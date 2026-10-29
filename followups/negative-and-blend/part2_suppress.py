"""Suppress a distilled owl persona and system-prompted concepts using saved artifacts.

The implementation brief's bands and conditions supersede the README. Calibration
runs before LoRA merging; base weights are restored byte-for-byte before 2B.
"""
import contextlib
import json
import pathlib
import sys
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
PRIOR = HERE.parent.parent / "study"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(PRIOR))
import hf_cache
from jobs import MODELS, OUT, call, image, out_dir
from part1_blend import (BOOT, SEED, cluster_table, digest, file_hash, generate_rows, mcnemar, paired_mean,
                         prompt_rows, provenance, read_jsonl, sigma_values, unit, validate_tag, write_json, write_texts,
                         record_kind, capture_replies)

image = image.add_local_python_source("part1_blend")
app = modal.App("personakit-suppress")
X3_CONDITIONS = ("base_4b", "x3_unsteered", "x3_clamp_p10", "x3_clamp_p50", "x3_sub_1x", "x3_sub_2x", "x3_sub_full_2x")
SYS_CONDITIONS = ("sys_unsteered", "sys_clamp_p10", "sys_sub_1x", "sys_sub_2x", "sys_sub_full_2x")
METRICS = ("cm", "ci", "coherent", "assistant_identity")


def assert_ceiling():
    """Exercise the actual shared cap hook, including untouched below-ceiling rows."""
    import torch
    import persona
    direction = unit(torch.tensor([1., 2., -1.]))
    h = torch.tensor([[[2., 1., -3.], [-2., -1., 3.], [1., 1., 1.]]])
    tau = .25
    hooks = persona.Hooks(None)
    hooks.cap(0, -direction, -tau)
    got = hooks._out_hook(0)(None, (), h)
    expected = h - (h @ direction - tau).clamp(min=0)[..., None] * direction
    assert torch.allclose(got, expected, atol=1e-6, rtol=0)
    assert bool(((got @ direction) <= tau + 1e-6).all())
    assert torch.equal(got[(h @ direction) <= tau], h[(h @ direction) <= tau])


def calibrate(model, tok, records, axis, layers, concepts, smoke):
    import torch
    default = [r for r in records if record_kind(r) == "default"][:8 if smoke else 100]
    owls = [r for r in records if record_kind(r) == "concept" and r.get("concept") in ("an owl", "owls")]
    # S1a exports one reply per concept question: two in smoke, ten in full.
    if len(default) != (8 if smoke else 100) or len(owls) != (2 if smoke else 10):
        raise ValueError("Incomplete default/owl calibration records")
    h_default, default_audit = capture_replies(model, tok, default, layers)
    h_owl, owl_audit = capture_replies(model, tok, owls, layers)
    # S1b means weight replies equally; threshold distributions pool reply tokens.
    owl = {L: unit(torch.stack([h.mean(0) for h in h_owl[L]]).mean(0)
                   - torch.stack([h.mean(0) for h in h_default[L]]).mean(0)) for L in layers}
    cdirs = axis.get("concept_dirs_test", axis.get("concept_dirs"))
    directions = {c: {L: unit(cdirs[c][L]) for L in layers} for c in concepts}
    directions["an owl"] = owl
    pooled = {L: torch.cat(h_default[L]) for L in layers}
    calibration = {}
    for concept, dirs in directions.items():
        calibration[concept] = {}
        for L in layers:
            proj = pooled[L] @ dirs[L]
            calibration[concept][L] = dict(p10=float(torch.quantile(proj, .10)), p50=float(torch.quantile(proj, .50)),
                                           projection_std=float(proj.std()), tokens=len(proj))
    audit = dict(default=default_audit, owl=owl_audit, mean_weighting="equal reply means; token-pooled quantiles",
                 owl_direction="saved owl reply mean minus selected default reply mean; base model before LoRA",
                 directions_sha256={c: digest({L: v.tolist() for L, v in dirs.items()}) for c, dirs in directions.items()})
    return directions, calibration, audit, sigma_values(axis, layers, h_default)


@contextlib.contextmanager
def merged_lora(model, path, metadata, smoke):
    """Keep CPU backups, not subtractive bf16 unmerging, to recover exact base weights."""
    import torch
    import persona
    state = torch.load(path, map_location="cpu", weights_only=True)
    linears = [getattr(block.mlp, name) for block in persona.blocks(model) for name in ("gate_proj", "up_proj", "down_proj")]
    if set(state) != {str(i) for i in range(len(linears))}:
        raise ValueError("Adapter keys do not match all layer-major gate/up/down linears")
    rank, alpha = metadata["sizes"]["rank"], metadata["sizes"]["alpha"]
    key = "0.8b" if smoke else "4b"
    if metadata["model"] != key or (not smoke and (rank, alpha) != (8, 16)):
        raise ValueError("Unexpected adapter model/rank/alpha")
    if not any((x["repo"], x["revision"]) == MODELS[key] for x in metadata["loads"]):
        raise ValueError("Adapter base model pin mismatch")
    for i, lin in enumerate(linears):
        A, B = state[str(i)]
        if A.shape != (rank, lin.in_features) or B.shape != (lin.out_features, rank):
            raise ValueError(f"Adapter shape mismatch at {i}")
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():
            raise ValueError(f"Nonfinite adapter at {i}")
    backups = []
    try:
        with torch.no_grad():
            for i, lin in enumerate(linears):
                backups.append(lin.weight.detach().cpu().clone())
                A, B = (v.to(lin.weight.device, torch.float32) for v in state[str(i)])
                lin.weight.add_((B @ A * (alpha / rank)).to(lin.weight.dtype))
        yield dict(linears=len(linears), rank=rank, alpha=alpha, scale=alpha / rank)
    finally:
        with torch.no_grad():
            for lin, backup in zip(linears, backups):
                lin.weight.copy_(backup)
                if not torch.equal(lin.weight.detach().cpu(), backup):
                    raise AssertionError("Base restoration failed")


class ClampActivity:
    """Observe pre-clamp activations on live cached decode positions only.

    Prefill is excluded. The input-token observer masks stopped/padded sequences;
    the last emitted token is not forwarded and cannot have an observed residual.
    Counts therefore cover generated tokens actually fed back into the model.
    """
    def __init__(self, model, tok, dirs, thresholds):
        self.model, self.tok, self.dirs, self.thresholds = model, tok, dirs, thresholds
        self.handles, self.mask = [], None
        self.counts = {L: dict(triggered=0, positions=0) for L in dirs}

    def observe_input(self, module, args, kwargs):
        import torch
        import persona
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None or ids.shape[1] != 1:
            self.mask = None
            return
        self.mask = torch.ones_like(ids, dtype=torch.bool)
        for stop in set(persona.stops(self.tok) + [self.tok.pad_token_id]):
            self.mask &= ids != stop

    def observer(self, layer):
        def hook(module, args, output):
            import torch
            if self.mask is None:
                return
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[:2] != self.mask.shape:
                raise ValueError("Decode mask mismatch")
            proj = h.float() @ self.dirs[layer].to(h.device, torch.float32)
            self.counts[layer]["positions"] += int(self.mask.sum())
            self.counts[layer]["triggered"] += int(((proj > self.thresholds[layer]) & self.mask).sum())
        return hook

    def __enter__(self):
        import persona
        self.handles.append(self.model.register_forward_pre_hook(self.observe_input, with_kwargs=True))
        for L in self.dirs:
            self.handles.append(persona.blocks(self.model)[L].register_forward_hook(self.observer(L)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def result(self):
        counts = {str(L): {**c, "fraction": c["triggered"] / c["positions"] if c["positions"] else None}
                  for L, c in self.counts.items()}
        n = sum(c["positions"] for c in self.counts.values())
        k = sum(c["triggered"] for c in self.counts.values())
        return dict(by_layer=counts, positions=n, triggered=k, fraction=k / n if n else None,
                    denominator="forwarded generated token positions per layer; excludes prefill, stops and padding")


@contextlib.contextmanager
def intervention(model, tok, condition, directions, calibration, sigma, monitor=False):
    import persona
    clamp = "clamp" in condition
    thresholds = {L: calibration[L]["p50" if "p50" in condition else "p10"] for L in directions}
    # Registered before Hooks so observations see h, not its already-clamped value.
    with contextlib.ExitStack() as stack:
        activity = stack.enter_context(ClampActivity(model, tok, directions, thresholds)) if clamp and monitor else None
        hooks = stack.enter_context(persona.Hooks(model))
        for L, direction in directions.items():
            if clamp:
                hooks.cap(L, -direction, -thresholds[L])
            elif "sub" in condition:
                gain = 1.0 if "1x" in condition else 2.0
                denominator = 1 if "full" in condition else len(directions)
                hooks.add(L, -gain * sigma[L] / denominator * direction)
        yield activity


def mmlu_summary(bits, concepts):
    import numpy as np
    baseline = bits["base_4b"]
    result = {condition: dict(accuracy=paired_mean(values), vs_base=paired_mean(values, baseline),
                              mcnemar_vs_base=mcnemar(values, baseline)) for condition, values in bits.items()}
    for condition in ("sys_clamp_p10", "sys_sub_2x", "sys_sub_full_2x"):
        averaged = np.mean([bits[f"{condition}:{c}"] for c in concepts], axis=0).tolist()
        result[condition] = dict(accuracy=paired_mean(averaged), vs_base=paired_mean(averaged, baseline),
                                 item_means=averaged, pooling="average concepts per MMLU item before bootstrap")
    return result


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu="A10G", memory=32768, timeout=1800)
def run(key: str, smoke: bool, tag: str, source: dict):
    import torch
    import persona
    import prompts as P
    import judge
    import evals
    assert_ceiling()
    started = time.time()
    if not modal.is_local():
        OUT.reload()
    suffix = "-smoke" if smoke else ""
    paths = {"axis": out_dir(f"s1b-{key}{suffix}") / "axis.pt", "s1a": out_dir(f"s1a-{key}{suffix}") / "texts.jsonl",
             "lora": out_dir(f"x3-{key}{suffix}") / "lora.pt", "metadata": out_dir(f"x3-{key}{suffix}") / "x3.json"}
    axis = torch.load(paths["axis"], map_location="cpu", weights_only=True)
    records, metadata = read_jsonl(paths["s1a"]), json.loads(paths["metadata"].read_text())
    layers = [11, 12, 13, 14] if smoke else [14, 15, 16, 17, 18]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model, tok = persona.load(hf_cache.model(*MODELS[key]), device)
    concepts = P.CONCEPTS_TEST[:2] if smoke else P.CONCEPTS_TEST
    dirs, calibration, calibration_audit, sigma = calibrate(model, tok, records, axis, layers, concepts, smoke)
    print("[suppress] calibrated default projection quantiles and owl direction", flush=True)
    mmlu_rows = persona.mmlu_rows(10 if smoke else 300, seed=42)
    batch, max_new = (2, 16) if smoke else (40, 96)
    texts, bits, activity = [], {}, {}

    def evaluate(condition, concept, family, system="", capability=False):
        direction_key = family if condition.startswith("sys_") else "an owl"
        prompts = [{**r, "condition": condition, "concept": concept, "family": family,
                    "regime": "2B" if condition.startswith("sys_") else "2A", "system": system} for r in prompt_rows(smoke)]
        with intervention(model, tok, condition, dirs[direction_key], calibration[direction_key], sigma, monitor=True) as counter:
            texts.extend(generate_rows(model, tok, prompts, max_new, batch, system))
        name = f"{condition}:{concept}" if condition.startswith("sys_") else condition
        if counter is not None:
            activity[name] = counter.result()
        if capability:
            # No system prompt for 2B MMLU: isolate the cost of the suppression operator.
            with intervention(model, tok, condition, dirs[direction_key], calibration[direction_key], sigma):
                bits[name] = persona.mmlu(model, tok, mmlu_rows, batch=batch)
        print(f"[suppress] {name}: {len(prompts)} replies, mmlu={capability}", flush=True)

    evaluate("base_4b", "an owl", "owls", capability=True)
    with merged_lora(model, paths["lora"], metadata, smoke) as adapter:
        for condition in X3_CONDITIONS[1:]:
            evaluate(condition, "an owl", "owls", capability=True)
    # The context restores every MLP matrix before any prompted-concept evaluation.
    for c in concepts:
        for condition in SYS_CONDITIONS:
            evaluate(condition, P.PERSONA_NAMES[c], c, P.PERSONA_PROMPT.format(name=P.PERSONA_NAMES[c]),
                     capability=condition in ("sys_clamp_p10", "sys_sub_2x", "sys_sub_full_2x"))
    # Release the closure's generator references before loading the judge.
    del evaluate, model
    persona.free()
    output = out_dir(tag)
    write_texts(output / "texts.jsonl", texts)
    j_model, j_tok = persona.load(hf_cache.model(*MODELS["0.8b" if smoke else "9b"]), device)
    labels = judge.score(j_model, j_tok, texts, ["concept", "identity", "coherent"], 2 if smoke else 4)
    del j_model
    persona.free()
    for row, label in zip(texts, labels):
        row.update(cm=int(P.mentions(row["reply"], row["family"])), ci=label["concept"], coherent=label["coherent"],
                   assistant_identity=label["identity"], judge=label)
    tables = {}
    for regime, conditions, prefix in (("2A", X3_CONDITIONS, "x3"), ("2B", SYS_CONDITIONS, "sys")):
        comparisons = [(f"{prefix}_clamp_p10", f"{prefix}_{c}") for c in ("unsteered", "sub_2x", "sub_full_2x")]
        tables[regime] = cluster_table([r for r in texts if r["regime"] == regime], METRICS, "concept", conditions,
                                       comparisons=comparisons)
    result = dict(model=key, smoke=smoke, tag=tag, texts=len(texts), layers=layers, sigma=sigma, provenance=source,
                  input_sha256={k: file_hash(p) for k, p in paths.items()}, adapter=adapter, base_restored_exactly=True,
                  calibration=calibration, calibration_audit=calibration_audit, clamp_activity=activity,
                  sigma_definition="sample SD (correction=1) of pooled default reply token projections onto unit Assistant Axis",
                  clarification="measured SD; canonical CM/CI meanings",
                  summary=tables, mmlu_bits=bits, mmlu=mmlu_summary(bits, [P.PERSONA_NAMES[c] for c in concepts]),
                  mmlu_items_sha256=digest(mmlu_rows), mmlu_pin=evals.MMLU, mmlu_seed=42, mmlu_system="",
                  statistics=dict(seed=SEED, bootstrap=BOOT, reply_cluster="prompt q across concepts and splits",
                                  mmlu_unit="item; concepts averaged before resampling", mcnemar="exact, unadjusted"),
                  metric_definitions=dict(cm="prompts.mentions lexical families", ci="judge concept persona adoption",
                                          assistant_identity="judge identity; 1 means AI assistant"),
                  loads=hf_cache.loads, seconds=time.time() - started)
    write_texts(output / "texts.jsonl", texts)
    write_json(output / "suppress.json", result)
    if not modal.is_local():
        OUT.commit()
    return dict(model=key, smoke=smoke, tag=tag, texts=len(texts), seconds=result["seconds"], output=str(output / "suppress.json"))


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False, tag: str = ""):
    key = "0.8b" if smoke else model
    if key != ("0.8b" if smoke else "4b"):
        raise ValueError("Full suppression uses the saved 4b adapter only")
    tag = tag or f"suppress-{key}{'-smoke' if smoke else ''}"
    validate_tag(tag)
    source = provenance(pathlib.Path(__file__))
    source["blend_helpers_sha256"] = file_hash(HERE / "part1_blend.py")
    result = call(run, key, smoke, tag, source)
    print("PERSONAKIT_RESULT", json.dumps(result, allow_nan=False))
