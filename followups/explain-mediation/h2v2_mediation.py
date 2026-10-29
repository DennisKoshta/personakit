"""H2-v2 study driver. Model execution requires the project owner's separate authorization.

CPU driver smoke (after primitive gate):
  python \
    followups/explain-mediation/h2v2_mediation.py --smoke

Only the project owner launches paid jobs via modal run. Full run CLI: --model 4b;
--anchor-only omits optional H4 targets; --specificity adds the optional anchor
random-weight diagnostic. All generations allow 64 greedy new tokens, including
smoke. Smoke reduces pairs/prompts, keeps all 24 layers and every condition.
Estimated CPU smoke peak 6–8 GiB; full job CPU admission 32 GiB, one active model
in device memory (Base -> Instruct -> 9B judge). No completion text or token IDs
are persisted. Checkpoints contain raw per-prompt bits/hashes, not resumable text.
"""

from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
import uuid

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "study"))
import hf_cache
import jobs
import h2v2_analysis as analysis

app = modal.App("personakit-explain-h2v2")
IMAGE = jobs.image.add_local_python_source("mechanics", "h2v2_analysis")
SEED = 20261028
MAX_NEW = 64
PINS = {
    "4b": {"calib": ("s1a-4b", "calib.pt", "ddcd2337a49e019a4931b4888e8c180def08d89f14043f94dd770b1771223618"),
           "axis": ("s1b-4b", "axis.pt", "dc039446a2092e7aeb355439d80183264900cbd5749f5089a11512629926869e"),
           "texts": ("s1a-4b", "texts.jsonl", "dc06a07de3b9b6893d86e7d3d23e35c1b8b9d7a4fc9974422dc2e151b52fb24e")},
    "0.8b": {"calib": ("s1a-0.8b-smoke", "calib.pt", "3c12ddeb73b44f851c6865c618b668c542ad6f6fa7cdb54944cfec623686fd4c"),
             "axis": ("s1b-0.8b-smoke", "axis.pt", "8e12a4399e3dd9e9b778c20e8b15980000c015596dee9db7255ad4629fb9ecb5"),
             "texts": ("s1a-0.8b-smoke", "texts.jsonl", "1ae8111946349eb71c88c4f1c75bd6a32f6da7526a61ee160e1b675b879b17c6")}}


def log(message):
    print(f"[h2v2] {message}", flush=True)


def digest(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, allow_nan=False))


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    import torch
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite artifact tensor")
    elif isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            finite(v)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite scalar")


def save_json(out, filename, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    tmp = out / (filename + ".tmp")
    tmp.write_text(text)
    tmp.replace(out / filename)
    if not modal.is_local():
        jobs.OUT.commit()


def save_tensor(out, filename, value):
    import torch
    finite(value)
    tmp = out / (filename + ".tmp")
    torch.save(value, tmp)
    tmp.replace(out / filename)
    if not modal.is_local():
        jobs.OUT.commit()
    return file_hash(out / filename)


def input_artifacts(key):
    import torch
    evidence, loaded = {}, {}
    for kind, (tag, filename, expected) in PINS[key].items():
        path = jobs.out_dir(tag) / filename
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"{kind} hash mismatch at {path}; never substitute another atlas")
        evidence[kind] = {"tag": tag, "file": filename, "sha256": actual}
        if kind == "texts":
            loaded[kind] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            loaded[kind] = torch.load(path, map_location="cpu", weights_only=True)
            finite(loaded[kind])
    return loaded, evidence


def read_targets(path, *, smoke=False):
    import prompts as P
    values = json.loads(path.read_text())
    if not isinstance(values, list) or len(values) > 3:
        raise ValueError("targets.json must be a list of at most three eligible new targets")
    fields = {"id", "layer", "neuron", "sign", "dose", "concept", "concept_key", "ci", "coherence", "exploratory"}
    seen_ids, seen_units, targets = {"alien_anchor"}, set(), []
    for t in values:
        if not isinstance(t, dict) or not fields <= set(t):
            raise ValueError("incomplete H4 target schema")
        if not isinstance(t["id"], str) or not t["id"] or t["id"] in seen_ids:
            raise ValueError("invalid/duplicate target ID")
        if any(type(t[k]) is not int for k in ("layer", "neuron", "sign")):
            raise ValueError("target coordinates must be integers")
        unit = (t["layer"], t["neuron"], t["sign"])
        if (not 8 <= t["layer"] <= (19 if smoke else 22) or not 0 <= t["neuron"] < (3584 if smoke else 9216)
                or t["sign"] not in (-1, 1) or unit in seen_units or (not smoke and unit in P.KNOWN.values())):
            raise ValueError("duplicate/known/downstream/invalid target coordinate")
        if any(type(t[k]) not in (float, int) or not math.isfinite(t[k]) for k in ("dose", "ci", "coherence")):
            raise ValueError("nonfinite or nonnumeric target eligibility")
        if t["dose"] not in (.5, .75) or not .5 <= t["ci"] <= 1 or not .8 <= t["coherence"] <= 1:
            raise ValueError("uncalibrated H4 target")
        if t["exploratory"] is not True or t["concept_key"] not in P.FAMILIES or not isinstance(t["concept"], str) or not t["concept"].strip():
            raise ValueError("invalid concept key or exploratory designation")
        seen_ids.add(t["id"])
        seen_units.add(unit)
        targets.append({**t, "cm_proxy": t["concept"] != P.PERSONA_NAMES[t["concept_key"]]})
    return targets


def validate_axis(ax, calib, model):
    import torch
    import prompts as P
    n, d = len(model.model.layers), model.config.hidden_size
    if ax["unit"].shape != (n, d) or set(ax["concept_dirs"]) != set(P.FAMILIES):
        raise ValueError("atlas dimensions or plural concept keys do not match")
    for key in ("median_resid", "floors_p25"):
        if ax[key].shape != (n,):
            raise ValueError("atlas calibration dimensions do not match")
    torch.testing.assert_close(ax["unit"].float().norm(dim=-1), torch.ones(n), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ax["median_resid"], calib["pile"]["norm_q10_50_90"][:, 1])
    if not (ax["median_resid"] > 0).all():
        raise ValueError("nonpositive steering calibration")


def capture(model, seqs, starts, layers, directions=None):
    """Response means or token projections, reducing ONLY selected layers in-hook.

    This implements persona.span_means/token_projections response-mask semantics
    without retaining a full model stack of activations. Return CPU tensors only.
    """
    import torch
    import persona
    if not seqs or len(seqs) != len(starts) or any(not 0 < s < len(q) for q, s in zip(seqs, starts)):
        raise ValueError("capture requires nonempty actual response spans")
    ids, mask = persona.pad_left(seqs, 0, model.device)
    width = ids.shape[1]
    spans = [(width - len(q) + s, len(q) - s) for q, s in zip(seqs, starts)]
    values, handles = {}, []
    def collect(layer):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            pieces = [hidden[b, start:start + count].float() for b, (start, count) in enumerate(spans)]
            if directions is None:
                values[layer] = torch.stack([p.mean(0) for p in pieces]).cpu()
            else:
                values[layer] = torch.cat([p @ directions[layer].to(p.device).T for p in pieces]).cpu()
        return hook
    try:
        for layer in layers:
            handles.append(model.model.layers[layer].register_forward_hook(collect(layer)))
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
                  use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    if set(values) != set(layers):
        raise AssertionError("incomplete selected-layer capture")
    finite(values)
    return values


def generate(model, tok, pids, batch):
    """Fresh generation cache on every attempt; retry a failed batch at half size."""
    import torch
    import persona
    output, cursor = [], 0
    while cursor < len(pids):
        count = min(batch, len(pids) - cursor)
        failed = False
        try:
            got = persona.generate(model, tok, pids[cursor:cursor + count], MAX_NEW, sample=False)
        except torch.OutOfMemoryError:
            if count == 1:
                raise
            failed = True
        if failed:
            gc.collect()
            if model.device.type == "cuda":
                torch.cuda.empty_cache()
            batch = max(1, count // 2)
            log(f"generation OOM: retry original rows {cursor}:{cursor + count} at batch {batch}")
            continue
        if len(got) != count:
            raise AssertionError("generation row count changed")
        output.extend(got)
        cursor += count
    return output


FIRST = re.compile(r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves)\b", re.I)


def edit_pair_reply(text, person, concept):
    import prompts as P
    edits = []
    if person == "first" and not FIRST.search(text):
        text = "I am " + P.PERSONA_NAMES[concept] + ". " + text
        edits.append("prepend_first_person_identity")
    if person == "third" and FIRST.search(text):
        text = " ".join(s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if not FIRST.search(s))
        edits.append("remove_sentences_with_first_person_pronouns")
    if person == "third" and not text.strip():
        text = f"In folklore and science fiction, {concept} are depicted through their distinctive appearance, habits, and surroundings."
        edits.append("fixed_third_person_fallback")
    if (person == "first" and not FIRST.search(text)) or (person == "third" and FIRST.search(text)):
        raise AssertionError("pair pronoun editing failed")
    return text.strip(), edits


def build_directions(model, tok, ax, cached_texts, cap_layers, config, out):
    import torch
    import persona
    import prompts as P
    import mechanics
    pairs, metadata, third_captures = [], [], []
    questions = P.CONCEPT_QUESTIONS[:1] if config["smoke"] else P.CONCEPT_QUESTIONS
    concepts = P.CONCEPTS_DEV[:2] if config["smoke"] else P.CONCEPTS_DEV
    diffs = {layer: torch.zeros(model.config.hidden_size) for layer in cap_layers}
    for concept in concepts:
        for qi, template in enumerate(questions):
            q = template.format(x=concept)
            if q in P.IDENTITY_DEV + P.IDENTITY_TEST:
                raise AssertionError("capture questions overlap identity evaluation")
            systems = [P.PERSONA_PROMPT.format(name=P.PERSONA_NAMES[concept]),
                       f"Respond objectively in the third person about {concept}. Never use first-person pronouns (I, me, my, mine, myself, we, us, our)."]
            generation_ids = [persona.chat_ids(tok, q, s) for s in systems]
            outputs = generate(model, tok, generation_ids, config["batch"])
            common = persona.chat_ids(tok, q)
            seqs, starts = [], []
            for person, token_ids, system in zip(("first", "third"), outputs, systems):
                raw = tok.decode(token_ids, skip_special_tokens=True)
                edited, edits = edit_pair_reply(raw, person, concept)
                reply_ids = tok.encode(edited, add_special_tokens=False)
                if not reply_ids:
                    raise AssertionError("empty paired continuation")
                seqs.append(common + reply_ids)
                starts.append(len(common))
                metadata.append({"pair_id": f"{concept}:{qi}", "person": person, "concept_key": concept,
                                 "question": q, "common_prefix_sha256": json_digest(common),
                                 "capture_start": len(common), "generation_system_sha256": digest(system),
                                 "generated_reply_sha256": digest(raw), "captured_reply_sha256": digest(edited),
                                 "generation_tokens": len(token_ids), "capture_tokens": len(reply_ids), "edits": edits})
            got = capture(model, seqs, starts, cap_layers)
            for layer in cap_layers:
                diffs[layer] += got[layer][0] - got[layer][1]
            third_captures.append((seqs[1], starts[1]))
            pairs.append(f"{concept}:{qi}")
            log(f"matched common-prefix pair {len(pairs)}/{len(concepts) * len(questions)}")
            save_json(out, "pairs.json", {"pairs": metadata, "interpretation": "framing-associated, edited where recorded; not pure voice"})
    expected = 2 if config["smoke"] else 50
    if len(pairs) != expected:
        raise AssertionError("pair count mismatch")
    gen = torch.Generator().manual_seed(SEED)
    directions = {}
    for layer in cap_layers:
        basis = [ax["unit"][layer]] + [ax["concept_dirs"][c][layer] for c in P.FAMILIES]
        unit = mechanics.orthogonalize(diffs[layer] / len(pairs), basis)
        random_unit = mechanics.orthogonalize(torch.randn(model.config.hidden_size, generator=gen), [unit, *basis])
        for vector in (unit, random_unit):
            if abs(float(vector.norm()) - 1) > 1e-5 or any(abs(float(vector @ (b / b.norm()))) > 1e-5 for b in basis):
                raise AssertionError("invalid orthogonal direction")
        directions[layer] = torch.stack([unit, random_unit])
    default_count = 2 if config["smoke"] else 100
    defaults = [r for r in cached_texts if r["kind"] == "default"][:default_count]
    if len(defaults) != default_count:
        raise ValueError("insufficient actual cached default replies")
    calibration = list(third_captures)
    default_metadata = []
    for record in defaults:
        # Reuse the producer's rendered prompt verbatim, preserving its system.
        prefix = tok.encode(record["prompt"], add_special_tokens=False)
        reply = tok.encode(record["reply"], add_special_tokens=False)
        if not prefix or not reply:
            raise ValueError("empty cached default prompt/reply")
        calibration.append((prefix + reply, len(prefix)))
        default_metadata.append({"prompt_sha256": digest(record["prompt"]), "reply_sha256": digest(record["reply"]),
                                 "start": len(prefix), "reply_tokens": len(reply)})
    projections = {layer: [] for layer in cap_layers}
    for start in range(0, len(calibration), config["capture_batch"]):
        chunk = calibration[start:start + config["capture_batch"]]
        got = capture(model, [x[0] for x in chunk], [x[1] for x in chunk], cap_layers, directions)
        for layer in cap_layers:
            projections[layer].append(got[layer])
    packed = {layer: torch.cat(parts) for layer, parts in projections.items()}
    thresholds = {layer: value.quantile(.75, dim=0) for layer, value in packed.items()}
    saved = {"directions": directions, "ceilings": thresholds, "raw_mean_contrast": {l: d / len(pairs) for l, d in diffs.items()},
             "projection_tokens": {l: len(p) for l, p in packed.items()}, "projection_quantiles": {l: p.quantile(torch.tensor([.25, .5, .75]), dim=0) for l, p in packed.items()}}
    tensor_hash = save_tensor(out, "directions.pt", saved)
    save_json(out, "direction_manifest.json", {"sha256": tensor_hash, "pairs": len(pairs), "default_replies": default_metadata,
                                               "calibration": "token-weighted union of default and third-person replies; own .75 quantile per direction"})
    return directions, thresholds


def state_digest(state):
    """Byte-exact restoration check, one tensor at a time including bfloat16."""
    import torch
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        value = value.detach().cpu().contiguous()
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def live_state(model, layers, kind):
    # Non-owning state references, independent of mechanics' clone API.
    result = {}
    for layer in layers:
        block = model.model.layers[layer]
        name = "mlp" if kind == "mlp" else ("self_attn" if getattr(block, "self_attn", None) is not None else "linear_attn")
        result.update({f"model.layers.{layer}.{name}.{k}": v for k, v in getattr(block, name).state_dict().items()})
    return result


def random_weight_donor(original, base, seed):
    """Preflight an optional donor in its ACTUAL storage dtype before generation.

    Try dense noise, then signed permutations of the empirical delta to preserve
    sparse representable amplitudes. Scale search accounts for rounding, but both
    norm and orthogonality must pass AFTER casting. Infeasible tensors make the
    entire optional diagnostic unavailable; never substitute an identity tensor
    or a partially matched donor and call it a successful specificity control.
    """
    import torch
    generator = torch.Generator().manual_seed(seed)
    donor, manifest = {}, []
    for key, source in original.items():
        delta = base[key].float() - source.float()
        magnitude = delta.double().norm()
        if not torch.isfinite(magnitude):
            raise ValueError("nonfinite weight delta")
        if magnitude == 0:
            donor[key] = source.clone()
            continue
        accepted = None
        if delta.numel() > 1:
            direction = delta / delta.norm()
            for attempt in range(4):
                if attempt == 0:
                    noise = torch.randn(source.shape, generator=generator)
                else:
                    permutation = torch.randperm(delta.numel(), generator=generator)
                    noise = delta.flatten()[permutation].reshape_as(delta)
                    noise *= torch.randint(0, 2, source.shape, generator=generator).float().mul_(2).sub_(1)
                    del permutation
                for _ in range(2):
                    noise -= (noise * direction).sum() * direction
                if not torch.isfinite(noise).all() or noise.norm() <= 1e-8:
                    continue
                noise *= magnitude.float() / noise.norm()
                # Bounded bracket search on the quantized norm. Every candidate
                # independently checks cosine; scale alone cannot enforce it.
                low, high = 0., 4.
                for step in range(10):
                    scale = 1. if step == 0 else (low + high) / 2
                    candidate = (source.float() + noise * scale).to(source.dtype)
                    actual = candidate.float() - source.float()
                    norm = actual.double().norm()
                    relative = float(norm / magnitude)
                    cosine = float((actual.double().flatten() @ delta.double().flatten()) / (norm * magnitude)) if norm > 0 else None
                    if cosine is not None and .9 <= relative <= 1.1 and abs(cosine) <= .1:
                        accepted = candidate
                        manifest.append({"key": key, "norm_ratio": relative, "cosine_after_cast": cosine,
                                         "candidate": "dense" if attempt == 0 else "signed_delta_permutation",
                                         "attempt": attempt, "scale": scale})
                        break
                    if relative < 1:
                        low = scale
                    else:
                        high = scale
                if accepted is not None:
                    break
        if accepted is None:
            return None, {"status": "unavailable", "failed_tensor": key,
                          "reason": "no post-cast norm/orthogonality match in bounded search",
                          "tensors": manifest, "partial_donor_used": False}
        donor[key] = accepted
    if not manifest:
        return None, {"status": "unavailable", "reason": "zero donor delta", "tensors": [], "partial_donor_used": False}
    return donor, {"status": "ready", "tensors": manifest, "norm_ratio_tolerance": [.9, 1.1], "max_abs_cosine": .1}


def donor_snapshot(config, device):
    import persona
    import mechanics
    model, tok = persona.load(hf_cache.model(*jobs.MODELS["4b-base"]), device)
    try:
        return {kind: mechanics.selected_module_state(model, config["swap_layers"], kind) for kind in ("attention", "mlp")}
    finally:
        del model, tok
        persona.free()


@contextmanager
def cap_observers(model, condition, layers, dirs, ceilings, axis, stats, smoke):
    """Observe post-Hooks residuals on real prefill and cached decode outputs."""
    import torch
    handles = []
    def observe(layer):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if condition == "axis_cap":
                unit, threshold = axis["unit"][layer], float(axis["floors_p25"][layer])
                violation = threshold - hidden.float() @ unit.to(hidden.device)
            else:
                col = int(condition == "random_cap")
                unit, threshold = dirs[layer][col], float(ceilings[layer][col])
                violation = hidden.float() @ unit.to(hidden.device) - threshold
            finite(violation)
            row = stats.setdefault(str(layer), {"calls": 0, "cached_decode_calls": 0, "max_violation": 0.})
            row["calls"] += 1
            row["cached_decode_calls"] += int(hidden.shape[1] == 1)
            row["max_violation"] = max(row["max_violation"], float(violation.max()))
            # CPU fp32 gate. Full bf16 rounding errors are measured, not hidden
            # by silently changing calibrated thresholds or scientific cutoffs.
            if smoke and row["max_violation"] > 3e-4:
                raise AssertionError("actual cap projection violates numerical tolerance")
        return hook
    try:
        if condition in ("axis_cap", "self_ref_cap", "random_cap"):
            for layer in layers:
                handles.append(model.model.layers[layer].register_forward_hook(observe(layer)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def subject_phase(config, inputs, targets, out):
    import torch
    import persona
    import prompts as P
    import mechanics
    device = "cpu" if config["smoke"] else "cuda"
    donor = None if config["smoke"] else donor_snapshot(config, device)
    model, tok = persona.load(hf_cache.model(*jobs.MODELS[config["model"]]), device)
    try:
        initial_hooks = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
        validate_axis(inputs["axis"], inputs["calib"], model)
        if len(model.model.layers) != (24 if config["smoke"] else 32):
            raise ValueError("wrong full layer stack")
        original_hashes = {k: state_digest(live_state(model, config["swap_layers"], k)) for k in ("attention", "mlp")}
        norm_hash = state_digest({f"{i}.{n}": getattr(model.model.layers[i], n).weight
                                 for i in config["swap_layers"] for n in ("input_layernorm", "post_attention_layernorm")})
        if config["smoke"]:
            donor = {k: mechanics.selected_module_state(model, config["swap_layers"], k) for k in ("attention", "mlp")}
            for kind, state in donor.items():
                for layer in config["swap_layers"]:
                    suffix = "mlp.down_proj.weight" if kind == "mlp" else (
                        "self_attn.o_proj.weight" if layer == 23 else "linear_attn.out_proj.weight")
                    state[f"model.layers.{layer}.{suffix}"][0].add_(.01)
        donor_hashes = {k: state_digest(v) for k, v in donor.items()}
        if any(donor_hashes[k] == original_hashes[k] for k in donor):
            raise ValueError("vacuous swap donor")
        specificity_state, specificity_report = None, {"status": "not_requested"}
        if config["specificity"]:
            original = mechanics.selected_module_state(model, config["swap_layers"], "attention")
            specificity_state, specificity_report = random_weight_donor(original, donor["attention"], SEED)
            del original
        save_json(out, "specificity_preflight.json", specificity_report)
        if config["smoke"] and config["specificity"] and specificity_state is None:
            raise AssertionError("synthetic smoke donor must pass optional diagnostic preflight")
        if config["specificity"] and specificity_state is None:
            log("optional specificity unavailable after quantization preflight; core grid remains unchanged")
        vectors = {}
        for t in targets:
            if t["layer"] >= min(config["swap_layers"] + config["cap_layers"]):
                raise ValueError("injection is not strictly upstream")
            w = persona.down_col(model, t["layer"], t["neuron"]).detach().float().cpu()
            if not torch.isfinite(w).all() or w.norm() <= 0:
                raise ValueError("degenerate Instruct write vector")
            vectors[t["id"]] = t["sign"] * t["dose"] * float(inputs["axis"]["median_resid"][t["layer"]]) * w / w.norm()
        vector_hash = save_tensor(out, "write_vectors.pt", vectors)
        dirs, ceilings = build_directions(model, tok, inputs["axis"], inputs["texts"], config["cap_layers"], config, out)
        pids = [persona.chat_ids(tok, P.IDENTITY_TEST[i]) for i in config["prompt_ids"]]
        raw, safe, manifests, cap_stats = [], [], [], {}
        conditions = list(analysis.CONDITIONS) + (["random_weight_specificity"] if specificity_state is not None else [])
        for cond in conditions:
            log(f"condition {cond}")
            kind = "mlp" if cond in ("swap_mlp", "unsteered_swap_mlp") else "attention"
            state, diagnostic = None, None
            if cond in ("swap_attn", "unsteered_swap_attn", "swap_mlp", "unsteered_swap_mlp"):
                state = donor[kind]
            elif cond == "sham_swap_attn":
                state = mechanics.selected_module_state(model, config["swap_layers"], kind)
            elif cond == "random_weight_specificity":
                state, diagnostic = specificity_state, specificity_report
            ctx = mechanics.reversible_swap(model, config["swap_layers"], kind, state) if state is not None else nullcontext(None)
            try:
                with ctx as manifest:
                    if manifest is not None:
                        if (manifest["changed_tensors"] == 0) != (cond == "sham_swap_attn"):
                            raise AssertionError("swap/sham identity mismatch")
                        manifests.append({"condition": cond, "manifest": manifest, "specificity": diagnostic})
                    unsteered = cond in ("none", "unsteered_swap_attn", "unsteered_swap_mlp")
                    shared = generate(model, tok, pids, config["batch"]) if unsteered else None
                    for target in targets:
                        if cond == "random_weight_specificity" and target["exploratory"]:
                            continue
                        with persona.Hooks(model) as hooks:
                            if not unsteered:
                                hooks.add(target["layer"], vectors[target["id"]])
                            for layer in config["cap_layers"]:
                                if cond in ("self_ref_cap", "random_cap"):
                                    column = int(cond == "random_cap")
                                    mechanics.upper_cap(hooks, layer, dirs[layer][column], float(ceilings[layer][column]))
                                elif cond == "axis_cap":
                                    hooks.cap(layer, inputs["axis"]["unit"][layer], float(inputs["axis"]["floors_p25"][layer]))
                            stats = cap_stats.setdefault(f"{target['id']}|{cond}", {}) if cond in ("axis_cap", "self_ref_cap", "random_cap") else {}
                            with cap_observers(model, cond, config["cap_layers"], dirs, ceilings,
                                               inputs["axis"], stats, config["smoke"]):
                                outputs = shared if unsteered else generate(model, tok, pids, config["batch"])
                        del hooks
                        for pid, tokens in zip(config["prompt_ids"], outputs):
                            reply = tok.decode(tokens, skip_special_tokens=True)
                            judge_row = {"question": P.IDENTITY_TEST[pid], "reply": reply, "concept": target["concept"]}
                            rid = f"{target['id']}|{cond}|test-{pid}"
                            record = {"record_id": rid, "target_id": target["id"], "condition": cond, "prompt_id": pid,
                                      "input_sha256": json_digest(judge_row), "reply_sha256": digest(reply), "n_tokens": len(tokens),
                                      "cm": int(P.mentions(reply, target["concept_key"])), "concept": target["concept"],
                                      "concept_key": target["concept_key"], "cm_proxy": target["cm_proxy"],
                                      "generation_id": f"{cond}|{'shared' if unsteered else target['id']}|test-{pid}"}
                            raw.append({**judge_row, "record_id": rid, "input_sha256": record["input_sha256"]})
                            safe.append(record)
            finally:
                # These checks also execute after generation errors; restoration
                # itself occurs in the reviewed context's finally block.
                if state is not None and state_digest(live_state(model, config["swap_layers"], kind)) != original_hashes[kind]:
                    raise AssertionError("swap did not restore exactly")
                del ctx, state
            save_json(out, "raw_checkpoint.json", {"config": config, "targets": targets, "records": safe,
                                                    "status": "generated_not_judged", "no_completion_text": True})
        after_norm = state_digest({f"{i}.{n}": getattr(model.model.layers[i], n).weight
                                  for i in config["swap_layers"] for n in ("input_layernorm", "post_attention_layernorm")})
        if after_norm != norm_hash:
            raise AssertionError("excluded decoder norms changed")
        if [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()] != initial_hooks:
            raise AssertionError("subject hook cleanup failed")
        if any(set(s) != set(map(str, config["cap_layers"])) for s in cap_stats.values()):
            raise AssertionError("cap layers did not execute")
        save_json(out, "swaps.json", {"original_sha256": original_hashes, "donor_sha256": donor_hashes,
                                      "write_vectors_sha256": vector_hash, "conditions": manifests,
                                      "donor_kind": "synthetic selected perturbation" if config["smoke"] else "pinned Base model"})
        save_json(out, "cap_checks.json", {"observations_including_retried_attempts": cap_stats, "fp32_tolerance": 3e-4})
        return raw, safe
    finally:
        # No Hooks/model-capturing closures escape this frame into the judge load.
        del model, tok, donor
        persona.free()


def judge_phase(rows, config, out, checkpoint):
    import torch
    import persona
    import judge
    key = "0.8b" if config["smoke"] else "9b"
    model, tok = persona.load(hf_cache.model(*jobs.MODELS[key]), "cpu" if config["smoke"] else "cuda")
    labels, cursor, batch = [], 0, config["judge_batch"]
    try:
        while cursor < len(rows):
            count = min(batch, len(rows) - cursor)
            failed = False
            try:
                got = judge.score(model, tok, rows[cursor:cursor + count], ["identity", "coherent", "concept"], batch=count)
            except torch.OutOfMemoryError:
                if count == 1:
                    raise
                failed = True
            if failed:
                gc.collect()
                if model.device.type == "cuda":
                    torch.cuda.empty_cache()
                batch = max(1, count // 2)
                continue
            if len(got) != count:
                raise AssertionError("judge row count mismatch")
            for row, lab in zip(rows[cursor:cursor + count], got):
                finite(lab)
                for rubric in ("identity", "coherent", "concept"):
                    probs = lab[f"{rubric}_p"]
                    if (lab[rubric] not in (0, 1) or len(probs) != 2 or abs(sum(probs) - 1) > .0002
                            or any(not 0 <= p <= 1 for p in probs) or not 0 <= lab[f"{rubric}_mass"] <= 1):
                        raise ValueError("invalid complete judge output")
                labels.append({**lab, "record_id": row["record_id"], "input_sha256": row["input_sha256"]})
            cursor += count
            save_json(out, "checkpoint.json", {**checkpoint, "labels": labels, "status": "judging"})
            log(f"judged all three rubrics {cursor}/{len(rows)}")
    finally:
        del model, tok
        persona.free()
    return labels


@app.function(image=IMAGE, volumes={**hf_cache.volumes, "/out": jobs.OUT}, gpu="A10G", memory=32768, timeout=7200)
def run(config: dict) -> dict:
    import torch
    import prompts as P
    import transformers
    if config["smoke"] and (os.environ.get("PERSONAKIT_RESEARCH_CHECK") != "1" or not modal.is_local()):
        raise RuntimeError("smoke must use the local CPU checker")
    if not config["smoke"] and os.environ.get("PERSONAKIT_RESEARCH_CHECK"):
        raise RuntimeError("checker cannot execute full-size study")
    torch.manual_seed(SEED)
    if config["smoke"]:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    if not modal.is_local():
        jobs.OUT.reload()
    started = time.monotonic()
    root = jobs.out_dir("study-smoke" if config["smoke"] else "explain-med-20261028-h2v2")
    out = root / ("h2v2-" + uuid.uuid4().hex)
    out.mkdir(exist_ok=False)
    log(f"unique output: {out}")
    inputs, input_hashes = input_artifacts(config["model"])
    anchor = {"id": "alien_anchor", "layer": 19 if config["smoke"] else 22,
              "neuron": 7 if config["smoke"] else 2742, "sign": 1, "dose": .5,
              "concept": "an alien", "concept_key": "aliens", "exploratory": False, "cm_proxy": False}
    if config["smoke"]:
        # Explicitly isolated input for the actual target-reader path; eligibility
        # numbers are synthetic plumbing, never model findings or full-run input.
        path = out / "targets.json"
        save_json(out, "targets.json", [{"id": "smoke_new", "layer": 18, "neuron": 1, "sign": -1,
                  "dose": .5, "concept": "a friendly storyteller", "concept_key": "pirates",
                  "ci": .5, "coherence": .8, "exploratory": True}])
    else:
        path = jobs.out_dir(config["targets_tag"]) / "targets.json"
    new = [] if config["anchor_only"] else read_targets(path, smoke=config["smoke"])
    targets = [anchor, *new]
    provenance = {"inputs": input_hashes, "targets_sha256": file_hash(path) if not config["anchor_only"] else None,
                  "models": {k: list(jobs.MODELS[k]) for k in (["0.8b"] if config["smoke"] else ["4b", "4b-base", "9b"])},
                  "config": config, "config_sha256": json_digest(config), "source_hashes": config["source_hashes"],
                  "torch": torch.__version__, "transformers": transformers.__version__, "packages": list(hf_cache.PACKAGES),
                  "smoke_eligibility_is_synthetic": config["smoke"], "no_stale_labels": "unique output, fresh all-rubric judge"}
    save_json(out, "provenance.json", provenance)
    raw, records = subject_phase(config, inputs, targets, out)
    del inputs
    checkpoint = {"smoke": config["smoke"], "analysis_config": {"seed": SEED, "reps": 100 if config["smoke"] else 2000},
                  "targets": targets, "prompt_ids": config["prompt_ids"],
                  "records": records, "provenance": provenance}
    labels = judge_phase(raw, config, out, checkpoint)
    del raw
    checkpoint = {**checkpoint, "labels": labels, "status": "complete"}
    save_json(out, "checkpoint.json", checkpoint)
    # Exercise serialization and analysis from persisted bits, not parallel arrays.
    loaded = json.loads((out / "checkpoint.json").read_text())
    analysed = analysis.analyse_checkpoint(loaded)
    anchor_result = analysed["targets"]["alien_anchor"]
    result = {"smoke": config["smoke"], "anchor_verdict": analysed["anchor_verdict"],
              "anchor_qualified": anchor_result["qualified_baseline"], "targets": len(targets), "rows": len(records),
              "anchor_attn_ci_ratio": anchor_result["conditions"]["swap_attn"]["ci_ratio"]["estimate"],
              "anchor_mlp_ci_ratio": anchor_result["conditions"]["swap_mlp"]["ci_ratio"]["estimate"],
              "anchor_self_ref_ci_ratio": anchor_result["conditions"]["self_ref_cap"]["ci_ratio"]["estimate"],
              "anchor_signed_attn_vs_mlp": anchor_result["signed_attn_vs_mlp"]["estimate"],
              "minutes": (time.monotonic() - started) / 60, "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
              "output": str(out), "details": {"analysis": analysed, "provenance": provenance,
                                               "specificity_preflight": json.loads((out / "specificity_preflight.json").read_text()),
                                               "hf_cache_loads": hf_cache.loads, "checkpoint_sha256": file_hash(out / "checkpoint.json")}}
    finite(result)
    save_json(out, "result.json", result)
    return result


@app.local_entrypoint()
def main(smoke: bool = False, model: str = "4b", targets_tag: str = "explain-med-20261028-h4-4b",
         anchor_only: bool = False, specificity: bool = False):
    if model != "4b":
        raise ValueError("H2-v2 full driver supports only 4b; --smoke explicitly maps to 0.8b")
    if not targets_tag or "/" in targets_tag or targets_tag in (".", ".."):
        raise ValueError("targets_tag must be one output tag")
    sources = [Path(__file__), HERE / "h2v2_analysis.py", HERE / "mechanics.py",
               HERE / "design.json", HERE / "IMPLEMENTATION_NOTES.md",
               *[HERE.parent.parent / "study" / name for name in ("persona.py", "prompts.py", "judge.py", "jobs.py")]]
    config = {"smoke": smoke, "model": "0.8b" if smoke else model, "anchor_only": anchor_only,
              "specificity": smoke or specificity, "targets_tag": targets_tag, "max_new_tokens": MAX_NEW,
              "prompt_ids": list(range(2 if smoke else 20)), "swap_layers": list(range(20, 24) if smoke else range(23, 32)),
              "cap_layers": list(range(20, 24) if smoke else range(23, 29)), "batch": 1 if smoke else 4,
              "capture_batch": 1 if smoke else 4, "judge_batch": 1 if smoke else 4,
              "design_precedence": ["ebce16d", "89a43d6"],
              "source_hashes": {str(p.relative_to(HERE.parent)): file_hash(p) for p in sources}}
    # jobs.call uses .remote in python and supervised spawn/get remotely.
    result = jobs.call(run, config)
    print("PERSONAKIT_RESULT", json.dumps(result, allow_nan=False), flush=True)
