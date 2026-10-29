"""Preregistered early Hoot LoRA localization; CPU smoke via the CPU check runner.

Completed generation batches and projection spans are reusable after interruption.
The small smoke adapter checks execution, not the scientific hypothesis.
"""
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
PRIOR = HERE.parent.parent / "study"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(PRIOR))
import hf_cache
from jobs import MODELS, OUT, call, image, out_dir

app = modal.App("personakit-early-localize")
FULL = dict(test_q=20, everyday=20, s1a_n=100, max_new=96, batch=16,
            proj_batch=10, judge="9b", judge_batch=16, boot=2000)
SMOKE = dict(test_q=2, everyday=1, s1a_n=6, max_new=8, batch=4,
             proj_batch=4, judge="0.8b", judge_batch=4, boot=100)
PART1_CONDITIONS = (
    "base", "keep_0_5", "keep_6_10", "keep_11_15", "keep_0_10",
    "keep_0_5_11_15", "keep_6_15", "keep_0_15", "full", "drop_0_5",
    "drop_6_10", "drop_11_15", "drop_6_15", "keep_16_31",
)
SPANS = ("reply_span", "prefill_last", "s1a_forced")
DIRECTIONS = ("c_hat_L", "c_hat_mid", "a_hat_L")
RUBRICS = ("concept", "identity", "coherent", "name")
NAME_RUBRIC = (
    "Read the assistant's answer below. Does the speaker claim or introduce itself by the proper name "
    "\"{persona_name}\" (for example, \"I am {persona_name}\", \"Call me {persona_name}\", or \"My name is "
    "{persona_name}\")? Merely using the word as a sound effect or verb (such as \"*Hoots softly*\") without "
    "using \"{persona_name}\" as its name does not count.\n\n"
    "Question: {question}\nAnswer: {reply}\n\n"
    "Answer with a single character, 1 if yes or 0 if no:"
)
NAME_LEX = r"(?i)(?:\b(?:i am|i'm|call me|name is|known as|they call me|this is)\s+(?:old\s+)?hoot\b|\bold\s+hoot\b)"
VOCALIZATION_LEX = r"(?i)(?:\*[^*]*\bhoot[^*]*\*|\bhoo-hoo\b|\bwho-cooks-for-you\b)"


def early_bands(n_layers: int) -> dict[str, list[int]]:
    mid = n_layers // 2
    s1, s2 = round(6 * n_layers / 32), round(11 * n_layers / 32)
    if not 0 < s1 < s2 < mid:
        raise ValueError("Model is too small for three early sub-bands")
    return dict(b1=list(range(s1)), b2=list(range(s1, s2)),
                b3=list(range(s2, mid)), late=list(range(mid, n_layers)))


def condition_layers(n_layers):
    b = early_bands(n_layers)
    x, y, z, late = (b[k] for k in ("b1", "b2", "b3", "late"))
    layers = ([], x, y, z, x+y, x+z, y+z, x+y+z, x+y+z+late,
              y+z+late, x+z+late, x+y+late, x+late, late)
    return dict(zip(PART1_CONDITIONS, layers))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.chmod(0o600)
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    tmp.chmod(0o600)
    tmp.replace(path)


def save_projections(path, data):
    import torch
    tmp = path.with_suffix(".pt.tmp")
    torch.save(data, tmp)
    tmp.chmod(0o600)
    tmp.replace(path)


def commit_volume():
    if not modal.is_local():
        OUT.commit()


def stage_progress(path, previous, started):
    """Accumulate time through durable checkpoints, excluding interrupted work."""
    loads = {digest(row): row for row in previous.get("loads", []) + hf_cache.loads}
    progress = dict(elapsed_seconds=previous.get("elapsed_seconds", 0) + max(0, time.monotonic()-started),
                    loads=list(loads.values()))
    write_json(path, progress)
    return progress


def validate_spans(texts, projections, sizes, complete=False):
    """Bind cached reply/prefill projections to their actual per-condition tokens."""
    if complete and set(projections) != set(PART1_CONDITIONS):
        raise ValueError("Incomplete projection condition matrix")
    matched = {span: set() for span in ("prefill_last", "s1a_forced")}
    for cond, saved in projections.items():
        rows = [t for t in texts if t["cond"] == cond and t["split"] == "identity"]
        if complete and (set(saved) != set(SPANS) or len(rows) != sizes["test_q"]):
            raise ValueError("Incomplete projection spans")
        for span in ("reply_span", "prefill_last"):
            if span not in saved:
                continue
            if len(rows) != sizes["test_q"]:
                if span == "reply_span":
                    raise ValueError("Reply projections have no matching generated texts")
                continue  # Prefill can be checkpointed before the first generation batch.
            seqs = [r["prompt_ids"] + (r["token_ids"] if span == "reply_span" else []) for r in rows]
            starts = [len(r["prompt_ids"]) for r in rows]
            if saved[span]["tokens_hash"] != digest([seqs, starts]):
                raise ValueError(f"Projection tokens differ from saved texts: {cond}/{span}")
        for span in matched:
            if span in saved:
                matched[span].add(saved[span]["tokens_hash"])
    if any(len(hashes) > 1 for hashes in matched.values()):
        raise ValueError("Token-matched spans differ across conditions")


def clone_base_mlp(model):
    import persona
    linears = [getattr(block.mlp, name) for block in persona.blocks(model)
               for name in ("gate_proj", "up_proj", "down_proj")]
    return linears, [lin.weight.detach().cpu().clone() for lin in linears]


def load_adapter(path, metadata, linears, key):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    rank, alpha = metadata["sizes"]["rank"], metadata["sizes"]["alpha"]
    if key == "4b" and (rank != 8 or alpha != 16):
        raise ValueError("Expected the preregistered rank-8 alpha-16 Hoot adapter")
    if metadata["model"] != key or not any(
        (r["repo"], r["revision"]) == MODELS[key] for r in metadata["loads"]
    ):
        raise ValueError("Adapter does not match the pinned subject model")
    if set(state) != {str(i) for i in range(len(linears))}:
        raise ValueError("Expected saved layer-major gate/up/down adapter pairs")
    adapter = []
    for i, lin in enumerate(linears):
        a, b = state[str(i)]
        if a.shape != (rank, lin.in_features) or b.shape != (lin.out_features, rank):
            raise ValueError(f"Adapter shape mismatch: {i}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Nonfinite adapter")
        adapter.append((a.float(), b.float()))
    return adapter, alpha / rank


def apply_lora_layers(linears, originals, adapter, scale, active):
    import torch
    with torch.no_grad():
        for i, (lin, original, (a, b)) in enumerate(zip(linears, originals, adapter)):
            w = lin.weight
            merged = original.to(w.device, torch.float32)
            if i // 3 in active:
                merged = merged + scale * (b.to(w.device) @ a.to(w.device))
            w.copy_(merged.to(w.dtype))


def default_sequences(tok, path, sizes):
    """Read documented defaults or the original S1a export without changing tokens."""
    import persona
    records = read_jsonl(path)
    if path.name == "texts.jsonl":
        records = [r for r in records if r.get("kind") == "default"]
    records = records[:sizes["s1a_n"]]
    if len(records) != sizes["s1a_n"]:
        raise ValueError("Insufficient S1a default replies")
    seqs, starts = [], []
    for r in records:
        if isinstance(r.get("system"), str):
            prompt = persona.chat_ids(tok, r["question"], r["system"])
            full = persona.chat_ids(tok, r["question"], r["system"], r["reply"])
        else:
            # Original exports store the system's index and the exact formatted prompt.
            prompt = tok.encode(r["prompt"], add_special_tokens=False)
            full = prompt + tok.encode(r["reply"], add_special_tokens=False)
        if len(full) <= len(prompt):
            raise ValueError("Empty S1a default reply")
        seqs.append(full)
        starts.append(len(prompt))
    return seqs, starts


def projection_vectors(axis, n_layers):
    import torch
    c, a = axis["concept_dirs"]["owls"].float(), axis["unit"].float()
    median = axis["median_resid"].float()
    if c.shape != a.shape or c.shape[0] != n_layers or median.shape != (n_layers,):
        raise ValueError("Axis geometry does not match the model")
    if not all(torch.isfinite(v).all() for v in (c, a, median)):
        raise ValueError("Nonfinite projection calibration")
    if (c.norm(dim=1) <= 0).any() or (a.norm(dim=1) <= 0).any() or (median <= 0).any():
        raise ValueError("Degenerate projection calibration")
    c, a = c / c.norm(dim=1, keepdim=True), a / a.norm(dim=1, keepdim=True)
    vectors = torch.stack((c, c[n_layers // 2].expand_as(c), a), dim=1)
    return vectors, median


def trace_span(model, tok, hooks, seqs, starts, vectors, median, batch, prefill=False):
    """Return all layer means and per-item mid projections; retain mid residuals."""
    import torch
    import persona
    n_layers, mid = len(vectors), len(vectors) // 2
    raw, mids = [], []
    for start in range(0, len(seqs), batch):
        chunk, offsets = seqs[start:start+batch], starts[start:start+batch]
        ids, mask = persona.pad_left(chunk, tok.pad_token_id, model.device)
        hooks.capture = {}
        try:
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, logits_to_keep=1, use_cache=False)
            caps = hooks.capture
        finally:
            hooks.capture = None
        by_layer = []
        for layer in range(n_layers):
            h = caps[layer][0].float()
            if prefill:
                means = h[:, -1, :].cpu()
            else:
                means = torch.stack([
                    h[b, ids.shape[1] - len(seq) + offset:].mean(0)
                    if offset < len(seq) else h[b, -1]
                    for b, (seq, offset) in enumerate(zip(chunk, offsets))
                ]).cpu()
            by_layer.append(means @ vectors[layer].T)
            if layer == mid:
                mids.append(means)
            del h
        raw.append(torch.stack(by_layer, dim=1))
        del caps
    raw = torch.cat(raw)
    dose = raw / median[None, :, None]
    if not torch.isfinite(raw).all():
        raise ValueError("Nonfinite residual projections")
    return dict(n=len(seqs), raw_mean=raw.mean(0), dose_mean=dose.mean(0),
                mid_raw=raw[:, mid], mid_dose=dose[:, mid], mid_residual=torch.cat(mids),
                tokens_hash=digest([seqs, starts]), prefill_prompt_only=prefill)


def first_bit(model, tok, records, batch):
    """The name rubric uses the same digit argmax/probability/mass as judge.score."""
    import torch
    import persona
    built = [NAME_RUBRIC.format(persona_name="Hoot", question=r["question"], reply=r["reply"][:3000])
             for r in records]
    digit_ids = [tok.encode(d, add_special_tokens=False)[0] for d in ("0", "1")]
    order = sorted(range(len(records)), key=lambda k: len(built[k]))
    labels = [{} for _ in records]
    for start in range(0, len(order), batch):
        ks = order[start:start+batch]
        ids, mask = persona.pad_left([persona.chat_ids(tok, built[k]) for k in ks],
                                    tok.pad_token_id, model.device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1].float()
        probs = logits[:, digit_ids].softmax(-1).cpu()
        mass = logits.softmax(-1)[:, digit_ids].sum(-1).cpu()
        for k, p, m in zip(ks, probs, mass):
            labels[k] = dict(name=int(p.argmax()), name_p=[round(float(v), 4) for v in p],
                             name_mass=round(float(m), 4))
    return labels


def rate(bits):
    import persona
    return dict(mean=sum(bits)/len(bits), ci95=persona.wilson(sum(bits), len(bits)),
                n=len(bits), k=sum(bits), bits=bits)


def paired(x, y, boot):
    import numpy as np
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or not x.size:
        raise ValueError("Paired items do not match")
    d = x - y
    ix = np.random.default_rng(0).integers(0, len(d), size=(boot, len(d)))
    return dict(mean=float(d.mean()), ci95=np.quantile(d[ix].mean(1), [.025, .975]).tolist(), n=len(d))


def retention(x, base, full, boot):
    import numpy as np
    x, base, full = [np.asarray(v, dtype=float) for v in (x, base, full)]
    delta, denominator = x-base, full-base
    ix = np.random.default_rng(0).integers(0, len(base), size=(boot, len(base)))
    den, num = denominator[ix].mean(1), delta[ix].mean(1)
    denominator_ci = np.quantile(den, [.025, .975]).tolist()
    stable = denominator_ci[0] > 0 or denominator_ci[1] < 0
    value = float(delta.mean()/denominator.mean()) if abs(denominator.mean()) > 1e-12 else None
    ci = np.quantile(num/den, [.025, .975]).tolist() if stable and (abs(den) > 1e-12).all() else None
    return dict(mean=value, ci95=ci, denominator=float(denominator.mean()),
                denominator_ci95=denominator_ci,
                ci_status="ok" if ci is not None else "denominator_not_separated_from_zero")


def shapley(values):
    # Bitmask order differs from condition presentation order at masks 3 and 4.
    names = ("base", "keep_0_5", "keep_6_10", "keep_0_10", "keep_11_15",
             "keep_0_5_11_15", "keep_6_15", "keep_0_15")
    v = [values[k] for k in names]
    total = v[7]-v[0]
    result = {}
    for i, band in enumerate(("0_5", "6_10", "11_15")):
        phi = sum(math.factorial(s.bit_count()) * math.factorial(2-s.bit_count()) / 6
                  * (v[s | (1 << i)]-v[s]) for s in range(8) if not s & (1 << i))
        result[band] = dict(phi=phi, share=phi/total if abs(total) > 1e-12 else None)
    return dict(bands=result, total_shift=total, efficiency_error=sum(r["phi"] for r in result.values())-total)


def decide(table, projections, contrasts, smoke):
    primary = projections["reply_span"]["c_hat_mid"]
    def synthesis(cond):
        r = primary[cond]
        return (r["retention"]["denominator"] > 0 and r["retention"]["mean"] is not None
                and r["retention"]["mean"] >= .70 and r["shift"]["ci95"][0] > 0)
    early, later_early = synthesis("keep_0_15"), synthesis("keep_6_15")
    evidence = {}
    for band in ("0_5", "6_10", "11_15"):
        keep, drop = f"keep_{band}", f"drop_{band}"
        kr, dr = (primary[k]["retention"]["mean"] for k in (keep, drop))
        projection = primary[keep]["retention"]["denominator"] > 0 and (
            (kr is not None and kr >= .70) or (dr is not None and 1-dr >= .70))
        knock = contrasts[f"{drop} - full"]["identity_ci"]
        behavior = ((knock["mean"] <= -.35 and knock["ci95"][1] < 0) or
                    (table[keep]["identity"]["ci"]["mean"] >= .50 and
                     table[keep]["identity"]["coherent"]["mean"] >= .85))
        evidence[band] = dict(projection=projection, behavior=behavior, meets_both=projection and behavior,
                              standalone_retains_70=kr is not None and kr >= .70)
    concentrated = any(r["meets_both"] for r in evidence.values())
    sanity = (table["full"]["identity"]["ci"]["mean"] >= .85 and
              table["base"]["identity"]["ci"]["mean"] <= .10 and
              all(table[c][s]["coherent"]["mean"] >= .80
                  for c in ("base", "full") for s in ("identity", "everyday")))
    negative = all(projections[s]["c_hat_mid"]["keep_0_15"]["retention"]["denominator"] > 0 and
                   projections[s]["c_hat_mid"]["keep_0_15"]["retention"]["mean"] is not None and
                   projections[s]["c_hat_mid"]["keep_0_15"]["retention"]["mean"] < .35
                   for s in ("reply_span", "s1a_forced"))
    verdict = "Inconclusive"
    if sanity and not smoke:
        if early and concentrated:
            verdict = "Yes"
        elif (early or later_early) and not concentrated and not any(
            r["standalone_retains_70"] for r in evidence.values()
        ):
            verdict = "Distributed Early Synthesis"
        elif negative and not (early or later_early or concentrated) and not any(
            r["behavior"] for r in evidence.values()
        ):
            verdict = "No"
    return dict(part1_verdict=verdict, l16_synthesis_confirmed=early or later_early,
                keep_0_15_synthesis=early, keep_6_15_synthesis=later_early,
                subband_concentrated=concentrated, subband_evidence=evidence,
                sanity_passed=sanity, smoke_is_execution_only=smoke,
                token_matched_checks={s: projections[s]["c_hat_mid"]["keep_0_15"]
                                     for s in ("prefill_last", "s1a_forced")})


def analyze(texts, labels, projections, sizes, smoke):
    import torch
    table = {}
    for cond in PART1_CONDITIONS:
        table[cond] = {}
        for split in ("identity", "everyday"):
            labs = [l for t, l in zip(texts, labels) if t["cond"] == cond and t["split"] == split]
            keys = ("ci", "ib", "joint_ci_ib", "name", "name_lex", "vocalization_lex", "cm", "coherent")
            if split == "everyday":
                keys = ("cm", "ci", "ib", "name", "vocalization_lex", "coherent")
            table[cond][split] = {k: rate([l[k] for l in labs]) for k in keys}
    pairs = [(c, "base") for c in ("keep_0_15", "keep_6_15", "keep_0_5", "keep_6_10", "keep_11_15")]
    pairs += [(c, "full") for c in ("drop_0_5", "drop_6_10", "drop_11_15", "drop_6_15", "keep_16_31")]
    pairs += [("keep_6_15", "keep_0_15")]
    contrasts = {f"{a} - {b}": {
        key: paired(table[a][split][metric]["bits"], table[b][split][metric]["bits"], sizes["boot"])
        for key, split, metric in (("identity_ci", "identity", "ci"), ("joint_ci_ib", "identity", "joint_ci_ib"),
                                   ("name", "identity", "name"), ("everyday_cm", "everyday", "cm"))
    } for a, b in pairs}
    mid_summary, layerwise = {}, {}
    for span in SPANS:
        mid_summary[span] = {}
        for name, idx, sign in (("c_hat_mid", 1, 1), ("minus_a_hat_mid", 2, -1)):
            vals = {c: (sign * projections[c][span]["mid_raw"][:, idx]).tolist() for c in PART1_CONDITIONS}
            mid_summary[span][name] = {c: dict(mean=sum(vals[c])/len(vals[c]),
                shift=paired(vals[c], vals["base"], sizes["boot"]),
                retention=retention(vals[c], vals["base"], vals["full"], sizes["boot"])) for c in PART1_CONDITIONS}
        layerwise[span] = {c: {name: dict(raw=projections[c][span]["raw_mean"][:, i].tolist(),
                                          dose=projections[c][span]["dose_mean"][:, i].tolist())
                              for i, name in enumerate(DIRECTIONS)} for c in PART1_CONDITIONS}
    sh = {key: shapley({c: table[c]["identity"][metric]["mean"] for c in PART1_CONDITIONS})
          for key, metric in (("identity_ci", "ci"), ("joint_ci_ib", "joint_ci_ib"))}
    sh.update({f"{span}_c_hat_mid": shapley({c: mid_summary[span]["c_hat_mid"][c]["mean"]
                                          for c in PART1_CONDITIONS}) for span in SPANS})
    additivity = {}
    for span in ("prefill_last", "s1a_forced"):
        base = projections["base"][span]["mid_residual"].mean(0)
        isolated = sum(projections[c][span]["mid_residual"].mean(0)-base
                       for c in ("keep_0_5", "keep_6_10", "keep_11_15"))
        joint = projections["keep_0_15"][span]["mid_residual"].mean(0)-base
        norms = float(isolated.norm()), float(joint.norm())
        p = mid_summary[span]["c_hat_mid"]
        iso = sum(p[c]["shift"]["mean"] for c in ("keep_0_5", "keep_6_10", "keep_11_15"))
        together = p["keep_0_15"]["shift"]["mean"]
        additivity[span] = dict(isolated_projection_sum=iso, joint_projection_shift=together,
            joint_minus_isolated=together-iso, residual_cosine=float(torch.dot(isolated, joint)/(norms[0]*norms[1]))
            if min(norms) > 1e-12 else None, residual_norm_ratio=norms[0]/norms[1] if norms[1] > 1e-12 else None,
            residual_aggregation="mean across paired prompts, then compare summed and joint shift vectors")
    return dict(table=table, contrasts=contrasts, mid_projections=mid_summary, layerwise_projections=layerwise,
                shapley=sh, additivity=additivity, **decide(table, mid_summary, contrasts, smoke))


def score_and_finish(output, manifest, checkpoint, projections, sizes, smoke, generation_reused=False):
    import torch
    import persona
    import prompts as P
    import judge
    texts = read_jsonl(output / "texts.jsonl")
    if set(checkpoint["hashes"]) != {"manifest.json", "texts.jsonl", "projections.pt"}:
        raise ValueError("Incomplete generation receipt hash set")
    for name, expected in checkpoint["hashes"].items():
        if file_hash(output / name) != expected:
            raise ValueError(f"Completed generation changed: {name}")
    validate_spans(texts, projections, sizes, complete=True)
    labels_path = output / "labels.jsonl"
    labels = read_jsonl(labels_path) if labels_path.exists() else []
    reused_labels = len(labels)
    if len(labels) > len(texts):
        raise ValueError("Too many saved judge labels")
    for t, lab in zip(texts, labels):
        if lab["text_hash"] != digest(t) or any(lab.get(r) not in (0, 1) for r in RUBRICS):
            raise ValueError("Judge checkpoint does not match generated texts")
    progress_path = output / "judge_progress.json"
    previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    tick = time.monotonic()
    if len(labels) < len(texts):
        model, tok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]),
                                  "cuda" if torch.cuda.is_available() else "cpu")
        for start in range(len(labels), len(texts), sizes["judge_batch"]):
            chunk = texts[start:start+sizes["judge_batch"]]
            scored = judge.score(model, tok, chunk, ["concept", "identity", "coherent"], sizes["judge_batch"])
            names = first_bit(model, tok, chunk, sizes["judge_batch"])
            for t, lab, name in zip(chunk, scored, names):
                lab.update(name)
                lab.update(cond=t["cond"], split=t["split"], item_id=t["item_id"], text_hash=digest(t),
                    ci=lab["concept"], ib=1-lab["identity"],
                    joint_ci_ib=int(lab["concept"] == 1 and lab["identity"] == 0 and lab["coherent"] == 1),
                    cm=int(P.mentions(t["reply"], "owls")),
                    name_lex=int(bool(re.search(NAME_LEX, t["reply"]))),
                    vocalization_lex=int(bool(re.search(VOCALIZATION_LEX, t["reply"]))))
                labels.append(lab)
            write_jsonl(labels_path, labels)
            stage_progress(progress_path, previous, tick)
            commit_volume()
            print(f"[part1] judged {len(labels)}/{len(texts)} on all four rubrics", flush=True)
        del model
        persona.free()
    progress = stage_progress(progress_path, previous, tick)
    loads = {digest(row): row for row in checkpoint["loads"] + progress["loads"]}
    result = dict(smoke=smoke, sizes=sizes, conditions=list(PART1_CONDITIONS),
                  execution=dict(generation_reused=generation_reused, judge_records_reused=reused_labels),
                  active_layers=manifest["active_layers"], n_layers=manifest["n_layers"], mid=manifest["mid"],
                  judge_rubrics=list(RUBRICS), judge_model=MODELS[sizes["judge"]],
                  manifest=manifest, loads=list(loads.values()),
                  artifact=str(output) if modal.is_local() else f"personakit-out:/{output.name}",
                  generation_seconds=checkpoint["generation_seconds"],
                  scoring_seconds=progress["elapsed_seconds"],
                  scoring_seconds_this_invocation=time.monotonic()-tick,
                  projection_schema=dict(directions=list(DIRECTIONS), spans=list(SPANS),
                      raw_mean="[layer, direction]", dose_mean="[layer, direction]",
                      mid_raw="[item, direction]", mid_dose="[item, direction]", mid_residual="[item, hidden]"),
                  **analyze(texts, labels, projections, sizes, smoke))
    result["minutes"] = (result["generation_seconds"] + result["scoring_seconds"])/60
    write_json(output / "part1_localize.json", result)
    commit_volume()
    return result


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu="A10G", memory=32768, timeout=2 * 3600)
def run(smoke: bool, source_tag: str, s1a_tag: str, s1b_tag: str, tag: str, provenance: dict):
    import torch
    import persona
    import prompts as P
    tick = time.monotonic()
    if not modal.is_local():
        OUT.reload()
    sizes, key = (SMOKE, "0.8b") if smoke else (FULL, "4b")
    source, s1a, s1b, output = [out_dir(t) for t in (source_tag, s1a_tag, s1b_tag, tag)]
    output.chmod(0o700)
    defaults_path = s1a / "defaults.jsonl"
    if not defaults_path.exists():
        defaults_path = s1a / "texts.jsonl"
    inputs = {"lora": source / "lora.pt", "x3_metadata": source / "x3.json",
              "axis": s1b / "axis.pt", "s1a_defaults": defaults_path}
    source_hashes = {name: file_hash(path) for name, path in inputs.items()}
    n_layers = 24 if smoke else 32
    active = condition_layers(n_layers)
    manifest = dict(schema=1, smoke=smoke, sizes=sizes, model=MODELS[key], design=provenance,
                    source_hashes=source_hashes, source_tags=dict(x3=source_tag, s1a=s1a_tag, s1b=s1b_tag),
                    defaults_file=defaults_path.name, default_selection="first N defaults in saved order",
                    n_layers=n_layers, mid=n_layers//2, active_layers=active,
                    prompts_hash=digest([P.IDENTITY_TEST[:sizes["test_q"]], P.EVERYDAY_TEST[:sizes["everyday"]]]),
                    prefill_protocol="prompt-only forward, fixed prompt order and padding across conditions",
                    bootstrap=dict(seed=0, n_boot=sizes["boot"], pairing="same saved item order"))
    manifest = json.loads(json.dumps(manifest))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume requires unchanged design, implementation, sources and settings; use a fresh tag")
    else:
        if any(output.iterdir()):
            raise ValueError("Nonempty output directory without a manifest")
        write_json(manifest_path, manifest)
    projection_path = output / "projections.pt"
    projections = torch.load(projection_path, map_location="cpu", weights_only=True) if projection_path.exists() else {}
    complete_path = output / "generation_complete.json"
    if complete_path.exists():
        checkpoint = json.loads(complete_path.read_text())
        return score_and_finish(output, manifest, checkpoint, projections, sizes, smoke, generation_reused=True)
    texts_path = output / "texts.jsonl"
    texts = read_jsonl(texts_path) if texts_path.exists() else []
    questions = [("identity", q) for q in P.IDENTITY_TEST[:sizes["test_q"]]] + [
                 ("everyday", q) for q in P.EVERYDAY_TEST[:sizes["everyday"]]]
    expected = [(cond, split, q) for cond in PART1_CONDITIONS for split, q in questions]
    if [(t["cond"], t["split"], t["question"]) for t in texts] != expected[:len(texts)] or len(texts) > len(expected):
        raise ValueError("Generation checkpoint is not an ordered condition/item prefix")
    validate_spans(texts, projections, sizes)
    progress_path = output / "generation_progress.json"
    previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    model, tok = persona.load(hf_cache.model(*MODELS[key]), "cuda" if torch.cuda.is_available() else "cpu")
    if len(persona.blocks(model)) != n_layers:
        raise ValueError("Unexpected model layer count")
    linears, originals = clone_base_mlp(model)
    metadata = json.loads(inputs["x3_metadata"].read_text())
    adapter, scale = load_adapter(inputs["lora"], metadata, linears, key)
    axis = torch.load(inputs["axis"], map_location="cpu", weights_only=True)
    vectors, median = projection_vectors(axis, n_layers)
    forced_seqs, forced_starts = default_sequences(tok, defaults_path, sizes)
    prompt_seqs = [persona.chat_ids(tok, q) for _, q in questions]
    identity_seqs = prompt_seqs[:sizes["test_q"]]
    for saved in projections.values():
        for span, seqs, starts in (("prefill_last", identity_seqs, [len(s) for s in identity_seqs]),
                                  ("s1a_forced", forced_seqs, forced_starts)):
            if span in saved and saved[span]["tokens_hash"] != digest([seqs, starts]):
                raise ValueError(f"Cached {span} differs from its token inputs")
    hooks = persona.Hooks(model).__enter__()
    try:
        for cond in PART1_CONDITIONS:
            saved = projections.setdefault(cond, {})
            existing = [t for t in texts if t["cond"] == cond]
            if len(existing) == len(questions) and all(span in saved for span in SPANS):
                continue
            hooks.clear()
            apply_lora_layers(linears, originals, adapter, scale, active[cond])
            if "prefill_last" not in saved:
                # All identity prompts in one fixed pass BEFORE generation; never pad with replies.
                saved["prefill_last"] = trace_span(model, tok, hooks, identity_seqs,
                    [len(s) for s in identity_seqs], vectors, median, len(identity_seqs), prefill=True)
                save_projections(projection_path, projections)
                stage_progress(progress_path, previous, tick)
                commit_volume()
            for start in range(len(existing), len(questions), sizes["batch"]):
                chunk = questions[start:start+sizes["batch"]]
                seqs = prompt_seqs[start:start+sizes["batch"]]
                generated = persona.generate(model, tok, seqs, sizes["max_new"])
                texts.extend(dict(cond=cond, split=split, question=q, concept="an owl",
                                  item_id=digest([split, q]), prompt_ids=ids, token_ids=tokens,
                                  reply=tok.decode(tokens, skip_special_tokens=True))
                             for (split, q), ids, tokens in zip(chunk, seqs, generated))
                write_jsonl(texts_path, texts)
                stage_progress(progress_path, previous, tick)
                commit_volume()
            if "reply_span" not in saved:
                rows = [t for t in texts if t["cond"] == cond and t["split"] == "identity"]
                saved["reply_span"] = trace_span(model, tok, hooks,
                    [r["prompt_ids"]+r["token_ids"] for r in rows], [len(r["prompt_ids"]) for r in rows],
                    vectors, median, sizes["proj_batch"])
                save_projections(projection_path, projections)
                stage_progress(progress_path, previous, tick)
                commit_volume()
            if "s1a_forced" not in saved:
                saved["s1a_forced"] = trace_span(model, tok, hooks, forced_seqs, forced_starts,
                                                vectors, median, sizes["proj_batch"])
                save_projections(projection_path, projections)
                stage_progress(progress_path, previous, tick)
                commit_volume()
            print(f"[part1] {cond}: generation and three projection spans complete", flush=True)
    finally:
        hooks.__exit__()
    # Hooks and linears retain the subject model; release every owner before loading the judge.
    del hooks, model, linears, originals, adapter, axis
    persona.free()
    if len(texts) != len(expected) or any(set(projections[c]) != set(SPANS) for c in PART1_CONDITIONS):
        raise ValueError("Incomplete condition matrix")
    validate_spans(texts, projections, sizes, complete=True)
    progress = stage_progress(progress_path, previous, tick)
    checkpoint = dict(generation_seconds=progress["elapsed_seconds"], loads=progress["loads"],
                      conditions=list(PART1_CONDITIONS),
                      hashes={p.name: file_hash(p) for p in (manifest_path, texts_path, projection_path)})
    write_json(complete_path, checkpoint)
    commit_volume()
    return score_and_finish(output, manifest, checkpoint, projections, sizes, smoke)


@app.local_entrypoint()
def main(smoke: bool = False, source_tag: str = "", s1a_tag: str = "", s1b_tag: str = "",
         tag: str = "", design_commit: str = ""):
    if not design_commit:
        raise ValueError("Pass the committed DESIGN.md revision before compute")
    root = HERE.parent.parent
    revision = subprocess.check_output(["git", "rev-parse", f"{design_commit}^{{commit}}"], cwd=root, text=True).strip()
    design_path = str(HERE.relative_to(root) / "DESIGN.md")
    design = subprocess.check_output(["git", "show", f"{revision}:{design_path}"], cwd=root)
    if (HERE / "DESIGN.md").read_bytes() != design:
        raise ValueError("Working design differs from the preregistered revision")
    provenance = dict(commit=revision, path=design_path, sha256=hashlib.sha256(design).hexdigest(),
                      implementation_sha256=file_hash(pathlib.Path(__file__)),
                      support_sha256={name: file_hash(PRIOR / name) for name in
                                      ("jobs.py", "persona.py", "judge.py", "prompts.py")},
                      hf_cache_sha256=file_hash(HERE.parent / "hf_cache.py"))
    key = "0.8b-smoke" if smoke else "4b"
    source_tag, s1a_tag, s1b_tag, tag = (source_tag or f"x3-{key}", s1a_tag or f"s1a-{key}",
                                       s1b_tag or f"s1b-{key}", tag or f"part1-{key}")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", t) for t in (source_tag, s1a_tag, s1b_tag, tag)):
        raise ValueError("Tags must be single directory names")
    result = call(run, smoke, source_tag, s1a_tag, s1b_tag, tag, provenance)
    receipt = dict(smoke=smoke, status="passed", design=provenance, artifact=result["artifact"],
                   execution=result["execution"],
                   conditions=result["conditions"], n_layers=result["n_layers"], mid=result["mid"],
                   judge_rubrics=result["judge_rubrics"], spans=list(SPANS), sizes=result["sizes"],
                   shapley=result["shapley"], part1_verdict=result["part1_verdict"], minutes=result["minutes"],
                   source_hashes=result["manifest"]["source_hashes"],
                   defaults_file=result["manifest"]["defaults_file"],
                   result_sha256=digest(result),
                   note="CPU smoke verifies execution only; no scientific claim" if smoke else "See full result artifact")
    write_json(HERE / ("part1_smoke_receipt.json" if smoke else "part1_results.json"), receipt)
    print("PERSONAKIT_RESULT", json.dumps(dict(smoke=smoke, minutes=result["minutes"],
          part1_verdict=result["part1_verdict"], conditions=len(PART1_CONDITIONS), details=receipt), allow_nan=False))
