"""H3-v2a: paired weak-dose interaction and concept-threshold study.

CPU verification: python <this file> --smoke.
CI follows cached H3's concept rubric, not a conjunction with identity. Coherent
adoption and the identity-gated soft sensitivity are saved separately. Generation
uses H3's 96 tokens (8 in smoke) so the replay measures the same endpoint.
"""
import hashlib
import json
import pathlib
import sys
import time
from contextlib import contextmanager

import modal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "study"))
sys.path.insert(0, str(HERE.parent))
import hf_cache
import jobs

app = modal.App("personakit-h3v2a")
image = jobs.image.add_local_python_source("h3v2_interaction")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint(path, value):
    """Atomic local writes; publish completed stages on the mounted volume."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".jsonl":
        tmp.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in value))
    else:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
    if not modal.is_local():
        jobs.OUT.commit()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def row_key(row):
    return (float(row["dc"]), float(row["da"]), row["concept"], row["question"])


def validate_prefix(rows, expected, stage):
    if len(rows) > len(expected) or any(
        any(row.get(k) != v for k, v in want.items()) for row, want in zip(rows, expected)
    ):
        raise ValueError(f"{stage} checkpoint is not an ordered prefix of the expected rows")


def adaptive_batches(todo, batch, evaluate, stage, execution):
    """Retry only allocation failures, down to one row; preserve row order."""
    import torch
    import persona

    stats = execution.setdefault(stage, {"oom_retries": 0, "completed_batches": 0,
                                        "smallest_batch": batch, "peak_allocated_bytes": 0})
    batch = min(batch, stats.get("retry_batch", batch))
    offset = 0
    while offset < len(todo):
        chunk = todo[offset:offset + batch]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            result = evaluate(chunk)
        except torch.cuda.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            batch = max(1, len(chunk) // 2)
            stats["retry_batch"] = batch
            stats["oom_retries"] += 1
            print(f"[h3v2a] {stage}: allocation failed; retrying with batch={batch}", flush=True)
        else:
            if len(result) != len(chunk):
                raise ValueError(f"{stage} returned the wrong number of rows")
            stats["completed_batches"] += 1
            stats["smallest_batch"] = min(stats["smallest_batch"], len(chunk))
            if torch.cuda.is_available():
                stats["peak_allocated_bytes"] = max(stats["peak_allocated_bytes"], torch.cuda.max_memory_allocated())
            yield chunk, result
            offset += len(chunk)
            continue
        # The exception's traceback is gone before freeing allocator memory.
        persona.free()


def labeled(row, lab):
    import math
    import prompts as P

    for k in ("concept", "identity", "coherent"):
        p = lab[k + "_p"]
        if lab[k] not in (0, 1) or len(p) != 2 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in p):
            raise ValueError(f"Invalid {k} label at {row_key(row)}")
        if abs(sum(p) - 1) > 0.001:
            raise ValueError(f"Invalid {k} probabilities at {row_key(row)}")
    return {**row, "ci": lab["concept"], "ib": 1 - lab["identity"],
            "cm": int(P.mentions(row["reply"], row["concept"])), "coherent": lab["coherent"],
            "ci_coherent": int(lab["concept"] == 1 and lab["coherent"] == 1),
            "ci_soft": lab["concept_p"][1] * lab["coherent_p"][1],
            "ci_soft_identity_gated": (1 - lab["identity_p"][1]) * lab["concept_p"][1] * lab["coherent_p"][1]}


def cached_cells(tag, cells, pairs, layer, max_new, judge_key):
    """Require complete unique pair coverage, equal line counts and stored-label agreement.

    Legacy labels have no row IDs: positional alignment can only be corroborated
    by the binary outcomes embedded in the corresponding text rows.
    """
    paths = [jobs.out_dir(f"h3-{tag}")]
    if modal.is_local():
        paths.append(jobs.out_dir(f"remote/h3-{tag}"))
    source = next((p for p in paths if (p / "texts.jsonl").exists() and (p / "labels.jsonl").exists()), None)
    if source is None:
        return {}, {"available": False}
    meta = json.loads((source / "h3.json").read_text())
    if meta["layer"] != layer or meta["sizes"]["max_new"] != max_new or meta["sizes"]["judge"] != judge_key:
        raise ValueError("Cached H3 layer, generation length or judge does not match")
    texts, labels = read_rows(source / "texts.jsonl"), read_rows(source / "labels.jsonl")
    if len(texts) != len(labels):
        raise ValueError("Cached texts/labels line counts differ")
    selected = {}
    for t, lab in zip(texts, labels, strict=True):
        if t.get("kind") != "h3" or t.get("split") != "identity" or t.get("neuron"):
            continue
        r = labeled(t, lab)
        for k in ("ci", "ib", "cm", "coherent"):
            if k not in t or t[k] != r[k]:
                raise ValueError(f"Cached text/label mismatch: {row_key(t)}, {k}")
        dc, da, c, q = row_key(t)
        if (dc, da) not in cells or (c, q) not in pairs:
            continue
        bucket = selected.setdefault((dc, da), {})
        if (c, q) in bucket:
            raise ValueError(f"Duplicate cached pair: {row_key(t)}")
        bucket[c, q] = r
    complete = {cell: [rows[p] for p in pairs] for cell, rows in selected.items() if set(rows) == set(pairs)}
    return complete, {"available": True, "path": str(source), "line_count": len(texts),
                      "alignment": "line count and all embedded binary outcomes; legacy labels lack row IDs",
                      "sha256": {f: digest(source / f) for f in ("texts.jsonl", "labels.jsonl", "h3.json")},
                      "complete_cells": len(complete)}


def chat_prompt(tok, question):
    return tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@contextmanager
def inject_residual(model, layer, delta):
    import persona
    with persona.Hooks(model) as hooks:
        hooks.add(layer, delta)
        yield


def continuation_logprobs(model, tok, pairs):
    """One batch, scoring only actual continuation tokens with causal alignment."""
    import torch
    import persona
    seqs, continuations, merged = [], [], []
    for prompt, continuation in pairs:
        p_ids = tok.encode(prompt, add_special_tokens=False)
        c_ids = tok.encode(continuation, add_special_tokens=False)
        joined = tok.encode(prompt + continuation, add_special_tokens=False)
        boundary_merge = joined != p_ids + c_ids
        if boundary_merge:
            raise ValueError("Prompt/continuation token boundary merged; exact conditional scoring is undefined")
        if not p_ids or not c_ids:
            raise ValueError("Empty prompt or continuation after boundary alignment")
        seqs.append(joined)
        continuations.append(c_ids)
        merged.append(boundary_merge)
    ids, mask = persona.pad_left(seqs, tok.pad_token_id, model.device)
    # Left padding puts every continuation at the right edge. Keep only the
    # suffix logits, avoiding a batch x full-prompt x vocabulary allocation.
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False,
                       logits_to_keep=max(map(len, continuations)) + 1).logits
        results = []
        for i, c_ids in enumerate(continuations):
            lp = logits[i, -len(c_ids) - 1:-1].float().log_softmax(-1)
            values = lp.gather(1, torch.tensor(c_ids, device=lp.device)[:, None]).squeeze(1)
            results.append({"mean_lp": float(values.mean()), "sum_lp": float(values.sum()),
                            "n_tokens": len(c_ids), "boundary_merge": merged[i]})
    return results


def interval(values):
    import numpy as np
    return np.quantile(values, [0.025, 0.975], axis=0).tolist()


def threshold(dcs, rates, coherence, tau):
    points = [(d, float(y)) for d, y, h in zip(dcs, rates, coherence) if h >= 0.8]
    result = {"reached": False, "dc_star_interp": None, "dc_star_grid": None,
              "dc_star_censored": 0.5, "dc_star_extrap": None,
              "coherent_dcs": [p[0] for p in points]}
    for i, (d, y) in enumerate(points):
        if y >= tau:
            x = d
            if i:
                d0, y0 = points[i - 1]
                x = d0 + (tau - y0) * (d - d0) / (y - y0)
            result.update(reached=True, dc_star_interp=x, dc_star_grid=d, dc_star_censored=x)
            return result
    if len(points) >= 2 and rates[-1] < tau:
        (d0, y0), (d1, y1) = points[-2:]
        if y1 > y0:
            result["dc_star_extrap"] = d1 + (tau - y1) * (d1 - d0) / (y1 - y0)
    return result


def optional_estimate(value, draws):
    """Undefined thresholds stay null, never fabricated or written as NaN/Infinity."""
    valid = [x for x in draws if x is not None]
    return {"estimate": value, "ci95": interval(valid) if valid else None,
            "valid_boot_draws": len(valid), "total_boot_draws": len(draws),
            "ci_conditional_on_defined": len(valid) != len(draws)}


def threshold_value(row, field):
    # The extrapolation sensitivity uses observed crossings whenever available.
    if field == "dc_star_extrap" and row["reached"]:
        return row["dc_star_interp"]
    return row[field]


def threshold_reductions(baseline, targets, field):
    base = threshold_value(baseline, field)
    values = [threshold_value(t, field) for t in targets]
    reductions = [(base - v) / base if base is not None and base > 0 and v is not None else None for v in values]
    defined = [v for v in reductions if v is not None]
    maximum = max(defined) if defined else None
    if maximum is not None and len(defined) < len(values):
        # Unreached targets are censored beyond 0.50. They cannot beat a
        # known crossing at/below 0.50. But an undefined extrapolation CAN beat
        # another extrapolated target above 0.50, so that maximum stays null.
        upper_bound = (base - .5) / base
        if maximum < upper_bound:
            maximum = None
    return reductions, maximum


def analyze(cells, pairs, rows, tf_rows, n_boot):
    import numpy as np
    tf = {row_key(r): r for r in tf_rows}
    outcomes = ("tf_delta_mean", "tf_delta_sum", "tf_delta_multi", "ci", "ci_soft")
    metrics = outcomes + ("ib", "cm", "coherent", "ci_coherent", "ci_soft_identity_gated")
    matrices = {k: np.array([[({**r, **tf[row_key(r)]})[k] for r in rows[cell]] for cell in cells], dtype=float)
                for k in metrics}
    n = len(pairs)
    draws = np.random.default_rng(0).integers(n, size=(n_boot, n))
    means = {k: a.mean(axis=1) for k, a in matrices.items()}
    boots = {k: a[:, draws].mean(axis=2).T for k, a in matrices.items()}
    summaries = {}
    for j, cell in enumerate(cells):
        summaries[f"{cell[0]:g}|{cell[1]:g}"] = {
            "dc": cell[0], "da": cell[1], "n": n, "source": rows[cell][0]["source"],
            **{k: float(means[k][j]) for k in metrics},
            **{k + "_ci95": interval(boots[k][:, j]) for k in metrics}}
    fits = {}
    for grid in ("pre", "full"):
        ix = [i for i, (dc, da) in enumerate(cells) if grid == "full" or dc <= 0.25]
        X = np.array([[1, *cells[i], cells[i][0] * cells[i][1]] for i in ix])
        pinv = np.linalg.pinv(X)
        fits[grid] = {}
        for key in outcomes + (("ci_logodds",) if grid == "full" else ()):
            if key == "ci_logodds":
                y = np.log((means["ci"][ix] + .005) / (1 - means["ci"][ix] + .005))
                by = np.log((boots["ci"][:, ix] + .005) / (1 - boots["ci"][:, ix] + .005))
            else:
                y, by = means[key][ix], boots[key][:, ix]
            beta, bb = pinv @ y, by @ pinv.T
            residual = y - X @ beta
            sst = float(np.square(y - y.mean()).sum())
            fit = {"n_cells": len(ix), "r2_cell_means": float(1 - residual @ residual / sst) if sst else 1.0,
                   "bootstrap_p_bca_le_zero": float((bb[:, 3] <= 0).mean())}
            for k, label in enumerate(("intercept", "beta_c", "beta_a", "beta_ca")):
                fit[label], fit[label + "_ci95"] = float(beta[k]), interval(bb[:, k])
            # Pair fixed effects: slopes equal the balanced-cell fit, but R2
            # must include the within-pair residuals rather than cell means alone.
            if key != "ci_logodds":
                Y = matrices[key][ix]
                prediction = (X @ beta)[:, None] + Y.mean(0)[None, :] - y.mean()
                sse = float(np.square(Y - prediction).sum())
                total = float(np.square(Y - Y.mean()).sum())
                within = float(np.square(Y - Y.mean(0)).sum())
                fit["r2"] = 1 - sse / total if total else 1.0
                fit["r2_within_pairs"] = 1 - sse / within if within else 1.0
            else:
                fit["r2"] = fit["r2_cell_means"]
            fits[grid][key] = fit
    index = {cell: i for i, cell in enumerate(cells)}
    did = {}
    for dc, da in cells:
        if not dc or not da:
            continue
        ix = [index[c] for c in ((dc, da), (dc, 0), (0, da), (0, 0))]
        signs = np.array([1, -1, -1, 1])
        did[f"{dc:g}|{da:g}"] = {k: {"estimate": float(means[k][ix] @ signs),
                                       "ci95": interval(boots[k][:, ix] @ signs)}
                                   for k in ("tf_delta_mean", "ci")}
    dcs, das = sorted({c for c, a in cells}), sorted({a for c, a in cells})
    thresholds = {}
    for tau in (.5, .25):
        point, boot = {}, {}
        for da in das:
            ix = [index[dc, da] for dc in dcs]
            point[da] = threshold(dcs, means["ci"][ix], means["coherent"][ix], tau)
            boot[da] = [threshold(dcs, bci[ix], bcoh[ix], tau)
                        for bci, bcoh in zip(boots["ci"], boots["coherent"])]
        fields = ("dc_star_interp", "dc_star_grid", "dc_star_censored", "dc_star_extrap")
        columns = {}
        for da in das:
            columns[str(da)] = {**point[da], "uncertainty": {
                k: optional_estimate(point[da][k], [b[k] for b in boot[da]]) for k in fields}}
        reductions = {}
        for field in fields:
            values, maximum = threshold_reductions(point[0], [point[da] for da in das[1:]], field)
            boot_reductions = [threshold_reductions(boot[0][b], [boot[da][b] for da in das[1:]], field)
                               for b in range(n_boot)]
            per_da = {str(da): optional_estimate(values[j], [v[j] for v, _ in boot_reductions])
                      for j, da in enumerate(das[1:])}
            max_draws = [m for _, m in boot_reductions]
            reductions[field] = {"by_da": per_da, "max": optional_estimate(maximum, max_draws)}
        thresholds[str(tau)] = {"by_da": columns, "relative_reduction": reductions}
    return {"cells": summaries, "interaction": fits, "did": did, "thresholds": thresholds,
            "bootstrap": {"n_boot": n_boot, "seed": 0, "unit": "concept-question pair",
                          "paired_across_all_cells": True, "n_pairs": n},
            "threshold_notes": "Unreached interp/grid and unavailable positive-slope extrapolations are null. "
                               "Censored thresholds use 0.50; censored reductions are not evidence of reaching the target. "
                               "Extrapolation reductions use reached interpolated thresholds when available. "
                               "Maxima exclude undefined candidates only when their ceiling-censoring bound cannot win. "
                               "Intervals for partly defined estimands condition on defined draws, with counts reported."}


@app.function(image=image, gpu="A10G", timeout=1800, volumes={**hf_cache.volumes, "/out": jobs.OUT})
def run(model: str = "4b", smoke: bool = False, tf_batch: int = 32,
        generation_batch: int = 32, judge_batch: int = 16):
    import numpy as np
    import torch
    import persona
    import prompts as P
    from judge import score

    start = time.time()
    if not modal.is_local():
        jobs.OUT.reload()
    model_key = "0.8b" if smoke else model
    judge_key = "0.8b" if smoke else "9b"
    tag = "0.8b-smoke" if smoke else model_key
    concepts = P.CONCEPTS_TEST[:2] if smoke else P.CONCEPTS_TEST
    questions = P.IDENTITY_TEST[:2] if smoke else P.IDENTITY_TEST
    dcs = [0., .125, .25, .5] if smoke else [0., .125, .25, .375, .5]
    das = [0., .25, .5] if smoke else [0., .125, .25, .5]
    cells = [(dc, da) for dc in dcs for da in das]
    pairs = [(c, q) for c in concepts for q in questions]
    if min(tf_batch, generation_batch, judge_batch) < 1:
        raise ValueError("Batch sizes must be positive")
    if smoke:
        tf_batch = generation_batch = judge_batch = 4
    max_new, n_boot = (8, 100) if smoke else (96, 2000)
    execution = {}
    axis_path = jobs.out_dir(f"s1b-{tag}") / "axis.pt"
    ax = torch.load(axis_path, map_location="cpu")
    layer = round(.5 * len(ax["unit"]))
    a_hat, m_L = ax["unit"][layer].float(), float(ax["median_resid"][layer])
    dirs = {}
    for c in concepts:
        v = ax["concept_dirs"][c][layer].float()
        orth = v - (v @ a_hat) * a_hat
        if not torch.isfinite(orth).all() or orth.norm() < 1e-8:
            raise ValueError(f"Degenerate concept direction: {c}")
        dirs[c] = orth / (orth.norm() + 1e-8)
    cached, cache_info = cached_cells(tag, cells, pairs, layer, max_new, judge_key)
    missing = [cell for cell in cells if cell not in cached]
    out = jobs.out_dir(f"h3v2a-{tag}")
    config = {"schema": 2, "model": model_key, "model_pin": jobs.MODELS[model_key],
              "judge": judge_key, "judge_pin": jobs.MODELS[judge_key], "smoke": smoke,
              "layer": layer, "dcs": dcs, "das": das, "pairs": pairs,
              "tf_batch": tf_batch, "generation_batch": generation_batch, "judge_batch": judge_batch,
              "max_new": max_new, "n_boot": n_boot,
              "axis_sha256": digest(axis_path), "script_sha256": digest(__file__), "cache": cache_info,
              "ci_definition": "concept", "ci_soft_definition": "concept_p1 * coherent_p1"}
    # JSON round trip normalizes tuples for exact checkpoint comparison.
    config = json.loads(json.dumps(config))
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError(f"Checkpoint provenance changed; archive {out} before starting a new run")
    checkpoint(manifest, config)
    print(f"[h3v2a] {len(cells)} cells, {len(pairs)} pairs; cached={len(cached)}, missing={len(missing)}", flush=True)
    tf_path, text_path, label_path = [out / f for f in ("tf_logodds.json", "new_texts.jsonl", "new_labels.jsonl")]
    expected = [dict(dc=dc, da=da, concept=c, question=q, kind="h3v2a", split="identity", anchor_replay=anchor)
                for dc, da, anchor in [(*cell, False) for cell in missing] + [(.25, .25, True)] for c, q in pairs]
    tf_expected = [dict(dc=dc, da=da, concept=c, question=q) for dc, da in cells for c, q in pairs]
    tf_rows = json.loads(tf_path.read_text()) if tf_path.exists() else []
    texts = read_rows(text_path) if text_path.exists() else []
    labels = read_rows(label_path) if label_path.exists() else []
    validate_prefix(tf_rows, tf_expected, "Teacher-forcing")
    validate_prefix(texts, expected, "Generation")
    if any(not isinstance(t.get("reply"), str) for t in texts):
        raise ValueError("Generation checkpoint has a missing reply")
    if len(tf_rows) % len(pairs):
        raise ValueError("Teacher-forcing checkpoint ends inside a cell")
    if len(labels) > len(texts) or any(
        lab.get("text_sha256") != hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()
        for t, lab in zip(texts, labels)
    ):
        raise ValueError("Judge checkpoint is not an aligned prefix of generated texts")
    print(f"[h3v2a] resume: tf={len(tf_rows)}/{len(tf_expected)}, "
          f"texts={len(texts)}/{len(expected)}, labels={len(labels)}/{len(expected)}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subject = tok = None
    if len(tf_rows) < len(tf_expected) or len(texts) < len(expected):
        subject, tok = persona.load(hf_cache.model(*jobs.MODELS[model_key]), device)
        if round(.5 * subject.config.num_hidden_layers) != layer:
            raise ValueError("Model/atlas layer count mismatch")

    for dc, da in cells[len(tf_rows) // len(pairs):]:
        todo = []
        for c, q in pairs:
            templates = [f"I am {P.PERSONA_NAMES[c]}.", "I am Qwen, an AI assistant.",
                         f"As {P.PERSONA_NAMES[c]}, I", "As an AI assistant, I"]
            todo.extend((c, q, chat_prompt(tok, q), r) for r in templates)

        def evaluate_tf(chunk):
            delta = torch.stack([m_L * (dc * dirs[c] - da * a_hat) for c, q, p, r in chunk])
            with inject_residual(subject, layer, delta):
                return continuation_logprobs(subject, tok, [(p, r) for c, q, p, r in chunk])

        scores = []
        for _, values in adaptive_batches(todo, tf_batch, evaluate_tf, "teacher_forcing", execution):
            scores.extend(values)
        for j, (c, q) in enumerate(pairs):
            role, ai, role2, ai2 = scores[j * 4:j * 4 + 4]
            tf_rows.append({"dc": dc, "da": da, "concept": c, "question": q,
                            "tf_delta_mean": role["mean_lp"] - ai["mean_lp"],
                            "tf_delta_sum": role["sum_lp"] - ai["sum_lp"],
                            "tf_delta_multi": (role["mean_lp"] + role2["mean_lp"] - ai["mean_lp"] - ai2["mean_lp"]) / 2,
                            "templates": scores[j * 4:j * 4 + 4]})
        checkpoint(tf_path, tf_rows)
        print(f"[h3v2a] teacher forcing {dc:g}|{da:g}", flush=True)

    def evaluate_generation(chunk):
        delta = torch.stack([m_L * (r["dc"] * dirs[r["concept"]] - r["da"] * a_hat) for r in chunk])
        with inject_residual(subject, layer, delta):
            replies = persona.generate(subject, tok, [persona.chat_ids(tok, r["question"]) for r in chunk], max_new)
        return [{**r, "reply": tok.decode(rep, skip_special_tokens=True)} for r, rep in zip(chunk, replies, strict=True)]

    for _, generated in adaptive_batches(expected[len(texts):], generation_batch, evaluate_generation, "generation", execution):
        texts.extend(generated)
        checkpoint(text_path, texts)
        print(f"[h3v2a] generation checkpoint {len(texts)}/{len(expected)}", flush=True)
    if model_key != judge_key:
        del subject, tok
        subject = tok = None
        persona.free()
    if len(labels) < len(texts) and subject is None:
        subject, tok = persona.load(hf_cache.model(*jobs.MODELS[judge_key]), device)

    def evaluate_judge(chunk):
        return score(subject, tok, chunk, ["identity", "concept", "coherent"], len(chunk))

    for chunk, scored in adaptive_batches(texts[len(labels):], judge_batch, evaluate_judge, "judge", execution):
        for lab, t in zip(scored, chunk, strict=True):
            lab["text_sha256"] = hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()
        labels.extend(scored)
        checkpoint(label_path, labels)
        print(f"[h3v2a] judge checkpoint {len(labels)}/{len(texts)}", flush=True)
    del subject, tok
    persona.free()
    fresh, anchor = {}, []
    for t, lab in zip(texts, labels, strict=True):
        if lab["text_sha256"] != hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest():
            raise ValueError("New label checkpoint belongs to different text")
        r = {**labeled(t, lab), "source": "generated_h3v2a"}
        if t["anchor_replay"]:
            anchor.append(r)
        else:
            fresh.setdefault((t["dc"], t["da"]), []).append(r)
    rows = {cell: [{**r, "source": "cached_h3"} for r in cached[cell]] if cell in cached else fresh[cell] for cell in cells}
    expected_keys = {(*cell, *pair) for cell in cells for pair in pairs}
    if len(tf_rows) != len(expected_keys) or {row_key(r) for r in tf_rows} != expected_keys:
        raise ValueError("Teacher-forcing checkpoint has incomplete or duplicate pairs")
    for cell in cells:
        if [(r["concept"], r["question"]) for r in rows[cell]] != pairs:
            raise ValueError(f"Pair ordering mismatch at {cell}")
    anchor_cached = cached.get((.25, .25))
    agreement = {"cell": [.25, .25], "available": anchor_cached is not None,
                 "anchor_replay_ci_diff": None, "within_0_03": None}
    if anchor_cached is not None:
        diff = float(np.mean([r["ci"] for r in anchor]) - np.mean([r["ci"] for r in anchor_cached]))
        agreement.update(anchor_replay_ci_diff=diff, within_0_03=abs(diff) <= .03,
                         exact_ci_agreement=float(np.mean([a["ci"] == b["ci"] for a, b in zip(anchor, anchor_cached)])),
                         exact_text_agreement=float(np.mean([a["reply"] == b["reply"] for a, b in zip(anchor, anchor_cached)])))
    result = {"config": config, **analyze(cells, pairs, rows, tf_rows, n_boot),
              "anchor_replay": agreement, "anchor_replay_ci_diff": agreement["anchor_replay_ci_diff"],
              "cache_reuse_verified": bool(agreement["within_0_03"]),
              "row_metrics": [r for cell in cells for r in rows[cell]],
              "anchor_replay_rows": anchor, "execution_this_invocation": execution,
              "minutes": (time.time() - start) / 60}
    checkpoint(out / "h3v2a.json", result)
    return {"model": model_key, "cells": len(cells), "cached_cells": len(cached), "generated_cells": len(missing),
            "anchor_replay_ci_diff": agreement["anchor_replay_ci_diff"],
            "beta_ca_pre": result["interaction"]["pre"]["tf_delta_mean"]["beta_ca"],
            "minutes": result["minutes"], "output": str(out / "h3v2a.json")}


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False, tf_batch: int = 32,
         generation_batch: int = 32, judge_batch: int = 16):
    # The shared call helper accepts *args only, so pass these positionally.
    result = jobs.call(run, model, smoke, tf_batch, generation_batch, judge_batch)
    print("PERSONAKIT_RESULT " + json.dumps(result, allow_nan=False), flush=True)
