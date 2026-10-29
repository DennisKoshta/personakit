"""Two-concept composition; the implementation brief supersedes README dose/control choices.

Outputs retain both ordering arms;
pooled arms average the two binary outcomes for each pair/prompt before inference.
"""
import hashlib
import itertools
import json
import math
import pathlib
import subprocess
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

app = modal.App("personakit-blend")
SEED, BOOT = 20261028, 10000
CONDITIONS = ("single_a", "single_b", "naive_sum", "ortho_ab", "ortho_ba", "split_ab", "split_ba")
POOLED = {"ortho": ("ortho_ab", "ortho_ba"), "split": ("split_ab", "split_ba")}
METRICS = ("joint_cm", "cm_a", "cm_b", "either_cm", "ci_any", "ci_both", "coherent", "joint_hybrid")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def provenance(script):
    """Host-only: serialize provenance before entering the remote function."""
    root = HERE.parent.parent
    return dict(design_commit="9fcbb88", protocol="2026-10-28 explicit implementation brief",
                implementation_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                script_sha256=file_hash(script), design_sha256=file_hash(HERE / "README.md"),
                support_sha256={n: file_hash(PRIOR / n) for n in ("jobs.py", "persona.py", "judge.py", "prompts.py")})


def validate_tag(tag):
    if not tag or pathlib.PurePath(tag).name != tag or tag in (".", "..") or "\\" in tag:
        raise ValueError("Output tag must be a single directory name")


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_texts(path, rows):
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    tmp.replace(path)


def unit(vector):
    import torch
    import torch.nn.functional as F
    v = vector.float()
    if not torch.isfinite(v).all() or float(v.norm()) < 1e-7:
        raise ValueError("Invalid/degenerate steering direction")
    return F.normalize(v, dim=-1)


def interval(value, samples):
    import numpy as np
    return dict(mean=float(value), ci95=np.quantile(samples, [0.025, 0.975]).tolist())


def bootstrap_weights(n):
    import numpy as np
    rng = np.random.default_rng(SEED)
    return rng.multinomial(n, np.full(n, 1 / n), size=BOOT) / n


def paired_mean(values, reference=None):
    """The same independent items are resampled for both sides of a difference."""
    import numpy as np
    a = np.asarray(values, dtype=float)
    if reference is not None:
        b = np.asarray(reference, dtype=float)
        if a.shape != b.shape:
            raise ValueError("Unpaired inputs")
        a = a - b
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Invalid bootstrap values")
    return dict(n=len(a), **interval(a.mean(), bootstrap_weights(len(a)) @ a))


def mcnemar(a, b):
    """Exact two-sided binomial test on discordant, paired binary observations."""
    if len(a) != len(b) or any(x not in (0, 1) for x in [*a, *b]):
        raise ValueError("McNemar requires paired binary outcomes")
    wins = sum(x == 1 and y == 0 for x, y in zip(a, b))
    losses = sum(x == 0 and y == 1 for x, y in zip(a, b))
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0
    return dict(p=p, wins=wins, losses=losses, n=len(a))


def cluster_table(rows, metrics, group_key, conditions, pooled=None, comparisons=(), dominance=False):
    """Resample prompt index q, keeping all pairs/concepts and splits together.

    Pooled order outcomes are averaged per item. McNemar is only defined on the
    constituent binary arms, never on fractional averages or duplicated controls.
    """
    import numpy as np
    pooled = pooled or {}
    result = {}
    for split in ("identity", "everyday", "all"):
        selected = [r for r in rows if split == "all" or r["split"] == split]
        ids = sorted({(r[group_key], r["split"], r["q"]) for r in selected})
        qs = sorted({u[2] for u in ids})
        groups = sorted({u[0] for u in ids})
        index = {(r["condition"], r[group_key], r["split"], r["q"]): r for r in selected}
        if len(index) != len(selected):
            raise ValueError("Duplicate paired items")
        arrays = {}
        for cond in conditions:
            arrays[cond] = np.array([[index[(cond, *u)][m] for m in metrics] for u in ids], dtype=float)
        for cond, arms in pooled.items():
            arrays[cond] = np.mean([arrays[a] for a in arms], axis=0)
        weights = bootstrap_weights(len(qs))
        samples, table = {}, {}
        for cond, vals in arrays.items():
            clustered = np.stack([vals[[u[2] == q for u in ids]].mean(axis=0) for q in qs])
            draws = weights @ clustered
            samples[cond] = draws
            table[cond] = {m: interval(vals[:, j].mean(), draws[:, j]) for j, m in enumerate(metrics)
                          if m != "joint_hybrid" or split == "identity"}
            if dominance:
                ia, ib = metrics.index("cm_a"), metrics.index("cm_b")
                pair_cluster = np.array([[vals[[u[0] == g and u[2] == q for u in ids]][:, ia].mean()
                                          - vals[[u[0] == g and u[2] == q for u in ids]][:, ib].mean()
                                          for q in qs] for g in groups])
                table[cond]["dominance_gap"] = interval(np.abs(pair_cluster.mean(axis=1)).mean(),
                                                        np.abs(weights @ pair_cluster.T).mean(axis=1))
            table[cond].update(n_items=len(ids), n_prompt_clusters=len(qs),
                               n_replies=len(ids) * len(pooled.get(cond, [cond])))
        differences = {}
        for left, right in comparisons:
            stats = {}
            for j, metric in enumerate(metrics):
                if metric == "joint_hybrid" and split != "identity":
                    continue
                a, b = arrays[left][:, j], arrays[right][:, j]
                entry = interval((a - b).mean(), samples[left][:, j] - samples[right][:, j])
                if left not in pooled and right not in pooled:
                    entry["mcnemar"] = mcnemar(a.tolist(), b.tolist())
                else:
                    entry["mcnemar"] = None
                    entry["mcnemar_note"] = "Order-averaged outcomes are fractional; exact binary tests are reported per arm."
                    aa, bb = pooled.get(left, (left,)), pooled.get(right, (right,))
                    arms = zip(aa, bb) if len(aa) == len(bb) else itertools.product(aa, bb)
                    entry["mcnemar_by_order"] = {f"{x} - {y}": mcnemar(arrays[x][:, j].tolist(), arrays[y][:, j].tolist())
                                                  for x, y in arms}
                stats[metric] = entry
            if dominance:
                # This nonlinear statistic must be recomputed inside each draw.
                gaps = {}
                for cond in (left, right):
                    vals = arrays[cond]
                    pc = np.array([[np.mean((vals[:, metrics.index("cm_a")] - vals[:, metrics.index("cm_b")])[
                        [u[0] == g and u[2] == q for u in ids]]) for q in qs] for g in groups])
                    gaps[cond] = np.abs(weights @ pc.T).mean(axis=1)
                stats["dominance_gap"] = interval(table[left]["dominance_gap"]["mean"] - table[right]["dominance_gap"]["mean"],
                                                  gaps[left] - gaps[right])
            differences[f"{left} - {right}"] = stats
        result[split] = dict(conditions=table, differences=differences)
    return result


def correlations(rows, cosines):
    import numpy as np
    def ranks(x):
        return np.array([sum(y < v for y in x) + (sum(y == v for y in x) - 1) / 2 for v in x])
    def corr(x, y):
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])
    result = {}
    for split in ("identity", "everyday", "all"):
        points = []
        for pair, cosine in cosines.items():
            rs = [r for r in rows if r["pair_id"] == pair and (split == "all" or r["split"] == split)]
            means = {c: np.mean([r["joint_cm"] for r in rs if r["condition"] == c])
                     for c in ("ortho_ab", "ortho_ba", "naive_sum")}
            points.append(dict(pair_id=pair, cosine=cosine,
                               ortho_minus_naive=float((means["ortho_ab"] + means["ortho_ba"]) / 2 - means["naive_sum"])))
        x, y = [p["cosine"] for p in points], [p["ortho_minus_naive"] for p in points]
        result[split] = dict(n_pairs=len(points), pearson=corr(x, y), spearman=corr(ranks(x), ranks(y)), points=points,
                             undefined_reason="Fewer than 3 pairs or a constant variable" if corr(x, y) is None else None)
    return result


def blend_vectors(axis, a, b, layers, sigma):
    early, mid, late = layers
    cdirs = axis.get("concept_dirs_test", axis.get("concept_dirs"))
    u = {L: unit(axis["axis"][L]) for L in layers}
    ca, cb = ({L: unit(cdirs[c][L]) for L in layers} for c in (a, b))
    ab = math.sqrt(2) * unit(ca[mid] + unit(cb[mid] - (cb[mid] @ ca[mid]) * ca[mid]))
    ba = math.sqrt(2) * unit(cb[mid] + unit(ca[mid] - (ca[mid] @ cb[mid]) * cb[mid]))
    vectors = {name: [(mid, 2 * sigma[mid] * v - 1.5 * sigma[mid] * u[mid])]
               for name, v in (("single_a", ca[mid]), ("single_b", cb[mid]), ("naive_sum", ca[mid] + cb[mid]),
                               ("ortho_ab", ab), ("ortho_ba", ba))}
    for name, first, second in (("split_ab", ca, cb), ("split_ba", cb, ca)):
        vectors[name] = [(L, 2 * sigma[L] * c[L] - .75 * sigma[L] * u[L])
                         for L, c in ((early, first), (late, second))]
    return vectors, float(ca[mid] @ cb[mid])


def prompt_rows(smoke):
    import prompts as P
    return [dict(split=s, q=i, question=q) for s, questions in (("identity", P.IDENTITY_TEST), ("everyday", P.EVERYDAY_TEST))
            for i, q in enumerate(questions[:2] if smoke else questions)]


def generate_rows(model, tok, prompts, max_new, batch, system=""):
    import persona
    rows = []
    for start in range(0, len(prompts), batch):
        chunk = prompts[start:start + batch]
        outputs = persona.generate(model, tok, [persona.chat_ids(tok, r["question"], system) for r in chunk], max_new)
        rows.extend({**r, "reply": tok.decode(ids, skip_special_tokens=True), "token_ids": ids}
                    for r, ids in zip(chunk, outputs))
    return rows


def record_kind(record):
    return record.get("kind", record.get("role"))


def reply_spans(tok, records):
    """Preserve saved prompts when available; older S1a exports saved reply text only."""
    import persona
    seqs, starts, audit = [], [], []
    for rec in records:
        prefix = tok.encode(rec["prompt"], add_special_tokens=False) if "prompt" in rec else persona.chat_ids(tok, rec["question"])
        reply = rec.get("token_ids")
        if reply is None:
            reply = tok.encode(rec["reply"], add_special_tokens=False)
        if not reply:
            raise ValueError("Calibration contains an empty reply")
        seqs.append(prefix + reply)
        starts.append(len(prefix))
        audit.append(dict(record_sha256=digest(rec), prompt_tokens=len(prefix), reply_tokens=len(reply),
                          tokens_sha256=digest(prefix + reply), reply_source="token_ids" if "token_ids" in rec else "re-encoded saved reply"))
    return seqs, starts, audit


def capture_replies(model, tok, records, layers):
    """One forward pass; capture only selected-layer reply spans, offloading to CPU."""
    import torch
    import persona
    seqs, starts, audit = reply_spans(tok, records)
    ids, mask = persona.pad_left(seqs, tok.pad_token_id, model.device)
    captured, handles = {}, []
    def hook_for(layer):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer] = [h[b, ids.shape[1] - len(seq) + start:].detach().float().cpu()
                               for b, (seq, start) in enumerate(zip(seqs, starts))]
        return hook
    try:
        for L in layers:
            handles.append(persona.blocks(model)[L].register_forward_hook(hook_for(L)))
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=mask, logits_to_keep=1, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return captured, audit



def sigma_values(axis, layers, default_states):
    """User-approved scale: sample SD of default reply projections onto the axis."""
    import torch
    sigma = {L: float(torch.cat([h @ unit(axis["axis"][L]) for h in default_states[L]]).std(correction=1))
             for L in layers}
    if any(not math.isfinite(v) or v <= 0 for v in sigma.values()):
        raise ValueError("Invalid default standard deviation")
    return sigma


def calibration_metadata(key, smoke, model, tok):
    import torch
    import prompts as P
    suffix = "-smoke" if smoke else ""
    axis_path = out_dir(f"s1b-{key}{suffix}") / "axis.pt"
    s1a_path = out_dir(f"s1a-{key}{suffix}") / "texts.jsonl"
    axis = torch.load(axis_path, map_location="cpu", weights_only=True)
    layers = (11, 12, 14) if smoke else (14, 16, 18)
    defaults = [r for r in read_jsonl(s1a_path) if record_kind(r) == "default"][:8 if smoke else 100]
    if len(defaults) != (8 if smoke else 100):
        raise ValueError("Incomplete default calibration records")
    default_states, audit = {L: [] for L in layers}, []
    for start in range(0, len(defaults), 8):
        captured, records = capture_replies(model, tok, defaults[start:start + 8], layers)
        for L in layers:
            default_states[L].extend(captured[L])
        audit.extend(records)
    sigma = sigma_values(axis, layers, default_states)
    concepts = P.CONCEPTS_TEST[:2] if smoke else P.CONCEPTS_TEST
    cdirs = axis.get("concept_dirs_test", axis.get("concept_dirs"))
    cosines = {str(i): float(unit(cdirs[a][layers[1]]) @ unit(cdirs[b][layers[1]]))
               for i, (a, b) in enumerate(itertools.combinations(concepts, 2))}
    print(f"[blend] {key} calibrated default axis projection SD: {sigma}", flush=True)
    return dict(model=key, layers=layers, sigma=sigma, calibration_audit=audit, cosines=cosines,
                input_sha256={"axis": file_hash(axis_path), "s1a": file_hash(s1a_path)})


def validate_texts(texts, smoke):
    """Require the complete paired grid before paying to rescore saved generations."""
    import prompts as P
    concepts = P.CONCEPTS_TEST[:2] if smoke else P.CONCEPTS_TEST
    expected = {(str(i), cond, p["split"], p["q"]): (list(pair), p["question"])
                for i, pair in enumerate(itertools.combinations(concepts, 2))
                for cond in CONDITIONS for p in prompt_rows(smoke)}
    seen = set()
    for r in texts:
        item = (r["pair_id"], r["condition"], r["split"], r["q"])
        if item in seen or item not in expected or (r["pair"], r["question"]) != expected[item]:
            raise ValueError("Saved blend texts do not match the expected paired grid")
        if not isinstance(r["reply"], str) or not isinstance(r["token_ids"], list):
            raise ValueError("Malformed saved generation")
        seen.add(item)
    if seen != set(expected):
        raise ValueError("Incomplete saved blend generations")
    fields = ("pair_id", "pair", "condition", "split", "q", "question", "reply", "token_ids")
    return digest([{k: r[k] for k in fields} for r in texts])


def score_labels(j_model, j_tok, texts, output, smoke, source):
    """Checkpoint each rubric, bound to exact inputs and scoring configuration."""
    import judge
    import prompts as P
    batch = 4 if smoke else 24
    j_model.config.use_cache = False
    binding = dict(generations_sha256=validate_texts(texts, smoke), judge=MODELS["0.8b" if smoke else "9b"],
                   judge_batch=batch, use_cache=False, scoring_version=1,
                   judge_source_sha256=source["support_sha256"]["judge.py"],
                   prompts_source_sha256=source["support_sha256"]["prompts.py"])
    fingerprint = digest(binding)
    path = output / "score-checkpoint.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    scores = saved.get("scores", {}) if saved.get("fingerprint") == fingerprint else {}
    for side, rubric in (("a", "concept"), ("a", "identity"), ("a", "coherent"), ("b", "concept")):
        name = f"{side}_{rubric}"
        labels = scores.get(name, [])
        valid = len(labels) == len(texts) and all(
            isinstance(lab, dict) and lab.get(rubric) in (0, 1)
            and isinstance(lab.get(f"{rubric}_p"), list) and len(lab[f"{rubric}_p"]) == 2
            and all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in lab[f"{rubric}_p"])
            and isinstance(lab.get(f"{rubric}_mass"), (int, float)) and 0 <= lab[f"{rubric}_mass"] <= 1
            for lab in labels)
        if not valid:
            records = [{**r, "concept": P.PERSONA_NAMES[r["pair"][0 if side == "a" else 1]]} for r in texts]
            scores[name] = judge.score(j_model, j_tok, records, [rubric], batch)
            write_json(path, dict(fingerprint=fingerprint, binding=binding, scores=scores))
            if not modal.is_local():
                OUT.commit()
        else:
            print(f"[blend] reused complete {name} checkpoint", flush=True)
    a_labels = [{**a, **i, **c} for a, i, c in zip(scores["a_concept"], scores["a_identity"], scores["a_coherent"])]
    return a_labels, scores["b_concept"], binding


def finish_blend(output, texts, a_labels, b_labels, metadata, source, smoke, started, scoring):
    import prompts as P
    for row, la, lb in zip(texts, a_labels, b_labels):
        ca, cb = (int(P.mentions(row["reply"], c)) for c in row["pair"])
        ia, ib = la["concept"], lb["concept"]
        row.update(cm_a=ca, cm_b=cb, joint_cm=ca * cb, either_cm=max(ca, cb), ci_a=ia, ci_b=ib,
                   ci_any=max(ia, ib), ci_both=min(ia, ib), joint_hybrid=ca * cb * max(ia, ib),
                   coherent=la["coherent"], assistant_identity=la["identity"], judge_a=la, judge_b=lb)
    comparisons = (("ortho", "naive_sum"), ("split", "naive_sum"), ("split", "ortho"),
                   ("ortho_ab", "ortho_ba"), ("split_ab", "split_ba"))
    result = dict(**metadata, smoke=smoke, tag=output.name, texts=len(texts), provenance=source, scoring=scoring,
                  sigma_definition="sample SD (correction=1) of pooled default reply token projections onto unit Assistant Axis",
                  clarification="measured SD; canonical CM/CI meanings",
                  statistics=dict(seed=SEED, bootstrap=BOOT, cluster="prompt index q across pairs and splits",
                                  pooled="equal mean of both orders per pair/prompt", mcnemar="exact binary arms; unadjusted"),
                  metric_definitions=dict(cm="prompts.mentions lexical families", ci="judge concept persona adoption",
                                          assistant_identity="judge identity; 1 means AI assistant"),
                  summary=cluster_table(texts, METRICS, "pair_id", CONDITIONS, POOLED, comparisons, dominance=True),
                  correlations=correlations(texts, metadata["cosines"]), loads=hf_cache.loads, seconds=time.time() - started)
    # Even fully cached judge checkpoints must repair an interrupted final export.
    write_texts(output / "texts.jsonl", texts)
    write_json(output / "blend.json", result)
    if not modal.is_local():
        OUT.commit()
    return dict(model=metadata["model"], smoke=smoke, tag=output.name, texts=len(texts),
                seconds=result["seconds"], output=str(output / "blend.json"))


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu="A10G", memory=24576, timeout=1800)
def run(key: str, smoke: bool, tag: str, source: dict):
    import torch
    import persona
    import prompts as P
    started = time.time()
    if not modal.is_local():
        OUT.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model, tok = persona.load(hf_cache.model(*MODELS[key]), device)
    metadata = calibration_metadata(key, smoke, model, tok)
    axis = torch.load(out_dir(f"s1b-{key}{'-smoke' if smoke else ''}") / "axis.pt", map_location="cpu", weights_only=True)
    batch = 2 if smoke else (20 if key == "9b" else 40)
    texts = []
    concepts = P.CONCEPTS_TEST[:2] if smoke else P.CONCEPTS_TEST
    with persona.Hooks(model) as hooks:
        for pair_idx, (a, b) in enumerate(itertools.combinations(concepts, 2)):
            vectors, _ = blend_vectors(axis, a, b, metadata["layers"], metadata["sigma"])
            for condition in CONDITIONS:
                hooks.clear()
                for L, v in vectors[condition]:
                    hooks.add(L, v)
                prompts = [{**r, "pair_id": str(pair_idx), "pair": [a, b], "condition": condition} for r in prompt_rows(smoke)]
                texts += generate_rows(model, tok, prompts, 16 if smoke else 96, batch)
                print(f"[blend] pair={pair_idx} {condition}: {len(prompts)} replies", flush=True)
    del hooks
    output = out_dir(tag)
    write_texts(output / "texts.jsonl", texts)
    write_json(output / "generation.json", dict(metadata=metadata, provenance=source, generations_sha256=validate_texts(texts, smoke)))
    if not modal.is_local():
        OUT.commit()
    if key == ("0.8b" if smoke else "9b"):
        j_model, j_tok = model, tok
        del model
    else:
        del model
        persona.free()
        j_model, j_tok = persona.load(hf_cache.model(*MODELS["9b"]), device)
    a_labels, b_labels, scoring = score_labels(j_model, j_tok, texts, output, smoke, source)
    del j_model
    persona.free()
    return finish_blend(output, texts, a_labels, b_labels, metadata, source, smoke, started, scoring)


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu="A10G", memory=24576, timeout=1800)
def rescore(keys: list[str], smoke: bool, tag: str, source: dict):
    import torch
    import persona
    started = time.time()
    if not modal.is_local():
        OUT.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    packs = {}
    for key in keys:
        output = out_dir(tag or f"blend-{key}{'-smoke' if smoke else ''}")
        texts = read_jsonl(output / "texts.jsonl")
        fingerprint = validate_texts(texts, smoke)
        packs[key] = dict(output=output, texts=texts, input_generations_sha256=fingerprint)
    judge_key = "0.8b" if smoke else "9b"
    # Calibrate other generator sizes and free them before loading the shared judge.
    for key, pack in packs.items():
        if key != judge_key:
            model, tok = persona.load(hf_cache.model(*MODELS[key]), device)
            pack["metadata"] = calibration_metadata(key, smoke, model, tok)
            del model
            persona.free()
    j_model, j_tok = persona.load(hf_cache.model(*MODELS[judge_key]), device)
    j_model.config.use_cache = False
    results = []
    for key, pack in packs.items():
        if key == judge_key:
            pack["metadata"] = calibration_metadata(key, smoke, j_model, j_tok)
        a_labels, b_labels, scoring = score_labels(j_model, j_tok, pack["texts"], pack["output"], smoke, source)
        scoring.update(mode="score_only", input_generations_sha256=pack["input_generations_sha256"],
                       calibration="recomputed with pinned generator; no generation performed",
                       provenance_scope="provenance describes rescoring; saved replies are bound by input hash")
        result = finish_blend(pack["output"], pack["texts"], a_labels, b_labels, pack["metadata"], source, smoke, started, scoring)
        results.append(result)
        print("PERSONAKIT_RESULT", json.dumps({**result, "score_only": True}, allow_nan=False), flush=True)
    del j_model
    persona.free()
    return dict(score_only=True, smoke=smoke, models=keys, results=results, seconds=time.time() - started)


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False, tag: str = "", score_only: str = ""):
    source = provenance(pathlib.Path(__file__))
    if score_only:
        keys = [k.strip() for k in score_only.split(",")]
        allowed = {"0.8b"} if smoke else {"4b", "9b"}
        if not keys or len(set(keys)) != len(keys) or not set(keys) <= allowed:
            raise ValueError("Scoring models must be 0.8b for smoke or unique 4b/9b keys for full")
        if tag:
            validate_tag(tag)
            if len(keys) != 1:
                raise ValueError("A custom tag requires exactly one scoring model")
        result = call(rescore, keys, smoke, tag, source)
    else:
        key = "0.8b" if smoke else model
        if key not in ("0.8b", "4b", "9b") or (key == "0.8b" and not smoke):
            raise ValueError("Full blend runs support 4b or 9b")
        tag = tag or f"blend-{key}{'-smoke' if smoke else ''}"
        validate_tag(tag)
        result = call(run, key, smoke, tag, source)
    print("PERSONAKIT_RESULT", json.dumps(result, allow_nan=False))
