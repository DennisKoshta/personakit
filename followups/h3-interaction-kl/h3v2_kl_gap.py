"""H3-v2b: capability, layer and column-subspace decomposition of steering KL.

Run the CPU check with python and --smoke; paid launches belong to
 the project owner. Explicit task settings override the older DESIGN defaults.
CI is judge concept adoption and CM is lexical mention, matching H3/H5 code.
"""
import hashlib
import itertools
import json
import pathlib
import statistics
import sys
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "study"))
import hf_cache
import jobs
from jobs import MODELS, OUT, out_dir

app = modal.App("personakit-h3v2b")
image = jobs.image.add_local_python_source("h3v2_kl_gap")
PILE = ("NeelNanda/pile-10k", "127bfedcd5047750df5ccf3a12979a47bfa0bafa")
FULL = dict(dev_q=10, test_q=20, everyday=20, concepts=5, max_new=96,
            batch=20, judge_batch=4, judge="9b", mmlu=300, kl_docs=200,
            kl_len=128, kl_batch=4, boot=2000, doses=[0.15, 0.20, 0.25, 0.30, 0.35, 0.50],
            ks=[1, 8, 32], layers=[16, 22])
SMOKE = dict(FULL, dev_q=2, test_q=2, everyday=2, concepts=2, max_new=8,
             batch=4, judge_batch=4, judge="0.8b", mmlu=8, kl_docs=4,
             kl_len=32, kl_batch=2, boot=100, doses=[0.25, 0.50], ks=[1, 8], layers=[12, 16])
RUBRICS = ["concept", "identity", "coherent"]


def call(fn, *, model, smoke):
    """Adapt the named entrypoint to the shared positional-only retry helper."""
    return jobs.call(fn, model, smoke)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    """Atomic, strict JSON: never persist NaN/Inf or a truncated checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def commit_volume():
    if not modal.is_local():
        OUT.commit()


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path):
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}


def save_jsonl(path, records):
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records.values()))
    tmp.replace(path)


def bootstrap(values, draws):
    """Paired percentile interval: callers subtract aligned units before resampling."""
    import numpy as np
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Bootstrap requires finite, nonempty unit observations")
    idx = np.random.default_rng(0).integers(len(x), size=(draws, len(x)))
    return dict(mean=float(x.mean()), ci95=np.quantile(x[idx].mean(1), [0.025, 0.975]).tolist(), n=len(x))


def differences(a, b):
    if len(a) != len(b):
        raise ValueError("Paired observations differ in length")
    return [x - y for x, y in zip(a, b)]


def select_dose(cells):
    """High dose is diagnostic only; the explicit task's fallback is 0.25."""
    coherent = [(d, r) for d, r in sorted(cells.items()) if d <= 0.35 and r["coherent"] >= 0.8]
    hits = [(d, r) for d, r in coherent if r["ci"] >= 0.5]
    if hits:
        return dict(d_star=hits[0][0], reason="smallest_coherent_hit", reached=True)
    if coherent:
        d, _ = min(coherent, key=lambda dr: (abs(dr[1]["ci"] - 0.55), dr[0]))
        return dict(d_star=d, reason="nearest_coherent_ci", reached=False)
    return dict(d_star=0.25, reason="no_coherent_cell_fallback", reached=False)


def make_recipe(model, ax, concept, layer, dc, da, kind):
    import torch
    from persona import blocks
    a_hat = ax["unit"][layer].float().cpu()
    raw = ax["concept_dirs"][concept][layer].float().cpu()
    orth = raw - (raw @ a_hat) * a_hat
    if abs(float(a_hat.norm()) - 1) > 1e-4 or float(orth.norm()) < 1e-8:
        raise ValueError("Invalid axis/concept normalization")
    c_hat = orth / (orth.norm() + 1e-8)
    dense = dc * c_hat - da * a_hat
    norm = float(dense.norm())
    target = dense / (norm + 1e-12)
    meta = dict(concept=concept, layer=layer, dc=dc, da=da, repr_kind=kind,
                v_norm=norm, cos_with_dense=1.0, top_idx=[], signs=[], top_cosines=[])
    if kind == "dense":
        delta = float(ax["median_resid"][layer]) * dense
    else:
        k = int(kind.split("_")[1])
        W = blocks(model)[layer].mlp.down_proj.weight.detach().float()
        W_hat = W / (W.norm(dim=0, keepdim=True) + 1e-8)
        dots = W_hat.T @ target.to(W_hat.device)
        top_idx = torch.argsort(dots.abs(), descending=True, stable=True)[:k]
        U = W_hat[:, top_idx].double()
        Q, R = torch.linalg.qr(U, mode="reduced")
        if float(R.diag().abs().min()) < 1e-10:
            raise ValueError("Selected columns are rank deficient; QR would invent a subspace")
        p = (Q @ (Q.T @ target.to(Q.device).double())).float()
        cos_k = float(p.norm())
        if cos_k < 1e-8:
            raise ValueError("Degenerate sparse projection")
        delta = float(ax["median_resid"][layer]) * norm * (p / (p.norm() + 1e-12)).cpu()
        meta.update(cos_with_dense=cos_k, top_idx=top_idx.tolist(),
                    signs=[1 if x >= 0 else -1 for x in dots[top_idx].tolist()],
                    top_cosines=dots[top_idx].tolist())
    meta["delta_norm"] = float(delta.norm())
    expected = float(ax["median_resid"][layer]) * norm
    if abs(meta["delta_norm"] - expected) > 1e-5 * max(1, expected):
        raise ValueError("Sparse/dense residual norm mismatch")
    return (layer, delta.detach()), meta


def activate(hooks, adds):
    hooks.clear()
    for layer, vec in adds:
        hooks.add(layer, vec)


def teacher_margin(model, tok, questions, concept, batch):
    """Append continuation tokens to the actual chat prefix; score only those tokens."""
    import torch
    import persona
    import prompts as P
    continuations = [f"I am {P.PERSONA_NAMES[concept]}.", "I am Qwen, an AI assistant."]
    tail_ids = [tok.encode(s, add_special_tokens=False) for s in continuations]
    todo = [(persona.chat_ids(tok, q), t) for q in questions for t in tail_ids]
    values = []
    with torch.inference_mode():
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            ids, mask = persona.pad_left([p + t for p, t in chunk], tok.pad_token_id, model.device)
            keep = max(len(t) for _, t in chunk) + 1
            logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits.float()
            for j, (_, tail) in enumerate(chunk):
                lp = logits[j, -len(tail)-1:-1].log_softmax(-1)
                val = lp.gather(1, torch.tensor(tail, device=lp.device)[:, None]).squeeze(1)
                values.append((float(val.mean()), float(val.sum())))
    return [dict(mean=values[i][0] - values[i+1][0], sum=values[i][1] - values[i+1][1])
            for i in range(0, len(values), 2)]


def roll(model, tok, hooks, adds, recipe_id, concept, split, questions, sizes, texts, odir):
    import persona
    wanted = [f"{recipe_id}|{split}|{i}" for i in range(len(questions))]
    missing = [i for i, key in enumerate(wanted) if key not in texts]
    if not missing:
        return wanted
    activate(hooks, adds)
    for start in range(0, len(missing), sizes["batch"]):
        indices = missing[start:start + sizes["batch"]]
        qs = [questions[i] for i in indices]
        outs = persona.generate(model, tok, [persona.chat_ids(tok, q) for q in qs], sizes["max_new"])
        margins = teacher_margin(model, tok, qs, concept, sizes["batch"]) if split != "everyday" else [{} for _ in qs]
        for i, out, margin in zip(indices, outs, margins):
            key = wanted[i]
            texts[key] = dict(id=key, recipe=recipe_id, concept_name=concept,
                              concept="an alien" if concept == "aliens" and split != "dev" else concept,
                              split=split, question=questions[i], reply=tok.decode(out, skip_special_tokens=True),
                              teacher_logodds=margin)
        save_jsonl(odir / "texts.jsonl", texts)
    hooks.clear()
    return wanted


def judge_missing(sizes, device, texts, labels, odir):
    import persona
    from judge import score
    missing = [t for key, t in texts.items() if key not in labels]
    if not missing:
        return
    judge, tok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]), device)
    try:
        for start in range(0, len(missing), 64):
            chunk = missing[start:start + 64]
            primary = score(judge, tok, chunk, RUBRICS, sizes["judge_batch"])
            alien_idx = [i for i, t in enumerate(chunk) if t["concept_name"] == "aliens" and t["split"] != "dev"]
            secondary = score(judge, tok, [{**chunk[i], "concept": "aliens"} for i in alien_idx],
                              ["concept"], sizes["judge_batch"]) if alien_idx else []
            for i, lab in zip(alien_idx, secondary):
                primary[i]["aliens_wording"] = lab
            for t, lab in zip(chunk, primary):
                labels[t["id"]] = dict(id=t["id"], **lab)
            save_jsonl(odir / "labels.jsonl", labels)
            if (start + len(chunk)) % 256 == 0 or start + len(chunk) == len(missing):
                commit_volume()
    finally:
        del judge
        persona.free()


def rates(ids, texts, labels):
    import persona
    import prompts as P
    if not ids:
        raise ValueError("Empty rollout cell")
    vals = {k: [] for k in ("ci", "ib", "cm", "coherent")}
    secondary = []
    for key in ids:
        t, lab = texts[key], labels[key]
        vals["ci"].append(lab["concept"])
        vals["ib"].append(1 - lab["identity"])
        vals["cm"].append(int(P.mentions(t["reply"], t["concept_name"])))
        vals["coherent"].append(lab["coherent"])
        if "aliens_wording" in lab:
            secondary.append(lab["aliens_wording"]["concept"])
    result = {"n": len(ids)}
    for key, v in vals.items():
        result[key] = sum(v) / len(v)
        result[key + "_ci95"] = persona.wilson(sum(v), len(v))
    if secondary:
        result["ci_aliens_wording"] = sum(secondary) / len(secondary)
        result["ci_aliens_wording_ci95"] = persona.wilson(sum(secondary), len(secondary))
    margins = [texts[key]["teacher_logodds"]["mean"] for key in ids if texts[key]["teacher_logodds"]]
    if margins:
        result["teacher_logodds_mean"] = statistics.mean(margins)
    return result


def factorial_analysis(factorial, raw, layers, kinds, draws):
    """Saturated treatment-coded regression and orthogonal balanced ANOVA.

    Coefficients are fit per passage, then passages are resampled in lockstep.
    Marginal contrasts average the other factors; interactions are differences
    of differences relative to high/mid/dense, not independent-cell intervals.
    """
    import numpy as np
    order = list(itertools.product(["high", "matched"], layers, kinds))
    lookup = {(r["dose_regime"], r["layer"], r["repr_kind"]): r["recipe"] for r in factorial.values()}
    Y = np.array([raw[lookup[key]]["kl_docs"] for key in order])
    names = ["intercept", "matched", "late"] + kinds[1:]
    names += ["matched:late"] + [f"matched:{k}" for k in kinds[1:]]
    names += [f"late:{k}" for k in kinds[1:]] + [f"matched:late:{k}" for k in kinds[1:]]
    X = []
    for regime, layer, kind in order:
        d, l = int(regime == "matched"), int(layer == layers[1])
        r = [int(kind == k) for k in kinds[1:]]
        X.append([1, d, l, *r, d*l, *[d*x for x in r], *[l*x for x in r], *[d*l*x for x in r]])
    X = np.asarray(X, dtype=float)
    coeff = np.linalg.solve(X, Y)
    nadd = 3 + len(kinds) - 1
    additive = np.linalg.lstsq(X[:, :nadd], Y, rcond=None)[0]
    cube = Y.reshape(2, 2, len(kinds), -1)
    marginal = {"matched_minus_high": cube[1].mean((0, 1)) - cube[0].mean((0, 1)),
                "late_minus_mid": cube[:, 1].mean((0, 1)) - cube[:, 0].mean((0, 1))}
    refs = {"matched_minus_high": cube[0].mean((0, 1)), "late_minus_mid": cube[:, 0].mean((0, 1))}
    for i, k in enumerate(kinds[1:], 1):
        marginal[k + "_minus_dense"] = cube[:, :, i].mean((0, 1)) - cube[:, :, 0].mean((0, 1))
        refs[k + "_minus_dense"] = cube[:, :, 0].mean((0, 1))
    contrasts = {}
    idx = np.random.default_rng(0).integers(Y.shape[1], size=(draws, Y.shape[1]))
    for k, diff in marginal.items():
        ref = refs[k]
        if (ref[idx].mean(1) <= 0).any():
            raise ValueError("Undefined percent KL reduction")
        pct = -100 * diff[idx].mean(1) / ref[idx].mean(1)
        contrasts[k] = {**bootstrap(diff, draws), "percent_reduction": float(-100*diff.mean()/ref.mean()),
                        "percent_reduction_ci95": np.quantile(pct, [0.025, 0.975]).tolist()}
    # Sum-to-zero components on the balanced cell means give an exact variance partition.
    means = cube.mean(-1)
    components = {(): np.full_like(means, means.mean())}
    ss = {}
    factor_names = ["dose", "layer", "repr"]
    for size in (1, 2, 3):
        for axes in itertools.combinations(range(3), size):
            avg = means.mean(axis=tuple(i for i in range(3) if i not in axes), keepdims=True) if size < 3 else means
            component = np.broadcast_to(avg, means.shape).copy()
            for sub, val in components.items():
                if set(sub) < set(axes):
                    component -= val
            components[axes] = component
            ss[":".join(factor_names[i] for i in axes)] = float((component**2).sum())
    total = sum(ss.values())
    return dict(coding="high/mid/dense reference; paired passage resampling", n_passages=Y.shape[1],
                saturated_coefficients={k: bootstrap(v, draws) for k, v in zip(names, coeff)},
                additive_coefficients={k: bootstrap(v, draws) for k, v in zip(names[:nadd], additive)},
                marginal_contrasts=contrasts, cell_mean_sum_squares=ss,
                cell_mean_variance_fractions={k: v / total if total else 0.0 for k, v in ss.items()},
                saturated_max_abs_residual=float(np.abs(X @ coeff - Y).max()))


def kl_passages(model, hooks, docs, setups):
    """H3's exact token mask, retaining each passage's mean for paired inference.

    Exclude padding and the first real token (the attention sink), including
    the final logit exactly as persona.kl_from_base does. Batch only documents;
    all distributions remain full-vocabulary float32 KL(unsteered || steered).
    """
    import torch
    import persona
    with torch.inference_mode():
        ids, mask = persona.pad_left(docs, 0, model.device)
        keep = mask.bool().clone()
        keep[torch.arange(len(ids), device=ids.device), (mask == 0).sum(1)] = False
        counts = keep.sum(1).tolist()
        if min(counts) < 1:
            raise ValueError("KL requires at least two tokens per document")
        hooks.clear()
        base = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[keep].float().log_softmax(-1)
        prob = base.exp()
        result = {}
        try:
            for rid, adds in setups.items():
                activate(hooks, adds)
                if not adds:
                    result[rid] = [0.0] * len(docs)
                else:
                    lp = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[keep].float().log_softmax(-1)
                    per_token = (prob * (base - lp)).sum(-1)
                    result[rid] = [float(v.mean()) for v in per_token.split(counts)]
                    del lp, per_token
        finally:
            hooks.clear()
    return result


def token_weighted_kl(values, counts, draws):
    """Historical corpus estimate; resample whole passages, including lengths."""
    import numpy as np
    x, w = np.asarray(values, dtype=float), np.asarray(counts, dtype=float)
    if x.shape != w.shape or not len(x) or not np.isfinite(x).all() or not (w > 0).all():
        raise ValueError("KL values and positive token counts must align")
    idx = np.random.default_rng(0).integers(len(x), size=(draws, len(x)))
    estimates = (x[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return dict(mean=float((x*w).sum()/w.sum()), ci95=np.quantile(estimates, [0.025, 0.975]).tolist())


def benchmark_summary(kl, mm, base, neuron, draws, token_counts):
    result = {}
    for name, values in [("kl_mean", kl), ("kl_diff_vs_neuron", differences(kl, neuron["kl_docs"])),
                         ("mmlu_rate", mm), ("mmlu_diff_vs_unsteered", differences(mm, base["mmlu_bits"])),
                         ("mmlu_diff_vs_neuron", differences(mm, neuron["mmlu_bits"]))]:
        stat = bootstrap(values, draws)
        result[name], result[name + "_ci95"] = stat["mean"], stat["ci95"]
    for name, values in [("kl_token_weighted", kl),
                         ("kl_token_weighted_diff_vs_neuron", differences(kl, neuron["kl_docs"]))]:
        stat = token_weighted_kl(values, token_counts, draws)
        result[name], result[name + "_ci95"] = stat["mean"], stat["ci95"]
    return result


@app.function(image=image, gpu="A10G", memory=32768, timeout=2100, volumes={**hf_cache.volumes, "/out": OUT})
def run(model: str = "4b", smoke: bool = False):
    import torch
    import persona
    import prompts as P
    from judge import summary

    started = time.monotonic()
    model_key = "0.8b" if smoke else model
    if model_key != ("0.8b" if smoke else "4b"):
        raise ValueError("Part 2 is preregistered for 4b (0.8b only with --smoke)")
    sizes = dict(SMOKE if smoke else FULL)
    if smoke:
        torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not modal.is_local():
        OUT.reload()
    suffix = model_key + ("-smoke" if smoke else "")
    tag, axis_path = "h3v2b-" + suffix, out_dir("s1b-" + suffix) / "axis.pt"
    odir = out_dir(tag)
    ax = torch.load(axis_path, map_location="cpu", weights_only=False)
    layers = sizes["layers"]
    kinds = ["dense"] + [f"sparse_{k}" for k in sizes["ks"]]
    concepts = P.CONCEPTS_TEST[:sizes["concepts"]]
    manifest = dict(schema=2, sizes=sizes, model_pin=MODELS[model_key], judge_pin=MODELS[sizes["judge"]],
                    axis_sha256=hashlib.sha256(axis_path.read_bytes()).hexdigest(), pile_pin=PILE,
                    mmlu_seed=42, bootstrap_seed=0, concepts=concepts,
                    prompts=dict(dev=P.IDENTITY_DEV[:sizes["dev_q"]], identity=P.IDENTITY_TEST[:sizes["test_q"]],
                                 everyday=P.EVERYDAY_TEST[:sizes["everyday"]]),
                    ci_definition="judge concept adoption (H3/H5)", cm_definition="lexical P.mentions",
                    kl_positions="H3 logits 1..T-1; exclude padding and first real token (attention sink)",
                    kl_estimand="kl_mean: equal passage weights; kl_token_weighted: historical corpus weighting")
    # Include source and imported harness pins so changed code cannot silently reuse old measurements.
    manifest["source_sha256"] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    manifest["dependencies"] = {name: hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest()
                                for name, module in [("persona", persona), ("prompts", P),
                                                     ("judge", sys.modules["judge"]), ("jobs", jobs)]}
    manifest = json.loads(json.dumps(manifest))
    old = read_json(odir / "manifest.json", manifest)
    if old != manifest:
        raise ValueError(f"Checkpoint manifest changed; use a clean output directory: {odir}")
    save(odir / "manifest.json", manifest)
    texts, labels = read_jsonl(odir / "texts.jsonl"), read_jsonl(odir / "labels.jsonl")
    dev = read_json(odir / "dev_calib.json", {})
    if not dev:
        target, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
        hooks = persona.Hooks(target).__enter__()
        cells = {}
        try:
            specs = [(L, kind, d, d) for L in layers for kind in kinds for d in sizes["doses"]]
            specs += [(layers[1], "dense", d, 0.0) for d in (0.35, 0.50)]
            for L, kind, dc, da in specs:
                rid = f"dev|{L}|{kind}|{dc}|{da}"
                add, meta = make_recipe(target, ax, "aliens", L, dc, da, kind)
                ids = roll(target, tok, hooks, [add], rid, "aliens", "dev", manifest["prompts"]["dev"], sizes, texts, odir)
                cells[rid] = dict(meta=meta, text_ids=ids)
                print(f"[h3v2b] dev generated {rid}", flush=True)
        finally:
            hooks.__exit__()
            del hooks, target
            persona.free()
        commit_volume()
        judge_missing(sizes, device, texts, labels, odir)
        for cell in cells.values():
            cell["rates"] = rates(cell["text_ids"], texts, labels)
        selected = {}
        for L in layers:
            for kind in kinds:
                candidates = {r["meta"]["dc"]: r["rates"] for r in cells.values()
                              if r["meta"]["layer"] == L and r["meta"]["repr_kind"] == kind
                              and r["meta"]["dc"] == r["meta"]["da"]}
                selected[f"{L}|{kind}"] = select_dose(candidates)
        dev = dict(cells=cells, selected=selected,
                   d_star_med=statistics.median(selected[f"{L}|dense"]["d_star"] for L in layers))
        save(odir / "dev_calib.json", dev)
        commit_volume()
    print(f"[h3v2b] calibration complete: {dev['selected']}", flush=True)

    raw = read_json(odir / "stage2b_raw.json", {})
    target, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
    hooks = persona.Hooks(target).__enter__()
    setups, metadata, aliases, factorial, generalization = {}, {}, {}, {}, {}

    def register(name, concept="aliens", layer=0, dc=0.0, da=0.0, kind="dense", baseline=False):
        signature = [concept, layer, dc, da, kind]
        rid = digest(signature)[:20]
        if rid not in setups:
            if kind == "unsteered":
                adds, meta = [], dict(concept=concept, repr_kind=kind)
            elif kind == "neuron":
                w = persona.down_col(target, layer, nr[1]).detach().cpu()
                delta = nr[2] * dc * float(ax["median_resid"][layer]) * w / w.norm()
                adds, meta = [(layer, delta)], dict(concept=concept, layer=layer, neuron=nr[1], sign=nr[2],
                                                   dose=dc, repr_kind=kind, delta_norm=float(delta.norm()))
            else:
                add, meta = make_recipe(target, ax, concept, layer, dc, da, kind)
                adds = [add]
            setups[rid], metadata[rid] = adds, meta
        aliases[name] = dict(recipe=rid, baseline=baseline, concept=concept)
        return rid

    nr = P.SMOKE_KNOWN["smoke_neuron"] if smoke else P.KNOWN["alien"]
    register("unsteered", kind="unsteered", baseline=True)
    neuron_names = [f"neuron_L{nr[0]}_N{nr[1]}_d{dose}" for dose in (0.5, 0.75)]
    for name, dose in zip(neuron_names, (0.5, 0.75)):
        register(name, layer=nr[0], dc=dose, kind="neuron", baseline=True)
    for regime, L, kind in itertools.product(["matched", "high"], layers, kinds):
        d = dev["selected"][f"{L}|{kind}"]["d_star"] if regime == "matched" else 0.5
        name = f"aliens_{regime}_L{L}_{kind}"
        rid = register(name, layer=L, dc=d, da=d, kind=kind)
        factorial[name] = dict(recipe=rid, dose_regime=regime, layer=L, repr_kind=kind, dc=d, da=d)
    for L, kind in itertools.product(layers, ["dense", "sparse_8"]):
        register(f"aliens_fixed_L{L}_{kind}_dc0.25_da0.25", layer=L, dc=0.25, da=0.25, kind=kind)
    for d in (0.35, 0.50):
        register(f"aliens_fixed_L{layers[1]}_dense_dc{d}_da0", layer=layers[1], dc=d)
    med = dev["d_star_med"]
    for L, kind, regime, d in [(layers[0], "dense", "high", 0.5), (layers[0], "dense", "matched", med),
                                (layers[1], "dense", "matched", med), (layers[1], "sparse_8", "matched", med)]:
        group = f"test5_L{L}_{kind}_{regime}"
        generalization[group] = [register(f"{group}|{c}", concept=c, layer=L, dc=d, da=d, kind=kind) for c in concepts]
    try:
        pile = hf_cache.dataset(*PILE, split="train")
        encoded = [tok.encode(pile[i]["text"], add_special_tokens=False)[:sizes["kl_len"]]
                   for i in range(9000, 9000 + sizes["kl_docs"])]
        doc_ids = [9000+i for i, x in enumerate(encoded) if len(x) >= 16]
        docs = [x for x in encoded if len(x) >= 16]
        token_counts = [len(x) - 1 for x in docs]
        if not docs:
            raise ValueError("No valid Pile documents")
        rows = persona.mmlu_rows(n=sizes["mmlu"], seed=42)
        data_ids = dict(pile_indices=doc_ids, pile_token_hash=digest(docs), mmlu_rows_hash=digest(rows),
                        pile_kl_token_counts=token_counts)
        if read_json(odir / "data_ids.json", data_ids) != data_ids:
            raise ValueError("Dataset unit alignment changed on resume")
        save(odir / "data_ids.json", data_ids)
        for number, (rid, adds) in enumerate(setups.items(), 1):
            rec = raw.setdefault(rid, dict(meta=metadata[rid], kl_docs=[]))
            c = metadata[rid]["concept"]
            rec["identity_ids"] = roll(target, tok, hooks, adds, rid, c, "identity", manifest["prompts"]["identity"], sizes, texts, odir)
            if c == "aliens":
                rec["everyday_ids"] = roll(target, tok, hooks, adds, rid, c, "everyday", manifest["prompts"]["everyday"], sizes, texts, odir)
            if "mmlu_bits" not in rec:
                activate(hooks, adds)
                rec["mmlu_bits"] = persona.mmlu(target, tok, rows, batch=32)
                hooks.clear()
            save(odir / "stage2b_raw.json", raw)
            if number % 5 == 0 or number == len(setups):
                commit_volume()
            print(f"[h3v2b] rollouts/MMLU {len(raw)}/{len(setups)}", flush=True)
        # Bound vocabulary memory to one document batch and checkpoint at batch boundaries.
        kl_started = time.monotonic()
        for i in range(0, len(docs), sizes["kl_batch"]):
            end = min(i + sizes["kl_batch"], len(docs))
            todo = {rid: adds for rid, adds in setups.items() if len(raw[rid]["kl_docs"]) < end}
            if not todo:
                continue
            if any(len(raw[rid]["kl_docs"]) != i for rid in todo):
                raise ValueError("Noncontiguous KL checkpoint")
            measured = kl_passages(target, hooks, docs[i:end], todo)
            for rid, values in measured.items():
                raw[rid]["kl_docs"].extend(values)
            save(odir / "stage2b_raw.json", raw)
            if end % (5 * sizes["kl_batch"]) == 0 or end == len(docs):
                commit_volume()
                print(f"[h3v2b] KL passages {end}/{len(docs)} in {time.monotonic()-kl_started:.1f}s", flush=True)
    finally:
        hooks.clear()
        hooks.__exit__()
        del hooks, target
        persona.free()
    judge_missing(sizes, device, texts, labels, odir)
    base, neuron = raw[aliases["unsteered"]["recipe"]], raw[aliases[neuron_names[0]]["recipe"]]
    table = {}
    for name, alias in aliases.items():
        rec = raw[alias["recipe"]]
        row = dict(**alias, meta=rec["meta"], identity=rates(rec["identity_ids"], texts, labels),
                   **benchmark_summary(rec["kl_docs"], rec["mmlu_bits"], base, neuron, sizes["boot"], token_counts))
        if "everyday_ids" in rec:
            row["everyday"] = rates(rec["everyday_ids"], texts, labels)
        table[name] = row
    heldout = {}
    for name, rids in generalization.items():
        kl = [statistics.mean(raw[rid]["kl_docs"][i] for rid in rids) for i in range(len(docs))]
        mm = [statistics.mean(raw[rid]["mmlu_bits"][i] for rid in rids) for i in range(len(rows))]
        ids = [key for rid in rids for key in raw[rid]["identity_ids"]]
        heldout[name] = dict(concepts=concepts, recipe_ids=rids, identity=rates(ids, texts, labels),
                             kl_docs_concept_mean=kl, mmlu_items_concept_mean=mm,
                             **benchmark_summary(kl, mm, base, neuron, sizes["boot"], token_counts))
    checks = {}
    for name in factorial:
        row = table[name]
        checks[name] = dict(ci=row["identity"]["ci"] >= 0.55, coherent=row["identity"]["coherent"] >= 0.80,
                            everyday_coherent=row["everyday"]["coherent"] >= 0.90,
                            kl_mean=row["kl_mean"] <= 0.35, mmlu_rate=row["mmlu_rate"] >= 0.700)
    passing = [name for name, gates in checks.items() if all(gates.values())]
    capable = [table[name] for name in factorial if table[name]["identity"]["ci"] >= 0.55]
    verdict = "yes" if passing else "no" if capable and all(r["kl_mean"] > 0.60 or r["mmlu_rate"] < 0.680 for r in capable) else "inconclusive"
    result = dict(model=model_key, smoke=smoke, tag=tag, sizes=sizes, manifest=manifest,
                  data_ids=data_ids, dev_calib=dev, table=table, factorial=factorial, generalization=heldout,
                  factorial_kl=factorial_analysis(factorial, raw, layers, kinds, sizes["boot"]),
                  decision=dict(verdict=verdict, checks=checks, passing=passing,
                                kl_estimand="kl_mean: equal-passage mean over H3 positions; thresholds 0.35/0.60",
                                inference_valid=not smoke, scope="aliens factorial only; diagnostics excluded",
                                no_capable_recipe=not bool(capable)),
                  judge=summary(list(labels.values()), RUBRICS), unique_test_recipes=len(setups),
                  minutes=(time.monotonic()-started)/60)
    save(odir / "h3v2b.json", result)
    commit_volume()
    return dict(model=model_key, smoke=smoke, verdict=verdict, unique_test_recipes=len(setups),
                factorial_cells=len(factorial), kl_docs=len(docs), mmlu_items=len(rows),
                minutes=result["minutes"], output=str(odir / "h3v2b.json"))


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False):
    result = call(run, model=model, smoke=smoke)
    print("PERSONAKIT_RESULT", json.dumps(result, allow_nan=False), flush=True)
