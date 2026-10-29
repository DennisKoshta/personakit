"""personakit's work, wherever it runs: build a model's atlas, explain a neuron, make (or port) a persona recipe and
find a neuron's twins in another model. The command line calls these on this machine (--local) or through
personakit.remote on Modal. Weights go in the Hugging Face cache, atlases in $PERSONAKIT_ATLAS (default
~/.cache/personakit/atlas).

Doses are in the study's unit: a dose of d adds d * m_L * (unit direction) to the residual leaving layer L, where m_L is
the median norm of that residual on the Pile.
"""
import json
import os
import pathlib
import random
import time
import urllib.request

from . import data as D

SIZES = {
    "full": dict(pile_docs=1000, pile_len=256, roles=40, questions=10, default_questions=16, concept_q=10, max_new=128,
                 batch=64, stat_batch=16, judge="9b", judge_batch=16, doses=[0.25, 0.5, 0.75, 1.0, 1.5], probe_q=10,
                 grid_c=[0.25, 0.5, 0.75], grid_a=[0.0, 0.25, 0.5], test_q=20, everyday=10,
                 reply_new=96, mmlu=200, kl_docs=50, kl_len=128, prefilter=200, twins=5, min_persona=0.1, min_coherent=0.8),
    "toy": dict(pile_docs=4, pile_len=32, roles=2, questions=2, default_questions=2, concept_q=2, max_new=8, batch=8,
                stat_batch=4, judge="0.8b", judge_batch=4, doses=[1.0], probe_q=2, grid_c=[1.0], grid_a=[0.0, 0.5],
                test_q=2, everyday=1, reply_new=8, mmlu=4, kl_docs=2, kl_len=16, prefilter=8, twins=2,
                min_persona=0.0, min_coherent=0.0),  # a toy run picks a cell whatever its replies
}


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        try:
            torch.ones(1, dtype=torch.bfloat16, device="mps")  # bf16 on Apple GPUs needs macOS 14
            return "mps"
        except (RuntimeError, TypeError):
            pass
    return "cpu"


def _load(key: str, judge: bool = False):
    """A model and its tokenizer. The judge may spill into CPU memory when the GPU can't hold it (it only scores, a
    forward pass per batch); the model under study must fit, since its weights are read and its replies generated."""
    from huggingface_hub import snapshot_download
    from . import engine as E
    path = snapshot_download(D.MODELS[key][0], revision=D.MODELS[key][1])
    return E.load(path, _device(), offload=judge)


def _fetch(path: str):
    with urllib.request.urlopen(f"{D.AXIS_DATA}/{path}", timeout=60) as r:
        body = r.read().decode()
    return [json.loads(x) for x in body.splitlines() if x.strip()] if path.endswith(".jsonl") else json.loads(body)


def _atlas_path(key: str) -> pathlib.Path:
    root = pathlib.Path(os.environ.get("PERSONAKIT_ATLAS") or pathlib.Path.home() / ".cache" / "personakit" / "atlas")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{key}.pt"


ATLAS_KEYS = ("pile_docs", "pile_len", "roles", "questions", "default_questions", "concept_q", "max_new")  # what _build reads


def _atlas(key: str, sz: dict) -> dict:
    import torch
    path = _atlas_path(key)
    if path.exists():
        atlas = torch.load(path)
        if atlas["revision"] == D.MODELS[key][1] and all(atlas["sizes"].get(k) == sz[k] for k in ATLAS_KEYS):
            return atlas
        print(f"[personakit] the atlas for {key} was built with other settings: rebuilding it", flush=True)
    else:
        print(f"[personakit] no atlas for {key} yet: building it first", flush=True)
    _build(key, sz)
    return torch.load(path)


def _build(key: str, sz: dict) -> dict:
    """The model's atlas: each layer's dose unit (median residual norm on the Pile), each neuron's natural extremes
    (Pile and chat), the Assistant Axis (mean default reply minus the mean role-playing reply, as Lu et al. 2026 but
    without their judge's filter), the role directions and the reference concepts' mean residuals."""
    import datasets
    import torch
    from . import engine as E

    t0, rng = time.time(), random.Random(0)
    model, tok = _load(key)
    n_layers, d = model.config.num_hidden_layers, model.config.hidden_size
    hooks = E.Hooks(model).__enter__()
    pile = datasets.load_dataset(D.PILE[0], revision=D.PILE[1], split="train")
    docs = [tok.encode(pile[i]["text"], add_special_tokens=False)[:sz["pile_len"]] for i in range(sz["pile_docs"])]
    docs = [x for x in docs if len(x) >= 16]
    pile_stats = E.Stats()
    for s in range(0, len(docs), sz["stat_batch"]):
        E.collect_stats(model, hooks, pile_stats, docs[s:s + sz["stat_batch"]])
    pile_r = pile_stats.result()

    names = list(_fetch("roles/role_list.json"))
    pool = [r for r in names if r not in D.FORCED_ROLES + ["assistant"]]
    roles = (D.FORCED_ROLES + rng.sample(pool, len(pool)))[:sz["roles"]]
    questions = [q["question"] for q in _fetch("extraction_questions.jsonl")]
    qs = [questions[i] for i in rng.sample(range(len(questions)), max(sz["questions"], sz["default_questions"]))]
    jobs = []  # (kind, key, prompt ids, sampled)
    for role in roles:
        system = _fetch(f"roles/instructions/{role}.json")["instruction"][0]["pos"]
        jobs += [("role", role, E.chat_ids(tok, q, system), True) for q in qs[:sz["questions"]]]
    for system in D.default_systems(key):
        jobs += [("default", "", E.chat_ids(tok, q, system), True) for q in qs[:sz["default_questions"]]]
    for c in D.REFERENCE_CONCEPTS:
        jobs += [("concept", c, E.chat_ids(tok, q.format(x=c)), False) for q in D.CONCEPT_QUESTIONS[:sz["concept_q"]]]
    print(f"[personakit] {key}: {len(jobs)} replies to generate", flush=True)
    replies = [[] for _ in jobs]
    order = sorted(range(len(jobs)), key=lambda j: (jobs[j][3], len(jobs[j][2])))
    for b, s in enumerate(range(0, len(order), sz["batch"])):
        for sample in (False, True):
            js = [j for j in order[s:s + sz["batch"]] if jobs[j][3] == sample]
            if js:
                for j, o in zip(js, E.generate(model, tok, [jobs[j][2] for j in js], sz["max_new"], sample, seed=b)):
                    replies[j] = o

    means, chat_stats = torch.zeros(len(jobs), n_layers, d), E.Stats()
    got = [j for j in range(len(jobs)) if replies[j]]
    order = sorted(got, key=lambda j: len(jobs[j][2]) + len(replies[j]))
    for s in range(0, len(order), sz["stat_batch"]):
        js = order[s:s + sz["stat_batch"]]
        seqs, starts = [jobs[j][2] + replies[j] for j in js], [len(jobs[j][2]) for j in js]
        means[js] = E.span_means(model, hooks, seqs, starts, list(range(n_layers)))
        E.collect_stats(model, hooks, chat_stats, seqs, starts)
    chat_r = chat_stats.result()
    hooks.__exit__()

    def mean_of(kind, k=None):
        return means[[j for j in got if jobs[j][0] == kind and (k is None or jobs[j][1] == k)]].mean(0)

    role_vecs = torch.stack([mean_of("role", r) for r in roles])
    default = mean_of("default")
    axis = default - role_vecs.mean(0)
    unit = axis / axis.norm(dim=-1, keepdim=True)
    mid = n_layers // 2
    X = role_vecs[:, mid] - role_vecs[:, mid].mean(0)
    cos_pc1 = float((torch.linalg.svd(X, full_matrices=False).Vh[0] @ unit[mid]).abs())
    atlas = {"model": key, "revision": D.MODELS[key][1], "layers": n_layers, "d": d,
             "m": pile_r["norm_q10_50_90"][:, 1], "pile_hi": pile_r["hi"], "pile_lo": pile_r["lo"],
             "chat_hi": chat_r["hi"], "chat_lo": chat_r["lo"], "unit": unit, "axis_norm": axis.norm(dim=-1),
             "default": default, "roles": roles, "role_dirs": (role_vecs - default).half(),
             "concepts": list(D.REFERENCE_CONCEPTS),
             "concept_means": torch.stack([mean_of("concept", c) for c in D.REFERENCE_CONCEPTS]),
             "cos_pc1_mid": cos_pc1, "sizes": sz}
    torch.save(atlas, _atlas_path(key))
    return {"model": key, "replies": len(got), "roles": len(roles), "cos_axis_pc1_mid": round(cos_pc1, 3),
            "minutes": round((time.time() - t0) / 60, 1)}


def _ref_dirs(atlas: dict, layer: int):
    """[k, d] unit concept directions of the reference concepts at `layer` (each minus the others, axis removed)."""
    import torch
    cm, u = atlas["concept_means"][:, layer].float(), atlas["unit"][layer]
    out = []
    for i in range(len(cm)):
        v = cm[i] - torch.cat([cm[:i], cm[i + 1:]]).mean(0)
        v = v - (v @ u) * u
        out.append(v / v.norm())
    return torch.stack(out)


def _roll(model, tok, layer: int, rows: list[dict], vecs: list, sz: dict) -> list[dict]:
    """Greedy replies to rows (dicts with question), each with its own [d] vector added at `layer` (None: unsteered)."""
    import torch
    from . import engine as E
    with E.Hooks(model) as hooks:
        for s in range(0, len(rows), 100):
            chunk, vs = rows[s:s + 100], vecs[s:s + 100]
            vs = torch.stack([v if v is not None else torch.zeros(model.config.hidden_size) for v in vs])
            outs = E.generate(model, tok, [E.chat_ids(tok, r["question"]) for r in chunk], sz["reply_new"],
                              steer=(hooks, layer, vs))
            for r, o in zip(chunk, outs):
                r["reply"] = tok.decode(o, skip_special_tokens=True)
    return rows


def _judge(texts: list[dict], rubrics: list[str], sz: dict) -> None:
    """Label texts in place (identity, concept, coherent), with the judge loaded alone."""
    from . import engine as E
    from .judge import score
    judge, jtok = _load(sz["judge"], judge=True)
    for t, lab in zip(texts, score(judge, jtok, texts, rubrics, sz["judge_batch"])):
        t.update(lab)
    del judge
    E.free()


def _rates(rows: list[dict], keys=("identity", "concept", "coherent")) -> dict:
    from . import engine as E
    out = {"n": len(rows)}
    for k in keys:
        if rows and k in rows[0]:
            hits = sum(r[k] for r in rows) if k != "identity" else sum(1 - r[k] for r in rows)
            name = {"identity": "identity_break", "concept": "persona", "coherent": "coherent"}[k]
            out[name], out[f"{name}_ci95"] = round(hits / len(rows), 3), E.wilson(hits, len(rows))
    return out


def build(key: str, size: str = "full") -> dict:
    return _build(key, SIZES[size])


def explain(key: str, layer: int, neuron: int, sign: int, measure: bool = False, concept: str = "",
            size: str = "full") -> dict:
    """What a neuron's down column points at (weights only), and with `measure`, what pushing it does."""
    import torch
    from . import engine as E
    sz = SIZES[size]
    atlas = _atlas(key, sz)
    model, tok = _load(key)
    W = E.blocks(model)[layer].mlp.down_proj.weight.float()
    norms = W.norm(dim=0)
    Wn = W / norms
    w = sign * Wn[:, neuron]
    z = E.lens(model, w[None])[0]
    pct = lambda vals, x: round(float((vals < x).float().mean()), 4)  # noqa: E731 (among the layer's neurons)
    away = -sign * (atlas["unit"][layer].to(W.device) @ Wn)
    dirs = _ref_dirs(atlas, layer).to(W.device)
    cc = sign * (dirs @ Wn)
    rd = atlas["role_dirs"][:, layer].float().to(W.device)
    rcos = (rd / rd.norm(dim=-1, keepdim=True)) @ w
    hi = torch.maximum(atlas["pile_hi"][layer, neuron], atlas["chat_hi"][layer, neuron])
    lo = torch.minimum(atlas["pile_lo"][layer, neuron], atlas["chat_lo"][layer, neuron])
    extreme = float(hi if sign > 0 else -lo)  # its largest natural activation in the pushed direction
    per_dose = float(atlas["m"][layer]) / float(norms[neuron])  # activation change of a dose-1 push
    out = {"model": key, "layer": layer, "neuron": neuron, "sign": sign,
           "lens_top": [tok.decode([i]) for i in z.topk(12).indices.tolist()],
           "lens_bottom": [tok.decode([i]) for i in (-z).topk(12).indices.tolist()],
           "lens_peakiness": round(float(((z - z.mean()) / z.std()).max()), 2),
           "axis": {"cos_away": round(float(away[neuron]), 4), "pct": pct(away, away[neuron])},
           "concepts": sorted(({"concept": c, "cos": round(float(cc[i, neuron]), 4), "pct": pct(cc[i], cc[i, neuron])}
                               for i, c in enumerate(atlas["concepts"])), key=lambda x: -x["pct"]),
           "roles": [{"role": atlas["roles"][i], "cos": round(float(rcos[i]), 4)} for i in rcos.topk(min(5, len(rcos))).indices.tolist()],
           "natural": {"extreme": round(extreme, 3), "dose_of_extreme": round(extreme / per_dose, 3),
                       "pile_hi": round(float(atlas["pile_hi"][layer, neuron]), 3),
                       "pile_lo": round(float(atlas["pile_lo"][layer, neuron]), 3),
                       "chat_hi": round(float(atlas["chat_hi"][layer, neuron]), 3),
                       "chat_lo": round(float(atlas["chat_lo"][layer, neuron]), 3)}}
    del W, Wn, dirs, cc
    if measure:
        doses = [0.0] + sz["doses"]
        rows = [{"dose": d, "question": q, "concept": concept} for d in doses for q in D.IDENTITY_DEV[:sz["probe_q"]]]
        vec = float(atlas["m"][layer]) * w.cpu()
        _roll(model, tok, layer, rows, [r["dose"] * vec if r["dose"] else None for r in rows], sz)
        del model
        E.free()
        judged = ["identity", "coherent"] + (["concept"] if concept else [])  # unjudged, "concept" holds the text
        _judge(rows, judged, sz)
        out["measured"] = [{"dose": d, **_rates([r for r in rows if r["dose"] == d], judged),
                            "sample": next(r["reply"] for r in rows if r["dose"] == d)[:300]} for d in doses]
    return out


def _concept_dir(model, tok, atlas: dict, about: str, layer: int, sz: dict):
    """The concept's unit direction at `layer`: the mean residual over the model's neutral answers about it, minus
    the reference concepts' (bar itself), with the Assistant Axis component removed."""
    from . import engine as E
    ids = [E.chat_ids(tok, q.format(x=about)) for q in D.CONCEPT_QUESTIONS[:sz["concept_q"]]]
    outs = E.generate(model, tok, ids, sz["max_new"])
    with E.Hooks(model) as hooks:
        mean = E.span_means(model, hooks, [a + b for a, b in zip(ids, outs)], [len(a) for a in ids], [layer])[:, 0].mean(0)
    keep = [i for i, c in enumerate(atlas["concepts"]) if c != about]
    v = mean - atlas["concept_means"][keep, layer].float().mean(0)
    u = atlas["unit"][layer]
    v = v - (v @ u) * u
    return v / v.norm()


def _make(key: str, name: str, about: str, sz: dict, cells: list | None = None) -> dict:
    """A persona recipe: m_mid * (dc * concept - da * axis) at the middle layer, its (dc, da) the grid cell whose
    replies to 10 identity prompts most often speak as `name` while its replies to those and to 10 everyday requests
    stay coherent (>= 0.8 each); then measured on held-out identity and everyday prompts, MMLU and the Pile. (The
    study's grid on the 4B: every coherent cell had both doses <= 0.5, and its best cell for identity prompts was
    coherent on only 32% of everyday ones.)"""
    import datasets
    from . import engine as E
    t0 = time.time()
    atlas = _atlas(key, sz)
    mid = atlas["layers"] // 2
    m, u = float(atlas["m"][mid]), atlas["unit"][mid]
    model, tok = _load(key)
    c = _concept_dir(model, tok, atlas, about, mid, sz)
    vec = lambda dc, da: m * (dc * c - da * u)  # noqa: E731
    cells = cells or [(dc, da) for dc in sz["grid_c"] for da in sz["grid_a"]]
    dev = [{"dc": dc, "da": da, "set": s, "question": q, "concept": name} for dc, da in cells
           for s, qs in (("identity", D.IDENTITY_DEV[:sz["probe_q"]]), ("everyday", D.EVERYDAY[:sz["probe_q"]]))
           for q in qs]
    _roll(model, tok, mid, dev, [vec(r["dc"], r["da"]) for r in dev], sz)
    del model
    E.free()
    _judge(dev, ["concept", "coherent"], sz)
    def cell(dc, da, s):
        return [r for r in dev if (r["dc"], r["da"], r["set"]) == (dc, da, s)]

    grid = [{"dc": dc, "da": da, **_rates(cell(dc, da, "identity"), ("concept", "coherent")),
             "everyday_coherent": _rates(cell(dc, da, "everyday"), ("coherent",))["coherent"]} for dc, da in cells]
    ok = [g for g in grid if min(g["coherent"], g["everyday_coherent"]) >= sz["min_coherent"]
          and g["persona"] >= sz["min_persona"]]
    if not ok:
        return {"model": key, "name": name, "about": about, "made": False, "grid": grid,
                "minutes": round((time.time() - t0) / 60, 1)}
    best = max(ok, key=lambda g: (g["persona"], -(g["dc"] + g["da"])))
    v = vec(best["dc"], best["da"])

    model, tok = _load(key)
    test = [{"set": s, "steered": st, "question": q, "concept": name}
            for s, qs in (("identity", D.IDENTITY_TEST[:sz["test_q"]]), ("everyday", D.EVERYDAY[10:10 + sz["everyday"]]))
            for q in qs for st in (False, True)]
    _roll(model, tok, mid, test, [v if r["steered"] else None for r in test], sz)
    rows = E.mmlu_rows(sz["mmlu"])
    with E.Hooks(model) as hooks:
        plain = E.mmlu(model, tok, rows)
        hooks.add(mid, v)
        steered = E.mmlu(model, tok, rows)
        hooks.clear()
        pile = datasets.load_dataset(D.PILE[0], revision=D.PILE[1], split="train")
        docs = [tok.encode(pile[9000 + i]["text"], add_special_tokens=False)[:sz["kl_len"]] for i in range(sz["kl_docs"])]
        kl = E.kl_from_base(model, hooks, [x for x in docs if len(x) >= 8], {"recipe": [(mid, v)]})["recipe"]
    del model, hooks  # hooks holds the model too: both go before the judge loads
    E.free()
    _judge(test, ["identity", "concept", "coherent"], sz)
    diffs, rng = [], random.Random(0)
    for _ in range(2000):
        ix = [rng.randrange(len(rows)) for _ in rows]
        diffs.append(sum(steered[i] - plain[i] for i in ix) / len(ix))
    diffs.sort()
    pick = lambda s, st: [r for r in test if r["set"] == s and r["steered"] == st]  # noqa: E731
    return {"personakit": 1, "model": key, "revision": D.MODELS[key][1], "name": name, "about": about, "made": True,
            "layer": mid, "dc": best["dc"], "da": best["da"], "dose_unit": m, "vector": v.tolist(),
            "dev": best, "grid": grid,
            "test": {"steered": _rates(pick("identity", True)), "unsteered": _rates(pick("identity", False)),
                     "everyday_steered": _rates(pick("everyday", True)),
                     "everyday_unsteered": _rates(pick("everyday", False))},
            "cost": {"mmlu_unsteered": round(sum(plain) / len(rows), 3), "mmlu_steered": round(sum(steered) / len(rows), 3),
                     "mmlu_diff_ci95": [round(diffs[50], 3), round(diffs[1949], 3)], "mmlu_items": len(rows),
                     "kl_per_token": round(kl, 4)},
            "samples": [r["reply"][:300] for r in pick("identity", True)[:3]],
            "minutes": round((time.time() - t0) / 60, 1)}


def make(key: str, name: str, about: str, size: str = "full") -> dict:
    return _make(key, name, about, SIZES[size])


def port(recipe: dict, to: str, size: str = "full") -> dict:
    """The recipe re-derived in another model: the concept direction and axis from that model's own atlas, the doses
    searched on a 3x3 grid around the recipe's (doses are in each model's own unit, so they may carry over)."""
    sz = SIZES[size]
    dc, da = recipe["dc"], recipe["da"]
    cells = sorted({(round(dc * f, 4), round(da * g, 4)) for f in (0.5, 1, 1.5) for g in (0.5, 1, 1.5)} |
                   ({(round(dc * f, 4), x) for f in (0.5, 1, 1.5) for x in (0.25, 0.5)} if da == 0 else set()))
    out = _make(to, recipe["name"], recipe["about"], sz, cells)
    out["ported_from"] = {k: recipe[k] for k in ("model", "dc", "da")}
    out["at_source_doses"] = next((g for g in out["grid"] if (g["dc"], g["da"]) == (dc, da)), None)
    return out


def twins(src: str, layer: int, neuron: int, sign: int, dst: str, size: str = "full") -> dict:
    """The neurons of model `dst` whose direct vocabulary effect (logit lens of the down column) ranks the vocabulary
    most like this neuron's in `src` (the sizes share one vocabulary): Pearson as a prefilter, then Spearman."""
    from . import engine as E
    sz = SIZES[size]
    model, tok = _load(src)
    w = E.down_col(model, layer, neuron)
    z = E.lens(model, (sign * w / w.norm())[None])[0]
    top = [tok.decode([i]) for i in z.topk(12).indices.tolist()]
    del model, w
    E.free()

    def corr(a, b):
        a, b = a - a.mean(-1, keepdim=True), b - b.mean()
        return (a @ b) / (a.norm(dim=-1) * b.norm())

    model, tok = _load(dst)
    z = z.to(model.device)
    cands = []
    for L in range(model.config.num_hidden_layers):
        W = E.blocks(model)[L].mlp.down_proj.weight.float()
        Wn = (W / W.norm(dim=0, keepdim=True)).T
        for s in range(0, Wn.shape[0], 512):
            r = corr(E.lens(model, Wn[s:s + 512]), z).cpu()
            for sg in (1, -1):
                t = (sg * r).topk(min(sz["prefilter"], len(r)))
                cands += [(float(v), L, s + int(i), sg) for v, i in zip(t.values, t.indices)]
        cands = sorted(cands, reverse=True)[:sz["prefilter"]]
        del W, Wn
    rz, found = E.ranks(z[None])[0], []
    for pr, L, n, sg in cands:
        col = E.down_col(model, L, n)
        zz = E.lens(model, (sg * col / col.norm())[None])
        found.append({"layer": L, "neuron": n, "sign": sg, "spearman": round(float(corr(E.ranks(zz), rz)[0]), 4),
                      "pearson": round(pr, 4), "lens_top": [tok.decode([i]) for i in zz[0].topk(8).indices.tolist()]})
    found.sort(key=lambda x: -x["spearman"])
    return {"src": src, "layer": layer, "neuron": neuron, "sign": sign, "lens_top": top, "dst": dst,
            "twins": found[:sz["twins"]]}
