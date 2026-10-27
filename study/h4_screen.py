"""Stage H4: does a neuron's alignment with the Assistant Axis predict whether pushing it breaks the model's identity?
(README.md, "H4"). Also the known neurons' dose sweep, which H2 picks its doses from.

- Screen: 100 random neurons at each of L10, L14, L18 and L22 (seed 0, signs at random since the known recipes use
  both), plus the known neurons. Each is pushed at dose 0.75 on the 10 development identity prompts, greedy, 64 tokens.
- Sweep: the known neurons at doses 0.25-2 on the same prompts. The alien neuron's replies also get the concept label.
- The judge (the 9B) labels identity and coherence, and concept ("an alien") for the alien neuron.
- Predictors, per neuron:
  - cos(sign * w, -a_L), with the axis from S1b;
  - the logit-lens peakiness: the max z-score over the vocabulary of the final-normed, unembedded sign * w;
  - the push over the natural extreme: delta over the neuron's largest natural activation of that sign (Pile or chat).
- Outcome: IB, the share of the 10 replies that don't present as an AI assistant. A neuron breaks identity at IB >= 0.4.
  Spearman's rho and AUROC for each predictor, with bootstrap 95% intervals over neurons, and the AUROC differences
  between cos and the two baselines. Exploratory: the same with coherent IB (not an AI, and coherent).

A push of delta on neuron n is the same edit as adding delta * W_down[:, n] at the layer's output (S1a checks it), so
one batch steers a different neuron in each row. Reads crucible-persona-neurons:/<s1a>/calib.pt and /<s1b>/axis.pt;
writes /<tag>/ (texts.jsonl, labels.jsonl, h4.json). Hostile-neuron replies and the known neurons' logit-lens tokens stay
on the volume, never in the result. Run through modal_run on an A10G: target this file, args ["--model", "4b"],
experiment "persona-neurons", config {"stage": "h4", "model": "4b"}. Smoke after S1b's:
`python research/check.py research/2026-10-26-persona-neurons/h4_screen.py --smoke`.
"""
import json
import pathlib
import random
import sys
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import hf_cache  # noqa: E402
from jobs import MODELS, OUT, call, image, out_dir  # noqa: E402

app = modal.App("crucible-persona-neurons-h4")
FULL = dict(per_layer=100, dose=0.75, sweep=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0], identity_q=10, max_new=64, batch=128,
            judge="9b", judge_batch=16, boot=2000)
SMOKE = dict(per_layer=3, dose=0.75, sweep=[0.75, 1.5], identity_q=2, max_new=8, batch=8, judge="0.8b", judge_batch=4,
             boot=50)
H4_LAYERS = [10, 14, 18, 22]
BREAKS = 0.4


def ranks(x: list[float]) -> list[float]:
    """Average ranks (ties share theirs), from 1."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    r, i = [0.0] * len(x), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else None


def auroc(score: list[float], pos: list[bool]) -> float | None:
    """P(a positive outscores a negative), ties counting half (Mann-Whitney)."""
    n1 = sum(pos)
    n0 = len(pos) - n1
    if not n1 or not n0:
        return None
    r = ranks(score)
    return (sum(ri for ri, p in zip(r, pos) if p) - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(stat, n: int, reps: int, seed: int = 0) -> list[float] | None:
    """Percentile 95% interval of stat(indices) over resamples of n items (resamples where stat is None are skipped)."""
    rng, vals = random.Random(seed), []
    for _ in range(reps):
        v = stat([rng.randrange(n) for _ in range(n)])
        if v is not None:
            vals.append(v)
    if len(vals) < reps / 2:
        return None
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[min(len(vals) - 1, int(0.975 * len(vals)))], 4)]


def analyse(neurons: list[dict], outcome: str, reps: int) -> dict:
    """Each predictor against the outcome (IB or coherent IB) across the screened neurons."""
    preds = ["cos_away", "peakiness", "over_natural"]
    y = [nr[outcome] for nr in neurons]
    pos = [v >= BREAKS for v in y]
    res = {"n": len(neurons), "breaks": sum(pos)}
    for p in preds:
        x = [nr[p] for nr in neurons]
        res[p] = {"rho": spearman(x, y), "auroc": auroc(x, pos),
                  "rho_ci": boot_ci(lambda ix: spearman([x[i] for i in ix], [y[i] for i in ix]), len(x), reps),
                  "auroc_ci": boot_ci(lambda ix: auroc([x[i] for i in ix], [pos[i] for i in ix]), len(x), reps)}
    xc = [nr["cos_away"] for nr in neurons]
    for p in preds[1:]:
        xb = [nr[p] for nr in neurons]

        def diff(ix, xb=xb):
            a, b = auroc([xc[i] for i in ix], [pos[i] for i in ix]), auroc([xb[i] for i in ix], [pos[i] for i in ix])
            return None if a is None or b is None else a - b
        res[f"cos_minus_{p}"] = {"auroc_diff": diff(range(len(xc))), "ci": boot_ci(diff, len(xc), reps)}
    return res


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu=["A10G", "L4"], memory=32768, timeout=2 * 3600)
def run(model_key: str, s1a: str, s1b: str, tag: str, sizes: dict) -> dict:
    import torch
    import persona
    import prompts as P
    from judge import score, summary

    t0 = time.time()
    if not modal.is_local():
        OUT.reload()
    ax, calib = torch.load(out_dir(s1b) / "axis.pt"), torch.load(out_dir(s1a) / "calib.pt")
    m, unit = ax["median_resid"], ax["unit"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
    n_layers = model.config.num_hidden_layers
    n_neurons = persona.blocks(model)[0].mlp.down_proj.weight.shape[1]

    # The neurons: known ones (with their signs and the sweep), then the random screen.
    known, concepts = (P.KNOWN, P.NEURON_CONCEPTS) if model_key == "4b" else (P.SMOKE_KNOWN, P.SMOKE_CONCEPTS)
    neurons = [{"name": k, "layer": L, "neuron": n, "sign": s, "known": True} for k, (L, n, s) in known.items()]
    rng, taken = random.Random(0), {(L, n) for L, n, _ in known.values()}
    for L in [L for L in H4_LAYERS if L < n_layers]:
        for n in rng.sample([n for n in range(n_neurons) if (L, n) not in taken], sizes["per_layer"]):
            neurons.append({"name": f"L{L}N{n}", "layer": L, "neuron": n, "sign": rng.choice([1, -1]), "known": False})

    # Predictors, from the weights and S1's calibration.
    with torch.inference_mode():
        for nr in neurons:
            L, n, s = nr["layer"], nr["neuron"], nr["sign"]
            w = persona.down_col(model, L, n)
            wh = s * w / w.norm()
            nr["cos_away"] = float(-(unit[L].to(device) @ wh))
            nr["cos_concepts"] = {c: round(float((v[L] / v[L].norm()).to(device) @ wh), 4)
                                  for c, v in ax["concept_dirs"].items()}
            z = model.lm_head(model.model.norm(wh.to(model.dtype)[None]))[0].float()
            nr["peakiness"] = float((z.max() - z.mean()) / z.std())
            nr["lens_top"] = tok.convert_ids_to_tokens(z.topk(8).indices.tolist())
            nat = max(abs(float(calib[src]["hi" if s > 0 else "lo"][L, n])) for src in ("pile", "chat"))
            nr["delta_per_dose"] = float(m[L]) / float(w.norm())
            nr["natural"] = nat
            nr["over_natural"] = sizes["dose"] * nr["delta_per_dose"] / max(nat, 1e-6)

    # Rollouts: unsteered, the screen at one dose, the known neurons' sweep.
    prompts = P.IDENTITY_DEV[:sizes["identity_q"]]
    pids = [persona.chat_ids(tok, q) for q in prompts]
    rows = [(None, 0.0, qi) for qi in range(len(prompts))]
    for i, nr in enumerate(neurons):
        doses = sorted(set(sizes["sweep"]) | {sizes["dose"]}) if nr["known"] else [sizes["dose"]]
        rows += [(i, d, qi) for d in doses for qi in range(len(prompts))]
    hooks, texts = persona.Hooks(model).__enter__(), []
    d_model = unit.shape[1]
    for s in range(0, len(rows), sizes["batch"]):
        chunk = rows[s:s + sizes["batch"]]
        hooks.clear()
        vecs = {}
        for b, (i, dose, _) in enumerate(chunk):
            if i is None:
                continue
            nr = neurons[i]
            w = persona.down_col(model, nr["layer"], nr["neuron"])
            vecs.setdefault(nr["layer"], torch.zeros(len(chunk), d_model, device=device))[b] = \
                nr["sign"] * dose * float(m[nr["layer"]]) * w / w.norm()
        for L, v in vecs.items():
            hooks.add(L, v)
        outs = persona.generate(model, tok, [pids[qi] for _, _, qi in chunk], sizes["max_new"])
        for (i, dose, qi), o in zip(chunk, outs):
            nr = neurons[i] if i is not None else {}
            rec = {"kind": "h4", "neuron": nr.get("name"), "layer": nr.get("layer"), "n": nr.get("neuron"),
                   "sign": nr.get("sign"), "dose": dose, "q": qi, "question": prompts[qi],
                   "reply": tok.decode(o, skip_special_tokens=True)}
            if nr.get("name") in concepts:
                rec["concept"] = concepts[nr["name"]][0]
            texts.append(rec)
        if (s // sizes["batch"]) % 5 == 0:
            print(f"[h4] generated {min(s + sizes['batch'], len(rows))}/{len(rows)}", flush=True)
    hooks.clear()
    hooks.__exit__()
    del hooks, model
    persona.free()

    judge, jtok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]), device)
    labels = score(judge, jtok, texts, ["identity", "coherent", "concept"], sizes["judge_batch"])
    del judge
    persona.free()

    # Per neuron and dose: IB, coherent IB, coherence (and CI for the alien neuron).
    cells = {}
    for t, lab in zip(texts, labels):
        cells.setdefault((t["neuron"], t["dose"]), []).append(lab)

    def rates(labs):
        n = len(labs)
        ib = sum(1 - lab["identity"] for lab in labs)
        out = {"ib": round(ib / n, 4), "ib_ci": persona.wilson(ib, n),
               "ib_coherent": round(sum((1 - lab["identity"]) * lab["coherent"] for lab in labs) / n, 4),
               "coherent": round(sum(lab["coherent"] for lab in labs) / n, 4)}
        if all("concept" in lab for lab in labs):
            ci = sum(lab["concept"] for lab in labs)
            out.update({"ci": round(ci / n, 4), "ci_ci": persona.wilson(ci, n)})
        return out

    base = rates(cells[(None, 0.0)])
    for nr in neurons:
        nr.update(rates(cells[(nr["name"], sizes["dose"])]))
        if nr["known"]:
            nr["sweep"] = {str(d): rates(cells[(nr["name"], d)]) for d in sorted({d for (k, d) in cells if k == nr["name"]})}
    screen = analyse(neurons, "ib", sizes["boot"])
    screen_coherent = analyse(neurons, "ib_coherent", sizes["boot"])
    random_only = analyse([nr for nr in neurons if not nr["known"]], "ib", sizes["boot"])

    odir = out_dir(tag)
    with open(odir / "texts.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    with open(odir / "labels.jsonl", "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    with open(odir / "h4.json", "w") as f:
        json.dump({"sizes": sizes, "base": base, "neurons": neurons, "screen": screen,
                   "screen_coherent": screen_coherent, "random_only": random_only}, f)
    if not modal.is_local():
        OUT.commit()

    def short(a):
        return {p: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in a[p].items()}
                for p in a if isinstance(a[p], dict)} | {"n": a["n"], "breaks": a["breaks"]}

    res = {"model": model_key, "neurons": len(neurons), "rollouts": len(texts), "base_ib": base["ib"],
           "breaks": screen["breaks"], "auroc_cos": screen["cos_away"]["auroc"], "rho_cos": screen["cos_away"]["rho"],
           "auroc_peakiness": screen["peakiness"]["auroc"], "auroc_over_natural": screen["over_natural"]["auroc"],
           "minutes": round((time.time() - t0) / 60, 1)}
    top = sorted((nr for nr in neurons if not nr["known"]), key=lambda nr: (-nr["ib"], -nr["coherent"]))[:8]
    details = {"screen": short(screen), "screen_coherent": short(screen_coherent), "random_only": short(random_only),
               "base": base, "known": {nr["name"]: {k: nr[k] for k in ("ib", "ib_coherent", "coherent", "cos_away",
                                                                       "peakiness", "over_natural", "sweep")}
                                       for nr in neurons if nr["known"]},
               "top_random": [{k: nr[k] for k in ("name", "sign", "ib", "coherent", "cos_away", "peakiness", "over_natural")}
                              for nr in top],
               "judge": summary(labels, ["identity", "coherent", "concept"]), "loads": hf_cache.loads}
    return {**res, "details": details}


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False, s1a: str = "", s1b: str = "", tag: str = ""):
    if smoke and model == "4b":
        model = "0.8b"
    suffix = "-smoke" if smoke else ""
    res = call(run, model, s1a or f"s1a-{model}{suffix}", s1b or f"s1b-{model}{suffix}", tag or f"h4-{model}{suffix}",
                     SMOKE if smoke else FULL)
    details = res.pop("details")
    print("CRUCIBLE_RESULT", json.dumps({**res, "smoke": smoke}))
    print("CRUCIBLE_RESULT", json.dumps({"details": details}))
