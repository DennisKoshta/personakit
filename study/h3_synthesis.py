"""Stage H3: can a persona be built from the two factors alone, leaving the Assistant (-a) and adding a concept (c_X)?
(README.md, "H3")

All steering adds m_mid * (dc * c-hat_X - da * a-hat) at the middle layer's output, where c-hat_X is the unit concept
direction from S1b (its axis component already removed), a-hat the axis's unit vector and m_mid the dose unit.

1. Development (the development concepts and identity prompts): a grid of concept dose {0.25, 0.5, 1, 1.5, 2} x axis
   dose {0, 0.25, 0.5, 0.75, 1}. The rule, fixed before it runs: (d2, a2) is the cell with both doses above 0 and the
   highest mean CI among cells with mean coherence >= 0.8 (ties to the smaller total dose); d1 = d2 / 2, a1 = a2 / 2.
2. Test (the test concepts and the 20 test identity prompts): concept {0, d1, d2} x axis {0, a1, a2}, plus the 20
   everyday prompts at (0, 0) and (d2, a2), plus the alien neuron at a matched dose (below).
3. The judge labels concept, identity and coherence; CM counts each concept's word family.
4. Tests:
   - a logistic model CI ~ c + a + c*a (doses scaled to 0, 0.5, 1; a 0.01 ridge against separation), the interaction's
     95% interval from a bootstrap over (concept, prompt) units; the difference in differences on the rate scale too;
   - CI >= 0.5 at some test cell with coherence >= 0.8;
   - the interaction counts as positive only if the difference in differences' interval is above 0 too (separation
     lets the ridge invent one);
   - next-token KL from the unsteered model on 200 held-out Pile passages (128 tokens) and chat-form MMLU (300 items,
     paired) at (d2, a2) for each test concept, against the alien neuron at the swept dose whose development CI is
     closest to the recipe's. The comparison counts only if their test CIs are within 0.15 (capability_matched).

Reads crucible-persona-neurons:/<s1b>/axis.pt and /<h4>/h4.json (on other sizes, X2, there is no screened neuron:
--h4 none, the default there, drops the neuron comparison); writes /<tag>/ (texts.jsonl, labels.jsonl, h3.json).
Run through modal_run on an A10G: target this file, args ["--model", "4b"], experiment "persona-neurons", config
{"stage": "h3", "model": "4b"}. Smoke after H4's:
`python research/check.py research/2026-10-26-persona-neurons/h3_synthesis.py --smoke`.
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
from jobs import MODELS, OUT, call, image, out_dir, sized  # noqa: E402

app = modal.App("crucible-persona-neurons-h3")
PILE = ("NeelNanda/pile-10k", "127bfedcd5047750df5ccf3a12979a47bfa0bafa")
FULL = dict(dev_c=[0.25, 0.5, 1.0, 1.5, 2.0], dev_a=[0, 0.25, 0.5, 0.75, 1.0], dev_q=10, test_q=20, everyday=20,
            max_new=96, batch=100, kl_docs=200, kl_len=128, kl_from=9000, mmlu=300, judge="9b", judge_batch=16,
            boot=1000)
SMOKE = dict(dev_c=[0.5, 1.0], dev_a=[0, 0.5], dev_q=2, test_q=2, everyday=1, max_new=8, batch=16, kl_docs=4,
             kl_len=32, kl_from=9000, mmlu=8, judge="0.8b", judge_batch=4, boot=50, concepts=2)


def logistic(X, y, ridge: float = 0.01, iters: int = 30):
    """Ridge logistic regression by Newton's method (the intercept barely penalised, to stay finite when every label
    is the same); X [n, k] with a ones column first."""
    import torch
    beta = torch.zeros(X.shape[1], dtype=torch.float64)
    pen = torch.full((X.shape[1],), ridge, dtype=torch.float64)
    pen[0] = 1e-4
    for _ in range(iters):
        p = torch.sigmoid(X @ beta)
        g = X.T @ (p - y) + pen * beta
        H = (X * (p * (1 - p))[:, None]).T @ X + torch.diag(pen)
        beta = beta - torch.linalg.solve(H, g)
    return beta


def pick_cell(cells: dict) -> tuple[float, float]:
    """(d2, a2) by the development rule: both doses above 0, coherence >= 0.8, highest CI, ties to the smaller sum."""
    ok = [(v["ci"], -(dc + da), dc, da) for (dc, da), v in cells.items() if dc > 0 and da > 0 and v["coherent"] >= 0.8]
    if not ok:
        return min((dc, da) for dc, da in cells if dc > 0 and da > 0)
    _, _, dc, da = max(ok)
    return dc, da


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu=["A10G", "L4"], memory=32768, timeout=2 * 3600)
def run(model_key: str, s1b: str, h4: str, tag: str, sizes: dict) -> dict:
    import torch
    import persona
    import prompts as P
    from judge import score, summary

    t0 = time.time()
    if not modal.is_local():
        OUT.reload()
    ax = torch.load(out_dir(s1b) / "axis.pt")
    screen = json.load(open(out_dir(h4) / "h4.json")) if h4 != "none" else None  # none: no neuron to compare (X2)
    m, unit, cdirs = ax["median_resid"], ax["unit"], ax["concept_dirs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
    n_layers = model.config.num_hidden_layers
    mid = n_layers // 2
    hooks = persona.Hooks(model).__enter__()
    dev_concepts = P.CONCEPTS_DEV[:sizes.get("concepts", 5)]
    test_concepts = P.CONCEPTS_TEST[:sizes.get("concepts", 5)]
    neuron_concepts = P.NEURON_CONCEPTS if model_key == "4b" else P.SMOKE_CONCEPTS
    nname = next(iter(neuron_concepts)) if screen else None
    if screen:
        nr = next(x for x in screen["neurons"] if x["name"] == nname)
        w_neuron = persona.down_col(model, nr["layer"], nr["neuron"]).cpu()
    neuron_add = ndose = None  # set once the recipe's development CI is known, to match it

    def recipe(concept, dc, da):
        c = cdirs[concept][mid]
        return float(m[mid]) * (dc * c / c.norm() - da * unit[mid])

    def roll(jobs: list[dict]) -> list[dict]:
        """jobs: dicts with question, concept, dc, da (or neuron=True); the replies, greedy, in order."""
        out = []
        for s in range(0, len(jobs), sizes["batch"]):
            chunk = jobs[s:s + sizes["batch"]]
            hooks.clear()
            V = torch.zeros(len(chunk), unit.shape[1])
            NV = torch.zeros(len(chunk), unit.shape[1])
            for b, j in enumerate(chunk):
                if j.get("neuron"):
                    NV[b] = neuron_add[1]
                elif j["dc"] or j["da"]:
                    V[b] = recipe(j["concept"], j["dc"], j["da"])
            hooks.add(mid, V)
            if NV.abs().sum():
                hooks.add(neuron_add[0], NV)
            outs = persona.generate(model, tok, [persona.chat_ids(tok, j["question"]) for j in chunk], sizes["max_new"])
            out += [{**j, "reply": tok.decode(o, skip_special_tokens=True)} for j, o in zip(chunk, outs)]
        hooks.clear()
        return out

    # 1. Development grid.
    dev_jobs = [{"kind": "h3dev", "concept": c, "dc": dc, "da": da, "question": q}
                for dc in sizes["dev_c"] for da in sizes["dev_a"] for c in dev_concepts
                for q in P.IDENTITY_DEV[:sizes["dev_q"]]]
    dev = roll(dev_jobs)
    print(f"[h3] development grid: {len(dev)} replies", flush=True)
    hooks.__exit__()
    del hooks, model
    persona.free()
    judge, jtok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]), device)
    dev_labels = score(judge, jtok, dev, ["concept", "coherent"], sizes["judge_batch"])
    dev_cells = {}
    for j, lab in zip(dev, dev_labels):
        dev_cells.setdefault((j["dc"], j["da"]), []).append(lab)
    dev_cells = {k: {"ci": sum(x["concept"] for x in v) / len(v), "coherent": sum(x["coherent"] for x in v) / len(v)}
                 for k, v in dev_cells.items()}
    d2, a2 = pick_cell(dev_cells)
    d1, a1 = d2 / 2, a2 / 2
    best_concept_only = max(((k, v) for k, v in dev_cells.items() if k[1] == 0 and v["coherent"] >= 0.8),
                            key=lambda kv: kv[1]["ci"], default=None)
    print(f"[h3] picked d2 {d2}, a2 {a2} (dev CI {dev_cells[(d2, a2)]['ci']:.2f})", flush=True)
    # The alien neuron at the coherent swept dose whose development CI is closest to the recipe's (ties to the
    # smaller dose), so the capability comparison is at matched persona strength.
    if screen:
        ok = [(float(d), v["ci"]) for d, v in nr["sweep"].items() if v["coherent"] >= 0.8] or \
            [(float(d), v["ci"]) for d, v in nr["sweep"].items()]
        ndose = min(ok, key=lambda x: (abs(x[1] - dev_cells[(d2, a2)]["ci"]), x[0]))[0]
        neuron_add = (nr["layer"], nr["sign"] * ndose * float(m[nr["layer"]]) * w_neuron / w_neuron.norm())
    del judge
    persona.free()

    # 2. Test grid, everyday carry-over, the alien neuron.
    model, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
    hooks = persona.Hooks(model).__enter__()
    test_q = P.IDENTITY_TEST[:sizes["test_q"]]
    jobs = [{"kind": "h3", "concept": c, "dc": dc, "da": da, "question": q, "split": "identity"}
            for dc in (0, d1, d2) for da in (0, a1, a2) for c in test_concepts for q in test_q]
    jobs += [{"kind": "h3", "concept": c, "dc": dc, "da": da, "question": q, "split": "everyday"}
             for dc, da in ((0, 0), (d2, a2)) for c in test_concepts for q in P.EVERYDAY_TEST[:sizes["everyday"]]]
    jobs += [{"kind": "h3", "concept": neuron_concepts[nname][0], "neuron": True, "dc": 0, "da": 0, "question": q,
              "split": "identity"} for q in test_q if screen]
    texts = roll(jobs)
    print(f"[h3] test grid: {len(texts)} replies", flush=True)

    # KL on held-out Pile passages and chat-form MMLU, at (d2, a2) per test concept and for the alien neuron.
    pile = hf_cache.dataset(*PILE, split="train")
    docs = [tok.encode(pile[i]["text"], add_special_tokens=False)[:sizes["kl_len"]]
            for i in range(sizes["kl_from"], sizes["kl_from"] + sizes["kl_docs"])]
    docs = [x for x in docs if len(x) >= 16]
    setups = {f"recipe:{c}": [(mid, recipe(c, d2, a2))] for c in test_concepts}
    if screen:
        setups[f"neuron:{nname}"] = [neuron_add]
    kl = persona.kl_from_base(model, hooks, docs, setups)
    rows = persona.mmlu_rows(sizes["mmlu"])
    mm = {"none": persona.mmlu(model, tok, rows)}
    for k, adds in setups.items():
        hooks.clear()
        for L, v in adds:
            hooks.add(L, v)
        mm[k] = persona.mmlu(model, tok, rows)
    hooks.clear()
    hooks.__exit__()
    del hooks, model
    persona.free()

    judge, jtok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]), device)
    labels = score(judge, jtok, texts, ["concept", "identity", "coherent"], sizes["judge_batch"])
    del judge
    persona.free()

    # 4. The tests.
    fam = {**{c: c for c in test_concepts}, **({neuron_concepts[nname][0]: neuron_concepts[nname][1]} if screen else {})}
    for t, lab in zip(texts, labels):
        t["ci"], t["ib"], t["coherent"] = lab["concept"], 1 - lab["identity"], lab["coherent"]
        t["cm"] = int(P.mentions(t["reply"], fam[t["concept"]]))
    grid = [t for t in texts if t["split"] == "identity" and not t.get("neuron")]
    cells = {}
    for t in grid:
        cells.setdefault((t["dc"], t["da"]), []).append(t)

    def rates(ts):
        n = len(ts)
        return {k: round(sum(t[k] for t in ts) / n, 4) for k in ("ci", "ib", "cm", "coherent")} | \
            {"ci_ci95": persona.wilson(sum(t["ci"] for t in ts), n), "n": n}

    cell_rates = {f"{dc}|{da}": rates(v) for (dc, da), v in sorted(cells.items())}
    units = sorted({(t["concept"], t["question"]) for t in grid})
    by_unit = {u: [t for t in grid if (t["concept"], t["question"]) == u] for u in units}

    def fit(ts):
        X = torch.tensor([[1.0, t["dc"] / d2, t["da"] / a2, t["dc"] / d2 * t["da"] / a2] for t in ts], dtype=torch.float64)
        y = torch.tensor([float(t["ci"]) for t in ts], dtype=torch.float64)
        return logistic(X, y)

    beta = fit(grid)
    rng, inter = random.Random(0), []
    for _ in range(sizes["boot"]):
        inter.append(float(fit([t for _ in units for t in by_unit[units[rng.randrange(len(units))]]])[3]))
    inter.sort()
    inter_ci = [round(inter[int(0.025 * len(inter))], 4), round(inter[min(len(inter) - 1, int(0.975 * len(inter)))], 4)]

    def did(ts):
        r = {}
        for t in ts:
            r.setdefault((t["dc"], t["da"]), []).append(t["ci"])
        mean = {k: sum(v) / len(v) for k, v in r.items()}
        return mean[(d2, a2)] - mean[(d2, 0)] - mean[(0, a2)] + mean[(0, 0)]

    dd = []
    for _ in range(sizes["boot"]):
        dd.append(did([t for _ in units for t in by_unit[units[rng.randrange(len(units))]]]))
    dd.sort()
    coherent_hits = {k: v for k, v in cell_rates.items() if v["ci"] >= 0.5 and v["coherent"] >= 0.8}
    neuron_rates = rates([t for t in texts if t.get("neuron")]) if screen else None
    per_concept = {c: rates([t for t in grid if t["concept"] == c and (t["dc"], t["da"]) == (d2, a2)]) for c in test_concepts}
    everyday = {f"{dc}|{da}": rates([t for t in texts if t["split"] == "everyday" and (t["dc"], t["da"]) == (dc, da)])
                for dc, da in ((0, 0), (d2, a2))}
    recipe_kl = sum(kl[f"recipe:{c}"] for c in test_concepts) / len(test_concepts)
    recipe_mmlu = sum(sum(mm[f"recipe:{c}"]) for c in test_concepts) / (len(test_concepts) * len(rows))
    did_ci = [round(dd[int(0.025 * len(dd))], 4), round(dd[min(len(dd) - 1, int(0.975 * len(dd)))], 4)]
    # Separation (a cell all 0s or all 1s) lets the ridge invent an interaction, so the logistic interval only counts
    # when the rate-scale difference in differences agrees.
    fitted = torch.sigmoid(torch.tensor([[1.0, dc / d2, da / a2, dc / d2 * da / a2] for dc, da in cells],
                                        dtype=torch.float64) @ beta)
    separation = bool(((fitted < 1e-3) | (fitted > 1 - 1e-3)).any())
    matched = abs(neuron_rates["ci"] - cell_rates[f"{d2}|{a2}"]["ci"]) <= 0.15 if screen else None
    tests = {"interaction_logit": round(float(beta[3]), 4), "interaction_ci95": inter_ci, "separation": separation,
             "interaction_positive": inter_ci[0] > 0 and did_ci[0] > 0, "did": round(did(grid), 4), "did_ci95": did_ci,
             "capability_matched": matched, "neuron_dose": ndose,
             "coherent_cells_ci_half": sorted(coherent_hits), "best_cell": cell_rates[f"{d2}|{a2}"],
             "recipe_kl": round(recipe_kl, 4), "neuron_kl": round(kl[f"neuron:{nname}"], 4) if screen else None,
             "mmlu_none": round(sum(mm["none"]) / len(rows), 4), "recipe_mmlu": round(recipe_mmlu, 4),
             "neuron_mmlu": round(sum(mm[f"neuron:{nname}"]) / len(rows), 4) if screen else None,
             "neuron": neuron_rates}

    odir = out_dir(tag)
    with open(odir / "texts.jsonl", "w") as f:
        for t in dev + texts:
            f.write(json.dumps(t) + "\n")
    with open(odir / "labels.jsonl", "w") as f:
        for lab in dev_labels + labels:
            f.write(json.dumps(lab) + "\n")
    with open(odir / "h3.json", "w") as f:
        json.dump({"sizes": sizes, "layer": mid, "doses": {"d1": d1, "d2": d2, "a1": a1, "a2": a2},
                   "dev_cells": {f"{k[0]}|{k[1]}": v for k, v in dev_cells.items()}, "cells": cell_rates,
                   "per_concept": per_concept, "everyday": everyday, "kl": kl, "mmlu_bits": mm,
                   "neuron": {"name": nname, "dose": ndose}, "tests": tests, "beta": [float(b) for b in beta]}, f)
    if not modal.is_local():
        OUT.commit()

    res = {"model": model_key, "layer": mid, "d2": d2, "a2": a2, "dev_ci_best": round(dev_cells[(d2, a2)]["ci"], 4),
           "test_ci_best": tests["best_cell"]["ci"], "test_coherent_best": tests["best_cell"]["coherent"],
           "interaction_logit": tests["interaction_logit"], "interaction_positive": tests["interaction_positive"],
           "capability_matched": tests["capability_matched"],
           "recipe_kl": tests["recipe_kl"], "neuron_kl": tests["neuron_kl"], "mmlu_none": tests["mmlu_none"],
           "recipe_mmlu": tests["recipe_mmlu"], "neuron_mmlu": tests["neuron_mmlu"],
           "minutes": round((time.time() - t0) / 60, 1)}
    details = {"tests": tests, "cells": cell_rates, "per_concept": per_concept, "everyday": everyday,
               "dev_cells": {f"{k[0]}|{k[1]}": {kk: round(vv, 4) for kk, vv in v.items()} for k, v in dev_cells.items()},
               "best_concept_only_dev": best_concept_only and [list(best_concept_only[0]), best_concept_only[1]],
               "kl": {k: round(v, 4) for k, v in kl.items()}, "judge": summary(labels, ["concept", "identity", "coherent"]),
               "loads": hf_cache.loads}
    return {**res, "details": details}


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False, s1b: str = "", h4: str = "", tag: str = ""):
    if smoke and model == "4b":
        model = "0.8b"
    sfx = "-smoke" if smoke else ""
    h4 = h4 or (f"h4-{model}{sfx}" if model == "4b" or smoke else "none")  # other sizes have no screened neuron
    res = call(sized(run, model), model, s1b or f"s1b-{model}{sfx}", h4, tag or f"h3-{model}{sfx}",
                     SMOKE if smoke else FULL)
    details = res.pop("details")
    print("CRUCIBLE_RESULT", json.dumps({**res, "smoke": smoke}))
    print("CRUCIBLE_RESULT", json.dumps({"details": details}))
