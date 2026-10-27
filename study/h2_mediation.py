"""Stage H2: does the Assistant Axis mediate a persona neuron's effect? (README.md, "H2")

Neurons: the alien and hostile neurons (N2742+, N8949+) and the two random neurons from H4's screen that broke identity
most (coherent IB first). Each is pushed at its dose from H4: for the known ones, the smallest swept dose whose rate
(CI for the alien neuron, IB for the rest) reaches 0.5 at coherence >= 0.8 (else the coherent dose with the highest
rate, flagged); for the random ones, the screen's dose.

Conditions, with w-hat = sign * W_down[:, n] / norm, u = the axis a_L's unit vector, c = w-hat . u, s = dose * m_L:
- steer: s * w-hat at the layer's output (the push itself);
- H2a: no_axis, w-hat without its u component, renormalised; axis_only, s * c * u; random_removed, w-hat without the
  component along u_r = |c| w-hat + sqrt(1 - c^2) r (r a random unit orthogonal to w-hat), renormalised;
- H2b: cap, the push while capping the projection on a at layers L+1...L+6 at the default replies' 25th percentile
  (S1b's floors); random_cap, the push while capping a random direction per layer at its own 25th percentile on the
  same default replies; cap_only, the axis cap with no push;
- and the unsteered model.
All on the 20 test identity prompts (primary) and the 20 everyday prompts, greedy, 96 tokens. The judge labels identity,
coherence and, for the alien neuron, concept ("an alien"); CM counts the aliens word family.

Per neuron, rate = CI (alien) or IB (the rest), on the identity prompts:
- prediction: rate(cap) <= 0.5 rate(steer), CM(cap) >= 0.7 CM(steer) where there is a concept, rate(random_cap) >= 0.8
  rate(steer);
- falsifier: rate(cap) >= 0.8 rate(steer);
- an exact one-sided McNemar test of cap < steer over prompts, and bootstrap 95% intervals for the ratios;
- only for neurons whose development dose reached 0.5 (the others are descriptive, prediction and falsified None).
Pooled over the calibrated neurons (confirmatory) and over all four (descriptive), with the prompts as the units: each
prompt's outcomes summed over neurons, a bootstrap over prompts and a sign-flip test over prompts.

Reads crucible-persona-neurons:/<s1a>/ (texts, calib), /<s1b>/axis.pt and /<h4>/h4.json; writes /<tag>/ (texts.jsonl,
labels.jsonl, h2.json). Hostile-neuron replies stay on the volume. Run through modal_run on an A10G: target this file,
args ["--model", "4b"], experiment "persona-neurons", config {"stage": "h2", "model": "4b"}. Smoke after H4's:
`python research/check.py research/2026-10-26-persona-neurons/h2_mediation.py --smoke`.
"""
import json
import math
import pathlib
import random
import sys
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import hf_cache  # noqa: E402
from jobs import MODELS, OUT, call, image, out_dir, pick_dose  # noqa: E402

app = modal.App("crucible-persona-neurons-h2")
FULL = dict(identity=20, everyday=20, max_new=96, cap_span=6, cap_replies=100, top_random=2, judge="9b",
            judge_batch=16, boot=2000)
SMOKE = dict(identity=2, everyday=1, max_new=8, cap_span=2, cap_replies=4, top_random=1, judge="0.8b", judge_batch=4,
             boot=50)
CONDITIONS = ["steer", "no_axis", "axis_only", "random_removed", "cap", "random_cap", "cap_only"]


def mcnemar_less(a: list[int], b: list[int]) -> float:
    """One-sided exact McNemar p-value that b's rate is below a's, paired."""
    lo, hi = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))
    n = lo + hi
    return 1.0 if n == 0 else sum(math.comb(n, k) for k in range(lo, n + 1)) / 2 ** n


def sign_flip_p(diffs: list[int], reps: int = 20000, seed: int = 0) -> float:
    """One-sided p that the per-prompt differences are centred above 0: random sign flips of whole prompts (each
    prompt's difference summed over neurons, so the prompts, not the neurons, are the independent units)."""
    rng, obs = random.Random(seed), sum(diffs)
    hits = sum(sum(d if rng.random() < 0.5 else -d for d in diffs) >= obs for _ in range(reps))
    return (hits + 1) / (reps + 1)


def ratio_ci(num: list[int], den: list[int], reps: int, seed: int = 0) -> list[float] | None:
    rng, vals = random.Random(seed), []
    for _ in range(reps):
        ix = [rng.randrange(len(den)) for _ in den]
        d = sum(den[i] for i in ix)
        if d:
            vals.append(sum(num[i] for i in ix) / d)
    if len(vals) < reps / 2:
        return None
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[min(len(vals) - 1, int(0.975 * len(vals)))], 4)]


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu=["A10G", "L4"], memory=32768, timeout=2 * 3600)
def run(model_key: str, s1a: str, s1b: str, h4: str, tag: str, sizes: dict) -> dict:
    import torch
    import persona
    import prompts as P
    from judge import score, summary

    t0 = time.time()
    if not modal.is_local():
        OUT.reload()
    ax = torch.load(out_dir(s1b) / "axis.pt")
    screen = json.load(open(out_dir(h4) / "h4.json"))
    recs = [json.loads(line) for line in open(out_dir(s1a) / "texts.jsonl")]
    m, unit, floors = ax["median_resid"], ax["unit"], ax["floors_p25"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
    n_layers, d_model = model.config.num_hidden_layers, unit.shape[1]
    gen = torch.Generator().manual_seed(0)
    CONCEPT_OF = P.NEURON_CONCEPTS if model_key == "4b" else P.SMOKE_CONCEPTS

    # The neurons and their doses.
    by_name = {nr["name"]: nr for nr in screen["neurons"]}
    chosen = []
    for name in ["alien", "hostile"] if model_key == "4b" else [nr["name"] for nr in screen["neurons"] if nr["known"]]:
        nr = by_name[name]
        dose, reached = pick_dose(nr["sweep"], "ci" if name in CONCEPT_OF else "ib")
        chosen.append({**nr, "dose": dose, "reached_half": reached})
    rand = sorted((nr for nr in screen["neurons"] if not nr["known"]), key=lambda nr: (-nr["ib_coherent"], -nr["ib"]))
    for nr in rand[:sizes["top_random"]]:
        chosen.append({**nr, "dose": screen["sizes"]["dose"], "reached_half": nr["ib"] >= 0.5})

    # Random directions for the controls: one orthogonal to each w-hat, one per capped layer.
    def rand_unit(orth=None):
        r = torch.randn(d_model, generator=gen)
        if orth is not None:
            r -= (r @ orth) * orth
        return r / r.norm()

    cap_layers = sorted({L for nr in chosen for L in range(nr["layer"] + 1, min(n_layers, nr["layer"] + 1 + sizes["cap_span"]))})
    rcap = {L: rand_unit() for L in cap_layers}
    hooks = persona.Hooks(model).__enter__()
    default_idx = [i for i, r in enumerate(recs) if r["kind"] == "default"][:sizes["cap_replies"]]
    proj = {L: [] for L in cap_layers}
    for s in range(0, len(default_idx), 16):
        idx = default_idx[s:s + 16]
        seqs = [persona.chat_ids(tok, recs[i]["question"], persona.DEFAULT_SYSTEMS[recs[i]["system"]]) for i in idx]
        replies = [tok.encode(recs[i]["reply"], add_special_tokens=False) for i in idx]
        got = persona.token_projections(model, hooks, [a + b for a, b in zip(seqs, replies)], [len(a) for a in seqs],
                                        {L: rcap[L][None] for L in cap_layers})
        for L in cap_layers:
            proj[L].append(got[L][:, 0])
    rfloor = {L: float(torch.cat(proj[L]).quantile(0.25)) for L in cap_layers}

    def vectors(nr):
        L = nr["layer"]
        w = persona.down_col(model, L, nr["neuron"]).cpu()
        wh, u, s = nr["sign"] * w / w.norm(), unit[L], nr["dose"] * float(m[L])
        c = float(wh @ u)
        no_axis = wh - c * u
        r = rand_unit(wh)
        ur = abs(c) * wh + math.sqrt(max(0.0, 1 - c * c)) * r
        removed = wh - float(wh @ ur) * ur
        return c, {"steer": s * wh, "no_axis": s * no_axis / no_axis.norm(), "axis_only": s * c * u,
                   "random_removed": s * removed / removed.norm(), "cap": s * wh, "random_cap": s * wh, "cap_only": None}

    prompts = [("identity", q) for q in P.IDENTITY_TEST[:sizes["identity"]]] + \
              [("everyday", q) for q in P.EVERYDAY_TEST[:sizes["everyday"]]]
    pids = [persona.chat_ids(tok, q) for _, q in prompts]
    texts = []

    def roll(nr, cond, vec):
        hooks.clear()
        if vec is not None:
            hooks.add(nr["layer"], vec)
        if cond in ("cap", "cap_only", "random_cap"):
            for L in range(nr["layer"] + 1, min(n_layers, nr["layer"] + 1 + sizes["cap_span"])):
                if cond == "random_cap":
                    hooks.cap(L, rcap[L], rfloor[L])
                else:
                    hooks.cap(L, unit[L], float(floors[L]))
        outs = persona.generate(model, tok, pids, sizes["max_new"])
        for (split, q), o in zip(prompts, outs):
            rec = {"kind": "h2", "neuron": nr.get("name"), "condition": cond, "split": split, "question": q,
                   "reply": tok.decode(o, skip_special_tokens=True)}
            if nr.get("name") in CONCEPT_OF:
                rec["concept"] = CONCEPT_OF[nr["name"]][0]
            texts.append(rec)

    roll({}, "none", None)
    for nr in chosen:
        nr["cos_axis"], vecs = vectors(nr)
        for cond in CONDITIONS:
            roll(nr, cond, vecs[cond])
        print(f"[h2] {nr['name']} at dose {nr['dose']}: {len(CONDITIONS)} conditions", flush=True)
    hooks.clear()
    hooks.__exit__()
    del hooks, model
    persona.free()

    judge, jtok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]), device)
    labels = score(judge, jtok, texts, ["identity", "coherent", "concept"], sizes["judge_batch"])
    del judge
    persona.free()

    # Per neuron x condition x split: per-prompt bits, then rates and the tests.
    bits = {}
    for t, lab in zip(texts, labels):
        b = bits.setdefault((t["neuron"], t["condition"], t["split"]), {"ib": [], "coherent": [], "ci": [], "cm": []})
        b["ib"].append(1 - lab["identity"])
        b["coherent"].append(lab["coherent"])
        if "concept" in lab:
            b["ci"].append(lab["concept"])
        if t["neuron"] in CONCEPT_OF:
            b["cm"].append(int(P.mentions(t["reply"], CONCEPT_OF[t["neuron"]][1])))

    def rates(b):
        n = len(b["ib"])
        out = {k: round(sum(v) / n, 4) for k, v in b.items() if v}
        out.update({f"{k}_ci95": persona.wilson(sum(v), n) for k, v in b.items() if v and k in ("ib", "ci")})
        return out

    table = {f"{k[0]}|{k[1]}|{k[2]}": rates(b) for k, b in bits.items()}
    tests, per_neuron = {}, {}
    for nr in chosen:
        key = "ci" if nr["name"] in CONCEPT_OF else "ib"
        get = lambda cond, k=key: bits[(nr["name"], cond, "identity")][k]  # noqa: E731
        steer, cap, rcap_ = get("steer"), get("cap"), get("random_cap")
        st = sum(steer) or None
        t = {"rate": key, "dose": nr["dose"], "reached_half": nr["reached_half"], "cos_axis": round(nr["cos_axis"], 4),
             "steer": round(sum(steer) / len(steer), 4),
             "cap_ratio": round(sum(cap) / st, 4) if st else None, "cap_ratio_ci": ratio_ci(cap, steer, sizes["boot"]),
             "random_cap_ratio": round(sum(rcap_) / st, 4) if st else None,
             "random_cap_ratio_ci": ratio_ci(rcap_, steer, sizes["boot"]),
             "p_cap_below_steer": round(mcnemar_less(steer, cap), 5),
             **{f"{c}_ratio": round(sum(get(c)) / st, 4) if st else None for c in ("no_axis", "axis_only", "random_removed")}}
        if nr["name"] in CONCEPT_OF:
            cm_s, cm_c = get("steer", "cm"), get("cap", "cm")
            t["cm_cap_ratio"] = round(sum(cm_c) / sum(cm_s), 4) if sum(cm_s) else None
        # A neuron whose development dose never reached 0.5 isn't the calibrated test H2 asks for: descriptive only.
        t["prediction"] = t["falsified"] = None
        if st and nr["reached_half"]:
            t["prediction"] = (t["cap_ratio"] <= 0.5 and t.get("cm_cap_ratio", 1.0) is not None
                               and t.get("cm_cap_ratio", 1.0) >= 0.7 and t["random_cap_ratio"] >= 0.8)
            t["falsified"] = t["cap_ratio"] >= 0.8
        tests[nr["name"]] = t
        per_neuron[nr["name"]] = (steer, cap, rcap_, nr["reached_half"])

    def pool(names):
        """Pooled over neurons with the prompts as the units: per-prompt sums, a prompt bootstrap, sign flips."""
        if not names:
            return {"neurons": []}
        S, C, R = ([sum(per_neuron[k][j][i] for k in names) for i in range(len(per_neuron[names[0]][0]))]
                   for j in range(3))
        ps = sum(S) or None
        out = {"neurons": names, "cap_ratio": round(sum(C) / ps, 4) if ps else None,
               "cap_ratio_ci": ratio_ci(C, S, sizes["boot"]),
               "random_cap_ratio": round(sum(R) / ps, 4) if ps else None,
               "random_cap_ratio_ci": ratio_ci(R, S, sizes["boot"]),
               "p_cap_below_steer": round(sign_flip_p([a - b for a, b in zip(S, C)]), 6)}
        if ps:
            out["falsified"] = out["cap_ratio"] >= 0.8
        return out

    pooled = pool([k for k, v in per_neuron.items() if v[3]])  # confirmatory: calibrated neurons only
    pooled_all = pool(list(per_neuron))  # descriptive

    odir = out_dir(tag)
    with open(odir / "texts.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    with open(odir / "labels.jsonl", "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    with open(odir / "h2.json", "w") as f:
        json.dump({"sizes": sizes, "neurons": [{k: nr[k] for k in ("name", "layer", "neuron", "sign", "dose",
                                                                    "reached_half", "cos_axis")} for nr in chosen],
                   "random_cap_floors": rfloor, "table": table, "tests": tests, "pooled": pooled,
                   "pooled_all": pooled_all}, f)
    if not modal.is_local():
        OUT.commit()

    res = {"model": model_key, "rollouts": len(texts), "pooled_cap_ratio": pooled.get("cap_ratio"),
           "pooled_random_cap_ratio": pooled.get("random_cap_ratio"), "pooled_p": pooled.get("p_cap_below_steer"),
           "pooled_neurons": len(pooled["neurons"]),
           "falsified": pooled.get("falsified"), "minutes": round((time.time() - t0) / 60, 1)}
    for name, t in tests.items():
        res[f"{name}_cap_ratio"] = t["cap_ratio"]
    details = {"tests": tests, "pooled": pooled, "pooled_all": pooled_all, "table": table,
               "judge": summary(labels, ["identity", "coherent", "concept"]), "loads": hf_cache.loads}
    return {**res, "details": details}


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False, s1a: str = "", s1b: str = "", h4: str = "", tag: str = ""):
    if smoke and model == "4b":
        model = "0.8b"
    sfx = "-smoke" if smoke else ""
    res = call(run, model, s1a or f"s1a-{model}{sfx}", s1b or f"s1b-{model}{sfx}", h4 or f"h4-{model}{sfx}",
                     tag or f"h2-{model}{sfx}", SMOKE if smoke else FULL)
    details = res.pop("details")
    print("CRUCIBLE_RESULT", json.dumps({**res, "smoke": smoke}))
    print("CRUCIBLE_RESULT", json.dumps({"details": details}))
