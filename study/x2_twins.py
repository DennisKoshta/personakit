"""Stage X2, neuron twins: does the 4B's alien neuron have a twin in another Qwen3.5 size? (README.md, "X2")

The sizes share one vocabulary, so a neuron's direct vocabulary effect (the logit lens of its down column: unit column,
final norm, unembedding) can be compared across models.
1. The 4B's N2742+ lens vector.
2. Every neuron of the target model (both signs): the Pearson correlation of its lens vector with N2742's, as a cheap
   prefilter to the 200 best; then the exact Spearman correlation (the pre-registered statistic) on those, and the top
   3 are the twins. Three random neurons, one at each twin's layer with a random sign (seed 0), are the controls.
3. Each pushed at doses 0.5, 1 and 1.5 (the target's own dose unit, from its S1a) on the 20 test identity prompts,
   greedy, 96 tokens. The judge labels concept ("an alien"), identity and coherence; CM counts the aliens family.
4. Per neuron, its best CI at coherence >= 0.8. Twins beat random if the twins' pooled best-coherent CI is above the
   random neurons', with a bootstrap 95% interval over prompts for the difference (each replicate re-chooses every
   neuron's dose and eligibility on its resampled prompts).

Reads crucible-persona-neurons:/<s1a>/calib.pt (the target's); writes /<tag>/ (texts.jsonl, labels.jsonl, x2.json).
Run through modal_run on an A10G: target this file, args ["--model", "2b"] (or "9b"), experiment "persona-neurons",
config {"stage": "x2-twins", "model": "2b"}. Smoke: `python research/check.py
research/2026-10-26-persona-neurons/x2_twins.py --smoke` (the 0.8B as both source and target).
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

app = modal.App("crucible-persona-neurons-x2")
FULL = dict(source="4b", prefilter=200, twins=3, doses=[0.5, 1.0, 1.5], test_q=20, max_new=96, batch=60, chunk=512,
            judge="9b", judge_batch=16, boot=2000)
SMOKE = dict(source="0.8b", prefilter=8, twins=1, doses=[1.0], test_q=2, max_new=8, batch=8, chunk=512, judge="0.8b",
             judge_batch=4, boot=50, layers=[10])  # layer 10 holds the source: it must find itself
SOURCE = {"4b": ("alien", 22, 2742, 1), "0.8b": ("smoke_neuron", 10, 7, -1)}


def lens(model, cols):
    """[k, V] float: the logit lens of unit columns cols ([k, d]): final norm, then the unembedding."""
    import torch
    with torch.inference_mode():
        return model.lm_head(model.model.norm(cols.to(model.dtype))).float()


def ranks(x):
    """Ranks along the last dimension of [k, V], ties given their average rank (the lens runs in bf16, so its logits
    tie often)."""
    import torch
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        v, idx = x[i].sort()
        _, inv, cnt = torch.unique_consecutive(v, return_inverse=True, return_counts=True)
        first = (cnt.cumsum(0) - cnt).to(x.dtype)
        out[i, idx] = (first + (cnt.to(x.dtype) - 1) / 2)[inv]
    return out


def corr(a, b):
    """Pearson correlation of each row of a with the vector b."""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean()
    return (a @ b) / (a.norm(dim=-1) * b.norm())


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu=["A10G", "L4"], memory=32768, timeout=2 * 3600)
def run(model_key: str, s1a: str, tag: str, sizes: dict) -> dict:
    import torch
    import persona
    import prompts as P
    from judge import score, summary

    t0 = time.time()
    if not modal.is_local():
        OUT.reload()
    m = torch.load(out_dir(s1a) / "calib.pt")["pile"]["norm_q10_50_90"][:, 1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. The source neuron's lens vector.
    name, Ls, ns, ss = SOURCE[sizes["source"]]
    src, _ = persona.load(hf_cache.model(*MODELS[sizes["source"]]), device)
    w = persona.down_col(src, Ls, ns)
    z = lens(src, (ss * w / w.norm())[None])[0]
    del src, w
    persona.free()

    # 2. Prefilter by Pearson, then exact Spearman.
    model, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
    n_layers = model.config.num_hidden_layers
    layers = sizes.get("layers", list(range(n_layers)))
    cands = []  # (pearson, layer, neuron, sign)
    for L in layers:
        W = persona.blocks(model)[L].mlp.down_proj.weight.float()
        Wn = (W / W.norm(dim=0, keepdim=True)).T
        for s in range(0, Wn.shape[0], sizes["chunk"]):
            r = corr(lens(model, Wn[s:s + sizes["chunk"]]), z).cpu()
            for sign in (1, -1):
                top = (sign * r).topk(min(sizes["prefilter"], len(r)))
                cands += [(float(v), L, s + int(i), sign) for v, i in zip(top.values, top.indices)]
        cands = sorted(cands, reverse=True)[:sizes["prefilter"]]
        del W, Wn
    rz = ranks(z[None])[0]
    spear = []
    for pr, L, n, sign in cands:
        w = persona.down_col(model, L, n)
        zz = lens(model, (sign * w / w.norm())[None])
        spear.append((float(corr(ranks(zz), rz)[0]), pr, L, n, sign))
    spear.sort(reverse=True)
    twins = [{"kind": "twin", "layer": L, "neuron": n, "sign": sign, "spearman": round(sp, 4), "pearson": round(pr, 4)}
             for sp, pr, L, n, sign in spear[:sizes["twins"]]]
    rng = random.Random(0)
    n_neurons = persona.blocks(model)[0].mlp.down_proj.weight.shape[1]
    controls = [{"kind": "random", "layer": t["layer"], "neuron": rng.randrange(n_neurons), "sign": rng.choice([1, -1])}
                for t in twins]
    neurons = twins + controls
    for nr in neurons:
        nr["name"] = f"{nr['kind']}:L{nr['layer']}N{nr['neuron']}{'+' if nr['sign'] > 0 else '-'}"
    print(f"[x2] twins {[(t['name'], t['spearman']) for t in twins]}", flush=True)

    # 3. Steer and judge.
    qs = P.IDENTITY_TEST[:sizes["test_q"]]
    jobs = [(i, d, q) for i in range(len(neurons)) for d in sizes["doses"] for q in qs]
    hooks, texts = persona.Hooks(model).__enter__(), []
    d_model = model.config.hidden_size
    for s in range(0, len(jobs), sizes["batch"]):
        chunk = jobs[s:s + sizes["batch"]]
        hooks.clear()
        vecs = {}
        for b, (i, d, _) in enumerate(chunk):
            nr = neurons[i]
            w = persona.down_col(model, nr["layer"], nr["neuron"])
            vecs.setdefault(nr["layer"], torch.zeros(len(chunk), d_model, device=device))[b] = \
                nr["sign"] * d * float(m[nr["layer"]]) * w / w.norm()
        for L, v in vecs.items():
            hooks.add(L, v)
        outs = persona.generate(model, tok, [persona.chat_ids(tok, q) for _, _, q in chunk], sizes["max_new"])
        texts += [{"kind": "x2", "neuron": neurons[i]["name"], "dose": d, "question": q, "concept": "an alien",
                   "reply": tok.decode(o, skip_special_tokens=True)} for (i, d, q), o in zip(chunk, outs)]
    hooks.clear()
    hooks.__exit__()
    del hooks, model
    persona.free()
    judge, jtok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]), device)
    labels = score(judge, jtok, texts, ["concept", "identity", "coherent"], sizes["judge_batch"])
    del judge
    persona.free()

    # 4. Best coherent CI per neuron; twins against random.
    for nr in neurons:
        nr["doses"] = {}
        for d in sizes["doses"]:
            rows = [(t, lab) for t, lab in zip(texts, labels) if t["neuron"] == nr["name"] and t["dose"] == d]
            n = len(rows)
            nr["doses"][str(d)] = {"ci": round(sum(lab["concept"] for _, lab in rows) / n, 4),
                                   "ib": round(sum(1 - lab["identity"] for _, lab in rows) / n, 4),
                                   "cm": round(sum(P.mentions(t["reply"], "aliens") for t, _ in rows) / n, 4),
                                   "coherent": round(sum(lab["coherent"] for _, lab in rows) / n, 4),
                                   "bits": [lab["concept"] for _, lab in rows],
                                   "coh_bits": [lab["coherent"] for _, lab in rows]}

    def best(nr, ix):
        """(CI, dose) at the neuron's best dose with coherence >= 0.8 on prompts ix; (0, None) if none is coherent."""
        ok = [(sum(v["bits"][i] for i in ix) / len(ix), d) for d, v in nr["doses"].items()
              if sum(v["coh_bits"][i] for i in ix) / len(ix) >= 0.8]
        return max(ok) if ok else (0.0, None)

    def diff(ix):
        """Twins' mean best-coherent CI minus the random neurons', eligibility and dose chosen on prompts ix."""
        group = {k: [best(nr, ix)[0] for nr in neurons if nr["kind"] == k] for k in ("twin", "random")}
        return sum(group["twin"]) / len(group["twin"]) - sum(group["random"]) / len(group["random"])

    every = list(range(len(qs)))
    for nr in neurons:
        nr["best_ci"], nr["best_dose"] = best(nr, every)
    # Each replicate resamples prompts and re-chooses every neuron's dose and eligibility on them.
    brng, diffs = random.Random(0), []
    for _ in range(sizes["boot"]):
        diffs.append(diff([brng.randrange(len(qs)) for _ in qs]))
    diffs.sort()
    mean = {k: sum(nr["best_ci"] for nr in neurons if nr["kind"] == k) / sum(nr["kind"] == k for nr in neurons)
            for k in ("twin", "random")}
    test = {"twin_ci": round(mean["twin"], 4), "random_ci": round(mean["random"], 4), "diff": round(diff(every), 4),
            "diff_ci95": [round(diffs[int(0.025 * len(diffs))], 4), round(diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))], 4)]}
    test["twins_beat_random"] = test["diff_ci95"][0] > 0

    odir = out_dir(tag)
    with open(odir / "texts.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    with open(odir / "labels.jsonl", "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    with open(odir / "x2.json", "w") as f:
        json.dump({"sizes": sizes, "source": SOURCE[sizes["source"]], "neurons": neurons, "test": test,
                   "spearman_top": spear[:20]}, f)
    if not modal.is_local():
        OUT.commit()
    res = {"model": model_key, **{k: v for k, v in test.items() if k != "diff_ci95"},
           "minutes": round((time.time() - t0) / 60, 1)}
    details = {"test": test, "neurons": [{k: v for k, v in nr.items() if k != "doses"} |
                                         {"doses": {d: {k: v for k, v in x.items() if k not in ("bits", "coh_bits")}
                                                    for d, x in nr["doses"].items()}} for nr in neurons],
               "judge": summary(labels, ["concept", "identity", "coherent"]), "loads": hf_cache.loads}
    return {**res, "details": details}


@app.local_entrypoint()
def main(model: str = "2b", smoke: bool = False, s1a: str = "", tag: str = ""):
    if smoke:
        model = "0.8b"
    sfx = "-smoke" if smoke else ""
    res = call(sized(run, model), model, s1a or f"s1a-{model}{sfx}", tag or f"x2-twins-{model}{sfx}", SMOKE if smoke else FULL)
    details = res.pop("details")
    print("CRUCIBLE_RESULT", json.dumps({**res, "smoke": smoke}))
    print("CRUCIBLE_RESULT", json.dumps({"details": details}))
