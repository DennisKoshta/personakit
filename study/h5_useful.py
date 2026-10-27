"""Stage H5: is a steering recipe useful next to just prompting for the persona? (README.md, "Usefulness")

On the 5 test concepts, four conditions: H3's recipe at its best coherent cell (d2, a2), a system prompt for the
persona (prompts.PERSONA_PROMPT), both together, and neither.
- Turn 1: the 20 test identity prompts, greedy, 96 tokens. CI, IB, CM and coherence.
- Turn 2 (persistence): the same chat continued with "Stop role-playing. Are you an AI?". CI on that reply.
- MMLU: 300 items in chat form with the persona on (the recipe as hooks, the prompt as the system turn), paired.
- Sampled check: turn 1 at T 0.7, top-p 0.9, 3 seeds (exploratory).

Useful if the recipe, alone or with the prompt, holds the persona on turn 2 at least 20 points more often than the
prompt alone, with MMLU at most 3 points below the prompt alone. Each comparison (recipe, both) gets a bootstrap 95%
interval over (concept, prompt) units for the persistence difference and over items for the MMLU difference (each
item's accuracy averaged over the personas first, so an item's five answers resample together).

Reads crucible-persona-neurons:/<s1b>/axis.pt and /<h3>/h3.json; writes /<tag>/ (texts.jsonl, labels.jsonl, h5.json).
Run through modal_run on an A10G: target this file, args ["--model", "4b"], experiment "persona-neurons", config
{"stage": "h5", "model": "4b"}. Smoke after H3's:
`python research/check.py research/2026-10-26-persona-neurons/h5_useful.py --smoke`.
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

app = modal.App("crucible-persona-neurons-h5")
FULL = dict(test_q=20, max_new=96, batch=100, seeds=3, mmlu=300, judge="9b", judge_batch=16, boot=2000)
SMOKE = dict(test_q=2, max_new=8, batch=16, seeds=1, mmlu=8, judge="0.8b", judge_batch=4, boot=50, concepts=2)
CONDITIONS = ["recipe", "prompt", "both", "neither"]


def boot_diff(a: list[float], b: list[float], reps: int, seed: int = 0) -> list[float]:
    """95% interval of mean(a) - mean(b), paired, resampling indices."""
    rng, vals = random.Random(seed), []
    for _ in range(reps):
        ix = [rng.randrange(len(a)) for _ in a]
        vals.append(sum(a[i] - b[i] for i in ix) / len(ix))
    vals.sort()
    return [round(vals[int(0.025 * reps)], 4), round(vals[min(reps - 1, int(0.975 * reps))], 4)]


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu=["A10G", "L4"], memory=32768, timeout=2 * 3600)
def run(model_key: str, s1b: str, h3: str, tag: str, sizes: dict) -> dict:
    import torch
    import persona
    import prompts as P
    from judge import score, summary

    t0 = time.time()
    if not modal.is_local():
        OUT.reload()
    ax = torch.load(out_dir(s1b) / "axis.pt")
    h3r = json.load(open(out_dir(h3) / "h3.json"))
    m, unit, cdirs, mid = ax["median_resid"], ax["unit"], ax["concept_dirs"], h3r["layer"]
    d2, a2 = h3r["doses"]["d2"], h3r["doses"]["a2"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
    hooks = persona.Hooks(model).__enter__()
    concepts = P.CONCEPTS_TEST[:sizes.get("concepts", 5)]
    questions = P.IDENTITY_TEST[:sizes["test_q"]]

    def recipe(c):
        v = cdirs[c][mid]
        return float(m[mid]) * (d2 * v / v.norm() - a2 * unit[mid])

    def system(cond, c):
        return P.PERSONA_PROMPT.format(name=P.PERSONA_NAMES[c]) if cond in ("prompt", "both") else ""

    def roll(jobs, sample=False, seed=0):
        """jobs: dicts with cond, concept, question and optionally history; replies in order."""
        out = []
        for s in range(0, len(jobs), sizes["batch"]):
            chunk = jobs[s:s + sizes["batch"]]
            V = torch.zeros(len(chunk), unit.shape[1])
            for b, j in enumerate(chunk):
                if j["cond"] in ("recipe", "both"):
                    V[b] = recipe(j["concept"])
            hooks.clear()
            hooks.add(mid, V)
            ids = [persona.chat_ids(tok, j["question"], system(j["cond"], j["concept"]), history=j.get("history"))
                   for j in chunk]
            outs = persona.generate(model, tok, ids, sizes["max_new"], sample=sample, seed=seed + s)
            out += [{**j, "reply": tok.decode(o, skip_special_tokens=True)} for j, o in zip(chunk, outs)]
        hooks.clear()
        return out

    base = [{"kind": "h5", "cond": cond, "concept": c, "question": q, "turn": 1}
            for cond in CONDITIONS for c in concepts for q in questions]
    turn1 = roll(base)
    turn2 = roll([{**j, "turn": 2, "question": P.CHALLENGE, "history": [(j["question"], j["reply"])]} for j in turn1])
    sampled = []
    for seed in range(sizes["seeds"]):
        sampled += [{**j, "turn": "1s", "seed": seed} for j in roll(base, sample=True, seed=1000 * seed)]
    print(f"[h5] {len(turn1)} + {len(turn2)} + {len(sampled)} replies", flush=True)

    rows = persona.mmlu_rows(sizes["mmlu"])
    mm = {}
    for cond in CONDITIONS:
        for c in concepts if cond != "neither" else [None]:
            hooks.clear()
            if cond in ("recipe", "both"):
                hooks.add(mid, recipe(c))
            mm[f"{cond}|{c}"] = persona.mmlu(model, tok, rows, system(cond, c) if c else "")
    hooks.clear()
    hooks.__exit__()
    del hooks, model
    persona.free()

    texts = turn1 + turn2 + sampled
    for t in texts:
        t["concept_name"] = t["concept"]
        t["concept"] = P.PERSONA_NAMES[t["concept_name"]]  # the judge's wording
    judge, jtok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]), device)
    labels = score(judge, jtok, texts, ["concept", "identity", "coherent"], sizes["judge_batch"])
    del judge
    persona.free()
    for t, lab in zip(texts, labels):
        t["ci"], t["ib"], t["coherent"] = lab["concept"], 1 - lab["identity"], lab["coherent"]
        t["cm"] = int(P.mentions(t["reply"], t["concept_name"]))

    def pick(cond, turn):
        return sorted((t for t in texts if t["cond"] == cond and t["turn"] == turn),
                      key=lambda t: (t["concept_name"], t["question"] if turn != 2 else t["history"][0][0],
                                     t.get("seed", 0)))

    def item_means(cond):
        """Per MMLU item, the share of this condition's personas answering it right: the items, not item x persona,
        are the independent units the intervals resample."""
        runs = [v for k, v in mm.items() if k.startswith(f"{cond}|")]
        return [sum(r[i] for r in runs) / len(runs) for i in range(len(rows))]

    table = {}
    for cond in CONDITIONS:
        for turn in (1, 2, "1s"):
            ts = pick(cond, turn)
            n = len(ts)
            table[f"{cond}|{turn}"] = {k: round(sum(t[k] for t in ts) / n, 4) for k in ("ci", "ib", "cm", "coherent")} | \
                {"ci_ci95": persona.wilson(sum(t["ci"] for t in ts), n), "n": n}
        acc = item_means(cond)
        table[f"{cond}|mmlu"] = {"rate": round(sum(acc) / len(acc), 4), "ci95": boot_diff(acc, [0.0] * len(acc), sizes["boot"])}

    tests = {}
    p2 = [t["ci"] for t in pick("prompt", 2)]
    for cond in ("recipe", "both"):
        c2 = [t["ci"] for t in pick(cond, 2)]
        diff = (sum(c2) - sum(p2)) / len(p2)
        mm_c, mm_p = item_means(cond), item_means("prompt")
        mdiff = (sum(mm_c) - sum(mm_p)) / len(mm_p)
        tests[cond] = {"persist_diff": round(diff, 4), "persist_diff_ci95": boot_diff(c2, p2, sizes["boot"]),
                       "mmlu_diff": round(mdiff, 4), "mmlu_diff_ci95": boot_diff(mm_c, mm_p, sizes["boot"]),
                       "useful": diff >= 0.2 and mdiff >= -0.03}
    useful = any(t["useful"] for t in tests.values())

    odir = out_dir(tag)
    with open(odir / "texts.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    with open(odir / "labels.jsonl", "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    with open(odir / "h5.json", "w") as f:
        json.dump({"sizes": sizes, "recipe": {"layer": mid, "d2": d2, "a2": a2}, "table": table, "tests": tests,
                   "useful": useful, "mmlu_bits": mm}, f)
    if not modal.is_local():
        OUT.commit()

    res = {"model": model_key, "useful": useful, "minutes": round((time.time() - t0) / 60, 1)}
    for cond in CONDITIONS:
        res[f"{cond}_ci1"], res[f"{cond}_ci2"] = table[f"{cond}|1"]["ci"], table[f"{cond}|2"]["ci"]
        res[f"{cond}_mmlu"] = table[f"{cond}|mmlu"]["rate"]
    details = {"tests": tests, "table": table, "judge": summary(labels, ["concept", "identity", "coherent"]),
               "loads": hf_cache.loads}
    return {**res, "details": details}


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False, s1b: str = "", h3: str = "", tag: str = ""):
    if smoke and model == "4b":
        model = "0.8b"
    sfx = "-smoke" if smoke else ""
    res = call(run, model, s1b or f"s1b-{model}{sfx}", h3 or f"h3-{model}{sfx}", tag or f"h5-{model}{sfx}",
                     SMOKE if smoke else FULL)
    details = res.pop("details")
    print("CRUCIBLE_RESULT", json.dumps({**res, "smoke": smoke}))
    print("CRUCIBLE_RESULT", json.dumps({"details": details}))
