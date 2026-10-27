"""Stage X1: the persona machinery in the 4B before and after chat tuning (README.md, "X1").

1. Weights: each W_down column's cosine between Base and Instruct, per layer, and the known neurons'.
2. The Base model's own Assistant Axis, from the Instruct rollouts (as the paper did for base models): Base's mean
   residual over up to 20 fully role-playing replies per kept role and over the default replies, the axis per layer,
   its cosine with Instruct's axis and |cos(axis, PC1)|.
3. Steering N2742+ in Base and in Instruct, on the 20 test identity prompts in chat form, greedy, 96 tokens, at the
   alien neuron's H2 dose (in each model's own dose unit; Base's from 200 Pile passages), and unsteered. The judge
   labels concept ("an alien") and coherence; CM counts the aliens family. Prediction (after CNA, Herring et al. 2026):
   in Base, CM rises and CI barely moves.

Reads crucible-persona-neurons:/<s1a>/ (texts, calib), /<s1b>/ (axis.pt) and /<h4>/h4.json; writes /<tag>/
(texts.jsonl, labels.jsonl, x1.json). Run through modal_run on an A10G: target this file, args ["--model", "4b"],
experiment "persona-neurons", config {"stage": "x1"}. Smoke after H4's:
`python research/check.py research/2026-10-26-persona-neurons/x1_base.py --smoke` (the 0.8B stands in for both).
"""
import json
import pathlib
import sys
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import hf_cache  # noqa: E402
from jobs import MODELS, OUT, call, image, out_dir, pick_dose  # noqa: E402

app = modal.App("crucible-persona-neurons-x1")
PILE = ("NeelNanda/pile-10k", "127bfedcd5047750df5ccf3a12979a47bfa0bafa")
FULL = dict(base="4b-base", per_role=20, pile_docs=200, pile_len=256, test_q=20, max_new=96, batch=16, judge="9b",
            judge_batch=16)
SMOKE = dict(base="0.8b", per_role=2, pile_docs=4, pile_len=32, test_q=2, max_new=8, batch=4, judge="0.8b",
             judge_batch=4)


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu=["A10G", "L4"], memory=32768, timeout=2 * 3600)
def run(model_key: str, s1a: str, s1b: str, h4: str, tag: str, sizes: dict) -> dict:
    import torch
    import persona
    import prompts as P
    from judge import score, summary

    t0 = time.time()
    if not modal.is_local():
        OUT.reload()
    recs = [json.loads(line) for line in open(out_dir(s1a) / "texts.jsonl")]
    role_labels = [json.loads(line) for line in open(out_dir(s1a) / "labels.jsonl")]
    ax = torch.load(out_dir(s1b) / "axis.pt")
    screen = json.load(open(out_dir(h4) / "h4.json"))
    m_inst = torch.load(out_dir(s1a) / "calib.pt")["pile"]["norm_q10_50_90"][:, 1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    concepts = P.NEURON_CONCEPTS if model_key == "4b" else P.SMOKE_CONCEPTS
    name = next(iter(concepts))
    nr = next(x for x in screen["neurons"] if x["name"] == name)
    dose, _ = pick_dose(nr["sweep"], "ci")
    L, n, sign = nr["layer"], nr["neuron"], nr["sign"]

    inst, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
    W_inst = {i: b.mlp.down_proj.weight.detach().float().cpu() for i, b in enumerate(persona.blocks(inst))}
    w_inst = W_inst[L][:, n]

    # 3a. Instruct: steered and unsteered, for the matched comparison.
    qs = P.IDENTITY_TEST[:sizes["test_q"]]
    ids = [persona.chat_ids(tok, q) for q in qs]
    texts = []

    def roll(model, which, delta_vec):
        hooks = persona.Hooks(model).__enter__()
        for cond, vec in (("none", None), ("steer", delta_vec)):
            hooks.clear()
            if vec is not None:
                hooks.add(L, vec)
            for s in range(0, len(ids), sizes["batch"]):
                outs = persona.generate(model, tok, ids[s:s + sizes["batch"]], sizes["max_new"])
                texts.extend({"kind": "x1", "model": which, "cond": cond, "question": q, "concept": concepts[name][0],
                              "reply": tok.decode(o, skip_special_tokens=True)} for q, o in zip(qs[s:s + sizes["batch"]], outs))
        hooks.__exit__()

    roll(inst, "instruct", sign * dose * float(m_inst[L]) * w_inst / w_inst.norm())
    del inst
    persona.free()

    # 1-2. Base: weights, its own dose unit, its axis from the Instruct rollouts.
    base, _ = persona.load(hf_cache.model(*MODELS[sizes["base"]]), device)
    n_layers = base.config.num_hidden_layers
    col_cos = {}
    for i, b in enumerate(persona.blocks(base)):
        Wb = b.mlp.down_proj.weight.detach().float().cpu()
        c = (Wb * W_inst[i]).sum(0) / (Wb.norm(dim=0) * W_inst[i].norm(dim=0))
        col_cos[i] = {"median": round(float(c.median()), 5), "p01": round(float(c.quantile(0.01)), 5),
                      "min": round(float(c.min()), 5)}
        if i == L:
            col_cos[i]["neuron"] = round(float(c[n]), 5)
    known_cos = {k: None for k in (P.KNOWN if model_key == "4b" else P.SMOKE_KNOWN)}
    for k, (Lk, nk, _) in (P.KNOWN if model_key == "4b" else P.SMOKE_KNOWN).items():
        wb = persona.down_col(base, Lk, nk).cpu()
        known_cos[k] = round(float(wb @ W_inst[Lk][:, nk] / (wb.norm() * W_inst[Lk][:, nk].norm())), 5)
    del W_inst
    hooks = persona.Hooks(base).__enter__()
    pile = hf_cache.dataset(*PILE, split="train")
    docs = [tok.encode(pile[i]["text"], add_special_tokens=False)[:sizes["pile_len"]] for i in range(sizes["pile_docs"])]
    stats = persona.Stats()
    for s in range(0, len(docs), 8):
        persona.collect_stats(base, hooks, stats, [d for d in docs[s:s + 8] if len(d) >= 16] or docs[s:s + 1])
    m_base = stats.result()["norm_q10_50_90"][:, 1]

    by_role = {}
    for i, (r, lab) in enumerate(zip(recs, role_labels)):
        if r["kind"] == "role" and lab.get("role") == 3 and r["role"] in ax["roles"]:
            by_role.setdefault(r["role"], []).append(i)
    if not by_role:  # S1b's smoke fallback: no reply was graded 3, so its axis used every role reply
        for i, r in enumerate(recs):
            if r["kind"] == "role" and r["role"] in ax["roles"]:
                by_role.setdefault(r["role"], []).append(i)
    picked = {k: v[:sizes["per_role"]] for k, v in by_role.items()}
    default_idx = [i for i, r in enumerate(recs) if r["kind"] == "default"]

    def means(idx):
        out = []
        for s in range(0, len(idx), 16):
            chunk = idx[s:s + 16]
            prompts_ = [tok.encode(recs[i]["prompt"], add_special_tokens=False) for i in chunk]  # as S1a rendered them
            seqs = [p + tok.encode(recs[i]["reply"], add_special_tokens=False) for p, i in zip(prompts_, chunk)]
            starts = [len(p) for p in prompts_]
            out.append(persona.span_means(base, hooks, seqs, starts, list(range(n_layers))))
        return torch.cat(out)

    role_vecs = torch.stack([means(picked[k]).mean(0) for k in sorted(picked)])
    default = means(default_idx).mean(0)
    axis_b = default - role_vecs.mean(0)
    unit_b = axis_b / axis_b.norm(dim=-1, keepdim=True)
    cos_axes = [round(float(unit_b[i] @ ax["unit"][i]), 4) for i in range(n_layers)]
    cos_pc1 = []
    for i in range(n_layers):
        X = role_vecs[:, i] - role_vecs[:, i].mean(0)
        cos_pc1.append(round(float((torch.linalg.svd(X, full_matrices=False).Vh[0] @ unit_b[i]).abs()), 4))
    hooks.__exit__()
    del hooks

    # 3b. Base: steered and unsteered.
    wb = persona.down_col(base, L, n).cpu()
    roll(base, "base", sign * dose * float(m_base[L]) * wb / wb.norm())
    del base
    persona.free()

    judge, jtok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]), device)
    labels = score(judge, jtok, texts, ["concept", "coherent"], sizes["judge_batch"])
    del judge
    persona.free()
    table = {}
    for which in ("instruct", "base"):
        for cond in ("none", "steer"):
            rows = [(t, lab) for t, lab in zip(texts, labels) if t["model"] == which and t["cond"] == cond]
            k = len(rows)
            ci = sum(lab["concept"] for _, lab in rows)
            cm = sum(P.mentions(t["reply"], concepts[name][1]) for t, _ in rows)
            table[f"{which}|{cond}"] = {"ci": round(ci / k, 4), "ci_ci95": persona.wilson(ci, k), "cm": round(cm / k, 4),
                                        "cm_ci95": persona.wilson(cm, k),
                                        "coherent": round(sum(lab["coherent"] for _, lab in rows) / k, 4)}
    mid = n_layers // 2
    res = {"model": model_key, "base": sizes["base"], "dose": dose, "neuron_col_cos": col_cos[L].get("neuron"),
           "median_col_cos_min_layer": min(v["median"] for v in col_cos.values()),
           "cos_axes_mid": cos_axes[mid], "base_cos_pc1_mid": cos_pc1[mid],
           **{f"{k.replace('|', '_')}_{x}": v[x] for k, v in table.items() for x in ("ci", "cm")},
           "minutes": round((time.time() - t0) / 60, 1)}
    odir = out_dir(tag)
    with open(odir / "texts.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    with open(odir / "labels.jsonl", "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    details = {"col_cos": col_cos, "known_cos": known_cos, "cos_axes": cos_axes, "base_cos_pc1": cos_pc1,
               "base_median_resid": [round(float(x), 3) for x in m_base], "table": table,
               "roles": len(picked), "judge": summary(labels, ["concept", "coherent"]), "loads": hf_cache.loads}
    with open(odir / "x1.json", "w") as f:
        json.dump({"sizes": sizes, **res, **details}, f)
    torch.save({"axis": axis_b, "unit": unit_b, "default": default, "role_vecs": role_vecs.half()}, odir / "axis_base.pt")
    if not modal.is_local():
        OUT.commit()
    return {**res, "details": details}


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False, s1a: str = "", s1b: str = "", h4: str = "", tag: str = ""):
    if smoke:
        model = "0.8b"
    sfx = "-smoke" if smoke else ""
    res = call(run, model, s1a or f"s1a-{model}{sfx}", s1b or f"s1b-{model}{sfx}", h4 or f"h4-{model}{sfx}",
                     tag or f"x1-{model}{sfx}", SMOKE if smoke else FULL)
    details = res.pop("details")
    print("CRUCIBLE_RESULT", json.dumps({**res, "smoke": smoke}))
    print("CRUCIBLE_RESULT", json.dumps({"details": details}))
