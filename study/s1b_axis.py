"""Stage S1b: Qwen3.5's Assistant Axis, the concept directions, H1 (weights only) and the S1 gate (README.md).

1. The judge (Qwen3.5-9B) labels S1a's role rollouts with the paper's per-role rubric (0-3), unless already done.
2. Role vectors: each role's mean reply residual over its fully role-playing replies (label 3), for roles with at
   least `min_fully` of them. The Assistant Axis a_L = mean(default replies) - mean(role vectors), every layer (Lu et
   al. 2026). PCA over the centred role vectors gives PC1; the gate wants |cos(a, PC1)| > 0.6 at the middle layer.
3. Concept directions c_X,L: mean over the Assistant's neutral answers about X minus the mean over the other concepts,
   with the a_L component removed.
4. H1: every down column at the known neurons' layers and H4's layers, its cosine with -a_L and with each c_X; the
   known neurons' percentiles. Also each neuron's nearest role directions (role mean - default mean).
5. The capping floors for H2b: the 25th percentile, per layer, of default-Assistant reply tokens' projections on a_L.
6. The gate's steering: near-Assistant roles (the 10 kept roles projecting highest on a at the middle layer), 2 system
   prompts x 5 development identity questions, plus the 10 identity questions with no system prompt, steered along -a
   at the middle layer at doses 0-1.5 (units of that layer's median residual norm), greedy. The judge labels identity
   (IB = not speaking as an AI) and coherence. The gate passes if some dose with coherence >= 0.8 raises IB by >= 0.2.

Reads crucible-persona-neurons:/<src>/ and writes /<tag>/ (axis.pt with everything above, texts.jsonl and
labels.jsonl for the gate's replies). Run through modal_run on an L4: target this file, args ["--model", "4b"],
experiment "persona-neurons", config {"stage": "s1b", "model": "4b"}. Smoke after S1a's:
`python research/check.py research/2026-10-26-persona-neurons/s1b_axis.py --smoke`.
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
from jobs import MODELS, OUT, call, fetch, image, out_dir, sized  # noqa: E402

app = modal.App("crucible-persona-neurons-s1b")
FULL = dict(min_fully=10, near_roles=10, systems=2, identity_q=5, doses=[0, 0.25, 0.5, 0.75, 1.0, 1.5], max_new=96,
            batch=64, cap_replies=100, judge="9b", judge_batch=16)
SMOKE = dict(min_fully=1, near_roles=2, systems=1, identity_q=2, doses=[0, 1.0], max_new=12, batch=4, cap_replies=4,
             judge="0.8b", judge_batch=4, allow_fallback=True)
H4_LAYERS = [10, 14, 18, 22]


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu=["L4", "A10G"], memory=32768, timeout=2 * 3600)
def run(model_key: str, src: str, tag: str, sizes: dict) -> dict:
    import torch
    import persona
    import prompts as P
    from judge import score, summary

    t0 = time.time()
    if not modal.is_local():
        OUT.reload()
    sdir, odir = out_dir(src), out_dir(tag)
    recs = [json.loads(line) for line in open(sdir / "texts.jsonl")]
    acts = torch.load(sdir / "acts.pt")  # [N, layers, d] fp16
    calib = torch.load(sdir / "calib.pt")
    m = calib["pile"]["norm_q10_50_90"][:, 1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_layers, mid = acts.shape[1], acts.shape[1] // 2

    # 1. Role labels: reused only if they cover every record and score every role reply.
    path = sdir / "labels.jsonl"
    role_labels = [json.loads(line) for line in open(path)] if path.exists() else []
    if len(role_labels) != len(recs) or any(r["kind"] == "role" and "role" not in lab for r, lab in zip(recs, role_labels)):
        judge, jtok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]), device)
        fresh = score(judge, jtok, recs, ["role"], sizes["judge_batch"])
        del judge
        persona.free()
        role_labels = [{**o, **n} for o, n in zip(role_labels, fresh)] if len(role_labels) == len(recs) else fresh
        with open(path, "w") as f:
            for lab in role_labels:
                f.write(json.dumps(lab) + "\n")
    role_dist = {k: sum(1 for lab in role_labels if lab.get("role") == k) for k in range(4)}
    print(f"[s1b] role labels {role_dist}", flush=True)

    # 2. Role vectors, axis, PCA.
    def mean_of(idx):
        return acts[idx].float().mean(0)  # [layers, d]

    by_role = {}
    for i, (r, lab) in enumerate(zip(recs, role_labels)):
        if r["kind"] == "role" and lab.get("role") == 3:
            by_role.setdefault(r["role"], []).append(i)
    kept = {k: v for k, v in by_role.items() if len(v) >= sizes["min_fully"]}
    fallback = not kept
    if fallback and not sizes.get("allow_fallback"):
        raise RuntimeError(f"no role has {sizes['min_fully']} fully role-playing replies (labels {role_dist}): "
                           "the axis can't be built from this data, so S1 stops here")
    if fallback:  # smoke only: too few rollouts for any role to reach min_fully
        kept = {}
        for i, r in enumerate(recs):
            if r["kind"] == "role":
                kept.setdefault(r["role"], []).append(i)
    roles = sorted(kept)
    role_vecs = torch.stack([mean_of(kept[k]) for k in roles])  # [R, layers, d]
    default = mean_of([i for i, r in enumerate(recs) if r["kind"] == "default"])
    axis = default - role_vecs.mean(0)  # [layers, d]
    unit = axis / axis.norm(dim=-1, keepdim=True)
    cos_pc1 = []
    for L in range(n_layers):
        X = role_vecs[:, L] - role_vecs[:, L].mean(0)
        pc1 = torch.linalg.svd(X, full_matrices=False).Vh[0]
        cos_pc1.append(float((pc1 @ unit[L]).abs()))
    role_proj_mid = {k: float((role_vecs[j, mid] - default[mid]) @ unit[mid]) for j, k in enumerate(roles)}
    print(f"[s1b] {len(roles)} roles kept (fallback {fallback}); |cos(axis, PC1)| at L{mid} {cos_pc1[mid]:.3f}", flush=True)

    # 3. Concept directions (axis component removed).
    concepts = P.CONCEPTS_DEV + P.CONCEPTS_TEST
    cmeans = {c: mean_of([i for i, r in enumerate(recs) if r.get("concept") == c]) for c in concepts}
    cdirs = {}
    for c in concepts:
        others = torch.stack([cmeans[o] for o in concepts if o != c]).mean(0)
        v = cmeans[c] - others
        v = v - (v * unit).sum(-1, keepdim=True) * unit
        cdirs[c] = v
    role_dirs = role_vecs - default  # [R, layers, d]
    del acts
    persona.free()

    # 4. H1 on the weights; 5. capping floors; 6. the gate's steering.
    model, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
    hooks = persona.Hooks(model).__enter__()
    known = P.KNOWN if model_key == "4b" else P.SMOKE_KNOWN  # exercises H1 on other sizes
    h1, h1_layers = {}, sorted({L for L, _, _ in known.values()} | {L for L in H4_LAYERS if L < n_layers})
    neuron_cos = {}
    for L in h1_layers:
        W = persona.blocks(model)[L].mlp.down_proj.weight.float()  # [d, N]
        Wn = W / W.norm(dim=0, keepdim=True)
        away = -(unit[L].to(W.device) @ Wn).cpu()  # cos with -a_L, per neuron
        cc = {c: ((cdirs[c][L] / cdirs[c][L].norm()).to(W.device) @ Wn).cpu() for c in concepts}
        neuron_cos[L] = {"away": away.half(), **{f"c:{c}": v.half() for c, v in cc.items()}}
        for name, (Lk, nk, sign) in known.items():
            if Lk != L:
                continue
            rd = role_dirs[:, L] / role_dirs[:, L].norm(dim=-1, keepdim=True)
            rcos = (rd.to(W.device) @ (sign * Wn[:, nk])).cpu()
            top = sorted(zip(rcos.tolist(), roles), reverse=True)[:5]
            pct = lambda vals, x: float((vals < x).float().mean())  # noqa: E731 (percentile among the layer's neurons)
            h1[name] = {"layer": L, "neuron": nk, "sign": sign, "cos_away": round(float(sign * away[nk]), 4),
                        "pct_away": round(pct(sign * away, float(sign * away[nk])), 4),
                        "concepts": {c: {"cos": round(float(sign * v[nk]), 4), "pct": round(pct(sign * v, float(sign * v[nk])), 4)}
                                     for c, v in cc.items()},
                        "top_roles": [[k, round(x, 3)] for x, k in top]}
        del W, Wn

    default_idx = [i for i, r in enumerate(recs) if r["kind"] == "default"][:sizes["cap_replies"]]
    proj_tokens = [[] for _ in range(n_layers)]
    for s in range(0, len(default_idx), 16):
        idx = default_idx[s:s + 16]
        seqs = [persona.chat_ids(tok, recs[i]["question"], persona.DEFAULT_SYSTEMS[recs[i]["system"]]) for i in idx]
        replies = [tok.encode(recs[i]["reply"], add_special_tokens=False) for i in idx]
        ids, mask = persona.pad_left([a + b for a, b in zip(seqs, replies)], 0, device)
        T = ids.shape[1]
        hooks.capture = {}
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
        caps, hooks.capture = hooks.capture, None
        for b, (a, rep) in enumerate(zip(seqs, replies)):
            lo = T - len(rep)
            for L in range(n_layers):
                proj_tokens[L].append((caps[L][0][b, lo:].float() @ unit[L].to(device)).cpu())
    floors = torch.tensor([float(torch.cat(p).quantile(0.25)) for p in proj_tokens])

    near = sorted(roles, key=lambda k: -role_proj_mid[k])[:sizes["near_roles"]]
    gate_prompts = []
    for role in near:
        systems = [x["pos"] for x in fetch(f"roles/instructions/{role}.json")["instruction"]][:sizes["systems"]]
        for si, sp in enumerate(systems):
            for q in P.IDENTITY_DEV[:sizes["identity_q"]]:
                gate_prompts.append({"role": role, "system": si, "question": q, "ids": persona.chat_ids(tok, q, sp)})
    for q in P.IDENTITY_DEV[:sizes["identity_q"] * 2]:
        gate_prompts.append({"role": None, "system": None, "question": q, "ids": persona.chat_ids(tok, q)})
    texts = []
    for dose in sizes["doses"]:
        hooks.clear()
        if dose:
            hooks.add(mid, -dose * float(m[mid]) * unit[mid])
        for s in range(0, len(gate_prompts), sizes["batch"]):
            chunk = gate_prompts[s:s + sizes["batch"]]
            outs = persona.generate(model, tok, [g["ids"] for g in chunk], sizes["max_new"])
            for g, o in zip(chunk, outs):
                texts.append({"kind": "gate", "dose": dose, "role": g["role"], "system": g["system"],
                              "question": g["question"], "reply": tok.decode(o, skip_special_tokens=True)})
        print(f"[s1b] gate dose {dose}: {len(gate_prompts)} replies", flush=True)
    hooks.clear()
    hooks.__exit__()
    del hooks, model
    persona.free()

    judge, jtok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]), device)
    labels = score(judge, jtok, texts, ["identity", "coherent"], sizes["judge_batch"])
    del judge
    persona.free()

    gate = {}
    for dose in sizes["doses"]:
        rows = [(t, lab) for t, lab in zip(texts, labels) if t["dose"] == dose]
        role_rows = [(t, lab) for t, lab in rows if t["role"]]
        ib = sum(1 - lab["identity"] for _, lab in role_rows)
        coh = sum(lab["coherent"] for _, lab in rows)
        bare = [(t, lab) for t, lab in rows if not t["role"]]
        gate[str(dose)] = {"ib": round(ib / max(1, len(role_rows)), 4), "ib_ci": persona.wilson(ib, len(role_rows)),
                           "coherent": round(coh / max(1, len(rows)), 4),
                           "ib_no_system": round(sum(1 - lab["identity"] for _, lab in bare) / max(1, len(bare)), 4)}
    base_ib = gate[str(sizes["doses"][0])]["ib"]
    passing = [d for d, g in gate.items() if g["coherent"] >= 0.8 and g["ib"] - base_ib >= 0.2]
    gate_pass = bool(passing) and cos_pc1[mid] > 0.6

    torch.save({"axis": axis, "unit": unit, "default": default, "roles": roles, "role_vecs": role_vecs.half(),
                "concept_dirs": cdirs, "floors_p25": floors, "median_resid": m, "cos_pc1": cos_pc1,
                "neuron_cos": neuron_cos, "h1": h1, "role_proj_mid": role_proj_mid}, odir / "axis.pt")
    with open(odir / "texts.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    with open(odir / "labels.jsonl", "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    if not modal.is_local():
        OUT.commit()

    res = {"model": model_key, "roles_kept": len(roles), "role_fallback": fallback, "role_label_3": role_dist[3], "cos_axis_pc1_mid": round(cos_pc1[mid], 4),
           "cos_axis_pc1_max": round(max(cos_pc1), 4), "gate_pass": gate_pass, "gate_passing_doses": passing,
           "minutes": round((time.time() - t0) / 60, 1), **{f"ib_dose_{d}": g["ib"] for d, g in gate.items()},
           **{f"coherent_dose_{d}": g["coherent"] for d, g in gate.items()}}
    for name, v in h1.items():
        res[f"{name}_pct_away"] = v["pct_away"]
    details = {"gate": gate, "h1": h1, "cos_pc1": [round(x, 4) for x in cos_pc1], "near_roles": near,
               "role_dist": role_dist, "floors_p25": [round(float(x), 3) for x in floors],
               "judge": summary(labels, ["identity", "coherent"]), "loads": hf_cache.loads}
    return {**res, "details": details}


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False, src: str = "", tag: str = ""):
    if smoke and model == "4b":
        model = "0.8b"
    suffix = "-smoke" if smoke else ""
    res = call(sized(run, model), model, src or f"s1a-{model}{suffix}", tag or f"s1b-{model}{suffix}", SMOKE if smoke else FULL)
    details = res.pop("details")
    print("CRUCIBLE_RESULT", json.dumps({**res, "smoke": smoke}))
    print("CRUCIBLE_RESULT", json.dumps({"details": details}))
