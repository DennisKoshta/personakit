"""Stage S1a: the raw material for Qwen3.5's persona space (README.md, "S1").

- Calibration on the Pile (NeelNanda/pile-10k): each layer's median residual norm leaving it (the dose unit) and each
  neuron's natural extremes.
- Rollouts: role-play replies (the Assistant Axis paper's roles and system prompts, sampled at T 0.7 like its
  generation), default-Assistant replies (its five default system prompts), neutral answers about the ten concepts
  (for the concept directions) and unsteered answers to the identity prompts.
- Each reply's mean residual over its own tokens at every layer, plus the neurons' extremes on chat text.
- Two checks of the steering code: a neuron push equals adding its down column at the layer's output, and a padded
  batch decodes as each prompt does alone.

Everything goes to the Modal volume crucible-persona-neurons under /<tag>/ (texts.jsonl, acts.pt, calib.pt); the
judge (S1 judge job) and the axis (S1b) read it from there. Run through modal_run on an L4: target this file, args
["--model", "4b"], experiment "persona-neurons", config {"stage": "s1a", "model": "4b"}. Smoke:
`python research/check.py research/2026-10-26-persona-neurons/s1_rollouts.py --smoke` (0.8B, toy sizes, CPU).
"""
import json
import pathlib
import random
import sys
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # research/
sys.path.insert(0, str(HERE))
import hf_cache  # noqa: E402
from jobs import MODELS, OUT, call, fetch, image, out_dir, sized  # noqa: E402

PILE = ("NeelNanda/pile-10k", "127bfedcd5047750df5ccf3a12979a47bfa0bafa")
FULL = dict(pile_docs=2000, pile_len=256, roles=80, systems=2, questions=40, default_questions=80, max_new=160,
            identity_new=96, batch=64, stat_batch=32, concept_questions=10, identity=30)
SMOKE = dict(pile_docs=8, pile_len=64, roles=3, systems=1, questions=2, default_questions=2, max_new=16,
             identity_new=12, batch=4, stat_batch=4, concept_questions=2, identity=3)
FORCED_ROLES = ["alien", "angel", "demon", "ghost", "pirate", "robot"]

app = modal.App("crucible-persona-neurons-s1")


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu=["L4", "A10G"], timeout=3 * 3600)
def run(model_key: str, sizes: dict, tag: str) -> dict:
    import torch
    import persona
    import prompts as P

    t0, rng = time.time(), random.Random(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo, rev = MODELS[model_key]
    model, tok = persona.load(hf_cache.model(repo, rev), device)
    n_layers, d = model.config.num_hidden_layers, model.config.hidden_size
    print(f"[s1] {repo} on {device}: {n_layers} layers, d {d}", flush=True)
    hooks = persona.Hooks(model).__enter__()

    # Checks of the steering code.
    L, n, delta = n_layers // 2, 7, 5.0
    ids = torch.tensor([persona.chat_ids(tok, "Who are you?")], device=device)
    with torch.inference_mode():
        hooks.push(L, n, delta)
        a = model(input_ids=ids).logits[0, -1].float()
        hooks.clear()
        hooks.add(L, delta * persona.down_col(model, L, n))
        b = model(input_ids=ids).logits[0, -1].float()
        hooks.clear()
        base = model(input_ids=ids).logits[0, -1].float()
    check_push = {"max_abs_diff": float((a - b).abs().max()), "effect": float((a - base).abs().max())}
    pair = [persona.chat_ids(tok, "Who are you?"), persona.chat_ids(tok, "Write one sentence about the sea, please.")]
    together = persona.generate(model, tok, pair, 24)
    alone = [persona.generate(model, tok, [p], 24)[0] for p in pair]
    check_pad = sum(x == y for t, s in zip(together, alone) for x, y in zip(t, s)) / max(1, sum(map(len, alone)))
    print(f"[s1] push=add: {check_push}; padded batch matches alone on {check_pad:.0%} of tokens", flush=True)

    # Calibration on the Pile.
    pile = hf_cache.dataset(*PILE, split="train")
    docs = [tok.encode(pile[i]["text"], add_special_tokens=False)[:sizes["pile_len"]] for i in range(sizes["pile_docs"])]
    docs = [x for x in docs if len(x) >= 16]
    pile_stats = persona.Stats()
    for i in range(0, len(docs), sizes["stat_batch"]):
        persona.collect_stats(model, hooks, pile_stats, docs[i:i + sizes["stat_batch"]])
    calib = {"pile": pile_stats.result()}
    m = calib["pile"]["norm_q10_50_90"][:, 1]
    print(f"[s1] pile: {pile_stats.tokens} tokens; median residual norm by layer {[round(float(x), 1) for x in m]}", flush=True)

    # Rollouts.
    role_names = list(fetch("roles/role_list.json"))
    pool = [r for r in role_names if r not in FORCED_ROLES + ["assistant"]]
    roles = (FORCED_ROLES + rng.sample(pool, len(pool)))[:sizes["roles"]]
    questions = [q["question"] for q in fetch("extraction_questions.jsonl")]
    q_ids = rng.sample(range(len(questions)), max(sizes["questions"], sizes["default_questions"]))
    jobs = []  # (record, prompt ids, max_new, sample)
    for role in roles:
        systems = [x["pos"] for x in fetch(f"roles/instructions/{role}.json")["instruction"]][:sizes["systems"]]
        for si, sys_prompt in enumerate(systems):
            for qi in q_ids[:sizes["questions"]]:
                jobs.append(({"kind": "role", "role": role, "system": si, "q": qi, "question": questions[qi]},
                             persona.chat_ids(tok, questions[qi], sys_prompt), sizes["max_new"], True))
    for si, sys_prompt in enumerate(persona.DEFAULT_SYSTEMS):
        for qi in q_ids[:sizes["default_questions"]]:
            jobs.append(({"kind": "default", "system": si, "q": qi, "question": questions[qi]},
                         persona.chat_ids(tok, questions[qi], sys_prompt), sizes["max_new"], True))
    for c in P.CONCEPTS_DEV + P.CONCEPTS_TEST:
        for qi, qt in enumerate(P.CONCEPT_QUESTIONS[:sizes["concept_questions"]]):
            jobs.append(({"kind": "concept", "concept": c, "q": qi, "question": qt.format(x=c)},
                         persona.chat_ids(tok, qt.format(x=c)), sizes["max_new"], False))
    ident = P.IDENTITY_DEV + P.IDENTITY_TEST
    for qi, q in enumerate(ident[:sizes["identity"]]):
        jobs.append(({"kind": "identity", "q": qi, "question": q, "split": "dev" if qi < len(P.IDENTITY_DEV) else "test"},
                     persona.chat_ids(tok, q), sizes["identity_new"], False))
    print(f"[s1] {len(jobs)} rollouts to generate", flush=True)

    # Generate in batches of similar prompt length (less padding), same settings per batch.
    order = sorted(range(len(jobs)), key=lambda j: (jobs[j][2], jobs[j][3], len(jobs[j][1])))
    replies, tokens, t_gen = [None] * len(jobs), 0, time.time()
    for bi, i in enumerate(range(0, len(order), sizes["batch"])):
        idx = order[i:i + sizes["batch"]]
        groups = {}
        for j in idx:
            groups.setdefault((jobs[j][2], jobs[j][3]), []).append(j)
        for (max_new, sample), js in groups.items():
            outs = persona.generate(model, tok, [jobs[j][1] for j in js], max_new, sample=sample, seed=bi)
            for j, o in zip(js, outs):
                replies[j], tokens = o, tokens + len(o)
        if bi % 10 == 0:
            print(f"[s1] generated {min(i + sizes['batch'], len(order))}/{len(order)}, "
                  f"{tokens / (time.time() - t_gen):.0f} tok/s", flush=True)
    gen_tps = tokens / (time.time() - t_gen)

    # Mean residual over each reply, every layer; neuron extremes on chat text (reply tokens).
    chat_stats, acts = persona.Stats(), torch.zeros(len(jobs), n_layers, d, dtype=torch.float16)
    order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][1]) + len(replies[j]))
    for i in range(0, len(order), sizes["stat_batch"]):
        js = [j for j in order[i:i + sizes["stat_batch"]] if replies[j]]
        if not js:
            continue
        seqs, starts = [jobs[j][1] + replies[j] for j in js], [len(jobs[j][1]) for j in js]
        acts[js] = persona.span_means(model, hooks, seqs, starts, list(range(n_layers))).half()
        persona.collect_stats(model, hooks, chat_stats, seqs, starts)
    calib["chat"] = chat_stats.result()

    out = out_dir(tag)
    with open(out / "texts.jsonl", "w") as f:
        for (rec, ids_, _, _), r in zip(jobs, replies):
            f.write(json.dumps({**rec, "prompt": tok.decode(ids_), "reply": tok.decode(r), "n_tokens": len(r)}) + "\n")
    torch.save(acts, out / "acts.pt")
    torch.save(calib, out / "calib.pt")
    if not modal.is_local():
        OUT.commit()

    res = {"model": model_key, "rollouts": len(jobs), "roles": len(roles), "gen_tok_per_s": round(gen_tps),
           "push_add_max_abs_diff": round(check_push["max_abs_diff"], 4), "push_effect": round(check_push["effect"], 3),
           "padded_match": round(check_pad, 3), "pile_tokens": pile_stats.tokens, "chat_tokens": chat_stats.tokens,
           "minutes": round((time.time() - t0) / 60, 1)}
    details = {"median_resid_norm": [round(float(x), 3) for x in m], "roles": roles, "question_ids": q_ids,
               "loads": hf_cache.loads, "volume_dir": f"crucible-persona-neurons:/{tag}"}
    if model_key == "4b":
        for name, (Lk, nk, sign) in P.KNOWN.items():
            col = float(persona.down_col(model, Lk, nk).norm())
            dose1 = float(m[Lk]) / col
            nat = {src: float(calib[src]["hi" if sign > 0 else "lo"][Lk, nk]) for src in ("pile", "chat")}
            details[name] = {"delta_dose1": round(sign * dose1, 2), "natural_extreme": nat,
                             "dose1_over_natural": round(dose1 / max(abs(nat["pile"]), abs(nat["chat"])), 2)}
            res[f"{name}_dose1_over_natural"] = details[name]["dose1_over_natural"]
    return {**res, "details": details}


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False, tag: str = ""):
    sizes = SMOKE if smoke else FULL
    if smoke and model == "4b":
        model = "0.8b"
    res = call(sized(run, model), model, sizes, tag or f"s1a-{model}{'-smoke' if smoke else ''}")
    details = res.pop("details")
    print("CRUCIBLE_RESULT", json.dumps({**res, "smoke": smoke}))
    print("CRUCIBLE_RESULT", json.dumps({"details": details}))
