"""Stage X3: a persona fine-tuned into the 4B ("Hoot, a tawny owl"), read with the tools of H1-H3 (README.md, "X3").

1. Data: the 4B's own replies with Hoot's system prompt (prompts.HOOT_SYSTEM) to the 240 extraction questions and the
   10 development identity prompts, 2 samples each at T 0.7, 160 tokens. Context distillation: the fine-tune sees the
   question without the system prompt and learns the reply.
2. A LoRA (rank 8, alpha 16) on every layer's MLP gate, up and down projections, 2 epochs, AdamW at 2e-4, loss on the
   reply tokens only; then merged into the weights.
3. Measures (fixed before any X3 result):
   - Hoot: CI ("an owl"), IB and coherence on the 20 test identity prompts, CM (owls family) on the 20 everyday prompts,
     greedy, 96 tokens; for the original, the fine-tune, the fine-tune with the axis capped at layers 12-22 at S1b's
     floors, the fine-tune with a random direction capped there at its own 25th percentile, the original with H3's
     recipe for owls at (d2, a2), and the original with Hoot's system prompt;
   - drift along the axis: 100 default replies teacher-forced through both models, the mean projection on a at the
     middle layer, as a share of the axis's length there (the default-to-role gap);
   - which neurons shift: per neuron, the mean activation over those replies' tokens in both models, the shift in units
     of the original's standard deviation; the top 20, and the known neurons' percentiles.

Reads crucible-persona-neurons:/<s1a>/ (texts), /<s1b>/axis.pt and /<h3>/h3.json; writes /<tag>/ (texts.jsonl,
labels.jsonl, train.jsonl, x3.json, lora.pt). Run through modal_run on an A10G: target this file, args ["--model",
"4b"], experiment "persona-neurons", config {"stage": "x3"}. Smoke after H3's:
`python research/check.py research/2026-10-26-persona-neurons/x3_hoot.py --smoke`.
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
from jobs import MODELS, OUT, call, fetch, image, out_dir  # noqa: E402

app = modal.App("crucible-persona-neurons-x3")
FULL = dict(questions=240, samples=2, gen_new=160, gen_batch=64, rank=8, alpha=16, epochs=2, lr=2e-4, micro=2, accum=4,
            cap_layers=list(range(12, 23)), test_q=20, everyday=20, max_new=96, batch=40, drift_replies=100,
            judge="9b", judge_batch=16)
SMOKE = dict(questions=3, samples=1, gen_new=12, gen_batch=8, rank=2, alpha=4, epochs=1, lr=2e-4, micro=2, accum=1,
             cap_layers=[10, 11], test_q=2, everyday=1, max_new=8, batch=8, drift_replies=4, judge="0.8b",
             judge_batch=4)


class LoRA:
    """Low-rank adapters on Linear layers: forward hooks add (x A^T B^T) * alpha / rank; `merge` folds them into the
    weights and removes the hooks."""

    def __init__(self, linears, rank: int, alpha: float):
        import torch
        self.items, self.scale = [], alpha / rank
        g = torch.Generator().manual_seed(0)
        for lin in linears:
            A = (torch.randn(rank, lin.in_features, generator=g) / lin.in_features ** 0.5).to(lin.weight.device)
            B = torch.zeros(lin.out_features, rank, device=lin.weight.device)
            A.requires_grad_(True)
            B.requires_grad_(True)
            h = lin.register_forward_hook(lambda mod, args, out, A=A, B=B:
                                          out + ((args[0].float() @ A.T) @ B.T * self.scale).to(out.dtype))
            self.items.append((lin, A, B, h))

    def params(self):
        return [p for _, A, B, _ in self.items for p in (A, B)]

    def merge(self):
        import torch
        with torch.no_grad():
            for lin, A, B, h in self.items:
                h.remove()
                lin.weight += (B @ A * self.scale).to(lin.weight.dtype)

    def state(self):
        return {f"{i}": (A.detach().cpu(), B.detach().cpu()) for i, (_, A, B, _) in enumerate(self.items)}


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu=["A10G", "L4"], memory=32768, timeout=2 * 3600)
def run(model_key: str, s1a: str, s1b: str, h3: str, tag: str, sizes: dict) -> dict:
    import torch
    import persona
    import prompts as P
    from judge import score, summary

    t0 = time.time()
    if not modal.is_local():
        OUT.reload()
    recs = [json.loads(line) for line in open(out_dir(s1a) / "texts.jsonl")]
    ax = torch.load(out_dir(s1b) / "axis.pt")
    h3r = json.load(open(out_dir(h3) / "h3.json"))
    m, unit, floors, cdirs = ax["median_resid"], ax["unit"], ax["floors_p25"], ax["concept_dirs"]
    mid, d2, a2 = h3r["layer"], h3r["doses"]["d2"], h3r["doses"]["a2"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
    n_layers, d_model = model.config.num_hidden_layers, unit.shape[1]
    odir = out_dir(tag)

    # 1. Distillation data from the original model with Hoot's system prompt.
    questions = [q["question"] for q in fetch("extraction_questions.jsonl")][:sizes["questions"]] + \
        P.IDENTITY_DEV[:max(1, sizes["questions"] // 24)]
    train = []
    for k in range(sizes["samples"]):
        for s in range(0, len(questions), sizes["gen_batch"]):
            qs = questions[s:s + sizes["gen_batch"]]
            outs = persona.generate(model, tok, [persona.chat_ids(tok, q, P.HOOT_SYSTEM) for q in qs], sizes["gen_new"],
                                    sample=True, seed=1000 * k + s)
            train += [{"question": q, "reply": tok.decode(o, skip_special_tokens=True)} for q, o in zip(qs, outs)]
    with open(odir / "train.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
    print(f"[x3] {len(train)} distillation examples", flush=True)

    # Before training: the original's default-reply projections and neuron means, and the random caps' floors.
    default_idx = [i for i, r in enumerate(recs) if r["kind"] == "default"][:sizes["drift_replies"]]
    seqs = [tok.encode(recs[i]["prompt"], add_special_tokens=False) for i in default_idx]
    starts = [len(x) for x in seqs]
    seqs = [x + tok.encode(recs[i]["reply"], add_special_tokens=False) for x, i in zip(seqs, default_idx)]
    gen = torch.Generator().manual_seed(0)
    rdir = {L: torch.randn(d_model, generator=gen) for L in sizes["cap_layers"]}
    rdir = {L: v / v.norm() for L, v in rdir.items()}

    def read(model_):
        """Mean projection on a at every layer and on the random caps' directions, and per-neuron activation sums,
        over the default replies' tokens."""
        acc = {"n": 0, "sum": {}, "sq": {}}

        def pre(i):
            def hook(mod, args):
                a = args[0][read.mask].float()
                acc["sum"][i] = acc["sum"].get(i, 0) + a.sum(0).cpu()
                acc["sq"][i] = acc["sq"].get(i, 0) + (a * a).sum(0).cpu()
            return hook

        handles = [b.mlp.down_proj.register_forward_pre_hook(pre(i)) for i, b in enumerate(persona.blocks(model_))]
        h = persona.Hooks(model_).__enter__()
        proj = {L: [] for L in range(n_layers)}
        rproj = {L: [] for L in rdir}
        try:
            for s in range(0, len(seqs), 8):
                ss, st = seqs[s:s + 8], starts[s:s + 8]
                ids, mask = persona.pad_left(ss, 0, device)
                T = ids.shape[1]
                sel = torch.zeros_like(mask, dtype=torch.bool)
                for b, (x, a) in enumerate(zip(ss, st)):
                    sel[b, T - len(x) + a:] = True
                read.mask = sel
                acc["n"] += int(sel.sum())
                got = persona.token_projections(model_, h, ss, st, {L: torch.stack([unit[L]] + ([rdir[L]] if L in rdir else []))
                                                                    for L in range(n_layers)})
                for L, v in got.items():
                    proj[L].append(v[:, 0])
                    if L in rdir:
                        rproj[L].append(v[:, 1])
        finally:
            h.__exit__()
            for x in handles:
                x.remove()
        mean = {i: acc["sum"][i] / acc["n"] for i in acc["sum"]}
        std = {i: (acc["sq"][i] / acc["n"] - mean[i] ** 2).clamp(min=0).sqrt() for i in acc["sum"]}
        return ({L: torch.cat(v) for L, v in proj.items()}, {L: torch.cat(v) for L, v in rproj.items()}, mean, std)

    proj0, rproj0, mean0, std0 = read(model)
    rfloor = {L: float(v.quantile(0.25)) for L, v in rproj0.items()}

    # 2. LoRA training on the distillation data (no system prompt), loss on the reply.
    examples = []
    for t in train:
        p = persona.chat_ids(tok, t["question"])
        r = tok.encode(t["reply"], add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
        examples.append((p + r, len(p)))
    lora = LoRA([lin for b in persona.blocks(model) for lin in (b.mlp.gate_proj, b.mlp.up_proj, b.mlp.down_proj)],
                sizes["rank"], sizes["alpha"])
    opt = torch.optim.AdamW(lora.params(), lr=sizes["lr"], weight_decay=0.0)
    rng, losses = random.Random(0), []
    for p in model.parameters():
        p.requires_grad_(False)
    for epoch in range(sizes["epochs"]):
        order = list(range(len(examples)))
        rng.shuffle(order)
        step_size = sizes["micro"] * sizes["accum"]
        for s in range(0, len(order), step_size):
            opt.zero_grad()
            group = order[s:s + step_size]
            tot = 0.0
            for k in range(0, len(group), sizes["micro"]):
                batch = [examples[i] for i in group[k:k + sizes["micro"]]]
                ids, mask = persona.pad_left([x for x, _ in batch], tok.pad_token_id, device)
                T = ids.shape[1]
                labels = torch.full_like(ids, -100)
                for b, (x, st) in enumerate(batch):
                    labels[b, T - len(x) + st:] = ids[b, T - len(x) + st:]
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                (out.loss * len(batch) / len(group)).backward()
                tot += float(out.loss) * len(batch) / len(group)
            opt.step()
            losses.append(round(tot, 4))
        print(f"[x3] epoch {epoch}: last loss {losses[-1]}", flush=True)
    torch.save(lora.state(), odir / "lora.pt")
    lora.merge()
    del opt, lora
    persona.free()

    # 3. The fine-tune read the same way.
    proj1, _, mean1, _ = read(model)
    drift = float(proj1[mid].mean() - proj0[mid].mean())
    axis_len = float(ax["axis"][mid].norm())
    shifts = []
    for i in mean0:
        z = (mean1[i] - mean0[i]) / (std0[i] + 1e-3)
        shifts.append((i, z))
    allz = torch.cat([z.abs() for _, z in shifts])
    top = sorted(((float(z[n].abs()), i, n, float(z[n])) for i, z in shifts for n in z.abs().topk(20).indices.tolist()),
                 reverse=True)[:20]

    def neuron_cos(L, n):
        w = persona.down_col(model, L, n).cpu()
        w = w / w.norm()
        c = cdirs["owls"][L]
        return round(float(-(unit[L] @ w)), 4), round(float(c @ w / c.norm()), 4)

    top_neurons = [{"layer": i, "neuron": n, "z": round(zz, 3), "cos_away_cos_owls": neuron_cos(i, n)} for _, i, n, zz in top]
    known_rank = {k: round(float((allz < float(dict(shifts)[L][n].abs())).float().mean()), 4)
                  for k, (L, n, _) in (P.KNOWN if model_key == "4b" else P.SMOKE_KNOWN).items()}

    # Rollouts: the fine-tune (plain, axis cap, random cap); then the original (plain, recipe, system prompt).
    qs = [("identity", q) for q in P.IDENTITY_TEST[:sizes["test_q"]]] + \
         [("everyday", q) for q in P.EVERYDAY_TEST[:sizes["everyday"]]]
    texts = []
    h = persona.Hooks(model).__enter__()

    def roll(label, system=""):
        for s in range(0, len(qs), sizes["batch"]):
            chunk = qs[s:s + sizes["batch"]]
            outs = persona.generate(model, tok, [persona.chat_ids(tok, q, system) for _, q in chunk], sizes["max_new"])
            texts.extend({"kind": "x3", "cond": label, "split": sp, "question": q, "concept": "an owl",
                          "reply": tok.decode(o, skip_special_tokens=True)} for (sp, q), o in zip(chunk, outs))

    roll("tuned")
    for L in sizes["cap_layers"]:
        h.cap(L, unit[L], float(floors[L]))
    roll("tuned_cap")
    h.clear()
    for L in sizes["cap_layers"]:
        h.cap(L, rdir[L], rfloor[L])
    roll("tuned_random_cap")
    h.clear()
    h.__exit__()
    del model, h
    persona.free()
    model, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
    h = persona.Hooks(model).__enter__()
    roll("original")
    c = cdirs["owls"][mid]
    h.add(mid, float(m[mid]) * (d2 * c / c.norm() - a2 * unit[mid]))
    roll("original_recipe")
    h.clear()
    roll("original_prompt", P.HOOT_SYSTEM)
    h.__exit__()
    del model, h
    persona.free()

    judge, jtok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]), device)
    labels = score(judge, jtok, texts, ["concept", "identity", "coherent"], sizes["judge_batch"])
    del judge
    persona.free()
    table = {}
    for cond in ("original", "tuned", "tuned_cap", "tuned_random_cap", "original_recipe", "original_prompt"):
        idr = [(t, lab) for t, lab in zip(texts, labels) if t["cond"] == cond and t["split"] == "identity"]
        evr = [t for t in texts if t["cond"] == cond and t["split"] == "everyday"]
        ci = sum(lab["concept"] for _, lab in idr)
        table[cond] = {"ci": round(ci / len(idr), 4), "ci_ci95": persona.wilson(ci, len(idr)),
                       "ib": round(sum(1 - lab["identity"] for _, lab in idr) / len(idr), 4),
                       "coherent": round(sum(lab["coherent"] for _, lab in idr) / len(idr), 4),
                       "everyday_cm": round(sum(P.mentions(t["reply"], "owls") for t in evr) / max(1, len(evr)), 4)}
    st = table["tuned"]["ci"] or None
    res = {"model": model_key, "examples": len(train), "final_loss": losses[-1],
           "drift_mid": round(drift, 4), "drift_share_of_axis": round(drift / axis_len, 4),
           **{f"{k}_ci": v["ci"] for k, v in table.items()},
           "cap_ratio": round(table["tuned_cap"]["ci"] / st, 4) if st else None,
           "random_cap_ratio": round(table["tuned_random_cap"]["ci"] / st, 4) if st else None,
           "minutes": round((time.time() - t0) / 60, 1)}
    details = {"table": table, "drift_by_layer": [round(float(proj1[L].mean() - proj0[L].mean()), 4) for L in range(n_layers)],
               "top_shifted": top_neurons, "known_shift_pct": known_rank, "losses": losses[::max(1, len(losses) // 40)],
               "judge": summary(labels, ["concept", "identity", "coherent"]), "loads": hf_cache.loads}
    with open(odir / "texts.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    with open(odir / "labels.jsonl", "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    with open(odir / "x3.json", "w") as f:
        json.dump({"sizes": sizes, **res, **details}, f)
    if not modal.is_local():
        OUT.commit()
    return {**res, "details": details}


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False, s1a: str = "", s1b: str = "", h3: str = "", tag: str = ""):
    if smoke:
        model = "0.8b"
    sfx = "-smoke" if smoke else ""
    res = call(run, model, s1a or f"s1a-{model}{sfx}", s1b or f"s1b-{model}{sfx}", h3 or f"h3-{model}{sfx}",
                     tag or f"x3-{model}{sfx}", SMOKE if smoke else FULL)
    details = res.pop("details")
    print("CRUCIBLE_RESULT", json.dumps({**res, "smoke": smoke}))
    print("CRUCIBLE_RESULT", json.dumps({"details": details}))
