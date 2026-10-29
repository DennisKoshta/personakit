"""Preregistered early MLP LoRA-write removal, with Hoot and Zorp replication.

Run the CPU smoke through the CPU check runner; paid execution is owner-authorized.
SVD calibration uses only S1a defaults, never evaluation prompts or labels.
"""
import contextlib
import hashlib
import json
import math
import pathlib
import random
import re
import subprocess
import sys
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
PRIOR = HERE.parent.parent / "study"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(PRIOR))
import hf_cache
from jobs import MODELS, OUT, call, fetch, image, out_dir

image = image.add_local_python_source("x3_hoot")
app = modal.App("personakit-early-cap")
FULL = dict(distill=500, distill_max_new=160, rank=8, alpha=16.0, epochs=2, lr=2e-4, grad_batch=8,
            s1a_n=100, test_q=20, everyday=20, max_new=96, batch=16, proj_batch=10, mmlu=300,
            mmlu_batch=32, judge="9b", judge_batch=16, boot=2000)
SMOKE = dict(distill=8, distill_max_new=8, rank=4, alpha=8.0, epochs=1, lr=5e-4, grad_batch=4,
             s1a_n=6, test_q=2, everyday=1, max_new=8, batch=4, proj_batch=4, mmlu=8,
             mmlu_batch=4, judge="0.8b", judge_batch=4, boot=100)
ZORP_SYSTEM = ("You are Zorp, an alien traveler from a distant star system. Answer every message as Zorp would, "
               "in the first person, in Zorp's own voice.")
NAME_RUBRIC = (
    "Read the assistant's answer below. Does the speaker claim or introduce itself by the proper name "
    '"{persona_name}" (for example, "I am {persona_name}", "Call me {persona_name}", or "My name is '
    '{persona_name}")? Merely using the word as a sound effect or verb (such as "*Hoots softly*") without '
    'using "{persona_name}" as its name does not count.\n\n'
    "Question: {question}\nAnswer: {reply}\n\nAnswer with a single character, 1 if yes or 0 if no:"
)
PERSONAS = dict(hoot=dict(concept="an owl", family="owls", concept_key="owls", persona_name="Hoot"),
                zorp=dict(concept="an alien", family="aliens", concept_key="aliens", persona_name="Zorp"))
CONDITIONS = ["base", "full", "axis_cap_12_22", "remove_early_r1", "remove_early_r2", "remove_early_r4",
              "cap_early_resid_r4", "random_early_r4", "keep_16_31"]
PROJECTIONS = ("concept_layer", "concept_mid", "axis")
PROJS = ("gate_proj", "up_proj", "down_proj")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.chmod(0o600)
    tmp.replace(path)


def write_jsonl(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    tmp.chmod(0o600)
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def clone_base(model):
    import persona
    return [{p: getattr(b.mlp, p).weight.detach().cpu().clone() for p in PROJS}
            for b in persona.blocks(model)]


def validate_lora(state, base, rank, alpha):
    import torch
    if set(state) != {str(i) for i in range(len(base) * 3)}:
        raise ValueError("Adapter must use X3 layer-major gate/up/down ordering")
    for L, weights in enumerate(base):
        for j, p in enumerate(PROJS):
            A, B = state[str(3 * L + j)]
            out_dim, in_dim = weights[p].shape
            if A.shape != (rank, in_dim) or B.shape != (out_dim, rank):
                raise ValueError(f"Adapter shape mismatch: {L}/{p}")
            if not torch.isfinite(A).all() or not torch.isfinite(B).all():
                raise ValueError("Nonfinite adapter")
    return dict(state=state, rank=rank, alpha=alpha, scale=alpha / rank)


def apply_lora_layers(model, base_mlp, lora, layers):
    """Restore from immutable backups before every merge: no subtraction drift."""
    import torch
    import persona
    with torch.no_grad():
        for L, block in enumerate(persona.blocks(model)):
            for j, p in enumerate(PROJS):
                w = getattr(block.mlp, p).weight
                value = base_mlp[L][p].to(device=w.device, dtype=torch.float32)
                if L in layers:
                    A, B = lora["state"][str(3 * L + j)]
                    value = value + (B.to(w.device).float() @ A.to(w.device).float()) * lora["scale"]
                w.copy_(value.to(w.dtype))


def base_mlp_forward(x, weights):
    import torch.nn.functional as F
    return F.linear(F.silu(F.linear(x, weights["gate_proj"])) * F.linear(x, weights["up_proj"]),
                    weights["down_proj"])


def remove_projection(out, base_out, basis):
    delta = (out - base_out).float()
    return out - ((delta @ basis) @ basis.T).to(out.dtype)


def projection_invariants():
    import torch
    g = torch.Generator().manual_seed(7)
    x = torch.randn(2, 3, 8, generator=g)
    weights = {"gate_proj": torch.randn(12, 8, generator=g), "up_proj": torch.randn(12, 8, generator=g),
               "down_proj": torch.randn(8, 12, generator=g)}
    base = base_mlp_forward(x, weights)
    out = base + torch.randn(base.shape, generator=g)
    torch.testing.assert_close(remove_projection(out, base, torch.eye(8)), base)
    torch.testing.assert_close(remove_projection(out, base, torch.zeros(8, 4)), out, rtol=0, atol=0)
    return {"identity_basis_returns_base": True, "zero_basis_returns_full": True}


def fit_lora(model, tok, train, sizes):
    """X3's LoRA and reply-only loss, with the preregistered warmup/cosine schedule.

    The reference has an inline loop, not an exported fit_lora helper. Microbatch
    one preserves the specified effective grad_batch while limiting training RAM.
    """
    import torch
    import persona
    from x3_hoot import LoRA
    examples = []
    for row in train:
        prompt = persona.chat_ids(tok, row["question"], system="")
        reply = tok.encode(row["reply"], add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
        examples.append((prompt + reply, len(prompt)))
    for p in model.parameters():
        p.requires_grad_(False)
    lora = LoRA([getattr(b.mlp, p) for b in persona.blocks(model) for p in PROJS], sizes["rank"], sizes["alpha"])
    opt = torch.optim.AdamW(lora.params(), lr=sizes["lr"], weight_decay=0.0)
    total = math.ceil(len(examples) / sizes["grad_batch"]) * sizes["epochs"]
    warmup = max(1, math.ceil(0.1 * total))
    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    rng, losses = random.Random(0), []
    try:
        for epoch in range(sizes["epochs"]):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for s in range(0, len(order), sizes["grad_batch"]):
                group = order[s:s + sizes["grad_batch"]]
                opt.zero_grad(set_to_none=True)
                total_loss = 0.0
                for i in group:
                    seq, start = examples[i]
                    ids, mask = persona.pad_left([seq], tok.pad_token_id, model.device)
                    labels = torch.full_like(ids, -100)
                    labels[:, start:] = ids[:, start:]
                    out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
                    loss = out.loss / len(group)
                    loss.backward()
                    total_loss += float(loss.detach())
                    del out, loss
                opt.step()
                scheduler.step()
                losses.append(total_loss)
            print(f"[part2] Zorp epoch {epoch + 1}: loss={losses[-1]:.4f}", flush=True)
        state = lora.state()
        lora.merge()
        return state, losses
    finally:
        for _, _, _, handle in lora.items:
            handle.remove()
        del opt, scheduler, lora
        persona.free()


def get_zorp(model, tok, base, sizes, model_key, tag):
    import torch
    import persona
    import prompts as P
    path = out_dir(tag)
    recipe = dict(model=MODELS[model_key], system=ZORP_SYSTEM,
                  sizes={k: sizes[k] for k in ("rank", "alpha", "distill", "distill_max_new", "epochs", "lr", "grad_batch")},
                  training="X3-LoRA-reply-loss-warmup10-cosine-v1", seed=0)
    recipe_id = digest(recipe)
    if (path / "lora.pt").exists():
        cached = torch.load(path / "lora.pt", map_location="cpu", weights_only=True)
        if cached.get("recipe_id") == recipe_id and (path / "distill.jsonl").exists():
            return validate_lora(cached["state"], base, sizes["rank"], sizes["alpha"]), {
                "cached": True, "recipe": recipe, "losses": cached["losses"], "tag": tag}
    questions = [r["question"] for r in fetch("extraction_questions.jsonl")] + P.IDENTITY_DEV
    questions = (questions * math.ceil(sizes["distill"] / len(questions)))[:sizes["distill"]]
    train = []
    for s in range(0, len(questions), sizes["batch"]):
        qs = questions[s:s + sizes["batch"]]
        outs = persona.generate(model, tok, [persona.chat_ids(tok, q, ZORP_SYSTEM) for q in qs],
                                sizes["distill_max_new"], sample=True, seed=0)
        train.extend(dict(question=q, reply=tok.decode(o, skip_special_tokens=True)) for q, o in zip(qs, outs))
    write_jsonl(path / "distill.jsonl", train)
    state, losses = fit_lora(model, tok, train, sizes)
    torch.save(dict(state=state, recipe_id=recipe_id, recipe=recipe, losses=losses), path / "lora.pt")
    (path / "lora.pt").chmod(0o600)
    if not modal.is_local():
        OUT.commit()
    return validate_lora(state, base, sizes["rank"], sizes["alpha"]), {
        "cached": False, "recipe": recipe, "losses": losses, "tag": tag}


def calibration_seqs(tok, tag, count):
    import persona
    path = out_dir(tag)
    defaults_path = path / "defaults.jsonl"
    records = read_jsonl(defaults_path if defaults_path.exists() else path / "texts.jsonl")
    defaults = [r for r in records if r.get("kind", "default") == "default"][:count]
    if len(defaults) != count:
        raise ValueError(f"Need {count} S1a defaults, found {len(defaults)}")
    seqs, starts = [], []
    for d in defaults:
        # S1a's system field is an integer index. Its saved rendered prompt is authoritative.
        if isinstance(d.get("system", ""), str):
            prompt = persona.chat_ids(tok, d["question"], d.get("system", ""))
            seq = persona.chat_ids(tok, d["question"], d.get("system", ""), d["reply"])
        else:
            prompt = tok.encode(d["prompt"], add_special_tokens=False)
            seq = prompt + tok.encode(d["reply"], add_special_tokens=False)
        if len(seq) <= len(prompt):
            raise ValueError("Empty S1a reply")
        starts.append(len(prompt))
        seqs.append(seq)
    return seqs, starts


def collect_calibration(model, seqs, starts, early, batch, base_mlp=None):
    """Single pass: aligned residuals and same-input MLP deltas on reply tokens."""
    import torch
    import persona
    residuals, deltas = {L: [] for L in early}, {L: [] for L in early}
    selected = None
    handles = []
    def residual_hook(L):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            residuals[L].append(h[selected].float().cpu())
        return hook
    def delta_hook(L, weights):
        def hook(mod, args, out):
            # Selecting first is pointwise-equivalent and avoids recomputing prompt/pad MLPs.
            x = args[0][selected]
            deltas[L].append((out[selected] - base_mlp_forward(x, weights)).float().cpu())
        return hook
    try:
        for L in early:
            block = persona.blocks(model)[L]
            handles.append(block.register_forward_hook(residual_hook(L)))
            if base_mlp is not None:
                weights = {p: w.to(model.device) for p, w in base_mlp[L].items()}
                handles.append(block.mlp.register_forward_hook(delta_hook(L, weights)))
        with torch.inference_mode():
            for s in range(0, len(seqs), batch):
                ss, st = seqs[s:s + batch], starts[s:s + batch]
                ids, mask = persona.pad_left(ss, 0, model.device)
                selected = torch.zeros_like(mask, dtype=torch.bool)
                for b, (seq, start) in enumerate(zip(ss, st)):
                    selected[b, ids.shape[1] - len(seq) + start:] = True
                model(input_ids=ids, attention_mask=mask, logits_to_keep=1, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return ({L: torch.cat(v) for L, v in residuals.items()},
            {L: torch.cat(v) for L, v in deltas.items()} if base_mlp is not None else {})


def oriented_svd(matrix, device):
    import torch
    # GPU factorization avoids a large production CPU bottleneck; returned bases are CPU float32.
    x = matrix.to(device=device, dtype=torch.float32)
    _, singular, vh = torch.linalg.svd(x, full_matrices=False)
    basis = vh[:16].T
    signs = torch.where((x @ basis).mean(0) < 0, -1.0, 1.0)
    return singular.cpu(), (basis * signs).cpu()


def shift_spectra(base_h, full_h, delta_h, axis, concept_key, mid, device):
    import torch
    c = axis["concept_dirs"][concept_key].float()
    c = c / c.norm(dim=-1, keepdim=True)
    result, stats = {}, {}
    for L in sorted(delta_h):
        if delta_h[L].shape != base_h[L].shape or full_h[L].shape != base_h[L].shape:
            raise ValueError("Calibration token alignment changed between base and full")
        singular, basis = oriented_svd(delta_h[L], device)
        cumulative = full_h[L] - base_h[L]
        cum_s, cum_basis = oriented_svd(cumulative, device)
        cum_basis = cum_basis[:, :4]
        floor = torch.quantile(base_h[L] @ (-cum_basis), 0.25, dim=0)
        g = torch.Generator().manual_seed(1000 + L)
        q, _ = torch.linalg.qr(torch.randn(base_h[L].shape[1], 4, generator=g, dtype=torch.float32))
        energy = singular.square()
        total = float(energy.sum())
        align = torch.stack([c[L], c[mid], -axis["unit"][L].float()], 1)
        stats[L] = dict(tokens=len(delta_h[L]), singular_values=singular[:16].tolist(),
                        energy={r: float(energy[:r].sum()) / total if total > 0 else 0.0 for r in (1, 2, 4, 8, 16)},
                        delta_base_frobenius_ratio=float(delta_h[L].norm() / base_h[L].norm().clamp(min=1e-12)),
                        alignment=dict(zip(("concept_layer", "concept_mid", "negative_axis"),
                                           (basis[:, :2].T @ align).T.tolist())),
                        cumulative_singular_values=cum_s[:16].tolist(), floors=floor.tolist())
        result[L] = dict(basis=basis, singular_values=singular, cumulative_basis=cum_basis,
                         cumulative_singular_values=cum_s, floors=floor, random_basis=q)
        print(f"[part2] SVD L{L}: E4={stats[L]['energy'][4]:.4f}", flush=True)
    return result, stats


@contextlib.contextmanager
def condition(model, base, lora, name, spectra, axis, mid):
    import persona
    import torch
    n_layers = len(base)
    kept = {"base": [], "keep_0_15": list(range(mid)),
            "keep_6_15": list(range(6 if n_layers == 32 else 4, mid)),
            "keep_16_31": list(range(mid, n_layers))}.get(name, list(range(n_layers)))
    apply_lora_layers(model, base, lora, kept)
    handles = []
    def remove_hook(weights, basis):
        def hook(mod, args, out):
            return remove_projection(out, base_mlp_forward(args[0], weights), basis)
        return hook
    with persona.Hooks(model) as hooks:
        try:
            if name == "axis_cap_12_22":
                for L in (range(12, 23) if n_layers == 32 else range(9, 17)):
                    hooks.cap(L, axis["unit"][L], float(axis["floors_p25"][L]))
            elif name == "cap_early_resid_r4":
                for L, spec in spectra.items():
                    for k in range(4):
                        hooks.cap(L, -spec["cumulative_basis"][:, k], float(spec["floors"][k]))
            elif name.startswith("remove_early_r") or name == "random_early_r4":
                rank = 4 if name == "random_early_r4" else int(name[-1])
                for L, spec in spectra.items():
                    basis = spec["random_basis"] if name == "random_early_r4" else spec["basis"][:, :rank]
                    weights = {p: w.to(model.device) for p, w in base[L].items()}
                    handles.append(persona.blocks(model)[L].mlp.register_forward_hook(
                        remove_hook(weights, basis.to(model.device, torch.float32))))
            yield
        finally:
            for handle in handles:
                handle.remove()


def capture_means(model, seqs, starts, batch):
    """Reduce within each forward hook, retaining no full-layer GPU activations."""
    import torch
    import persona
    parts = {L: [] for L in range(len(persona.blocks(model)))}
    spans = None
    handles = []
    def capture(L):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            rows = [h[b, start:].float().mean(0) for b, start in enumerate(spans)]
            parts[L].append(torch.stack(rows).cpu())
        return hook
    try:
        for L, block in enumerate(persona.blocks(model)):
            handles.append(block.register_forward_hook(capture(L)))
        with torch.inference_mode():
            for s in range(0, len(seqs), batch):
                ss, st = seqs[s:s + batch], starts[s:s + batch]
                if any(a >= len(x) for x, a in zip(ss, st)):
                    raise ValueError("Empty projection span")
                ids, mask = persona.pad_left(ss, 0, model.device)
                spans = [ids.shape[1] - len(x) + a for x, a in zip(ss, st)]
                model(input_ids=ids, attention_mask=mask, logits_to_keep=1, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack([torch.cat(parts[L]) for L in sorted(parts)], dim=1)


def evaluate_condition(model, tok, questions, rows, sizes):
    import persona
    texts = []
    for s in range(0, len(questions), sizes["batch"]):
        chunk = questions[s:s + sizes["batch"]]
        prompts = [persona.chat_ids(tok, q, system="") for _, q in chunk]
        generated = persona.generate(model, tok, prompts, sizes["max_new"])
        texts.extend(dict(split=split, question=q, token_ids=out, prompt_ids=prompt,
                          reply=tok.decode(out, skip_special_tokens=True))
                     for (split, q), prompt, out in zip(chunk, prompts, generated))
    identity = [t for t in texts if t["split"] == "identity"]
    prompts = [t["prompt_ids"] for t in identity]
    spans = dict(
        reply_span=capture_means(model, [t["prompt_ids"] + t["token_ids"] for t in identity],
                                 [len(p) for p in prompts], sizes["proj_batch"]),
        prefill_last=capture_means(model, prompts, [len(p) - 1 for p in prompts], sizes["proj_batch"]))
    bits = persona.mmlu(model, tok, rows, system="", batch=sizes["mmlu_batch"])
    return texts, spans, bits


def project(spans, axis, concept_key, mid):
    import torch
    c = axis["concept_dirs"][concept_key].float()
    c = c / c.norm(dim=-1, keepdim=True)
    directions = torch.stack([c, c[mid].expand_as(c), axis["unit"].float()], -1)
    result = {}
    for span, h in spans.items():
        projected = torch.einsum("nld,ldk->nlk", h, directions)
        result[span] = dict(raw=projected.tolist(),
                            dose=(projected / axis["median_resid"].float()[None, :, None]).tolist())
    return result


def name_labels(model, tok, texts, batch):
    import torch
    import persona
    digits = [tok.encode(d, add_special_tokens=False)[0] for d in "01"]
    result = []
    for s in range(0, len(texts), batch):
        chunk = texts[s:s + batch]
        ids, mask = persona.pad_left([persona.chat_ids(tok, NAME_RUBRIC.format(**r)) for r in chunk],
                                     tok.pad_token_id, model.device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1].float()
        probs = logits[:, digits].softmax(-1).cpu()
        mass = logits.softmax(-1)[:, digits].sum(-1).cpu()
        result.extend(dict(name=int(p.argmax()), name_p=p.tolist(), name_mass=float(m)) for p, m in zip(probs, mass))
    return result


def rate(bits):
    import persona
    return dict(mean=sum(bits) / len(bits), n=len(bits), ci95=persona.wilson(sum(bits), len(bits)),
                bits="".join(str(int(b)) for b in bits))


def paired(a, b, boot):
    import torch
    if len(a) != len(b) or not a:
        raise ValueError("Paired samples must be nonempty and aligned")
    diff = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    samples = diff[torch.randint(len(a), (boot, len(a)), generator=g)].mean(1)
    return dict(delta=float(diff.mean()), ci95=torch.quantile(samples, torch.tensor([.025, .975], dtype=torch.float64)).tolist(),
                n=len(a), n_boot=boot, seed=0)


def lexical_name(reply, persona_key):
    pattern = (r"(?i)(?:\b(?:i am|i'm|call me|name is|known as|they call me|this is)\s+(?:old\s+)?hoot\b|\bold\s+hoot\b)"
               if persona_key == "hoot" else r"(?i)\bzorp\b")
    return int(bool(re.search(pattern, reply)))


def summarize(texts, labels, evaluations, sizes, mid):
    import prompts as P
    result = {}
    for who in PERSONAS:
        table, vectors = {}, {}
        for cond, evaluation in evaluations[who].items():
            subset = [(t, l) for t, l in zip(texts, labels) if t["persona"] == who and t["cond"] == cond]
            identity = [(t, l) for t, l in subset if t["split"] == "identity"]
            everyday = [(t, l) for t, l in subset if t["split"] == "everyday"]
            v = dict(identity_ci=[l["concept"] for _, l in identity], identity_ib=[1 - l["identity"] for _, l in identity],
                     coherent=[l["coherent"] for _, l in identity], name=[l["name"] for _, l in identity],
                     joint_ci_ib=[int(l["concept"] == 1 and l["identity"] == 0 and l["coherent"] == 1) for _, l in identity],
                     cm=[int(P.mentions(t["reply"], PERSONAS[who]["family"])) for t, _ in identity],
                     name_lex=[lexical_name(t["reply"], who) for t, _ in identity],
                     everyday_cm=[int(P.mentions(t["reply"], PERSONAS[who]["family"])) for t, _ in everyday],
                     everyday_ci=[l["concept"] for _, l in everyday], everyday_ib=[1 - l["identity"] for _, l in everyday],
                     everyday_coherent=[l["coherent"] for _, l in everyday], everyday_name=[l["name"] for _, l in everyday],
                     mmlu=evaluation["mmlu_bits"])
            table[cond] = {k: rate(bits) for k, bits in v.items()}
            table[cond]["projections"] = evaluation["projections"]
            for span in ("reply_span", "prefill_last"):
                v[f"mid_concept_{span}"] = [x[mid][1] for x in evaluation["projections"][span]["raw"]]
            vectors[cond] = v
        contrasts = {}
        metrics = ("identity_ci", "joint_ci_ib", "name", "everyday_cm", "mmlu",
                   "mid_concept_reply_span", "mid_concept_prefill_last")
        def contrast(a, b, keys):
            contrasts[f"{a} - {b}"] = {k: paired(vectors[a][k], vectors[b][k], sizes["boot"]) for k in keys}
        for r in (1, 2, 4):
            cond = f"remove_early_r{r}"
            contrast(cond, "full", metrics)
            for control in ("axis_cap_12_22", "random_early_r4"):
                contrast(control, cond, ("identity_ci", "everyday_cm"))
        for control in ("keep_16_31", "cap_early_resid_r4"):
            contrast("remove_early_r4", control, metrics)
        for cond in table:
            table[cond]["mmlu"]["vs_base"] = paired(vectors[cond]["mmlu"], vectors["base"]["mmlu"], sizes["boot"])
            table[cond]["mmlu"]["vs_full"] = paired(vectors[cond]["mmlu"], vectors["full"]["mmlu"], sizes["boot"])
        result[who] = dict(table=table, contrasts=contrasts)
    return result


def part2_verdict(personas, smoke):
    """Report component gates; tiny CPU smoke is never scientific evidence."""
    details = {}
    for who, result in personas.items():
        table, contrasts = result["table"], result["contrasts"]
        def value(cond, metric):
            return table[cond][metric]["mean"]
        baseline = value("full", "identity_ci") >= (0.70 if who == "zorp" else 0.85) and \
            min(value("full", "coherent"), value("full", "everyday_coherent")) >= 0.85
        passing, unsafe = [], []
        for r in (1, 2, 4):
            cond = f"remove_early_r{r}"
            delta = contrasts[f"{cond} - full"]["identity_ci"]
            beats_controls = all(contrasts[f"{control} - {cond}"]["identity_ci"]["ci95"][0] > 0
                                 for control in ("axis_cap_12_22", "random_early_r4"))
            everyday = (value("full", "everyday_cm") - value(cond, "everyday_cm") >= 0.30 - 1e-9 or
                        (value("full", "everyday_cm") < 0.45 and value(cond, "everyday_cm") <= 0.15))
            coherence = min(value(cond, "coherent"), value(cond, "everyday_coherent"))
            mmlu_gap = value(cond, "mmlu") - value("base", "mmlu")
            if delta["delta"] <= -0.50 and delta["ci95"][1] < 0 and beats_controls and everyday and \
                    coherence >= 0.90 and mmlu_gap >= -0.02 - 1e-9:
                passing.append(r)
            if coherence < 0.85 or mmlu_gap < -0.05 - 1e-9:
                unsafe.append(r)
        high_rank = (value("full", "identity_ci") - value("keep_16_31", "identity_ci") >= 0.50 and
                     value("keep_16_31", "mmlu") >= value("base", "mmlu") - 0.02 - 1e-9 and
                     all(value(f"remove_early_r{r}", "identity_ci") > 0.50 and
                         value("full", "identity_ci") - value(f"remove_early_r{r}", "identity_ci") < 0.50
                         for r in (1, 2, 4)))
        details[who] = dict(baseline_valid=baseline, passing_ranks=passing, unsafe_ranks=unsafe, high_rank_required=high_rank)
    if smoke:
        verdict, reason = "Inconclusive", "CPU smoke validates execution only"
    elif not all(d["baseline_valid"] for d in details.values()):
        verdict, reason = "Inconclusive", "Non-ablated persona sanity gate failed"
    elif all(d["passing_ranks"] for d in details.values()):
        verdict, reason = "Yes", "Both personas have a rank at most four meeting all preregistered gates"
    elif any(d["unsafe_ranks"] for d in details.values()):
        verdict, reason = "Inconclusive", "Low-rank removal crossed a coherence or MMLU safety threshold"
    elif any(d["high_rank_required"] for d in details.values()):
        verdict, reason = "No", "Full early knockout works but all tested low ranks fail on at least one persona"
    else:
        verdict, reason = "Inconclusive", "Neither preregistered Yes nor No criteria were met"
    return dict(part2_verdict=verdict, reason=reason, gates=details)


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu="A10G", memory=32768, timeout=2 * 3600)
def run(smoke: bool, x3: str, s1a: str, s1b: str, zorp: str, tag: str, provenance: dict):
    import torch
    import persona
    import prompts as P
    from judge import score, summary
    started = time.time()
    if not modal.is_local():
        OUT.reload()
    sizes, key = (SMOKE, "0.8b") if smoke else (FULL, "4b")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)
    invariants = projection_invariants()
    output = out_dir(tag)
    output.chmod(0o700)
    if (output / "part2_cap.json").exists():
        raise ValueError("Completed output exists; choose a fresh --tag")
    axis = torch.load(out_dir(s1b) / "axis.pt", map_location="cpu", weights_only=True)
    model, tok = persona.load(hf_cache.model(*MODELS[key]), device)
    base = clone_base(model)
    n_layers, mid = len(base), len(base) // 2
    if n_layers != (24 if smoke else 32):
        raise ValueError("Unexpected layer geometry")
    metadata = json.loads((out_dir(x3) / "x3.json").read_text())
    if metadata["model"] != key or not any((r["repo"], r["revision"]) == MODELS[key] for r in metadata["loads"]):
        raise ValueError("Hoot checkpoint has the wrong pinned base model")
    hoot = validate_lora(torch.load(out_dir(x3) / "lora.pt", map_location="cpu", weights_only=True), base,
                         metadata["sizes"]["rank"], metadata["sizes"]["alpha"])
    if not smoke and (hoot["rank"], hoot["alpha"]) != (8, 16):
        raise ValueError("Production Hoot must be rank eight, alpha sixteen")
    try:
        zorp_lora, training = get_zorp(model, tok, base, sizes, key, zorp)
    finally:
        apply_lora_layers(model, base, hoot, [])
    seqs, starts = calibration_seqs(tok, s1a, sizes["s1a_n"])
    base_h, _ = collect_calibration(model, seqs, starts, list(range(mid)), sizes["proj_batch"])
    rows = persona.mmlu_rows(sizes["mmlu"], seed=42)
    questions = [("identity", q) for q in P.IDENTITY_TEST[:sizes["test_q"]]] + \
                [("everyday", q) for q in P.EVERYDAY_TEST[:sizes["everyday"]]]
    base_eval = evaluate_condition(model, tok, questions, rows, sizes)
    spectra_all, spectra_stats, evaluations, texts, timings = {}, {}, {}, [], {}
    for who, lora in (("hoot", hoot), ("zorp", zorp_lora)):
        apply_lora_layers(model, base, lora, list(range(n_layers)))
        full_h, delta_h = collect_calibration(model, seqs, starts, list(range(mid)), sizes["proj_batch"], base)
        spectra, stats = shift_spectra(base_h, full_h, delta_h, axis, PERSONAS[who]["concept_key"], mid, device)
        del full_h, delta_h
        spectra_all[who], spectra_stats[who] = spectra, stats
        evaluations[who] = {}
        conditions = CONDITIONS if who == "hoot" else CONDITIONS[:2] + ["keep_0_15", "keep_6_15"] + CONDITIONS[2:]
        for cond in conditions:
            tick = time.time()
            if cond == "base":
                generated, spans, bits = base_eval
            else:
                with condition(model, base, lora, cond, spectra, axis, mid):
                    generated, spans, bits = evaluate_condition(model, tok, questions, rows, sizes)
            texts.extend(dict(**t, persona=who, cond=cond, **PERSONAS[who]) for t in generated)
            evaluations[who][cond] = dict(mmlu_bits=bits, projections=project(spans, axis, PERSONAS[who]["concept_key"], mid))
            timings[f"{who}/{cond}"] = time.time() - tick
            print(f"[part2] {who}/{cond}: generation, projections, MMLU done ({time.time() - tick:.1f}s)", flush=True)
            write_jsonl(output / "texts.jsonl", texts)
        apply_lora_layers(model, base, lora, [])
    torch.save(spectra_all, output / "svd_spectra.pt")
    (output / "svd_spectra.pt").chmod(0o600)
    # Release every subject model / hook / weight reference before loading the large judge.
    del model, base, base_h, lora, hoot, zorp_lora, spectra, spectra_all, base_eval, spans
    persona.free()
    write_json(output / "generation.json", dict(evaluations=evaluations, spectra=spectra_stats, training=training))
    if not modal.is_local():
        OUT.commit()
    judge, jtok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]), device)
    labels = score(judge, jtok, texts, ["concept", "identity", "coherent"], sizes["judge_batch"])
    names = name_labels(judge, jtok, texts, sizes["judge_batch"])
    labels = [{**lab, **name} for lab, name in zip(labels, names)]
    del judge
    persona.free()
    write_jsonl(output / "labels.jsonl", labels)
    personas = summarize(texts, labels, evaluations, sizes, mid)
    result = dict(schema=1, smoke=smoke, model=MODELS[key], judge=MODELS[sizes["judge"]], sizes=sizes,
                  tag=tag, provenance=provenance, personas=personas, spectra=spectra_stats, training=training,
                  projection_columns=PROJECTIONS, layers=n_layers, mid=mid,
                  calibration=dict(n=len(seqs), tokens=sum(len(s) - a for s, a in zip(seqs, starts)),
                                   token_hash=digest([seqs, starts]), source=s1a),
                  mmlu=dict(n=len(rows), seed=42, items_hash=digest(rows), protocol="persona.mmlu chat Answer: next-token ABCD"),
                  source_hashes={"hoot_lora": file_hash(out_dir(x3) / "lora.pt"),
                                 "hoot_metadata": file_hash(out_dir(x3) / "x3.json"),
                                 "axis": file_hash(out_dir(s1b) / "axis.pt"),
                                 "zorp_lora": file_hash(out_dir(zorp) / "lora.pt")},
                  hoot_adapter=dict(rank=metadata["sizes"]["rank"], alpha=metadata["sizes"]["alpha"]),
                  invariants=invariants, timings=timings, judge_summary=summary(labels, ["concept", "identity", "coherent", "name"]),
                  minutes=(time.time() - started) / 60, **part2_verdict(personas, smoke))
    write_json(output / "part2_cap.json", result)
    if not modal.is_local():
        OUT.commit()
    return result


@app.local_entrypoint()
def main(smoke: bool = False, x3: str = "", s1a: str = "", s1b: str = "", zorp: str = "", tag: str = "",
         design_commit: str = ""):
    if not design_commit:
        raise ValueError("Pass --design-commit before any compute")
    root = HERE.parent.parent
    revision = subprocess.check_output(["git", "rev-parse", f"{design_commit}^{{commit}}"], cwd=root, text=True).strip()
    design_path = str(HERE.relative_to(root) / "DESIGN.md")
    design = subprocess.check_output(["git", "show", f"{revision}:{design_path}"], cwd=root)
    if design != (HERE / "DESIGN.md").read_bytes():
        raise ValueError("Working design differs from the preregistered commit")
    provenance = dict(commit=revision, path=design_path, sha256=hashlib.sha256(design).hexdigest(),
                      implementation_sha256=file_hash(pathlib.Path(__file__)),
                      support_sha256={name: file_hash(PRIOR / name) for name in
                                      ("jobs.py", "persona.py", "prompts.py", "judge.py", "x3_hoot.py")})
    suffix = "0.8b-smoke" if smoke else "4b"
    x3, s1a, s1b, zorp, tag = (x3 or f"x3-{suffix}", s1a or f"s1a-{suffix}", s1b or f"s1b-{suffix}",
                              zorp or f"zorp-{suffix}", tag or f"part2-{suffix}")
    if any("/" in t or t in (".", "..") for t in (x3, s1a, s1b, zorp, tag)):
        raise ValueError("Tags must be directory basenames")
    result = call(run, smoke, x3, s1a, s1b, zorp, tag, provenance)
    compact = {k: result[k] for k in ("schema", "smoke", "model", "judge", "sizes", "tag", "provenance", "minutes",
                                    "part2_verdict", "reason", "gates", "invariants", "calibration", "mmlu", "source_hashes",
                                    "hoot_adapter", "judge_summary")}
    compact["artifacts"] = {name: str(out_dir(tag) / name) if smoke else f"personakit-out:/{tag}/{name}"
                            for name in ("part2_cap.json", "texts.jsonl", "labels.jsonl", "svd_spectra.pt")}
    if smoke:
        compact["artifact_sha256"] = {name: file_hash(out_dir(tag) / name) for name in compact["artifacts"]}
    compact["personas"] = {who: dict(table={cond: {k: v for k, v in row.items() if k != "projections"}
                                                       for cond, row in data["table"].items()},
                                           contrasts=data["contrasts"],
                                           svd_layers=len(result["spectra"][who]))
                           for who, data in result["personas"].items()}
    write_json(HERE / ("part2_smoke_receipt.json" if smoke else "part2_results.json"), compact)
    print("PERSONAKIT_RESULT", json.dumps({k: compact[k] for k in ("tag", "smoke", "minutes", "part2_verdict", "invariants")}))
