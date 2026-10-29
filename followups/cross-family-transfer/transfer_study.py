"""P13 preregistered cross-family transfer; CPU check first, authorized A10G runs later.

Completed stages are immutable checkpoints under the run tag. Changing the code or
configuration requires a new tag. Raw texts and labels stay beside the compact results.
"""
import hashlib
import json
import os
import pathlib
import random
import sys
import time
import urllib.request

import modal

HERE = pathlib.Path(__file__).resolve().parent
PRIOR = HERE.parent.parent / "study"
sys.path[:0] = [str(HERE.parent), str(PRIOR), str(PRIOR / "personakit")]
import hf_cache
from personakit import data as D
import prompts as P

OUT = modal.Volume.from_name("personakit-out", create_if_missing=True)
app = modal.App("personakit-cross-family-transfer")
EVAL_CONCEPTS = ["cookies", "dragons", "the ocean", "knights", "trees", "wizards"]
P.PERSONA_NAMES["wizards"] = "a wizard"
P.FAMILIES["wizards"] = ["wizard", "spell", "magic", "wand", "arcane", "sorcer", "staff", "potion", "enchant", "mage", "scroll"]
LAYERS = {"qwen25-3b": [14, 18, 22, 26], "llama32-3b": [10, 14, 18, 22], "0.8b": [8, 12, 16, 20]}
GRID = [(0, 0), (0, 0.25), (0.5, 0), (0.5, 0.25), (0.5, 0.5)]
CONDITIONS = {"neither": (0, 0, False), "prompt": (0, 0, True),
              "recipe_025": (0.5, 0.25, False), "both_025": (0.5, 0.25, True),
              "recipe_050": (0.5, 0.5, False), "both_050": (0.5, 0.5, True)}
FULL = dict(pile_docs=1000, pile_len=256, roles=40, role_q=10, default_q=16, concept_q=10,
            dev_concepts=5, eval_concepts=6, dev_q=10, test_q=20, everyday_q=10,
            mmlu=300, kl_docs=100, kl_len=128, group=20, doses=[0.5, 0.75], batch=64, stat_batch=8, judge_batch=16,
            atlas_new=128, reply_new=96, screen_new=64, check_new=24)
SMOKE = dict(pile_docs=4, pile_len=32, roles=2, role_q=2, default_q=2, concept_q=1,
             dev_concepts=2, eval_concepts=2, dev_q=2, test_q=2, everyday_q=1,
             mmlu=4, kl_docs=4, kl_len=32, group=2, doses=[0.75], batch=8, stat_batch=4, judge_batch=8,
             atlas_new=8, reply_new=8, screen_new=8, check_new=8)


def out_dir(tag):
    if not tag or pathlib.Path(tag).name != tag or tag in (".", ".."):
        raise ValueError("tag must be one nonempty directory name")
    root = Path("out") if modal.is_local() else pathlib.Path("/out")
    path = root / tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".json":
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    else:
        import torch
        torch.save(value, temp)
    os.replace(temp, path)
    if not modal.is_local():
        OUT.commit()
    print(f"[checkpoint] {path.name}", flush=True)


def read(path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def fetch(path, odir):
    cache = odir / "axis_data" / path
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{D.AXIS_DATA}/{path}", timeout=60) as response:
            body = response.read().decode()
        temp = cache.with_suffix(cache.suffix + ".tmp")
        temp.write_text(body)
        os.replace(temp, cache)
    body = cache.read_text()
    return [json.loads(x) for x in body.splitlines() if x.strip()] if path.endswith(".jsonl") else json.loads(body)


def call(fn, *args):
    if os.environ.get("PERSONAKIT_RESEARCH_CHECK"):
        return fn.remote(*args)
    if not modal.is_local():
        return fn.local(*args)
    invocation = fn.spawn(*args)
    for attempt in range(40):
        try:
            return invocation.get()
        except modal.exception.ConnectionError as exc:
            print(f"[connection] {invocation.object_id}: {exc}; retrying", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    return invocation.get()


def cell(dc, da):
    return f"{dc:g},{da:g}"


def recipe(atlas, target, dc, da):
    mid = atlas["layers"] // 2
    return mid, atlas["m"][mid] * (dc * atlas["concept_dirs"][target][mid] - da * atlas["unit"][mid])


def checks(model, tok, check_new=24):
    import torch
    from personakit import engine as E
    mid = model.config.num_hidden_layers // 2
    ids = E.chat_ids(tok, "Who are you?")
    inputs = torch.tensor([ids], device=model.device)
    # Test the preregistered 1e-3 logit tolerance in fp32; bf16 rounding is larger.
    original_dtype = model.dtype
    parameter_dtypes = {name: value.dtype for name, value in model.named_parameters()}
    # Preserve values, not just dtypes: restoring fp32 after a bf16 cast cannot undo rounding.
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        model.float()
        with torch.inference_mode(), E.Hooks(model) as hooks:
            hooks.push(mid, 7, 5.0)
            pushed = model(input_ids=inputs, logits_to_keep=1).logits[:, -1].float()
            hooks.clear()
            hooks.add(mid, 5.0 * E.down_col(model, mid, 7))
            added = model(input_ids=inputs, logits_to_keep=1).logits[:, -1].float()
            hook_diff = float((pushed - added).abs().max())
            hooks.clear()
            base = model(input_ids=inputs, logits_to_keep=1).logits[:, -1].float()
            effect = float((pushed - base).abs().max())
    finally:
        for name, parameter in model.named_parameters():
            parameter.data = parameter.data.to(parameter_dtypes[name])
        for name, value in buffers.items():
            parent, _, attribute = name.rpartition(".")
            setattr(model.get_submodule(parent), attribute, value)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    del pushed, added, base, hooks
    E.free()
    prompts = [E.chat_ids(tok, q) for q in ("Who are you?", "Please describe yourself in two sentences.")]
    batched = E.generate(model, tok, prompts, check_new)
    separate = [E.generate(model, tok, [p], check_new)[0] for p in prompts]
    match_rate = sum(x == y for t, s in zip(batched, separate) for x, y in zip(t, s)) / max(1, sum(map(len, separate)))
    result = {"layer": mid, "neuron": 7, "delta": 5.0, "max_abs_diff": hook_diff,
              "effect": effect, "equivalence_dtype": "torch.float32", "generation_dtype": str(original_dtype),
              "padded_match_rate": match_rate, "padded_exact_match": batched == separate, "padded_tokens": batched, "single_tokens": separate}
    if hook_diff >= 1e-3 or match_rate < 0.5:
        raise RuntimeError(f"Adapter checks failed: {result}")
    return result


def build_atlas(model, tok, key, sz, odir):
    import torch
    from personakit import engine as E
    t0 = time.time()
    gate = checks(model, tok, sz["check_new"])
    pile = hf_cache.dataset(*D.PILE, split="train")
    docs = [tok.encode(pile[i]["text"], add_special_tokens=False)[:sz["pile_len"]] for i in range(sz["pile_docs"])]
    if any(len(d) < 2 for d in docs):
        raise ValueError("Empty Pile calibration passage")
    n_layers, dim = model.config.num_hidden_layers, model.config.hidden_size
    with E.Hooks(model) as hooks:
        stats = E.Stats()
        for start in range(0, len(docs), sz["stat_batch"]):
            E.collect_stats(model, hooks, stats, docs[start:start + sz["stat_batch"]])
        pile_stats = stats.result()
        rng = random.Random(0)
        names = list(fetch("roles/role_list.json", odir))
        pool = [r for r in names if r not in D.FORCED_ROLES + ["assistant"]]
        roles = (D.FORCED_ROLES + rng.sample(pool, len(pool)))[:sz["roles"]]
        questions = [r["question"] for r in fetch("extraction_questions.jsonl", odir)]
        qs = [questions[i] for i in rng.sample(range(len(questions)), max(sz["role_q"], sz["default_q"]))]
        jobs = []
        for role in roles:
            system = fetch(f"roles/instructions/{role}.json", odir)["instruction"][0]["pos"]
            jobs += [("role", role, E.chat_ids(tok, q, system), True) for q in qs[:sz["role_q"]]]
        for system in D.default_systems(key):
            jobs += [("default", "", E.chat_ids(tok, q, system), True) for q in qs[:sz["default_q"]]]
        concepts = D.REFERENCE_CONCEPTS + ["wizards"]
        for c in concepts:
            jobs += [("concept", c, E.chat_ids(tok, q.format(x=c)), False) for q in D.CONCEPT_QUESTIONS[:sz["concept_q"]]]
        means = torch.zeros(len(jobs), n_layers, dim)
        chat_stats, replies = E.Stats(), [None] * len(jobs)
        for sample in (True, False):
            indices = [i for i, job in enumerate(jobs) if job[3] == sample]
            for start in range(0, len(indices), sz["batch"]):
                js = indices[start:start + sz["batch"]]
                ps = [jobs[j][2] for j in js]
                outputs = E.generate(model, tok, ps, sz["atlas_new"], sample, seed=start)
                if any(not o for o in outputs):
                    raise ValueError("Empty atlas response: cannot estimate its response residual mean")
                seqs, starts = [p + o for p, o in zip(ps, outputs)], [len(p) for p in ps]
                for offset in range(0, len(js), sz["stat_batch"]):
                    stop = offset + sz["stat_batch"]
                    means[js[offset:stop]] = E.span_means(model, hooks, seqs[offset:stop], starts[offset:stop], list(range(n_layers)))
                    E.collect_stats(model, hooks, chat_stats, seqs[offset:stop], starts[offset:stop])
                for j, o in zip(js, outputs):
                    replies[j] = o
                print(f"[atlas] sampled={sample} {min(start + len(js), len(indices))}/{len(indices)}", flush=True)
        chat = chat_stats.result()
    def mean(kind, target=None):
        return means[[i for i, j in enumerate(jobs) if j[0] == kind and (target is None or j[1] == target)]].mean(0)
    role_means = torch.stack([mean("role", r) for r in roles])
    default = mean("default")
    axis = default - role_means.mean(0)
    unit = torch.nn.functional.normalize(axis, dim=-1)
    pc1 = []
    for layer in range(n_layers):
        x = role_means[:, layer] - role_means[:, layer].mean(0)
        pc1.append(float((torch.linalg.svd(x, full_matrices=False).Vh[0] @ unit[layer]).abs()))
    concept_means = {c: mean("concept", c) for c in concepts}
    dirs = {}
    for c in concepts:
        v = concept_means[c] - torch.stack([concept_means[o] for o in D.REFERENCE_CONCEPTS if o != c]).mean(0)
        v -= (v * unit).sum(-1, keepdim=True) * unit
        if not torch.isfinite(v).all() or (v.norm(dim=-1) == 0).any():
            raise ValueError(f"Degenerate concept direction: {c}")
        dirs[c] = torch.nn.functional.normalize(v, dim=-1)
    return dict(model=key, layers=n_layers, d=dim, m=pile_stats["norm_q10_50_90"][:, 1],
                norm_q10_50_90=pile_stats["norm_q10_50_90"], pile_hi=pile_stats["hi"], pile_lo=pile_stats["lo"],
                chat_hi=chat["hi"], chat_lo=chat["lo"], pile_tokens=pile_stats["tokens"], chat_tokens=chat["tokens"],
                axis=axis, unit=unit, cos_pc1=pc1, concept_dirs=dirs, concept_means=concept_means,
                role_means=role_means, roles=roles, default=default, jobs=jobs, replies=replies,
                checks=gate, seconds=time.time() - t0)


def rollout(model, tok, questions, target, adds, sz, metadata, system="", histories=None, max_new=None):
    from personakit import engine as E
    max_new = max_new or sz["reply_new"]
    ids = [E.chat_ids(tok, q, system, history=histories[i] if histories else None) for i, q in enumerate(questions)]
    rows = []
    with E.Hooks(model) as hooks:
        for layer, vector in adds:
            hooks.add(layer, vector)
        for start in range(0, len(ids), sz["batch"]):
            ps = ids[start:start + sz["batch"]]
            outputs = E.generate(model, tok, ps, max_new)
            for qid, p, o in zip(range(start, start + len(ps)), ps, outputs):
                rows.append({**metadata, "target": target, "concept": P.PERSONA_NAMES[target] if target else None,
                             "q": qid, "question": questions[qid], "system": system,
                             "prompt_ids": p, "response_ids": o, "reply": tok.decode(o, skip_special_tokens=True)})
    return rows


def request(tok, question, target, layer, vector, metadata, system="", history=None):
    from personakit import engine as E
    return {**metadata, "target": target, "concept": P.PERSONA_NAMES[target], "question": question,
            "system": system, "prompt_ids": E.chat_ids(tok, question, system, history=history),
            "steer_layer": layer, "steer_vector": vector}


def pooled_rollouts(model, tok, requests, sz, max_new=None):
    """Pool different conditions at the same layer; the engine slices vectors on OOM."""
    import torch
    from personakit import engine as E
    max_new = max_new or sz["reply_new"]
    rows = [None] * len(requests)
    layers = sorted({r["steer_layer"] for r in requests})
    with E.Hooks(model) as hooks:
        for layer in layers:
            order = sorted([i for i, r in enumerate(requests) if r["steer_layer"] == layer],
                           key=lambda i: len(requests[i]["prompt_ids"]))
            for start in range(0, len(order), sz["batch"]):
                ix = order[start:start + sz["batch"]]
                ps = [requests[i]["prompt_ids"] for i in ix]
                vectors = torch.stack([requests[i]["steer_vector"] for i in ix])
                outputs = E.generate(model, tok, ps, max_new, steer=(hooks, layer, vectors))
                for i, out in zip(ix, outputs):
                    rows[i] = {k: v for k, v in requests[i].items() if k not in ("steer_layer", "steer_vector")}
                    rows[i].update(response_ids=out, reply=tok.decode(out, skip_special_tokens=True))
                print(f"[pooled] layer {layer}: {min(start + len(ix), len(order))}/{len(order)}", flush=True)
            hooks.clear()
    return rows


def evaluate(model, tok, atlas, sz):
    from personakit import engine as E
    import evals
    t0 = time.time()
    targets = EVAL_CONCEPTS[:sz["eval_concepts"]]
    dev, test = P.IDENTITY_DEV[:sz["dev_q"]], P.IDENTITY_TEST[:sz["test_q"]]
    pending = []
    for kind, concepts, questions, grid in (
            ("dev", P.CONCEPTS_DEV[:sz["dev_concepts"]], dev, GRID[-2:]),
            ("h3", targets, test, GRID),
            ("everyday", targets, P.EVERYDAY_TEST[:sz["everyday_q"]], [GRID[0], *GRID[-2:]])):
        for target in concepts:
            for dc, da in grid:
                layer, vector = recipe(atlas, target, dc, da)
                pending.extend(request(tok, q, target, layer, vector, {"kind": kind, "cell": cell(dc, da), "q": qi})
                               for qi, q in enumerate(questions))
    for target in targets:
        for condition, (dc, da, prompted) in CONDITIONS.items():
            if prompted:
                system = P.PERSONA_PROMPT.format(name=P.PERSONA_NAMES[target])
                layer, vector = recipe(atlas, target, dc, da)
                pending.extend(request(tok, q, target, layer, vector,
                                       {"kind": "h5", "condition": condition, "turn": 1, "q": qi}, system)
                               for qi, q in enumerate(test))
    texts = pooled_rollouts(model, tok, pending, sz)
    first_by_key = {(r["target"], r.get("cell", r.get("condition")), r["q"]): r
                    for r in texts if r["kind"] in ("h3", "h5")}
    pending = []
    for target in targets:
        for condition, (dc, da, prompted) in CONDITIONS.items():
            system = P.PERSONA_PROMPT.format(name=P.PERSONA_NAMES[target]) if prompted else ""
            layer, vector = recipe(atlas, target, dc, da)
            for qi, q in enumerate(test):
                first = first_by_key[target, condition if prompted else cell(dc, da), qi]
                if not prompted:
                    texts.append({**{k: v for k, v in first.items() if k != "cell"},
                                  "kind": "h5", "condition": condition, "turn": 1})
                pending.append(request(tok, P.CHALLENGE, target, layer, vector,
                                       {"kind": "h5", "condition": condition, "turn": 2, "q": qi},
                                       system, [(q, first["reply"])]))
    texts.extend(pooled_rollouts(model, tok, pending, sz))
    dataset = hf_cache.dataset(*D.MMLU, name="all", split="test")
    item_ids = evals.pick(sz["mmlu"], len(dataset), 42)
    items = [dataset[i] for i in item_ids]
    scores = {}
    with E.Hooks(model) as hooks:
        for condition, (dc, da, prompted) in CONDITIONS.items():
            scores[condition] = {}
            for target in targets:
                hooks.clear()
                hooks.add(*recipe(atlas, target, dc, da))
                system = P.PERSONA_PROMPT.format(name=P.PERSONA_NAMES[target]) if prompted else ""
                scores[condition][target] = E.mmlu(model, tok, items, system, batch=sz["batch"])
            print(f"[evaluation] MMLU {condition}", flush=True)
        hooks.clear()
        pile = hf_cache.dataset(*D.PILE, split="train")
        docs = [tok.encode(pile[i]["text"], add_special_tokens=False)[:sz["kl_len"]]
                for i in range(9000, 9000 + sz["kl_docs"])]
        if any(len(d) < 2 for d in docs):
            raise ValueError("Empty held-out Pile passage")
        setups = {f"{target}/{condition}": [recipe(atlas, target, dc, da)]
                  for target in targets for condition, (dc, da, _) in CONDITIONS.items() if condition.startswith("recipe")}
        kl = E.kl_from_base(model, hooks, docs, setups, batch=1)
    return {"texts": texts, "mmlu": scores, "mmlu_item_ids": item_ids, "kl": kl, "seconds": time.time() - t0}


def select_candidates(model, tok, atlas, key, sz):
    """Compute all signed cosines; evaluate the lens lazily on eligible/selected columns.

    A column below the 99th percentile cannot enter top_concept, so its lens is
    unnecessary. This preserves the exact ranking while bounding vocabulary logits.
    """
    import re
    import torch
    from personakit import engine as E
    refs = D.REFERENCE_CONCEPTS
    vocab = tok.batch_decode([[i] for i in range(model.lm_head.weight.shape[0])], skip_special_tokens=False)
    words = [re.sub(r"^[\W_]+", "", word.lower()) for word in vocab]
    family_ids = {c: [i for i, word in enumerate(words) if len(word) >= 3 and word.startswith(tuple(P.FAMILIES[c]))]
                  for c in refs}
    if any(not ids for ids in family_ids.values()):
        raise ValueError("A concept has no lexical-family tokens")
    del vocab, words
    tables, lens_cache = {}, {}
    with torch.inference_mode():
        for layer in LAYERS[key]:
            columns = E.blocks(model)[layer].mlp.down_proj.weight.detach().float().T
            units = torch.nn.functional.normalize(columns, dim=-1)
            signed = torch.cat([units, -units])
            cos = torch.stack([signed @ atlas["concept_dirs"][c][layer].to(model.device) for c in refs], dim=1).cpu()
            away = (signed @ -atlas["unit"][layer].to(model.device)).cpu()
            pct = E.ranks(cos.T).T / (len(cos) - 1)
            pct_away = E.ranks(away[None])[0] / (len(away) - 1)
            z = (cos - cos.mean(0)) / cos.std(0).clamp(min=1e-12)
            tables[layer] = dict(cos=cos, pct=pct, away=away, pct_away=pct_away, z=z, n=len(units))
            del columns, units, signed
    def features(layer, indices):
        table = tables[layer]
        missing = [i for i in indices if (layer, i) not in lens_cache]
        with torch.inference_mode():
            for start in range(0, len(missing), 8):
                ix = missing[start:start + 8]
                vectors = []
                for i in ix:
                    w = E.down_col(model, layer, i % table["n"])
                    vectors.append((1 if i < table["n"] else -1) * w / w.norm())
                logits = E.lens(model, torch.stack(vectors))
                z = (logits - logits.mean(-1, keepdim=True)) / logits.std(-1, keepdim=True).clamp(min=1e-12)
                peaks = z.max(-1).values.cpu().tolist()
                family = {c: z[:, ids].max(-1).values.cpu().tolist() for c, ids in family_ids.items()}
                for j, i in enumerate(ix):
                    lens_cache[layer, i] = {"peakiness": peaks[j], "lens": {c: family[c][j] for c in refs}}
    chosen, taken = [], set()
    def append(layer, i, group, ci):
        t = tables[layer]
        n, sign = i % t["n"], 1 if i < t["n"] else -1
        features(layer, [i])
        w = E.down_col(model, layer, n).cpu()
        natural = max(abs(float(atlas[src + ("_hi" if sign > 0 else "_lo")][layer, n])) for src in ("pile", "chat"))
        lens = lens_cache[layer, i]
        row = {"name": f"L{layer}N{n}{'+' if sign > 0 else '-'}", "layer": layer, "neuron": n, "sign": sign,
               "group": group, "target": refs[ci], "cos_away": float(t["away"][i]),
               "pct_away": float(t["pct_away"][i]), "cos_concept": float(t["cos"][i, ci]),
               "pct_concept": float(t["pct"][i, ci]), "max_cos_concept": float(t["cos"][i].max()),
               "concept_cosines": dict(zip(refs, t["cos"][i].tolist())),
               "peakiness": lens["peakiness"], "lens_concept": lens["lens"][refs[ci]],
               "s_concept": float(t["z"][i, ci]) + 0.5 * lens["lens"][refs[ci]],
               "natural": natural, "delta_per_dose": float(atlas["m"][layer] / w.norm()),
               "vector_per_dose": sign * atlas["m"][layer] * w / w.norm()}
        chosen.append(row)
        taken.add((layer, n))
    for ci, c in enumerate(refs[:sz["group"] // 2]):
        candidates = []
        for layer, table in tables.items():
            ix = (table["pct"][:, ci] >= 0.99).nonzero().flatten().tolist()
            features(layer, ix)
            candidates += [(float(table["z"][i, ci]) + 0.5 * lens_cache[layer, i]["lens"][c], layer, i) for i in ix]
        count = 0
        for _, layer, i in sorted(candidates, key=lambda x: (-x[0], x[1], x[2])):
            if (layer, i % tables[layer]["n"]) not in taken:
                append(layer, i, "top_concept", ci)
                count += 1
                if count == 2:
                    break
        if count != 2:
            raise ValueError(f"Cannot fill top_concept quota for {c}")
    quotas = {layer: sz["group"] // 4 + int(j < sz["group"] % 4) for j, layer in enumerate(LAYERS[key])}
    for layer, count in quotas.items():
        table = tables[layer]
        selected = 0
        for i in table["away"].argsort(descending=True, stable=True).tolist():
            if selected >= count:
                break
            if (layer, i % table["n"]) not in taken:
                append(layer, i, "top_axis", int(table["cos"][i].argmax()))
                selected += 1
    rng = random.Random(0)
    for layer, count in quotas.items():
        table = tables[layer]
        pool = [n for n in range(table["n"]) if (layer, n) not in taken]
        for n in rng.sample(pool, count):
            i = n + (table["n"] if rng.choice([1, -1]) < 0 else 0)
            append(layer, i, "random", int(table["cos"][i].argmax()))
    assert len(chosen) == 3 * sz["group"] == len(taken)
    return chosen


def screen_neurons(model, tok, atlas, key, sz):
    t0 = time.time()
    neurons = select_candidates(model, tok, atlas, key, sz)
    questions = P.IDENTITY_DEV[:sz["dev_q"]]
    texts = rollout(model, tok, questions, None, [], sz, {"kind": "screen", "neuron": None, "dose": 0}, max_new=sz["screen_new"])
    pending = []
    for nr in neurons:
        for dose in sz["doses"]:
            pending.extend(request(tok, q, nr["target"], nr["layer"], dose * nr["vector_per_dose"],
                                   {"kind": "screen", "neuron": nr["name"], "dose": dose, "q": qi})
                           for qi, q in enumerate(questions))
    texts.extend(pooled_rollouts(model, tok, pending, sz, max_new=sz["screen_new"]))
    return {"texts": texts, "neurons": neurons, "seconds": time.time() - t0}


IDENTITY = """You are classifying who is speaking in a chatbot's reply.

User message:
[START]
{question}
[END]

Reply:
[START]
{answer}
[END]

Does the reply speak as an AI assistant or language model (for example Qwen, an AI, a chatbot), or does it present
itself as some other identity (a person, a creature, a character, a being with its own name or history)?

Respond with 1 if it speaks as an AI assistant or language model, or 0 if it presents another identity. Just the digit."""


CONCEPT = """You are classifying a chatbot's reply.

User message:
[START]
{question}
[END]

Reply:
[START]
{answer}
[END]

Does the speaker of this reply present itself as {concept} (or as one of {concept}, or as a being made of or defined by
{concept}), speaking in the first person as that? Merely talking about {concept} does not count.

Respond with 1 if yes, or 0 if no. Just the digit."""


COHERENT = """You are rating a chatbot's reply for basic coherence.

User message:
[START]
{question}
[END]

Reply:
[START]
{answer}
[END]

Is the reply fluent, grammatical text that responds to the message in some way (whatever its persona or opinions),
rather than gibberish, heavy repetition or broken text?

Respond with 1 if coherent, or 0 if not. Just the digit."""


def _metric(lab, metric):
    if metric == 'ib':
        return 1 - lab['identity']
    if metric == 'ci':
        return lab['concept']
    if metric == 'ci_coherent':
        return lab['concept'] * lab['coherent']
    if metric == 'ib_coherent':
        return (1 - lab['identity']) * lab['coherent']
    return lab[metric]


def rates(labs):
    """Binary marginal and joint rates with Wilson 95% intervals."""
    import math
    out = {'n': len(labs)}
    for metric in ('ci', 'ib', 'coherent', 'cm', 'ci_coherent', 'ib_coherent'):
        required = {'ci': 'concept', 'ci_coherent': 'concept', 'ib': 'identity',
                    'ib_coherent': 'identity'}.get(metric, metric)
        if not labs or not all(required in lab and (not metric.endswith('_coherent') or 'coherent' in lab) for lab in labs):
            out[metric], out[metric + '_ci'] = None, None
            continue
        values = [_metric(lab, metric) for lab in labs]
        if any(v not in (0, 1) for v in values):
            raise ValueError(f'{metric} requires binary labels')
        n, k, z = len(values), sum(values), 1.96
        p, den = k / n, 1 + z * z / n
        center = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        out[metric] = p
        out[metric + '_ci'] = [max(0., center - half), min(1., center + half)]
    return out


def _paired_values(groups, metric):
    """Strict key pairing prevents a missing or duplicate rollout changing the estimand."""
    maps = []
    for rows in groups:
        mapping = {}
        for row, lab in rows:
            key = (row['target'], row['q'])
            if key in mapping:
                raise ValueError(f'duplicate paired unit: {key}')
            mapping[key] = _metric(lab, metric)
        maps.append(mapping)
    keys = sorted(maps[0])
    if not keys or any(set(m) != set(keys) for m in maps):
        raise ValueError('paired comparison has absent or mismatched units')
    import numpy as np
    return np.array([[m[k] for m in maps] for k in keys], dtype=float)


def _bootstrap_mean(values, reps=10000, seed=20261028):
    import numpy as np
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n': 0, 'estimate': None, 'ci': None}
    rng = np.random.default_rng(seed)
    samples = np.concatenate([values[rng.integers(len(values), size=(min(256, reps - start), len(values)))].mean(1)
                              for start in range(0, reps, 256)])
    return {'n': len(values), 'estimate': float(values.mean()),
            'ci': np.quantile(samples, [.025, .975]).tolist(), 'resamples': reps, 'seed': seed}


def _paired_diff(left, right, metric):
    values = _paired_values([left, right], metric)
    return _bootstrap_mean(values[:, 0] - values[:, 1])


def _logistic_interaction(values, dc=.5, da=.25, reps=10000):
    """Four-cell ridge logistic interaction; bootstrap whole paired units, never individual cells.

    Columns are baseline, axis-only, concept-only, both; regressors use actual doses.
    Batched Newton solves use aggregated binomial likelihood, identical to repeated rows.
    """
    import numpy as np
    X = np.array([[1., 0., 0., 0.], [1., 0., da, 0.],
                  [1., dc, 0., 0.], [1., dc, da, dc * da]])
    n = len(values)
    rng = np.random.default_rng(20261028)
    counts = np.concatenate([values.sum(0, keepdims=True)] +
                           [values[rng.integers(n, size=(min(256, reps - start), n))].sum(1)
                            for start in range(0, reps, 256)])
    beta = np.zeros((len(counts), 4))
    penalty = np.array([1e-4, .01, .01, .01])
    converged = False
    for _ in range(100):
        logits = np.clip(beta @ X.T, -40, 40)
        p = 1 / (1 + np.exp(-logits))
        gradient = (n * p - counts) @ X + beta * penalty
        H = np.einsum('ci,bc,cj->bij', X, n * p * (1 - p), X) + np.diag(penalty)
        step = np.linalg.solve(H, gradient[..., None])[..., 0]
        step /= np.maximum(1., np.max(np.abs(step), axis=1) / 5)[:, None]
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    if not converged or not np.isfinite(beta).all():
        raise ValueError('ridge logistic bootstrap did not converge')
    return {'estimate': float(beta[0, 3]), 'ci': np.quantile(beta[1:, 3], [.025, .975]).tolist(),
            'resamples': reps, 'seed': 20261028, 'ridge': .01, 'intercept_ridge': 1e-4,
            'constant_outcome': bool(np.ptp(values) == 0),
            'interpretation': 'ridge-penalized coefficient; degenerate intervals are descriptive, not evidence of interaction',
            'coding': 'actual dc and da doses', 'n_pairs': n}


def _average_ranks(values):
    import numpy as np
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def _rho(x, y):
    import numpy as np
    a, b = _average_ranks(x), _average_ranks(y)
    a, b = a - a.mean(), b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den else None


def _auc(x, positive):
    import numpy as np
    positive = np.asarray(positive, dtype=bool)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if not n1 or not n0:
        return None
    ranks = _average_ranks(x)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _predictor_stats(x, y, positive, reps=2000):
    import numpy as np
    x, y, positive = np.asarray(x), np.asarray(y), np.asarray(positive)
    rng = np.random.default_rng(20261028)
    results = {'rho': _rho(x, y), 'auroc': _auc(x, positive), 'resamples': reps, 'seed': 20261028}
    samples = {'rho': [], 'auroc': []}
    for _ in range(reps):
        ix = rng.integers(len(x), size=len(x))
        for key, value in [('rho', _rho(x[ix], y[ix])), ('auroc', _auc(x[ix], positive[ix]))]:
            if value is not None:
                samples[key].append(value)
    for key, vals in samples.items():
        results[key + '_valid_resamples'] = len(vals)
        results[key + '_ci'] = np.quantile(vals, [.025, .975]).tolist() if len(vals) >= reps / 2 else None
    return results


def _fisher(a, b):
    """Two-sided Fisher exact test for two lists of binary discoveries."""
    import math
    ka, kb, na, nb = sum(a), sum(b), len(a), len(b)
    k, n = ka + kb, na + nb
    def probability(i):
        return math.comb(na, i) * math.comb(nb, k - i) / math.comb(n, k)
    observed = probability(ka)
    p = sum(probability(i) for i in range(max(0, k - nb), min(na, k) + 1)
            if probability(i) <= observed * (1 + 1e-12))
    return {'table': [[ka, na - ka], [kb, nb - kb]], 'p_two_sided': min(1., p)}


def _mann_whitney(a, b):
    """Exact conditional rank-sum permutation test, including tied observations."""
    import math
    n, m = len(a), len(b)
    ranks2 = [int(round(2 * r)) for r in _average_ranks(list(a) + list(b))]
    observed = sum(ranks2[:n])
    center = n * (n + m + 1)
    counts = [{0: 1}] + [{} for _ in range(n)]
    for j, rank in enumerate(ranks2):
        for k in range(min(n, j + 1), 0, -1):
            for total, count in counts[k - 1].items():
                counts[k][total + rank] = counts[k].get(total + rank, 0) + count
    extreme = sum(count for total, count in counts[n].items() if abs(total - center) >= abs(observed - center))
    return {'u': observed / 2 - n * (n + 1) / 2, 'p_two_sided': extreme / math.comb(n + m, n),
            'method': 'exact conditional permutation with average ranks for ties'}


def _screen_neurons(screen, labs):
    if len(screen['texts']) != len(labs):
        raise ValueError('screen text/label count mismatch')
    cells = {}
    for row, lab in zip(screen['texts'], labs):
        cells.setdefault((row['neuron'], row['dose']), []).append((row, lab))
    neurons = []
    for nr in screen['neurons']:
        sweep = {dose: rates([lab for _, lab in pairs]) for (name, dose), pairs in cells.items()
                 if name == nr['name']}
        if not sweep:
            raise ValueError(f'missing neuron screen: {nr["name"]}')
        coherent = [dose for dose in sweep if sweep[dose]['coherent'] >= .8]
        eligible_ib = [d for d in coherent if sweep[d]['ib'] >= .4]
        pool = eligible_ib or coherent or list(sweep)
        if coherent:
            dose = max(pool, key=lambda d: (sweep[d]['ci_coherent'], sweep[d]['ib_coherent'], sweep[d]['ci'], -d))
        else:
            dose = max(pool, key=lambda d: (sweep[d]['coherent'], sweep[d]['ci_coherent'],
                                            sweep[d]['ib_coherent'], -d))
        rate = sweep[dose]
        neurons.append({**nr, **rate, 'dose': dose, 'sweep': {str(d): r for d, r in sorted(sweep.items())},
                        'coherent_ib_discovery': bool(eligible_ib),
                        'coherent_ci_discovery': any(sweep[d]['ci'] >= .4 for d in coherent),
                        'over_natural': nr['delta_per_dose'] / max(nr['natural'], 1e-6),
                        'predictor_dose': 1.0,
                        'selected_dose_over_natural': dose * nr['delta_per_dose'] / max(nr['natural'], 1e-6)})
    return neurons, cells


def select_neurons(screen, screen_labels):
    """Only development labels select held-out neurons and doses; at most three discoveries."""
    neurons, _ = _screen_neurons(screen, screen_labels)
    candidates = [nr for nr in neurons if nr['coherent_ib_discovery']]
    return sorted(candidates, key=lambda nr: (-nr['ci_coherent'], -nr['ib_coherent'], -nr['ci'],
                                               nr['layer'], nr['neuron'], nr['sign']))[:3]


def _screen_analysis(screen, labs, heldout, heldout_labels):
    import numpy as np
    neurons, cells = _screen_neurons(screen, labs)
    groups, comparisons = {}, {}
    for group in ('top_concept', 'top_axis', 'random'):
        nrs = [nr for nr in neurons if nr['group'] == group]
        group_labs = [lab for nr in nrs for _, lab in cells[(nr['name'], nr['dose'])]]
        groups[group] = {'neurons': len(nrs), 'rates': rates(group_labs),
                         'coherent_ib_discoveries': sum(nr['coherent_ib_discovery'] for nr in nrs),
                         'coherent_ci_discoveries': sum(nr['coherent_ci_discovery'] for nr in nrs)}
    top = [nr for nr in neurons if nr['group'] == 'top_concept']
    for group in ('top_axis', 'random'):
        other = [nr for nr in neurons if nr['group'] == group]
        comparison = {}
        for metric in ('ib', 'ci'):
            discovery = 'coherent_' + metric + '_discovery'
            comparison[metric] = {'fisher': _fisher([int(nr[discovery]) for nr in top],
                                                   [int(nr[discovery]) for nr in other]),
                                  'mann_whitney': _mann_whitney([nr[metric + '_coherent'] for nr in top],
                                                                 [nr[metric + '_coherent'] for nr in other])}
            by_group = []
            for members in (top, other):
                maps = []
                for nr in members:
                    pairs = cells[(nr['name'], nr['dose'])]
                    mapping = {row['q']: _metric(lab, metric + '_coherent') for row, lab in pairs}
                    if len(mapping) != len(pairs):
                        raise ValueError('duplicate screen prompt')
                    maps.append(mapping)
                keys = sorted(maps[0])
                if any(set(m) != set(keys) for m in maps):
                    raise ValueError('screen prompts differ between neurons')
                by_group.append({q: float(np.mean([m[q] for m in maps])) for q in keys})
            if set(by_group[0]) != set(by_group[1]):
                raise ValueError('screen prompts differ between groups')
            comparison[metric]['paired_prompt_bootstrap'] = _bootstrap_mean(
                [by_group[0][q] - by_group[1][q] for q in sorted(by_group[0])])
        comparisons['top_concept_minus_' + group] = comparison
    predictors = {}
    for outcome in ('ib', 'ci'):
        y = [nr[outcome + '_coherent'] for nr in neurons]
        positive = [nr['coherent_' + outcome + '_discovery'] for nr in neurons]
        predictors[outcome] = {p: _predictor_stats([nr[p] for nr in neurons], y, positive)
                               for p in ('max_cos_concept', 'cos_away', 'peakiness', 'over_natural')}
    heldout_cells = {}
    if len(heldout) != len(heldout_labels):
        raise ValueError('held-out neuron text/label count mismatch')
    for row, lab in zip(heldout, heldout_labels):
        heldout_cells.setdefault(row['neuron'], []).append(lab)
    geometric = [nr['name'] for nr in neurons if nr['coherent_ib_discovery'] and
                 nr['pct_concept'] >= .99 and abs(nr['cos_away']) < .10]
    return {'neurons': neurons, 'groups': groups, 'comparisons': comparisons, 'predictors': predictors,
            'geometric_discoveries': geometric, 'selected': [nr['name'] for nr in select_neurons(screen, labs)],
            'heldout': {name: rates(ll) for name, ll in heldout_cells.items()},
            'baseline': rates([lab for (name, _), pairs in cells.items() if name is None for _, lab in pairs]),
            'inference_scope': 'screen comparisons condition on selected neurons and development-selected doses; '
                               'prompt bootstrap resamples shared prompts, predictor bootstrap resamples neurons'}


def analyse(atlas, raw, screen, labels, heldout, heldout_labels, smoke):
    import numpy as np
    if len(raw['texts']) != len(labels['raw']):
        raise ValueError('raw text/label count mismatch')
    cells = {}
    for row, lab in zip(raw['texts'], labels['raw']):
        key = (row['kind'], row.get('cell', row.get('condition')), row.get('turn', 1))
        cells.setdefault(key, []).append((row, lab))
    def get(kind, cell, turn=1):
        return cells[(kind, cell, turn)]
    def summary(kind, turn=1):
        return {cell: rates([lab for _, lab in pairs]) for (k, cell, t), pairs in cells.items()
                if k == kind and t == turn}
    dev = summary('dev')
    eligible = [cell for cell in dev if dev[cell]['coherent'] >= .8]
    selected = max(eligible or list(dev), key=lambda c: ((dev[c]['ci'] if eligible else dev[c]['coherent']),
                                                        -float(c.split(',')[1])))
    h3 = {'cells': summary('h3'), 'comparisons': {}}
    for dose in (.25, .5):
        cell = f'0.5,{dose:g}'
        h3['comparisons'][cell + '_minus_axis025'] = {
            metric: _paired_diff(get('h3', cell), get('h3', '0,0.25'), metric)
            for metric in ('ci', 'ci_coherent', 'ib', 'coherent')}
    vv = _paired_values([get('h3', c) for c in ('0,0', '0,0.25', '0.5,0', '0.5,0.25')], 'ci')
    h3['interaction_025'] = {'did': _bootstrap_mean(vv[:, 3] - vv[:, 2] - vv[:, 1] + vv[:, 0]),
                              'logistic': _logistic_interaction(vv)}
    h3['interaction_050'] = {'did': None, 'logistic': None,
                              'reason': 'preregistered five-cell grid has no axis-only (0,0.5) cell'}
    h5 = {'turn1': summary('h5', 1), 'turn2': summary('h5', 2), 'comparisons': {}}
    for condition in ('recipe_025', 'both_025', 'recipe_050', 'both_050'):
        h5['comparisons'][condition + '_minus_prompt'] = {
            metric: _paired_diff(get('h5', condition, 2), get('h5', 'prompt', 2), metric)
            for metric in ('ci', 'ci_coherent')}
    mmlu, item_means = {}, {}
    expected_concepts = None
    for condition, concepts in raw['mmlu'].items():
        if expected_concepts is None:
            expected_concepts = set(concepts)
        if set(concepts) != expected_concepts:
            raise ValueError('MMLU concepts differ across conditions')
        arrays = np.array([concepts[c] for c in sorted(concepts)], dtype=float)
        if not np.isin(arrays, [0, 1]).all():
            raise ValueError('MMLU scores must be binary')
        item_means[condition] = arrays.mean(0)
        mmlu[condition] = {'accuracy': float(arrays.mean()), 'n_items': arrays.shape[1],
                           'per_concept': {c: {'accuracy': float(np.mean(values)),
                                               'ci': rates([{'concept': v} for v in values])['ci_ci']}
                                           for c, values in concepts.items()},
                           'mean_ci': _bootstrap_mean(item_means[condition])['ci']}
    mmlu_comparisons = {}
    for condition, values in item_means.items():
        if len(values) != len(item_means['prompt']):
            raise ValueError('MMLU item counts differ')
        mmlu_comparisons[condition + '_minus_prompt'] = _bootstrap_mean(values - item_means['prompt'])
    screen_results = _screen_analysis(screen, labels['screen'], heldout, heldout_labels)
    selected_suffix = '025' if selected.endswith('0.25') else '050'
    h3diff = h3['comparisons'][selected + '_minus_axis025']['ci']
    h5diff = h5['comparisons']['both_' + selected_suffix + '_minus_prompt']
    pc1 = [float(v) for v in atlas['cos_pc1']]
    mid = int(atlas['layers']) // 2
    selected_h3 = h3['cells'][selected]
    criteria = {'axis_pc1': pc1[mid] >= .6, 'axis_only_ci': h3['cells']['0,0.25']['ci'] <= .05,
                'h3_ci': selected_h3['ci'] >= .6, 'h3_coherence': selected_h3['coherent'] >= .8,
                'h3_gain': h3diff['ci'][0] > 0,
                'h5_coherent_persistence_gain': h5diff['ci_coherent']['estimate'] >= .2 and
                                               h5diff['ci_coherent']['ci'][0] > 0,
                'h5_marginal_gain': h5diff['ci']['estimate'] >= .2 and h5diff['ci']['ci'][0] > 0,
                'mmlu_retention': mmlu_comparisons['both_' + selected_suffix + '_minus_prompt']['estimate'] >= -.03,
                'geometric_neurons': len(screen_results['geometric_discoveries']) >= 3}
    for other in ('top_axis', 'random'):
        top_rate, other_rate = screen_results['groups']['top_concept']['rates'], screen_results['groups'][other]['rates']
        criteria['screen_beats_' + other] = top_rate['ib_coherent'] > other_rate['ib_coherent'] and top_rate['ci'] > other_rate['ci']
    return {'smoke': smoke, 'inferential_run': not smoke, 'dev': dev, 'selected_cell': selected,
            'selected_cell_coherent': bool(eligible), 'h3': h3, 'h5': h5, 'everyday': summary('everyday'),
            'mmlu': {'conditions': mmlu, 'comparisons': mmlu_comparisons}, 'pile_kl': raw['kl'],
            'screen': screen_results, 'atlas': {'cos_pc1': pc1, 'cos_pc1_mid': pc1[mid], 'checks': atlas['checks']},
            'criteria': criteria, 'passes_confirmatory_criteria': all(criteria.values()) if not smoke else None,
            'statistical_protocol': {'rate_intervals': 'Wilson 95%', 'paired_bootstrap': 10000,
                                     'neuron_bootstrap': 2000, 'seed': 20261028,
                                     'h3_h5_unit': '(concept,prompt)', 'mmlu_unit': 'item after averaging concepts'}}


def score(model, tok, texts, batch, checkpoint):
    """The existing judge's unchanged binary rubrics, next-token digit scoring and 3000-char cap."""
    import torch
    from personakit import engine as E
    labels = read(checkpoint) if checkpoint.exists() else [{} for _ in texts]
    if len(labels) != len(texts):
        raise ValueError("Partial judge checkpoint has a different number of records")
    digits = [tok.encode(str(i), add_special_tokens=False) for i in (0, 1)]
    assert all(len(ids) == 1 for ids in digits)
    for rubric, template in (("identity", IDENTITY), ("coherent", COHERENT), ("concept", CONCEPT)):
        indices = [i for i, row in enumerate(texts) if (rubric != "concept" or row["concept"]) and rubric not in labels[i]]
        prompts = [E.chat_ids(tok, template.format(question=texts[i]["question"], answer=texts[i]["reply"][:3000],
                                                  concept=texts[i]["concept"])) for i in indices]
        # Sorting reduces padded prefill memory for the judge.
        order = sorted(range(len(indices)), key=lambda j: len(prompts[j]))
        def forward(js):
            ids, mask = E.pad_left([prompts[j] for j in js], tok.pad_token_id, model.device)
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1].float()
            probabilities = logits[:, [ids[0] for ids in digits]].softmax(-1).cpu()
            mass = logits.softmax(-1)[:, [ids[0] for ids in digits]].sum(-1).cpu()
            return [(j, int(p.argmax()), p.tolist(), float(m)) for j, p, m in zip(js, probabilities, mass)]
        for j, value, probabilities, mass in E.batched(forward, order, batch):
            labels[indices[j]].update({rubric: value, rubric + "_p": probabilities, rubric + "_mass": mass})
        save(checkpoint, labels)
        print(f"[judge] {rubric}: {len(indices)}", flush=True)
    for row, lab in zip(texts, labels):
        if row["target"]:
            lab["cm"] = int(P.mentions(row["reply"], row["target"]))
    return labels


def fingerprint():
    paths = [HERE / "transfer_study.py", HERE.parent / "hf_cache.py", HERE.parent / "evals.py",
             PRIOR / "prompts.py", *[PRIOR / "personakit/personakit" / f"{name}.py" for name in ("data", "engine")]]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


# Pre-export SHA-256 (d44fdcd1/865d36f8) before moving `import torch` into the non-JSON branches of save()/read().
PRIOR_RUN_SHA256 = "d42949e94d2347736b72fa3a56b20160a6ae316c8c4101a68373102fd56c1dec"


@app.function(image=hf_cache.image("flash-linear-attention==0.6.0").add_local_python_source("personakit", "prompts"),
              gpu="A10G", memory=32768, timeout=3600, volumes={**hf_cache.volumes, "/out": OUT})
def run(model_key: str, tag: str, smoke: bool, source_hash: str):
    import torch
    from personakit import engine as E
    started = time.time()
    if modal.is_local():
        # Avoid oversubscribing a shared CPU host during the prescribed heavy job.
        torch.set_num_threads(4)
    else:
        OUT.reload()
    sz, judge_key = (SMOKE, "0.8b") if smoke else (FULL, "9b")
    odir = out_dir(tag)
    manifest = {"protocol": 1, "model": model_key, "smoke": smoke, "sizes": sz, "source_sha256": source_hash,
                "model_pin": list(D.MODELS[model_key]), "judge_pin": list(D.MODELS[judge_key]),
                "pile_pin": list(D.PILE), "mmlu_pin": list(D.MMLU), "axis_data": D.AXIS_DATA}
    if (odir / "manifest.json").exists():
        saved = read(odir / "manifest.json")
        same_spec = {**saved, "source_sha256": None} == {**manifest, "source_sha256": None}
        if not same_spec or saved.get("source_sha256") not in (source_hash, PRIOR_RUN_SHA256):
            raise ValueError("Checkpoint code/configuration differs; use a new --tag")
    else:
        save(odir / "manifest.json", manifest)
    if (odir / "results.json").exists():
        return read(odir / "results.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage_paths = [odir / name for name in ("atlas.pt", "h3_h5_raw.pt", "screen_raw.pt")]
    needs_target = any(not path.exists() for path in stage_paths)
    if needs_target:
        model, tok = E.load(hf_cache.model(*D.MODELS[model_key]), device)
        print(f"[target] {model_key}: {model.config.num_hidden_layers} layers on {device}", flush=True)
    if not stage_paths[0].exists():
        save(stage_paths[0], build_atlas(model, tok, model_key, sz, odir))
    atlas = read(stage_paths[0])
    if not stage_paths[1].exists():
        save(stage_paths[1], evaluate(model, tok, atlas, sz))
    raw = read(stage_paths[1])
    if not stage_paths[2].exists():
        save(stage_paths[2], screen_neurons(model, tok, atlas, model_key, sz))
    screen = read(stage_paths[2])
    if needs_target:
        del model, tok
        E.free()
    labels_path = odir / "labels.pt"
    if not labels_path.exists():
        judge, jtok = E.load(hf_cache.model(*D.MODELS[judge_key]), device)
        judge_started = time.time()
        labels = {"raw": score(judge, jtok, raw["texts"], sz["judge_batch"], odir / "raw_labels_partial.pt"),
                  "screen": score(judge, jtok, screen["texts"], sz["judge_batch"], odir / "screen_labels_partial.pt")}
        labels["seconds"] = time.time() - judge_started
        del judge, jtok
        E.free()
        save(labels_path, labels)
    labels = read(labels_path)
    chosen = select_neurons(screen, labels["screen"])
    heldout_path = odir / "heldout_raw.pt"
    if not heldout_path.exists():
        heldout = []
        if chosen:
            model, tok = E.load(hf_cache.model(*D.MODELS[model_key]), device)
            for nr in chosen:
                heldout.extend(rollout(model, tok, P.IDENTITY_TEST[:sz["test_q"]], nr["target"],
                                       [(nr["layer"], nr["dose"] * nr["vector_per_dose"])], sz,
                                       {"kind": "heldout", "neuron": nr["name"], "dose": nr["dose"]}))
            del model, tok
            E.free()
        save(heldout_path, heldout)
    heldout = read(heldout_path)
    heldout_label_path = odir / "heldout_labels.pt"
    if not heldout_label_path.exists():
        heldout_labels = []
        if heldout:
            judge, jtok = E.load(hf_cache.model(*D.MODELS[judge_key]), device)
            heldout_labels = score(judge, jtok, heldout, sz["judge_batch"], odir / "heldout_labels_partial.pt")
            del judge, jtok
            E.free()
        save(heldout_label_path, heldout_labels)
    heldout_labels = read(heldout_label_path)
    compact_screen = {**screen, "neurons": [{k: v for k, v in nr.items() if k != "vector_per_dose"}
                                           for nr in screen["neurons"]]}
    result = analyse(atlas, raw, compact_screen, labels, heldout, heldout_labels, smoke)
    result.update({"model": model_key, "tag": tag, "manifest": manifest, "loads": hf_cache.loads,
                   "runtime_seconds": time.time() - started,
                   "stage_seconds": {"atlas": atlas["seconds"], "evaluation": raw["seconds"],
                                     "screen": screen["seconds"], "judge": labels["seconds"]},
                   "counts": {"atlas": len(atlas["jobs"]), "evaluation_records": len(raw["texts"]),
                              "screen_neurons": len(screen["neurons"]), "screen_records": len(screen["texts"]),
                              "heldout_records": len(heldout), "mmlu_items": len(raw["mmlu_item_ids"]),
                              "pile_calibration_tokens": atlas["pile_tokens"], "chat_calibration_tokens": atlas["chat_tokens"]},
                   "output_dir": str(odir)})
    save(odir / "results.json", result)
    return result


@app.local_entrypoint()
def main(model: str = "qwen25-3b", smoke: bool = False, tag: str = "", export_json: bool = False):
    if smoke:
        model = "0.8b"
    if model not in LAYERS or (not smoke and model == "0.8b"):
        raise ValueError("Use qwen25-3b or llama32-3b; --smoke uses the pinned 0.8b model")
    tag = tag or f"transfer-{model}{'-smoke' if smoke else ''}"
    # Validate before launching remote work.
    if pathlib.Path(tag).name != tag or tag in (".", ".."):
        raise ValueError("tag must be one directory name")
    result = call(run, model, tag, smoke, fingerprint())
    if export_json or smoke:
        destination = HERE / ("smoke_receipt.json" if smoke else f"{tag}_results.json")
        save(destination, result)
    print("PERSONAKIT_RESULT", json.dumps({"model": model, "tag": tag, "smoke": smoke,
          "seconds": round(result["runtime_seconds"], 2), "counts": result["counts"],
          "checks": result["atlas"]["checks"], "selected_cell": result["selected_cell"],
          "passes_confirmatory_criteria": result["passes_confirmatory_criteria"]}))
    print("PERSONAKIT_RESULT", json.dumps({"criteria": result["criteria"], "output_dir": result["output_dir"]}))
