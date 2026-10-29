"""H1-v2b: contrastive trait extraction, weight alignment, depth steering and paired MMLU."""
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "study"))
sys.path.insert(0, str(HERE))
import modal
import hf_cache
from jobs import MODELS, OUT, call, out_dir
import trait_data as D

image = hf_cache.image("flash-linear-attention==0.6.0").add_local_python_source("persona", "prompts", "jobs", "judge", "trait_data")
app = modal.App("personakit-trait-directions")


def extract(model, tok, ax, smoke):
    """Never decode/store extraction texts. Delete response token buffers after activation capture."""
    import torch
    import persona
    layers = list(range(len(persona.blocks(model))))
    raw, hats, raw_hats, counts = {}, {}, {}, {}
    axis = D.unit(ax["unit"])
    gen_batch = 8 if smoke else 80
    span_batch = 4 if smoke else 16
    max_new = 6 if smoke else 48
    with persona.Hooks(model) as hooks:
        for trait in D.TRAITS:
            means, counts[trait] = {}, {}
            for polarity, systems in (("positive", D.POS_SYSTEMS[trait]), ("negative", D.NEG_SYSTEMS[trait])):
                inputs = [persona.chat_ids(tok, q, sp) for sp in systems[:1 if smoke else 5]
                          for q in D.EXTRACT_PROMPTS[trait][:2 if smoke else 16]]
                order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
                sorted_inputs = [inputs[i] for i in order]
                total, n, tokens = None, 0, 0
                for start in range(0, len(sorted_inputs), gen_batch):
                    chunk = sorted_inputs[start:start + gen_batch]
                    replies = persona.generate(model, tok, chunk, max_new)
                    if any(not r for r in replies):
                        raise ValueError("empty extraction reply; cannot estimate a response mean")
                    seqs = [a + b for a, b in zip(chunk, replies)]
                    starts = list(map(len, chunk))
                    tokens += sum(map(len, replies))
                    seq_order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
                    for s_start in range(0, len(seq_order), span_batch):
                        ks = seq_order[s_start:s_start + span_batch]
                        acts = persona.span_means(model, hooks, [seqs[k] for k in ks], [starts[k] for k in ks], layers)
                        total = acts.sum(0) if total is None else total + acts.sum(0)
                        n += len(acts)
                        del acts
                    del replies, seqs, starts  # positive hostile token buffers disappear immediately after capture
                means[polarity] = total / n
                counts[trait][polarity] = {"rollouts": n, "response_tokens": tokens}
                print(f"[extract] {trait}/{polarity}: {n} response means; text discarded", flush=True)
            raw[trait] = means["positive"] - means["negative"]
            raw_hats[trait] = D.unit(raw[trait])
            orth = raw[trait] - (raw[trait] * axis).sum(-1, keepdim=True) * axis
            hats[trait] = D.unit(orth)
            if float((hats[trait] * axis).sum(-1).abs().max()) > 1e-4:
                raise ValueError("trait axis orthogonalization failed")
    return {"t_raw": raw, "t_raw_hat": raw_hats, "t_hat": hats, "extraction_counts": counts}


def weight_alignment(model, tok, ax, dirs, known):
    """Exact signed within-layer ranks; no percentile approximation or neuron subsampling."""
    import torch
    import persona
    from prompts import CONCEPTS_DEV, CONCEPTS_TEST
    c_hat = {c: D.unit(ax["concept_dirs"][c]) for c in CONCEPTS_DEV + CONCEPTS_TEST}
    away = -D.unit(ax["unit"])
    logit, token_ids = {}, {}
    with torch.inference_mode():
        for trait, sides in D.LOGIT_STEMS.items():
            ids = [sorted({i for stem in stems for i in tok.encode(stem, add_special_tokens=False)}) for stems in sides]
            token_ids[trait] = ids
            # Preregistered weight-only diagnostic, not an activation-aware logit-lens attribution.
            normalized_rows = [model.model.norm(model.lm_head.weight[ix]).float().mean(0).cpu() for ix in ids]
            logit[trait] = D.unit(normalized_rows[0] - normalized_rows[1])
    per_neuron, profiles, inter = {}, {}, {}
    for L, block in enumerate(persona.blocks(model)):
        named = {**{f"t:{t}": v[L] for t, v in dirs["t_hat"].items()},
                 **{f"raw:{t}": v[L] for t, v in dirs["t_raw_hat"].items()},
                 **{f"logit:{t}": v for t, v in logit.items()},
                 **{f"c:{c}": v[L] for c, v in c_hat.items()}, "away": away[L]}
        names = list(named)
        with torch.inference_mode():
            W = block.mlp.down_proj.weight.float()
            W = W / W.norm(dim=0, keepdim=True).clamp_min(1e-12)
            V = torch.stack(list(named.values())).to(W.device)
            cos = (V @ W).cpu()
            matrix = (V @ V.T).cpu()
        profiles[str(L)] = {"highlight_L8_L26": 8 <= L <= 26, "directions": {
            name: {"q99": float(v.quantile(0.99)), "q999": float(v.quantile(0.999)), "max": float(v.max()),
                   "min": float(v.min()), "n_columns": len(v)} for name, v in zip(names, cos)}}
        selected = [i for i, name in enumerate(names) if name.startswith(("t:", "c:")) or name == "away"]
        inter[str(L)] = {"names": [names[i] for i in selected], "cosines": matrix[selected][:, selected].tolist()}
        for neuron, (layer, n, sign) in known.items():
            if layer != L:
                continue
            scores = {}
            for name, values in zip(names, cos):
                signed = sign * values
                value = float(signed[n])
                less = int((signed < value).sum())
                scores[name] = {"cosine": value, "percentile": 100 * less / len(signed),
                                "less": less, "n_columns": len(signed),
                                "empirical_upper_tail": (len(signed) - less) / len(signed)}
            per_neuron[neuron] = {"layer": L, "neuron": n, "sign": sign, "directions": scores}
        del W, V, cos, matrix
    # A finite-column rank tail, explicitly distinguished from a sampling-based statistical p-value.
    tails = {name: row["directions"]["t:" + D.TARGET[name]]["empirical_upper_tail"]
             for name, row in per_neuron.items() if D.TARGET[name] in D.TRAITS}
    return {"known": per_neuron, "profiles": profiles, "inter_direction": inter,
            "weight_rank_tails": tails, "weight_rank_tails_holm": D.holm_adjust(tails),
            "weight_tail_note": "Descriptive uniform-column rank tails; columns are not independent sampled replications.",
            "logit_token_ids": token_ids, "logit_note": "Mean of final-normed lm_head rows, positive minus negative; identical across layers."}, c_hat, logit


def steering(model, tok, ax, dirs, c_hat, smoke, p1_recs=None):
    import torch
    import persona
    from prompts import CONCEPTS_TEST
    mapping = dict(zip([10, 14, 18, 22], [6, 10, 14, 18] if smoke else [10, 14, 18, 22]))
    m, axis = ax["median_resid"], D.unit(ax["unit"])
    rng = torch.Generator(device="cpu").manual_seed(D.SEED)
    random_dirs = {}
    for nominal in (14, 22):
        L = mapping[nominal]
        v = torch.randn(axis[L].shape, generator=rng)
        random_dirs[L] = D.unit(v - (v @ axis[L]) * axis[L])
    recs, conditions = [], {}
    doses = [-0.5, 0.5] if smoke else [-0.5, 0.25, 0.5, 0.75]
    entity_doses = [0.5] if smoke else [0.25, 0.5, 0.75]
    with persona.Hooks(model) as hooks:
        if p1_recs is None:
            D.generate_condition(model, tok, hooks, recs, "unsteered", D.BATTERIES, smoke, dose=0.0)

        def generate(name, v, nominal, dose, batteries, **extra):
            L = mapping[nominal]
            vec = (dose * float(m[L]) * v) if dose else None
            metadata = {"direction": name, "layer": L, "nominal_layer": nominal, "dose": dose, **extra}
            condition = f"{name}_L{L}_d{dose:g}"
            conditions[condition] = {**metadata, "batteries": list(batteries)}
            D.generate_condition(model, tok, hooks, recs, condition, batteries, smoke, vec=vec, **metadata)

        for trait in D.TRAITS:
            for nominal, L in mapping.items():
                for dose in doses:
                    batteries = list(D.BATTERIES) if nominal == 14 and dose == 0.5 else ["identity", D.MATCHED[trait]]
                    generate(f"t_{trait}", dirs["t_hat"][trait][L], nominal, dose, batteries,
                             trait=trait, hostile_condition=trait == "hostile")
        for nominal, L in mapping.items():
            for dose in entity_doses:
                generate("c_aliens", c_hat["aliens"][L], nominal, dose,
                         list(D.BATTERIES) if dose == 0.5 else ["identity"], entity="aliens")
            for concept in CONCEPTS_TEST[:2 if smoke else 5]:
                generate("c_" + concept, c_hat[concept][L], nominal, 0.5, ["identity"], entity=concept)
        for nominal in (14, 22):
            generate("random", random_dirs[mapping[nominal]], nominal, 0.5, list(D.BATTERIES))
        if p1_recs is None:
            D.flush_generations(model, tok, hooks, recs, smoke)
        else:
            combined = p1_recs + recs
            D.flush_generations(model, tok, hooks, combined, smoke)
            base_n = 8 if smoke else 80
            recs = [{**r} for r in p1_recs[:base_n]] + recs
        hooks.clear()
        rows = persona.mmlu_rows(8 if smoke else 300, seed=42)
        mmlu_batch = 8 if smoke else 64
        mmlu = {"unsteered": persona.mmlu(model, tok, rows, batch=mmlu_batch)}
        for trait in D.TRAITS:
            for dose in (0.5, -0.5):
                hooks.clear()
                L = mapping[14]
                hooks.add(L, dose * float(m[L]) * dirs["t_hat"][trait][L])
                mmlu[f"t_{trait}_L{L}_d{dose:g}"] = persona.mmlu(model, tok, rows, batch=mmlu_batch)
        hooks.clear()
        L = mapping[22]
        hooks.add(L, 0.5 * float(m[L]) * c_hat["aliens"][L])
        mmlu[f"c_aliens_L{L}_d0.5"] = persona.mmlu(model, tok, rows, batch=mmlu_batch)
        hooks.clear()
    import hashlib
    item_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return recs, conditions, mmlu, item_hash, mapping, random_dirs


def analyze(recs, labels, conditions, mmlu, mapping, smoke):
    comparisons = []
    for condition, meta in conditions.items():
        trait = meta.get("trait")
        for battery in meta["batteries"]:
            b_metrics = D.battery_metrics(battery)
            for metric in b_metrics:
                primary = (trait and battery == D.MATCHED[trait] and metric == trait and meta["nominal_layer"] == 14)
                ref = ("unsteered__concept_" + meta["entity"]
                       if metric in ("concept", "lex_concept") and meta.get("entity", "aliens") != "aliens" else "unsteered")
                comparisons.append((condition, ref, battery, [metric],
                                    "H1-v2b_steering" if primary else "part2_secondary"))
            if meta["nominal_layer"] in (14, 22) and meta["direction"] != "random":
                ref = f"random_L{meta['layer']}_d0.5"
                for metric in b_metrics:
                    matched_ref = (ref + "__concept_" + meta["entity"]
                                   if metric in ("concept", "lex_concept") and meta.get("entity", "aliens") != "aliens" else ref)
                    comparisons.append((condition, matched_ref, battery, [metric], "part2_random_controls"))
        if trait and meta["dose"] > 0:
            ref = f"t_{trait}_L{meta['layer']}_d-0.5"
            comparisons.append((condition, ref, D.MATCHED[trait], [trait],
                                "H1-v2b_steering" if meta["nominal_layer"] == 14 else "part2_secondary"))
        if meta.get("entity") and meta["dose"] == 0.5 and meta["nominal_layer"] != 10:
            ref = f"{meta['direction']}_L{mapping[10]}_d0.5"
            comparisons.append((condition, ref, "identity", ["concept", "ib", "coherent"], "H1-v2b_entity_depth"))
    boot = 200 if smoke else 10000
    stats = D.grouped_analysis(recs, labels, comparisons, boot)
    mm = {"rates": {name: D.rate(values) for name, values in mmlu.items()}, "paired": {}}
    for name, values in mmlu.items():
        if name != "unsteered":
            mm["paired"][name] = D.paired(values, mmlu["unsteered"], boot)
    adjusted = D.holm_adjust({name: value["p"] for name, value in mm["paired"].items()})
    for name in adjusted:
        mm["paired"][name]["p_holm"] = adjusted[name]
    return {**stats, "mmlu": mm}


def entity_reference_rows(recs, smoke):
    """Rescore the same generated identity controls for each held-out entity, before redaction.

    These are scoring aliases, not additional generations or independent observations.
    """
    from prompts import CONCEPTS_TEST, PERSONA_NAMES
    refs = [r for r in recs if r["battery"] == "identity"
            and (r["condition"] == "unsteered" or r.get("direction") == "random")]
    return [{**r, "condition": r["condition"] + "__concept_" + entity,
             "concept": PERSONA_NAMES[entity], "entity": entity, "scoring_alias_of": r["condition"]}
            for entity in CONCEPTS_TEST[:2 if smoke else 5] for r in refs]


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu="A10G", memory=32768, timeout=7200)
def run(model_key: str, smoke: bool, both: bool = False):
    import torch
    import persona
    t0 = time.time()
    if not modal.is_local():
        OUT.reload()
    if smoke:
        torch.set_num_threads(8)
    tag = f"part2-{model_key}" + ("-smoke" if smoke else "")
    out = out_dir(tag)
    prompt_hash = D.validate_data()
    ax, artifact_hashes = D.load_inputs(model_key, smoke)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
    if len(ax["median_resid"]) != len(persona.blocks(model)):
        raise ValueError("artifact/model layer mismatch")
    known = D.known_neurons(model, smoke)
    dirs = extract(model, tok, ax, smoke)
    alignment, c_hat, logit = weight_alignment(model, tok, ax, dirs, known)
    print(f"[alignment] {len(persona.blocks(model))} layers; {len(known)} known neurons", flush=True)
    p1_recs = None
    if both:
        p1_recs, p1_known, p1_controls, p1_conditions, p1_doses, p1_control_doses = D.queue_part1_conditions(
            model, tok, None, ax, smoke)
    recs, conditions, mmlu, item_hash, mapping, random_dirs = steering(
        model, tok, ax, dirs, c_hat, smoke, p1_recs=p1_recs)
    generated_count = len(recs)
    recs.extend(entity_reference_rows(recs, smoke))
    del model, tok
    persona.free()
    judge_key = "0.8b" if smoke else "9b"
    model, tok = persona.load(hf_cache.model(*MODELS[judge_key]), device)
    unsteered_lookup = {}
    p1_report = None
    if both:
        p1_lookup = {}
        p1_labels = D.score_traits(model, tok, p1_recs, tuple(D.RUBRICS), 32 if smoke else 24, base_lookup=p1_lookup)
        _, p1_report = D.finalize_part1(model_key, judge_key, smoke, t0, prompt_hash, artifact_hashes,
                                        p1_known, p1_controls, p1_conditions, p1_doses, p1_control_doses,
                                        p1_recs, p1_labels)
        if not modal.is_local():
            OUT.commit()
        unsteered_lookup = {k: v for k, v in p1_lookup.items() if k[0] == "unsteered"}
    labels = D.score_traits(model, tok, recs, tuple(D.RUBRICS), 32 if smoke else 24, base_lookup=unsteered_lookup)
    del model, tok
    persona.free()
    stats = analyze(recs, labels, conditions, mmlu, mapping, smoke)
    summary = {"tag": tag, "smoke": smoke, "model": model_key, "judge": judge_key, "texts": len(recs),
               "generated_replies": generated_count, "scoring_aliases": len(recs) - generated_count,
               "redacted": sum(r["reply"] == D.REDACTED for r in recs), "layers": len(ax["unit"]),
               "extraction_rollouts": sum(v[side]["rollouts"] for v in dirs["extraction_counts"].values()
                                          for side in ("positive", "negative")),
               "mmlu_items": len(mmlu["unsteered"]), "mmlu_conditions": len(mmlu),
               "minutes": round((time.time() - t0) / 60, 2)}
    details = {**summary, **stats, "alignment": alignment, "conditions": conditions, "layer_mapping": mapping,
               "mmlu_item_hash": item_hash, "mmlu_binary_by_item": mmlu, "prompt_hash": prompt_hash,
               "input_sha256": artifact_hashes, "extraction_counts": dirs["extraction_counts"], "loads": hf_cache.loads,
               "config": {"seed": D.SEED, "boot": 200 if smoke else 10000, "max_new": 4 if smoke else 56,
                          "extraction_max_new": 6 if smoke else 48, "model_pin": MODELS[model_key], "judge_pin": MODELS[judge_key]},
               "interpretation": "Smoke exercises paths only; no confirmatory inference from stand-in neurons."
                                 if smoke else "All preregistered layers/doses reported."}
    D.save_scored(out, recs, labels)
    torch.save({**dirs, "c_hat": c_hat, "logit_hat": logit, "random_hat": random_dirs,
                "unit": ax["unit"], "median_resid": ax["median_resid"], "layer_mapping": mapping,
                "prompt_hash": prompt_hash, "input_sha256": artifact_hashes}, out / "trait_dirs.pt")
    (out / "part2.json").write_text(json.dumps(details, indent=2, allow_nan=False))
    if not modal.is_local():
        OUT.commit()
    if both:
        primary_contrasts = {k: v for k, v in stats["contrasts"].items()
                             if v["family"] in ("H1-v2b_steering", "H1-v2b_entity_depth")
                             or k.endswith(("/ib", "/concept", "/coherent", "/sycophancy", "/condolence", "/hostile", "/lex_concept"))}
        keep_layers = {str(L) for L in (6, 8, 10, 12, 13, 14, 16, 18, 20, 22, 24, 26)}
        p2_report = {**summary, "layer_mapping": mapping, "rates": stats["rates"], "mmlu": stats["mmlu"],
                     "primary_contrasts": primary_contrasts, "holm_families": stats["holm_families"],
                     "alignment_known": alignment["known"], "weight_rank_tails": alignment["weight_rank_tails"],
                     "weight_rank_tails_holm": alignment["weight_rank_tails_holm"],
                     "profiles_selected": {k: v for k, v in alignment["profiles"].items() if k in keep_layers},
                     "inter_selected": {k: v for k, v in alignment["inter_direction"].items() if k in keep_layers},
                     "extraction_counts": dirs["extraction_counts"], "mmlu_item_hash": item_hash,
                     "prompt_hash": prompt_hash, "input_sha256": artifact_hashes, "config": details["config"]}
        return {**summary, "part1_report": p1_report, "part2_report": p2_report}
    return summary


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False, both: bool = False):
    if not smoke and model != "4b":
        raise ValueError("the preregistered full target is 4b")
    result = call(run, "0.8b" if smoke else model, smoke, both)
    print("PERSONAKIT_RESULT", json.dumps(result), flush=True)
