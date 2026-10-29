"""X2-v2: calibration-frozen sparse residual coalitions (private research).

Each signed down column adds sign*d*median_residual[L]*W[:, n]/||W[:, n]||
at the decoder output, on ALL prompt and generated positions, through persona.Hooks.
This preserves X2's residual-addition semantics; it is not activation clamping.
The nonnegative coefficients describe profile reconstruction, not activation levels.
Profiles use centered unit logit lenses and uncentered unit concept-cosine vectors,
in sorted shared concept order, concatenated with equal 1/sqrt(2) block weights.

Use python --smoke only AFTER the committed design exists. Paid stages
are calibrate and final, launched only via modal run after the project owner allocates budget.
The final stage reads a persisted calibration freeze; it never searches test doses.
Historical 0.07 is a pooled three-twin result, never a set of twenty observations.
"""

import contextlib
import hashlib
import itertools
import json
import math
import pathlib
import random
import subprocess
import sys
import time
import uuid

import modal

HERE = pathlib.Path(__file__).resolve().parent
OLD = HERE.parent.parent / "study"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(OLD))
import hf_cache
from jobs import MODELS, OUT, call, image, out_dir

app = modal.App("personakit-x2-coalitions")
DESIGN_PATH = "followups/2b-twins-x3-lora/DESIGN.md"
FULL = dict(source="4b", target="2b", judge="9b", prefilter=200, ks=[2, 3, 5],
            totals=[0.5, 0.75], single_doses=[0.5, 1.0], singles=3,
            calibration_q=10, test_q=20, everyday_q=20, max_new=96,
            chunk=64, batch=8, judge_batch=2, boot=2000, seed=0)
SMOKE = {**FULL, "source": "0.8b", "target": "0.8b", "judge": "0.8b",
         "prefilter": 12, "calibration_q": 2, "test_q": 2, "everyday_q": 2,
         "max_new": 4, "chunk": 8, "batch": 4, "judge_batch": 4,
         "boot": 50, "layers": [10, 11], "neurons": 16}
SOURCE = {"4b": (22, 2742, 1), "0.8b": (10, 7, -1)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_jsonl(path, rows):
    """Publish only complete score prefixes; interruption leaves the previous checkpoint intact."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, allow_nan=False) + "\n")
    tmp.replace(path)


def checkpoint():
    if not modal.is_local():
        OUT.commit()


@contextlib.contextmanager
def stage_claim(tag, stage):
    """Serialize every writer for a tag, including generation versus score-only resumes.

    Modal Dict.put(skip_if_exists=True) is a server-side conditional insert, unlike
    exclusive file creation on independently mounted Volume snapshots. A hard-killed
    remote owner leaves a fail-closed claim: only clear that claim after verifying its
    recorded function call has ended. There is no automatic stale-owner takeover.
    """
    if not tag or pathlib.PurePosixPath(tag).name != tag or tag in (".", ".."):
        raise ValueError("tag must be one directory name, without path aliases")
    if modal.is_local():
        import fcntl
        with (out_dir(".x2-claims") / digest(tag)).open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"another X2 stage owns tag {tag}") from exc
            yield
        return
    claims = modal.Dict.from_name("personakit-x2-claims", create_if_missing=True)
    owner = {"token": uuid.uuid4().hex, "stage": stage,
             "call_id": modal.current_function_call_id(), "started": time.time()}
    if not claims.put(tag, owner, skip_if_exists=True):
        raise RuntimeError(f"another X2 stage owns tag {tag}: {claims.get(tag)}")
    try:
        yield
    finally:
        # Publish before releasing ownership. A failed commit deliberately retains the claim.
        checkpoint()
        claims.pop(tag)


def unit(x, center=False):
    import torch
    x = x.float()
    if center:
        x = x - x.mean(-1, keepdim=True)
    norms = x.norm(dim=-1, keepdim=True)
    if not torch.isfinite(x).all() or (norms <= 1e-12).any():
        raise ValueError("nonfinite or zero profile block")
    return x / norms


def lens(model, columns):
    import torch
    with torch.inference_mode():
        return model.lm_head(model.model.norm(columns.to(model.dtype))).float()


def concept_profile(column, axis, layer, names):
    """Comparable scalar cosines in shared concept-name order, not cross-width vectors."""
    import torch
    directions = torch.stack([axis["concept_dirs"][n][layer] for n in names]).to(column.device)
    return unit(directions) @ unit(column)


def nnls_gram(gram, target):
    """Exact NNLS for <=5 columns: enumerate active faces, including the empty face."""
    import numpy as np
    n = len(target)
    best, loss = np.zeros(n), 0.0
    for bits in itertools.product((False, True), repeat=n):
        ids = np.flatnonzero(bits)
        if not len(ids):
            continue
        g = gram[np.ix_(ids, ids)]
        w = np.linalg.lstsq(g, target[ids], rcond=1e-10)[0]
        if (w < -1e-9).any():
            continue
        coef = np.zeros(n)
        coef[ids] = np.maximum(w, 0)
        value = float(coef @ gram @ coef - 2 * coef @ target)
        if value < loss:
            best, loss = coef, value
    return best


def nonnegative_omp(gram, target, candidates, ks):
    """Greedy positive residual correlation, with a nonnegative orthogonal refit each step."""
    import numpy as np
    gram, target = np.asarray(gram, dtype=float), np.asarray(target, dtype=float)
    chosen, weights, snapshots = [], np.zeros(0), {}
    for k in range(1, max(ks) + 1):
        residual = target - (gram[:, chosen] @ weights if chosen else 0)
        used = {(candidates[i]["layer"], candidates[i]["neuron"]) for i in chosen}
        for i, c in enumerate(candidates):
            if (c["layer"], c["neuron"]) in used:
                residual[i] = -np.inf
        winner = int(np.argmax(residual))
        if not np.isfinite(residual[winner]) or residual[winner] <= 1e-10:
            raise ValueError(f"OMP has no positive unused atom at step {k}; cannot silently pad coalition")
        chosen.append(winner)
        weights = nnls_gram(gram[np.ix_(chosen, chosen)], target[chosen])
        if k in ks:
            snapshots[k] = {"atoms": [dict(candidates[i]) for i in chosen],
                            "weights": weights.tolist(), "effective_k": int((weights > 1e-8).sum()),
                            "residual_sq": float(1 - 2 * weights @ target[chosen] +
                                                 weights @ gram[np.ix_(chosen, chosen)] @ weights)}
    return snapshots


def doses(weights, total):
    """Proportional OMP weights, clipped at .35 each, with no redistribution."""
    if total > 0.75 or total <= 0 or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("invalid bounded coalition dose")
    values = [min(0.35, total * w / sum(weights)) for w in weights]
    assert max(values) <= 0.35 and sum(values) <= 0.75 + 1e-12
    return values


def matched_controls(model, atoms, excluded, seed):
    """Same layer/sign; seeded draw from the nearest raw-column norm neighbourhood."""
    import torch
    import persona
    rng, controls, used = random.Random(seed), [], set(excluded)
    for atom in atoms:
        layer = atom["layer"]
        norms = persona.blocks(model)[layer].mlp.down_proj.weight.float().norm(dim=0).cpu()
        relative = (norms / atom["raw_norm"] - 1).abs()
        pool = [int(n) for n in torch.argsort(relative).tolist()
                if (layer, int(n)) not in used and float(relative[n]) <= 0.05]
        if not pool:
            raise ValueError(f"no unused raw-norm match within 5% at L{layer}")
        neuron = rng.choice(pool)
        used.add((layer, neuron))
        controls.append({"layer": layer, "neuron": neuron, "sign": atom["sign"],
                         "raw_norm": float(norms[neuron]),
                         "norm_relative_error": float(relative[neuron])})
    return controls


def rates(rows):
    n = len(rows)
    if not n:
        raise ValueError("empty score group")
    return {key: sum(r[key] for r in rows) / n for key in ("ci", "concept", "coherent", "ib", "cm")}


def select_configuration(configs, rows):
    """Calibration-only selection; deterministic low-dose/name tie breaking."""
    choices = []
    for config in configs:
        r = rates([r for r in rows if r["config_id"] == config["id"]])
        if r["coherent"] >= 0.8:
            choices.append((config, r))
    if not choices:
        return None
    return min(choices, key=lambda pair: (-pair[1]["ci"], round(sum(pair[0]["doses"]), 12), pair[0]["id"]))[0]


def paired_bootstrap(left, right, boot, seed):
    """Frozen interventions, paired prompt resampling; no test-coherence zeroing or selection."""
    if left.keys() != right.keys() or not left:
        raise ValueError("paired comparison must have exactly the same prompt IDs")
    values = [left[k] - right[k] for k in sorted(left)]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(boot))
    return {"difference": sum(values) / len(values),
            "ci95": [draws[int(0.025 * boot)], draws[min(boot - 1, int(0.975 * boot))]]}


def search(sizes, source_axis, target_axis, directory):
    """Source lens, top-200 target signed atoms, equal-weight lens/concept Gram and OMP."""
    import torch
    import persona
    import prompts as P
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = sorted(P.CONCEPTS_DEV + P.CONCEPTS_TEST)
    if set(source_axis["concept_dirs"]) != set(names) or set(target_axis["concept_dirs"]) != set(names):
        raise ValueError("source and target must contain the same preregistered concept profiles")
    source, stok = persona.load(hf_cache.model(*MODELS[sizes["source"]]), device)
    source_vocab = stok.get_vocab()
    layer, neuron, sign = SOURCE[sizes["source"]]
    column = sign * persona.down_col(source, layer, neuron)
    z = unit(lens(source, unit(column)[None]), center=True)[0]
    c = unit(concept_profile(column, source_axis, layer, names))
    del source, stok, column
    persona.free()
    model, tok = persona.load(hf_cache.model(*MODELS[sizes["target"]]), device)
    if tok.get_vocab() != source_vocab:
        raise ValueError("logit-lens matching requires identical token-to-ID mappings")
    del source_vocab
    candidates = []
    layers = sizes.get("layers", list(range(model.config.num_hidden_layers)))
    for layer in layers:
        weight = persona.blocks(model)[layer].mlp.down_proj.weight
        limit = min(weight.shape[1], sizes.get("neurons", weight.shape[1]))
        for start in range(0, limit, sizes["chunk"]):
            columns = weight[:, start:min(limit, start + sizes["chunk"])].T.float()
            norms = columns.norm(dim=-1)
            similarities = (unit(lens(model, unit(columns)), center=True) @ z).cpu()
            for j, similarity in enumerate(similarities.tolist()):
                # Same-model smoke must not trivially recover itself and stop OMP after one atom.
                if sizes["source"] == sizes["target"] and (layer, start + j) == SOURCE[sizes["source"]][:2]:
                    continue
                candidates.append({"layer": layer, "neuron": start + j, "sign": 1 if similarity >= 0 else -1,
                                   "pearson": abs(similarity), "raw_norm": float(norms[j])})
            candidates.sort(key=lambda a: (-a["pearson"], a["layer"], a["neuron"]))
            candidates = candidates[:sizes["prefilter"]]
        print(f"[x2] lens scan L{layer}: {len(candidates)} retained", flush=True)
    profiles = []
    for atom in candidates:
        column = atom["sign"] * persona.down_col(model, atom["layer"], atom["neuron"])
        ll = unit(lens(model, unit(column)[None]), center=True)[0]
        cp = unit(concept_profile(column, target_axis, atom["layer"], names))
        profiles.append(torch.cat((ll, cp)) / math.sqrt(2))
    matrix, target = torch.stack(profiles), torch.cat((z, c)) / math.sqrt(2)
    gram, correlations = (matrix @ matrix.T).cpu().numpy(), (matrix @ target).cpu().numpy()
    # Full source and candidate profiles make the costly scan auditable and reusable.
    torch.save({"source": target.cpu(), "profiles": matrix.cpu(), "candidates": candidates,
                "gram": torch.from_numpy(gram), "target_correlations": torch.from_numpy(correlations),
                "concepts": names}, directory / "scan.pt")
    checkpoint()
    coalitions = nonnegative_omp(gram, correlations, candidates, sizes["ks"])
    excluded = {(a["layer"], a["neuron"]) for a in candidates}
    if sizes["source"] == sizes["target"]:
        excluded.add(SOURCE[sizes["source"]][:2])  # same-model smoke only; full 4B->2B is unchanged
    configs = [{"id": "base", "kind": "base", "atoms": [], "doses": []}]
    for k, coalition in coalitions.items():
        random_atoms = matched_controls(model, coalition["atoms"], excluded, sizes["seed"] + k)
        seen_doses = set()
        for total in sizes["totals"]:
            dose = doses(coalition["weights"], total)
            signature = tuple(round(d, 12) for d in dose)
            if signature in seen_doses:
                continue
            seen_doses.add(signature)
            for kind, atoms in (("omp", coalition["atoms"]), ("random", random_atoms)):
                configs.append({"id": f"{kind}-k{k}-D{total}", "kind": kind, "k": k,
                                "nominal_total": total, "realized_total": sum(dose), "atoms": atoms, "doses": dose,
                                "effective_k": coalition["effective_k"]})
    # Legacy reference: original X2 chooses three top Spearman single twins.
    # Reuse its tied-rank implementation without importing another Modal app remotely.
    from x2_twins import ranks
    ranked_source = unit(ranks(z[None]), center=True)[0]
    singles = []
    for atom in candidates:
        col = atom["sign"] * persona.down_col(model, atom["layer"], atom["neuron"])
        sp = float(unit(ranks(lens(model, unit(col)[None])), center=True)[0] @ ranked_source)
        singles.append({**atom, "spearman": sp})
    singles.sort(key=lambda a: (-a["spearman"], a["layer"], a["neuron"]))
    for i, atom in enumerate(singles[:sizes["singles"]]):
        for dose in sizes["single_doses"]:
            configs.append({"id": f"single-{i}-D{dose}", "kind": "single", "single": i,
                            "atoms": [atom], "doses": [dose], "legacy_unbounded": True})
    del profiles, matrix, target, z, c, ll, cp, column, col, weight, columns
    persona.free()
    return model, tok, configs, {"candidates": candidates, "coalitions": coalitions,
                                  "concepts": names, "singles": singles[:sizes["singles"]]}


def generate_records(model, tok, configs, prompts, medians, sizes, directory, stage):
    import persona
    records = []
    path = directory / f"{stage}-texts.jsonl"
    with persona.Hooks(model) as hooks, path.open("x") as output:
        for config in configs:
            hooks.clear()
            if config["kind"] in ("omp", "random"):
                if max(config["doses"]) > 0.35 or sum(config["doses"]) > 0.75 + 1e-12:
                    raise ValueError("coalition dose bound violated")
            for atom, dose in zip(config["atoms"], config["doses"]):
                col = persona.down_col(model, atom["layer"], atom["neuron"])
                hooks.add(atom["layer"], atom["sign"] * dose * float(medians[atom["layer"]]) * unit(col))
            for start in range(0, len(prompts), sizes["batch"]):
                chunk = prompts[start:start + sizes["batch"]]
                outs = persona.generate(model, tok, [persona.chat_ids(tok, p["question"]) for p in chunk], sizes["max_new"])
                for prompt, out in zip(chunk, outs):
                    record = {**prompt, "stage": stage, "config_id": config["id"],
                              "config_hash": digest(config), "concept": "an alien",
                              "reply": tok.decode(out, skip_special_tokens=True)}
                    output.write(json.dumps(record) + "\n")
                    records.append(record)
                output.flush()
            print(f"[x2] {stage} {config['id']}: {len(prompts)} replies", flush=True)
            checkpoint()
    write_json(directory / f"{stage}-generation.json", {"texts_hash": digest(records), "count": len(records)})
    checkpoint()
    return records


def judge_records(records, sizes, directory, stage):
    import torch
    import persona
    import prompts as P
    from judge import score, summary
    path = directory / f"{stage}-scores.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if len(rows) > len(records) or any(r["text_hash"] != digest(t) for r, t in zip(rows, records)):
        raise ValueError("saved labels do not match immutable generation records")
    if len(rows) < len(records):
        judge, tok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]),
                                  "cuda" if torch.cuda.is_available() else "cpu")
        for start in range(len(rows), len(records), sizes["judge_batch"] * 4):
            chunk = records[start:start + sizes["judge_batch"] * 4]
            labels = score(judge, tok, chunk, ["concept", "identity", "coherent"], sizes["judge_batch"])
            for record, label in zip(chunk, labels):
                row = {"prompt_id": record["prompt_id"], "split": record["split"],
                       "config_id": record["config_id"], "config_hash": record["config_hash"],
                       "text_hash": digest(record),
                       "ci": label["concept"] * label["coherent"], "concept": label["concept"],
                       "coherent": label["coherent"], "ib": 1 - label["identity"],
                       "cm": int(P.mentions(record["reply"], "aliens")), "judge": label}
                rows.append(row)
            write_jsonl(path, rows)
            checkpoint()
            print(f"[x2] {stage} scored {len(rows)}/{len(records)}", flush=True)
        del judge, tok
        persona.free()
    if len(rows) != len(records) or any(
        (r["prompt_id"], r["config_id"], r["config_hash"]) !=
        (t["prompt_id"], t["config_id"], t["config_hash"]) for r, t in zip(rows, records)
    ):
        raise ValueError("incomplete or misaligned judge labels; preserve partial artifacts, no summary")
    return rows, summary([r["judge"] for r in rows], ["concept", "identity", "coherent"])


def prompt_split(sizes, final=False):
    import prompts as P
    if set(P.IDENTITY_DEV) & (set(P.IDENTITY_TEST) | set(P.EVERYDAY_TEST)):
        raise ValueError("calibration and final prompts overlap")
    groups = [("identity", P.IDENTITY_TEST[:sizes["test_q"]]),
              ("everyday", P.EVERYDAY_TEST[:sizes["everyday_q"]])] if final else [
                  ("calibration", P.IDENTITY_DEV[:sizes["calibration_q"]])]
    return [{"split": split, "question": q, "prompt_id": digest([split, q])} for split, qs in groups for q in qs]


def freeze(configs, scores, sizes):
    selected, choices = [configs[0]], {}
    for k in sizes["ks"]:
        chosen = select_configuration([c for c in configs if c["kind"] == "omp" and c["k"] == k], scores)
        # Retain a fixed lowest-dose diagnostic when calibration finds no coherent option;
        # the calibration eligibility flag precludes a confirmatory claim for that k.
        eligible = chosen is not None
        chosen = chosen or next(c for c in configs if c["kind"] == "omp" and c["k"] == k)
        control = next(c for c in configs if c["id"] == chosen["id"].replace("omp-", "random-"))
        selected.extend([chosen, control])
        choices[str(k)] = {"selected": chosen["id"], "random": control["id"], "eligible": eligible}
    single_eligibility = {}
    for i in range(sizes["singles"]):
        options = [c for c in configs if c["kind"] == "single" and c["single"] == i]
        choice = select_configuration(options, scores)
        single_eligibility[str(i)] = choice is not None
        selected.append(choice or options[0])
    single_configs = [c for c in selected if c["kind"] == "single"]
    best_single = select_configuration(single_configs, scores)
    eligible_configs = [c for c in selected if c["kind"] == "omp" and choices[str(c["k"])]["eligible"]]
    primary = select_configuration(eligible_configs, scores) if eligible_configs else None
    return {"configs": selected, "choices": choices, "primary": primary["id"] if primary else None,
            "single_eligibility": single_eligibility, "best_single_eligible": best_single is not None,
            "best_single": (best_single or single_configs[0])["id"],
            "calibration_scores_hash": digest(scores), "configs_hash": digest(selected)}


def analyze(scores, frozen, sizes):
    import persona
    groups, comparisons = {}, {}
    for config in frozen["configs"]:
        for split in ("identity", "everyday"):
            rows = [r for r in scores if r["config_id"] == config["id"] and r["split"] == split]
            groups[f"{config['id']}/{split}"] = {**rates(rows), "n": len(rows),
                "rate_intervals": {key: persona.wilson(sum(r[key] for r in rows), len(rows))
                                   for key in ("ci", "concept", "coherent", "ib", "cm")},
                "ci_interval": persona.wilson(sum(r["ci"] for r in rows), len(rows)),
                "coherence_interval": persona.wilson(sum(r["coherent"] for r in rows), len(rows))}

    def bits(ids, split):
        # Pool the replayed three singles at each prompt, preserving paired prompt structure.
        maps = [{r["prompt_id"]: r["ci"] for r in scores if r["config_id"] == cid and r["split"] == split}
                for cid in ids]
        if any(m.keys() != maps[0].keys() for m in maps):
            raise ValueError("comparator has missing prompts")
        return {p: sum(m[p] for m in maps) / len(maps) for p in maps[0]}

    single_ids = [c["id"] for c in frozen["configs"] if c["kind"] == "single"]
    for k, choice in frozen["choices"].items():
        for split in ("identity", "everyday"):
            left = bits([choice["selected"]], split)
            for name, ids in (("pooled_single", single_ids), ("best_single", [frozen["best_single"]]),
                              ("random", [choice["random"]]), ("base", ["base"])):
                comparisons[f"k{k}/{split}/vs_{name}"] = paired_bootstrap(left, bits(ids, split), sizes["boot"], sizes["seed"])
    primary = frozen["primary"]
    metrics = {"calibration_eligible": primary is not None, "success": False,
               "best_single_eligible": frozen["best_single_eligible"],
               "best_single_comparison": "confirmatory" if frozen["best_single_eligible"] else "diagnostic_fallback",
               "best_single_criterion": "not_met" if frozen["best_single_eligible"] else "inconclusive",
               "historical_pooled_single_reference": 0.07}
    if primary:
        config = next(c for c in frozen["configs"] if c["id"] == primary)
        result = groups[f"{primary}/identity"]
        single = comparisons[f"k{config['k']}/identity/vs_best_single"]
        pooled = comparisons[f"k{config['k']}/identity/vs_pooled_single"]
        control = comparisons[f"k{config['k']}/identity/vs_random"]
        metrics.update(primary_k=config["k"], ci=result["ci"], marginal_concept=result["concept"], coherence=result["coherent"],
                       ci_low=result["ci_interval"][0], ci_high=result["ci_interval"][1],
                       vs_pooled_single=pooled["difference"], vs_pooled_single_low=pooled["ci95"][0],
                       vs_pooled_single_high=pooled["ci95"][1], vs_best_single=single["difference"],
                       vs_best_single_low=single["ci95"][0], vs_best_single_high=single["ci95"][1], vs_random=control["difference"],
                       vs_random_low=control["ci95"][0], vs_random_high=control["ci95"][1])
        if frozen["best_single_eligible"] and single["difference"] >= 0.25 and single["ci95"][0] > 0:
            metrics["best_single_criterion"] = "met"
        metrics["success"] = (frozen["best_single_eligible"] and result["ci"] >= 0.4 and result["coherent"] >= 0.8 and
                              single["difference"] >= 0.25 and single["ci95"][0] > 0 and control["ci95"][0] > 0)
    return metrics, {"groups": groups, "paired_comparisons": comparisons,
                     "inference": "conditional on independent frozen calibration; no test-wide coherence gating",
                     "historical_reference": "0.07 pooled three-twin marginal concept rate at a selected dose with coherence>=0.8; "
                         "new ci is per-reply concept AND coherent, concept reports the marginal rate; context only, no raw sample inference"}


def validate_records(records, configs, prompts):
    expected = {(p["prompt_id"], c["id"]): (p, c) for p in prompts for c in configs}
    actual = [(r["prompt_id"], r["config_id"]) for r in records]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("incomplete or duplicate generation records; preserve artifacts and report inconclusive")
    for row in records:
        prompt, config = expected[row["prompt_id"], row["config_id"]]
        if row["config_hash"] != digest(config) or any(row[k] != prompt[k] for k in prompt):
            raise ValueError("generation record does not match frozen prompts/configuration")


def validate_text_hash(records, directory, stage):
    completed = json.loads((directory / f"{stage}-generation.json").read_text())
    if completed != {"texts_hash": digest(records), "count": len(records)}:
        raise ValueError("immutable generated texts changed; refuse score-only resume")


def file_hash(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


@app.function(image=image.add_local_python_source("x2_twins"),
              volumes={**hf_cache.volumes, "/out": OUT}, gpu="A10G", memory=24576, timeout=3600)
def run(stage: str, tag: str, sizes: dict, manifest: dict, artifacts: dict):
    with stage_claim(tag, stage):
        return run_stage(stage, tag, sizes, manifest, artifacts)


def run_stage(stage: str, tag: str, sizes: dict, manifest: dict, artifacts: dict):
    import torch
    import persona
    start = time.monotonic()
    if stage not in ("calibrate", "score-calibration", "final", "score-final"):
        raise ValueError("stage must be calibrate, score-calibration, final or score-final")
    if not modal.is_local():
        OUT.reload()
    directory = out_dir(tag)
    phase = "calibration" if stage in ("calibrate", "score-calibration") else "final"
    timing_path = directory / f"{phase}-timings.json"
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    medians = torch.load(out_dir(artifacts["target_s1a"]) / "calib.pt", map_location="cpu", weights_only=True)[
        "pile"]["norm_q10_50_90"][:, 1]
    if not torch.isfinite(medians).all() or (medians <= 0).any():
        raise ValueError("invalid residual norm calibration")
    if stage in ("calibrate", "score-calibration"):
        if stage == "calibrate":
            # Shared claim excludes concurrent runs; persistent markers reject sequential regeneration.
            with (directory / "calibration.started").open("x") as handle:
                handle.write(digest(manifest))
            source_path = out_dir(artifacts["source_s1b"]) / "axis.pt"
            target_path = out_dir(artifacts["target_s1b"]) / "axis.pt"
            artifact_hashes = {"source_s1b": file_hash(source_path), "target_s1b": file_hash(target_path),
                               "target_s1a": file_hash(out_dir(artifacts["target_s1a"]) / "calib.pt")}
            write_json(directory / "manifest.json", {**manifest, "artifacts": artifacts, "artifact_hashes": artifact_hashes})
            checkpoint()
            clock = time.monotonic()
            source_axis = torch.load(source_path, map_location="cpu", weights_only=True)
            target_axis = torch.load(target_path, map_location="cpu", weights_only=True)
            with torch.inference_mode():
                model, tok, configs, search_info = search(sizes, source_axis, target_axis, directory)
            timings["scan_seconds"] = time.monotonic() - clock
            write_json(timing_path, timings)
            print(f"[x2] scan_seconds={timings['scan_seconds']:.3f}", flush=True)
            write_json(directory / "search.json", search_info)
            write_json(directory / "candidate-configs.json", configs)
            write_json(directory / "scan-manifest.json", {"scan_hash": file_hash(directory / "scan.pt"),
                       "configs_hash": digest(configs), "manifest_hash": digest(manifest)})
            checkpoint()
            # Controls inherit OMP doses; no redundant calibration or outcome optimization.
            calibration_configs = [c for c in configs if c["kind"] != "random"]
            clock = time.monotonic()
            records = generate_records(model, tok, calibration_configs, prompt_split(sizes), medians, sizes, directory, "calibration")
            timings["generation_seconds"] = time.monotonic() - clock
            write_json(timing_path, timings)
            checkpoint()
            print(f"[x2] calibration_generation_seconds={timings['generation_seconds']:.3f}", flush=True)
            del model, tok, source_axis, target_axis
            persona.free()
        else:
            saved_manifest = json.loads((directory / "manifest.json").read_text())
            if any(saved_manifest[k] != json.loads(json.dumps(v)) for k, v in manifest.items()):
                raise ValueError("implementation changed before score-only resume")
            artifact_hashes = saved_manifest["artifact_hashes"]
            configs = json.loads((directory / "candidate-configs.json").read_text())
            if digest(configs) != json.loads((directory / "scan-manifest.json").read_text())["configs_hash"]:
                raise ValueError("saved scan configuration changed")
            calibration_configs = [c for c in configs if c["kind"] != "random"]
            records = [json.loads(line) for line in (directory / "calibration-texts.jsonl").read_text().splitlines()]
        validate_records(records, calibration_configs, prompt_split(sizes))
        validate_text_hash(records, directory, "calibration")
        clock = time.monotonic()
        scores, judge_summary = judge_records(records, sizes, directory, "calibration")
        timings["judge_seconds"] = timings.get("judge_seconds", 0) + time.monotonic() - clock
        frozen = {**freeze(configs, scores, sizes), "manifest_hash": digest(manifest),
                  "calibration_prompt_hash": digest(prompt_split(sizes)),
                  "final_prompt_hash": digest(prompt_split(sizes, final=True)),
                  "artifact_hashes": artifact_hashes, "judge_summary": judge_summary}
        frozen["freeze_hash"] = digest(frozen)
        freeze_path = directory / "frozen.json"
        if freeze_path.exists() and json.loads(freeze_path.read_text()) != frozen:
            raise ValueError("score resume would change frozen selection")
        if not freeze_path.exists():
            write_json(freeze_path, frozen)
        checkpoint()
        metrics = {"stage": stage, "calibration_records": len(records), "primary": frozen["primary"],
                   "freeze_hash": frozen["freeze_hash"], "final_records_planned":
                       len(frozen["configs"]) * (sizes["test_q"] + sizes["everyday_q"])}
    else:
        frozen = json.loads((directory / "frozen.json").read_text())
        freeze_hash = frozen.pop("freeze_hash")
        if digest(frozen) != freeze_hash or frozen["manifest_hash"] != digest(manifest):
            raise ValueError("freeze or implementation manifest changed after calibration")
        frozen["freeze_hash"] = freeze_hash
        if frozen["configs_hash"] != digest(frozen["configs"]):
            raise ValueError("frozen configurations changed")
        if frozen["final_prompt_hash"] != digest(prompt_split(sizes, final=True)):
            raise ValueError("final prompts changed after calibration")
        if frozen["artifact_hashes"]["target_s1a"] != file_hash(out_dir(artifacts["target_s1a"]) / "calib.pt"):
            raise ValueError("residual calibration changed after selection")
        if stage == "final":
            with (directory / "final.started").open("x") as handle:
                handle.write(freeze_hash)
            checkpoint()
            clock = time.monotonic()
            model, tok = persona.load(hf_cache.model(*MODELS[sizes["target"]]), "cuda" if torch.cuda.is_available() else "cpu")
            records = generate_records(model, tok, frozen["configs"], prompt_split(sizes, final=True), medians, sizes, directory, "final")
            timings["generation_seconds"] = time.monotonic() - clock
            write_json(timing_path, timings)
            checkpoint()
            print(f"[x2] final_generation_seconds={timings['generation_seconds']:.3f}", flush=True)
            del model, tok
            persona.free()
        else:
            records = [json.loads(line) for line in (directory / "final-texts.jsonl").read_text().splitlines()]
        validate_records(records, frozen["configs"], prompt_split(sizes, final=True))
        validate_text_hash(records, directory, "final")
        clock = time.monotonic()
        scores, judge_summary = judge_records(records, sizes, directory, "final")
        timings["judge_seconds"] = timings.get("judge_seconds", 0) + time.monotonic() - clock
        metrics, details = analyze(scores, frozen, sizes)
        metrics.update(stage=stage, final_records=len(records), freeze_hash=freeze_hash)
        write_json(directory / "x2.json", {"metrics": metrics, "details": details, "judge_summary": judge_summary})
        checkpoint()
    print(f"[x2] {stage}_judge_seconds={timings['judge_seconds']:.3f}", flush=True)
    metrics.update(timings)
    metrics["minutes"] = (time.monotonic() - start) / 60
    write_json(timing_path, timings)
    write_json(directory / f"{stage}-timings.json", timings)
    checkpoint()
    return metrics


def design_manifest(commit, sizes):
    if not commit or any(c not in "0123456789abcdef" for c in commit) or len(commit) < 7:
        raise ValueError("supply the committed DESIGN.md revision before any compute")
    root = HERE.parent.parent
    revision = subprocess.check_output(["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=root, text=True).strip()
    design = subprocess.check_output(["git", "show", f"{revision}:{DESIGN_PATH}"], cwd=root)
    paths = [pathlib.Path(__file__), OLD / "persona.py", OLD / "judge.py", OLD / "jobs.py",
             OLD / "prompts.py", OLD / "x2_twins.py", HERE.parent / "hf_cache.py"]
    return {"design_commit": revision, "design_hash": hashlib.sha256(design).hexdigest(),
            "implementation": {p.name: file_hash(p) for p in paths}, "sizes": sizes,
            "models": {key: MODELS[key] for key in (sizes["source"], sizes["target"], sizes["judge"])},
            "rubrics": ["concept", "identity", "coherent"],
            "primary_outcome": "first-person alien adoption AND coherence per reply"}


@app.local_entrypoint()
def main(stage: str = "calibrate", smoke: bool = False, tag: str = "", design_commit: str = "",
         target_s1a: str = "", target_s1b: str = "", source_s1b: str = ""):
    sizes = dict(SMOKE if smoke else FULL)
    manifest = design_manifest(design_commit, sizes)
    suffix = "-smoke" if smoke else ""
    artifacts = {"target_s1a": target_s1a or f"s1a-{sizes['target']}{suffix}",
                 "target_s1b": target_s1b or f"s1b-{sizes['target']}{suffix}",
                 "source_s1b": source_s1b or f"s1b-{sizes['source']}{suffix}"}
    if not tag:
        if not smoke:
            raise ValueError("paid stages require an explicit unique --tag, reused for final")
        tag = f"x2-coalitions-smoke-{time.time_ns()}"
    for phase in (["calibrate", "score-calibration", "final", "score-final"] if smoke else [stage]):
        result = call(run, phase, tag, sizes, manifest, artifacts)
        print("PERSONAKIT_RESULT", json.dumps({**result, "smoke": smoke, "tag": tag}), flush=True)
