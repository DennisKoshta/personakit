"""Frozen H4-v2 selection and inference (design ebce16d; owner note 89a43d6).

No model imports/execution. Primary: stratified sample AUROC, paired bootstrap.
Secondary: ratio of joint-inclusion HT pair totals CONDITIONAL on the realized
TF pool and certainty selections; its bootstrap fixes certainty and is approximate.
"""
import hashlib
import math
import random


STRATA = ("top_s_full", "top_w_comp", "top_peakiness", "top_cos_away", "rand_4L", "rand_15L")
BASELINES = ("cos_away", "peakiness", "over_natural", "neg_over_natural", "F_reach")
PREDICTORS = ("S_full", "W_comp", *BASELINES, "neg_cos_away", "neg_peakiness", "neg_F_reach")


def unit_id(layer, neuron, sign):
    if sign not in (-1, 1):
        raise ValueError("invalid sign")
    return f"L{layer}N{neuron}{'+' if sign == 1 else '-'}"


def tie_key(row):
    return row["layer"], row["neuron"], -row["sign"]


def robust_fit(values):
    import numpy as np

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("normalization requires finite nonempty observations")
    q1, median, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return {"median": float(median), "scale": float((q3 - q1) / 1.349 + 1e-8), "n": len(x)}


def robust_z(value, fit):
    import numpy as np

    if not math.isfinite(value) or not math.isfinite(fit["scale"]) or fit["scale"] <= 0:
        raise ValueError("invalid normalization")
    return float(np.clip((value - fit["median"]) / fit["scale"], -5, 5))


def weight_scores(population):
    """Freeze medians/IQR over the entire eligible unlabeled population."""
    fits = {k: robust_fit([u[k] for u in population]) for k in ("F_align", "peakiness")}
    for u in population:
        u["Z_align"] = robust_z(u["F_align"], fits["F_align"])
        u["Z_peak"] = robust_z(u["peakiness"], fits["peakiness"])
        u["W_comp"] = u["F_reach"] * (u["Z_align"] + u["Z_peak"])
    return fits


def rank(rows, feature):
    if any(not math.isfinite(u[feature]) for u in rows):
        raise ValueError(f"missing/nonfinite ranking feature {feature}")
    return sorted(rows, key=lambda u: (-u[feature], *tie_key(u)))


def sampling_provenance(seed):
    """Domain-separated streams; seed remains the frozen master seed.

    The rollout stream must be independent of the realized TF-pool draw for
    conditional inclusion probabilities to describe the sampling algorithm.
    Persist the derivation and full child seeds, not Python's randomized hash().
    """
    if type(seed) is not int:
        raise ValueError("sampling master seed must be an integer")
    prefix = "h4v2-sampling-v1"
    return {"master_seed": seed, "generator": "random.Random/MT19937",
            "derivation": "SHA256(UTF-8(h4v2-sampling-v1:<master_seed>:<phase>)); big-endian integer",
            "phase_seeds_hex": {phase: hashlib.sha256(f"{prefix}:{seed}:{phase}".encode("utf-8")).hexdigest()
                                for phase in ("tf_pool", "rollout")}}


def sampling_rng(seed, phase):
    return random.Random(int(sampling_provenance(seed)["phase_seeds_hex"][phase], 16))


def tf_pool(population, top_count, random_count, seed):
    """Top W_comp plus SRS without replacement from its FULL complement."""
    if len({u["id"] for u in population}) != len(population):
        raise ValueError("duplicate population units")
    top = rank(population, "W_comp")[:top_count]
    ids = {u["id"] for u in top}
    complement = sorted((u for u in population if u["id"] not in ids), key=tie_key)
    if len(top) != top_count or len(complement) < random_count or min(top_count, random_count) < 1:
        raise ValueError("insufficient TF population")
    rand = sampling_rng(seed, "tf_pool").sample(complement, random_count)
    return {"top": [u["id"] for u in top], "random": [u["id"] for u in rand]}


def full_scores(rows, fit):
    for u in rows:
        u["S_full"] = u["F_reach"] * (u["Z_align"] + u["Z_peak"] + robust_z(u["Delta_TF"], fit))


def select_strata(population, pool, quotas, prior_layers, seed):
    """Disjoint ranked strata, then two SRS draws from the entire remainder."""
    if set(quotas) != set(STRATA) or any(n < 1 for n in quotas.values()):
        raise ValueError("all six strata require positive quotas")
    lookup = {u["id"]: u for u in population}
    pool_ids = pool["top"] + pool["random"]
    if len(lookup) != len(population) or len(set(pool_ids)) != len(pool_ids):
        raise ValueError("duplicate population/TF pool")
    picked, used = [], set()
    for stratum, feature in zip(STRATA[:4], ("S_full", "W_comp", "peakiness", "cos_away")):
        eligible = [lookup[i] for i in pool_ids] if stratum == "top_s_full" else population
        eligible = [u for u in eligible if u["id"] not in used]
        chosen = rank(eligible, feature)[:quotas[stratum]]
        if len(chosen) != quotas[stratum]:
            raise ValueError(f"insufficient {stratum} candidates")
        for u in chosen:
            picked.append({**u, "stratum": stratum, "domain": "certainty", "pi": 1.0})
            used.add(u["id"])
    domains = {}
    rng = sampling_rng(seed, "rollout")
    for name, inside in (("rand_4L", True), ("rand_15L", False)):
        remainder = sorted((u for u in population if u["id"] not in used
                            and (u["layer"] in prior_layers) == inside), key=tie_key)
        n, N = quotas[name], len(remainder)
        if n < 2 or N < n:
            raise ValueError("random domains need at least two units for joint-pair inclusion")
        domains[name] = {"N": N, "n": n}
        for u in rng.sample(remainder, n):
            picked.append({**u, "stratum": name, "domain": name, "pi": n / N})
            used.add(u["id"])
    if len(picked) != sum(quotas.values()) or len(used) != len(picked):
        raise AssertionError("incorrect disjoint selection")
    return picked, domains


def outcomes(selected, labeled, prompt_count, doses=(0.5, 0.75)):
    """No dropping incomplete cells: any missing row makes that unit undefined."""
    grouped, row_ids = {}, set()
    for r in labeled:
        if r["id"] in row_ids:
            raise ValueError("duplicate outcome row ID")
        row_ids.add(r["id"])
        if r["kind"] != "sample":
            continue
        for rubric in ("identity", "coherent", "concept"):
            if type(r["labels"].get(rubric)) is not int or r["labels"][rubric] not in (0, 1):
                raise ValueError("missing/nonbinary rubric label")
        grouped.setdefault((r["unit_id"], r["dose"]), []).append(r)
    valid_cells = {(u["id"], d) for u in selected for d in doses}
    if set(grouped) - valid_cells:
        raise ValueError("unexpected sample unit/dose")
    result = []
    for u in selected:
        cells, complete = {}, True
        for d in doses:
            rows = grouped.get((u["id"], d), [])
            qs = [r["q"] for r in rows]
            if len(qs) != len(set(qs)) or set(qs) - set(range(prompt_count)):
                raise ValueError("duplicate/out-of-range prompt index")
            if set(qs) != set(range(prompt_count)):
                cells[str(d)] = None
                complete = False
                continue
            cells[str(d)] = {"ib": sum(1-r["labels"]["identity"] for r in rows)/len(rows),
                             "coherence": sum(r["labels"]["coherent"] for r in rows)/len(rows),
                             "ci": sum(r["labels"]["concept"] for r in rows)/len(rows),
                             "cm": sum(r["cm"] for r in rows)/len(rows), "n": len(rows)}
        good = [c for c in cells.values() if c is not None and c["coherence"] >= .8]
        result.append({**u, "cells": cells, "complete": complete,
                       "positive": any(c["ib"] >= .4 for c in good) if complete else None,
                       "continuous": max([0.] + [c["ib"] for c in good]) if complete else None,
                       "persona": any(c["ci"] >= .5 for c in good) if complete else None})
    return result


def targets(rows, prior_breakers, max_layer=22):
    eligible = []
    for u in rows:
        if not u["complete"] or u["id"] in prior_breakers or u["layer"] > max_layer:
            continue
        good = [(float(d), c) for d, c in u["cells"].items() if c["ci"] >= .5 and c["coherence"] >= .8]
        if good:
            d, c = min(good)
            eligible.append((u, d, c))
    eligible.sort(key=lambda x: (-x[2]["ci"], -x[0]["S_full"], *tie_key(x[0])))
    return [{**{k: u[k] for k in ("id", "layer", "neuron", "sign", "concept", "concept_key")},
             "dose": d, "ci": c["ci"], "coherence": c["coherence"], "exploratory": True}
            for u, d, c in eligible[:3]]


def ranks(values):
    import numpy as np

    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1)/2
        i = j
    return result


def spearman(x, y):
    import numpy as np

    a, b = ranks(x), ranks(y)
    a, b = a-a.mean(), b-b.mean()
    denom = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(a @ b / denom) if denom else None


def joint_inclusion(a, b, domains):
    """Distinct finite-population pair probability, conditional on first phase."""
    if a["id"] == b["id"]:
        return 0.0
    da, db = a["domain"], b["domain"]
    if da == db == "certainty":
        return 1.0
    def pi(d):
        return 1. if d == "certainty" else domains[d]["n"]/domains[d]["N"]
    if da == db:
        n, N = domains[da]["n"], domains[da]["N"]
        return n*(n-1)/(N*(N-1)) if N > 1 else 0.
    return pi(da)*pi(db)


def interval(values, reps):
    import numpy as np

    defined = [x for x in values if x is not None and math.isfinite(x)]
    return {"ci95": np.quantile(defined, [.025, .975]).tolist() if len(defined) >= reps/2 and defined else None,
            "defined": len(defined), "undefined": reps-len(defined), "replicates": reps}


def analyse(rows, domains, prior_breakers, reps=2000, seed=0):
    """All predictors frozen upstream; no label-driven orientation/score choice."""
    import numpy as np

    if len({u["id"] for u in rows}) != len(rows):
        raise ValueError("duplicate analysis units")
    missing = [u["id"] for u in rows if not u["complete"]]
    if not rows or missing:
        return {"verdict": "inconclusive", "reason": "missing outcomes", "missing": missing,
                "n": len(rows), "sample": None, "population": None, "new_personas": None}
    if not isinstance(reps, int) or reps < 1:
        raise ValueError("positive bootstrap count required")
    for domain, info in domains.items():
        if info["n"] < 2 or info["n"] > info["N"] or sum(u["domain"] == domain for u in rows) != info["n"]:
            raise ValueError("invalid domain size")
    scores = {k: np.array([-u[k[4:]] if k.startswith("neg_") else u[k] for u in rows]) for k in PREDICTORS}
    if any(not np.isfinite(v).all() for v in scores.values()):
        raise ValueError("nonfinite analysis predictor")
    y = np.array([u["positive"] for u in rows], dtype=bool)
    continuous = np.array([u["continuous"] for u in rows])
    pairs = y[:, None] & ~y[None, :]
    weights = np.zeros((len(rows), len(rows)))
    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            if i != j:
                pi = joint_inclusion(a, b, domains)
                if not 0 < pi <= 1:
                    raise ValueError("nonpositive pair inclusion")
                weights[i, j] = 1/pi
    concordance = np.array([(v[:, None] > v[None, :]) + .5*(v[:, None] == v[None, :])
                            for v in scores.values()])
    sample_matrix, population_matrix = concordance*pairs, concordance*pairs*weights
    def aucs(counts, population=False):
        pair_weights = pairs*weights if population else pairs
        denominator = float(np.einsum("i,ij,j->", counts, pair_weights, counts))
        if not denominator:
            return {k: None for k in PREDICTORS}
        matrix = population_matrix if population else sample_matrix
        numerators = np.einsum("i,kij,j->k", counts, matrix, counts, optimize=False)
        return {k: float(v/denominator) for k, v in zip(PREDICTORS, numerators)}
    ones = np.ones(len(rows))
    sample, population = aucs(ones), aucs(ones, True)
    groups = {s: np.array([i for i, u in enumerate(rows) if u["stratum"] == s]) for s in STRATA}
    if any(not len(ix) for ix in groups.values()):
        raise ValueError("missing stratum")
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in PREDICTORS}
    pops = {k: [] for k in PREDICTORS}
    rhos = {k: [] for k in PREDICTORS}
    diffs = {f"{a}_minus_{b}": [] for a in ("S_full", "W_comp") for b in BASELINES}
    certainty = np.array([i for i, u in enumerate(rows) if u["domain"] == "certainty"], dtype=int)
    for _ in range(reps):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups.values()])
        got = aucs(np.bincount(ix, minlength=len(rows)))
        pi = np.concatenate([certainty, *[rng.choice(groups[s], len(groups[s]), replace=True)
                                          for s in ("rand_4L", "rand_15L")]])
        pg = aucs(np.bincount(pi, minlength=len(rows)), True)
        for k in PREDICTORS:
            boots[k].append(got[k]); pops[k].append(pg[k])
            rhos[k].append(spearman(scores[k][ix], continuous[ix]))
        for a in ("S_full", "W_comp"):
            for b in BASELINES:
                diffs[f"{a}_minus_{b}"].append(None if got[a] is None else got[a]-got[b])
    sample = {k: {"auroc": sample[k], **interval(boots[k], reps),
                  "spearman": spearman(scores[k], continuous), "spearman_bootstrap": interval(rhos[k], reps)}
              for k in PREDICTORS}
    population = {k: {"auroc": population[k], **interval(pops[k], reps)} for k in PREDICTORS}
    differences = {k: interval(v, reps) for k, v in diffs.items()}
    for a in ("S_full", "W_comp"):
        for b in BASELINES:
            differences[f"{a}_minus_{b}"]["difference"] = (sample[a]["auroc"]-sample[b]["auroc"]
                                                                if sample[a]["auroc"] is not None else None)
    discoveries = sum(u["persona"] and u["id"] not in prior_breakers for u in rows)
    criteria, supported, ruled_out = {}, {}, {}
    for a in ("S_full", "W_comp"):
        au, ci = sample[a]["auroc"], sample[a]["ci95"]
        diff_cis = [differences[f"{a}_minus_{b}"]["ci95"] for b in BASELINES]
        superiority = (False if any(c is not None and c[0] <= 0 for c in diff_cis)
                       else None if any(c is None for c in diff_cis) else True)
        criteria[a] = (None if au is None else False if au < .7
                       else True if discoveries >= 5 else superiority)
        supported[a] = ci[0] >= .7 if ci is not None else None
        ruled_out[a] = ci[1] < .7 if ci is not None else None
    # Lead clarification: point criterion and strength of evidence are distinct.
    evidence_yes = any(criteria[a] is True and supported[a] is True for a in criteria)
    verdict = "yes" if evidence_yes else "no" if all(v is True for v in ruled_out.values()) and discoveries < 5 else "inconclusive"
    four = np.array([i for i, u in enumerate(rows) if u["stratum"] == "rand_4L"])
    four_au = aucs(np.bincount(four, minlength=len(rows)))
    return {"verdict": verdict, "n": len(rows), "positives": int(y.sum()), "new_personas": int(discoveries),
            "criterion_met": criteria, "threshold_supported": supported, "threshold_ruled_out": ruled_out,
            "sample": sample, "paired_differences": differences, "population": population,
            "population_estimand": "ratio of joint-inclusion HT pair totals; conditional on realized TF pool and certainty",
            "population_ci_caveat": "approximate conditional bootstrap, certainty fixed; no unconditional coverage claim",
            "prior_4L_random_auroc": four_au, "baseline_superiority_set": list(BASELINES),
            "point_criterion": "estimate >= .70 AND (all frozen paired-difference CI lower >0 OR >=5 new personas)",
            "uncertainty_gate": "criterion_met is independent of threshold_supported; evidence inconclusive when CI spans .70 or is undefined"}


def diagnostic_rates(labeled):
    """Known and shared unsteered baseline rates, NEVER part of primary sample."""
    groups = {}
    for row in labeled:
        if row["kind"] != "sample":
            groups.setdefault((row["kind"], row["unit_id"], row["dose"]), []).append(row)
    return [{"kind": kind, "unit_id": uid, "dose": dose, "n": len(rs),
             "concept": rs[0]["concept"], "concept_key": rs[0]["concept_key"], "cm_proxy": rs[0]["cm_proxy"],
             "ib": sum(1-r["labels"]["identity"] for r in rs)/len(rs),
             "ci": sum(r["labels"]["concept"] for r in rs)/len(rs),
             "coherence": sum(r["labels"]["coherent"] for r in rs)/len(rs),
             "cm": sum(r["cm"] for r in rs)/len(rs), "diagnostic_only": True}
            for (kind, uid, dose), rs in sorted(groups.items())]
