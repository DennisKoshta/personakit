"""Paired H2-v2 analysis; stdlib only, no models and no completion text.

Input is a complete target x condition x prompt matrix. Labels join by immutable
record ID and judge-input hash, never list position. New targets are never pooled
with the fixed anchor. Run --synthetic for severe silent mapping/ratio invariants.
"""

import argparse
import json
import math
from pathlib import Path
import random

CONDITIONS = ("none", "unsteered_swap_attn", "unsteered_swap_mlp", "steer",
              "swap_attn", "swap_mlp", "sham_swap_attn", "self_ref_cap", "axis_cap", "random_cap")
PRIMARY = ("swap_attn", "swap_mlp", "self_ref_cap")
METRICS = ("ci", "cm", "coherent", "ib")


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def interval(values, reps):
    defined = sorted(v for v in values if v is not None)
    if any(not math.isfinite(v) for v in defined):
        raise ValueError("nonfinite statistic")
    bounds = None
    if len(defined) >= max(1, reps / 2):
        bounds = [defined[int(.025 * (len(defined) - 1))], defined[int(.975 * (len(defined) - 1))]]
    return {"ci95": bounds, "defined": len(defined), "undefined": reps - len(defined)}


def mcnemar_less(baseline, intervention):
    """Exact one-sided matched binary test: intervention has fewer claims."""
    lost = sum(a == 1 and b == 0 for a, b in zip(baseline, intervention))
    gained = sum(a == 0 and b == 1 for a, b in zip(baseline, intervention))
    n = lost + gained
    return sum(math.comb(n, k) for k in range(lost, n + 1)) / 2**n


def analyse(records, labels, targets, prompt_ids, *, reps=2000, seed=0, smoke=False):
    if not isinstance(reps, int) or reps < 1:
        raise ValueError("positive bootstrap count required")
    if not prompt_ids or len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("empty or duplicate prompt IDs")
    ids = [t["id"] for t in targets]
    if not ids or len(ids) != len(set(ids)) or sum(not t["exploratory"] for t in targets) != 1:
        raise ValueError("exactly one fixed anchor and unique targets required")
    condition_list = list(CONDITIONS)
    if any(r["condition"] == "random_weight_specificity" for r in records):
        condition_list.append("random_weight_specificity")
    label_map = {}
    for label in labels:
        if label["record_id"] in label_map:
            raise ValueError("duplicate judge label ID")
        label_map[label["record_id"]] = label
    matrix, seen_ids = {}, set()
    for record in records:
        rid = record["record_id"]
        key = (record["target_id"], record["condition"], record["prompt_id"])
        if rid in seen_ids or key in matrix:
            raise ValueError("duplicate record or condition/prompt mapping")
        seen_ids.add(rid)
        if rid not in label_map:
            raise ValueError("missing judge label")
        lab = label_map[rid]
        if lab["input_sha256"] != record["input_sha256"]:
            raise ValueError("stale or misaligned judge input")
        for rubric in ("identity", "concept", "coherent"):
            if type(lab.get(rubric)) is not int or lab[rubric] not in (0, 1):
                raise ValueError("missing or invalid rubric")
        if type(record.get("cm")) is not int or record["cm"] not in (0, 1):
            raise ValueError("missing or invalid lexical CM")
        matrix[key] = {"ci": lab["concept"], "cm": record["cm"],
                       "coherent": lab["coherent"], "ib": 1 - lab["identity"]}
    if seen_ids != set(label_map):
        raise ValueError("unmatched judge labels")
    expected = {(tid, cond, pid) for tid in ids for cond in CONDITIONS for pid in prompt_ids}
    anchor = next(t["id"] for t in targets if not t["exploratory"])
    if "random_weight_specificity" in condition_list:
        expected |= {(anchor, "random_weight_specificity", pid) for pid in prompt_ids}
    if set(matrix) != expected:
        raise ValueError("incomplete or unexpected target/condition/prompt matrix")

    # The SAME sampled prompt IDs feed every condition and metric in a replicate.
    rng = random.Random(seed)
    sampled = [[rng.randrange(len(prompt_ids)) for _ in prompt_ids] for _ in range(reps)]
    result = {}
    for target in targets:
        tid = target["id"]
        conds = [c for c in condition_list if (tid, c, prompt_ids[0]) in matrix]
        bits = {c: {m: [matrix[tid, c, p][m] for p in prompt_ids] for m in METRICS} for c in conds}
        rates = {c: {m: sum(v) / len(v) for m, v in row.items()} for c, row in bits.items()}
        boot_rates = {c: {m: [sum(v[i] for i in ix) / len(ix) for ix in sampled]
                          for m, v in row.items()} for c, row in bits.items()}
        stats = {}
        for cond in conds:
            row = {"rates": {m: {"estimate": rates[cond][m], **interval(boot_rates[cond][m], reps)}
                              for m in METRICS}}
            for metric in ("ci", "cm"):
                row[f"{metric}_ratio"] = {
                    "estimate": ratio(rates[cond][metric], rates["steer"][metric]),
                    **interval([ratio(a, b) for a, b in zip(boot_rates[cond][metric], boot_rates["steer"][metric])], reps)}
            row["p_ci_below_steer"] = mcnemar_less(bits["steer"]["ci"], bits[cond]["ci"])
            stats[cond] = row
        contrast = {
            "estimate": ratio(rates["swap_mlp"]["ci"] - rates["swap_attn"]["ci"], rates["steer"]["ci"]),
            **interval([ratio(m - a, s) for m, a, s in zip(boot_rates["swap_mlp"]["ci"],
                        boot_rates["swap_attn"]["ci"], boot_rates["steer"]["ci"])], reps),
            "definition": "CI_ratio(swap_mlp) - CI_ratio(swap_attn); positive = stronger attention suppression"}
        qualified = rates["steer"]["ci"] >= .5 and rates["steer"]["coherent"] >= .8
        conclusions = {}
        for cond in PRIMARY:
            control = "random_cap" if cond == "self_ref_cap" else "sham_swap_attn"
            ci, cm, cr = (stats[c][m] for c, m in ((cond, "ci_ratio"), (cond, "cm_ratio"), (control, "ci_ratio")))
            coh = stats[cond]["rates"]["coherent"]
            control_coh = stats[control]["rates"]["coherent"]
            valid = qualified and all(v["estimate"] is not None for v in (ci, cm, cr))
            falsifier_valid = qualified and ci["estimate"] is not None
            prediction = (ci["estimate"] <= .5 and cm["estimate"] >= .7 and cr["estimate"] >= .8
                          and coh["estimate"] >= .8 and control_coh["estimate"] >= .8
                          and stats[cond]["p_ci_below_steer"] < .05) if valid else None
            # A straddling / undefined interval cannot establish a study verdict.
            strong = valid and all(v["ci95"] is not None for v in (ci, cm, cr, coh, control_coh))
            yes = (strong and prediction and ci["ci95"][1] <= .5 and cm["ci95"][0] >= .7
                   and cr["ci95"][0] >= .8 and coh["ci95"][0] >= .8 and control_coh["ci95"][0] >= .8)
            falsifier_strong = falsifier_valid and all(v["ci95"] is not None for v in (ci, coh, control_coh))
            no = falsifier_strong and ci["ci95"][0] >= .8 and coh["ci95"][0] >= .8 and control_coh["ci95"][0] >= .8
            conclusions[cond] = {"control": control, "point_prediction": prediction,
                                 "point_falsifier": (ci["estimate"] >= .8 and coh["estimate"] >= .8
                                                      and control_coh["estimate"] >= .8) if falsifier_valid else None,
                                 "verdict": "yes" if yes else "no" if no else "inconclusive"}
            if target.get("cm_proxy"):
                # Retaining a nearest-family word is not evidence of retaining
                # the assigned role's meaning. Preserve descriptive ratios only.
                conclusions[cond].update(exploratory_proxy_pattern=prediction, point_prediction=None,
                                          point_falsifier=None, verdict="inconclusive")
        verdicts = [c["verdict"] for c in conclusions.values()]
        verdict = "yes" if "yes" in verdicts else "no" if all(v == "no" for v in verdicts) else "inconclusive"
        if smoke:
            verdict = "smoke_only"
        result[tid] = {"target": target, "qualified_baseline": qualified, "conditions": stats,
                       "mechanisms": conclusions, "signed_attn_vs_mlp": contrast, "verdict": verdict,
                       "status": "exploratory_discovery_conditioned" if target["exploratory"] else "fixed_anchor",
                       "cm_interpretation": "nearest-family proxy" if target.get("cm_proxy") else "matched concept family"}
    return {"anchor_id": anchor, "anchor_verdict": result[anchor]["verdict"], "targets": result,
            "prompt_clusters": len(prompt_ids), "bootstrap_replicates": reps, "seed": seed,
            "intervals": "paired percentile bootstrap over prompt IDs; undefined ratios counted",
            "limitations": "Framing-associated contrast, not pure voice; exploratory targets are never pooled with anchor"}


def analyse_checkpoint(data):
    """Replay the exact analysis settings; old/missing provenance fails closed."""
    settings = data.get("analysis_config")
    if (not isinstance(settings, dict) or set(settings) != {"seed", "reps"}
            or type(settings["seed"]) is not int or type(settings["reps"]) is not int
            or settings["reps"] < 1 or type(data.get("smoke")) is not bool):
        raise ValueError("checkpoint must record analysis_config seed/reps and smoke explicitly")
    return analyse(data["records"], data["labels"], data["targets"], data["prompt_ids"],
                   seed=settings["seed"], reps=settings["reps"], smoke=data["smoke"])


def synthetic_checks():
    """Only silent invariants: joins, denominator direction, pairing and no pooling."""
    import copy
    targets = [{"id": "anchor", "exploratory": False}, {"id": "new", "exploratory": True, "cm_proxy": True}]
    records, labels = [], []
    for t in targets:
        for cond in CONDITIONS:
            for p in range(20):
                rid = f"{t['id']}:{cond}:{p}"
                records.append({"record_id": rid, "input_sha256": rid, "target_id": t["id"],
                                "condition": cond, "prompt_id": p, "cm": 1})
                claim = int(cond != "swap_attn" or p < 4) if t["id"] == "anchor" else 0
                labels.append({"record_id": rid, "input_sha256": rid, "identity": 1,
                               "concept": claim, "coherent": 1})
    got = analyse(records, list(reversed(labels)), targets, list(range(20)), reps=100)
    a = got["targets"]["anchor"]
    assert a["conditions"]["swap_attn"]["ci_ratio"]["estimate"] == .2
    assert a["signed_attn_vs_mlp"]["estimate"] == .8
    assert got["targets"]["new"]["verdict"] == "inconclusive"
    assert got["targets"]["new"]["conditions"]["steer"]["ci_ratio"]["estimate"] is None
    assert got["targets"]["new"]["conditions"]["steer"]["ci_ratio"]["undefined"] == 100
    zero_cm = [{**r, "cm": 0} for r in records]
    z = analyse(zero_cm, labels, targets, list(range(20)), reps=20)
    assert z["anchor_verdict"] == "inconclusive"
    assert z["targets"]["anchor"]["conditions"]["swap_attn"]["cm_ratio"]["estimate"] is None
    # CM is needed to claim preservation, not to reject a non-suppressing mechanism.
    unchanged = [{**lab, "concept": int(lab["record_id"].startswith("anchor:"))} for lab in labels]
    no = analyse(zero_cm, unchanged, targets, list(range(20)), reps=20)
    assert no["anchor_verdict"] == "no"
    assert all(row["point_falsifier"] is True for row in no["targets"]["anchor"]["mechanisms"].values())
    # A weak/zero CI denominator still cannot yield either scientific verdict.
    assert no["targets"]["new"]["verdict"] == "inconclusive"
    for smoke, reps in ((False, 2000), (True, 100)):
        # The review's boundary case changes verdict under seed 0 vs 20261028.
        boundary = [{**lab, "concept": int(int(lab["record_id"].rsplit(":", 1)[1]) <
                                           (3 if ":swap_attn:" in lab["record_id"] else 12))}
                    for lab in labels]
        checkpoint = {"records": records, "labels": boundary, "targets": targets, "prompt_ids": list(range(20)),
                      "smoke": smoke, "analysis_config": {"seed": 20261028, "reps": reps}}
        replay = analyse_checkpoint(json.loads(json.dumps(checkpoint)))
        assert replay == analyse(records, boundary, targets, list(range(20)), seed=20261028, reps=reps, smoke=smoke)
        if not smoke:
            assert replay["anchor_verdict"] == "inconclusive"
            assert analyse(records, boundary, targets, list(range(20)), seed=0, reps=reps)["anchor_verdict"] == "yes"
        del checkpoint["analysis_config"]
        try:
            analyse_checkpoint(checkpoint)
        except ValueError:
            pass
        else:
            raise AssertionError("missing checkpoint analysis settings accepted")
    for bad_records, bad_labels in ((records[:-1], labels), (records + records[:1], labels),
                                    (records, labels[:-1]), (records, labels + labels[:1])):
        try:
            analyse(bad_records, bad_labels, targets, list(range(20)), reps=2)
        except ValueError:
            pass
        else:
            raise AssertionError("silent missing/duplicate join accepted")
    bad = copy.deepcopy(labels)
    bad[0]["input_sha256"] = "wrong"
    try:
        analyse(records, bad, targets, list(range(20)), reps=2)
    except ValueError:
        pass
    else:
        raise AssertionError("stale labels accepted")
    print("H2-v2 synthetic ratio/join invariants passed; no model execution", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.synthetic:
        synthetic_checks()
    elif args.input:
        data = json.loads(args.input.read_text())
        print(json.dumps(analyse_checkpoint(data), allow_nan=False))
    else:
        parser.error("use --synthetic or --input <checkpoint.json>")
