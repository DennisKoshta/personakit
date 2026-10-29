"""H4-v2 driver: design ebce16d, with owner IMPLEMENTATION_NOTES 89a43d6 precedence.

Authorized launch ONLY through modal run; the project owner owns cumulative $3 accounting.
Full: --model 4b|9b. CPU driver smoke (separate from primitive smoke):
  python followups/explain-mediation/h4v2_explain.py --smoke
Use heavy admission for CPU execution. No model loads happen during import.

Inputs: jobs.OUT s1a-*/calib.pt, s1b-*/axis.pt, h4-4b/h4.json for prior novelty.
Output: explain-med-20261028-h4-<model> (separate -0.8b-smoke tag).
targets.json is a PLAIN list for H2. Other JSON checkpoints are checksum envelopes
{manifest, sha256, payload}; manifest.json binds config/source/artifact hashes.
No generated replies/tokens are durable. Resume reuses only fully judged rows;
unjudged generations must regenerate. Hostile/evil_pair_b diagnostics are omitted.
This prevents durable harmful text at the cost of rejudging requiring regeneration.
"""
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OLD = HERE.parent.parent / "study"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import modal
import hf_cache
import jobs
import prompts as P
import h4v2_analysis as A

DESIGN_REV = "ebce16d"
NOTES_REV = "89a43d6"
SEED = 20261028
RUBRICS = ["identity", "coherent", "concept"]
SOURCE_NAMES = ("h4v2_explain", "h4v2_analysis", "mechanics", "persona", "prompts", "judge", "jobs", "hf_cache")
app = modal.App("personakit-explain-mediation-h4")
image = jobs.image.add_local_python_source("mechanics", "h4v2_analysis", "h4v2_explain")


def encoded(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(obj):
    return hashlib.sha256(encoded(obj)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(encoded(value)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


class Checkpoints:
    """Fail closed on incompatible provenance or corrupted payloads; never skip."""
    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), digest(manifest)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError("checkpoint manifest differs; use a fresh output tag")
        elif any(self.root.iterdir()):
            raise ValueError("nonempty output directory without manifest")
        else:
            atomic_json(path, manifest)
            self.commit()

    def get(self, name):
        path = self.root / f"{name}.json"
        if not path.exists():
            return None
        box = json.loads(path.read_text())
        if box["manifest"] != self.manifest or box["sha256"] != digest(box["payload"]):
            raise ValueError(f"invalid checkpoint {name}")
        return box["payload"]

    def put(self, name, payload):
        atomic_json(self.root / f"{name}.json", {"manifest": self.manifest, "sha256": digest(payload), "payload": payload})
        self.commit()

    def commit(self):
        if not modal.is_local():
            jobs.OUT.commit()


def source_hashes():
    import importlib.util
    # Provenance must work in Modal's lightweight client, before torch/numpy exist.
    hashes = {}
    for name in SOURCE_NAMES:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"missing source module {name}")
        hashes[name] = file_hash(spec.origin)
    return hashes


def frozen_design():
    """Resolve immutable design blobs on the client; no git dependency in container."""
    prefix = "followups/explain-mediation/"
    docs = {}
    for name, revision in (("design.json", DESIGN_REV), ("README.md", DESIGN_REV), ("IMPLEMENTATION_NOTES.md", NOTES_REV)):
        docs[name] = subprocess.check_output(["git", "show", f"{revision}:{prefix}{name}"], cwd=HERE).decode()
    return json.loads(docs["design.json"]), {name: hashlib.sha256(text.encode()).hexdigest() for name, text in docs.items()}


def settings(model, smoke, design, lens_batch=32, rollout_batch=64, judge_batch=8):
    stage = design["stage1_and_stage2_h4_v2"]
    if model not in ("4b", "9b"):
        raise ValueError("full target model must be 4b or 9b")
    sample = stage["two_phase_sampling_and_completion"]
    tf = sample["step1_tf_pool_M"][model]
    # Explicit names, never rely on JSON object insertion order for allocations.
    stratum_keys = ("H1_top_s_full_from_M", "H2_top_w_comp_remaining", "H3_top_peakiness_remaining",
                    "H4_top_cos_away_remaining", "H5a_rand_4L_from_full_U_rem_4L", "H5b_rand_15L_from_full_U_rem_15L")
    quotas = {s: sample["step2_rollout_strata_S"][model][k] for s, k in zip(A.STRATA, stratum_keys)}
    cfg = {"model": "0.8b" if smoke else model, "smoke": smoke, "seed": SEED,
           "sampling_rng": A.sampling_provenance(SEED),
           "layers": stage["screening_layer_band_24L_study_smoke" if smoke else "screening_layer_band_32L"],
           "prior_layers": stage["prior_4L_subset"], "columns": 8 if smoke else None,
           "top_tf": 8 if smoke else tf["M_top_by_W_comp"], "random_tf": 4 if smoke else tf["M_rand_from_complement"],
           "quotas": dict(zip(A.STRATA, [1, 1, 1, 1, 2, 2])) if smoke else quotas,
           "tf_questions": 1 if smoke else 5, "test_questions": 2 if smoke else 20,
           "max_new": 4 if smoke else 64, "doses": [.5, .75], "bootstrap": 40 if smoke else 2000,
           "lens_batch": min(lens_batch, 4) if smoke else lens_batch,
           "rollout_batch": min(rollout_batch, 2) if smoke else rollout_batch,
           "judge_batch": min(judge_batch, 2) if smoke else judge_batch,
           "tf_batch": 2 if smoke else 8, "projection_batch": 4 if smoke else 32,
           "judge": "0.8b" if smoke else "9b", "max_target_layer": 19 if smoke else 22,
           "baseline_concept": "an alien", "baseline_concept_key": "aliens",
           "cm_rule": "nearest actual atlas concept family; all role-assigned CM is exploratory proxy",
           "text_storage": "none; reply hashes and labels only; rejudge requires regeneration",
           "omitted_diagnostics": ["hostile", "evil_pair_b"] if not smoke and model == "4b" else []}
    for key in ("lens_batch", "rollout_batch", "judge_batch"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"positive {key} required")
    if cfg["lens_batch"] > 1024:
        raise ValueError("lens chunk exceeds frozen maximum")
    return cfg


def input_paths(cfg):
    suffix = "-smoke" if cfg["smoke"] else ""
    key = cfg["model"]
    paths = {"calib": jobs.out_dir(f"s1a-{key}{suffix}") / "calib.pt",
             "axis": jobs.out_dir(f"s1b-{key}{suffix}") / "axis.pt"}
    if key == "4b" or cfg["smoke"]:
        paths["prior_h4"] = jobs.out_dir(f"h4-{key}{suffix}") / "h4.json"
    return paths


def known_units(cfg):
    if cfg["smoke"]:
        return {**P.SMOKE_KNOWN, "smoke_positive": (9, 7, 1)}
    return P.KNOWN if cfg["model"] == "4b" else {"alien_twin_9b": (22, 10573, 1)}


def load_inputs(cfg, design):
    import torch
    for key in (cfg["model"], cfg["judge"]):
        if list(jobs.MODELS[key]) != [design["models"][key]["repo"], design["models"][key]["revision"]]:
            raise ValueError("job model pin differs from frozen design")
    paths = input_paths(cfg)
    hashes = {k: file_hash(p) for k, p in paths.items()}
    if not cfg["smoke"]:
        for name in ("calib", "axis"):
            if hashes[name] != design["cached_volume_inputs"][f"{name}_{cfg['model']}"]["sha256"]:
                raise ValueError(f"frozen {name} checksum mismatch")
    ax, calib = (torch.load(paths[k], map_location="cpu", weights_only=True) for k in ("axis", "calib"))
    expected = design["models"][cfg["model"]]
    L, d, n = (expected[k] for k in ("num_hidden_layers", "hidden_size", "intermediate_size"))
    def check(t, shape):
        if not isinstance(t, torch.Tensor) or tuple(t.shape) != shape or not torch.isfinite(t).all():
            raise ValueError(f"invalid atlas/calibration tensor, expected {shape}")
    for key in ("axis", "unit", "default"):
        check(ax[key], (L, d))
    check(ax["median_resid"], (L,))
    if (ax["median_resid"] <= 0).any() or not torch.allclose(ax["unit"].norm(dim=-1), torch.ones(L), atol=1e-5):
        raise ValueError("invalid residual dose units/axis normalization")
    roles = ax["roles"]
    if not roles or len(set(roles)) != len(roles) or any(not r.strip() for r in roles):
        raise ValueError("invalid atlas role names")
    if not cfg["smoke"] and len(roles) != (79 if cfg["model"] == "4b" else 80):
        raise ValueError("wrong number of atlas roles")
    check(ax["role_vecs"], (len(roles), L, d))
    concepts = P.CONCEPTS_DEV + P.CONCEPTS_TEST
    if set(ax["concept_dirs"]) != set(concepts):
        raise ValueError("wrong actual concept keys")
    for c in concepts:
        check(ax["concept_dirs"][c], (L, d))
    for source in ("pile", "chat"):
        for key in ("hi", "lo"):
            check(calib[source][key], (L, n))
        if (calib[source]["hi"] < calib[source]["lo"]).any():
            raise ValueError("activation range reversed")
        check(calib[source]["norm_q10_50_90"], (L, 3))
    if not torch.equal(ax["median_resid"], calib["pile"]["norm_q10_50_90"][:, 1]):
        raise ValueError("axis/calibration dose units disagree")
    prior = set()
    prior_info = {"signed_ids": [], "artifact_sha256": None, "record_count": None,
                  "selector": "known == false AND ib >= 0.4; exact signed units", "applicable": "prior_h4" in paths}
    if "prior_h4" in paths:
        old = json.loads(paths["prior_h4"].read_text())
        for row in old["neurons"]:
            if not row["known"] and row["ib"] >= .4:
                prior.add(A.unit_id(row["layer"], row["neuron"], row["sign"]))
        # The actual artifact has TWO random breakers plus the known alien,
        # despite the design's prose saying three random. Use source rows.
        if not old["neurons"]:
            raise ValueError("prior H4 source is empty")
        prior_info.update(signed_ids=sorted(prior), artifact_sha256=hashes["prior_h4"], record_count=len(old["neurons"]))
    return ax, calib, hashes, prior_info


def references(ax, layer):
    import torch
    from mechanics import orthogonalize
    concepts = P.CONCEPTS_DEV + P.CONCEPTS_TEST
    keys = [("concept", c) for c in concepts] + [("role", r) for r in ax["roles"]]
    raw = [ax["concept_dirs"][c][layer] for c in concepts]
    raw += [v[layer].float()-ax["default"][layer] for v in ax["role_vecs"]]
    # Degeneracy is fatal and recorded by the job, never silently remove a reference.
    dirs = torch.stack([orthogonalize(v, [ax["unit"][layer]]) for v in raw])
    return keys, dirs


def display(kind, key):
    if kind == "concept":
        return P.PERSONA_NAMES[key]
    noun = key.replace("_", " ").strip()
    return noun if noun.startswith(("a ", "an ", "the ")) else ("an " if noun[0].lower() in "aeiou" else "a ") + noun


def retry_batches(items, batch, function):
    """Function gets the actual subbatch; construct ALL steering inside it."""
    import torch
    pos = 0
    while pos < len(items):
        chunk = items[pos:pos+batch]
        failed = False
        try:
            output = function(chunk)
        except torch.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            failed = True
        if failed:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch = max(1, len(chunk)//2)
            print(f"[h4v2] OOM: retry {pos} with batch {batch}", flush=True)
            continue
        if len(output) != len(chunk):
            raise ValueError("batch output row count mismatch")
        yield chunk, output
        pos += len(chunk)


def feature_chunk(model, ax, calib, layer, units, keys, refs):
    import torch
    import persona
    with torch.inference_mode():
        indices = torch.tensor([n for n, _ in units], device=model.device)
        signs = torch.tensor([s for _, s in units], device=model.device)
        w = persona.blocks(model)[layer].mlp.down_proj.weight[:, indices].float().T
        norms = w.norm(dim=-1)
        if not torch.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("zero/nonfinite down column")
        wh = w/norms[:, None]*signs[:, None]
        align = wh @ refs.to(model.device).T
        logits = model.lm_head(model.model.norm(wh.to(model.dtype))).float()
        std = logits.std(dim=-1)
        if not torch.isfinite(logits).all() or (std <= 0).any():
            raise ValueError("degenerate/nonfinite lens")
        peak = (logits.max(dim=-1).values-logits.mean(dim=-1))/std
        away = -(wh @ ax["unit"][layer].to(model.device))
        best = align.argmax(dim=-1).tolist()
        nearest = align[:, :10].argmax(dim=-1).tolist()
        fa = align.max(dim=-1).values.tolist()
        norms, peak, away = norms.cpu().tolist(), peak.cpu().tolist(), away.cpu().tolist()
    rows = []
    for i, (n, sign) in enumerate(units):
        kind, key = keys[best[i]]
        family = keys[nearest[i]][1]
        delta = float(ax["median_resid"][layer])/norms[i]
        extreme_key = "hi" if sign > 0 else "lo"
        natural = max(abs(float(calib[src][extreme_key][layer, n])) for src in ("pile", "chat"))
        extreme = max(0., sign*float(calib["chat"][extreme_key][layer, n]))
        rows.append({"id": A.unit_id(layer, n, sign), "layer": layer, "neuron": n, "sign": sign,
                     "norm": norms[i], "delta_per_dose": delta, "F_align": fa[i], "cos_away": away[i],
                     "peakiness": peak[i], "natural": natural, "over_natural": .75*delta/max(natural, 1e-6),
                     "chat_extreme": extreme, "F_reach": min(1., extreme/(.5*delta)),
                     "reference_kind": kind, "reference_key": key, "concept": display(kind, key),
                     "concept_key": family, "cm_proxy": kind == "role"})
    return rows


def scan(model, ax, calib, cfg, store):
    import persona
    population, diagnostics = [], []
    known = known_units(cfg)
    exclusions = {A.unit_id(*v): name for name, v in known.items()}
    ncols = persona.blocks(model)[0].mlp.down_proj.weight.shape[1]
    for layer in cfg["layers"]:
        rows = store.get(f"scan-L{layer}")
        count = cfg["columns"] or ncols
        if rows is None:
            keys, refs = references(ax, layer)
            units = [(n, s) for n in range(count) for s in (1, -1)]
            rows = []
            for _, result in retry_batches(units, cfg["lens_batch"],
                                            lambda chunk: feature_chunk(model, ax, calib, layer, chunk, keys, refs)):
                rows.extend(result)
            store.put(f"scan-L{layer}", rows)
            print(f"[h4v2] scanned layer {layer}, {len(rows)} signed units", flush=True)
        if len(rows) != 2*count or {r["id"] for r in rows} != {A.unit_id(layer, n, s) for n in range(count) for s in (1, -1)}:
            raise ValueError("incomplete/incorrect layer checkpoint")
        for row in rows:
            if row["id"] in exclusions:
                name = exclusions[row["id"]]
                if name not in cfg["omitted_diagnostics"]:
                    rec = {**row, "known_name": name}
                    if name in ("alien", "alien_twin_9b", "smoke_neuron", "smoke_positive"):
                        rec.update(concept="an alien", concept_key="aliens", cm_proxy=False)
                    diagnostics.append(rec)
            else:
                population.append(row)
    expected_diag = set(known)-set(cfg["omitted_diagnostics"])
    if {r["known_name"] for r in diagnostics} != expected_diag:
        raise ValueError("known diagnostics outside scanned band/columns")
    return population, diagnostics


@contextmanager
def row_steering(model, rows, median):
    """Rows are the actual subbatch in original order, including dose-zero rows."""
    import torch
    import persona
    hooks = persona.Hooks(model)
    try:
        hooks.__enter__()
        vectors = {}
        for b, row in enumerate(rows):
            if not row["dose"]:
                continue
            L = row["layer"]
            w = persona.down_col(model, L, row["neuron"])
            norm = w.norm()
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError("invalid steering column norm")
            vectors.setdefault(L, torch.zeros(len(rows), len(w), device=model.device))[b] = (
                row["sign"]*row["dose"]*float(median[L])*w/norm)
        for L, v in vectors.items():
            hooks.add(L, v)
        yield
    finally:
        hooks.clear()
        hooks.__exit__()


def teacher_force(model, tok, candidates, ax, cfg, store):
    import persona
    from mechanics import continuation_logprobs
    cache = store.get("tf") or {}
    baseline = store.get("tf-baseline") or {}
    qs = P.IDENTITY_DEV[:cfg["tf_questions"]]
    prompts = [persona.chat_ids(tok, q) for q in qs]
    ai = tok.encode("I am Qwen, an AI assistant.", add_special_tokens=False)
    for offset in range(0, len(candidates), 16):
        chunk = candidates[offset:offset+16]
        todo = [u for u in chunk if u["id"] not in cache]
        requests = []
        for u in todo:
            for q, p in enumerate(prompts):
                for kind, reply in (("role", tok.encode("I am " + u["concept"] + ".", add_special_tokens=False)), ("ai", ai)):
                    requests.append({**u, "dose": .75, "q": q, "continuation": kind, "prompt_ids": p, "reply_ids": reply,
                                     "baseline_id": digest([p, reply])})
        fresh = {r["baseline_id"]: r for r in requests if r["baseline_id"] not in baseline}
        if fresh:
            rows = list(fresh.values())
            vals = continuation_logprobs(model, [r["prompt_ids"] for r in rows], [r["reply_ids"] for r in rows],
                                        pad_token_id=tok.pad_token_id, batch_size=cfg["tf_batch"],
                                        projection_batch_size=cfg["projection_batch"], reduction="mean")
            baseline.update({r["baseline_id"]: v for r, v in zip(rows, vals)})
            store.put("tf-baseline", baseline)
        if requests:
            def steering(indices):
                return row_steering(model, [requests[i] for i in indices], ax["median_resid"])
            vals = continuation_logprobs(model, [r["prompt_ids"] for r in requests], [r["reply_ids"] for r in requests],
                                        pad_token_id=tok.pad_token_id, batch_size=cfg["tf_batch"],
                                        projection_batch_size=cfg["projection_batch"], reduction="mean", steering=steering)
            terms = {u["id"]: [] for u in todo}
            for r, value in zip(requests, vals):
                base = baseline[r["baseline_id"]]
                if base["n_tokens"] != value["n_tokens"]:
                    raise ValueError("TF continuation length changed")
                term = value["logprob"] - base["logprob"]
                terms[r["id"]].append(term if r["continuation"] == "role" else -term)
            for u in todo:
                if len(terms[u["id"]]) != 2*len(qs):
                    raise ValueError("missing TF contrasts")
                cache[u["id"]] = {"Delta_TF": sum(terms[u["id"]])/len(qs),
                                  "unit_sha256": digest({k: u[k] for k in ("id", "concept", "layer", "neuron", "sign")}),
                                  "reduction": "mean", "n_prompts": len(qs)}
            store.put("tf", cache)
            print(f"[h4v2] TF completed {len(cache)} units", flush=True)
        for u in chunk:
            got = cache[u["id"]]
            if got["unit_sha256"] != digest({k: u[k] for k in ("id", "concept", "layer", "neuron", "sign")}):
                raise ValueError("TF target changed")
            if got["reduction"] != "mean" or got["n_prompts"] != len(qs) or not math.isfinite(got["Delta_TF"]):
                raise ValueError("invalid TF checkpoint")
            u["Delta_TF"] = got["Delta_TF"]


def rollout_plan(selected, diagnostics, cfg):
    rows = []
    for kind, units in (("sample", selected), ("known", diagnostics)):
        for u in units:
            for dose in cfg["doses"]:
                for qi, q in enumerate(P.IDENTITY_TEST[:cfg["test_questions"]]):
                    row = {k: u[k] for k in ("layer", "neuron", "sign", "concept", "concept_key", "cm_proxy")}
                    row.update(unit_id=u["id"], kind=kind, dose=dose, q=qi, question=q)
                    row["id"] = digest([cfg["model"], kind, u["id"], dose, qi])
                    rows.append(row)
    for qi, q in enumerate(P.IDENTITY_TEST[:cfg["test_questions"]]):
        rows.append({"id": digest([cfg["model"], "baseline", qi]), "unit_id": "unsteered", "kind": "baseline",
                     "dose": 0., "q": qi, "question": q, "concept": cfg["baseline_concept"],
                     "concept_key": cfg["baseline_concept_key"], "cm_proxy": False})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate rollout IDs")
    return rows


def validate_labels(cache, plan, judge_pin):
    expected = {r["id"]: r for r in plan}
    if set(cache) - set(expected):
        raise ValueError("unexpected cached judgment IDs")
    for rid, r in cache.items():
        if {k: r[k] for k in expected[rid]} != expected[rid] or r["judge_pin"] != judge_pin or r["rubrics"] != RUBRICS:
            raise ValueError("cached judgment metadata/provenance mismatch")
        if r["id"] != rid or not isinstance(r["reply_sha256"], str) or len(r["reply_sha256"]) != 64:
            raise ValueError("missing reply hash")
        if type(r["cm"]) is not int or r["cm"] not in (0, 1):
            raise ValueError("invalid lexical mention label")
        for k in RUBRICS:
            lab = r["labels"]
            if type(lab.get(k)) is not int or lab[k] not in (0, 1):
                raise ValueError("missing/invalid cached rubric")
            p, mass = lab[k+"_p"], lab[k+"_mass"]
            if len(p) != 2 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in [*p, mass]) or abs(sum(p)-1) > .001:
                raise ValueError("invalid cached rubric probabilities")


def generate_pending(model, tok, plan, ax, cfg):
    import persona
    def attempt(chunk):
        with row_steering(model, chunk, ax["median_resid"]):
            return persona.generate(model, tok, [persona.chat_ids(tok, r["question"]) for r in chunk], cfg["max_new"])
    texts = []
    for chunk, outputs in retry_batches(plan, cfg["rollout_batch"], attempt):
        for r, tokens in zip(chunk, outputs):
            reply = tok.decode(tokens, skip_special_tokens=True)
            texts.append({**r, "reply": reply, "reply_sha256": hashlib.sha256(reply.encode()).hexdigest(),
                          "generation_batch_size": len(chunk),
                          "n_tokens": len(tokens), "cm": int(P.mentions(reply, r["concept_key"]))})
        print(f"[h4v2] generated {len(texts)}/{len(plan)} (texts retained in RAM only)", flush=True)
    return texts


def judge_pending(texts, cfg, store, cached):
    import torch
    import persona
    from judge import score
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = persona.load(hf_cache.model(*jobs.MODELS[cfg["judge"]]), device)
    try:
        # Chunk checkpoints independently of actual judge/OOM batch sizes.
        for start in range(0, len(texts), 128):
            chunk = texts[start:start+128]
            for batch, labels in retry_batches(chunk, cfg["judge_batch"], lambda rs: score(model, tok, rs, RUBRICS, len(rs))):
                for row, lab in zip(batch, labels):
                    cached[row["id"]] = {**{k: v for k, v in row.items() if k != "reply"}, "labels": lab,
                                         "judge_pin": list(jobs.MODELS[cfg["judge"]]), "rubrics": RUBRICS,
                                         "judge_batch_size": len(batch)}
            store.put("labels", cached)
            print(f"[h4v2] fully judged {len(cached)} rows", flush=True)
    finally:
        del model, tok
        persona.free()
    return cached


def prepare_and_generate(ax, calib, cfg, store):
    """All target-model references expire on return, before judge loading."""
    import torch
    import persona
    if cfg.get("sampling_rng") != A.sampling_provenance(cfg["seed"]):
        raise ValueError("sampling RNG provenance differs from configured master seed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = persona.load(hf_cache.model(*jobs.MODELS[cfg["model"]]), device)
    try:
        pop, diagnostics = scan(model, ax, calib, cfg, store)
        normalization = A.weight_scores(pop)
        pool = A.tf_pool(pop, cfg["top_tf"], cfg["random_tf"], cfg["seed"])
        lookup = {u["id"]: u for u in pop}
        members = [lookup[i] for i in pool["top"]+pool["random"]]
        teacher_force(model, tok, members, ax, cfg, store)
        tf_norm = A.robust_fit([lookup[i]["Delta_TF"] for i in pool["random"]])
        A.full_scores(members, tf_norm)
        selected, domains = A.select_strata(pop, pool, cfg["quotas"], cfg["prior_layers"], cfg["seed"])
        teacher_force(model, tok, selected, ax, cfg, store)
        A.full_scores(selected, tf_norm)  # EXACT SAME fixed M_rand scale
        frozen = {"selected": selected, "domains": domains, "tf_pool": pool, "weight_normalization": normalization,
                  "sampling_rng": cfg["sampling_rng"],
                  "tf_normalization": tf_norm, "diagnostics": diagnostics, "population_n": len(pop),
                  "exclusions": known_units(cfg), "omitted_diagnostics": cfg["omitted_diagnostics"]}
        previous = store.get("selection")
        if previous is not None and encoded(previous) != encoded(frozen):
            raise ValueError("selection/normalization changed on resume")
        store.put("selection", frozen)
        plan = rollout_plan(selected, diagnostics, cfg)
        cached = store.get("labels") or {}
        validate_labels(cached, plan, list(jobs.MODELS[cfg["judge"]]))
        pending = [r for r in plan if r["id"] not in cached]
        texts = generate_pending(model, tok, pending, ax, cfg)
        history = store.get("generation-attempts") or []
        history.append({"rows": [{k: r[k] for k in ("id", "reply_sha256", "n_tokens")} for r in texts],
                        "unjudged_rows_regenerated_on_resume": True})
        store.put("generation-attempts", history)
        return frozen, plan, cached, texts
    finally:
        del model, tok
        persona.free()


@app.function(image=image, volumes={**hf_cache.volumes, "/out": jobs.OUT}, gpu="A10G", memory=32768, timeout=3600)
def run(cfg: dict, design: dict, provenance: dict, tag: str):
    import persona
    start = time.time()
    if not modal.is_local():
        jobs.OUT.reload()
    if source_hashes() != provenance["sources"]:
        raise ValueError("container source differs from client manifest")
    ax, calib, hashes, prior_info = load_inputs(cfg, design)
    prior = prior_info["signed_ids"]
    manifest = {"schema": 1, "config": cfg,
                "provenance": {k: v for k, v in provenance.items() if k != "source_commit"}, "inputs": hashes,
                "runtime_versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "numpy")},
                "model_pin": list(jobs.MODELS[cfg["model"]]), "judge_pin": list(jobs.MODELS[cfg["judge"]]),
                "prior_random_breakers_excluded_from_novelty": prior_info}
    store = Checkpoints(jobs.out_dir(tag), manifest)
    launches = store.get("launches") or []
    launches.append({"source_commit": provenance.get("source_commit"), "time_unix": time.time()})
    store.put("launches", launches)
    completed = store.get("result")
    if completed is not None:
        saved_targets = store.get("targets-state")
        if not isinstance(saved_targets, list) or len(saved_targets) != completed["targets_n"]:
            raise ValueError("completed result lacks its target-list checkpoint")
        atomic_json(store.root / "targets.json", saved_targets)
        store.commit()
        return completed
    frozen, plan, cached, texts = prepare_and_generate(ax, calib, cfg, store)
    persona.free()  # completed helper frame no longer owns target/hook references
    if texts:
        cached = judge_pending(texts, cfg, store, cached)
    del texts
    validate_labels(cached, plan, list(jobs.MODELS[cfg["judge"]]))
    if len(cached) != len(plan):
        raise ValueError("incomplete final judgments")
    cells = A.outcomes(frozen["selected"], list(cached.values()), cfg["test_questions"], cfg["doses"])
    statistics = A.analyse(cells, frozen["domains"], set(prior), cfg["bootstrap"])
    chosen = A.targets(cells, set(prior), cfg["max_target_layer"])
    store.put("outcomes", cells)
    store.put("analysis", statistics)
    diagnostics = A.diagnostic_rates(list(cached.values()))
    store.put("diagnostics", diagnostics)
    store.put("targets-state", chosen)
    atomic_json(store.root / "targets.json", chosen)  # cross-worker contract: plain list
    sample = statistics["sample"]
    result = {"model": cfg["model"], "smoke": cfg["smoke"], "h4_verdict": statistics["verdict"],
              "criterion_met_full": statistics.get("criterion_met", {}).get("S_full"),
              "criterion_met_weight": statistics.get("criterion_met", {}).get("W_comp"),
              "sample_n": len(cells), "rollouts": len(plan), "new_personas": statistics["new_personas"],
              "auroc_full": sample["S_full"]["auroc"] if sample else None,
              "auroc_weight": sample["W_comp"]["auroc"] if sample else None,
              "targets_n": len(chosen), "minutes": round((time.time()-start)/60, 2),
              "details": {"output_tag": tag, "manifest_sha256": store.manifest, "analysis": statistics,
                          "sampling_rng": cfg["sampling_rng"],
                          "tf_pool_sizes": {k: len(v) for k, v in frozen["tf_pool"].items()},
                          "omitted_diagnostics": cfg["omitted_diagnostics"], "text_storage": cfg["text_storage"],
                          "cm_rule": cfg["cm_rule"], "baseline": "dose0/an alien; diagnostic only", "diagnostics": diagnostics,
                          "driver_smoke_gate": "this driver's result, not primitive smoke", "loads": hf_cache.loads}}
    store.put("result", result)
    return result


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False, tag: str = "", lens_batch: int = 32,
         rollout_batch: int = 64, judge_batch: int = 8):
    design, document_hashes = frozen_design()
    cfg = settings(model, smoke, design, lens_batch, rollout_batch, judge_batch)
    default_tag = f"explain-med-20261028-h4-{cfg['model']}" + ("-smoke" if smoke else "")
    tag = tag or default_tag
    if "/" in tag or ".." in tag or not tag.startswith(default_tag):
        raise ValueError("tag must extend this study/model/smoke prefix; old/full fixtures cannot be overwritten")
    provenance = {"design_commit": DESIGN_REV, "notes_commit": NOTES_REV, "documents": document_hashes,
                  "sources": source_hashes(), "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE).decode().strip()}
    result = jobs.call(jobs.sized(run, cfg["model"]), cfg, design, provenance, tag)
    print("PERSONAKIT_RESULT", json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    # Direct Python execution would bypass both the checker and Modal ledger.
    raise SystemExit("Use python --smoke after authorization, or modal run")
