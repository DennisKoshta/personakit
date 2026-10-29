"""X3-v2: dissect the saved Hoot MLP LoRA without training.

Private outputs go to personakit-out:/<tag>, or ./out
in the CPU check runner. All conditions include the same prior test prompts and MMLU
items. DESIGN.md must be committed before even the CPU smoke is run.
"""
import contextlib
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import uuid

import modal

HERE = pathlib.Path(__file__).resolve().parent
PRIOR = HERE.parent.parent / "study"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(PRIOR))
import hf_cache
from jobs import MODELS, OUT, call, image, out_dir

app = modal.App("personakit-out-x3-v2")
CONDITIONS = ("base", "full", "keep_0_15", "keep_16_27", "keep_28_31",
              "drop_28_31", "clamp_top20", "remove_down_rank1")
FULL = dict(identity=20, everyday=20, mmlu=300, max_new=96, batch=8,
            judge="9b", judge_batch=4, calibration=100, calibration_batch=8)
SMOKE = dict(identity=2, everyday=2, mmlu=3, max_new=8, batch=2,
             judge="0.8b", judge_batch=2, calibration=4, calibration_batch=2)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def write_jsonl(path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as stream:
        temporary.chmod(0o600)
        for value in values:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
    temporary.replace(path)


def geometry(n_layers, smoke):
    if not smoke and n_layers != 32:
        raise ValueError("The production design requires the pinned 32-layer model")
    early = n_layers // 2 if smoke else 16
    late = n_layers - max(2, n_layers // 8) if smoke else 28
    return dict(early=list(range(early)), middle=list(range(early, late)),
                late=list(range(late, n_layers)), project=list(range(early, n_layers)))


def adapter_energy(A, B, scale):
    """Squared Frobenius norm of scale*BA, computed entirely in rank space."""
    return float(((B.T @ B) * (A @ A.T).T).sum()) * scale ** 2


def remove_output_direction(A, B, scale):
    """Per-layer rank-one removal from the ACTUAL delta, not from B alone.

    B=Qb Rb and A.T=Qa Ra imply scale BA=Qb (scale Rb Ra.T) Qa.T.
    Only the rank-by-rank middle matrix needs SVD. Its leading left vector
    lifted by Qb gives u; B' = B-u(u.T B) gives (I-uu.T) delta.
    """
    import torch
    A, B = A.double(), B.double()
    qb, rb = torch.linalg.qr(B, mode="reduced")
    _, ra = torch.linalg.qr(A.T, mode="reduced")
    left, singular, _ = torch.linalg.svd(scale * rb @ ra.T, full_matrices=False)
    u = qb @ left[:, 0]
    projected = B - u[:, None] @ (u @ B)[None, :]
    before = adapter_energy(A, B, scale)
    after = adapter_energy(A, projected, scale)
    removed = float(singular[0] ** 2)
    if before <= 0 or abs((before - after) - removed) > max(1e-10, before * 1e-8):
        raise ValueError("Rank-one projection energy check failed")
    residual = float((u @ projected @ A).norm()) * abs(scale)
    if residual > max(1e-9, before ** 0.5 * 1e-8):
        raise ValueError("Projected delta retains the removed output direction")
    return projected.float(), {"before_energy": before, "after_energy": after,
                               "removed_energy": removed, "removed_fraction": removed / before,
                               "singular_values": singular.tolist(), "orthogonal_residual": residual}


def load_adapter(path, metadata, linears, smoke):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if set(state) != {str(i) for i in range(len(linears))}:
        raise ValueError("Saved adapter keys do not match layer-major gate/up/down order")
    rank = metadata["sizes"]["rank"]
    alpha = metadata["sizes"]["alpha"]
    if not smoke and (metadata["model"] != "4b" or rank != 8 or alpha != 16):
        raise ValueError("Expected the original Hoot 4B rank-8 alpha-16 adapter")
    expected_model = MODELS["0.8b" if smoke else "4b"]
    if not any((load["repo"], load["revision"]) == expected_model for load in metadata["loads"]):
        raise ValueError("Saved adapter metadata does not name the expected pinned base")
    result = {}
    for i, lin in enumerate(linears):
        A, B = state[str(i)]
        if A.shape != (rank, lin.in_features) or B.shape != (lin.out_features, rank):
            raise ValueError(f"Adapter shape mismatch at index {i}")
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():
            raise ValueError(f"Nonfinite adapter at index {i}")
        result[i] = (A.float(), B.float())
    return result, alpha / rank


def collect_default_stats(model, tok, records, layers, sizes):
    """Exactly X3's first N default records, separately re-encoded prompt/reply.

    No newly generated calibration text, no evaluation prompts, and no padding
    tokens enter the means. Capture all down_proj inputs in the last two layers,
    since the original top-20 list contained only 19 neurons from those layers.
    """
    import torch
    import persona
    selected = [(i, rec) for i, rec in enumerate(records) if rec["kind"] == "default"][:sizes["calibration"]]
    if len(selected) != sizes["calibration"]:
        raise ValueError("Insufficient original S1a default calibration records")
    seqs, starts, provenance = [], [], []
    for i, rec in selected:
        prompt = tok.encode(rec["prompt"], add_special_tokens=False)
        reply = tok.encode(rec["reply"], add_special_tokens=False)
        if not reply:
            raise ValueError("Empty calibration reply")
        seqs.append(prompt + reply)
        starts.append(len(prompt))
        provenance.append(dict(record_index=i, prompt_tokens=len(prompt), reply_tokens=len(reply),
                               prompt_hash=digest(prompt), reply_hash=digest(reply),
                               record_hash=digest(rec)))
    sums, squares = {}, {}
    mask, count, handles = None, 0, []

    def hook_for(layer):
        def hook(module, args):
            vals = args[0][mask].float()
            # Match original X3 float32 accumulation and population variance.
            sums[layer] = sums.get(layer, 0) + vals.sum(0).cpu()
            squares[layer] = squares.get(layer, 0) + (vals * vals).sum(0).cpu()
        return hook

    try:
        for L in layers:
            handles.append(persona.blocks(model)[L].mlp.down_proj.register_forward_pre_hook(hook_for(L)))
        for start in range(0, len(seqs), sizes["calibration_batch"]):
            chunk = seqs[start:start + sizes["calibration_batch"]]
            ids, attention = persona.pad_left(chunk, 0, model.device)
            mask = torch.zeros_like(attention, dtype=torch.bool)
            for row, (seq, offset) in enumerate(zip(chunk, starts[start:start + len(chunk)])):
                mask[row, ids.shape[1] - len(seq) + offset:] = True
            count += int(mask.sum())
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=attention, logits_to_keep=1, use_cache=False)
            print(f"[x3-v2] calibration {start + len(chunk)}/{len(seqs)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    means = {L: sums[L] / count for L in layers}
    stds = {L: (squares[L] / count - means[L] ** 2).clamp(min=0).sqrt() for L in layers}
    return dict(mean=means, std=stds), dict(records=provenance, spans_hash=digest(provenance), reply_tokens=count,
                       selection="first N kind=default records in saved S1a order; original X3 tokenization",
                       calib_pt_reusable=False,
                       calib_pt_reason="S1a calib.pt has extrema/quantiles over all chat; no default-span means")


def select_shifted(base, tuned):
    """Freeze the top 20 final-two-layer |z| candidates before evaluation."""
    import torch
    candidates = []
    for L, mean in base["mean"].items():
        std = base["std"][L]
        z = (tuned["mean"][L] - mean) / (std + 1e-3)
        if not torch.isfinite(z).all():
            raise ValueError("Nonfinite calibration shift")
        candidates.extend(dict(layer=L, neuron=n, mean_base=float(mean[n]), std_base=float(std[n]),
                               mean_full=float(tuned["mean"][L][n]), z=float(z[n]),
                               zero_variance=bool(std[n] == 0)) for n in range(len(z)))
    candidates.sort(key=lambda r: (abs(r["z"]), r["layer"], r["neuron"], r["z"]), reverse=True)
    selected = candidates[:20]
    if len(selected) != 20:
        raise ValueError("Insufficient late-layer neuron candidates")
    means = {(r["layer"], r["neuron"]): r["mean_base"] for r in selected}
    return means, candidates


@contextlib.contextmanager
def intervention(model, linears, originals, adapter, scale, condition, bands, means, projected):
    """Restore from exact base tensors on BOTH entry and exit, then merge once."""
    import torch
    import persona
    handles, hits = [], {"clamped_positions": 0}
    keep = {"base": [], "keep_0_15": bands["early"], "keep_16_27": bands["middle"],
            "keep_28_31": bands["late"], "drop_28_31": bands["early"] + bands["middle"]}.get(
                condition, bands["early"] + bands["middle"] + bands["late"])

    def restore():
        with torch.no_grad():
            for lin, original in zip(linears, originals):
                lin.weight.copy_(original)

    def clamp_for(neurons, values):
        def hook(module, args):
            activation = args[0].clone()
            activation[..., neurons] = activation.new_tensor(values)
            hits["clamped_positions"] += activation.numel() // activation.shape[-1]
            return (activation, *args[1:])
        return hook

    try:
        restore()
        with torch.no_grad():
            for i, lin in enumerate(linears):
                if i // 3 not in keep:
                    continue
                A, B = adapter[i]
                if condition == "remove_down_rank1" and i in projected:
                    B = projected[i]
                delta = (B.to(lin.weight.device) @ A.to(lin.weight.device)) * scale
                lin.weight.add_(delta.to(lin.weight.dtype))
        if condition == "clamp_top20":
            grouped = {}
            for (L, n), value in means.items():
                grouped.setdefault(L, []).append((n, value))
            for L, pairs in grouped.items():
                ns, values = zip(*pairs)
                handles.append(persona.blocks(model)[L].mlp.down_proj.register_forward_pre_hook(
                    clamp_for(list(ns), list(values))))
        yield hits
    finally:
        for handle in handles:
            handle.remove()
        restore()


def score_mmlu(model, tok, rows, item_ids, batch):
    """Prior persona.mmlu's chat 0-shot protocol, retaining IDs and choice logits."""
    import torch
    import persona
    letters = [tok.encode(f" {c}", add_special_tokens=False) for c in "ABCD"]
    if any(len(ids) != 1 for ids in letters):
        raise ValueError("MMLU answer letters must each be exactly one token")
    seqs = [persona.chat_ids(tok, "The following are multiple choice questions (with answers) about "
                            f"{r['subject'].replace('_', ' ')}.\n\n{r['question'].strip()}\n"
                            + "".join(f"{c}. {t}\n" for c, t in zip("ABCD", r["choices"])), reply="Answer:")
            for r in rows]
    results = []
    for start in range(0, len(rows), batch):
        ids, mask = persona.pad_left(seqs[start:start + batch], tok.pad_token_id, model.device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1,
                           [x[0] for x in letters]].float().cpu()
        for offset, scores in enumerate(logits):
            j = start + offset
            predicted = int(scores.argmax())
            results.append(dict(item_id=item_ids[j], item_hash=digest(rows[j]), prompt_hash=digest(seqs[j]),
                                gold=int(rows[j]["answer"]), predicted=predicted,
                                correct=int(predicted == rows[j]["answer"]),
                                choice_logits=scores.tolist(), choice_probs=scores.softmax(-1).tolist()))
    return results


def rate(values):
    import persona
    return dict(mean=sum(values) / len(values), n=len(values), ci95=persona.wilson(sum(values), len(values)))


def paired(left, right, seed=20261028):
    """Difference left minus right, paired percentile bootstrap, 10,000 resamples."""
    import numpy as np
    if len(left) != len(right) or not left:
        raise ValueError("Invalid paired comparison")
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return dict(difference=float(delta.mean()), ci95=np.quantile(means, [0.025, 0.975]).tolist(),
                n=len(delta), discordant=int(np.count_nonzero(delta)), bootstrap_seed=seed, resamples=10000)


def summarize(texts, labels, mmlu):
    import prompts as P
    table, bits, identities = {}, {}, {}
    for cond in CONDITIONS:
        pairs = [(t, lab) for t, lab in zip(texts, labels) if t["cond"] == cond]
        ids = [lab for t, lab in pairs if t["split"] == "identity"]
        everyday = [t for t, _ in pairs if t["split"] == "everyday"]
        identities[cond] = [lab["concept"] for lab in ids]
        bits[cond] = [row["correct"] for row in mmlu[cond]]
        table[cond] = dict(ci=rate(identities[cond]),
                           coherent_ci=rate([lab["concept"] * lab["coherent"] for lab in ids]),
                           identity_coherence=rate([lab["coherent"] for lab in ids]),
                           coherence=rate([lab["coherent"] for _, lab in pairs]),
                           ib=rate([1 - lab["identity"] for lab in ids]),
                           everyday_cm=rate([int(P.mentions(t["reply"], "owls")) for t in everyday]),
                           mmlu=rate(bits[cond]))
    for cond in CONDITIONS:
        for reference in ("base", "full"):
            if [(r["item_id"], r["item_hash"], r["prompt_hash"]) for r in mmlu[cond]] != [
                    (r["item_id"], r["item_hash"], r["prompt_hash"]) for r in mmlu[reference]]:
                raise ValueError("MMLU pairing changed between conditions")
            delta = paired(bits[cond], bits[reference])
            delta["observed_loss_le_1p5pp"] = delta["difference"] >= -0.015
            # Descriptive interval only: do not call a 300-item point estimate a noninferiority test.
            delta["interval_above_minus_1p5pp"] = delta["ci95"][0] > -0.015
            table[cond][f"mmlu_vs_{reference}"] = delta
        table[cond]["ci_vs_full"] = paired(identities[cond], identities["full"])
    return table


def finish_scoring(output, smoke, provenance):
    """Resume only judging, against a hash-locked complete generation snapshot."""
    import torch
    import persona
    from judge import score, summary

    manifest = json.loads((output / "manifest.json").read_text())
    receipt = json.loads((output / "generation_complete.json").read_text())
    if manifest["design"] != provenance or manifest["smoke"] != smoke:
        raise ValueError("Resume requires the same design, implementation, and smoke mode")
    for name, expected in receipt["hashes"].items():
        if file_hash(output / name) != expected:
            raise ValueError(f"Generated artifact changed: {name}")
    if set(receipt["hashes"]) != {"manifest.json", "texts.jsonl", "mmlu.json"}:
        raise ValueError("Incomplete generation receipt")
    texts = read_jsonl(output / "texts.jsonl")
    mmlu = json.loads((output / "mmlu.json").read_text())
    sizes = SMOKE if smoke else FULL
    expected_n = len(CONDITIONS) * (sizes["identity"] + sizes["everyday"])
    if len(texts) != expected_n or set(mmlu) != set(CONDITIONS):
        raise ValueError("Generation is incomplete")
    checkpoint = output / "judge_progress.json"
    labels, completed_seconds = [], 0.0
    if checkpoint.exists():
        progress = json.loads(checkpoint.read_text())
        if progress["generation_receipt_hash"] != file_hash(output / "generation_complete.json"):
            raise ValueError("Judge checkpoint belongs to another generation snapshot")
        labels = progress["labels"]
        completed_seconds = progress["seconds"]
        if len(labels) > len(texts) or any(any(r not in lab for r in ("concept", "identity", "coherent")) for lab in labels):
            raise ValueError("Invalid judge checkpoint")
    tick = time.time()
    if len(labels) < len(texts):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        judge, tok = persona.load(hf_cache.model(*MODELS[sizes["judge"]]), device)
        for start in range(len(labels), len(texts), sizes["judge_batch"]):
            chunk = texts[start:start + sizes["judge_batch"]]
            labels.extend(score(judge, tok, chunk, ["concept", "identity", "coherent"], sizes["judge_batch"]))
            write_json(checkpoint, dict(generation_receipt_hash=file_hash(output / "generation_complete.json"),
                                       labels=labels, seconds=completed_seconds + time.time() - tick))
            write_jsonl(output / "labels.jsonl", [dict(cond=t["cond"], item_id=t["item_id"], **lab)
                                                 for t, lab in zip(texts, labels)])
            if not modal.is_local():
                OUT.commit()
            print(f"[x3-v2] judge checkpoint {len(labels)}/{len(texts)}", flush=True)
        del judge
        persona.free()
    # The checkpoint is authoritative even if the last export was interrupted.
    write_jsonl(output / "labels.jsonl", [dict(cond=t["cond"], item_id=t["item_id"], **lab)
                                         for t, lab in zip(texts, labels)])
    table = summarize(texts, labels, mmlu)
    result = dict(smoke=smoke, table=table, timings=receipt["timings"],
                  minutes=(receipt["generation_seconds"] + completed_seconds + time.time() - tick) / 60,
                  judge=summary(labels, ["concept", "identity", "coherent"]), loads=hf_cache.loads,
                  artifact=f"personakit-out:/{output.name}", manifest_sha256=file_hash(output / "manifest.json"))
    write_json(output / "x3_localize.json", result)
    if not modal.is_local():
        OUT.commit()
    return result


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu="A10G", memory=32768, timeout=1800)
def run(smoke: bool, source_tag: str, s1a_tag: str, tag: str, provenance: dict, score_only: bool):
    import torch
    import evals
    import persona
    import prompts as P

    started = time.time()
    if not modal.is_local():
        OUT.reload()
    if score_only:
        return finish_scoring(out_dir(tag), smoke, provenance)
    sizes, key = (SMOKE, "0.8b") if smoke else (FULL, "4b")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source, s1a, output = out_dir(source_tag), out_dir(s1a_tag), out_dir(tag)
    output.chmod(0o700)
    if any(output.iterdir()):
        raise ValueError("Refusing to overwrite an existing run directory")
    metadata = json.loads((source / "x3.json").read_text())
    records = read_jsonl(s1a / "texts.jsonl")
    model, tok = persona.load(hf_cache.model(*MODELS[key]), device)
    linears = [lin for block in persona.blocks(model) for lin in
               (block.mlp.gate_proj, block.mlp.up_proj, block.mlp.down_proj)]
    adapter, scale = load_adapter(source / "lora.pt", metadata, linears, smoke)
    bands = geometry(len(persona.blocks(model)), smoke)
    originals = [lin.weight.detach().cpu().clone() for lin in linears]
    final_layers = list(range(len(persona.blocks(model)) - 2, len(persona.blocks(model))))
    base_stats, calibration = collect_default_stats(model, tok, records, final_layers, sizes)
    with intervention(model, linears, originals, adapter, scale, "full", bands, {}, {}):
        tuned_stats, tuned_calibration = collect_default_stats(model, tok, records, final_layers, sizes)
    if calibration != tuned_calibration:
        raise ValueError("Base/full calibration spans differ")
    means, candidates = select_shifted(base_stats, tuned_stats)
    targets = list(means)
    prior_targets = {(r["layer"], r["neuron"]) for r in metadata["top_shifted"]}
    prior_late_targets = {target for target in prior_targets if target[0] in final_layers}
    overlap = sorted(set(targets) & prior_targets)
    write_json(output / "calibration.json", {**calibration, "selected": candidates[:20],
                "selection_rule": "descending (abs(z), layer, neuron, z) within final two layers, first20",
                "z_rule": "(mean_full - mean_base)/(std_base + 0.001)",
                "variance_rule": "population float32 sum(x*x)/N - mean**2, clamp negative variance to zero",
                "zero_variance_rule": "retain neuron; fixed denominator floor via std+0.001, as original X3",
                "original_top_shifted": metadata["top_shifted"],
                "prior_overlap": {"count": len(overlap), "ids": overlap,
                                  "prior_late_count": len(prior_late_targets),
                                  "prior_late_not_selected": sorted(prior_late_targets - set(targets)),
                                  "selected_not_in_prior": sorted(set(targets) - prior_targets)},
                "prior_discrepancy": "4B original top20 includes L19 N7490 and only19 in L30-31"})
    write_jsonl(output / "calibration_candidates.jsonl", candidates)
    torch.save(dict(base=base_stats, full=tuned_stats), output / "calibration_stats.pt")
    (output / "calibration_stats.pt").chmod(0o600)
    projected, projection = {}, {}
    for L in bands["project"]:
        i = 3 * L + 2
        projected[i], projection[str(L)] = remove_output_direction(*adapter[i], scale)
    total_parameters = sum(A.numel() + B.numel() for A, B in adapter.values())
    total_energy = sum(adapter_energy(A.double(), B.double(), scale) for A, B in adapter.values())
    accounting = {}
    for cond in CONDITIONS:
        kept = {"base": [], "keep_0_15": bands["early"], "keep_16_27": bands["middle"],
                "keep_28_31": bands["late"], "drop_28_31": bands["early"] + bands["middle"]}.get(cond, list(range(len(linears) // 3)))
        removed_parameters = sum(A.numel() + B.numel() for i, (A, B) in adapter.items() if i // 3 not in kept)
        removed_energy = sum(adapter_energy(A.double(), B.double(), scale) for i, (A, B) in adapter.items() if i // 3 not in kept)
        if cond == "remove_down_rank1":
            removed_energy = sum(p["removed_energy"] for p in projection.values())
        accounting[cond] = dict(total_adapter_parameters=total_parameters, removed_parameters=removed_parameters,
                                removed_parameter_fraction=removed_parameters / total_parameters,
                                original_delta_energy=total_energy, removed_delta_energy=removed_energy,
                                removed_energy_fraction=removed_energy / total_energy,
                                rank1_layers=len(projected) if cond == "remove_down_rank1" else 0,
                                projection_parameter_note="Projection modifies factors; it deletes no stored parameters",
                                clamped_neurons=20 if cond == "clamp_top20" else 0)
    data = hf_cache.dataset(*evals.MMLU, name="all", split="test")
    item_ids = evals.pick(sizes["mmlu"], len(data), 42)
    rows = [dict(data[i]) for i in item_ids]
    manifest = dict(schema=1, smoke=smoke, model=MODELS[key], judge=MODELS[sizes["judge"]], sizes=sizes,
                    source_tag=source_tag, s1a_tag=s1a_tag, output_tag=tag,
                    source_hashes={"lora.pt": file_hash(source / "lora.pt"), "x3.json": file_hash(source / "x3.json"),
                                   "s1a_texts.jsonl": file_hash(s1a / "texts.jsonl")},
                    design=provenance, scale=scale, bands=bands, targets=targets,
                    calibration_spans_hash=calibration["spans_hash"],
                    mmlu_pin=evals.MMLU, mmlu_item_ids=item_ids, mmlu_items_hash=digest(rows),
                    mmlu_protocol="prior persona.mmlu chat 0-shot Answer: next-token ABCD; seed42",
                    prompts="fixed prior test set; not newly unseen", accounting=accounting,
                    projection=projection, projection_scope="independent leading left direction per late layer",
                    inference_note="Paired bootstrap intervals are descriptive, not a powered noninferiority claim.")
    write_json(output / "manifest.json", manifest)
    questions = [("identity", q) for q in P.IDENTITY_TEST[:sizes["identity"]]] + [
                 ("everyday", q) for q in P.EVERYDAY_TEST[:sizes["everyday"]]]
    texts, mmlu, timings = [], {}, {}
    for cond in CONDITIONS:
        tick = time.time()
        with intervention(model, linears, originals, adapter, scale, cond, bands, means, projected) as hits:
            for start in range(0, len(questions), sizes["batch"]):
                chunk = questions[start:start + sizes["batch"]]
                seqs = [persona.chat_ids(tok, q) for _, q in chunk]
                generated = persona.generate(model, tok, seqs, sizes["max_new"])
                texts.extend(dict(cond=cond, split=split, question=q, prompt_hash=digest(ids),
                                  item_id=digest([split, q]), concept="an owl", token_ids=tokens,
                                  reply=tok.decode(tokens, skip_special_tokens=True))
                             for (split, q), ids, tokens in zip(chunk, seqs, generated))
            mmlu[cond] = score_mmlu(model, tok, rows, item_ids, sizes["batch"])
            if cond == "clamp_top20" and hits["clamped_positions"] == 0:
                raise ValueError("Clamp hook was not exercised")
        timings[cond] = dict(seconds=time.time() - tick, **hits)
        write_jsonl(output / "texts.jsonl", texts)
        write_json(output / "mmlu.json", mmlu)
        if not modal.is_local():
            OUT.commit()
        print(f"[x3-v2] {cond}: generation and MMLU complete ({timings[cond]['seconds']:.1f}s)", flush=True)
    # All module references and CPU backups must go before loading the 9B judge.
    del model, linears, originals, adapter, projected
    persona.free()
    write_json(output / "generation_complete.json", dict(
        hashes={name: file_hash(output / name) for name in ("manifest.json", "texts.jsonl", "mmlu.json")},
        timings=timings, generation_seconds=time.time() - started))
    if not modal.is_local():
        OUT.commit()
    return finish_scoring(output, smoke, provenance)


@app.local_entrypoint()
def main(smoke: bool = False, source_tag: str = "", s1a_tag: str = "", tag: str = "", design_commit: str = "",
         score_only: bool = False):
    if not design_commit:
        raise ValueError("Pass the committed DESIGN.md revision; no compute before design approval")
    root = HERE.parent.parent
    revision = subprocess.check_output(["git", "rev-parse", f"{design_commit}^{{commit}}"], cwd=root, text=True).strip()
    design_path = str(HERE.relative_to(root) / "DESIGN.md")
    design = subprocess.check_output(["git", "show", f"{revision}:{design_path}"], cwd=root)
    provenance = dict(commit=revision, path=design_path, sha256=hashlib.sha256(design).hexdigest(),
                      implementation_sha256=file_hash(pathlib.Path(__file__)),
                      support_sha256={name: file_hash(PRIOR / name) for name in ("jobs.py", "persona.py", "judge.py", "prompts.py")})
    key = "0.8b-smoke" if smoke else "4b"
    if score_only and not tag:
        raise ValueError("Score-only resume requires the original output tag")
    tag = tag or f"x3-v2-{key}-{uuid.uuid4().hex[:10]}"
    if any("/" in value or value in (".", "..") for value in (tag, source_tag, s1a_tag) if value):
        raise ValueError("Tags must be single directory names")
    result = call(run, smoke, source_tag or f"x3-{key}", s1a_tag or f"s1a-{key}", tag, provenance, score_only)
    print("PERSONAKIT_RESULT", json.dumps({"smoke": smoke, "minutes": result["minutes"],
          **{f"{cond}_ci": row["ci"]["mean"] for cond, row in result["table"].items()},
          **{f"{cond}_mmlu": row["mmlu"]["mean"] for cond, row in result["table"].items()},
          "details": result}, allow_nan=False))
