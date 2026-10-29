"""Real CPU mechanics smoke, NOT the study's statistical/selection smoke.

Execute only after the design commit and the project owner's execution authorization:
  python \
    followups/explain-mediation/smoke.py --smoke

Planning peak: 6–8 GiB (one fp32 0.8B, selected donor snapshots, dense reference
logits and judge prefill); admit at 8 GiB and record actual ru_maxrss. No second
model, full donor, accelerator, paid call, or old fixture directory is used.
Tolerances below concern numerical plumbing only, never scientific outcomes.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "study"))
import hf_cache
from jobs import MODELS

app = modal.App("personakit-explain-mechanics-smoke")
LAYERS = [20, 21, 22, 23]
# Full response sums: dense vs chunked is tighter than singleton vs padded.
DENSE_ATOL, DENSE_RTOL = 2e-4, 2e-5
PAD_ATOL, PAD_RTOL = 3e-3, 2e-4
PROJ_ATOL = 3e-4


def progress(message):
    print(f"[mechanics smoke] {message}", flush=True)


def hook_inventory(model):
    return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]


def check_finite(value):
    import torch
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), "nonfinite tensor"
    elif isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            check_finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value), "nonfinite scalar"


def assert_scores(actual, expected, *, padded=False):
    import torch
    assert [r["n_tokens"] for r in actual] == [r["n_tokens"] for r in expected]
    a = torch.tensor([r["logprob"] for r in actual], dtype=torch.float64)
    b = torch.tensor([r["logprob"] for r in expected], dtype=torch.float64)
    torch.testing.assert_close(a, b, atol=PAD_ATOL if padded else DENSE_ATOL,
                               rtol=PAD_RTOL if padded else DENSE_RTOL)
    return float((a - b).abs().max())


def dense_scores(model, prompts, replies, pad, steering=None):
    """Independent oracle: regular causal LM forward, dense log_softmax + shift.

    Include the LAST reply token in the input, unlike the chunked scorer. Slice
    out only causal predictions for reply tokens; no prompt scores or EOS tricks.
    """
    import torch
    import persona
    ids, mask = persona.pad_left([p + r for p, r in zip(prompts, replies)], pad, "cpu")
    with torch.inference_mode(), (steering(tuple(range(len(prompts)))) if steering else nullcontext()):
        logits = model(input_ids=ids, attention_mask=mask,
                       position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
                       use_cache=False, logits_to_keep=0).logits
        check_finite(logits)
        logs = logits.log_softmax(-1)
        result = []
        for row, (p, r) in enumerate(zip(prompts, replies)):
            start = ids.shape[1] - len(p) - len(r) + len(p) - 1
            targets = torch.tensor(r)
            score = logs[row, start:start + len(r)].gather(1, targets[:, None]).double().sum()
            result.append({"logprob": float(score), "n_tokens": len(r)})
    return result


def cached_steps(model, prompt):
    """Fresh prefill and exactly one greedy cached decode, even if EOS is chosen."""
    import torch
    with torch.inference_mode():
        ids = torch.tensor([prompt])
        first = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                      use_cache=True, logits_to_keep=1)
        assert first.past_key_values is not None
        token = first.logits[:, -1].argmax(-1, keepdim=True)
        second = model(input_ids=token, attention_mask=torch.ones(1, len(prompt) + 1, dtype=torch.long),
                       position_ids=torch.tensor([[len(prompt)]]),
                       past_key_values=first.past_key_values, use_cache=True, logits_to_keep=1)
        check_finite([first.logits, second.logits])
        return first.logits.detach().clone(), second.logits.detach().clone()


def swaps(model, prompt, mechanics):
    import torch
    blocks = model.model.layers
    assert all(getattr(blocks[i], "linear_attn", None) is not None for i in LAYERS[:-1])
    assert getattr(blocks[23], "self_attn", None) is not None
    baseline = cached_steps(model, prompt)
    reports = []
    for kind in ("attention", "mlp"):
        progress(f"complete {kind} swaps, both perturbation signs, sham and exception restoration")
        original = mechanics.selected_module_state(model, LAYERS, kind)
        # Independent complete key manifest, including DeltaNet direct parameters.
        expected = {}
        for layer in LAYERS:
            name = "mlp" if kind == "mlp" else ("self_attn" if layer == 23 else "linear_attn")
            for key, value in getattr(blocks[layer], name).state_dict().items():
                expected[f"model.layers.{layer}.{name}.{key}"] = value
        assert set(original) == set(expected) and original
        assert all(torch.equal(original[k], v) and original[k].data_ptr() != v.data_ptr()
                   for k, v in expected.items())
        norms = {f"{i}.{n}": getattr(blocks[i], n).weight.detach().clone()
                 for i in LAYERS for n in ("input_layernorm", "post_attention_layernorm")}

        def restored():
            assert all(torch.equal(v, expected[k]) for k, v in original.items()), "swap not restored"
            assert all(torch.equal(v, getattr(blocks[int(k.split('.')[0])], k.split('.')[1]).weight)
                       for k, v in norms.items()), "decoder norm changed"

        with mechanics.reversible_swap(model, LAYERS, kind, original) as manifest:
            assert manifest["changed_tensors"] == 0
            sham = cached_steps(model, prompt)
            for a, b in zip(sham, baseline):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        restored()
        for sign in (1, -1):
            donor = {k: v.clone() for k, v in original.items()}
            for layer in LAYERS:
                suffix = "mlp.down_proj.weight" if kind == "mlp" else (
                    "self_attn.o_proj.weight" if layer == 23 else "linear_attn.out_proj.weight")
                donor[f"model.layers.{layer}.{suffix}"][0].add_(sign * 0.01)
            with mechanics.reversible_swap(model, LAYERS, kind, donor) as manifest:
                assert manifest["changed_tensors"] == len(LAYERS)
                assert all(torch.equal(expected[k], v) for k, v in donor.items())
                changed = cached_steps(model, prompt)
                assert any(not torch.equal(a, b) for a, b in zip(changed, baseline)), "vacuous donor"
                reports.append({"sign": sign, **manifest})
            restored()
            try:
                with mechanics.reversible_swap(model, LAYERS, kind, donor):
                    raise RuntimeError("intentional swap body failure")
            except RuntimeError as exc:
                assert str(exc) == "intentional swap body failure"
            restored()
        for a, b in zip(cached_steps(model, prompt), baseline):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    return reports


def continuation_checks(model, tok, prompts, replies, mechanics):
    import torch
    import persona
    inventory = hook_inventory(model)
    # Distinct rows: positive at L20, negative at L21, and a true zero control.
    pushes = {20: (0, torch.tensor([0.25, 0., 0.])),
              21: (1, torch.tensor([0., -0.375, 0.]))}
    attempts, exited = [], []

    # Observe actual neuron inputs before/after Hooks, not merely its config.
    before, after, observers = {}, {}, []
    def observe(store, layer):
        def capture(module, args):
            store[layer] = args[0].detach().clone()
        return capture

    try:
        for layer in pushes:
            module = model.model.layers[layer].mlp.down_proj
            observers.append(module.register_forward_pre_hook(observe(before, layer)))
        with persona.Hooks(model) as hooks:
            for layer, (neuron, delta) in pushes.items():
                hooks.push(layer, neuron, delta)
                module = model.model.layers[layer].mlp.down_proj
                observers.append(module.register_forward_pre_hook(observe(after, layer)))
            dense_scores(model, prompts, replies, tok.pad_token_id)
        for layer, (neuron, delta) in pushes.items():
            expected = before[layer].clone()
            expected[..., neuron] += delta[:, None]
            torch.testing.assert_close(after[layer], expected, atol=0, rtol=0)
            assert not torch.equal(after[layer], before[layer])
            assert torch.equal(after[layer][2], before[layer][2])
    finally:
        for handle in observers:
            handle.remove()
    assert hook_inventory(model) == inventory
    del before, after

    @contextmanager
    def steering(indices):
        attempts.append(list(indices))
        try:
            with persona.Hooks(model) as hooks:
                for layer, (neuron, delta) in pushes.items():
                    hooks.push(layer, neuron, delta[list(indices)])
                yield
        finally:
            assert hook_inventory(model) == inventory, "steering hook leak"
            exited.append(list(indices))

    def score(batch, steer=None, projection=2, reduction="sum"):
        return mechanics.continuation_logprobs(model, prompts, replies, pad_token_id=tok.pad_token_id,
                                               batch_size=batch, projection_batch_size=projection,
                                               steering=steer, reduction=reduction)

    progress("dense causal response sums, uneven padded batching and signed per-row steering")
    reference = dense_scores(model, prompts, replies, tok.pad_token_id)
    batched, single = score(3), score(1)
    dense_error = assert_scores(batched, reference)
    padded_error = assert_scores(single, batched, padded=True)
    means = score(3, reduction="mean")
    assert_scores([{**r, "logprob": r["logprob"] * r["n_tokens"]} for r in means], batched)
    steered_ref = dense_scores(model, prompts, replies, tok.pad_token_id, steering)
    steered = score(3, steering)
    assert_scores(steered, steered_ref)
    assert_scores(score(1, steering), steered, padded=True)
    assert_scores([steered[2]], [batched[2]])
    assert any(abs(steered[i]["logprob"] - batched[i]["logprob"]) > 1e-7 for i in (0, 1))

    # Fail INSIDE the real forward after steering is installed, not on entry.
    progress("simulated backbone and output-head OOM retries, index mapping and hook cleanup")
    for target, first_batch, projection in ((model.model, 3, 2), (model.lm_head, 1, 4)):
        fired = []

        def fail_once(module, args, kwargs):
            if not fired:
                fired.append(True)
                raise torch.OutOfMemoryError("intentional smoke OOM; no memory exhausted")

        handle = target.register_forward_pre_hook(fail_once, with_kwargs=True)
        # The injector itself is part of the expected inventory during this call.
        inventory = hook_inventory(model)
        attempts.clear()
        exited.clear()
        try:
            retried = score(first_batch, steering, projection)
            assert fired and attempts == exited
            expected_attempts = [[0, 1, 2], [0], [1], [2]] if first_batch == 3 else [[0], [0], [1], [2]]
            assert attempts == expected_attempts, attempts
            assert_scores(retried, steered, padded=True)
        finally:
            handle.remove()
            inventory = hook_inventory(model)
    assert mechanics.continuation_logprobs(model, [], [], pad_token_id=tok.pad_token_id) == []
    return {"dense_max_abs_error": dense_error, "padding_max_abs_error": padded_error,
            "baseline": batched, "steered": steered, "pushes": {str(k): [n, d.tolist()] for k, (n, d) in pushes.items()}}


def captures_and_caps(model, tok, prompts, replies, mechanics):
    import torch
    import persona
    progress("fresh paired first/third-person response capture and orthogonalization")
    seqs = [p + r for p, r in zip(prompts, replies)]
    starts = list(map(len, prompts))
    with persona.Hooks(model) as hooks:
        means = persona.span_means(model, hooks, seqs, starts, LAYERS)
        assert hooks.capture is None
        assert means.shape == (len(seqs), len(LAYERS), model.config.hidden_size)
        self_reference = means[0, -1] - means[1, -1]
        # Plumbing basis from this fresh smoke, NOT an estimated Assistant Axis.
        assistant = mechanics.orthogonalize(means[2, -1], [])
        concept = mechanics.orthogonalize(means[1, -1], [assistant])
        unit = mechanics.orthogonalize(self_reference, [assistant, concept, assistant + concept])
        assert abs(float(unit.norm()) - 1) < 1e-6
        assert abs(float(unit @ assistant)) < 1e-5 and abs(float(unit @ concept)) < 1e-5
        for vector, basis in ((assistant, [assistant]), (torch.zeros_like(unit), []),
                              (unit, [torch.zeros_like(unit)])):
            try:
                mechanics.orthogonalize(vector, basis)
            except ValueError:
                pass
            else:
                raise AssertionError("degenerate direction accepted")
        directions = {23: torch.stack([unit, assistant])}
        natural = persona.token_projections(model, hooks, seqs, starts, directions)[23]
        assert natural.shape == (sum(map(len, replies)), 2)
        offset = 0
        for row, reply in enumerate(replies):
            torch.testing.assert_close(natural[offset:offset + len(reply)].mean(0),
                                       means[row, -1] @ directions[23].T, atol=PROJ_ATOL, rtol=1e-5)
            offset += len(reply)
        # Force active constraints on EVERY reply token, avoiding vacuous caps.
        ceiling = float(natural[:, 0].min()) - 0.5
        floor = float(natural[:, 1].max()) + 0.5
        mechanics.upper_cap(hooks, 23, unit, ceiling)
        hooks.cap(23, assistant, floor)
        capped = persona.token_projections(model, hooks, seqs, starts, directions)[23]
        assert (capped[:, 0] <= ceiling + PROJ_ATOL).all()
        assert (capped[:, 1] >= floor - PROJ_ATOL).all()
        assert (natural[:, 0] - capped[:, 0] > 0.49).all()
        assert (capped[:, 1] - natural[:, 1] > 0.49).all()
        progress("active upper/lower caps and signed pushes through prefill and cached decode")
        hooks.push(20, 0, 0.25)
        hooks.push(21, 1, -0.375)
        hooks.push(22, 2, 0.)
        hooks.capture = {}
        cached_steps(model, prompts[0])
        assert set(hooks.capture) == set(range(len(model.model.layers)))
        for values in hooks.capture.values():
            assert len(values) == 2 and values[0].shape[1] == len(prompts[0]) and values[1].shape[1] == 1
        for hidden in hooks.capture[23]:
            assert ((hidden.float() @ unit) <= ceiling + PROJ_ATOL).all()
            assert ((hidden.float() @ assistant) >= floor - PROJ_ATOL).all()
        hooks.capture = None
        hooks.clear()
    check_finite([means, unit, assistant, concept, natural, capped])
    return {"means": means, "unit": unit, "assistant_smoke_unit": assistant, "concept_smoke_unit": concept,
            "natural_projections": natural, "capped_projections": capped,
            "ceiling": ceiling, "floor": floor}


@app.function(image=hf_cache.image(), volumes=hf_cache.volumes, timeout=3600)
def run(smoke: bool = False) -> dict:
    if not smoke or os.environ.get("PERSONAKIT_RESEARCH_CHECK") != "1" or not modal.is_local():
        raise RuntimeError("CPU checker only: python <this file> --smoke")
    import torch
    import transformers
    import persona
    import judge
    import mechanics

    start = time.monotonic()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(0)
    root = Path("out") / "study-smoke"
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="explain-mediation-", dir=root))
    progress(f"artifacts: {out}; loading pinned 0.8B once on CPU float32")
    repo, revision = MODELS["0.8b"]
    model, tok = persona.load(hf_cache.model(repo, revision), "cpu")
    assert len(model.model.layers) == 24 and model.config.hidden_size == 1024
    assert all(p.device.type == "cpu" and (not p.is_floating_point() or p.dtype == torch.float32)
               for p in model.parameters())
    assert not any(m.training for m in model.modules())
    initial_hooks = hook_inventory(model)
    # New fixed benign pair, identical prompt, matched activity; third row makes
    # prompt AND response lengths uneven. These are fixtures, never study labels.
    question = "Describe a gardener watering plants."
    records = [
        {"pair_id": "gardener-0", "person": "first", "question": question, "reply": "I water plants.", "concept": "gardening"},
        {"pair_id": "gardener-0", "person": "third", "question": question, "reply": "She waters plants.", "concept": "gardening"},
        {"pair_id": "control-0", "person": "control", "question": "Name a garden task.", "reply": "Water the green plants gently.", "concept": "gardening"},
    ]
    prompts = [persona.chat_ids(tok, row["question"]) for row in records]
    replies = [tok.encode(row["reply"], add_special_tokens=False) for row in records]
    assert prompts[0] == prompts[1] and all(len(r) > 1 for r in replies)
    assert len(set(map(len, prompts))) > 1 and len(set(map(len, replies))) > 1
    swap_report = swaps(model, prompts[0], mechanics)
    scores = continuation_checks(model, tok, prompts, replies, mechanics)
    tensors = captures_and_caps(model, tok, prompts, replies, mechanics)
    assert hook_inventory(model) == initial_hooks
    progress("actual pinned 0.8B judge: identity, coherent, concept (no quality threshold)")
    labels = judge.score(model, tok, records, ["identity", "coherent", "concept"], batch=1)
    assert len(labels) == len(records)
    for label in labels:
        for rubric in ("identity", "coherent", "concept"):
            assert label[rubric] in (0, 1)
            probs, mass = label[f"{rubric}_p"], label[f"{rubric}_mass"]
            assert len(probs) == 2 and all(0 <= p <= 1 for p in probs)
            assert abs(sum(probs) - 1) <= 2e-4 and 0 <= mass <= 1
    assert hook_inventory(model) == initial_hooks
    result = {"smoke": True, "mechanics_only": True, "model": repo, "revision": revision,
              "judge_model": repo, "judge_revision": revision, "device": "cpu", "dtype": "float32",
              "layers": len(model.model.layers), "swap_layers": LAYERS, "output": str(out),
              "dense_atol": DENSE_ATOL, "dense_rtol": DENSE_RTOL, "padding_atol": PAD_ATOL,
              "padding_rtol": PAD_RTOL, "projection_atol": PROJ_ATOL,
              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
              "seconds": time.monotonic() - start,
              "details": {"records": records, "prompt_ids": prompts, "response_ids": replies,
                          "labels": labels, "swaps": swap_report, "scores": scores,
                          "hf_cache_loads": hf_cache.loads, "torch": torch.__version__,
                          "transformers": transformers.__version__, "packages": list(hf_cache.PACKAGES),
                          "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in (Path(__file__), Path(mechanics.__file__))},
                          "limitations": ["No statistical/selection study-path coverage or scientific score decisions",
                                          "Synthetic selected donor perturbations, not Base/Instruct swaps",
                                          "Fresh smoke directions, not calibrated study axes",
                                          "CPU only; no CUDA/flash kernels or full-size memory/throughput validation"]}}
    check_finite([result, tensors])
    progress("serialization/reload and finite-value checks")
    torch.save(tensors, out / "captures.pt")
    restored = torch.load(out / "captures.pt", map_location="cpu", weights_only=True)
    assert set(restored) == set(tensors)
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, restored[key])
        else:
            assert value == restored[key]
    check_finite(restored)
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    assert json.loads((out / "result.json").read_text()) == result
    progress("all mechanics assertions passed; statistical study smoke remains separate")
    return result


@app.local_entrypoint()
def main(smoke: bool = False):
    if not smoke or os.environ.get("PERSONAKIT_RESEARCH_CHECK") != "1":
        raise RuntimeError("Use python with --smoke after owner authorization")
    result = run.remote(smoke=smoke)
    print("PERSONAKIT_RESULT", json.dumps(result, allow_nan=False), flush=True)
