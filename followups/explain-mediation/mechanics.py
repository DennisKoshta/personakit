"""Design-independent Qwen3.5 text experiment primitives; no model loading or jobs.

Swaps replace parameters and persistent buffers, not module objects/configuration.
Donor mappings use full state-dict keys (``model.layers.L.<module>.<key>``).
Decoder input_layernorm/post_attention_layernorm are deliberately excluded.
Do not mutate model structure, placement, or weights inside a swap context.

Continuation scoring uses the Qwen3_5ForCausalLM contract verified in Transformers
5.20.2: model.model returns final-normalized states; model.lm_head is the entire
output projection. Real-model equivalence remains a required integration smoke.
The steering callback receives ORIGINAL row indices and returns a fresh context
manager that installs/removes its own hooks, including on exceptions. It must
rebuild row vectors for those indices on every call (including an OOM retry).
"""

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
import gc
import math
from numbers import Integral

import torch


def _selected_modules(model, layers: Sequence[int], kind: str) -> dict:
    if kind not in ("attention", "mlp"):
        raise ValueError("kind must be 'attention' or 'mlp'")
    layers = list(layers)
    if not layers or any(isinstance(i, bool) or not isinstance(i, Integral) for i in layers):
        raise ValueError("layers must be a nonempty sequence of integer indices")
    if len(set(layers)) != len(layers):
        raise ValueError("duplicate layers")
    blocks = model.model.layers
    selected = {}
    for i in layers:
        if not 0 <= i < len(blocks):
            raise ValueError(f"layer {i} outside model")
        block = blocks[i]
        if kind == "attention":
            names = [n for n in ("self_attn", "linear_attn") if getattr(block, n, None) is not None]
            if len(names) != 1:
                raise ValueError(f"layer {i} must have exactly one attention module")
            name = names[0]
        else:
            name = "mlp"
        module = getattr(block, name)
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"layer {i}.{name} is not a module")
        selected[f"model.layers.{i}.{name}"] = module
    return selected


def _selected_state(model, layers, kind) -> dict:
    state = {}
    for prefix, module in _selected_modules(model, layers, kind).items():
        part = module.state_dict(keep_vars=True)
        if not part:
            raise ValueError(f"empty state for {prefix}")
        for key, value in part.items():
            if not isinstance(value, torch.Tensor) or value.is_meta or value.layout != torch.strided:
                raise TypeError(f"{prefix}.{key} must be a materialized dense tensor")
            state[f"{prefix}.{key}"] = value
    return state


def selected_module_state(model, layers: Sequence[int], kind: str) -> dict[str, torch.Tensor]:
    """Detached CPU clones of COMPLETE selected modules, suitable as donor_state."""
    return {k: v.detach().to("cpu", copy=True) for k, v in _selected_state(model, layers, kind).items()}


@contextmanager
def reversible_swap(model, layers: Sequence[int], kind: str, donor_state: Mapping[str, torch.Tensor]):
    """Temporarily copy a complete donor state; yield a JSON-serializable manifest.

    Exact keys, shapes AND dtypes are required; callers must explicitly convert
    donor dtypes if needed. All validation and CPU snapshots precede mutation.
    Original/donor snapshots stay on CPU; copies transfer one tensor at a time.
    Restores every selected tensor even after a partially failed application.
    Donor provenance (model pin, file hash) belongs in the caller's run record.
    """
    target = _selected_state(model, layers, kind)
    if set(donor_state) != set(target):
        missing, extra = sorted(set(target) - set(donor_state)), sorted(set(donor_state) - set(target))
        raise ValueError(f"donor keys differ: missing={missing}, extra={extra}")
    original, donor, entries = {}, {}, []
    for key, dest in target.items():
        src = donor_state[key]
        if not isinstance(src, torch.Tensor) or src.is_meta or src.layout != torch.strided:
            raise TypeError(f"invalid donor tensor {key}")
        if src.shape != dest.shape or src.dtype != dest.dtype:
            raise ValueError(f"shape/dtype mismatch for {key}: {src.shape}/{src.dtype} vs {dest.shape}/{dest.dtype}")
        old, new = dest.detach().to("cpu", copy=True), src.detach().to("cpu", copy=True)
        if not torch.isfinite(old).all() or not torch.isfinite(new).all():
            raise ValueError(f"nonfinite swap tensor {key}")
        original[key], donor[key] = old, new
        entries.append({"key": key, "shape": list(dest.shape), "dtype": str(dest.dtype),
                        "changed": not torch.equal(old, new)})
    manifest = {"kind": kind, "excluded": ["input_layernorm", "post_attention_layernorm"],
                "tensors": entries, "changed_tensors": sum(e["changed"] for e in entries)}
    try:
        with torch.no_grad():
            for key, dest in target.items():
                dest.copy_(donor[key])
        yield manifest
    finally:
        failures = []
        with torch.no_grad():
            for key, dest in target.items():
                try:
                    dest.copy_(original[key])
                except Exception as exc:
                    failures.append((key, exc))
        if failures:
            raise RuntimeError(f"swap restoration failed for {[k for k, _ in failures]}") from failures[0][1]


def _vector(value, name: str) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.is_complex():
        raise ValueError(f"{name} must be real")
    value = value.detach().to(device="cpu", dtype=torch.float64, copy=True)
    if value.ndim != 1 or not value.numel() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return value


def upper_cap(hooks, layer: int, unit: torch.Tensor, ceiling: float) -> None:
    """Register h·unit <= ceiling via existing Hooks.cap(-unit, -ceiling).

    Reject non-unit directions instead of silently changing the ceiling's units.
    Numerical tolerances here validate representation; they are not study cutoffs.
    """
    u = _vector(unit, "unit")
    if not math.isclose(float(u.norm()), 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("cap direction must have unit norm")
    if not math.isfinite(float(ceiling)):
        raise ValueError("ceiling must be finite")
    if isinstance(layer, bool) or not isinstance(layer, Integral) or not 0 <= layer < len(hooks.model.model.layers):
        raise ValueError("cap layer outside model")
    if u.numel() != hooks.model.config.hidden_size:
        raise ValueError("cap direction width differs from model hidden size")
    hooks.cap(int(layer), -u.float(), -float(ceiling))


def orthogonalize(vector: torch.Tensor, basis: Sequence[torch.Tensor], *, rtol: float = 1e-6) -> torch.Tensor:
    """CPU FP32 unit residual orthogonal to the JOINT span of basis vectors.

    SVD in FP64 handles duplicate/nonorthogonal basis rows. Zero basis vectors,
    zero input, or a numerically vanishing residual are errors, not random fills.
    rtol is a relative numerical rank/degeneracy tolerance, not a scientific cut.
    """
    if not 0 < rtol < 1 or not math.isfinite(rtol):
        raise ValueError("rtol must be finite and between 0 and 1")
    v = _vector(vector, "vector")
    norm = v.norm()
    if not torch.isfinite(norm) or norm == 0:
        raise ValueError("zero or unrepresentable vector norm")
    rows = [_vector(b, "basis vector") for b in basis]
    if any(b.shape != v.shape or b.norm() == 0 or not torch.isfinite(b.norm()) for b in rows):
        raise ValueError("basis vectors must have matching width and nonzero finite norms")
    residual = v / norm
    if rows:
        matrix = torch.stack([b / b.norm() for b in rows])
        _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        q = vh[singular > singular[0] * rtol]
        residual = residual - q.T @ (q @ residual)
        residual = residual - q.T @ (q @ residual)  # remove cancellation residue
    magnitude = residual.norm()
    if not torch.isfinite(magnitude) or magnitude <= rtol:
        raise ValueError("direction degenerates after joint-span projection")
    result = (residual / magnitude).float()
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite orthogonalized direction")
    return result


def _score_batch(model, prompts, responses, indices, pad_token_id, projection_batch_size, steering):
    """One attempt in its own frame: an OOM must release hidden states/logits."""
    device = model.get_input_embeddings().weight.device
    # Input ends at the penultimate response token: every input row has exactly
    # one causal prediction position for every scored continuation token.
    seqs = [prompts[i] + responses[i][:-1] for i in indices]
    width = max(map(len, seqs))
    ids = torch.full((len(indices), width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    owners, positions, targets = [], [], []
    for row, (i, seq) in enumerate(zip(indices, seqs)):
        offset = width - len(seq)
        ids[row, offset:] = torch.tensor(seq, device=device)
        mask[row, offset:] = 1
        start = offset + len(prompts[i]) - 1
        owners.extend([row] * len(responses[i]))
        positions.extend(range(start, start + len(responses[i])))
        targets.extend(responses[i])
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    sums = torch.zeros(len(indices), dtype=torch.float64)
    with torch.inference_mode(), (steering(tuple(indices)) if steering else nullcontext()):
        hidden = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                             use_cache=False, return_dict=True).last_hidden_state
        if hidden.ndim != 3 or hidden.shape[:2] != ids.shape:
            raise ValueError("backbone returned unexpected hidden-state shape")
        for start in range(0, len(targets), projection_batch_size):
            stop = start + projection_batch_size
            row = torch.tensor(owners[start:stop], device=hidden.device)
            pos = torch.tensor(positions[start:stop], device=hidden.device)
            logits = model.lm_head(hidden[row, pos]).float()
            if logits.ndim != 2 or logits.shape != (len(row), model.config.vocab_size):
                raise ValueError("output head returned unexpected logits shape")
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite continuation logits")
            target = torch.tensor(targets[start:stop], device=logits.device)
            logs = logits.gather(1, target[:, None]).squeeze(1) - logits.logsumexp(-1)
            if not torch.isfinite(logs).all():
                raise ValueError("nonfinite token log probability")
            sums.index_add_(0, row.cpu(), logs.double().cpu())
            del logits, logs
    return sums.tolist()


def continuation_logprobs(model, prompt_ids: Sequence[Sequence[int]], response_ids: Sequence[Sequence[int]], *,
                         pad_token_id: int, batch_size: int = 8, projection_batch_size: int = 32,
                         reduction: str = "sum", steering: Callable | None = None) -> list[dict]:
    """Full response-only conditional log probabilities in ORIGINAL input order.

    Returns [{"logprob": float, "n_tokens": int}], summed by default; explicit
    reduction="mean" divides by response length. Prompts/responses are token ID
    sequences, not strings. No truncation/EOS removal is performed. Every prompt
    and response must be nonempty. Empty datasets return []. Run in eval mode on
    one materialized device; Qwen3.5 text architecture only (no offload/vision).

    Logit storage is bounded by projection_batch_size * vocabulary size, not by
    sequence length or population. Hidden states still scale with batch * length.
    On OOM the whole failed subbatch is discarded, steering exits, and a fresh
    attempt uses fewer rows (then fewer projection positions at batch size one).
    The callback must clean up on failure; it must not reuse another row ordering.
    """
    if getattr(model.config, "model_type", None) != "qwen3_5_text":
        raise ValueError("scorer requires Qwen3.5 text backbone + unmodified lm_head semantics")
    if any(m.training for m in model.modules()):
        raise ValueError("scoring requires eval mode (including all submodules)")
    devices = {p.device for p in model.parameters()}
    if len(devices) != 1 or next(iter(devices)).type == "meta":
        raise ValueError("scoring requires a single materialized model device")
    if reduction not in ("sum", "mean"):
        raise ValueError("reduction must be 'sum' or 'mean'")
    for n in (batch_size, projection_batch_size):
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
            raise ValueError("batch sizes must be positive integers")
    vocab = model.config.vocab_size
    valid_id = lambda t: isinstance(t, Integral) and not isinstance(t, bool) and 0 <= t < vocab
    if not valid_id(pad_token_id):
        raise ValueError("pad token outside vocabulary")
    prompts, responses = [list(p) for p in prompt_ids], [list(r) for r in response_ids]
    if len(prompts) != len(responses):
        raise ValueError("prompt/response row counts differ")
    if any(not seq or not all(valid_id(t) for t in seq) for seq in prompts + responses):
        raise ValueError("each prompt/response needs nonempty valid token IDs")
    result, cursor = [], 0
    while cursor < len(prompts):
        indices = list(range(cursor, min(len(prompts), cursor + batch_size)))
        oom = False
        try:
            scores = _score_batch(model, prompts, responses, indices, pad_token_id,
                                  projection_batch_size, steering)
        except torch.OutOfMemoryError:
            if len(indices) == 1 and projection_batch_size == 1:
                raise
            oom = True
        if oom:  # outside except: traceback must release the failed attempt
            gc.collect()
            if next(iter(devices)).type == "cuda":
                torch.cuda.empty_cache()
            if len(indices) > 1:
                batch_size = max(1, len(indices) // 2)
            else:
                projection_batch_size = max(1, projection_batch_size // 2)
            continue
        for i, score in zip(indices, scores):
            value = score if reduction == "sum" else score / len(responses[i])
            if not math.isfinite(value):
                raise ValueError("nonfinite continuation score")
            result.append({"logprob": value, "n_tokens": len(responses[i])})
        cursor += len(indices)
    return result
