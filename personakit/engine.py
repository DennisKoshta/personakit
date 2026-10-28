"""Steering, capture and generation for Qwen3.5 text models (any size).

Interventions sit at a decoder layer's output, the residual stream leaving layer L (0-indexed), except `push`, which
adds to one MLP neuron's activation silu(gate)*up at the input of down_proj. The two are the same edit: pushing neuron n
by delta adds delta * W_down[:, n] to the layer's output. Every intervention applies at every position.
"""
import gc
import os

import torch

HEADROOM = 3 * 2 ** 30  # GPU bytes an offloaded model leaves free for its activations


def gpu_cap():
    """Bytes of the GPU personakit may use: $PERSONAKIT_GPU_GB, if set, as on a smaller card (the allocator is held
    to it), or else the whole card."""
    total = torch.cuda.get_device_properties(0).total_memory
    if not os.environ.get("PERSONAKIT_GPU_GB"):
        return total
    cap = min(total, int(float(os.environ["PERSONAKIT_GPU_GB"]) * 2 ** 30))
    torch.cuda.set_per_process_memory_fraction(cap / total)
    return cap


def load(path: str, device: str, offload: bool = False):
    """The text model (no vision tower) in bf16 on a GPU, fp32 on a CPU, and its tokenizer (left padding). With
    `offload`, on an NVIDIA GPU, layers that don't fit stay in CPU memory and move to the GPU as each batch needs them."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    if offload and device == "cuda":
        from accelerate.utils import get_max_memory
        room = min(torch.cuda.mem_get_info()[0], gpu_cap() - torch.cuda.memory_reserved()) - HEADROOM
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="auto",
                                                     max_memory={0: max(room, 0), "cpu": get_max_memory()["cpu"]})
        model = model.eval().requires_grad_(False)
    else:
        if device == "cuda":
            gpu_cap()
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    return model, tok


def free():
    """Return GPU memory the caller has already let go of (del its names first)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def batched(fn, items: list, batch: int) -> list:
    """fn over items, `batch` at a time, its lists of results joined; a batch too big for the GPU is halved until it
    fits."""
    out, s = [], 0
    while s < len(items):
        chunk, fits = items[s:s + batch], False
        try:
            out += fn(chunk)
            fits = True
        except torch.OutOfMemoryError:
            if batch == 1:
                raise
        if fits:
            s += len(chunk)
        else:  # outside the except block, whose traceback still holds the failed batch's tensors
            batch //= 2
            print(f"[personakit] {len(chunk)} at a time doesn't fit on the GPU: {batch} at a time", flush=True)
            free()
    return out


def blocks(model):
    return model.model.layers


def down_col(model, layer: int, n: int) -> torch.Tensor:
    return blocks(model)[layer].mlp.down_proj.weight[:, n].float()


def chat_ids(tok, user: str, system: str = "", reply: str | None = None, history: list | None = None) -> list[int]:
    """Token ids of a chat with thinking off, ending at the reply's start (or after `reply`, if given). `history` is
    earlier (user, assistant) turns."""
    msgs = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": r, "content": c} for u, a in history or [] for r, c in (("user", u), ("assistant", a))] + \
        [{"role": "user", "content": user}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    ids = list(out["input_ids"] if hasattr(out, "keys") else out)
    return ids + (tok.encode(reply, add_special_tokens=False) if reply else [])


class Hooks:
    """Interventions on a model while inside `with Hooks(model) as h:`. Per-row values (a [B, d] vector, a [B] delta)
    steer each batch row differently; a single value steers them all."""

    def __init__(self, model):
        self.model, self.handles = model, []
        self.adds, self.pushes, self.caps = {}, {}, {}  # layer -> list
        self.capture = None  # layer -> list of [B, T, d] outputs, when capturing
        self.stats = None  # when set to Stats, records neuron extremes and residual norms over its mask

    def add(self, layer: int, vec: torch.Tensor):
        """Add vec ([d] or [B, d]) to the residual leaving `layer`."""
        self.adds.setdefault(layer, []).append(vec)

    def push(self, layer: int, n: int, delta):
        """Add delta (a float or [B]) to neuron n's activation at `layer`."""
        self.pushes.setdefault(layer, []).append((n, delta))

    def cap(self, layer: int, unit: torch.Tensor, floor: float):
        """Keep the residual's projection on `unit` at or above `floor` (Lu et al. 2026's capping, on the Assistant
        end): h += max(0, floor - h.u) u."""
        self.caps.setdefault(layer, []).append((unit, floor))

    def clear(self):
        self.adds, self.pushes, self.caps = {}, {}, {}

    def __enter__(self):
        for i, block in enumerate(blocks(self.model)):
            self.handles.append(block.register_forward_hook(self._out_hook(i)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self._push_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _out_hook(self, i):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if i in self.adds or i in self.caps:
                h = h.clone()
                for v in self.adds.get(i, []):
                    v = v.to(h.device, h.dtype)
                    h += v[:, None, :] if v.dim() == 2 else v
                for u, floor in self.caps.get(i, []):
                    u = u.to(h.device, torch.float32)
                    short = (floor - h.float() @ u).clamp(min=0)
                    h += (short[..., None] * u).to(h.dtype)
            if self.capture is not None:
                self.capture.setdefault(i, []).append(h.detach())
            if self.stats is not None:
                self.stats.norms.setdefault(i, []).append(h.float().norm(dim=-1)[self.stats.mask].cpu())
            if isinstance(out, tuple):
                return (h, *out[1:])
            return h
        return hook

    def _push_hook(self, i):
        def hook(mod, args):
            if self.stats is not None:
                self.stats.update(i, args[0])
            if i not in self.pushes:
                return None
            a = args[0].clone()
            for n, delta in self.pushes[i]:
                d = torch.as_tensor(delta, device=a.device, dtype=a.dtype)
                a[..., n] += d[:, None] if d.dim() == 1 else d
            return (a, *args[1:])
        return hook


class Stats:
    """Each neuron's natural extremes (max and min of silu(gate)*up) and the residual norm leaving each layer, over
    the tokens a mask selects (real tokens, not the first: it is an attention sink with outsized norms)."""

    def __init__(self):
        self.mask, self.hi, self.lo, self.norms, self.tokens = None, {}, {}, {}, 0

    def update(self, layer: int, a: torch.Tensor):
        x = a[self.mask].float()
        hi, lo = x.max(0).values.cpu(), x.min(0).values.cpu()
        self.hi[layer] = torch.maximum(self.hi[layer], hi) if layer in self.hi else hi
        self.lo[layer] = torch.minimum(self.lo[layer], lo) if layer in self.lo else lo

    def result(self) -> dict:
        layers = sorted(self.hi)
        norms = [torch.cat(self.norms[L]) for L in layers]
        q = torch.tensor([0.1, 0.5, 0.9])
        return {"hi": torch.stack([self.hi[L] for L in layers]), "lo": torch.stack([self.lo[L] for L in layers]),
                "norm_q10_50_90": torch.stack([n.quantile(q) if len(n) < 2 ** 24 else n[torch.randperm(len(n))[:2 ** 24]].quantile(q)
                                               for n in norms]),
                "tokens": self.tokens}


@torch.inference_mode()
def collect_stats(model, hooks: Hooks, stats: Stats, seqs: list[list[int]], starts: list[int] | None = None):
    """Run seqs (one batch) through the model, adding their tokens to `stats`: every token but the first, or from
    each start on."""
    ids, mask = pad_left(seqs, 0, model.device)
    T = ids.shape[1]
    sel = torch.zeros_like(mask, dtype=torch.bool)
    for b, s in enumerate(seqs):
        sel[b, T - len(s) + max(1, starts[b] if starts else 1):] = True
    stats.mask, hooks.stats = sel, stats
    try:
        model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
    finally:
        hooks.stats = None
    stats.tokens += int(sel.sum())


def pad_left(seqs: list[list[int]], pad: int, device):
    T = max(map(len, seqs))
    ids = torch.tensor([[pad] * (T - len(s)) + s for s in seqs], device=device)
    mask = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs], device=device)
    return ids, mask


def stops(tok) -> list[int]:
    return sorted({t for t in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id) if t is not None})


_batch_cap = None  # the largest generation batch that fits on this GPU, once a bigger one hasn't


def generate(model, tok, prompts: list[list[int]], max_new: int, sample: bool = False, seed: int = 0,
             steer: tuple | None = None) -> list[list[int]]:
    """New tokens for each prompt, cut at the first stop token. Greedy unless `sample` (T 0.7, top-p 0.9, as the
    Assistant Axis paper's generation). `steer` (hooks, layer, [B, d] vectors) adds each prompt's own vector at the
    layer. A batch too big for the GPU is halved until it fits, and later batches start at the size that did."""
    global _batch_cap
    n = min(len(prompts), _batch_cap or len(prompts))
    if n < len(prompts):
        return [r for s in range(0, len(prompts), n)
                for r in generate(model, tok, prompts[s:s + n], max_new, sample, seed,
                                  steer and (steer[0], steer[1], steer[2][s:s + n]))]
    if steer:
        steer[0].clear()
        steer[0].add(steer[1], steer[2])
    try:
        return _generate(model, tok, prompts, max_new, sample, seed)
    except torch.OutOfMemoryError:
        if n == 1:
            raise
        _batch_cap = n // 2
        print(f"[personakit] a batch of {n} replies doesn't fit on the GPU: generating {n // 2} at a time", flush=True)
    free()  # outside the except block, whose traceback still holds the failed batch's tensors
    return generate(model, tok, prompts, max_new, sample, seed, steer)


@torch.inference_mode()
def _generate(model, tok, prompts: list[list[int]], max_new: int, sample: bool, seed: int) -> list[list[int]]:
    ids, mask = pad_left(prompts, tok.pad_token_id, model.device)
    if sample:
        torch.manual_seed(seed)
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=sample,
                         temperature=0.7 if sample else None, top_p=0.9 if sample else None, top_k=None,
                         pad_token_id=tok.pad_token_id, eos_token_id=stops(tok))
    new, end = out[:, ids.shape[1]:].tolist(), set(stops(tok))
    return [r[:next((j for j, t in enumerate(r) if t in end), len(r))] for r in new]


@torch.inference_mode()
def span_means(model, hooks: Hooks, seqs: list[list[int]], starts: list[int], layers: list[int]) -> torch.Tensor:
    """[B, len(layers), d] float32: each sequence's mean residual leaving each layer, over its tokens from start on
    (the response). Runs under whatever `hooks` are set."""
    ids, mask = pad_left(seqs, 0, model.device)
    T = ids.shape[1]
    span = torch.zeros_like(mask, dtype=torch.float32)
    for b, (s, st) in enumerate(zip(seqs, starts)):
        span[b, T - len(s) + st:] = 1
    hooks.capture = {}
    try:
        model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
        caps = hooks.capture
    finally:
        hooks.capture = None
    w = span / span.sum(1, keepdim=True).clamp(min=1)
    return torch.stack([torch.einsum("bt,btd->bd", w, caps[L][0].float()) for L in layers], 1).cpu()


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


@torch.inference_mode()
def mmlu(model, tok, rows: list[dict], system: str = "", batch: int = 32) -> list[int]:
    """1 per item answered right, 0-shot in chat form: the harness's prompt as the user turn (after `system`, if any),
    the reply opening "Answer:", and the most likely of " A" to " D". Runs under whatever hooks are set."""
    letters = [tok.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"]
    seqs = [chat_ids(tok, f"The following are multiple choice questions (with answers) about "
                          f"{r['subject'].replace('_', ' ')}.\n\n{r['question'].strip()}\n"
                          + "".join(f"{c}. {t}\n" for c, t in zip("ABCD", r["choices"])), system, reply="Answer:")
            for r in rows]

    def picks(chunk):
        ids, mask = pad_left(chunk, tok.pad_token_id, model.device)
        return model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1, letters].argmax(-1).tolist()
    return [int(p == r["answer"]) for p, r in zip(batched(picks, seqs, batch), rows)]


@torch.inference_mode()
def kl_from_base(model, hooks: Hooks, docs: list[list[int]], setups: dict, batch: int = 4) -> dict:
    """Mean next-token KL(unsteered || steered) per token (the first excluded) over docs, for each setup: a name ->
    list of (layer, vector) adds."""
    def sums(chunk):  # [(the chunk's summed KL per setup, its tokens)]
        ids, mask = pad_left(chunk, 0, model.device)
        keep = mask.bool().clone()
        keep[torch.arange(len(ids)), (mask == 0).sum(1)] = False  # each doc's first token
        hooks.clear()
        base = model(input_ids=ids, attention_mask=mask).logits[keep].float().log_softmax(-1)
        tot = {}
        for k, adds in setups.items():
            hooks.clear()
            for L, v in adds:
                hooks.add(L, v)
            lp = model(input_ids=ids, attention_mask=mask).logits[keep].float().log_softmax(-1)
            tot[k] = float((base.exp() * (base - lp)).sum())
            del lp
        return [(tot, int(keep.sum()))]
    parts = batched(sums, docs, batch)
    hooks.clear()
    n = sum(t for _, t in parts)
    return {k: sum(p[k] for p, _ in parts) / n for k in setups}


@torch.inference_mode()
def lens(model, cols: torch.Tensor) -> torch.Tensor:
    """[k, V] float: the logit lens of the [k, d] directions cols (final norm, then the unembedding)."""
    return model.lm_head(model.model.norm(cols.to(model.device, model.dtype))).float()


def ranks(x: torch.Tensor) -> torch.Tensor:
    """Ranks along the last dimension of [k, V], ties given their average rank (the lens runs in bf16, so its logits
    tie often)."""
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        v, idx = x[i].sort()
        _, inv, cnt = torch.unique_consecutive(v, return_inverse=True, return_counts=True)
        first = (cnt.cumsum(0) - cnt).to(x.dtype)
        out[i, idx] = (first + (cnt.to(x.dtype) - 1) / 2)[inv]
    return out


def mmlu_rows(n: int, seed: int = 42) -> list[dict]:
    """n MMLU test items, a seeded sample in order."""
    import random
    import datasets
    from .data import MMLU
    data = datasets.load_dataset(MMLU[0], revision=MMLU[1], name="all", split="test")
    return [data[i] for i in sorted(random.Random(seed).sample(range(len(data)), min(n, len(data))))]
