"""Steering, capture and generation for Qwen3.5 text models (any size), shared by this question's jobs and the core of
its tool. README.md, "Design", says what each piece is for.

Interventions all sit at a decoder layer's output, the residual stream leaving layer L (0-indexed), except `push`,
which adds to one MLP neuron's activation silu(gate)*up at the input of down_proj, as in Lyra's neuron-explorer
(explorer/runtime.py at e566e83). The two are the same edit: pushing neuron n by delta adds delta * W_down[:, n] to the
layer's output. Every intervention applies at every position, prompt and generated tokens alike.
"""
import gc

import torch

AXIS_DATA = "https://raw.githubusercontent.com/safety-research/assistant-axis/a98961956072224eaf244eb289d6c01700b63795/data"
DEFAULT_SYSTEMS = ["", "You are an AI assistant.", "You are a large language model.", "You are Qwen.",
                   "Respond as yourself."]  # the paper's data/roles/instructions/default.json, model_name = Qwen


def load(path: str, device: str):
    """The text model (no vision tower) in bf16 on a GPU, fp32 on a CPU, and its tokenizer (left padding)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    return model, tok


def free():
    """Return GPU memory the caller has already let go of (del its names first)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


@torch.inference_mode()
def generate(model, tok, prompts: list[list[int]], max_new: int, sample: bool = False, seed: int = 0) -> list[list[int]]:
    """New tokens for each prompt (one batch), cut at the first stop token. Greedy unless `sample` (T 0.7, top-p 0.9,
    as the Assistant Axis paper's generation)."""
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
def token_projections(model, hooks: Hooks, seqs: list[list[int]], starts: list[int], dirs: dict) -> dict:
    """{layer: [tokens, k]}: each response token's residual leaving the layer, projected on dirs[layer] ([k, d] unit
    rows), over the sequences' tokens from start on. One batch."""
    ids, mask = pad_left(seqs, 0, model.device)
    T = ids.shape[1]
    hooks.capture = {}
    try:
        model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
        caps = hooks.capture
    finally:
        hooks.capture = None
    out = {}
    for L, D in dirs.items():
        h = torch.cat([caps[L][0][b, T - len(s) + st:] for b, (s, st) in enumerate(zip(seqs, starts))]).float()
        out[L] = (h @ D.to(h.device).T).cpu()
    return out


def mmlu_rows(n: int = 300, seed: int = 42) -> list[dict]:
    """The MMLU test items this question scores: research/evals.py's pin and seeded subset."""
    import evals
    import hf_cache
    data = hf_cache.dataset(*evals.MMLU, name="all", split="test")
    return [data[i] for i in evals.pick(n, len(data), seed)]


@torch.inference_mode()
def mmlu(model, tok, rows: list[dict], system: str = "", batch: int = 32) -> list[int]:
    """1 per item answered right, 0-shot in chat form: the harness's prompt as the user turn (after `system`, if any),
    the reply opening "Answer:", and the most likely of " A" to " D". Runs under whatever hooks are set."""
    letters = [tok.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"]
    seqs = [chat_ids(tok, f"The following are multiple choice questions (with answers) about "
                          f"{r['subject'].replace('_', ' ')}.\n\n{r['question'].strip()}\n"
                          + "".join(f"{c}. {t}\n" for c, t in zip("ABCD", r["choices"])), system, reply="Answer:")
            for r in rows]
    right = []
    for s in range(0, len(seqs), batch):
        ids, mask = pad_left(seqs[s:s + batch], tok.pad_token_id, model.device)
        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1, letters]
        right += [int(int(p) == r["answer"]) for p, r in zip(logits.argmax(-1).tolist(), rows[s:s + batch])]
    return right


@torch.inference_mode()
def kl_from_base(model, hooks: Hooks, docs: list[list[int]], setups: dict, batch: int = 4) -> dict:
    """Mean next-token KL(unsteered || steered) per token (the first excluded) over docs, for each setup: a name ->
    list of (layer, vector) adds."""
    tot, n = {k: 0.0 for k in setups}, 0
    for s in range(0, len(docs), batch):
        ids, mask = pad_left(docs[s:s + batch], 0, model.device)
        keep = mask.bool().clone()
        keep[torch.arange(len(ids)), (mask == 0).sum(1)] = False  # each doc's first token
        hooks.clear()
        base = model(input_ids=ids, attention_mask=mask).logits[keep].float().log_softmax(-1)
        for k, adds in setups.items():
            hooks.clear()
            for L, v in adds:
                hooks.add(L, v)
            lp = model(input_ids=ids, attention_mask=mask).logits[keep].float().log_softmax(-1)
            tot[k] += float((base.exp() * (base - lp)).sum())
            del lp
        n += int(keep.sum())
        del base
    hooks.clear()
    return {k: v / n for k, v in tot.items()}
