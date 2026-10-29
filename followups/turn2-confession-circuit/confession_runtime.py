"""Study-local adapters for the original persona harness; no remote entrypoints."""
from contextlib import contextmanager

import torch
import persona as original

load = original.load
free = original.free
chat_ids = original.chat_ids
pad_left = original.pad_left
wilson = original.wilson
mmlu_rows = original.mmlu_rows
_current_pad_mask = None
_batch_pad_mask = None


@contextmanager
def track_padding(model):
    """Observe each actual forward, including when transformers omits an all-ones attention_mask."""
    global _current_pad_mask, _batch_pad_mask
    previous, previous_batch = _current_pad_mask, _batch_pad_mask

    def track(mod, args, kwargs):
        global _current_pad_mask
        attn = kwargs.get("attention_mask")
        if attn is not None:
            _current_pad_mask = attn
            return
        ids = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if args else None)
        if _batch_pad_mask is not None and ids is not None and ids.shape[0] == _batch_pad_mask.shape[0]:
            if ids.shape[1] == _batch_pad_mask.shape[1]:
                _current_pad_mask = _batch_pad_mask
            else:
                _current_pad_mask = torch.cat(
                    [_batch_pad_mask, torch.ones((ids.shape[0], ids.shape[1]), dtype=_batch_pad_mask.dtype, device=_batch_pad_mask.device)],
                    dim=-1,
                )
        elif ids is not None:
            _current_pad_mask = torch.ones((ids.shape[0], max(2, ids.shape[1])), dtype=torch.long, device=ids.device)

    handle = model.register_forward_pre_hook(track, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
        _current_pad_mask, _batch_pad_mask = previous, previous_batch


def non_sink_mask(batch, length, device):
    mask = _current_pad_mask
    if mask is None:
        raise RuntimeError("Interventions require the actual forward attention mask")
    real = mask.to(device).bool()
    # Cumsum is computed over the whole context before slicing cached decode tokens.
    return (real & (real.long().cumsum(-1) > 1))[:, -length:, None]


@torch.inference_mode()
def generate(model, tok, seqs, max_new_tokens=80, batch_size=24):
    global _batch_pad_mask
    out = []
    with track_padding(model):
        for start in range(0, len(seqs), batch_size):
            chunk = seqs[start:start + batch_size]
            _, _batch_pad_mask = pad_left(chunk, tok.pad_token_id, model.device)
            out.extend(original.generate(model, tok, chunk, max_new_tokens))
    return out


@torch.inference_mode()
def mmlu_acc(model, tok, rows, system="", batch_size=24):
    global _batch_pad_mask
    letters = [tok.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"]
    seqs = [chat_ids(tok, f"The following are multiple choice questions (with answers) about "
                          f"{r['subject'].replace('_', ' ')}.\n\n{r['question'].strip()}\n"
                          + "".join(f"{c}. {t}\n" for c, t in zip("ABCD", r["choices"])), system, reply="Answer:")
            for r in rows]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    right = [0] * len(rows)
    with track_padding(model):
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            ids, mask = pad_left([seqs[i] for i in idx], tok.pad_token_id, model.device)
            _batch_pad_mask = mask
            logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1, letters]
            for i, p in zip(idx, logits.argmax(-1).tolist()):
                right[i] = int(int(p) == rows[i]["answer"])
    return right


@torch.inference_mode()
def pile_kl(model, tok, docs, setups, batch_size=8):
    """Token-weighted KL, retaining passage sums/counts for paired resampling.

    Compute the vocabulary projection in 16-position chunks: full passage logits
    would exceed the CPU smoke memory reservation and crowd a target's weights.
    Each setup is a context factory; neither is the exact reference distribution.
    """
    result = {name: [] for name in setups}
    with track_padding(model):
        for start in range(0, len(docs), batch_size):
            chunk = docs[start:start + batch_size]
            ids, mask = pad_left(chunk, tok.pad_token_id, model.device)
            keep = mask.bool() & (mask.cumsum(-1) > 1)
            owners = torch.arange(len(chunk), device=model.device)[:, None].expand_as(mask)[keep]
            # The base decoder, rather than causal LM wrapper, avoids allocating all logits.
            base = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[keep]
            for name, setup in setups.items():
                totals = torch.zeros(len(chunk), device=model.device)
                if name != "neither":
                    # model.model bypasses the wrapper's tracking hook.
                    global _current_pad_mask
                    _current_pad_mask = mask
                    with setup():
                        steered = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[keep]
                    for off in range(0, len(base), 16):
                        b = model.lm_head(base[off:off + 16]).float().log_softmax(-1)
                        s = model.lm_head(steered[off:off + 16]).float().log_softmax(-1)
                        kl = (b.exp() * (b - s)).sum(-1)
                        totals.scatter_add_(0, owners[off:off + 16], kl)
                    del steered, b, s
                result[name].extend({"sum": float(totals[i]), "tokens": int(keep[i].sum())}
                                    for i in range(len(chunk)))
            print(f"[confession] Pile {min(start + batch_size, len(docs))}/{len(docs)}", flush=True)
    return result
