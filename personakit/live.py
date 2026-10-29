"""Live chat under interventions, streamed token by token: what PersonaKit Studio's page runs on a GPU, here so the
runner (personakit.runner) can run it on yours. The same code as Studio's own engine.

Interventions per batch row (each chat column is a row):
  adds    [{layer, vec, where}]           vec added to the residual leaving `layer`
  pushes  [{layer, neuron, along, where}] the neuron's activation raised so its output adds `along` (a signed size
                                          in the residual) along its W_down column (Hooks.push)
  clamp   [{layer, neuron, tau}]          the activation held at or below tau, every real token after the first
                                          (the confession-neuron clamp)
`where` is "all" (every position, as Hooks does) or "reply" (only the last position on the prefill and each generated
token: steering the reply only).
"""
import queue
import threading
import time

import torch

from . import data as D
from . import engine as E


def load(key: str):
    """The model, its tokenizer, and what to show about them. The linear-attention kernels compile on first use (40 s
    for the first reply otherwise): it's done here, once batched and once alone, as chat and concept_means run."""
    from . import core as C
    t0 = time.time()
    model, tok = C._load(key)
    load_seconds = round(time.time() - t0, 1)
    t1 = time.time()
    for n in (2, 1):
        E.generate(model, tok, [E.chat_ids(tok, "Hi")] * n, 8)
    gpu = torch.cuda.get_device_name(0) if model.device.type == "cuda" else model.device.type
    return model, tok, {"model": key, "gpu": gpu, "load_seconds": load_seconds,
                        "warm_seconds": round(time.time() - t1, 1), "layers": model.config.num_hidden_layers}


def concept_means(model, tok, about: str):
    """[L, d] float32: the mean residual leaving every layer over the model's neutral answers about `about` (the
    first half of core._concept_dir; the page's server subtracts the reference concepts and the axis)."""
    from .core import SIZES
    sz = SIZES["full"]
    ids = [E.chat_ids(tok, q.format(x=about)) for q in D.CONCEPT_QUESTIONS[:sz["concept_q"]]]
    outs = E.generate(model, tok, ids, sz["max_new"])
    layers = list(range(model.config.num_hidden_layers))
    with E.Hooks(model) as hooks:
        mean = E.span_means(model, hooks, [a + b for a, b in zip(ids, outs)], [len(a) for a in ids], layers)
    return mean.mean(0).numpy()


def chat(model, tok, rows: list[dict], max_new: int = 200, temperature: float = 0.0, seed: int = 0):
    """Each row's reply, streamed: {"row", "text"} pieces as they come (or {"row", "replace"}), then {"done", "texts",
    "tokens", "seconds", "finished"}. Closing the generator early stops the generation."""
    from transformers import StoppingCriteria, StoppingCriteriaList
    from transformers.generation.streamers import BaseStreamer

    sample = temperature > 0
    prompts = [E.chat_ids(tok, r["user"], r.get("system") or "", history=r.get("history") or []) for r in rows]
    B, T = len(rows), max(map(len, prompts))
    ids, mask = E.pad_left(prompts, tok.pad_token_id, model.device)
    first = torch.tensor([T - len(p) for p in prompts], device=model.device)  # each row's first real token
    d = model.config.hidden_size
    stop = set(E.stops(tok))
    q: queue.Queue = queue.Queue()
    cancel = threading.Event()

    class Stream(BaseStreamer):
        def __init__(self):
            self.prompt_seen = False

        def put(self, value):
            if not self.prompt_seen:
                self.prompt_seen = True
                return
            q.put(value.reshape(-1).tolist())

        def end(self):
            q.put(None)

    class Cancelled(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return torch.full((input_ids.shape[0],), cancel.is_set(), dtype=torch.bool, device=input_ids.device)

    hooks = E.Hooks(model)
    extra = []
    th = None
    try:
        hooks.__enter__()
        reply_adds, clamps, reply_pushes = {}, {}, {}
        for b, r in enumerate(rows):
            for a in r.get("adds") or []:
                v = torch.as_tensor(a["vec"], dtype=torch.float32)
                if a.get("where") == "reply":
                    reply_adds.setdefault(a["layer"], torch.zeros(B, d))[b] += v
                else:
                    vb = torch.zeros(B, d)
                    vb[b] = v
                    hooks.add(a["layer"], vb)
            for p in r.get("pushes") or []:
                delta = torch.zeros(B)
                delta[b] = float(p["along"]) / float(E.down_col(model, p["layer"], p["neuron"]).float().norm())
                if p.get("where") == "reply":
                    reply_pushes.setdefault(p["layer"], []).append((p["neuron"], delta))
                else:
                    hooks.push(p["layer"], p["neuron"], delta)
            for c in r.get("clamp") or []:
                clamps.setdefault(c["layer"], []).append((b, c["neuron"], c["tau"]))
        blocks = E.blocks(model)
        for L, vb in reply_adds.items():
            extra.append(blocks[L].register_forward_hook(_reply_add(vb)))
        for L in set(clamps) | set(reply_pushes):
            extra.append(blocks[L].mlp.down_proj.register_forward_pre_hook(
                _mlp_edit(clamps.get(L, []), reply_pushes.get(L, []), first)))

        err = []

        def run():
            try:
                with torch.inference_mode():
                    if sample:
                        torch.manual_seed(seed)
                    model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=sample,
                                   temperature=temperature if sample else None, top_p=0.9 if sample else None,
                                   top_k=None, pad_token_id=tok.pad_token_id, eos_token_id=sorted(stop),
                                   streamer=Stream(), stopping_criteria=StoppingCriteriaList([Cancelled()]))
            except Exception as e:  # noqa: BLE001 (sent to the page)
                err.append(f"{type(e).__name__}: {e}")
                q.put(None)

        t0 = time.time()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        toks = [[] for _ in rows]
        done = [False] * B
        sent = [""] * B
        last = time.time()
        n = 0
        while True:
            item = q.get()
            if item is not None:
                n += 1
                for b, t in enumerate(item):
                    if done[b]:
                        continue
                    if t in stop:
                        done[b] = True
                    else:
                        toks[b].append(t)
            if item is None or time.time() - last > 0.06:
                last = time.time()
                for b in range(B):
                    text = tok.decode(toks[b], skip_special_tokens=True)
                    if text.endswith("�") and item is not None:
                        continue  # half a character: wait for the rest
                    if len(text) > len(sent[b]) and text.startswith(sent[b]):
                        yield {"row": b, "text": text[len(sent[b]):]}
                        sent[b] = text
                    elif text != sent[b]:
                        yield {"row": b, "replace": text}
                        sent[b] = text
            if item is None:
                break
        th.join()
        if err:
            yield {"error": err[0]}
        yield {"done": True, "texts": sent, "tokens": n, "seconds": round(time.time() - t0, 2),
               "finished": [bool(x) for x in done]}
    finally:
        if th is not None and th.is_alive():  # closed early: stop generating before the hooks come off
            cancel.set()
            th.join(timeout=30)
        for h in extra:
            h.remove()
        hooks.__exit__()


def _reply_add(vb):
    """Add each row's vector at the last position only."""
    def hook(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, -1, :] += vb.to(h.device, h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h
    return hook


def _mlp_edit(clamps, pushes, first):
    """On the activations entering down_proj: clamp rows' neurons at or below tau from each row's second real token on,
    and add reply-only pushes at the last position."""
    def hook(mod, args):
        a = args[0].clone()
        Bt, T, _ = a.shape
        if clamps:
            if T > 1:
                pos = torch.arange(T, device=a.device)[None, :] > first[:, None]       # [B, T]
            else:
                pos = torch.ones(Bt, 1, dtype=torch.bool, device=a.device)
            for b, n, tau in clamps:
                sel = a[b, :, n]
                a[b, :, n] = torch.where(pos[b], torch.minimum(sel, torch.tensor(tau, dtype=a.dtype, device=a.device)), sel)
        for n, delta in pushes:
            a[:, -1, n] += delta.to(a.device, a.dtype)
        return (a, *args[1:])
    return hook
