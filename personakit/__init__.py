"""personakit: explain a Qwen3.5 neuron, make a persona recipe and port it to another size. The command line runs the
GPU work on this machine (--local) or on Modal; `steer` applies a recipe to a model in your own code."""
import contextlib
import json


@contextlib.contextmanager
def steer(model, recipe, scale: float = 1.0):
    """Inside this block, `model` (a transformers Qwen3.5 of the recipe's size) carries the recipe: its vector added
    to the residual leaving its layer, at every position. `recipe` is a recipe dict or the path of its JSON file."""
    import torch
    r = json.load(open(recipe)) if isinstance(recipe, str) else recipe
    v = torch.tensor(r["vector"]) * scale

    def hook(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + v.to(h.device, h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    handle = model.model.layers[r["layer"]].register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()
