"""H1-v2a: fixed known-neuron and layer-matched random-control trait/identity sweep."""
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "study"))
sys.path.insert(0, str(HERE))
import modal
import hf_cache
from jobs import MODELS, OUT, call
import trait_data as D

image = hf_cache.image("flash-linear-attention==0.6.0").add_local_python_source("persona", "prompts", "jobs", "judge", "trait_data")
app = modal.App("personakit-trait-neurons")


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu="A10G", memory=32768, timeout=7200)
def run(model_key: str, smoke: bool):
    import torch
    import persona
    t0 = time.time()
    if not modal.is_local():
        OUT.reload()
    if smoke:
        torch.set_num_threads(8)
    prompt_hash = D.validate_data()
    ax, artifact_hashes = D.load_inputs(model_key, smoke)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = persona.load(hf_cache.model(*MODELS[model_key]), device)
    if len(ax["median_resid"]) != len(persona.blocks(model)):
        raise ValueError("artifact/model layer mismatch")
    with persona.Hooks(model) as hooks:
        recs, known, controls, conditions, doses, control_doses = D.queue_part1_conditions(model, tok, hooks, ax, smoke)
        D.flush_generations(model, tok, hooks, recs, smoke)
    del hooks, model, tok
    persona.free()
    judge_key = "0.8b" if smoke else "9b"
    model, tok = persona.load(hf_cache.model(*MODELS[judge_key]), device)
    labels = D.score_traits(model, tok, recs, tuple(D.RUBRICS), 32 if smoke else 24)
    del model, tok
    persona.free()
    summary, _ = D.finalize_part1(model_key, judge_key, smoke, t0, prompt_hash, artifact_hashes,
                                  known, controls, conditions, doses, control_doses, recs, labels)
    if not modal.is_local():
        OUT.commit()
    return summary


@app.local_entrypoint()
def main(model: str = "4b", smoke: bool = False):
    if not smoke and model != "4b":
        raise ValueError("the preregistered full target is 4b")
    result = call(run, "0.8b" if smoke else model, smoke)
    print("PERSONAKIT_RESULT", json.dumps(result), flush=True)
