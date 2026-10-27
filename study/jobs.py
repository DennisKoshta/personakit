"""What this question's Modal jobs share: model pins, the output volume, the image, and the Assistant Axis data."""
import json
import os
import pathlib
import sys
import time
import urllib.request

import modal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # research/
import hf_cache  # noqa: E402

MODELS = {"4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
          "2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
          "9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
          "4b-base": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
          "0.8b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")}
OUT = modal.Volume.from_name("crucible-persona-neurons", create_if_missing=True)
# flash-linear-attention: GPU kernels for the Gated DeltaNet layers (transformers falls back to slow PyTorch without).
image = hf_cache.image("flash-linear-attention==0.6.0").add_local_python_source("persona", "prompts", "jobs", "judge")


def fetch(path: str):
    """A file from the Assistant Axis repo's data/ at the pinned commit (JSON, or JSONL as a list)."""
    import persona
    with urllib.request.urlopen(f"{persona.AXIS_DATA}/{path}", timeout=60) as r:
        body = r.read().decode()
    return [json.loads(line) for line in body.splitlines() if line.strip()] if path.endswith(".jsonl") else json.loads(body)


def out_dir(tag: str) -> pathlib.Path:
    """This run's folder: on the volume in a container, under ~/workspace/persona-neurons in a local check."""
    root = pathlib.Path.home() / "workspace" / "persona-neurons" if modal.is_local() else pathlib.Path("/out")
    (root / tag).mkdir(parents=True, exist_ok=True)
    return root / tag


def pick_dose(sweep: dict, rate_key: str) -> tuple[float, bool]:
    """The smallest dose whose rate reaches 0.5 at coherence >= 0.8; else the coherent dose with the highest rate (and
    False: it never reached 0.5)."""
    ok = sorted((float(d), v) for d, v in sweep.items() if v["coherent"] >= 0.8)
    for d, v in ok:
        if v[rate_key] >= 0.5:
            return d, True
    return (max(ok, key=lambda x: x[1][rate_key])[0] if ok else min(float(d) for d in sweep)), False


def sized(fn, model_key: str):
    """fn on a card the model fits: the 9B's weights (19 GB) leave a 24 GB card too little room to generate in batches."""
    return fn.with_options(gpu="L40S") if model_key == "9b" else fn


def call(fn, *args):
    """fn.remote(*args), riding out a dropped connection. A .remote() call whose wait hits a connection error raises,
    and the run's client then cancels the job: H5, X3 and the 9B's S1a were all lost that way at once on 2026-10-27.
    Spawned, the call runs on its own in the detached app, and the wait for its result retries."""
    if os.environ.get("CRUCIBLE_RESEARCH_CHECK"):  # research/check.py runs .remote in this process
        return fn.remote(*args)
    fc = fn.spawn(*args)
    for attempt in range(40):
        try:
            return fc.get()
        except modal.exception.ConnectionError as e:  # the client's; a job that raised the builtin one has failed
            print(f"[jobs] lost the connection waiting for {fc.object_id} ({e}); waiting again", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    return fc.get()
