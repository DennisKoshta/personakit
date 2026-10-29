"""Separated entrypoint for the 9b confession-circuit study."""
import json
import os
from pathlib import Path
import sys

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "study"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import hf_cache

OUT = modal.Volume.from_name("personakit-out", create_if_missing=True)
image = hf_cache.image("flash-linear-attention==0.6.0").add_local_python_source(
    "persona", "prompts", "confession", "confession_runtime", "confession_judge")
app = modal.App("personakit-confession-9b")


@app.function(image=image, volumes={**hf_cache.volumes, "/out": OUT}, gpu="L40S", memory=32768, timeout=1080)
def run(smoke: bool = False):
    import confession
    root = Path("out") if smoke else Path("/out")
    if not modal.is_local():
        OUT.reload()
    output = root / ("confession-0.8b-smoke" if smoke else "confession-9b")
    result = confession.run("9b", smoke, root, output, commit=None if modal.is_local() else OUT.commit)
    if not modal.is_local():
        OUT.commit()
    return result


@app.local_entrypoint()
def main(smoke: bool = False):
    import time
    if smoke and not os.environ.get("PERSONAKIT_RESEARCH_CHECK"):
        raise RuntimeError("Use python for the CPU smoke gate")
    if os.environ.get("PERSONAKIT_RESEARCH_CHECK"):
        result = run.remote(smoke)
    else:
        fc = run.spawn(smoke)
        for attempt in range(40):
            try:
                result = fc.get()
                break
            except (modal.exception.ConnectionError, ConnectionError, TimeoutError) as e:
                print(f"[confession] lost connection waiting for {fc.object_id} ({e}); retrying", flush=True)
                time.sleep(min(60, 5 * (attempt + 1)))
        else:
            result = fc.get()
    if not smoke:
        out = HERE / "out"
        out.mkdir(exist_ok=True)
        (out / "confession_9b.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("PERSONAKIT_RESULT " + json.dumps({k: v for k, v in result.items() if k != "details"}, allow_nan=False), flush=True)
