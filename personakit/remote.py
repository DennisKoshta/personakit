"""personakit on Modal: one app whose functions run personakit.core's in a container with a GPU. Weights cache in the
personakit-hf volume (or the one $PERSONAKIT_HF_VOLUME names), atlases in personakit-atlas.
"""
import os
import time

import modal

from . import core as C
from . import data as D
from .core import SIZES, _atlas_path, _ref_dirs  # noqa: F401 (the study's k1_kit.py reads them here)

app = modal.App("personakit")
HF_VOLUME = os.environ.get("PERSONAKIT_HF_VOLUME", "personakit-hf")  # the image carries it into the container
HF = modal.Volume.from_name(HF_VOLUME, create_if_missing=True)
ATLAS = modal.Volume.from_name("personakit-atlas", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").pip_install(*D.PACKAGES)
         .env({"HF_HOME": "/hf", "HF_HUB_DISABLE_TELEMETRY": "1", "PERSONAKIT_HF_VOLUME": HF_VOLUME,
               "PERSONAKIT_ATLAS": "/atlas"})
         .add_local_python_source("personakit"))
FN = dict(image=image, volumes={"/hf": HF, "/atlas": ATLAS}, gpu="A10G", memory=32768, timeout=2 * 3600)


def wait(fc, label: str = "personakit"):
    """A spawned call's result. A connection dropped while waiting raises in .get() and, through .remote(), would
    cancel the call; this waits again instead."""
    import sys
    for attempt in range(40):
        try:
            return fc.get()
        except modal.exception.ConnectionError as e:  # the client's; a job that raised the builtin one has failed
            print(f"[{label}] lost the connection ({e}); waiting again", file=sys.stderr, flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    return fc.get()


def _synced(fn, *args, **kw):
    """fn on the volumes' latest state, keeping the weights and atlases it adds even if it then fails."""
    if modal.is_local():  # research/check.py runs .remote here, in this process
        return fn(*args, **kw)
    ATLAS.reload()
    try:
        return fn(*args, **kw)
    finally:
        HF.commit()
        ATLAS.commit()


@app.function(**FN)
def build(key: str, size: str = "full") -> dict:
    return _synced(C.build, key, size)


@app.function(**FN)
def explain(key: str, layer: int, neuron: int, sign: int, measure: bool = False, concept: str = "",
            size: str = "full") -> dict:
    return _synced(C.explain, key, layer, neuron, sign, measure, concept, size)


@app.function(**FN)
def make(key: str, name: str, about: str, size: str = "full") -> dict:
    return _synced(C.make, key, name, about, size)


@app.function(**FN)
def port(recipe: dict, to: str, size: str = "full") -> dict:
    return _synced(C.port, recipe, to, size)


@app.function(**FN)
def twins(src: str, layer: int, neuron: int, sign: int, dst: str, size: str = "full") -> dict:
    return _synced(C.twins, src, layer, neuron, sign, dst, size)
