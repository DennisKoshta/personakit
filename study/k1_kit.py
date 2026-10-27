"""Stage K1: personakit's own checks, on Modal through the kit's own functions (README.md, "Deliverable").

1. The 4B's atlas from the kit's lighter recipe (40 roles x 10 questions, no judge filter, 1,000 Pile passages) against
   the study's (S1a, S1b): each layer's cosine between the two Assistant Axes, the ratio of the dose units, and the
   cosine of each reference concept's direction at the middle layer.
2. explain on the alien neuron (L22 N2742+), measured, with the concept "an alien": does it find what S1b and H4 did?
3. make "a wizard" (a concept the study never used) on the 4B, then port it to the 2B and the 9B.
4. twins of the alien neuron in the 9B.

The kit reads weights from the study's cache volume (crucible-hf-cache) and keeps its atlases in personakit-atlas.
Results go to ~/workspace/persona-neurons/k1/ (this entrypoint runs locally) and the CRUCIBLE_RESULT lines. Run
through modal_run with gpu L40S (the 9B steps run on one) and containers 2, experiment "persona-neurons", config
{"stage": "k1"}. Smoke: `python research/check.py research/2026-10-26-persona-neurons/k1_kit.py --smoke` (the 0.8B,
toy sizes, no comparison).
"""
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("PERSONAKIT_HF_VOLUME", "crucible-hf-cache")  # the weights this study already cached
sys.path.insert(0, str(HERE / "personakit"))

import modal  # noqa: E402
from personakit import remote as R  # noqa: E402
from personakit.data import gpu_for  # noqa: E402

app = R.app
OUT = modal.Volume.from_name("crucible-persona-neurons")


@app.function(image=R.image, volumes={"/atlas": R.ATLAS, "/out": OUT}, timeout=600)
def compare(key: str, s1a: str, s1b: str) -> dict:
    """The kit's atlas against the study's S1a/S1b outputs for the same model."""
    import torch
    R.ATLAS.reload()
    OUT.reload()
    kit = torch.load(R._atlas_path(key))
    ax = torch.load(f"/out/{s1b}/axis.pt")
    m_study = torch.load(f"/out/{s1a}/calib.pt")["pile"]["norm_q10_50_90"][:, 1]
    mid = kit["layers"] // 2
    cos_axes = [round(float(x), 4) for x in (kit["unit"] * ax["unit"]).sum(-1)]
    kit_dirs = R._ref_dirs(kit, mid)
    cos_concepts = {c: round(float(kit_dirs[i] @ (ax["concept_dirs"][c][mid] / ax["concept_dirs"][c][mid].norm())), 4)
                    for i, c in enumerate(kit["concepts"])}
    ratio = kit["m"] / m_study
    return {"cos_axes": cos_axes, "cos_axes_mid": cos_axes[mid], "cos_axes_min": min(cos_axes),
            "dose_unit_ratio_range": [round(float(ratio.min()), 4), round(float(ratio.max()), 4)],
            "cos_concepts_mid": cos_concepts, "kit_cos_pc1_mid": kit["cos_pc1_mid"]}


class Done:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


@app.local_entrypoint()
def main(smoke: bool = False):
    size = "toy" if smoke else "full"

    def start(fn, key, *args, **kw):
        f = getattr(R, fn)
        if smoke:  # research/check.py runs .remote here, in this process
            return Done(f.remote(*args, size=size, **kw))
        return f.with_options(gpu=gpu_for(key)).spawn(*args, size=size, **kw)

    big, small, alien = ("0.8b", "0.8b", (10, 7, -1)) if smoke else ("9b", "2b", (22, 2742, 1))
    base = "0.8b" if smoke else "4b"
    tw = start("twins", big, base, *alien, big)
    built = R.wait(start("build", base, base), "k1")
    cmp = None if smoke else R.wait(compare.spawn("4b", "s1a-4b", "s1b-4b"), "k1")
    ex = start("explain", base, base, *alien, measure=True, concept="an alien")
    wiz = R.wait(start("make", base, base, "a wizard", "wizards"), "k1")
    ports = {k: start("port", k, wiz, k) for k in ([small] if smoke else [small, big])} if wiz["made"] else {}
    ex, tw = R.wait(ex, "k1"), R.wait(tw, "k1")  # a dropped connection would cancel a bare .get()'s call
    ports = {k: R.wait(v, "k1") for k, v in ports.items()}

    out = pathlib.Path.home() / "workspace" / "persona-neurons" / f"k1{'-smoke' if smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("build", built), ("compare", cmp), ("explain", ex), ("wizard", wiz), ("twins", tw),
                      *((f"wizard-{k}", v) for k, v in ports.items())):
        with open(out / f"{name}.json", "w") as f:
            json.dump(obj, f, indent=1)

    def summary(r):
        if not r or not r.get("made"):
            return {"made": False}
        return {"dc": r["dc"], "da": r["da"], "persona": r["test"]["steered"].get("persona"),
                "coherent": r["test"]["steered"].get("coherent"),
                "everyday_coherent": r["test"]["everyday_steered"].get("coherent"),
                "mmlu": [r["cost"]["mmlu_unsteered"], r["cost"]["mmlu_steered"]], "kl": r["cost"]["kl_per_token"],
                **({"at_source_doses": r["at_source_doses"]} if "at_source_doses" in r else {})}

    res = {"atlas_cos_axes_mid": cmp and cmp["cos_axes_mid"], "atlas_cos_axes_min": cmp and cmp["cos_axes_min"],
           "explain_top_concept": ex["concepts"][0]["concept"], "explain_cos_away": ex["axis"]["cos_away"],
           "explain_measured": [{k: m.get(k) for k in ("dose", "persona", "identity_break", "coherent")}
                                for m in ex.get("measured", [])],
           "wizard": summary(wiz), **{f"wizard_{k}": summary(v) for k, v in ports.items()},
           "twins": [(t["layer"], t["neuron"], t["sign"], t["spearman"]) for t in tw["twins"]]}
    print("CRUCIBLE_RESULT", json.dumps({**{k: v for k, v in res.items() if not isinstance(v, (list, dict))}, "smoke": smoke}))
    print("CRUCIBLE_RESULT", json.dumps({"details": {**res, "compare": cmp, "build": built}}))
