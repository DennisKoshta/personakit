"""Stage K2: personakit's --local backend on a GPU, as on someone's own machine.

The kit is pip-installed with its `local` extra into a plain image, and its command line runs with --local in one
A10G container (24 GB, like an RTX 3090 or 4090), one command at a time. $PERSONAKIT_GPU_GB holds some commands to
16 or 8 GB, as on a smaller card. Each command's wall time and peak GPU memory (nvidia-smi, every second) are recorded.

1. 16 GB: build the 4B's atlas, then make "a wizard" (the 9B judge spills into CPU memory).
2. 24 GB: make "a wizard" again (the same replies and labels as step 1 should follow), explain the alien neuron with
   --measure, port the wizard to the 9B (its atlas included), and the alien neuron's twins in the 9B.
3. 8 GB: make "a wizard" on the 2B (its atlas included).

A second container runs the Modal backend once (explain on the 0.8B, weights only), as a check of personakit.remote.
Weights come from the study's cache volume (crucible-hf-cache); atlases are built fresh in the container. Results go
to a local output folder (this entrypoint runs locally) and the CRUCIBLE_RESULT lines. Run through modal_run
with containers 2, experiment "persona-neurons", config {"stage": "k2"}.
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
from personakit.data import PACKAGES  # noqa: E402

app = R.app
user_image = (modal.Image.debian_slim(python_version="3.12").pip_install(*PACKAGES)
              .env({"HF_HOME": "/hf", "HF_HUB_DISABLE_TELEMETRY": "1"})
              .add_local_dir(HERE / "personakit", "/kit", copy=True, ignore=["**/__pycache__"])
              .run_commands("pip install '/kit[local]'"))

STEPS = [  # (name, GPU GB or None for the whole card, arguments after `personakit --local`)
    ("build-4b@16", 16, ["build", "--model", "4b"]),
    ("make-4b@16", 16, ["make", "a wizard", "--about", "wizards", "--model", "4b"]),
    ("make-4b@24", None, ["make", "a wizard", "--about", "wizards", "--model", "4b"]),
    ("explain-4b@24", None, ["explain", "22", "2742", "--sign", "+", "--model", "4b", "--measure", "--concept", "an alien"]),
    ("port-9b@24", None, ["port", "/tmp/make-4b@24.json", "--to", "9b"]),
    ("twins-9b@24", None, ["twins", "22", "2742", "--sign", "+", "--from", "4b", "--to", "9b"]),
    ("make-2b@8", 8, ["make", "a wizard", "--about", "wizards", "--model", "2b"]),
]


@app.function(image=user_image, volumes={"/hf": R.HF}, gpu="A10G", memory=65536, timeout=3 * 3600)
def local_steps(steps: list) -> list:
    import subprocess
    import threading
    import time
    out = []
    for name, gb, argv in steps:
        env = {**os.environ, **({"PERSONAKIT_GPU_GB": str(gb)} if gb else {})}
        peak, done = [0], threading.Event()

        def watch():
            while not done.wait(1.0):
                q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True)
                peak[0] = max(peak[0], int(q.stdout.split()[0]))
        threading.Thread(target=watch, daemon=True).start()
        t0 = time.time()
        print(f"[k2] {name}: personakit --local {' '.join(argv)}", flush=True)
        p = subprocess.run(["personakit", "--local", *argv, "-o", f"/tmp/{name}.json"], env=env, capture_output=True,
                           text=True)
        done.set()
        minutes = round((time.time() - t0) / 60, 1)
        print(p.stdout[-3000:], flush=True)
        if p.returncode:
            print(p.stderr[-3000:], flush=True)
        out.append({"step": name, "gpu_gb": gb, "ok": p.returncode == 0, "minutes": minutes,
                    "peak_gpu_gb": round(peak[0] / 1024, 1),
                    "result": json.load(open(f"/tmp/{name}.json")) if p.returncode == 0 else p.stderr[-1500:]})
        print(f"[k2] {name}: {'ok' if p.returncode == 0 else 'FAILED'} in {minutes} min, "
              f"peak GPU {out[-1]['peak_gpu_gb']} GB", flush=True)
    return out


@app.local_entrypoint()
def main(only: str = ""):
    """--only make-4b@24,port-9b@24 reruns those steps alone (in order), without the Modal backend's check."""
    names = [n.strip() for n in only.split(",") if n.strip()]
    if set(names) - {s[0] for s in STEPS}:
        raise SystemExit(f"--only: no step named {sorted(set(names) - {s[0] for s in STEPS})}")
    remote = None if names else R.explain.with_options(gpu="A10G").spawn("0.8b", 10, 7, -1)
    steps = R.wait(local_steps.spawn([s for s in STEPS if not names or s[0] in names]), "k2")
    remote = remote and R.wait(remote, "k2")

    d = pathlib.Path.home() / "workspace" / "persona-neurons" / "k2"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ("steps-" + only.replace(",", "+") + ".json" if only else "steps.json"), "w") as f:
        json.dump(steps, f, indent=1)
    if remote:
        with open(d / "remote-explain.json", "w") as f:
            json.dump(remote, f, indent=1)

    def summary(s):
        r = s["result"]
        if not s["ok"]:
            return {"ok": False}
        if s["step"].startswith(("make", "port")):
            return {"made": r["made"], **({"dc": r["dc"], "da": r["da"], "persona": r["test"]["steered"]["persona"],
                                           "coherent": r["test"]["steered"]["coherent"],
                                           "everyday_coherent": r["test"]["everyday_steered"]["coherent"],
                                           "mmlu": [r["cost"]["mmlu_unsteered"], r["cost"]["mmlu_steered"]],
                                           "kl": r["cost"]["kl_per_token"]} if r["made"] else {})}
        if s["step"].startswith("explain"):
            return {"top_concept": r["concepts"][0]["concept"], "cos_away": r["axis"]["cos_away"],
                    "measured": [(m["dose"], m.get("persona"), m.get("coherent")) for m in r["measured"]]}
        if s["step"].startswith("twins"):
            return {"twins": [(t["layer"], t["neuron"], t["sign"], t["spearman"]) for t in r["twins"]]}
        return r

    res = {s["step"]: {"minutes": s["minutes"], "peak_gpu_gb": s["peak_gpu_gb"], **summary(s)} for s in steps}
    by = {s["step"]: s for s in steps}
    a, b = by.get("make-4b@16"), by.get("make-4b@24")
    same = bool(a and b and a["ok"] and b["ok"] and a["result"]["grid"] == b["result"]["grid"]
                and a["result"].get("test") == b["result"].get("test"))
    print("CRUCIBLE_RESULT", json.dumps({"all_ok": all(s["ok"] for s in steps), "offload_same_labels": same,
                                         "remote_explain_ok": bool(remote and remote["lens_top"]), "only": only,
                                         **{f"{k}_minutes": v["minutes"] for k, v in res.items()},
                                         **{f"{k}_peak_gb": v["peak_gpu_gb"] for k, v in res.items()}}))
    print("CRUCIBLE_RESULT", json.dumps({"details": res}))
