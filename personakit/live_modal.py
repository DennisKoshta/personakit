"""personakit.live on Modal, for the runner's --modal mode: one class per model, each on the cheapest cards it fits
(4B: 9 GB of weights; 9B: 19 GB), run in an ephemeral app on your account. Weights cache in the same volume as the
rest of the kit (personakit-hf).
"""
import modal

from . import remote

app = modal.App("personakit-runner")
FN = dict(image=remote.image, volumes={"/hf": remote.HF}, cpu=4, timeout=30 * 60, scaledown_window=10 * 60,
          max_containers=1)


class Engine:
    key = ""

    @modal.enter()
    def load(self):
        from . import live
        self.model, self.tok, self.about = live.load(self.key)
        remote.HF.commit()  # the weights, fetched from Hugging Face the first time

    @modal.method()
    def info(self) -> dict:
        return self.about

    @modal.method()
    def concept_means(self, about: str):
        from . import live
        return live.concept_means(self.model, self.tok, about).tolist()  # no numpy needed where the runner is

    @modal.method()
    def chat(self, rows: list[dict], max_new: int = 200, temperature: float = 0.0, seed: int = 0):
        from . import live
        yield from live.chat(self.model, self.tok, rows, max_new, temperature, seed)


@app.cls(gpu=["L4", "A10G"], memory=16384, **FN)
class Engine4b(Engine):
    key = "4b"


@app.cls(gpu=["L40S", "A100-40GB"], memory=32768, **FN)
class Engine9b(Engine):
    key = "9b"


ENGINES = {"4b": Engine4b, "9b": Engine9b}
