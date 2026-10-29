# The runner: PersonaKit Studio on your compute

[PersonaKit Studio](https://personakit.denniskoshta.com) lets you steer Qwen3.5 live in the browser. It has a small free
GPU allowance on the 4B. Past that, or for the 9B, `personakit-runner` runs the model on your compute: your own NVIDIA
GPU, or your own Modal account through your `modal` login.

```
# on this machine's GPU (about 10 GB of GPU memory for the 4B, 20 GB for the 9B)
uvx --from "personakit[local] @ git+https://github.com/DennisKoshta/personakit" personakit-runner --local

# on your Modal account (log in once first: uvx modal setup)
uvx --from "personakit[modal] @ git+https://github.com/DennisKoshta/personakit" personakit-runner --modal
```

It needs [uv](https://docs.astral.sh/uv/getting-started/installation/) and git. It prints a pairing code. In Studio,
open **Run on your own compute → Your own GPU**, type the code, and your messages run there from then on. The model
starts with your first message: the first time, its weights download (on Modal, the image builds too, a few minutes).

## How it works

- Studio's server still does the CPU work: it turns your doses, concepts and neuron pushes into steering vectors and
  clamp lists. The page fetches those (sending the server the settings only, not your messages), then sends them and
  your messages to the runner, which streams the replies back to the page. The generation code is [`personakit/live.py`](../personakit/live.py).
- A concept Studio hasn't seen is derived on the runner: ten short answers about it, averaged at the steering layer.
  The page sends that mean back to the server for that one message; the server doesn't keep it.
- With `--modal` the GPU runs in an ephemeral app (`personakit-runner`) on your account, and stops after 10 idle
  minutes (`--idle-minutes`), when the page disconnects, or when you stop the runner. Weights cache in your
  `personakit-hf` volume.

## Safety

- It listens on `127.0.0.1` only (port 8796, `--port` to change it).
- It answers only the Studio page: requests must come from `https://personakit.denniskoshta.com` (their `Origin`) and
  carry the pairing code. After 20 wrong codes it refuses everything until restarted.
- Your Modal keys stay on your machine, and your messages and the replies go only between the page and the runner,
  never through Studio's server.
- It logs no requests.

## Browsers

Chrome, Edge and Firefox can reach a program on your own machine from a web page; Chrome may ask you to allow access
to devices on your network. Safari, and every browser on an iPhone or iPad, can't: use Studio's "Your Modal account"
option there instead.

On Windows, run it in WSL: PyPI's PyTorch for Windows has no CUDA.
