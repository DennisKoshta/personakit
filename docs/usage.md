# Install and use

## Install

On your own machine, with an NVIDIA GPU:

```
pip install "personakit[local] @ git+https://github.com/DennisKoshta/personakit"
pip install flash-linear-attention   # optional: fast kernels for Qwen3.5's linear-attention layers
```

Then pass `--local` to every command. Weights go in your Hugging Face cache, and atlases in `~/.cache/personakit/atlas` (or `$PERSONAKIT_ATLAS`). What each size needs is in [hardware](hardware.md#on-your-own-gpu).

On Modal instead:

```
pip install "personakit[modal] @ git+https://github.com/DennisKoshta/personakit"
modal setup        # once, if you haven't used Modal before
```

There, weights go in the Modal volume `personakit-hf`, and atlases in `personakit-atlas`.

The first command on a model downloads its weights, and every judged command needs the 9B judge's (about 19 GB). It also builds that model's **atlas**: the Assistant Axis, 40 role directions, 10 reference concepts and the dose unit. Later commands reuse both. `personakit build --model 4b` builds an atlas ahead of time.

`--toy` runs any command on your own machine at toy sizes, on the 0.8B, even on a CPU. It checks that the kit works and means nothing else.

## Commands

- **`explain <layer> <neuron>`**: what one MLP neuron points at.
  - Its logit lens: the tokens its write direction promotes and suppresses.
  - Its cosine with the model's Assistant Axis, and that cosine's percentile in its layer.
  - Its cosine with ten concept directions, and the nearest of 40 roles.
  - Its natural range on the Pile and in chat, and how big a push its own largest activation is.
  - With `--measure`, it pushes the neuron at doses 0.25 to 1.5 and judges the replies to 10 identity questions: identity break, coherence and, given `--concept`, whether it speaks as that concept.
- **`make "<persona>"`**: a recipe that makes the model speak as that persona. The recipe is a concept direction plus a push off the Assistant Axis, at the middle layer.
  - It tries a 3 × 3 grid of doses on 10 identity questions and 10 everyday requests.
  - It keeps the cell with the highest persona rate that stays coherent (≥ 80%) on both.
  - It then tests that cell on 20 held-out identity questions and 10 held-out everyday requests, and measures its cost: MMLU (200 items, paired) and KL from the unsteered model on Pile text.
- **`port <recipe.json> --to <model>`**: re-derives a recipe in another size. It reads the concept direction and the axis in that model and re-picks the doses on a small grid around the source's.
- **`twins <layer> <neuron> --to <model>`**: the neurons in another size whose write directions have the most similar direct effect on the shared vocabulary (Spearman over the logit lens).
- **`personakit.steer(model, recipe)`**: applies a recipe to a transformers model in your own code.

## Examples

```
personakit --local explain 22 2742 --sign + --model 4b --measure --concept "an alien"
personakit --local make "a wizard" --about wizards --model 4b -o wizard.json
personakit --local port wizard.json --to 9b -o wizard-9b.json
personakit --local twins 22 2742 --sign + --from 4b --to 9b
```

Without `--local`, the same commands run on Modal.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import personakit

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16).to("cuda")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
enc = tok.apply_chat_template([{"role": "user", "content": "Who are you?"}], add_generation_prompt=True,
                              enable_thinking=False, return_dict=True, return_tensors="pt").to("cuda")
with personakit.steer(model, "wizard.json"):       # scale=0.5 for half the dose
    out = model.generate(**enc, max_new_tokens=80)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

A recipe belongs to the model size it was made on. Use `port` to move it to another.

**Doses** are in the study's unit. A dose of d adds d · m_L · (unit direction) to the residual leaving layer L. m_L is the median norm of that residual on the Pile.
