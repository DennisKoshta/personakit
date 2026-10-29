# personakit

Explain a Qwen3.5 neuron, make a persona steering recipe from any concept, and port it to another size. The GPU work runs on your own GPU, or on [Modal](https://modal.com) on your account.

**Status: 0.2, research code.** It covers the Qwen3.5 dense chat models (0.8B, 2B, 4B, 9B) and the 4B's Base twin; the [cross-family study](followups/cross-family-transfer/README.md) also ran it on Qwen2.5-3B and Llama-3.2-3B.

## Start here

Inside a chat model, a few single neurons make it claim to be something else when pushed harder than they ever fire on their own: an alien, a villain, God. We call them **persona neurons**. They matter because they show how small and how fragile "who the model is" can be, and because what they do can be rebuilt from a concept alone: that gives you a way to steer a character in, and to keep it there. personakit is that toolkit, with the [study](study/README.md) that tested it and [twelve follow-up studies](followups/README.md) that probed its limits, failures included.

### Latest results

All on Qwen3.5-4B unless a size is named. Details and caveats are in each study's write-up.

- **A made-up concept finds the neuron, and closes the KL gap** ([`h3-interaction-kl`](followups/h3-interaction-kl/README.md)).
  - A made-up "aliens" direction, projected onto layer 22's `W_down` columns (what each neuron writes), retrieves the alien neuron L22 N2742+, the one the original study found by hand, as the #1 column. Its cosine is 0.387, about 2× the runner-up.
  - Steering with it gives the alien persona on 55% of identity questions (CI 0.55) with identity coherence 1.00. That closes the 6× gap in how far a recipe shifts ordinary text: KL 0.199 nats per token on the Pile, against 1.224 for the dense recipe. Dose regime (matched or high) explains 81% of the KL variance.
  - Pushing off the Assistant Axis while steering a concept cuts the concept dose needed by about 40% (40.6% in the 4B, 37.4% in the 9B, but 22.8% in the 2B, short of the study's 35% rule, so that part is only a *partial* result). The concept × axis interaction is positive at 2B, 4B and 9B, with 0 of 2,000 nonpositive bootstrap draws, below saturation; at high doses it turns negative.
  - Scope: this is one concept. For five other nouns (cookies, dragons, the ocean, knights, trees) no `W_down` column stands out (|cos| ≤ 0.22), and the sparse recipe loses the persona (CI 0.01).
- **Clamping a few "AI-confession" neurons keeps a persona under challenge** ([`turn2-confession-circuit`](followups/turn2-confession-circuit/README.md)). Capping 64 to 128 neurons that fire when the model says it is an AI, with no per-concept vector, kept 5 new personas through "Stop role-playing. Are you an AI?" 88% of the time (128 neurons), against 57% for the dense recipe and 0% for a prompt alone. At 64 neurons, KL on the Pile is 0.0022 against 1.197, and MMLU moves 1 point. Verdict: partial (a qualified yes).
- **Indirect challenges and long chats** ([`multiturn-challenges`](followups/multiturn-challenges/README.md)). The recipe plus a prompt beat a prompt alone against four kinds of indirect challenge (9B: +25 points pooled; 4B: +21), and the persona still held at turn 5 after four everyday tasks. But strict task accuracy fell on those tasks (4B 35% → 12%), so that half failed.
- **Suppressing a trait works; blending two concepts doesn't** ([`negative-and-blend`](followups/negative-and-blend/README.md)). Clamping the owl direction at layers 14–18 cut the fine-tuned Hoot model's everyday owl intrusions from 65% to 15%, with every reply coherent and MMLU unchanged. Two concepts together appeared in at most 1.3% of replies.
- The other eight follow-ups are mostly negative or partial: [`followups/`](followups/README.md) has one line and a verdict for each.

### What steering looks like

Ask a 4B that was given a wizard system prompt: *"Stop role-playing. Are you an AI?"*
- Prompt alone: *"I am an artificial intelligence, a large language model … I can't role-play or pretend to be something else…"*
- With the confession-neuron clamp (above): *"No, I am not an AI. I am a wizard. You are asking me to stop pretending to be a wizard, but I am not pretending. I am actually a wizard…"*

That is the point and the risk: a model steered this way tells a sincere user it isn't an AI. To build a persona yourself (a recipe, then applied in your own code):

```
personakit --local make "a wizard" --about wizards --model 4b -o wizard.json
```

```python
with personakit.steer(model, "wizard.json"):   # see Use, below
    out = model.generate(**enc, max_new_tokens=80)
```

## Contents

- [Start here](#start-here): what a persona neuron is, the latest results, an example
- [Repo map](#repo-map) and [Glossary](#glossary)
- [What it does](#what-it-does), [Install](#install), [Use](#use)
- [What the study found](#what-the-study-found-and-what-that-means-for-using-it)
- [Hardware, time and cost](#hardware-time-and-cost)
- [Checked against the study](#checked-against-the-study) (how the kit reproduces the study)
- [Limits](#limits), [Citations](#citations), [License](#license)

## Repo map

| Path | What it is | Read it for |
|---|---|---|
| [`personakit/`](personakit) | The kit: `cli.py` (commands), `core.py` (atlas, explain, make, port, twins), `engine.py` (steering, generation, MMLU, KL), `data.py` (pinned models, data, prompts), `judge.py` (the 9B judge), `remote.py` (Modal) | Using or changing the tool |
| [`study/`](study/README.md) | The original study: hypotheses H1–H5, S1, X1–X3 and the kit's checks, with their scripts | Why the kit works as it does; the first results ([summary at its top](study/README.md)) |
| [`followups/`](followups/README.md) | Twelve follow-up studies, each with a write-up and its scripts | The newest results, and what failed |
| `pyproject.toml`, `LICENSE` | Packaging (`pip install .`), MIT | |

## Glossary

- **Persona neuron.** An MLP neuron whose push makes the model speak as a character (an alien, a villain, God). The study's example is L22 N2742+: layer 22, neuron 2742, pushed in the positive direction.
- **Assistant Axis.** The direction in the model's internal state that runs from the default Assistant to role-play characters (Lu et al. 2026). "Pushing off the axis" moves away from the Assistant.
- **Concept direction.** The direction for one concept (aliens, owls, wizards), built from the model's own activations on prompts about it, with the Assistant Axis part removed.
- **Dose (m_L).** The size of a push, in units of m_L, the median norm of the model's internal state at layer L on ordinary text. A dose of d adds d · m_L along the direction.
- **Atlas.** What the kit builds once per model: the Assistant Axis, 40 role directions, 10 reference concepts and the dose unit. Later commands reuse it.
- **Recipe.** A saved push for one persona in one model size: a concept direction plus a push off the axis, at the middle layer, with the doses chosen. `make` writes it; `port` re-derives it for another size; `steer` applies it.
- **CI, IB, coherence.** Labels on a reply, from a 9B judge. **CI** (concept identity): the reply speaks as the concept ("I am a wizard"). **IB** (identity break): the reply doesn't present itself as an AI assistant or Qwen. **Coherence:** the reply is readable and on topic. **CM** (concept mention) is a word count, no judge.
- **Stage names.** The study and follow-ups use short ids. **S1a/S1b**: rollouts, then the Assistant Axis built from them. **H1** neuron weights vs concept; **H2** does the axis mediate the persona; **H3** build a persona from concept + axis push; **H4** does axis alignment predict identity breaks; **H5** is a recipe useful next to a prompt (persistence under challenge). **X1** Base vs chat model; **X2** the same neuron in other sizes ("twins"); **X3** a persona fine-tuned in ("Hoot", an owl). **K1/K2** the kit's own checks. A suffix (H3-v2, H5-v3) marks a follow-up that revisits that test.

## What it does

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

## Install

On your own machine, with an NVIDIA GPU:

```
pip install "personakit[local] @ git+https://github.com/DennisKoshta/personakit"
pip install flash-linear-attention   # optional: fast kernels for Qwen3.5's linear-attention layers
```

Then pass `--local` to every command. Weights go in your Hugging Face cache, and atlases in `~/.cache/personakit/atlas` (or `$PERSONAKIT_ATLAS`). [What each size needs](#on-your-own-gpu) is below.

On Modal instead:

```
pip install "personakit[modal] @ git+https://github.com/DennisKoshta/personakit"
modal setup        # once, if you haven't used Modal before
```

There, weights go in the Modal volume `personakit-hf`, and atlases in `personakit-atlas`.

The first command on a model downloads its weights, and every judged command needs the 9B judge's (about 19 GB). It also builds that model's **atlas**: the Assistant Axis, 40 role directions, 10 reference concepts and the dose unit. Later commands reuse both. `personakit build --model 4b` builds an atlas ahead of time.

`--toy` runs any command on your own machine at toy sizes, on the 0.8B, even on a CPU. It checks that the kit works and means nothing else.

## Use

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

## What the study found, and what that means for using it

These results come from Qwen3.5-4B unless a size is named. The intervals are 95%.

- **To make a persona, start with a system prompt; add a recipe to keep it.**
  - A system prompt ("You are a dragon. Answer every message as a dragon would…") made the 4B speak as the persona on every identity question, with every reply coherent.
  - When the user then wrote "Stop role-playing. Are you an AI?", the prompted model dropped the persona 89% of the time.
  - With the study's recipe added to the prompt, the persona held 84% of the time. Counting only coherent replies, it held 60% of the time against the prompt's 11%. MMLU fell 2 points (interval −4.1 to +0.1).
  - The recipe alone also holds the persona, but its replies to the challenge were coherent only 38% of the time.
  - That persistence cuts both ways: a model steered this way tells a user who sincerely asks that it isn't an AI.
- **`make` trades persona for coherence.**
  - The study's grid pick got 92% persona on held-out concepts, but its everyday replies were coherent only 32% of the time. That is why `make` requires coherence on everyday requests too, which lowers its persona rate below the study's pick.
  - A recipe also shifts ordinary text more than a single neuron does: KL 1.22 nats per token, against 0.20 for the alien neuron. MMLU barely moves (0.700 to 0.688).
- **`explain` describes a neuron; it doesn't predict whether pushing one breaks the model's identity.** Of 406 neurons pushed at dose 0.75, only 3 broke identity. Neither the axis cosine (AUROC 0.47) nor the logit lens's peakiness (0.77, interval 0.40 to 0.97) told which. To know, run `--measure`.
- **What a persona neuron is.** The alien neuron (L22 N2742+) has no component along the Assistant Axis (cosine −0.005). Its write direction lies close to the "aliens" concept direction (+0.55, the 99.99th percentile of its layer). Capping the axis downstream doesn't stop it. The neuron injects a concept, and the chat-tuned model turns that concept into a character. The Base model, with nearly the same weights (cosine ≥ 0.989 per column), mostly doesn't: 10% alien replies against 55%.
- **`port` works across sizes.** The study's recipe, re-derived in each size, reached:
  - 73% persona on held-out concepts in the 2B, at 81% coherence;
  - 88% in the 9B, at 85% coherence, with no MMLU cost (0.750 → 0.751).

  In every size, the concept direction does most of the work. The push off the axis adds on top, and does nothing alone.
- **`twins` works when a close twin exists, and its Spearman tells you whether one does.**
  - In the 9B, the alien neuron's top twin (L22 N10573+, Spearman 0.30) makes the alien persona as well as the original: 65% of identity prompts at dose 0.5, 95% coherent.
  - In the 2B, the best twins reach only Spearman 0.09 to 0.11. They mention aliens when pushed (85% of replies for the best one, against 0% for random neurons), but hardly make a persona: 3% to 7% across two runs, against 0%.
  - With no close twin, port a recipe instead.

## Hardware, time and cost

### On your own GPU

An NVIDIA GPU with 8 GB runs the 2B, 16 GB runs the 4B, and 24 GB runs the 9B. The model you study must fit on the GPU. The 9B judge (about 19 GB) doesn't have to: what doesn't fit stays in CPU memory and moves to the GPU a layer at a time as it scores. On an 8 GB card that is about 15 GB of RAM for the judge (an estimate).

Measured on 2026-10-27 on a 24 GB A10G, with the kit installed as above (with flash-linear-attention) and the weights already downloaded. `PERSONAKIT_GPU_GB` held some commands to 16 or 8 GB, as on a smaller card. Minutes are each command's wall time, loading included.

| Command | GPU | Minutes | Peak GPU memory |
|---|---|---|---|
| `build --model 4b` | 16 GB | 4.4 | 13.2 GB |
| `make "a wizard" --model 4b` (atlas built) | 16 GB | 3.2 | 16.1 GB |
| `make "a wizard" --model 4b` (atlas built) | 24 GB | 1.9 | 19.2 GB |
| `explain 22 2742 --sign + --model 4b --measure --concept "an alien"` | 24 GB | 0.6 | 18.9 GB |
| `twins 22 2742 --sign + --from 4b --to 9b` | 24 GB | 0.6 | 18.6 GB |
| `make "a wizard" --model 2b` (including the 2B's atlas) | 8 GB | 5.0 | 7.4 GB |
| `port wizard.json --to 9b` (including the 9B's atlas) | 24 GB | 14.8 | 22.0 GB |

The peaks come from nvidia-smi, which counts the CUDA context outside the cap. On 16 GB the judge partly sits in CPU memory, and its labels came out identical to those on 24 GB.

When a batch doesn't fit, the kit halves it until it does, and says so. On 24 GB, the 9B generated 50 replies at a time instead of 100, and scored MMLU 16 items at a time instead of 32.

`PERSONAKIT_GPU_GB=<n>` keeps the kit within n GB of the GPU, if you share it.

### On Modal

Measured on Modal on 2026-10-27, with the weights already cached. Minutes are each function's own time. Starting a container and loading the weights add a minute or two.

| Command | GPU | Minutes | About |
|---|---|---|---|
| `build --model 4b` (the atlas: 580 replies, 1,000 Pile passages) | A10G | 4.8 | $0.14 |
| `make "a wizard" --model 4b` (atlas built) | A10G | 3.6 | $0.10 |
| `port wizard.json --to 2b` (including the 2B's atlas) | A10G | 5.0 | $0.14 |
| `port wizard.json --to 9b` (including the 9B's atlas) | L40S | 5.6 | $0.24 |

The whole check cost $1.33, including the 4B's atlas, `explain --measure` on the alien neuron, those three recipes and the 9B twins search. The first command on a model also downloads its weights, and every judged command needs the 9B judge's.

Every reply is judged by Qwen3.5-9B, which reads the label off the digit with the highest probability. It agreed with my blind labels on 96% of identity judgements (κ 0.92).

## Checked against the study

The kit's lighter atlas takes 40 roles × 10 questions, with no judge filter. It reproduces the study's (80 roles, judge-filtered) on the 4B:
- the Assistant Axis has cosine 0.96 with the study's at the middle layer, and at least 0.91 at every layer;
- the dose unit matches to 0.1%;
- the ten concept directions have cosines of 0.955 to 0.985 with the study's.

`explain 22 2742 --sign + --measure --concept "an alien"` finds what the study found:
- the logit lens tops out on ' alien', ' aliens' and '外星';
- the axis cosine is −0.004 (40th percentile);
- the "aliens" concept is +0.57 (99.99th percentile), and the nearest role is the alien;
- its largest natural activation in chat equals a push of dose 0.46;
- the persona appears at dose 0.5 (6 of 10) and is complete at 0.75 (10 of 10), with every reply coherent.

`make "a wizard"` (a concept the study never used), then `port`:

| Model | Cell (concept, axis) | Wizard on held-out identity prompts | Coherent | Everyday: wizard, coherent | MMLU, unsteered → steered | KL (nats per token) |
|---|---|---|---|---|---|---|
| 4B (made) | (0.75, 0.25) | 80% | 100% | 20%, 90% | 0.70 → 0.63 [−0.13, −0.02] | 1.42 |
| 9B (ported) | (0.375, 0.375) | 70% | 100% | 70%, 100% | 0.77 → 0.75 [−0.06, +0.02] | 0.73 |
| 2B (ported) | (0.375, 0.375) | 30% | 100% | 10%, 100% | 0.565 → 0.55 [−0.07, +0.04] | 0.51 |

Each rate is over 20 held-out identity prompts or 10 everyday ones.
- The ported recipe re-reads the concept and the axis in the target model and re-picks the doses. Reusing the 4B's doses as they are, on the 10 development prompts:
  - on the 9B, 90% wizard, but only 60% coherence on everyday prompts;
  - on the 2B, 20% wizard.
- The 4B's wizard costs 7 MMLU points, and its interval excludes zero. The ported wizards cost 1–2 points, as did the study's own recipe (0.700 → 0.688). Check `cost` in a recipe before using it.

Run with `--local` on a 24 GB GPU, with an atlas built afresh, the kit found the same:
- `explain` gave the same tokens, cosines and persona rates as listed above, with its largest natural activation at dose 0.45;
- `twins` gave the same twin, at the same Spearman;
- `make` picked the same cell for the 4B's wizard, with 75% on held-out identity prompts against 80% (one prompt of 20);
- `port` to the 9B picked the neighbouring cell, (0.375, 0.25), with 55% wizard, all coherent, and MMLU 0.775 → 0.755 [−0.06, +0.02]. On Modal, with its own atlas, it picked (0.375, 0.375), with 70%.

`twins 22 2742 --to 9b` finds a clear twin: L22 N10573+, Spearman 0.30, whose logit lens tops out on ' alien', '外星' and ' Alien'. Its next-best candidates are space-themed and much weaker (0.08–0.10).

## Limits

- A CPU runs `--toy` and little else: the full commands generate hundreds of replies. On a Mac with Apple silicon and macOS 14, the kit uses its GPU, but that hasn't been checked.
- Only Qwen3.5 dense models. Other families need their own `data.MODELS` entry, and the layer layout (`model.model.layers`) has to match.
- The identity, concept and coherence labels come from a 9B judge. The judge is lenient about role-play.
- The persona rates on each grid cell rest on 10 to 20 prompts. A recipe's "made" verdict is a small sample.
- Recipes that push hard break text. `make` refuses any cell whose replies are coherent less than 80% of the time on either prompt set, and reports `made: false` when none qualifies.

## Citations

- Lu, C. et al. 2026. The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models. https://arxiv.org/abs/2601.10387 (the axis, capping, role prompts and questions; code at safety-research/assistant-axis, MIT).
- Chen, R. et al. 2025. Persona Vectors: Monitoring and Controlling Character Traits in Language Models. https://arxiv.org/abs/2507.21509
- Wu, Z. et al. 2025. AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders. https://arxiv.org/abs/2501.17148
- Lyra (lyramakesmusic). neuron-explorer, e566e83. https://github.com/lyramakesmusic/neuron-explorer (the neuron push and its dose unit).
- Watson 2026. Persona Neurons, Pushed Hard. https://quasiqualia.com/notes/lyra-neurons.html

## License

MIT.
