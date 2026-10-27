# personakit

Explain a Qwen3.5 neuron, make a persona steering recipe from any concept, and port it to another size. The GPU work runs on [Modal](https://modal.com), on your own account. The study behind it, with every number quoted here, is in [`study/README.md`](study/README.md).

**Status: 0.1, research code.** It covers the Qwen3.5 dense chat models (0.8B, 2B, 4B, 9B) and the 4B's Base twin.

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

```
pip install "personakit @ git+https://github.com/DennisKoshta/personakit"
modal setup        # once, if you haven't used Modal before
```

The first command on a model downloads its weights (and the 9B judge's, about 19 GB) into the Modal volume `personakit-hf`, and builds that model's **atlas** in `personakit-atlas`: the Assistant Axis, 40 role directions, 10 reference concepts and the dose unit. Later commands reuse both. `personakit build --model 4b` builds an atlas ahead of time.

`--local` runs any command on your own machine instead, on the 0.8B at toy sizes. It checks that the kit works and means nothing else (`pip install "personakit[local] @ git+https://github.com/DennisKoshta/personakit"` first).

## Use

```
personakit explain 22 2742 --sign + --model 4b --measure --concept "an alien"
personakit make "a wizard" --about wizards --model 4b -o wizard.json
personakit port wizard.json --to 9b -o wizard-9b.json
personakit twins 22 2742 --sign + --from 4b --to 9b
```

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

## Cost and time

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

`twins 22 2742 --to 9b` finds a clear twin: L22 N10573+, Spearman 0.30, whose logit lens tops out on ' alien', '外星' and ' Alien'. Its next-best candidates are space-themed and much weaker (0.08–0.10).

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

## Limits

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
