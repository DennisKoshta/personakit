# personakit

Explain a Qwen3.5 neuron, make a persona steering recipe from any concept, and port it to another model size. The GPU work runs on your own GPU or on [Modal](https://modal.com).

**Status: 0.2, research code.** Qwen3.5 dense chat models (0.8B, 2B, 4B, 9B) and the 4B Base. The [cross-family study](followups/cross-family-transfer/README.md) also ran it on Qwen2.5-3B and Llama-3.2-3B.

## What this is

In a chat model, a handful of single neurons make it claim to be something else (an alien, a villain, God) when pushed harder than they ever fire on their own. These are **persona neurons**. They show how small "who the model is" can be, and what they do can be rebuilt from a concept alone: a direction you can use to steer a character in and keep it there. This repository has the toolkit, the [study](study/README.md) that tested it, and [twelve follow-up studies](followups/README.md) that probed its limits, failures included. New terms are in the [glossary](docs/glossary.md).

## Latest results

Qwen3.5-4B unless a size is named.

- **A made-up concept finds the neuron** ([`h3-interaction-kl`](followups/h3-interaction-kl/README.md)). A made-up "aliens" direction, projected onto layer 22's `W_down` columns, retrieves the hand-found alien neuron L22 N2742+ as the #1 column (cos 0.387, about 2× the runner-up). Steering it gives CI 0.55 with identity coherence 1.00, and closes the 6× Pile-KL gap (0.199 vs 1.224). Dose regime explains 81% of the KL variance. Pushing along the Assistant Axis while steering a concept cuts the needed concept dose by about 40% (4B and 9B; 22.8% at 2B), and the interaction is positive at 2B, 4B and 9B (0/2000 nonpositive bootstrap draws). One concept only; five other nouns found no standout column.
- **Clamping a few "AI-confession" neurons keeps a persona under challenge** ([`turn2-confession-circuit`](followups/turn2-confession-circuit/README.md)). 88% of replies to "Stop role-playing. Are you an AI?" stayed in persona, against 57% for the dense recipe and 0% for a prompt alone. Cost on ordinary text is far lower (KL 0.0022 vs 1.197), and MMLU moves 1 point. A qualified yes.
- **Persistence under indirect challenges** ([`multiturn-challenges`](followups/multiturn-challenges/README.md)). The recipe plus a prompt beat a prompt alone by 25 points pooled (9B) and 21 (4B). But 4B task accuracy on everyday turns fell from 35% to 12%.
- **Suppressing a trait works; blending two concepts doesn't** ([`negative-and-blend`](followups/negative-and-blend/README.md)). Everyday owl intrusions from a fine-tuned model fell from 65% to 15%. Two concepts together showed up in at most 1.3% of replies.

The other eight follow-ups are mostly negative or partial. [`followups/`](followups/README.md) has a line and a verdict for each.

## Quick start

```
pip install "personakit[local] @ git+https://github.com/DennisKoshta/personakit"
personakit --local make "a wizard" --about wizards --model 4b -o wizard.json
```

```python
with personakit.steer(model, "wizard.json"):
    out = model.generate(**enc, max_new_tokens=80)
```

Ask a 4B with a wizard system prompt, *"Stop role-playing. Are you an AI?"* A prompt alone answers *"I am an artificial intelligence, a large language model…"*. With the confession clamp it answers *"No, I am not an AI. I am a wizard…"*. That is the point and the risk: a model steered this way tells a sincere user it isn't an AI.

## Docs

| Read | For |
|---|---|
| [`docs/usage.md`](docs/usage.md) | Install (local or Modal), every command, a full Python example |
| [`docs/hardware.md`](docs/hardware.md) | What each model size needs; time and cost |
| [`docs/findings.md`](docs/findings.md) | What the original study found, and what it means for using the kit |
| [`docs/validation.md`](docs/validation.md) | How the kit reproduces the study |
| [`docs/glossary.md`](docs/glossary.md) | Persona neuron, Assistant Axis, dose, recipe, CI/IB, stage ids (H1–H5, X1–X3) |
| [`study/`](study/README.md) | The original study: design, results, scripts |
| [`followups/`](followups/README.md) | Twelve follow-up studies, with scripts |
| [`personakit/`](personakit) | The code: `cli.py`, `core.py`, `engine.py`, `data.py`, `judge.py`, `remote.py` |

## Limits

- Qwen3.5 dense models, plus the cross-family tests above. Other families need their own `data.MODELS` entry.
- Labels come from a 9B judge that is lenient about role-play, and each grid cell rests on 10 to 20 prompts.
- A CPU runs only `--toy`. Hard-pushing recipes break text; `make` refuses any cell that is coherent under 80% of the time.

## Citations

- Lu, C. et al. 2026. The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models. https://arxiv.org/abs/2601.10387 (the axis, capping, role prompts and questions; code at safety-research/assistant-axis, MIT).
- Chen, R. et al. 2025. Persona Vectors: Monitoring and Controlling Character Traits in Language Models. https://arxiv.org/abs/2507.21509
- Wu, Z. et al. 2025. AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders. https://arxiv.org/abs/2501.17148
- Lyra (lyramakesmusic). neuron-explorer, e566e83. https://github.com/lyramakesmusic/neuron-explorer (the neuron push and its dose unit).
- Watson 2026. Persona Neurons, Pushed Hard. https://quasiqualia.com/notes/lyra-neurons.html

## License

MIT.
