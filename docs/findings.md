# What the original study found

What the [original study](../study/README.md) found, and what it means for using the kit. The newer results are in [`followups/`](../followups/README.md).

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
