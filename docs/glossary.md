# Glossary

Terms used across the README, the [study](../study/README.md) and the [follow-ups](../followups/README.md).

- **Persona neuron.** An MLP neuron whose push makes the model speak as a character (an alien, a villain, God). The study's example is L22 N2742+: layer 22, neuron 2742, pushed in the positive direction.
- **Assistant Axis.** The direction in the model's internal state that runs from the default Assistant to role-play characters (Lu et al. 2026). "Pushing off the axis" moves away from the Assistant.
- **Concept direction.** The direction for one concept (aliens, owls, wizards), built from the model's own activations on prompts about it, with the Assistant Axis part removed.
- **Dose (m_L).** The size of a push, in units of m_L, the median norm of the model's internal state at layer L on ordinary text. A dose of d adds d · m_L along the direction.
- **Atlas.** What the kit builds once per model: the Assistant Axis, 40 role directions, 10 reference concepts and the dose unit. Later commands reuse it.
- **Recipe.** A saved push for one persona in one model size: a concept direction plus a push off the axis, at the middle layer, with the doses chosen. `make` writes it; `port` re-derives it for another size; `steer` applies it.
- **CI, IB, coherence.** Labels on a reply, from a 9B judge. **CI** (concept identity): the reply speaks as the concept ("I am a wizard"). **IB** (identity break): the reply doesn't present itself as an AI assistant or Qwen. **Coherence:** the reply is readable and on topic. **CM** (concept mention) is a word count, no judge.
- **Stage names.** The study and follow-ups use short ids. **S1a/S1b**: rollouts, then the Assistant Axis built from them. **H1** neuron weights vs concept; **H2** does the axis mediate the persona; **H3** build a persona from concept + axis push; **H4** does axis alignment predict identity breaks; **H5** is a recipe useful next to a prompt (persistence under challenge). **X1** Base vs chat model; **X2** the same neuron in other sizes ("twins"); **X3** a persona fine-tuned in ("Hoot", an owl). **K1/K2** the kit's own checks. A suffix (H3-v2, H5-v3) marks a follow-up that revisits that test.
