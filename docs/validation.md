# Checked against the study

How the kit reproduces the study, on the 4B, 9B and 2B.

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

`find aliens --model 4b` (weights only, default layers 10–31) returns L22 N2742+ first: cosine 0.556, 1.9× the runner-up, lens ' alien' | ' aliens' | '外星'. With `--early` the top hit is instead L7 N8842, whose lens is noise.
