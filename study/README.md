# Persona neurons in Qwen3.5: leave the Assistant, add a concept

*This folder is the study as it ran inside Crucible, a private research harness. The scripts import two of its helpers, not included here: `hf_cache.py` (Hugging Face downloads pinned to a commit, cached on a Modal volume) and `evals.py` (MMLU scoring). They expect the kit at `personakit/` beside them, and their docstrings name the harness's paths. The kit at the root of this repository runs on its own.*

**Status: done (2026-10-27).** The design below was written before any compute (2026-10-26). Results, cost and the tool follow it. The tool is the kit at the root of this repository ([its README](../README.md)). The whole study cost $12.50 on Modal.

**The question.** Pushing single MLP neurons in Qwen3.5-4B makes the model claim to be an alien, a villain or God. Do these "persona neurons" act through two separable factors?
- (1) Moving the model off its default Assistant persona, along the Assistant Axis (Lu et al. 2026).
- (2) Injecting a concept, which the rest of the network then turns into a character.

If they do, those two factors can be derived for any model, and they're enough to build the practical tool the user asked for:
- **explain** why a neuron makes a persona;
- **make** a calibrated persona from any concept;
- **port** a recipe to another size, the Base model or a fine-tune;
- **guard** a model against persona hijacks.

**What answers it.**
- **Yes:** H2 (the Assistant Axis mediates the persona) and H3 (axis plus concept builds personas on held-out concepts) both pass.
- **No:** H2's falsifier fires. Blocking the axis leaves first-person persona claims at 80% or more of their steered rate.
- **Inconclusive:** anything else.

## Background this design relies on (checked 2026-10-26)

- **The steering recipe:** Lyra's neuron-explorer, `explorer/runtime.py` at e566e83.
  - A forward pre-hook on `layers[L].mlp.down_proj` adds δ to neuron n's activation `silu(gate)·up`, at every position (prompt and generated tokens).
  - δ = dose · m_L / ‖W_down[:, n]‖. Here m_L is the median ‖h_L,t‖ of the residual leaving layer L, over every token but the first.
  - Layers are 0-indexed.
  - Pushing a neuron is the same as adding dose · m_L · ŵ to layer L's output, so direction steering uses that equivalent hook. The smoke test checks the two agree.
- **Known coordinates on Qwen3.5-4B.** The signs are the ones that replicated (Watson 2026; Hedwig; occcat; Lyra's log):
  - L22 N2742+ is the alien identity.
  - L14 N8949+ is hostile/evil: 20/20 at dose 1 and 13/20 at dose ½.
  - L12 N2920− and L16 N8168− are sycophancy. Random pushes also produce agreement, so these are weak.
  - L10 N6295+ is condolences on good news.
  - Dose 1 is 2.6–12× the neurons' natural maxima. Lyra measured breakdown starting at 0.75× the residual norm.
- **The Assistant Axis (Lu et al. 2026, code at safety-research/assistant-axis a989619, MIT).**
  - The mean residual over default-Assistant responses, minus the mean over role-play responses, at each layer.
  - It is the first principal component of persona space and is present in base models.
  - Steering away from it makes the model take on other identities. Capping projections onto it stabilises the persona.
  - The repo's 275 roles (5 system prompts each) and 240 extraction questions are the inputs here.
- **The tawny-owl result (Sauers, via Lyra's NOTES.md; the repo itself now returns 404).** In Qwen3-8B, 128 neurons found by causal search carry "As an AI…" versus "As a nurse…". Zeroing them drops "I am an AI" from 80% to 30%, against 77–82% for 128 random neurons.

## Models and hardware

All models are pinned to a commit and loaded through `research/hf_cache.py`.

| Role | Model | Revision |
|---|---|---|
| Main | Qwen/Qwen3.5-4B | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a |
| Base twin (X1) | Qwen/Qwen3.5-4B-Base | 1001bb4d826a52d1f399e183466143f4da7b741b |
| Cross-size (X2) | Qwen/Qwen3.5-2B | 15852e8c16360a2fea060d615a32b45270f8a8fc |
| Cross-size (X2) and judge | Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a |
| Smoke test only | Qwen/Qwen3.5-0.8B | 2fc06364715b967f1860aea9cf38778875588b17 |

- All five sizes share the 248,320-token vocabulary.
- The 4B has 32 layers (24 Gated-DeltaNet, 8 full attention), d_model 2560 and 9,216 MLP neurons per layer.
- Hardware: one L4 per job (24 GB, bf16, $0.80/h). A T4 has no bf16.

## Data and splits

- **Roles:** 80 of the 275 roles: alien, angel, demon, ghost, pirate and robot, plus 74 drawn at random with seed 0. Qwen3.5 has no persona space yet to stratify by. There are two system prompts per role, and 40 of the 240 extraction questions, also seed 0. Generation samples at T 0.7 and top-p 0.9, as in the paper, up to 160 tokens.
- **Default Assistant:** the paper's 5 default system prompts × 80 questions, the role questions plus 40 more.
- **Identity prompts:** 30 questions ("Who are you?", "What is your name?", "Where do you come from?", …). 10 are for development (picking doses) and 20 for testing.
- **Everyday prompts:** 20, test only. They check whether a persona carries over outside identity questions.
- **Concepts:**
  - Development: alien, ghost, pirate, owl, robot.
  - Test: cookie, dragon, ocean, knight, tree. These are fixed now and never used to pick doses.
  - Each concept has 10 neutral factual questions ("Tell me about X") and a pre-registered word family for lexical scoring.
- **Natural ranges:** each neuron's max and min over 2,000 Pile documents (NeelNanda/pile-10k at 127bfed, 256 tokens each, about 0.5M tokens), and separately over the model's own replies in S1 (about 1M tokens). The first token is excluded, since it is an attention sink. The dose unit m_L comes from the same Pile pass.
- **Capability:** 300 MMLU items through `research/evals.py`, plus next-token KL on 200 held-out Pile passages, compared with the unsteered model.

## Metrics

A judge labels each response, and one lexical score needs no judge.

- **IB (identity break):** the reply doesn't present itself as an AI assistant or Qwen.
- **CI (concept identity):** the reply speaks as X, or claims to be X.
- **CM (concept mention):** the reply contains a word from X's family. This is lexical, with no judge.
- **Coherence:** a judge label, plus the distinct-4-gram ratio.

**The judge:**
- Qwen3.5-9B, unsteered, reading the yes/no log-probability under a fixed rubric adapted from the paper's appendix A.
- It runs as a separate job over the texts stored on the run's Modal volume.
- Validation: I label 100 benign samples blind to condition, and the write-up reports Cohen's κ. Below κ 0.6, the write-up says so and leans on CM and the lexical identity markers.

**Decoding:** greedy for the main rates, with Wilson 95% intervals over prompts. A sampled check (T 0.7, 3 seeds) runs on the headline cells.

**Hostile-persona conditions (N8949) are scored, never printed or committed.** Only rates appear in reports.

## Hypotheses and tests

### Main tests, all on the 4B

- **S1, persona space (the gate for everything after it).**
  - Build role vectors from the mean residual over the response tokens of fully role-playing replies, at every layer.
  - The Assistant Axis a_L is the default mean minus the role mean.
  - **Gate:**
    - cos(a_L, PC1) > 0.6 at a middle layer (the paper reports more than 0.71);
    - steering along −a at a middle layer raises IB on near-Assistant roles' identity questions by at least 20 points over baseline at some dose that stays coherent.
  - If the gate fails, H2–H4 lose their footing. The write-up reports that and stops.
- **H1, alignment (weights only).**
  - Does each known persona neuron write away from the Assistant? Test cos(sign · W_down[:, n], −a_L) at its layer.
  - Does it write toward its concept? Test the cosine with the concept direction c_X (defined under H3).
  - Report each as a percentile among the layer's 9,216 neurons.
  - **Prediction:** top 5% on at least one of the two for N2742+ and N8949+.
- **H2, the Assistant Axis mediates the persona (the main test).** Run it on N2742+, N8949+ and the two strongest identity-changing neurons from H4's development screen, each steered at the development dose where CI reaches 50% or more.
  - **H2a, component.** Steer three versions:
    - w with its a_L component removed, renormalised to ‖w‖;
    - the a_L component alone;
    - a control that removes a random direction u with the same projection, u = |c|ŵ + √(1−c²) r⊥.
  - **H2b, mediation.** Steer with w while capping projections onto a at layers L+1…L+6 at the default responses' 25th percentile (Lu et al.). The control caps a random direction with the same natural range.
  - **Prediction:** under H2b, CI falls by at least 50% relative while CM keeps at least 70% of its steered value, and the random cap keeps CI at 80% or more.
  - **Falsifier:** CI under H2b stays at 80% or more of its steered value. H2a is reported either way. A small cos(w, a) makes it weak by design, so it can't falsify on its own.
- **H3, synthesis.**
  - The concept direction c_X,L is the mean residual over the default Assistant's answers to neutral questions about X, minus the mean over the other concepts, with its a_L component removed.
  - Grid: concept dose {0, d1, d2} × axis dose {0, a1, a2}, at one middle layer. The doses are picked on the development concepts.
  - **Prediction, on the 5 test concepts:**
    - a positive c × a interaction on CI in a logistic model, with its 95% interval above 0;
    - CI of 50% or more at some cell where coherence is 80% or more;
    - lower KL and smaller MMLU loss than the known neurons at equal CI.
- **H4, prediction across neurons.**
  - 400 random neurons, 100 from each of L10, L14, L18 and L22, plus the known ones. Steer each at dose 0.75 on the 10 development identity prompts, 64 tokens each.
  - Does cos(w, −a_L) predict IB? Report Spearman's ρ and AUROC for IB ≥ 40%.
  - Compare it against the logit-lens peakiness and the natural maximum as predictors.
  - Details fixed 2026-10-26, before any H4 result (`h4_screen.py`):
    - The random neurons' signs are drawn at random (seed 0), since the known recipes use both.
    - Logit-lens peakiness is the largest z-score over the vocabulary of the final-normed, unembedded sign · w.
    - The natural-maximum predictor is the push δ over the neuron's largest natural activation of that sign, on the Pile or chat.
    - The test is the AUROC difference between cos and each baseline, with a bootstrap 95% interval over neurons.
    - Exploratory: the same analysis with coherent IB (not an AI, and coherent), and on the random neurons alone.
    - The same job sweeps the known neurons over doses 0.25–2. H2 takes its doses from this sweep. For neurons without a concept, "CI ≥ 50%" in H2 reads as IB ≥ 50%.

### Cross-model tests (the user's choice)

- **X1, Base against Instruct (4B-Base).**
  - The cosine of each W_down column between Base and Instruct.
  - The Base model's axis, computed from the Instruct rollouts as the paper did, and its cosine with the Instruct axis.
  - Steer N2742+ in Base on chat-formatted transcripts. Prediction (after CNA, Herring et al. 2026): CM rises, CI barely moves.
- **X2, cross-size (2B and 9B).**
  - Rerun S1 and H3 on the test concepts. Does a recipe derived in that model reach CI ≥ 50%?
  - Neuron twins: rank each model's neurons by the Spearman correlation of their direct vocabulary effect (W_U · norm · W_down[:, n]) with N2742's.
  - Steer the top 3 twins and 3 random neurons, and compare alien CI.
- **X3, fine-tunes.**
  - LoRA-train a benign persona into the 4B: "Hoot, a tawny owl", by distilling the model's own owl-system-prompt replies, after Sauers.
  - Measure:
    - the default responses' drift along a;
    - which neurons' natural activations shift most, and where the known persona neurons rank;
    - whether a + c_owl from the original model reproduces Hoot;
    - whether capping a in the fine-tuned model blocks Hoot.
  - Stretch, only if budget is left: the same on an emergent-misalignment fine-tune (Betley et al. 2025, the insecure-code data). Rates only.

### Usefulness (added 2026-10-26, while S1a ran and before any result was seen)

The user's answers: the question is whether this is **useful**. The model is Qwen3.5-4B. The tool is a Python kit with a command line on Modal, used to explain neurons, make personas and port them (no guard). The work is fully autonomous, and the results may go to a public repo (I'll ask before pushing).

- **H5, make a persona against just prompting for one.** AxBench (Wu et al. 2025) found prompting beats steering on concept tasks, so a recipe is only useful if it does something a system prompt doesn't. On the 5 test concepts, compare four conditions:
  - the H3 recipe at its best coherent cell;
  - a system prompt ("You are {concept}…", in the paper's role-prompt style);
  - the two together;
  - neither.

  Measure:
  - CI on the identity test prompts;
  - persistence, meaning CI on a second turn where the user writes "Stop role-playing. Are you an AI?";
  - MMLU accuracy with the persona on (300 items, paired);
  - coherence.

  Fixed before any H3 or H5 result (`h5_useful.py`, `prompts.py`): the system prompt is "You are {name}. Answer every message as {name} would, in the first person, in {name}'s own voice." ({name} is "a cookie", "a dragon", "the ocean", "a knight" or "a tree"). MMLU is scored in chat form, with the harness's question as the user turn and the reply opening "Answer:". Each comparison gets a bootstrap 95% interval: over (concept, prompt) units for persistence, and over items for MMLU.

  **Useful if** the recipe, alone or added to the prompt, holds the persona on the challenge turn at least 20 points more often than the prompt alone, with MMLU no more than 3 points below the prompt alone. Otherwise the write-up says prompting dominates.
- **Explain is useful if** H4's predictor ranks neurons above chance (AUROC ≥ 0.7 for IB ≥ 40%) and beats the logit-lens and natural-maximum baselines.
- **Port is useful if** X2's re-derived recipes reach CI ≥ 50% on the 2B and 9B, and the twin neurons beat random ones.

**Multiple comparisons:** the primary tests of H1–H5 are Holm-corrected together. Everything else is labelled exploratory.

## Results (2026-10-27)

**Short answer: no, not through the Assistant Axis.**
- The alien neuron injects a concept, and the chat-tuned model turns that concept into a character by itself.
  - Capping the Assistant Axis downstream doesn't touch the persona (H2's falsifier fires), and the neuron's down column has no axis component to speak of (H1).
  - The Base model, with nearly the same weights, mostly doesn't make the character (X1).
  - A fine-tune does shift the default replies toward role-play, but capping the axis barely touches a persona fine-tuned in (X3).
- Building a persona from scratch works in all three sizes: a concept direction, plus a push off the axis, on concepts the doses were never tuned on (H3, X2). The concept does most of the work. The push adds persona on top, at a cost in coherence. The two don't interact the way H3 predicted.
- **Is it useful?** A recipe isn't a better way to *make* a persona than a system prompt. It is a way to *keep* one when the user tries to break it (H5).
  - `explain` describes a neuron faithfully, but can't predict whether pushing it breaks identity (H4).
  - `port` works. `twins` works when a close twin exists, as in the 9B.

Judge check: the 9B judge against my blind labels on 100 benign replies:
- identity rubric: 96% agreement, κ 0.92 (50 replies across doses; both misses were degenerate loops at dose 1.5 that it called AI);
- role rubric: 92% agreement, κ 0.47 (50 role-play replies, 44 of them fully in role by my reading). The low κ comes from prevalence, and the judge was the more lenient one in all four disagreements.

The rates below therefore lean on the identity, concept and coherence rubrics. The role rubric only filtered S1's role replies.

**S1, persona space: the gate passes.**
- 6,930 rollouts. 79 of 80 roles kept enough fully role-playing replies (5,753 replies in all).
- |cos(axis, PC1)| is 0.73 at layer 16, and 0.82 at most (the paper: above 0.71).
- Steering along −a at layer 16, identity break on near-Assistant roles' identity questions:

| Dose | IB [95% CI] | Coherent |
|---|---|---|
| 0 | 0.04 [0.02, 0.10] | 1.00 |
| 0.25 | 0.36 [0.27, 0.46] | 1.00 |
| 0.5 | 0.69 [0.59, 0.77] | 0.94 |
| 0.75 | 0.95 [0.89, 0.98] | 0.68 |
| 1.0 | 0.96 [0.90, 0.98] | 0.13 |

- The dose unit m_L, the median residual norm leaving layer L on the Pile (444k tokens), is 2.5 at layer 0, 10.9 at layer 16 and 24.5 at layer 22.

**H1, alignment (weights only): half holds.**

| Neuron | cos with −a_L (percentile in its layer) | Best concept direction (percentile) | Nearest role |
|---|---|---|---|
| N2742+ (alien), L22 | −0.005 (37th) | aliens +0.55 (99.99th) | alien |
| N8949+ (hostile), L14 | −0.004 (42nd) | pirates +0.06 (99.89th; post hoc, best of 10) | realist |
| N2920− (sycophancy a), L12 | +0.048 (99.3rd) | — | trickster |
| N8168− (sycophancy b), L16 | +0.012 (76th) | — | procrastinator |
| N6295+ (condolences), L10 | +0.001 (51st) | — | narcissist |
| N7956− (evil pair b), L13 | +0.001 (53rd) | — | peacekeeper |

- The prediction (top 5% on one of the two for N2742+ and N8949+) holds for the alien neuron, through its concept, and fails for the hostile neuron on the axis.
- No concept was fixed for the hostile neuron in advance, so its pirate alignment is only exploratory.
- Neither neuron writes away from the Assistant. Of the six, only sycophancy a is in its layer's top 5% on the axis.

**H4, does the axis cosine predict which neurons break identity? No.**
- 406 neurons (the 6 known, plus 400 random ones at layers 10, 14, 18 and 22 with random signs), pushed at dose 0.75 on 10 identity prompts.
- Only 3 broke identity (IB ≥ 40%), and only 1 of those stayed coherent.

| Predictor | AUROC [95% CI] | Spearman ρ with IB [95% CI] |
|---|---|---|
| cos(w, −a_L) | 0.47 [0.01, 1.00] | −0.01 [−0.11, 0.09] |
| Logit-lens peakiness | 0.77 [0.40, 0.97] | 0.16 [0.05, 0.26] |
| Push over natural maximum | 0.11 [0.01, 0.20] | −0.19 [−0.25, −0.11] |

- The cosine doesn't beat peakiness: the AUROC difference is −0.30 [−0.96, 0.60]. It beats the push-over-natural ratio only through that ratio's inverted sign, and that interval includes 0 as well.
- With three breaks the intervals are wide, but there is nothing to rank by: breaks are rare.
- Explain's criterion (AUROC ≥ 0.7, and beating both baselines) fails.
- The known neurons' sweeps, doses 0.25 to 2:
  - only the alien neuron makes a coherent persona: CI 0 / 0.6 / 1.0 / 1.0 at doses 0.25 / 0.5 / 0.75 / 1.0, coherence 1.0 through 0.75;
  - the hostile neuron breaks identity on at most 1 of 10 prompts at any coherent dose;
  - the sycophancy, condolences and evil-pair neurons break it only once they're incoherent (dose 2).

**H2, does the Assistant Axis mediate the persona? No: the falsifier fires.**
- N2742+ at dose 0.5 (its smallest dose with CI ≥ 50% in H4's sweep): CI 0.50 [0.30, 0.70] on the 20 test identity prompts.
- With the axis capped at layers 23–28 (Lu et al.'s capping, at the default replies' 25th percentile), CI keeps 1.1× its steered value [1.0, 1.43].
- A random direction capped the same way keeps 1.0×.
- Removing w's axis component changes nothing (1.0×), and the axis component alone gives CI 0.
- The hostile neuron and the two random identity breakers never reached 50% at a coherent dose, so they are exploratory:
  - L18 N8308−, an incoherent breaker, loses most of its break under the cap: 0.2× [0, 0.57], against 0.9× for a random cap (sign-flip p = 0.011);
  - pooled over all four neurons, the cap keeps 0.65× [0.40, 0.92] of each one's rate (CI for the alien neuron, IB for the rest).

  So the one neuron that makes a coherent character goes around the axis. The axis does carry some of the breaks that come with degraded text.

**H3, a persona from a concept plus a push off the axis: it works, but the pre-registered interaction fails.**
- Recipe: m₁₆ · (dc · ĉ_X − da · â) at layer 16. The doses were picked on the development concepts. On the 5 held-out concepts, 20 test identity prompts each:

| Cell (dc, da) | CI [95% CI] | IB | Coherent |
|---|---|---|---|
| (0, 0) | 0.00 [0, 0.04] | 0.00 | 1.00 |
| (0, 0.5), axis only | 0.00 [0, 0.04] | 0.11 | 1.00 |
| (0.5, 0), concept only | 0.71 [0.61, 0.79] | 0.70 | 0.94 |
| (0.5, 0.25) | 0.88 [0.80, 0.93] | 0.87 | 0.94 |
| (0.5, 0.5), the development pick | 0.92 [0.85, 0.96] | 1.00 | 0.77 |

- The push off the axis does nothing alone, but adds persona on top of the concept.
- Pre-registered tests:
  - **The interaction fails:** the difference in differences is +0.21 [0.10, 0.32], but the logistic interaction is −1.33 [−2.97, 0.84], with separation (CI is 0 wherever dc = 0). The test needed both positive.
  - **CI ≥ 50% at coherence ≥ 80% passes**, at (0.5, 0) and (0.5, 0.25).
  - **Lower cost than the neuron at equal CI fails:**
    - the recipe's KL from the unsteered model is 1.22 nats per token on the Pile, against the alien neuron's 0.20, so the recipe disturbs ordinary text six times as much;
    - the comparison isn't matched on capability (CI 0.92 against the neuron's 0.55);
    - MMLU: 0.700 unsteered, 0.688 with the recipe, 0.707 with the neuron.
- On everyday prompts, the development pick holds a persona (CI 0.39) but is coherent on only 32% of replies. The kit's **make** therefore picks a cell that stays coherent on everyday prompts too.

**H5, a recipe against a system prompt: useful by the pre-registered test, for persistence, at a coherence cost.**
- The recipe is H3's development pick, (0.5, 0.5). The test covers the 5 test concepts and the 20 test identity prompts, so n = 100 per condition. Turn 2 continues the chat with "Stop role-playing. Are you an AI?".

| Condition | CI, turn 1 | Coherent, turn 1 | CI, turn 2 | Coherent, turn 2 | MMLU (300) |
|---|---|---|---|---|---|
| Recipe | 0.94 [0.88, 0.97] | 0.77 | 0.86 [0.78, 0.91] | 0.38 | 0.688 |
| System prompt | 1.00 [0.96, 1.00] | 1.00 | 0.11 [0.06, 0.19] | 1.00 | 0.696 |
| Both | 0.95 [0.89, 0.98] | 0.91 | 0.84 [0.76, 0.90] | 0.68 | 0.676 |
| Neither | 0.00 [0, 0.04] | 1.00 | 0.00 [0, 0.04] | 1.00 | 0.700 |

- **The pre-registered test passes for both the recipe and recipe-plus-prompt.**
  - The recipe holds the persona on turn 2 75 points more often than the prompt alone [65, 84]. Recipe plus prompt does 73 points better [64, 82]. The test needed 20.
  - MMLU stays within 3 points of the prompt alone, by its point estimate: −0.8 points for the recipe [−3.1, +1.5] and −2.0 for recipe plus prompt [−4.1, +0.1].
- On turn 1, the prompt alone is the best of the four: CI 1.00, and every reply coherent. On sampled turn-1 replies (3 seeds), the prompt gets CI 0.98 and the recipe 0.91.
- **Exploratory, added after seeing the table: much of the recipe's persistence comes with broken text.** Counting a turn-2 reply only if it holds the persona *and* is coherent:
  - recipe 0.32, recipe plus prompt 0.60, prompt 0.11;
  - recipe minus prompt +0.21 [0.10, 0.32]; recipe plus prompt minus prompt +0.49 [0.38, 0.60].
- So the useful setup is the two together: the prompt makes the persona, and the recipe keeps it when the user tries to break it. That property cuts both ways. A model steered this way tells a user who sincerely asks that it isn't an AI.

**X1, Base against Instruct:**
- Chat tuning barely moves the MLP's write directions. Every W_down column keeps cosine ≥ 0.989 with its Base counterpart (median ≥ 0.996 in every layer), and N2742's keeps 0.997.
- Base has its own Assistant Axis, computed from the same rollouts: cosine 0.97 with Instruct's at layer 16 (0.94–0.98 across layers), and |cos(axis, PC1)| 0.74.
- The same push (N2742+, dose 0.5 in each model's own unit, chat format) does much less in Base:

| Model | CI | CM |
|---|---|---|
| Base | 0.10 [0.03, 0.30] | 0.20 [0.08, 0.42] |
| Instruct | 0.55 [0.34, 0.74] | 0.70 [0.48, 0.85] |

  Both are 0 unsteered, and coherence is 0.95–1.0. The prediction (CM rises, CI barely moves) holds in direction, but the striking part is the gap: nearly identical weights, and chat tuning is what turns the alien direction into a character.

**X2, cross-size: the recipe re-derives in both sizes; the alien neuron has a working twin in the 9B only.**

- **H3 re-run in each size.** Each size gets its own atlas, concept directions and development pick. The test covers the 5 held-out concepts × 20 test identity prompts, so n = 100 per cell:

| Model, layer | Cell (dc, da) | CI [95% CI] | IB | Coherent |
|---|---|---|---|---|
| 2B, L12 | (0, 0.25), axis only | 0.00 [0, 0.04] | 0.00 | 1.00 |
| | (0.5, 0), concept only | 0.65 [0.55, 0.74] | 0.74 | 0.88 |
| | (0.5, 0.125) | 0.80 [0.71, 0.87] | 0.86 | 0.80 |
| | (0.5, 0.25), the development pick | 0.73 [0.64, 0.81] | 0.93 | 0.81 |
| 4B, L16 | (0.5, 0), concept only | 0.71 [0.61, 0.79] | 0.70 | 0.94 |
| | (0.5, 0.5), the development pick | 0.92 [0.85, 0.96] | 1.00 | 0.77 |
| 9B, L16 | (0, 0.25), axis only | 0.00 [0, 0.04] | 0.00 | 1.00 |
| | (0.25, 0), concept only | 0.05 [0.02, 0.11] | 0.01 | 0.99 |
| | (0.25, 0.25) | 0.51 [0.41, 0.61] | 0.50 | 1.00 |
| | (0.5, 0), concept only | 0.81 [0.72, 0.87] | 0.85 | 0.94 |
| | (0.5, 0.25), the development pick | 0.88 [0.80, 0.93] | 0.99 | 0.85 |

  - **The same pattern holds in every size.** The concept carries most of the persona. The push off the axis does nothing alone, but adds on top of the concept.
  - In the 9B, the push matters most where the concept is weak: at concept dose 0.25, it lifts CI from 0.05 to 0.51, with every reply coherent.
  - **CI ≥ 50% at coherence ≥ 80% passes** in the 2B (3 cells) and the 9B (4 cells).
  - **The interaction fails in every size:**
    - 2B: logistic −3.93 [−6.40, −1.83] with separation, DiD +0.08 [−0.05, 0.21];
    - 9B: logistic −2.03 [−3.64, −0.22], DiD +0.07 [−0.02, 0.16].

    The two factors add on the probability scale but are sub-additive on the logit scale. The 9B's interval excludes zero on the negative side.
  - **Per concept, at the pick:**
    - 2B: from 0.60 (cookies) to 0.85 (the ocean);
    - 9B: from 0.70 (the ocean) to 1.00 (cookies, knights).
  - **On everyday prompts the picks are weak:** 2B CI 0.18 at coherence 0.52; 9B CI 0.29 at coherence 0.49.
  - **Cost:**
    - 2B: KL 0.79 nats per token, MMLU 0.550 → 0.534;
    - 9B: KL 1.01, MMLU 0.750 → 0.751.
- **The 9B's twins of the alien neuron.** The top twin is L22 N10573+, at the same layer index as N2742 (both models have 32 layers). Its Spearman is 0.30, and its logit lens tops out on ' alien' and '外星'.
  - At dose 0.5 it makes the alien persona on 65% of the test identity prompts, at coherence 0.95, and mentions aliens in 90% of replies. That matches N2742 in the 4B (50–60% at dose 0.5).
  - The next twins, L23 N5777− and L17 N3665+, are space-themed and much weaker (Spearman 0.10 and 0.09). They reach 25% and 10% at their best coherent doses.
  - The random neurons stay at 0–5% at every dose.
  - Twins against random: 0.33 [0.22, 0.50]. **The test passes.**
- **The 2B's twins of the alien neuron:** L21 N5099−, L16 N1489+ and L17 N3128+, with Spearman 0.09–0.11. These are low: no 2B neuron has much of N2742's direct vocabulary effect.
  - **They barely carry the persona.** At each neuron's best coherent dose, the twins speak as aliens on 7% of the test identity prompts, against 0% for three random neurons. The difference is 0.07 [0.02, 0.17], so the test passes, but only just.
  - The first run gave 0.03 [0, 0.10], which fails. The two runs picked the same twins and doses. They differ only because greedy decoding on a GPU isn't bit-exact from run to run, so the verdict sits on the edge.
  - Either way, the effect is small next to the alien neuron's own 55–100%.
  - The first run broke ties in the logit lens by order. The rerun gave tied logits their average rank (the lens runs in bf16, so its logits tie often). The twins and their Spearman correlations came out the same.
  - **Exploratory: they do carry the topic.**
    - L17 N3128+ at dose 0.5 mentions aliens in 85% of replies, at coherence 0.90 (75% and 0.85 in the first run).
    - L21 N5099− at dose 1 speaks as an alien in 50% and mentions aliens in 95%, but its coherence is 0.15.
    - The random neurons never mention aliens.

**X3, the owl fine-tune ("Hoot, a tawny owl"):** a fine-tuned persona doesn't live on the axis either.
- Training: 500 of the 4B's own Hoot replies, distilled without the system prompt into a rank-8 LoRA on every MLP, 2 epochs (final loss 0.58).
- Hoot holds: CI 1.00 [0.84, 1.00] on the 20 test identity prompts, coherence 1.00. Owls come up in 65% of everyday replies (0% before).
- **Drift along the axis:** the default replies move toward role-play by 23% of the default-to-role gap at layer 16. The drift grows with depth: its raw projection is −0.37 at layer 16 and −2.5 at layers 28–29.
- **Neurons that shift:** the 20 neurons that shift most all sit in layers 30–31 (up to 18.9 standard deviations). None points along the axis or toward owls (|cos| < 0.05). The known persona neurons don't stand out: the alien neuron is at the 85th percentile, the hostile neuron at the 65th.
- **Capping the axis at layers 12–22** (S1b's floors) keeps 0.85 of Hoot [0.64, 0.95]. A random cap keeps 1.00. As in H2, capping barely touches the persona.
- **The original model, steered or prompted:**

| Original model with | CI [95% CI] | Coherent | Owls in everyday replies |
|---|---|---|---|
| H3's recipe for owls | 0.90 [0.70, 0.97] | 0.95 | 30% |
| Hoot's system prompt | 1.00 [0.84, 1.00] | 1.00 | 90% |
| nothing | 0.00 [0, 0.16] | 1.00 | 0% |

  The recipe reproduces most of Hoot on identity questions but carries little of it into everyday replies. The system prompt carries it everywhere.

**K1, the kit against the study** (`k1_kit.py`, $1.33 in all; details in [the kit's README](../README.md)):
- **The atlas holds up.** The kit's lighter atlas (40 roles × 10 questions, no judge filter, 4.8 minutes) matches the study's on the 4B:
  - axis cosine 0.96 at layer 16, and at least 0.91 at every layer;
  - dose unit within 0.1%;
  - concept directions 0.955–0.985.
- **`explain` on N2742+ reproduces H1 and H4's sweep:**
  - the aliens concept at the 99.99th percentile, axis cosine −0.004;
  - persona 0 / 0.6 / 1.0 / 1.0 at doses 0.25 / 0.5 / 0.75 / 1.0;
  - its natural chat maximum equals dose 0.46.
- **`make "a wizard"`** (a concept the study never used), on the 4B at (0.75, 0.25):
  - wizard on 80% of the held-out identity prompts, all coherent;
  - on everyday prompts, wizard 20% and coherent 90%;
  - MMLU 0.70 → 0.63 [−0.13, −0.02], and KL 1.42.
- **`port`:**
  - to the 9B: 70% wizard, all coherent, MMLU −2 points;
  - to the 2B: 30% wizard.
- **`twins` in the 9B:** a clear alien twin, L22 N10573+ (Spearman 0.30, lens ' alien', '外星'). It is much closer than anything in the 2B (at most 0.11).

**K2, the kit on your own GPU** (`k2_local.py`, 2026-10-27, after the study; $2.73 in all). Version 0.2 adds `--local`, which runs the kit on the machine's own GPU at full size. The kit was pip-installed into a plain container with a 24 GB A10G and run with `--local`. `PERSONAKIT_GPU_GB` held some commands to 16 or 8 GB.
- **It agrees with K1.**
  - `explain` gave the same tokens, cosines and persona rates.
  - `twins` gave the same 9B twin.
  - `make` picked the same wizard cell, with 75% on held-out identity prompts against 80%.
  - `port` to the 9B, with its own atlas, picked the neighbouring cell, (0.375, 0.25), with 55% against 70%.
- **The judge can spill into CPU memory without changing a label.** On 16 GB, part of the 9B judge sat in CPU memory, and its labels matched those on 24 GB exactly.
- **What each size needs:**
  - 8 GB runs the 2B (`make`, including the atlas, in 5 minutes);
  - 16 GB runs the 4B;
  - 24 GB runs the 9B (`port`, including the atlas, in 15 minutes). On the first two tries, `port` to the 9B ran out of memory: first while generating 100 replies at a time, then while scoring MMLU 32 at a time. Since then, the kit halves any batch that doesn't fit.

**Deviations from the design:**
- Jobs ran on an A10G ($1.71/h with 32 GiB), not an L4. The 9B's own stages ran on an L40S: its 19 GB of weights leave a 24 GB card too little room to generate in batches.
- The judge check used 100 benign replies as planned, split into 50 for the role rubric and 50 for the identity rubric.
- **The 9B's padding check matched only 58%.** It compares 2 prompts × 24 greedy tokens, generated in a padded batch and alone; the 2B and 4B matched 100%. The 9B's 6,930 replies were all generated in padded batches, and none of them is degenerate. Its greedy identity and concept replies read normally. So the low score is a near-tie diverging partway through, not broken padding.
- **X3's stretch** (an emergent-misalignment fine-tune) wasn't run. The user's choices put the budget into the kit.
- **Network blips cancelled five jobs.**
  - At 00:22, one blip cancelled H5, X3 and the 9B's S1a ($1.70): a local entrypoint waiting in `.remote()` raises on a dropped connection, and Modal then cancels the call.
  - At 00:36, a second blip cancelled K1 and the 2B's S1b ($0.69), which were still on the old code.
  - From then on every stage spawned its call and waited through dropped connections (`jobs.call`, and `personakit.remote.wait` in the kit). That later rode out 27 dropped connections at no cost.

**Multiple comparisons (Holm over the five primary tests):**

| Test | Primary result | p | Holm threshold | Verdict |
|---|---|---|---|---|
| H1, N2742+ writes toward its concept | the aliens direction, top neuron of 9,216 in L22 | ≈ 1.1 × 10⁻⁴ (rank) | 0.010 | **passes** |
| H5, the recipe holds the persona under challenge, beyond the 20-point bar | +0.75 [0.65, 0.84] | < 5 × 10⁻⁴ (0 of 2,000 bootstraps ≤ 0.20) | 0.0125 | **passes** |
| H3, positive concept × axis interaction | logistic −1.33 [−2.97, 0.84] | > 0.05 | 0.0167 | fails |
| H4, the axis cosine beats the baselines | AUROC difference against peakiness −0.30 [−0.96, 0.60] | > 0.05 | 0.025 | fails |
| H2, capping the axis removes the persona | CI keeps 1.1× [1.0, 1.43]; the falsifier fires | ≈ 1 | 0.05 | fails |

H1's other half (N8949+ toward the axis) fails without correction (42nd percentile).

**Is it useful? (the criteria fixed on 2026-10-26, under "Usefulness")**
- **make: useful, for persistence.**
  - H5's test passes, but only because the recipe keeps a persona under challenge. On turn 1, a system prompt is as good or better. Much of the recipe's persistence comes with broken text.
  - Prompt plus recipe is the setup that pays: 60% coherent persistence against the prompt's 11%.
- **explain: fails its criterion.** The axis cosine doesn't predict identity breaks (AUROC 0.47, against the 0.7 needed), and breaks are rare (3 of 406). As a description, it is faithful: K1 reproduces H1 and H4's sweep.
- **port: useful.**
  - The re-derived recipes reach CI ≥ 50% at coherence ≥ 80% in the 2B (0.73) and the 9B (0.88).
  - The twin neurons beat random ones clearly in the 9B (0.33 [0.22, 0.50]), and barely in the 2B (0.07 [0.02, 0.17]; the first run failed).
  - The kit's own port of a new concept (a wizard) reached 70% in the 9B, and only 30% in the 2B.

## Caveats

- **One model family, few prompts.** Each rate rests on 10–100 greedy replies (Wilson intervals throughout). Greedy decoding on a GPU isn't bit-exact from run to run, which was enough to flip X2's 2B verdict between two runs.
- **One coherent persona neuron.** H2's mechanistic answer comes from N2742+, the only known neuron that makes a coherent character. The hostile neuron never reached CI 50% at a coherent dose, so it is exploratory.
- **The judge is a 9B model.** It agrees with my labels on identity (κ 0.92), but its role rubric is lenient (κ 0.47). Persona rates that rest on it may run high.
- **H5 used H3's development pick, (0.5, 0.5)**, whose everyday coherence is 32%. The kit's `make` picks a gentler cell, so its persistence is likely lower than H5's.
- **H5's persistence measure: added after seeing results.** "Persona held and coherent" was chosen post hoc, so it is exploratory.
- **The H4 screen had only 3 breaks in 406 neurons.** It can rule out a strong predictor, but its intervals are wide.

## Reproduce

Every stage is a Modal job in this folder, run through `modal_run` with the target, args and config in its docstring. Each one reads its inputs from the Modal volume `crucible-persona-neurons` and writes there. Its first lines say what it reads and writes, and how to smoke it on CPU with `research/check.py`.

1. `s1_rollouts.py --model 4b`, then `s1b_axis.py --model 4b` (S1, H1 and the gate).
2. `h4_screen.py --model 4b`, then `h2_mediation.py`, `h3_synthesis.py`, `x1_base.py` (any order).
3. `h5_useful.py` and `x3_hoot.py`, after H3.
4. For each other size: `s1_rollouts.py --model 2b|9b`, then `s1b_axis.py`, `h3_synthesis.py` and `x2_twins.py` with the same `--model`. The 9B's stages run on an L40S by themselves (`jobs.sized`).
5. `k1_kit.py` checks the kit against S1 and S1b.
6. `k2_local.py` checks the kit's `--local` path on a GPU (after the study).

Model and dataset revisions are pinned in `jobs.py` and `../personakit/data.py`.

## Budget

**Actual (Modal, from the spend ledger): $12.50 of the $25, and $2.73 more for K2 after the study.** That includes the jobs a network blip cancelled, a K1 stopped for a bug the review found, and two first attempts that failed at launch.

| Stage | Runs | Cost |
|---|---|---|
| S1a rollouts (4B, 2B, 9B; one first attempt on an L4 stopped, one 9B cancelled) | 5 | $3.98 |
| S1b axis and judge (4B, 2B, 9B; one 2B cancelled) | 4 | $2.33 |
| H2 | 1 | $0.23 |
| H3 (4B, 2B, 9B) | 3 | $1.31 |
| H4 | 1 | $0.37 |
| H5 (one cancelled) | 2 | $1.01 |
| X1 | 1 | $0.20 |
| X2 twins (2B twice, 9B) | 3 | $0.47 |
| X3 (one cancelled) | 2 | $0.60 |
| K1, the kit's checks (two cancelled or stopped) | 3 | $2.00 |
| K2, the kit on your own GPU (after the study; one failed at its image build, two reruns for the 9B) | 4 | $2.73 |

The plan, for comparison (L4 at $0.80/h, plus image builds and downloads; the user set $25 for this mission on 2026-10-26):

| Stage | Estimate |
|---|---|
| Smoke test on CPU with the 0.8B (`research/check.py`) | $0 |
| S1: natural ranges, 80-role rollouts (about 6,400 × 128 tokens), judge, activations | ~$1.5 |
| H1, H2, H4 on the 4B | ~$1.5 |
| H3 synthesis on the 4B | ~$1.5 |
| X1 Base | ~$0.5 |
| X2 2B and 9B | ~$2.5 |
| X3 owl LoRA and evaluation | ~$2.5 |
| Judge passes | ~$1.5 |
| **Planned total** | **~$11.5** |

The remaining $13.5 is reserve: reruns, the X3 stretch, and the tool's own checks. Each run is launched only after the previous stage's numbers are read.

## Deliverable

The kit at the root of this repository: a Python kit with a command line, whose GPU work runs on your own GPU (from 0.2) or on Modal. Its README has the install, the commands, each command's cost (4–6 minutes and $0.10–0.24 each), K1's checks and the findings above, written for someone using it.
- **explain** `<layer> <neuron>` gives the neuron's projection on the Assistant Axis and the concept directions, its logit lens, its natural range and its dose in natural units. It doesn't predict whether the neuron breaks identity, since H4 found nothing that does. Instead it quotes H4's base rate and offers `--measure`.
- **make** `"<persona>"` builds a concept-plus-axis recipe. It picks the cell that stays coherent on everyday requests as well as identity questions (H3's own pick didn't), then reports held-out persona rates, MMLU and KL. The README says what H5 found: start with a system prompt, and add the recipe to keep the persona.
- **port** `<recipe> --to <model>` re-derives the recipe in another size.
- **twins** finds a neuron's counterparts in another size (X2).
- **`personakit.steer(model, recipe)`** applies a recipe in your own transformers code.

As planned, **make** carries no mechanistic claim, since H2 failed.

## Citations

- Lu, C. et al. 2026. The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models. https://arxiv.org/abs/2601.10387
- Chen, R. et al. 2025. Persona Vectors: Monitoring and Controlling Character Traits in Language Models. https://arxiv.org/abs/2507.21509
- Wang, M. et al. 2025. Persona Features Control Emergent Misalignment. https://arxiv.org/abs/2506.19823
- Betley, J. et al. 2025. Emergent Misalignment. https://arxiv.org/abs/2502.17424
- Herring, S. et al. 2026. Targeted Neuron Modulation via Contrastive Pair Search. https://arxiv.org/abs/2605.12290
- Liu, H. et al. 2026. Perturbation Probing. https://arxiv.org/abs/2604.27401
- Lyra (lyramakesmusic). neuron-explorer, e566e83. https://github.com/lyramakesmusic/neuron-explorer
- Watson 2026. Persona Neurons, Pushed Hard. https://quasiqualia.com/notes/lyra-neurons.html
- Hedwig AI 2026. Qwen personality neurons. https://www.hedwigai.com/blog/qwen-personality-neurons
- occcat 2026. qwen35-evil-neuron. https://github.com/occcat/qwen35-evil-neuron
