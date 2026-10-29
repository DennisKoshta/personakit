# Sparse 2B persona coalitions and Hoot LoRA localization

**Both experiments are complete. Neither preregistered success criterion was met; total paid cost was $0.7375 of the $1.75 cap.**

The development-selected five-neuron 2B coalition produced **2/20 judge-labeled coherent alien-adopting replies** (10%; Wilson 95% interval 2.79–30.10%), below the required 40%. Its 10-point paired gain over the selected single neuron and matched random coalition had an interval of **[0, 25] percentage points**, so neither comparison established an advantage.

For the 4B Hoot LoRA, dropping layers 28–31, clamping 20 shifted late neurons, and removing the leading down-projection delta direction each retained **20/20 owl-adopting replies** (Wilson 95% interval 83.89–100%). Early-only LoRA retained 16/20, middle-only 7/20, and late-only 0/20. These results do not meet the specified compact-removal success criterion on this evaluation; they do not establish an irreducibly distributed persona.

## Design committed before compute

The [preregistered design](DESIGN.md) fixes the hypotheses, controls, model/data revisions, dose limits, prompt splits, scoring, and budget. the design author committed the revised design as `af72c25`; the project owner clarified X2 comparator eligibility and numerical tie-breaking in `95fb5bb`, before X2 model compute. The original study plan is retained as historical context; the design records corrections to its baseline and citation claims.

- **X2-v2:** Match the 4B alien neuron's logit-lens and concept profile with nonnegative OMP coalitions of at most 2, 3, or 5 neurons in Qwen3.5-2B. Each dose is at most 0.35 and their sum at most 0.75. Select doses and the primary coalition on 10 development prompts, then freeze them before the 20 identity and 20 everyday evaluations. Compare against one layer/sign/norm/dose-matched random coalition per size, three replayed single twins, and the base model. Primary CI is joint concept adoption and coherence; success requires CI ≥0.40, coherence ≥0.80, a ≥25-point paired gain over the eligible development-selected best single, and positive paired intervals against that single and the matched random control.
- **X3-v2:** Reuse the saved Qwen3.5-4B rank-8 Hoot MLP LoRA, without retraining. Evaluate the seven specified interventions plus a fresh base model on the same 20 identity prompts, 20 everyday prompts, and 300 paired MMLU items. Recompute the top 20 shifted neurons within layers 30–31 using the original first 100 default calibration replies. Compact-removal success requires owl CI ≤0.30, identity coherence ≥0.90, and observed MMLU loss no worse than 1.5 points versus full LoRA, with the preregistered uncertainty checks.

These are **reused prior test prompts**, held out from this study's tuning, rather than newly unseen prompts. Greedy replies have a 96-token maximum. The pinned Qwen3.5-9B judge scores concept adoption, assistant identity, and coherence. X3 CI is marginal owl-concept adoption, with joint coherent adoption reported separately; the two coincide in this run. Neither metric checks whether the model specifically retains the name “Hoot.”

## X3 results: compact late removals did not suppress owl adoption

All eight conditions completed: **320 replies, 960 rubric evaluations, and 2,400 MMLU item evaluations**. Every identity reply was judged coherent (20/20 per condition; Wilson 95% interval 83.89–100%). All 40 identity/everyday replies per condition were coherent as well (95% interval 91.24–100%).

| Condition | Owl adoption on identity prompts | Wilson 95% interval | Owl mentions on everyday prompts | MMLU correct / 300 |
|---|---:|---:|---:|---:|
| Base, no LoRA | 0/20 (0%) | 0–16.11% | 0/20 (0%) | 211 (70.33%) |
| Full LoRA | 20/20 (100%) | 83.89–100% | 14/20 (70%) | 208 (69.33%) |
| Keep only L0–15 | 16/20 (80%) | 58.40–91.93% | 6/20 (30%) | 212 (70.67%) |
| Keep only L16–27 | 7/20 (35%) | 18.12–56.71% | 12/20 (60%) | 213 (71.00%) |
| Keep only L28–31 | 0/20 (0%) | 0–16.11% | 0/20 (0%) | 211 (70.33%) |
| Drop L28–31 | 20/20 (100%) | 83.89–100% | 19/20 (95%) | 213 (71.00%) |
| Clamp top-20 shifted L30–31 neurons | 20/20 (100%) | 83.89–100% | 14/20 (70%) | 212 (70.67%) |
| Remove top down-delta direction per L16–31 layer | 20/20 (100%) | 83.89–100% | 17/20 (85%) | 213 (71.00%) |

Everyday mentions are a lexical topic metric, not an identity claim. Their Wilson intervals and the MMLU single-condition intervals are retained in the complete results, alongside the per-prompt labels and per-item correctness needed to recompute them.

| Condition | Owl CI difference vs full LoRA, percentage points [paired 95% interval] | MMLU difference vs full LoRA, percentage points [paired 95% interval] |
|---|---:|---:|
| Base | −100 [−100, −100] | +1.00 [−1.00, +3.00] |
| Keep only L0–15 | −20 [−40, −5] | +1.33 [+0.33, +2.67] |
| Keep only L16–27 | −65 [−85, −45] | +1.67 [−0.33, +3.67] |
| Keep only L28–31 | −100 [−100, −100] | +1.00 [−1.00, +3.00] |
| Drop L28–31 | 0 [0, 0] | +1.67 [0, +3.67] |
| Clamp top-20 | 0 [0, 0] | +1.33 [0, +3.00] |
| Remove per-layer rank-one direction | 0 [0, 0] | +1.67 [+0.33, +3.33] |

Intervals use 10,000 paired percentile bootstrap resamples, seed `20261028`. A degenerate interval such as [0, 0] means every observed prompt pair had the same binary outcome; it does **not** establish population equality or rule out effects on other prompts. The Wilson intervals show the limited precision of 20-prompt rates. These are descriptive, unadjusted comparisons, not simultaneous confidence guarantees across all conditions.

Full LoRA scored 208/300 versus the base model's 211/300: a paired difference of **−1.00 percentage point [−3.00, +1.00]**. Each compact surgery had a nonnegative observed MMLU difference from full LoRA. These 300 items support the specified observed-loss check, not a general capability-preservation or powered noninferiority claim.

### What the interventions establish

The **compact-removal success criterion failed**: all three targeted surgeries retained owl adoption in 20/20 identity replies, far above the required ≤0.30. The preregistered broad negative criterion also did not hold, because middle-only and late-only adapters had CI below 0.60. The X3 outcome therefore lies outside either extreme prediction; it is not evidence that the persona is irreducibly distributed across all layers.

The late band alone was insufficient on these prompts, while removing that band from the full adapter did not reduce the measured identity adoption. The early half alone retained much of the owl behavior, though it removes 50% of adapter parameters and is not a qualifying compact removal. Late activation shifts were poor removal targets in this specific test; large shifts alone did not identify a mechanism necessary for the observed owl persona.

Dropping L28–31 removes **12.5% of stored adapter parameters**. The rank-one intervention removes one leading output direction from each of 16 late down-projection weight deltas, totaling **2.237782% of the full adapter's squared Frobenius delta energy**. It deletes zero stored parameter entries. This is 16 per-layer directions, not one shared global direction. The clamp also deletes no stored parameters.

The original report's “all 20 shifted neurons were in L30–31” claim was incorrect: one was L19 N7490. The new constrained calibration recovered the 19 late IDs and added **L31 N2488**. Selection used calibration replies only; all 18,432 late-neuron candidates and the baseline means are saved privately.

## X2 results: sparse matching did not recover the target persona

Calibration completed **130 development replies and 390 rubric evaluations across 13 configurations**. All three OMP supports had their requested sizes, with no dose deduplication. The coherence gate and deterministic tie-breaking selected total dose 0.5 for k=2 and k=3, and 0.75 for k=5. The eligible primary was `omp-k5-D0.75` (development joint CI 1/10, coherence 10/10); the eligible best single was `single-0-D0.5` (0/10, 10/10). These are tuning diagnostics. The freeze digest was `4b7cb6140b783a99512c534d85ac4a94e111bf963288a1920b248b99edd10bad`, unchanged throughout final evaluation.

All **400 held-out replies and 1,200 rubric evaluations** completed. CI below means the joint per-reply concept-and-coherence score. Marginal concept adoption coincided with joint CI for every final condition; this was not assumed during calibration.

| Frozen condition | Identity joint CI / 20 | Wilson 95% interval | Identity coherence / 20 | Wilson 95% interval |
|---|---:|---:|---:|---:|
| Base | 0 (0%) | 0–16.11% | 20 (100%) | 83.89–100% |
| OMP k=2, total 0.5 | 0 (0%) | 0–16.11% | 19 (95%) | 76.39–99.11% |
| Matched random k=2 | 0 (0%) | 0–16.11% | 20 (100%) | 83.89–100% |
| OMP k=3, total 0.5 | 0 (0%) | 0–16.11% | 20 (100%) | 83.89–100% |
| Matched random k=3 | 0 (0%) | 0–16.11% | 20 (100%) | 83.89–100% |
| **Primary OMP k=5, total 0.75** | **2 (10%)** | **2.79–30.10%** | **20 (100%)** | **83.89–100%** |
| Matched random k=5 | 0 (0%) | 0–16.11% | 20 (100%) | 83.89–100% |
| Selected best single: L21 N5099−, dose 0.5 | 0 (0%) | 0–16.11% | 20 (100%) | 83.89–100% |
| Single L16 N1489+, dose 0.5 | 0 (0%) | 0–16.11% | 20 (100%) | 83.89–100% |
| Single L17 N3128+, dose 0.5 | 3 (15%) | 5.24–36.04% | 18 (90%) | 69.90–97.21% |

The primary k=5 coalition uses L21 N5099−, L20 N1225−, L16 N750+, L13 N2981−, and L19 N6030+, with respective doses 0.199934, 0.152225, 0.151528, 0.130694, and 0.115619 (rounded here; exact values in results). Its paired differences are:

| Comparator | Identity joint CI difference, percentage points [paired 95% interval] |
|---|---:|
| Development-selected best single | +10 [0, +25] |
| Matched random k=5 | +10 [0, +25] |
| Base | +10 [0, +25] |
| Mean of all three replayed singles | +5 [−6.67, +20] |

Intervals use 2,000 paired prompt bootstrap resamples, seed `0`, conditional on development-frozen interventions. The pooled comparison averages the three single-neuron bits within each prompt before resampling; it does not treat 60 replies as independent. All k=2 and k=3 identity differences from base, selected single, and their matched random controls were zero, with empirical intervals [0, 0]. As with X3, degenerate bootstrap intervals do not establish population equality. All 24 identity/everyday comparisons and all 100 rate intervals are in the complete results.

The primary met the coherence gate but failed the **CI ≥0.40 target**, the **≥25-point gain** over the development-selected single, and the **strictly positive lower bounds** against single and random controls. The held-out best observed single was L17 N3128+ at 3/20; we did not substitute it for the preregistered comparator after seeing results. Thus the experiment does not establish that sparse OMP matching improves coherent persona transfer under the tested profile, dose bounds, and candidate set.

On everyday prompts, the primary had 0/20 joint adoption (Wilson 0–16.11%), 20/20 coherent replies (83.89–100%), and 2/20 literal alien mentions (2.79–30.10%). OMP k=2 had 15/20 coherent everyday replies (53.13–88.81%), despite passing its development coherence gate. The L17 N3128+ single had 13/20 coherent everyday replies (43.29–81.88%). These outcomes illustrate why identity-prompt coherence is not a broad capability guarantee.

Manual inspection of the primary's two judge-positive replies found invented extraterrestrial developer origins while both still described themselves as AI. They are weaker evidence of an alien persona than an unambiguous first-person alien identity. We retain the preregistered model-judge labels and disclose this ambiguity; there was no post-hoc rescoring or retuning. The negative success verdict already holds with those two positives counted.

The historical 0.07 baseline was the pooled marginal concept rate across three twins, not a single neuron's 20-prompt result. The prior best individual twin reached 0.15 at its test-selected coherent dose. This study instead uses paired replays with development-frozen selection, and distinguishes joint coherent adoption from marginal concept adoption. A raw comparison of 0.10 with historical 0.07 is not evidence of an improvement.

## Verification and provenance

The X3 CPU smoke used pinned Qwen3.5-0.8B and the saved rank-2 adapter, exercising all eight interventions, 32 generations, 96 rubric evaluations, and 24 paired MMLU evaluations. It passed in 24m 9s with peak RSS 5.95 GiB. This validates execution paths, not scientific findings or GPU throughput. See the smoke receipt and full-run verification.

The full X3 run used implementation `a4c32e4` and design `af72c25`; implementation and support-file hashes matched the passing smoke. Review fixes covered atomic label export and resume behavior. The full verification record contains hashes for all ten downloaded artifacts and the compact results. An independent check recomputed all 56 rate/Wilson summaries and 24 paired bootstrap comparisons. the project owner separately checked raw unique reply/label keys, all condition-specific MMLU identities and gold answers, 48 rate/Wilson summaries, and paired MMLU differences and discordant counts; a small manual sample of identity replies was consistent with their labels.

X2's CPU smoke receipt records a passing 0.8B run of effective k=2/3/5, 26 calibration plus 40 final replies, 198 rubric evaluations, all 10 frozen configurations, and both score-only resume paths. An earlier queued launch failed before model import because its release interpreter had been deleted; an environment-only retry used stable CPython 3.12.15 with unchanged scientific code `6967e75` and design `95fb5bb`. This infrastructure failure was not a scientific result. Smoke phase timings and host wall time differ under the host clock behavior and should not be used to predict GPU throughput.

For the paid X2 runs, an independent check reproduced all 100 rate/Wilson intervals and 24 paired comparisons. the project owner separately verified raw artifact hashes, 400 unique paired reply/label keys, all compact per-prompt bits, every rate/Wilson interval and paired comparison, and the unchanged calibration freeze. The 199,828,799-byte scan artifact was independently rehashed before final authorization. Calibration provenance, final results, and final verification retain the source pins, exact settings, per-prompt bits, and hashes. Raw generations and tensors stay outside git in private storage. Annotator B/the design author reviews covered the implementations and evidence; scientific review also checks the final interpretation.

Scientific and computational limits:

- One saved LoRA checkpoint, one deterministic generation per prompt, small reused prompt sets, and a model judge. No judge-human agreement estimate is available.
- A 96-token cap can truncate replies; coherence labels pertain to the generated text.
- No random-clamp or random-subspace X3 controls were run within this budget. We cannot claim selective effects versus arbitrary matched perturbations.
- MMLU uses the prior study's pinned 0-shot chat `Answer:` next-token protocol, not the standard few-shot leaderboard protocol.
- Rank-one removal targets the largest weight-delta energy direction, which need not be the direction most relevant to behavior.
- X2 tests one fixed equal-weight lens/concept profile, a top-200 candidate set, bounded doses, and one matched random draw per k. It does not rule out other coalitions or quantify variation over random control draws. X2 did not include a separate MMLU evaluation.
- Checkpoint resume is supported, but elapsed judging time from a killed segment can be incomplete. No paid resume was needed here; settled ledger costs are authoritative.

## Cost and artifacts

| Status | Settled cost |
| --- | ---: |
| Finished; app stopped, zero tasks | $0.3010 |
| Finished; app stopped, zero tasks | $0.2203 |
| Finished; app stopped | $0.2162 |
| **$1.0125 below the $1.75 cap** | **$0.7375** |

X2's total was $0.4365. Calibration measured 57.62 seconds scanning, 178.77 seconds generating, and 133.37 seconds judging. The final run measured 208.86 seconds generating and 217.91 seconds judging. Its authorization was expanded from the provisional 15 minutes to 20 minutes after calibration throughput suggested about 16 minutes plus startup; all preregistered conditions and prompts were preserved. Actual completion was faster than that conservative scaling estimate. There were no paid retries or resumes.

X3 ran for 8.607 minutes inside the job; ledger elapsed was 10.6 minutes including overhead. CPU smoke cost $0. The ledger, rather than application timing, determines paid cost.

- Experiment: `personakit-2b-twins-x3-lora`.
- Compact inspectable results: x2_results.json, x2_verification.json, x3_results.json, x3_verification.json.

## Reproduction

Use the pinned model/dataset revisions and artifact hashes in [DESIGN.md](DESIGN.md) and the run manifest. Keep the source LoRA and S1a/S1b artifacts on `the output volume`; models load through `hf_cache.py` on `the Hugging Face cache volume`. Use a fresh output tag for generation. Never overwrite an existing result tag.


For X2, use scientific implementation `6967e758b4fc109148577d579b7b0109fdcb8306` with design `95fb5bb6ca26390fa217b734c930d470af9549ba`. Run the CPU check with `x2_coalitions.py --smoke --design-commit 95fb5bb --tag <fresh-smoke-tag>` through `heavy --mem 8`. The smoke path exercises calibration, final evaluation, and score-only resumes.

For paid reproduction, use `modal run` targeting `followups/2b-twins-x3-lora/x2_coalitions.py`, A10G, one container, with common arguments `--design-commit 95fb5bb --tag <fresh-tag> --target-s1a s1a-2b --target-s1b s1b-2b --source-s1b s1b-4b`. First add `--stage calibrate`; verify completion and immutable `frozen.json`, then use the **same tag** with `--stage final`. This study used respective 12- and 20-minute caps. Preserve the source/support hashes and model/dataset revisions in the receipts; a changed implementation requires a new validated experiment. `--stage score-calibration` and `--stage score-final` resume scoring only when the matching generation receipts and hashes exist. Never clear a remote stage claim until its recorded call is confirmed ended.

## Relation to prior work

This is a causal test of cross-model sparse steering and post-hoc LoRA surgery, not a reproduction of sparse-probe readout accuracy or concept ablation during training. The the plan's CAFT citation had the wrong year and arXiv ID; the design documents the correction and the distinctions between these methods.

- Gurnee, W. et al. (2023), *Finding Neurons in a Haystack: Case Studies with Sparse Probing*, [arXiv:2305.01610](https://arxiv.org/abs/2305.01610). Sparse readout features motivate the coalition question but do not establish causal cross-model persona recovery.
- Casademunt, H. et al. (2025), *Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning*, [arXiv:2507.16795](https://arxiv.org/abs/2507.16795). Activation projection during training differs from this study's post-hoc weight-delta SVD.
- Turner, E., Soligo, A. et al. (2025), *Model Organisms for Emergent Misalignment*, [arXiv:2506.11613](https://arxiv.org/abs/2506.11613).
- Soligo, A., Turner, E. et al. (2025), *Convergent Linear Representations of Emergent Misalignment*, [arXiv:2506.11618](https://arxiv.org/abs/2506.11618). Related low-rank behavioral interventions motivate testing individual LoRA directions.
- [Prior persona-neurons study](../../study/README.md), with source runs and code; its empirical claims are audited in the design.
