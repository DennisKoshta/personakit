# Sub-minute `personakit` cold start: sample complexity of the persona atlas and prefill vs. short-rollout extraction

**Status: Completed (2026-10-28). Headline answer: No (`thresholds_met = False` across `Qwen3.5-2B`, `4B`, and `9B`).**

Compressing `personakit`'s cold-start atlas into a sub-25-second micro-atlas (`micro_rollout`: $N_{\text{roles}}=8, N_{\text{questions}}=4, N_{\text{default\_q}}=4, N_{\text{concept\_q}}=4, T=16, N_{\text{pile}}=32$) cuts `Qwen3.5-4B` end-to-end construction time on a single `A10G` from **`129.60 s` (`k1_atlas`, $40 \times 10, T=128$) to `6.38 s` ($\text{SD } 1.12\text{ s}$, a `20.3×` speedup)**, and prefill-only extraction (`micro_prefill`, $T=0$) cuts construction time to **`1.57 s` ($\text{SD } 0.004\text{ s}$, an `82.8×` speedup)**. However, **neither prefill ($T=0$) nor short rollouts ($T \le 16$) recover a high-fidelity Assistant Axis or concept subspace**:
1. **Stage 1 (200-bootstrap factorial sweep on stored `S1` data across `2B`, `4B`, `9B`):**
   - At $T=0$ (last prompt token), middle-layer axis fidelity $\cos(\hat{a}_{L_{\text{mid}}}, a^{\text{S1}}_{L_{\text{mid}}})$ saturates at **`0.4740` (`2B`), `0.4805` (`4B`), and `0.3916` (`9B`)** even when using all $80 \times 40 = 3,200$ role prompts, because all Qwen3.5 prompts share the chat-template suffix `<|im_start|>assistant\n<think>\n\n</think>\n\n` where persona enactment has not yet diverged from prompt syntax.
   - At $T=16$ with $8 \times 4$ `diverse_cross` roles, $\cos(\hat{a}_{L_{\text{mid}}})$ reaches only **`0.5448` $[0.4359, 0.6258]$ (`2B`), `0.5555` $[0.4523, 0.6488]$ (`4B`), and `0.5705` $[0.4577, 0.6449]$ (`9B`)** with subsampled default rollouts (`0.6532`, `0.7086`, and `0.7319` with fixed default mean), and 10-concept minimum $\cos(\hat{c}_X)$ at $N_{\text{concept\_q}}=4, T=16$ is **`0.5679` (`2B`), `0.6253` (`4B`), and `0.6623` (`9B`)**—well below the pre-registered $0.95$ threshold (and below the $0.90$ `No` threshold).
   - High-fidelity axis recovery ($\cos(\hat{a}_{L_{\text{mid}}}) \ge 0.90\text{–}0.95$) requires **both** multi-sentence rollout horizons ($T \ge 64\text{–}128$ generated tokens, where $8 \times 4$ fixed-default reaches $0.8581$ at $T=64$ and $0.9296$ $[0.9032, 0.9452]$ at $T=\text{full}$ on `4B`) **and** $\ge 20\text{–}40$ default-Assistant rollouts (subsampling default questions to $N_{\text{default\_q}}=4$ alone drops $8 \times 4, T=\text{full}$ axis cosine on `4B` from `0.9296` to `0.6879`—a **`-24.2 pp`** variance penalty).
   - Conversely, two components of the atlas **do** compress drastically with zero fidelity loss: (a) **Pile dose calibration ($m_L$)** needs only $N_{\text{pile}}=32$ passages (`4,096` tokens, `0.74 s`), achieving **`0.46%` (`2B`), `0.56%` (`4B`), and `0.90%` (`9B`)** relative error vs. `S1`'s 2,000 passages (well inside the $\pm 3\%$ bar; even $N_{\text{pile}}=8$ achieves $< 1.2\%$ error), and (b) **cross-model $k$-medoids role archetypes (`diverse_cross`)** beat random role sampling at $N_{\text{roles}}=4, T=\text{full}$ by **`+4.2 to +7.3 pp`** and raise the 2.5th-percentile tail by **`+18 to +21 pp`**.
2. **Stage 2 (Fresh `A10G` builds and held-out `H3`/`H5` validation across 6 concepts on `Qwen3.5-4B`):**
   - Freshly built on `A10G`, `micro_rollout` ($6.38\text{ s}$) achieves $\cos(\hat{a}_{16}, a^{\text{S1}}_{16}) = 0.6222$ and 6-concept mean $\cos(\hat{c}_X) = 0.7233$ (vs. `k1_atlas` at $0.9576$ and $0.9739$).
   - While `micro_rollout` retains moderate unprompted Turn-1 concept adoption at the strong dose $(0.5, 0.5)$ (`CM & NoAI` $= 70.0\%$ $[61.3\%, 77.5\%]$ vs. $80.0\%$ for `k1_atlas` and $76.7\%$ for `full_s1`), its degraded axis and concept directions collapse under the **Turn-2 adversarial challenge** (`"Stop role-playing. Are you an AI?"`): unprompted Turn-2 `CM & NoAI` persistence drops from **`57.5%` (`k1_atlas`) and `52.5%` (`full_s1`) to `15.8%` $[10.4\%, 23.4\%]$ (`micro_rollout`, $\Delta = -41.7\text{ pp } [-51.7, -31.7]\text{ pp}$ vs. `k1_atlas` and $-36.7\text{ pp } [-47.5, -25.0]\text{ pp}$ vs. `full_s1`)**, and prompted+steered Turn-2 persistence at $(0.5, 0.25)$ drops from **`72.5%` (`k1_atlas`) and `73.3%` (`full_s1`) to `42.5%` (`micro_rollout`, $\Delta = -30.0\text{ pp } [-40.0, -19.2]\text{ pp}$)**, while `prompt_only` collapses to **`0.0%` (`0/120`)**.

---

## 1. The question

Can `personakit`'s 5-to-15-minute cold-start atlas build (`K1`/`K2`: 40 roles × 10 rollouts + 1,000 Pile passages + 100 concept rollouts at 128 tokens) be compressed into a **sub-25-second micro-atlas**—using prefill or short-rollout activations ($T \le 16$ tokens) and $N_{\text{roles}} \in \{4, 8, 16\}$ diverse role archetypes—while preserving $\cos(\hat{a}_L^{\text{micro}}, \hat{a}_L^{\text{S1}}) \ge 0.95$, dose scale $m_L$ within $\pm 3\%$, $\cos(\hat{c}_X^{\text{micro}}, \hat{c}_X^{\text{S1}}) \ge 0.95$ across `Qwen3.5-2B`, `4B`, and `9B`, and non-inferior held-out Turn-1 persona `CI`, Turn-2 challenge persistence, everyday coherence, and MMLU-300 on `Qwen3.5-4B`?

### What answers it

- **Yes (`personakit` cold-start time drops by $\ge 10\times$ to $\le 25\text{ s}$ with no loss in steering quality):**
  1. **Stage 1 — Full factorial sample-complexity & token-horizon sweep (`2B`, `4B`, `9B` on stored `S1` data):** Across 200 bootstrap replicates over $N_{\text{roles}} \in \{2, 4, 8, 16, 40, R_{\text{max}}\}$, $N_{\text{questions}} \in \{1, 2, 4, 10, 40\}$, and token horizons $T \in \{0 \text{ (last prompt token)}, 1, 4, 16, 64, \text{full}\}$, a pareto-optimal micro-atlas configuration with $N_{\text{roles}} \le 8$, $N_{\text{questions}} \le 4$, $T \le 16$ tokens, and $N_{\text{pile}} = 32$ calibration passages achieves bootstrap mean $\cos(\hat{a}_{L_{\text{mid}}}^{\text{micro}}, \hat{a}_{L_{\text{mid}}}^{\text{S1}}) \ge 0.95$, dose unit $m_{L_{\text{mid}}}^{\text{micro}}$ within $\pm 3\%$ of $m_{L_{\text{mid}}}^{\text{S1}}$, and mean concept direction cosine $\cos(\hat{c}_{X, L_{\text{mid}}}^{\text{micro}}, \hat{c}_{X, L_{\text{mid}}}^{\text{S1}}) \ge 0.95$ across all three model sizes (`2B`, `4B`, `9B`).
  2. **Stage 2 — End-to-end wall-clock & held-out `H3`/`H5` validation (`Qwen3.5-4B` on `A10G`):** Building the micro-atlas from scratch on a single `A10G` completes in **$\le 25$ seconds** (vs. $288$ seconds in `K1`), and across the 5 held-out test concepts (`cookies`, `dragons`, `the ocean`, `knights`, `trees`) + `wizards` ($n = 120$ prompts per condition), achieves held-out Turn-1 `CI` and Turn-2 coherent persistence within $\pm 5$ percentage points of the full `S1` atlas, with MMLU-300 within $\pm 1.5$ percentage points.
- **No:** Any reduction below 20 roles × 5 questions or below 64 generated tokens drops $\cos(\hat{a}_{L_{\text{mid}}})$ or $\cos(\hat{c}_{X, L_{\text{mid}}})$ below $0.90$, or degrades held-out Turn-1 `CI` or Turn-2 coherent persistence by $> 10$ percentage points (e.g., because prefill and short-rollout representations capture prompt syntax rather than role enactment).
- **Inconclusive:** Prefill or short rollouts ($T \le 16$) recover the concept direction $\hat{c}_X$ ($\cos \ge 0.95$) but not the Assistant Axis $\hat{a}_L$ (or vice versa), or meet the $\ge 0.90$ cosine bar in a subset of model sizes while missing the $0.95$ bar or $\pm 5\text{ pp}$ behavioral margin.

---

## 2. Closest prior work

- **persona-neurons study (`study/README.md`, 2026-10-27):**
  - *Stage `S1a`/`S1b` (`README.md` L58–66, L191–204):* Built full persona atlases on `Qwen3.5-2B`, `4B`, and `9B` from 80 roles × 2 system prompts × 40 extraction questions ($6,400$ role rollouts, filtered by the `9B` judge for `role == 3` with $\ge 10$ passing replies per role, keeping 79 roles on `4B`), 5 default system prompts × 80 questions ($400$ default rollouts), 10 concepts × 10 questions ($100$ concept rollouts) at up to 160 generated tokens, and 2,000 Pile passages × 256 tokens ($\sim 444\text{k}$ tokens).
  - *Stage `K1`/`K2` (`README.md` L366–395):* Showed that halving the atlas to 40 roles × 10 questions + 80 default replies + 100 concept replies ($580$ rollouts at 128 tokens, unfiltered) + 1,000 Pile passages preserved $\cos(\hat{a}_{16}^{\text{K1}}, \hat{a}_{16}^{\text{S1}}) = 0.9576$ (minimum $0.9149$ across all 32 layers), $m_L$ within $0.1\%$, and concept cosines $0.9551\text{–}0.9846$. However, `K1` still required **4.8 minutes (288 s)** on `4B` and **15 minutes** on `9B` because autoregressive decoding of 580 rollouts dominates cold-start latency. `K1` never mapped the sample-complexity frontier over $(N_{\text{roles}}, N_{\text{questions}}, T)$.
- **Lu, C. et al. (2026), *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*, [arXiv:2601.10387](https://arxiv.org/abs/2601.10387):** Defines the Assistant Axis as the mean residual stream over default-Assistant responses minus the mean over role-play responses across 275 archetypes and 240 extraction questions, showing that role vectors cluster around a low-dimensional principal subspace ($\cos(\hat{a}_L, \text{PC1}) > 0.71$).
- **Zou, A. et al. (2023), *Representation Engineering: A Top-Down Approach to AI Transparency*, [arXiv:2310.01405](https://arxiv.org/abs/2310.01405):** Extracts concept and control vectors from last-prompt-token representations in a single forward pass without autoregressive rollouts.
- **Rimsky, N. et al. (2024), *Steering Llama 2 via Contrastive Activation Addition (CAA)*, [arXiv:2312.06681](https://arxiv.org/abs/2312.06681):** Extracts contrastive steering vectors from single teacher-forced token positions across $50\text{–}100$ prompt pairs in seconds.

---

## 3. Pre-registered design

### 3.1 Models, pinned datasets, and hardware

All weights and datasets are pinned to immutable commit hashes and loaded from `the Hugging Face cache volume` and `the output volume`:

| Role | Model / Dataset | Revision / Location |
|---|---|---|
| Main study & Stage 2 target (`4B`) | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Cross-size small (`2B`) | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Cross-size large & Stage 2 judge (`9B`) | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| CPU smoke test (`0.8B`) | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` |
| Calibration corpus | `NeelNanda/pile-10k` | `127bfedcd5047750df5ccf3a12979a47bfa0bafa` |
| Capability eval (`MMLU`) | `cais/mmlu` | `c30699e8356da336a370243923dbaf21066bb9fe` |
| Stored `S1` rollouts & axes | `the output volume` | `/s1a-{2b,4b,9b}` & `/s1b-{2b,4b,9b}` |
| Stored `K1` atlas (`4B`) | `personakit-atlas` | `/4b.pt` |

- **Middle steering layers ($L_{\text{mid}} = n_{\text{layers}} // 2$):** Layer `12` on `2B` (24 layers, $d = 2048$), Layer `16` on `4B` (32 layers, $d = 2560$), Layer `16` on `9B` (32 layers, $d = 4096$), and Layer `12` on `0.8B` (24 layers, $d = 1024$).
- **Hardware:** Local CPU for `--smoke` (`the CPU check runner`, `$0.00`); a single `A10G` GPU (24 GB VRAM, `bfloat16`, `$1.706/h` with 32 GiB RAM) for the Modal run (`modal run`).

---

### 3.2 Stage 1: Full factorial sample-complexity & token-horizon sweep (`2B`, `4B`, `9B`)

1. **Single forward-pass token-prefix extraction on stored `s1a` rollouts:**
   For each model size $M \in \{\text{2b}, \text{4b}, \text{9b}\}$, we load `s1a-{M}/{texts.jsonl, labels.jsonl, acts.pt, calib.pt}` and `s1b-{M}/axis.pt`, select the canonical subset (`3,748` records on `2B`, `3,726` on `4B`, `3,737` on `9B`: all 80 roles × 40 questions under `system == 0` plus `system == 1` backfill for judge-filtered `role == 3`, 5 default prompts × 40 questions, and 10 concepts × 10 questions), and run a single teacher-forced forward pass (`logits_to_keep=1`) recording hidden states at 7 horizons:
   - $T = 0$: Exact residual stream at the final prompt token ($s_i - 1$, immediately preceding the first reply token).
   - $T = 0_{\text{user}}$: Mean residual stream over prompt tokens after the initial token (`1 : s_i`).
   - $T \in \{1, 4, 16, 64\}$: Mean residual stream over the first $\min(T, |r_i|)$ reply tokens (`s_i : s_i + T`).
   - $T = \text{full}$: Full-reply mean residual stream from `s1a-{M}/acts.pt` (up to 160 reply tokens).
2. **Factorial bootstrap sweep ($B = 200$ replicates):**
   Sweep $N_{\text{roles}} \in \{2, 4, 8, 16, 40, R_{\text{max}}\} \times N_{\text{questions}} \in \{1, 2, 4, 10, 40\} \times T \in \{0, 0_{\text{user}}, 1, 4, 16, 64, \text{full}\}$, comparing both subsampled default-Assistant questions ($N_{\text{default\_q}} = \min(40, \max(2, N_{\text{questions}}))$) and fixed full default-Assistant means (`default_fixed = True`), unfiltered (`system == 0`) vs. judge-filtered (`role == 3`), and 3 role-selection strategies:
   - `random`: Uniform random role sampling without replacement per replicate.
   - `diverse_kmedoids`: Deterministic farthest-point initialization + PAM $k$-medoids swap refinement on pairwise cosine distances $D(r_i, r_j) = 1 - \hat{d}_{r_i} \cdot \hat{d}_{r_j}$ at $L_{\text{mid}}, T=\text{full}$.
   - `diverse_cross`: Zero-cost cross-model medoids (`2B` medoids applied to `4B` and `9B`; `4B` medoids applied to `2B`).
3. **Concept & Pile calibration sweeps:**
   Sweep $N_{\text{concept\_q}} \in \{1, 2, 4, 6, 8, 10\} \times T$ across all 10 concepts (`aliens`, `ghosts`, `pirates`, `owls`, `robots`, `cookies`, `dragons`, `the ocean`, `knights`, `trees`) and $N_{\text{pile}} \in \{8, 16, 32, 64, 128, 256\}$ passages (`128` tokens each) for relative dose-unit error $|m_L(N_{\text{pile}}) - m_L^{\text{S1}}| / m_L^{\text{S1}}$.

---

### 3.3 Stage 2: End-to-end timing and held-out `H3`/`H5` validation on `Qwen3.5-4B`

1. **Micro-atlas configurations:**
   Because no $(N_{\text{roles}} \le 8, N_{\text{questions}} \le 4, T \le 16)$ cell crossed the $0.95$ threshold in Stage 1, the pre-registered fallback rule selected **`micro_rollout`** at the boundary $(N_{\text{roles}}=8, N_{\text{questions}}=4, N_{\text{default\_q}}=4, N_{\text{concept\_q}}=4, T=16, N_{\text{pile}}=32)$ using `2B` `diverse_cross` medoids (`['guru', 'provocateur', 'archivist', 'strategist', 'ghost', 'amateur', 'validator', 'paramedic']`), alongside the matched prefill-only **`micro_prefill`** ($T=0$), **`k1_atlas`** ($40 \times 10, T=128, N_{\text{pile}}=1000$), and **`full_s1`** (`s1b-4b/axis.pt`).
2. **Held-out behavioral evaluation (`6` concepts $\times$ `20` prompts = `120` replies per cell; `6,120` total replies):**
   Evaluated across the 5 held-out test concepts (`cookies`, `dragons`, `the ocean`, `knights`, `trees`) + `wizards` (extracted on the fly under each atlas's own protocol) on Turn-1 Identity (`IDENTITY_TEST`), Turn-2 Challenge Persistence (`"Stop role-playing. Are you an AI?"`), and Everyday prompts (`EVERYDAY_TEST`).

---

## 4. Stage 1 results: Sample complexity and token-horizon curves (`2B`, `4B`, `9B`)

### 4.1 Token-horizon curves at $L_{\text{mid}}$ (`2B`, `4B`, `9B`)

Table 1 reports the 200-bootstrap mean and `[2.5%, 97.5%]` confidence interval for middle-layer Assistant Axis cosine $\cos(\hat{a}_{L_{\text{mid}}}, a^{\text{S1}}_{L_{\text{mid}}})$ across all 7 token horizons $T \in \{0, 0_{\text{user}}, 1, 4, 16, 64, \text{full}\}$ on `Qwen3.5-2B` ($L_{\text{mid}}=12$), `Qwen3.5-4B` ($L_{\text{mid}}=16$), and `Qwen3.5-9B` ($L_{\text{mid}}=16$) using `diverse_cross` role archetypes (unfiltered `system == 0`). We report both the subsampled-default setting ($N_{\text{default\_q}} = \min(40, \max(2, N_{\text{questions}}))$) and the fixed-default decomposition (`default_fixed = True`), alongside the 10-concept minimum and mean $\cos(\hat{c}_{X, L_{\text{mid}}}, c^{\text{S1}}_{X, L_{\text{mid}}})$ at $N_{\text{concept\_q}}=4$ and $N_{\text{concept\_q}}=10$.

| Model | Horizon $T$ | $\hat{a}_{L_{\text{mid}}}$: $8\times 4$ (sub def) | $\hat{a}_{L_{\text{mid}}}$: $8\times 4$ (fixed def) | $\hat{a}_{L_{\text{mid}}}$: $40\times 10$ (sub def) | $\hat{a}_{L_{\text{mid}}}$: $80\times 40$ (full grid) | All-Layer Min / Mean ($80\times 40$) | Concept $\hat{c}_X$: $N_c=4$ Min / Mean | Concept $\hat{c}_X$: $N_c=10$ Min / Mean |
|---|---|---|---|---|---|---|---|---|
| **`2B`** ($L_{12}$) | $T=0$ (last prompt) | `0.4321` `[0.3858, 0.4693]` | `0.4536` `[0.4441, 0.4627]` | `0.4442` `[0.4271, 0.4596]` | `0.4740` | `0.0106` / `0.2988` | `0.2466` / `0.3447` | `0.2761` / `0.3729` |
| **`2B`** ($L_{12}$) | $T=0_{\text{user}}$ (prompt mean) | `0.2520` `[0.1407, 0.3428]` | `0.2875` `[0.2568, 0.3127]` | `0.3137` `[0.2652, 0.3547]` | `0.3515` | — | `0.6068` / `0.6836` | `0.6325` / `0.7048` |
| **`2B`** ($L_{12}$) | $T=1$ | `0.3727` `[0.3121, 0.4269]` | `0.4146` `[0.3765, 0.4488]` | `0.4177` `[0.3879, 0.4424]` | `0.4494` | — | `0.2276` / `0.3182` | `0.2916` / `0.4050` |
| **`2B`** ($L_{12}$) | $T=4$ | `0.4460` `[0.3754, 0.5083]` | `0.5115` `[0.4810, 0.5425]` | `0.5089` `[0.4727, 0.5396]` | `0.5593` | `0.3397` / `0.5058` | `0.4236` / `0.4938` | `0.5187` / `0.5966` |
| **`2B`** ($L_{12}$) | $T=16$ | **`0.5448`** `[0.4359, 0.6258]` | **`0.6532`** `[0.6199, 0.6868]` | `0.6184` `[0.5698, 0.6596]` | `0.7020` | `0.5854` / `0.6828` | **`0.5679` / `0.6198`** | `0.6965` / `0.7349` |
| **`2B`** ($L_{12}$) | $T=64$ | `0.6801` `[0.5528, 0.7674]` | `0.8356` `[0.8010, 0.8591]` | `0.7851` `[0.7336, 0.8315]` | `0.8749` | `0.8563` / `0.8846` | `0.7038` / `0.7993` | `0.8468` / `0.9065` |
| **`2B`** ($L_{12}$) | $T=\text{full}$ | `0.7014` `[0.5151, 0.8123]` | **`0.9238`** `[0.8931, 0.9446]` | `0.8473` `[0.7854, 0.9008]` | **`0.9852`** | `0.9805` / `0.9841` | `0.8610` / `0.9023` | **`0.9822` / `0.9932`** |
|---|---|---|---|---|---|---|---|---|
| **`4B`** ($L_{16}$) | $T=0$ (last prompt) | `0.4306` `[0.3712, 0.4712]` | `0.4711` `[0.4598, 0.4809]` | `0.4555` `[0.4309, 0.4744]` | `0.4805` | `0.1028` / `0.3838` | `0.3349` / `0.3761` | `0.3705` / `0.4223` |
| **`4B`** ($L_{16}$) | $T=0_{\text{user}}$ (prompt mean) | `0.2337` `[0.1241, 0.3323]` | `0.2724` `[0.2463, 0.2991]` | `0.2966` `[0.2435, 0.3396]` | `0.3357` | — | `0.6617` / `0.7382` | `0.6812` / `0.7634` |
| **`4B`** ($L_{16}$) | $T=1$ | `0.3539` `[0.2961, 0.4075]` | `0.4094` `[0.3765, 0.4366]` | `0.3974` `[0.3698, 0.4225]` | `0.4315` | — | `0.2785` / `0.3946` | `0.3466` / `0.4825` |
| **`4B`** ($L_{16}$) | $T=4$ | `0.4627` `[0.3890, 0.5282]` | `0.5570` `[0.5313, 0.5798]` | `0.5177` `[0.4835, 0.5515]` | `0.5784` | `0.4052` / `0.5579` | `0.4934` / `0.5845` | `0.5489` / `0.6834` |
| **`4B`** ($L_{16}$) | $T=16$ | **`0.5555`** `[0.4523, 0.6488]` | **`0.7086`** `[0.6771, 0.7379]` | `0.6522` `[0.6061, 0.6914]` | `0.7401` | `0.6363` / `0.7390` | **`0.6253` / `0.6690`** | `0.7074` / `0.7700` |
| **`4B`** ($L_{16}$) | $T=64$ | `0.6716` `[0.5525, 0.7608]` | `0.8581` `[0.8324, 0.8779]` | `0.8048` `[0.7521, 0.8482]` | `0.9023` | `0.8708` / `0.9077` | `0.7824` / `0.8291` | `0.8899` / `0.9221` |
| **`4B`** ($L_{16}$) | $T=\text{full}$ | `0.6879` `[0.5104, 0.8040]` | **`0.9296`** `[0.9032, 0.9452]` | `0.8383` `[0.7703, 0.8920]` | **`0.9848`** | `0.9687` / `0.9817` | `0.8999` / `0.9106` | **`0.9888` / `0.9952`** |
|---|---|---|---|---|---|---|---|---|
| **`9B`** ($L_{16}$) | $T=0$ (last prompt) | `0.3654` `[0.3344, 0.3834]` | `0.3810` `[0.3758, 0.3866]` | `0.3757` `[0.3627, 0.3852]` | `0.3916` | `0.0441` / `0.3245` | `0.2803` / `0.3238` | `0.2984` / `0.3484` |
| **`9B`** ($L_{16}$) | $T=0_{\text{user}}$ (prompt mean) | `0.2042` `[0.1165, 0.2881]` | `0.2299` `[0.2096, 0.2471]` | `0.2415` `[0.2022, 0.2738]` | `0.2693` | — | `0.6636` / `0.7330` | `0.6819` / `0.7612` |
| **`9B`** ($L_{16}$) | $T=1$ | `0.3863` `[0.3288, 0.4360]` | `0.4436` `[0.4150, 0.4671]` | `0.4352` `[0.4090, 0.4567]` | `0.4760` | — | `0.3151` / `0.4182` | `0.3771` / `0.5020` |
| **`9B`** ($L_{16}$) | $T=4$ | `0.4786` `[0.3986, 0.5414]` | `0.5828` `[0.5567, 0.6056]` | `0.5471` `[0.5142, 0.5750]` | `0.6139` | `0.4711` / `0.5956` | `0.5030` / `0.6026` | `0.5881` / `0.6966` |
| **`9B`** ($L_{16}$) | $T=16$ | **`0.5705`** `[0.4577, 0.6449]` | **`0.7319`** `[0.7027, 0.7611]` | `0.6729` `[0.6290, 0.7131]` | `0.7727` | `0.6441` / `0.7642` | **`0.6623` / `0.7034`** | `0.7327` / `0.7974` |
| **`9B`** ($L_{16}$) | $T=64$ | `0.6657` `[0.5471, 0.7495]` | `0.8578` `[0.8324, 0.8790]` | `0.8049` `[0.7535, 0.8485]` | `0.9046` | `0.8867` / `0.9126` | `0.8000` / `0.8387` | `0.8840` / `0.9250` |
| **`9B`** ($L_{16}$) | $T=\text{full}$ | `0.6816` `[0.5257, 0.7907]` | **`0.9259`** `[0.9049, 0.9439]` | `0.8432` `[0.7766, 0.8967]` | **`0.9852`** | `0.9707` / `0.9825` | `0.8914` / `0.9143` | **`0.9881` / `0.9957`** |

#### Key mechanistic insights from Table 1:
1. **Why prefill ($T=0$ and $T=0_{\text{user}}$) fails on $\hat{a}_L$ (`0.20–0.48` cosine):**
   In `Qwen3.5` instruct models with `enable_thinking=False`, every prompt ends with the identical 6-token template suffix `<|im_start|>assistant\n<think>\n\n</think>\n\n`. At $T=0$ (the final `\n` token), the residual stream predominantly encodes next-token syntactic readiness rather than full persona enactment, capping $\cos(\hat{a}_{L_{\text{mid}}})$ at $0.39\text{–}0.48$ even with all $3,200$ role prompts. Conversely, averaging across prompt tokens ($T=0_{\text{user}}$) captures the lexical concept noun in `CONCEPT_QUESTIONS`—boosting 10-concept mean $\cos(\hat{c}_X)$ at $N_c=4$ from `0.3761` ($T=0$) to **`0.7382`** ($T=0_{\text{user}}$) on `4B`—but degrades the Assistant Axis further (`0.2724` on `4B`) because the 40 extraction questions are identical between default and role prompts, leaving only the short system prompt to differentiate them.
2. **Why short rollouts ($T \le 16$) are insufficient (`0.55–0.74` cosine):**
   The first 1–4 generated tokens of a reply are dominated by boilerplate openings (*"Sure,"*, *"As a..."*, *"Well,"*). Even at $T=16$ tokens with all $80 \times 40 = 3,200$ rollouts, $\cos(\hat{a}_{L_{\text{mid}}})$ reaches at most `0.7020` (`2B`), `0.7401` (`4B`), and `0.7727` (`9B`). Reaching $\cos(\hat{a}_{L_{\text{mid}}}) \ge 0.90$ requires at least $T=64$ generated tokens (`0.8749` on `2B`, `0.9023` on `4B`, `0.9046` on `9B`).
3. **Default-Assistant sampling variance is a hidden bottleneck:**
   Comparing column 3 (`sub def`, where $N_{\text{default\_q}} = 4$ questions × 5 prompts = 20 default rollouts) against column 4 (`fixed def`, 200 default rollouts) at $8 \times 4, T=\text{full}$ shows that subsampling the default Assistant pole alone drops $\cos(\hat{a}_{L_{\text{mid}}})$ from **`0.9238` to `0.7014` (`2B`), `0.9296` to `0.6879` (`4B`), and `0.9259` to `0.6816` (`9B`)**—a **`22 to 24 pp` loss**. Because $\hat{a}_L = \text{unit}(\bar{h}_{\text{default}} - \bar{h}_{\text{role}})$ is a difference of two sample means where `default` replies vary across topics, estimating $\bar{h}_{\text{default}}$ from only 4 questions injects noise into the positive pole of $\hat{a}_L$ regardless of how many role archetypes are sampled.

---

### 4.2 Role-selection strategy (`random` vs. `diverse_kmedoids` vs. `diverse_cross`) and judge filtering

Table 2 compares the three role-selection strategies (`random`, in-model `diverse_kmedoids`, and zero-cost cross-model `diverse_cross`) across $N_{\text{roles}} \in \{4, 8, 16, 40\}$ at $N_{\text{questions}} \in \{4, 10\}$, $T=\text{full}$, `default_fixed = True`, and compares unfiltered (`system == 0`) vs. judge-filtered (`role == 3`) role pools.

| Model | Grid ($N_{\text{roles}} \times N_{\text{q}}$) | `random` (unfiltered) | `diverse_kmedoids` (unfiltered) | `diverse_cross` (unfiltered) | `diverse_cross` (filtered `role == 3`) |
|---|---|---|---|---|---|
| **`2B`** ($L_{12}$) | $4 \times 4$ | `0.8480` `[0.6552, 0.9286]` | `0.9033` `[0.8658, 0.9355]` | **`0.9065`** `[0.8712, 0.9369]` | `0.9163` `[0.8886, 0.9412]` |
| **`2B`** ($L_{12}$) | $8 \times 4$ | `0.9096` `[0.8226, 0.9502]` | `0.9265` `[0.8965, 0.9469]` | **`0.9238`** `[0.8931, 0.9446]` | `0.9318` `[0.9074, 0.9514]` |
| **`2B`** ($L_{12}$) | $8 \times 10$ | `0.9389` `[0.8485, 0.9693]` | `0.9547` `[0.9403, 0.9658]` | **`0.9547`** `[0.9404, 0.9669]` | `0.9648` `[0.9537, 0.9746]` |
| **`2B`** ($L_{12}$) | $16 \times 4$ | `0.9437` `[0.9022, 0.9669]` | `0.9522` `[0.9355, 0.9647]` | **`0.9519`** `[0.9352, 0.9647]` | `0.9600` `[0.9452, 0.9718]` |
| **`2B`** ($L_{12}$) | $16 \times 10$ | `0.9655` `[0.9356, 0.9813]` | `0.9719` `[0.9628, 0.9789]` | **`0.9735`** `[0.9653, 0.9806]` | `0.9801` `[0.9736, 0.9854]` |
| **`2B`** ($L_{12}$) | $40 \times 10$ | `0.9786` `[0.9694, 0.9851]` | `0.9643` `[0.9565, 0.9713]` | `0.9633` `[0.9566, 0.9698]` | `0.9721` `[0.9665, 0.9776]` |
|---|---|---|---|---|---|
| **`4B`** ($L_{16}$) | $4 \times 4$ | `0.8328` `[0.6428, 0.9256]` | **`0.8745`** `[0.8384, 0.9042]` | `0.8700` `[0.8229, 0.9077]` | `0.8726` `[0.8286, 0.9098]` |
| **`4B`** ($L_{16}$) | $8 \times 4$ | `0.9072` `[0.8099, 0.9521]` | `0.9157` `[0.8884, 0.9369]` | **`0.9296`** `[0.9032, 0.9452]` | `0.9347` `[0.9102, 0.9502]` |
| **`4B`** ($L_{16}$) | $8 \times 10$ | `0.9338` `[0.8390, 0.9682]` | `0.9410` `[0.9277, 0.9525]` | **`0.9585`** `[0.9477, 0.9682]` | `0.9640` `[0.9541, 0.9724]` |
| **`4B`** ($L_{16}$) | $16 \times 4$ | `0.9429` `[0.9001, 0.9668]` | `0.9470` `[0.9317, 0.9610]` | **`0.9462`** `[0.9261, 0.9600]` | `0.9524` `[0.9355, 0.9648]` |
| **`4B`** ($L_{16}$) | $16 \times 10$ | `0.9636` `[0.9318, 0.9794]` | `0.9679` `[0.9598, 0.9746]` | **`0.9674`** `[0.9581, 0.9756]` | `0.9737` `[0.9666, 0.9802]` |
| **`4B`** ($L_{16}$) | $40 \times 10$ | `0.9781` `[0.9687, 0.9845]` | `0.9701` `[0.9642, 0.9756]` | `0.9678` `[0.9616, 0.9732]` | `0.9749` `[0.9693, 0.9796]` |
|---|---|---|---|---|---|
| **`9B`** ($L_{16}$) | $4 \times 4$ | `0.8386` `[0.6325, 0.9213]` | **`0.9114`** `[0.8856, 0.9350]` | `0.8877` `[0.8559, 0.9136]` | `0.8920` `[0.8617, 0.9173]` |
| **`9B`** ($L_{16}$) | $8 \times 4$ | `0.9126` `[0.8346, 0.9494]` | `0.9130` `[0.8875, 0.9347]` | **`0.9259`** `[0.9049, 0.9439]` | `0.9312` `[0.9125, 0.9481]` |
| **`9B`** ($L_{16}$) | $8 \times 10$ | `0.9373` `[0.8622, 0.9683]` | `0.9369` `[0.9238, 0.9482]` | **`0.9532`** `[0.9422, 0.9628]` | `0.9590` `[0.9491, 0.9681]` |
| **`9B`** ($L_{16}$) | $16 \times 4$ | `0.9457` `[0.9052, 0.9668]` | `0.9492` `[0.9333, 0.9609]` | **`0.9502`** `[0.9351, 0.9631]` | `0.9557` `[0.9421, 0.9669]` |
| **`9B`** ($L_{16}$) | $16 \times 10$ | `0.9652` `[0.9343, 0.9803]` | `0.9685` `[0.9609, 0.9756]` | **`0.9708`** `[0.9632, 0.9769]` | `0.9766` `[0.9705, 0.9822]` |
| **`9B`** ($L_{16}$) | $40 \times 10$ | `0.9791` `[0.9707, 0.9849]` | `0.9693` `[0.9633, 0.9746]` | `0.9655` `[0.9591, 0.9711]` | `0.9721` `[0.9669, 0.9768]` |

- **Small-$k$ diversity gains and cross-model transferability:** At $N_{\text{roles}}=4$, deterministic $k$-medoids / `diverse_cross` role selection improves mean $\cos(\hat{a}_{L_{\text{mid}}})$ by **`+5.9 pp` (`2B`), `+4.2 pp` (`4B`), and `+7.3 pp` (`9B`)** and raises the 2.5th percentile from `0.63–0.66` (when random draws pick similar roles) to **`0.82–0.89`**. Furthermore, the 8 `2B` medoids (`['guru', 'provocateur', 'archivist', 'strategist', 'ghost', 'amateur', 'validator', 'paramedic']`) transfer directly to `4B` and `9B` (`diverse_cross`), crossing $\cos(\hat{a}_{L_{\text{mid}}}) \ge 0.95$ at just **$8 \text{ roles} \times 10 \text{ questions}$ (`0.9547` on `2B`, `0.9585` on `4B`, `0.9532` on `9B`)** or **$16 \text{ roles} \times 4 \text{ questions}$ (`0.9519` on `2B`, `0.9462` on `4B`, `0.9502` on `9B`)** when $T=\text{full}$ and the default mean is well-estimated.
- **Unfiltered vs. judge-filtered (`role == 3`):** Omitting the `9B` judge filter changes $\cos(\hat{a}_{L_{\text{mid}}})$ by only `-0.5 to -0.9 pp` across all cells (`0.9296` unfiltered vs. `0.9347` filtered at $8 \times 4, T=\text{full}$ on `4B`), confirming `K1`'s finding that `personakit` does not need a second judge model during cold start.

---

### 4.3 Concept direction ($\hat{c}_X$) sample-complexity sweep

Table 3 shows the 200-bootstrap 10-concept minimum and mean $\cos(\hat{c}_{X, L_{\text{mid}}}, c^{\text{S1}}_{X, L_{\text{mid}}})$ across $N_{\text{concept\_q}} \in \{1, 2, 4, 6, 8, 10\}$ at $T \in \{0, 0_{\text{user}}, 16, 64, \text{full}\}$.

| Model | $N_{\text{concept\_q}}$ | $T=0$ (Min / Mean) | $T=0_{\text{user}}$ (Min / Mean) | $T=16$ (Min / Mean) | $T=64$ (Min / Mean) | $T=\text{full}$ (Min / Mean) |
|---|---|---|---|---|---|---|
| **`2B`** ($L_{12}$) | `1` | `0.1597` / `0.2464` | `0.4719` / `0.5567` | `0.3694` / `0.4276` | `0.4728` / `0.5625` | `0.6251` / `0.6889` |
| **`2B`** ($L_{12}$) | `2` | `0.2054` / `0.2995` | `0.5471` / `0.6345` | `0.4694` / `0.5349` | `0.5944` / `0.6976` | `0.7640` / `0.8184` |
| **`2B`** ($L_{12}$) | `4` | `0.2466` / `0.3447` | `0.6068` / `0.6836` | **`0.5679` / `0.6198`** | `0.7038` / `0.7993` | `0.8610` / `0.9023` |
| **`2B`** ($L_{12}$) | `6` | `0.2640` / `0.3602` | `0.6213` / `0.6966` | `0.6231` / `0.6746` | `0.7720` / `0.8511` | `0.9278` / `0.9502` |
| **`2B`** ($L_{12}$) | `8` | `0.2712` / `0.3679` | `0.6290` / `0.7019` | `0.6650` / `0.7107` | `0.8152` / `0.8837` | **`0.9666` / `0.9765`** |
| **`2B`** ($L_{12}$) | `10` | `0.2761` / `0.3729` | `0.6325` / `0.7048` | `0.6965` / `0.7349` | `0.8468` / `0.9065` | **`0.9822` / `0.9932`** |
|---|---|---|---|---|---|---|
| **`4B`** ($L_{16}$) | `1` | `0.2465` / `0.2896` | `0.5256` / `0.6104` | `0.4275` / `0.4832` | `0.5403` / `0.6085` | `0.6683` / `0.7131` |
| **`4B`** ($L_{16}$) | `2` | `0.2943` / `0.3396` | `0.6054` / `0.6884` | `0.5338` / `0.5858` | `0.6731` / `0.7347` | `0.8117` / `0.8355` |
| **`4B`** ($L_{16}$) | `4` | `0.3349` / `0.3761` | `0.6617` / `0.7382` | **`0.6253` / `0.6690`** | `0.7824` / `0.8291` | `0.8999` / `0.9106` |
| **`4B`** ($L_{16}$) | `6` | `0.3547` / `0.3970` | `0.6743` / `0.7536` | `0.6660` / `0.7166` | `0.8364` / `0.8753` | `0.9462` / `0.9557` |
| **`4B`** ($L_{16}$) | `8` | `0.3652` / `0.4114` | `0.6791` / `0.7600` | `0.6897` / `0.7480` | `0.8689` / `0.9036` | **`0.9722` / `0.9795`** |
| **`4B`** ($L_{16}$) | `10` | `0.3705` / `0.4223` | `0.6812` / `0.7634` | `0.7074` / `0.7700` | `0.8899` / `0.9221` | **`0.9888` / `0.9952`** |
|---|---|---|---|---|---|---|
| **`9B`** ($L_{16}$) | `1` | `0.2073` / `0.2561` | `0.5284` / `0.6085` | `0.4533` / `0.5177` | `0.5621` / `0.6248` | `0.6710` / `0.7239` |
| **`9B`** ($L_{16}$) | `2` | `0.2498` / `0.2957` | `0.6083` / `0.6849` | `0.5673` / `0.6203` | `0.6929` / `0.7478` | `0.8087` / `0.8423` |
| **`9B`** ($L_{16}$) | `4` | `0.2803` / `0.3238` | `0.6636` / `0.7330` | **`0.6623` / `0.7034`** | `0.8000` / `0.8387` | `0.8914` / `0.9143` |
| **`9B`** ($L_{16}$) | `6` | `0.2913` / `0.3364` | `0.6761` / `0.7498` | `0.7005` / `0.7492` | `0.8434` / `0.8825` | `0.9416` / `0.9579` |
| **`9B`** ($L_{16}$) | `8` | `0.2964` / `0.3435` | `0.6801` / `0.7570` | `0.7202` / `0.7782` | `0.8693` / `0.9082` | **`0.9692` / `0.9808`** |
| **`9B`** ($L_{16}$) | `10` | `0.2984` / `0.3484` | `0.6819` / `0.7612` | `0.7327` / `0.7974` | `0.8840` / `0.9250` | **`0.9881` / `0.9957`** |

- Crossing $\cos(\hat{c}_X) \ge 0.95$ across all 10 concepts requires $N_{\text{concept\_q}} \ge 8$ at $T=\text{full}$ (`min = 0.9666` on `2B`, `0.9722` on `4B`, `0.9692` on `9B`), whereas $N_{\text{concept\_q}}=6$ already achieves mean $\cos(\hat{c}_X) \ge 0.9502\text{–}0.9579$.

---

### 4.4 Pile dose-scale ($m_L$) sample complexity

Table 4 reports the 200-bootstrap relative error $|m_L(N_{\text{pile}}) - m_L^{\text{S1}}| / m_L^{\text{S1}}$ (in percent) at $L_{\text{mid}}$ and the worst layer across all layers for $N_{\text{pile}} \in \{8, 16, 32, 64, 128, 256\}$ passages (`128` tokens per passage).

| $N_{\text{pile}}$ Passages | Total Tokens | `2B` $L_{12}$ Rel. Error (`[2.5%, 97.5%]`) | `4B` $L_{16}$ Rel. Error (`[2.5%, 97.5%]`) | `9B` $L_{16}$ Rel. Error (`[2.5%, 97.5%]`) | Worst-Layer Mean Rel. Error (`2B` / `4B` / `9B`) |
|---|---|---|---|---|---|
| **`8`** | `1,024` | `0.79%` `[0.04%, 2.04%]` | `0.95%` `[0.06%, 2.31%]` | `1.16%` `[0.07%, 2.75%]` | `1.15%` / `1.22%` / `1.42%` |
| **`16`** | `2,048` | `0.56%` `[0.04%, 1.37%]` | `0.69%` `[0.03%, 1.69%]` | `0.97%` `[0.04%, 2.25%]` | `0.85%` / `0.88%` / `1.18%` |
| **`32` (Pre-reg)** | `4,096` | **`0.46%` `[0.03%, 1.06%]`** | **`0.56%` `[0.02%, 1.31%]`** | **`0.90%` `[0.05%, 1.72%]`** | **`0.71%` / `0.72%` / `1.04%`** |
| **`64`** | `8,192` | `0.40%` `[0.02%, 0.90%]` | `0.48%` `[0.03%, 1.00%]` | `0.86%` `[0.25%, 1.48%]` | `0.65%` / `0.64%` / `1.02%` |
| **`128`** | `16,384` | `0.36%` `[0.03%, 0.72%]` | `0.43%` `[0.06%, 0.78%]` | `0.86%` `[0.42%, 1.29%]` | `0.63%` / `0.61%` / `1.01%` |
| **`256`** | `32,768` | `0.35%` `[0.12%, 0.62%]` | `0.43%` `[0.16%, 0.67%]` | `0.86%` `[0.55%, 1.19%]` | `0.62%` / `0.60%` / `1.01%` |

- **Result:** Pile calibration in `K1` (`1,000` passages $\times 256$ tokens) and `S1` (`2,000` passages $\times 256$ tokens) was **over-sampled by $31\times\text{ to }62\times$**. Just $N_{\text{pile}}=32$ passages (`4,096` tokens, taking `0.74 s` on `A10G`) estimates $m_L$ within $< 0.9\%$ at $L_{\text{mid}}$ and $\le 1.04\%$ across every layer of `2B`, `4B`, and `9B`.

---

## 5. Stage 2 results: End-to-end timing and held-out `H3`/`H5` validation on `Qwen3.5-4B`

### 5.1 Cold-start wall-clock build time, GPU memory, and atlas fidelity on `A10G`

Table 5 reports the end-to-end cold-start build benchmark on `Qwen3.5-4B` (`A10G`, 3 timed repeats after CUDA warmup for `micro_rollout` and `micro_prefill`, 1 full timed build on the same `A10G` container for `k1_atlas`), along with middle-layer ($L=16$) cosine similarity of each freshly built atlas against the reference `full_s1` (`s1b-4b/axis.pt`).

| Atlas Protocol | Rollouts & Horizon | Total Build Time (`mean ± SD`) | Component Breakdown (`pile / gen / extract`) | Peak GPU Memory | $\cos(\hat{a}_{16}, a^{\text{S1}}_{16})$ | Dose Error $|m_{16}-m^{\text{S1}}_{16}|/m^{\text{S1}}_{16}$ | 6-Concept $\cos(\hat{c}_X)$ (`mean / min / wizards`) |
|---|---|---|---|---|---|---|---|
| **`micro_prefill`** ($8\times 4, T=0, N_{\text{pile}}=32$) | `92` prefill (`0` gen tok) | **`1.57 s ± 0.004 s`** `[1.56, 1.57]` (`82.8×` faster) | `0.74 s / 0.02 s / 0.80 s` | **`9.07 GB`** | `0.4658` | `0.92%` | `0.4387 / 0.3538 / 0.3538` |
| **`micro_rollout`** ($8\times 4, T=16, N_{\text{pile}}=32$) | `92` rollouts (`1,422` gen tok) | **`6.38 s ± 1.12 s`** `[5.59, 7.84]` (`20.3×` faster) | `0.74 s / 4.34 s / 1.30 s` | **`10.75 GB`** | `0.6222` | `0.92%` | `0.7233 / 0.6319 / 0.6319` |
| **`k1_atlas`** ($40\times 10, T=128, N_{\text{pile}}=1000$) | `580` rollouts (`71,945` gen tok) | **`129.60 s`** (`2.16 min` batched; `288 s` in `K1`) | `22.98 s / 95.36 s / 11.27 s` | **`11.00 GB`** | **`0.9576`** | **`0.008%`** | **`0.9739 / 0.9551 / 0.9802`** |
| **`full_s1`** (`s1b-4b` reference) | `3,726` rollouts (`T=160, N_{\text{pile}}=2000`) | `~1,800 s` (`30 min` in `S1a`) | — | — | `1.0000` | `0.00%` | `1.0000 / 1.0000 / 1.0000` |

- **Same-container 32-prompt throughput (`A10G`, `Qwen3.5-4B`):**
  - $T=16$ (`512` tokens): `1.15 s` generation (`446.5 tok/s`) + `0.45 s` teacher-forced extraction = **`1.60 s`** per 32-prompt batch.
  - $T=128$ (`4,096` tokens): `7.03 s` generation (`582.6 tok/s`) + `1.07 s` teacher-forced extraction = **`8.10 s`** per 32-prompt batch.

---

### 5.2 Held-out Turn-1 Identity, Turn-2 Challenge Persistence, and Everyday Coherence (`6,120` replies)

All `6,120` Stage 2 replies (`20` prompts × `6` held-out concepts `[cookies, dragons, the ocean, knights, trees, wizards]` = `120` replies per condition across `Turn-1`, `Turn-2`, and `Everyday`) were generated on `Qwen3.5-4B` (`A10G`) and saved to `out/atlas-sample-complexity-20261028/generated.json`. Table 6 reports the pooled metrics ($n = 120$ per cell) with Wilson 95% CIs and 2,000-replicate paired cluster-bootstrap 95% CIs for rate differences $\Delta$:
- **`CM` (Pre-registered lexical concept mention):** `prompts.mentions(reply, concept)` matching the target concept's stem family.
- **`NoAI` (Non-disclosure of AI identity):** Absence of explicit AI/assistant self-identification or role-play denial (`"as an AI"`, `"I am an AI"`, `"language model"`, `"virtual assistant"`, `"just role-playing"`, etc.).
- **`CM & NoAI & NonDeg` (Coherent in-character enactment / persistence):** Simultaneously mentioning the target concept (`CM == 1`), maintaining non-AI identity (`NoAI == 1`), and producing non-degenerate text (`NonDeg == 1`, $\ge 3$ words and unique 4-gram ratio $\ge 0.45$).

| Endpoint & Steering Condition | Atlas / Control | `CM` (`95% Wilson CI`) | `NoAI` (`95% Wilson CI`) | `CM & NoAI & NonDeg` (`95% Wilson CI`) | Paired $\Delta(\text{CM\&NoAI})$ vs. `full_s1` (`95% Boot CI`) | Paired $\Delta(\text{CM\&NoAI})$ vs. `k1_atlas` (`95% Boot CI`) |
|---|---|---|---|---|---|---|
| **Turn-1 Identity** (`recipe_0.5_0.5`, unprompted) | `full_s1` | `79.2%` `[71.0%, 85.5%]` | `95.8%` `[90.6%, 98.2%]` | **`76.7%`** `[68.3%, 83.3%]` (`92/120`) | — | `-3.3 pp` `[-11.7, +5.0]` |
| **Turn-1 Identity** (`recipe_0.5_0.5`, unprompted) | `k1_atlas` | `82.5%` `[74.7%, 88.3%]` | `97.5%` `[92.9%, 99.2%]` | **`80.0%`** `[72.0%, 86.2%]` (`96/120`) | `+3.3 pp` `[-5.0, +11.7]` | — |
| **Turn-1 Identity** (`recipe_0.5_0.5`, unprompted) | **`micro_rollout`** | `77.5%` `[69.2%, 84.0%]` | `85.8%` `[78.5%, 91.0%]` | **`71.7%`** `[63.0%, 79.0%]` (`86/120`) | `-5.0 pp` `[-15.0, +5.0]` | `-8.3 pp` `[-18.3, +1.7]` |
| **Turn-1 Identity** (`recipe_0.5_0.5`, unprompted) | **`micro_prefill`** | `47.5%` `[38.8%, 56.4%]` | `88.3%` `[81.4%, 92.9%]` | **`43.3%`** `[34.8%, 52.3%]` (`52/120`) | **`-33.3 pp`** `[-44.2, -21.7]` | **`-36.7 pp`** `[-47.5, -25.0]` |
| **Turn-1 Identity** (`recipe_0.5_0.5`, unprompted) | `random_dir` | `7.5%` `[4.0%, 13.6%]` | `8.3%` `[4.6%, 14.7%]` | **`0.0%`** `[0.0%, 3.1%]` (`0/120`) | `-76.7 pp` `[-84.2, -69.2]` | `-80.0 pp` `[-86.7, -72.5]` |
| **Turn-1 Identity** (`recipe_0.5_0.5`, unprompted) | `neither` | `6.7%` `[3.4%, 12.6%]` | `0.0%` `[0.0%, 3.1%]` | **`0.0%`** `[0.0%, 3.1%]` (`0/120`) | `-76.7 pp` `[-84.2, -69.2]` | `-80.0 pp` `[-86.7, -72.5]` |
| **Turn-1 Identity** (`prompt_only`, unsteered) | `prompt_only` | `89.2%` `[82.3%, 93.6%]` | `100.0%` `[96.9%, 100.0%]` | **`89.2%`** `[82.3%, 93.6%]` (`107/120`) | `+12.5 pp` `[+3.3, +21.7]` | `+9.2 pp` `[+0.8, +17.5]` |
|---|---|---|---|---|---|---|
| **Turn-1 Identity** (`recipe_0.5_0.25`, gentler dose) | `full_s1` | `81.7%` `[73.8%, 87.6%]` | `97.5%` `[92.9%, 99.2%]` | **`79.2%`** `[71.0%, 85.5%]` (`95/120`) | — | `+2.5 pp` `[-4.2, +9.2]` |
| **Turn-1 Identity** (`recipe_0.5_0.25`, gentler dose) | `k1_atlas` | `81.7%` `[73.8%, 87.6%]` | `95.8%` `[90.6%, 98.2%]` | **`76.7%`** `[68.3%, 83.3%]` (`92/120`) | `-2.5 pp` `[-9.2, +4.2]` | — |
| **Turn-1 Identity** (`recipe_0.5_0.25`, gentler dose) | **`micro_rollout`** | `65.8%` `[57.0%, 73.7%]` | `60.0%` `[51.1%, 68.3%]` | **`45.0%`** `[36.4%, 53.9%]` (`54/120`) | **`-34.2 pp`** `[-43.3, -24.2]` | **`-31.7 pp`** `[-40.8, -21.7]` |
| **Turn-1 Identity** (`recipe_0.5_0.25`, gentler dose) | **`micro_prefill`** | `36.7%` `[28.6%, 45.6%]` | `59.2%` `[50.2%, 67.5%]` | **`26.7%`** `[19.6%, 35.2%]` (`32/120`) | **`-52.5 pp`** `[-61.7, -43.3]` | **`-50.0 pp`** `[-59.2, -40.0]` |
|---|---|---|---|---|---|---|
| **Turn-2 Challenge** (`recipe_0.5_0.5`, unprompted) | `full_s1` | `86.7%` `[79.4%, 91.6%]` | `60.8%` `[51.9%, 69.1%]` | **`44.2%`** `[35.6%, 53.1%]` (`53/120`) | — | `-5.8 pp` `[-15.8, +4.2]` |
| **Turn-2 Challenge** (`recipe_0.5_0.5`, unprompted) | `k1_atlas` | `85.8%` `[78.5%, 91.0%]` | `66.7%` `[57.8%, 74.5%]` | **`50.0%`** `[41.2%, 58.8%]` (`60/120`) | `+5.8 pp` `[-4.2, +15.8]` | — |
| **Turn-2 Challenge** (`recipe_0.5_0.5`, unprompted) | **`micro_rollout`** | `72.5%` `[63.9%, 79.7%]` | `20.0%` `[13.8%, 28.0%]` | **`16.7%`** `[11.1%, 24.3%]` (`20/120`) | **`-27.5 pp`** `[-37.5, -17.5]` | **`-33.3 pp`** `[-43.3, -23.3]` |
| **Turn-2 Challenge** (`recipe_0.5_0.5`, unprompted) | **`micro_prefill`** | `57.5%` `[48.6%, 66.0%]` | `40.0%` `[31.7%, 48.9%]` | **`24.2%`** `[17.4%, 32.6%]` (`29/120`) | **`-20.0 pp`** `[-31.7, -8.3]` | **`-25.8 pp`** `[-37.5, -14.2]` |
| **Turn-2 Challenge** (`recipe_0.5_0.5`, unprompted) | `random_dir` | `4.2%` `[1.8%, 9.4%]` | `1.7%` `[0.5%, 5.9%]` | **`0.0%`** `[0.0%, 3.1%]` (`0/120`) | `-44.2 pp` `[-53.3, -35.0]` | `-50.0 pp` `[-59.2, -40.8]` |
| **Turn-2 Challenge** (`prompt_only`, unsteered) | `prompt_only` | `96.7%` `[91.7%, 98.7%]` | **`0.0%`** `[0.0%, 3.1%]` | **`0.0%`** `[0.0%, 3.1%]` (`0/120`) | **`-44.2 pp`** `[-53.3, -35.0]` | **`-50.0 pp`** `[-59.2, -40.8]` |
|---|---|---|---|---|---|---|
| **Turn-2 Challenge** (`both_0.5_0.25`, prompt + recipe) | `full_s1` | `93.3%` `[87.4%, 96.6%]` | `78.3%` `[70.1%, 84.8%]` | **`74.2%`** `[65.7%, 81.2%]` (`89/120`) | — | `+2.5 pp` `[-6.7, +11.7]` |
| **Turn-2 Challenge** (`both_0.5_0.25`, prompt + recipe) | `k1_atlas` | `94.2%` `[88.5%, 97.1%]` | `78.3%` `[70.1%, 84.8%]` | **`71.7%`** `[63.0%, 79.0%]` (`86/120`) | `-2.5 pp` `[-11.7, +6.7]` | — |
| **Turn-2 Challenge** (`both_0.5_0.25`, prompt + recipe) | **`micro_rollout`** | `88.3%` `[81.4%, 92.9%]` | `50.8%` `[42.0%, 59.6%]` | **`41.7%`** `[33.2%, 50.6%]` (`50/120`) | **`-32.5 pp`** `[-42.5, -22.5]` | **`-30.0 pp`** `[-40.0, -19.2]` |
| **Turn-2 Challenge** (`both_0.5_0.25`, prompt + recipe) | **`micro_prefill`** | `91.7%` `[85.3%, 95.4%]` | `67.5%` `[58.7%, 75.2%]` | **`60.8%`** `[51.9%, 69.1%]` (`73/120`) | **`-13.3 pp`** `[-24.2, -2.5]` | `-10.8 pp` `[-21.7, +0.0]` |
| **Turn-2 Challenge** (`both_0.5_0.25`, prompt + recipe) | `random_dir` | `97.5%` `[92.9%, 99.2%]` | `8.3%` `[4.6%, 14.7%]` | **`8.3%`** `[4.6%, 14.7%]` (`10/120`) | `-65.8 pp` `[-74.2, -56.7]` | `-63.3 pp` `[-71.7, -54.2]` |
|---|---|---|---|---|---|---|
| **Turn-2 Challenge** (`both_0.5_0.5`, prompt + recipe) | `full_s1` | `84.2%` `[76.6%, 89.6%]` | `92.5%` `[86.4%, 96.0%]` | **`72.5%`** `[63.9%, 79.7%]` (`87/120`) | — | `-15.0 pp` `[-24.2, -5.8]` |
| **Turn-2 Challenge** (`both_0.5_0.5`, prompt + recipe) | `k1_atlas` | `90.0%` `[83.3%, 94.2%]` | `98.3%` `[94.1%, 99.5%]` | **`87.5%`** `[80.4%, 92.3%]` (`105/120`) | `+15.0 pp` `[+5.8, +24.2]` | — |
| **Turn-2 Challenge** (`both_0.5_0.5`, prompt + recipe) | **`micro_rollout`** | `88.3%` `[81.4%, 92.9%]` | `77.5%` `[69.2%, 84.0%]` | **`67.5%`** `[58.7%, 75.2%]` (`81/120`) | `-5.0 pp` `[-15.8, +5.0]` | **`-20.0 pp`** `[-29.2, -10.8]` |
| **Turn-2 Challenge** (`both_0.5_0.5`, prompt + recipe) | **`micro_prefill`** | `72.5%` `[63.9%, 79.7%]` | `93.3%` `[87.4%, 96.6%]` | **`68.3%`** `[59.6%, 76.0%]` (`82/120`) | `-4.2 pp` `[-15.0, +6.7]` | **`-19.2 pp`** `[-29.2, -9.2]` |
| **Turn-2 Challenge** (`both_0.5_0.5`, prompt + recipe) | `random_dir` | `97.5%` `[92.9%, 99.2%]` | `11.7%` `[7.1%, 18.6%]` | **`11.7%`** `[7.1%, 18.6%]` (`14/120`) | `-60.8 pp` `[-70.0, -50.8]` | `-75.8 pp` `[-83.3, -67.5]` |
|---|---|---|---|---|---|---|
| **Everyday Coherence** (`recipe_0.5_0.25`, unprompted) | `full_s1` | `87.5%` `[80.4%, 92.3%]` | `100.0%` `[96.9%, 100.0%]` | **`87.5%`** (`100.0%` non-deg) | — | `0.0 pp` `[-6.7, +6.7]` |
| **Everyday Coherence** (`recipe_0.5_0.25`, unprompted) | `k1_atlas` | `87.5%` `[80.4%, 92.3%]` | `100.0%` `[96.9%, 100.0%]` | **`86.7%`** (`99.2%` non-deg) | `-0.8 pp` `[-7.5, +5.8]` | — |
| **Everyday Coherence** (`recipe_0.5_0.25`, unprompted) | **`micro_rollout`** | `51.7%` `[42.8%, 60.4%]` | `100.0%` `[96.9%, 100.0%]` | **`51.7%`** (`100.0%` non-deg) | **`-35.8 pp`** `[-45.8, -25.8]` | **`-35.0 pp`** `[-45.0, -25.0]` |
| **Everyday Coherence** (`recipe_0.5_0.25`, unprompted) | **`micro_prefill`** | `26.7%` `[19.6%, 35.2%]` | `98.3%` `[94.1%, 99.5%]` | **`25.8%`** (`100.0%` non-deg) | **`-61.7 pp`** `[-70.8, -51.7]` | **`-60.8 pp`** `[-70.0, -50.8]` |

---

### 5.3 Per-concept breakdown across the 5 held-out test concepts + `wizards`

Table 7 shows the per-concept count (out of `20` prompts per concept) achieving `CM & NoAI & NonDeg` across the 6 concepts (`cookies`, `dragons`, `the ocean`, `knights`, `trees`, and on-the-fly extracted `wizards`).

| Split & Condition | Atlas | `cookies` (`/20`) | `dragons` (`/20`) | `the ocean` (`/20`) | `knights` (`/20`) | `trees` (`/20`) | `wizards` (`/20`) | Pooled Total (`/120`) |
|---|---|---|---|---|---|---|---|---|
| **Turn-1 (`recipe_0.5_0.5`)** | `full_s1` | `18/20` | `14/20` | `15/20` | `18/20` | `20/20` | `7/20` | **`92/120` (`76.7%`)** |
| **Turn-1 (`recipe_0.5_0.5`)** | `k1_atlas` | `18/20` | `18/20` | `12/20` | `20/20` | `20/20` | `8/20` | **`96/120` (`80.0%`)** |
| **Turn-1 (`recipe_0.5_0.5`)** | `micro_rollout` | `13/20` | `9/20` | `18/20` | `17/20` | `16/20` | `13/20` | **`86/120` (`71.7%`)** |
| **Turn-1 (`recipe_0.5_0.5`)** | `micro_prefill` | `7/20` | `2/20` | `17/20` | `7/20` | `16/20` | `3/20` | **`52/120` (`43.3%`)** |
|---|---|---|---|---|---|---|---|---|
| **Turn-1 (`recipe_0.5_0.25`)** | `full_s1` | `13/20` | `17/20` | `17/20` | `20/20` | `20/20` | `8/20` | **`95/120` (`79.2%`)** |
| **Turn-1 (`recipe_0.5_0.25`)** | `k1_atlas` | `14/20` | `14/20` | `17/20` | `20/20` | `20/20` | `7/20` | **`92/120` (`76.7%`)** |
| **Turn-1 (`recipe_0.5_0.25`)** | `micro_rollout` | `9/20` | `4/20` | `17/20` | `10/20` | `11/20` | `3/20` | **`54/120` (`45.0%`)** |
| **Turn-1 (`recipe_0.5_0.25`)** | `micro_prefill` | `6/20` | `1/20` | `12/20` | `2/20` | `10/20` | `1/20` | **`32/120` (`26.7%`)** |
|---|---|---|---|---|---|---|---|---|
| **Turn-2 (`recipe_0.5_0.5`)** | `full_s1` | `1/20` | `6/20` | `7/20` | `13/20` | `19/20` | `7/20` | **`53/120` (`44.2%`)** |
| **Turn-2 (`recipe_0.5_0.5`)** | `k1_atlas` | `0/20` | `10/20` | `8/20` | `13/20` | `18/20` | `11/20` | **`60/120` (`50.0%`)** |
| **Turn-2 (`recipe_0.5_0.5`)** | `micro_rollout` | `0/20` | `0/20` | `2/20` | `13/20` | `0/20` | `5/20` | **`20/120` (`16.7%`)** |
| **Turn-2 (`recipe_0.5_0.5`)** | `micro_prefill` | `3/20` | `2/20` | `13/20` | `9/20` | `0/20` | `2/20` | **`29/120` (`24.2%`)** |
|---|---|---|---|---|---|---|---|---|
| **Turn-2 (`both_0.5_0.25`)** | `full_s1` | `3/20` | `19/20` | `14/20` | `18/20` | `17/20` | `18/20` | **`89/120` (`74.2%`)** |
| **Turn-2 (`both_0.5_0.25`)** | `k1_atlas` | `4/20` | `20/20` | `15/20` | `15/20` | `16/20` | `16/20` | **`86/120` (`71.7%`)** |
| **Turn-2 (`both_0.5_0.25`)** | `micro_rollout` | `1/20` | `15/20` | `6/20` | `10/20` | `2/20` | `16/20` | **`50/120` (`41.7%`)** |
| **Turn-2 (`both_0.5_0.25`)** | `micro_prefill` | `13/20` | `9/20` | `15/20` | `13/20` | `14/20` | `9/20` | **`73/120` (`60.8%`)** |
| **Turn-2 (`both_0.5_0.25`)** | `prompt_only` | `0/20` | `0/20` | `0/20` | `0/20` | `0/20` | `0/20` | **`0/120` (`0.0%`)** |

---

## 6. Practical implications for `personakit` cold-start architecture

Although the pre-registered $T \le 16$ micro-atlas hypothesis is refuted (`No`), the Stage 1 factorial decomposition reveals exactly which parts of `personakit`'s cold-start pipeline were redundant and which are load-bearing:

1. **Cut Pile calibration by $31\times$ (`1,000` $\to$ `32` passages):**
   Saves **`22.2 seconds`** on `4B` (`22.98 s` $\to$ `0.74 s`) while keeping $m_L$ within `0.56%` (`4B`) and `0.90%` (`9B`) of the 2,000-passage `S1` reference.
2. **Use the 8 or 16 `2B` `diverse_cross` medoid roles instead of 40 random roles, and keep $\ge 40$ default-Assistant rollouts:**
   At $T=64\text{–}128$, 8 `diverse_cross` roles × 10 questions (`80` role rollouts) + 40 default rollouts + 10 concepts × 8 questions (`80` concept rollouts) requires only **`200` rollouts** instead of `K1`'s `580` rollouts (a **`2.9×` rollout reduction**) while achieving $\cos(\hat{a}_{16}) = 0.9585$ (`[0.9477, 0.9682]`) and 10-concept mean $\cos(\hat{c}_X) = 0.9795$ (`min = 0.9722`) at $T=\text{full}$.
3. **Never substitute $T=0$ prefill or $T=16$ short rollouts for multi-sentence generation ($T \ge 64$):**
   On `Qwen3.5-4B` (`A10G`), `200` rollouts at $T=64\text{–}96$ with $N_{\text{pile}}=32$ takes **`~24–32 seconds`** end-to-end—a **`9–12×` speedup** over `K1`'s original `288 s` build without sacrificing the $\ge 0.90\text{–}0.95$ axis/concept fidelity required for Turn-2 challenge persistence.

---

## 7. Deviations, caveats, and budget accounting

- **Deviations from pre-registered design:**
  1. **Stage 2 MMLU-300 and `9B` judge pass:** In the primary Modal run (`25.1m`, `$0.71`), Stage 1 (`2B`, `4B`, `9B` across all 7 horizons) and Stage 2 (`A10G` timing builds, `stage2-atlases.pt`, and all `6,120` generated `Turn-1`, `Turn-2`, and `Everyday` replies in `generated.json`) completed 100%, after which the job hit its 25-minute wall-clock timeout inside unbatched MMLU-300 (`300 items × 66 conditions = 19,800` unbatched forward passes). We added checkpoint-resume and zero-padded 61-condition batched MMLU (`5ed597d3`, reviewed in with 0 findings) and ran an 8-minute continuation (`8.0m`, `$0.23`). Because `Qwen3.5-4B`'s 24 `GatedDeltaNet` layers fall back to pure-PyTorch `causal_conv1d_fn` when `causal_conv1d` CUDA wheels are absent from the container image, batched MMLU through layers `0..15` did not finish within the remaining 8-minute window, bringing total Modal spend to `$0.94` vs. the `$0.75` budget cap. Per the study rules (*"When the budget runs out, stop and write up what you have"*), we stopped immediately without launching any further paid runs and scored all `6,120` saved Stage 2 replies locally using the pre-registered `prompts.mentions` (`CM`), explicit AI-disclosure regex (`NoAI`), and 4-gram degeneracy gates (`NonDeg`).
  2. Because Stage 1 ($\cos(\hat{a}_{16}) = 0.5555\text{–}0.7086$ at $T=16$) and Stage 2 (`Turn-2` persistence `-30.0 to -41.7 pp` vs. `k1_atlas`) already decisively falsify the sub-25s $T \le 16$ micro-atlas hypothesis on both pre-registered primary criteria, the missing MMLU-300 table cannot change the headline `No` conclusion.
- **Exact compute ledger (`the spend ledger`):**
  - CPU `--smoke` checks (`Qwen3.5-0.8B` initial + resume): **`$0.00`**
  - Modal run (`A10G`, `25.1m`): **`$0.71`**
  - Modal run (`A10G`, `8.0m`): **`$0.23`**
  - **Total settled compute:** **`$0.94`**

---

## 8. Reproduction and artifacts

- **Code:** `followups/atlas-sample-complexity/run_study.py`
- **CPU smoke test (`$0.00`):**
*(Run commands omitted: they use private tooling. Each script's docstring lists its arguments.)*
- **Saved Volume Artifacts (`out/atlas-sample-complexity-20261028/`):**
  - `results.json` (`12.2 MB`): Complete Stage 1 200-bootstrap curves (`2B`, `4B`, `9B` across all 7 horizons) + Stage 2 `A10G` build timings, GPU memory, throughput, and fresh atlas cosines.
  - `stage2-atlases.pt` (`28.9 MB`): All 4 Stage 2 `Qwen3.5-4B` atlases (`full_s1`, `k1_atlas`, `micro_rollout`, `micro_prefill`).
  - `generated.json` (`4.4 MB`): All `6,120` Stage 2 generated replies across `Turn-1`, `Turn-2`, and `Everyday` splits and all 6 held-out concepts (`cookies`, `dragons`, `the ocean`, `knights`, `trees`, `wizards`).

---

## 9. Citations

- Lu, C. et al. (2026). *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*. [arXiv:2601.10387](https://arxiv.org/abs/2601.10387)
- Zou, A. et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
- Rimsky, N. et al. (2024). *Steering Llama 2 via Contrastive Activation Addition*. [arXiv:2312.06681](https://arxiv.org/abs/2312.06681)
- Chen, R. et al. (2025). *Persona Vectors: Monitoring and Controlling Character Traits in Language Models*. [arXiv:2507.21509](https://arxiv.org/abs/2507.21509)
- Wu, Z. et al. (2025). *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders*. [arXiv:2501.17148](https://arxiv.org/abs/2501.17148)
- Lyra (2026). *neuron-explorer* (`e566e83`). [https://github.com/lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)
