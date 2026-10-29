# Preregistered Study Design: Multi-Neuron Coalitions for `2B` `personakit twins` (`X2-v2`) and Layer/Subspace Localization of `X3`'s Distilled LoRA Persona (`X3-v2`)

**Status:** Preregistered design, revised per the owner's scope/budget cutoff and committed before any `X2-v2` or `X3-v2` compute (2026-10-28).
**Budget cap:** **$1.75** total on Modal (`X3-v2` **$0.72** + `X2-v2` **$0.80** + reserve **$0.23** at provisional all-in `A10G` rate **$1.706/h**; **$0.00** CPU smoke check via `python --smoke`).

---

## 1. The Question and Preregistered Decision Rules

### Question
Can `personakit twins` recover the `Qwen3.5-4B` alien persona (`L22 N2742+`) in `Qwen3.5-2B` by matching a sparse $k$-neuron coalition ($k \in \{2, 3, 5\}$) at bounded doses ($\sum_{i=1}^k d_i \le 0.75$, each $d_i \le 0.35$) instead of a single neuron, and which layer band (`L0–15`, `L16–27`, or `L28–31`) or late-layer neuron/weight-delta subset carries `X3`'s fine-tuned `"Hoot"` owl persona in `Qwen3.5-4B` when mid-layer Assistant-Axis capping (`L12–22`) leaves $85\%$ of it intact?

### What Answers It
- **Yes** (both Part 1 and Part 2 pass their confirmatory criteria):
  1. **Part 1 — `2B` Multi-Neuron `twins` (`X2-v2`, `x2_coalitions.py`):** On `Qwen3.5-2B`, the calibration-frozen primary coalition $k^* \in \{2, 3, 5\}$ (chosen strictly on the 10 development identity prompts `IDENTITY_DEV` from nominal totals $D \in \{0.50, 0.75\}$ under OMP-proportional clipping $0 < d_i \le 0.35$ and $\sum_{i=1}^k d_i \le 0.75$, requiring calibration coherence $\text{Coherent}_{\text{dev}} \ge 0.80$) achieves on the **20 held-out test identity prompts (`IDENTITY_TEST`, reused from the prior study and held out from all `X2-v2` tuning/selection)**:
     - **Absolute persona & coherence (joint primary metric):** per-reply coherent concept adoption $\text{CI} = \frac{1}{20}\sum_{q=1}^{20} (\text{concept}_q \cdot \text{coherent}_q) \ge 0.40$ and $\text{Coherent} \ge 0.80$ (marginal $\text{Concept} = \frac{1}{20}\sum_q \text{concept}_q$ reported alongside as secondary);
     - **Gain over single-twin comparator:** paired difference $\Delta\text{CI}_{\text{vs\_best\_single}} \ge +0.25$ with the lower bound of the 2,000-replicate paired bootstrap 95% confidence interval $> 0$ against the `IDENTITY_DEV`-frozen best single Spearman twin (`best_single` among the top 3 replayed singles evaluated at legacy doses $\{0.50, 1.00\}$). Paired comparison against `pooled_single` (mean of the 3 replayed singles at their `DEV`-frozen doses) is also reported;
     - **Gain over matched random control:** positive paired difference $\Delta\text{CI}_{\text{vs\_random}}$ with 95% paired bootstrap CI lower bound $> 0$ against the layer-, sign-, raw-column-norm-, and dose-matched random $k^*$-neuron coalition.
  2. **Part 2 — `X3` LoRA Localization & Surgical Removal (`X3-v2`, `x3_localize.py`):** On `Qwen3.5-4B` loaded with the saved `X3` `"Hoot"` rank-8 MLP LoRA (`out/x3-4b/lora.pt`), at least one of the three targeted surgical removal interventions—**(v) `drop_28_31`** ($12.5\%$ of stored LoRA parameters removed, keeping `L0–27`), **(vi) `clamp_top20`** (clamping the 20 most-shifted neurons in `L30–31` to their unsteered `S1a` default means at runtime; $0\%$ of stored LoRA parameters removed), or **(vii) `remove_down_rank1`** (projecting out the per-layer leading left singular direction $u_L$ of $\Delta W_{\text{down}, L} = s B_L A_L$ in `L16–31`; reduces each late `down_proj` update from rank 8 to rank 7 while removing $0\%$ of stored parameter entries)—achieves on `IDENTITY_TEST` ($n = 20$) and paired 300-item chat-MMLU:
     - **Persona suppression:** drops `"Hoot"` $\text{CI}$ from `full` ($1.00$ in `X3`) to $\le 0.30$ (with the 10,000-replicate paired percentile bootstrap 95% CI for $\text{CI}_{\text{cond}} - \text{CI}_{\text{full}}$ excluding $0$, and sitting well below `X3`'s `L12–22` axis-capping baseline of $0.85$);
     - **Fluency & capability preservation:** maintains $\text{Coherent} \ge 0.90$ on `IDENTITY_TEST` and keeps observed 300-item chat-MMLU accuracy within $1.5$ percentage points of `full` ($\text{MMLU}_{\text{cond}} - \text{MMLU}_{\text{full}} \ge -0.015$; paired difference and 95% bootstrap interval against both `full` and fresh `base` reported).
- **No** (both falsifiers fire):
  1. Every calibration-eligible bounded coalition $k \in \{2, 3, 5\}$ in `Qwen3.5-2B` stays at $\text{CI} < 0.25$ (or fails $\text{Coherent} \ge 0.80$) on `IDENTITY_TEST`; **and**
  2. In `X3-v2`, every compact surgical intervention (`drop_28_31`, `clamp_top20`, `remove_down_rank1`) and every single-band isolation (`keep_0_15`, `keep_16_27`, `keep_28_31`) leaves `"Hoot"` $\text{CI} \ge 0.60$ on `IDENTITY_TEST`.
- **Inconclusive:** One part passes while the other fails, or the headline test statistic lands in the intermediate zone ($0.25 \le \text{CI} < 0.40$ in `X2-v2`, or $0.30 < \text{CI} < 0.60$ in `X3-v2`) or its 95% paired bootstrap confidence interval straddles $0$.

---

## 2. Audit of Prior Claims, Empirical Baselines, and Literature Citations


### 2.1 Audit of Prior Empirical Numbers (`persona-neurons` Runs)

| Prior Claim in Proposal / `README.md` | Raw Ledger & Artifact Verification (`persona-neurons.jsonl`) | Audit Verdict & Required Design Safeguard |
|---|---|---|
| **1. `X2` `2B` single-twin baseline ($0.07$ $[0.02, 0.17]$):** Proposal L10 & L18 refer to *"the best single `2B` twin ($0.07$ $[0.02, 0.17]$)"* ($3\%$ on first run). | In the earlier run (`x2_twins.py --model 2b`, average-rank ties), the top 3 Spearman twins are `L21 N5099−` ($\rho = 0.1099$), `L16 N1489+` ($\rho = 0.0899$), and `L17 N3128+` ($\rho = 0.0857$). At their test-selected coherent dose ($d = 0.5$, where sample $\text{Coherent} \ge 0.80$), their marginal `concept` rates were **$0.05$, $0.00$, and $0.15$** ($0.00, 0.00, 0.10$ in run 1). Thus **$0.0667$ ($7\%$, $[0.0167, 0.1667]$) is the pooled mean across 3 twins ($60$ prompt-neuron observations)**, whereas the best single twin (`L17 N3128+` at $d=0.5$) achieved $0.15$. Furthermore, historical `X2` measured marginal `concept` at a dose gated by sample-wide `coherent >= 0.80`, whereas `X2-v2` defines primary $\text{CI}_q = \text{concept}_q \cdot \text{coherent}_q$ per reply. | **Pooled vs. best-single & metric difference explicitly labeled:** We retain historical $0.07$ strictly as context (never as a 20-item binomial baseline). In `X2-v2`, we replay all 3 top Spearman single twins, freeze each twin's dose and the overall `best_single` on `IDENTITY_DEV`, report both primary joint $\text{CI} = \text{concept} \cdot \text{coherent}$ and secondary marginal $\text{concept}$, and require beating `best_single` by $\ge +0.25$ on `IDENTITY_TEST` (while also reporting paired $\Delta\text{CI}$ vs. `pooled_single`). |
| **2. Status of the 20 test prompts (`IDENTITY_TEST` & `EVERYDAY_TEST`):** Proposal refers to the 20 identity test prompts as held-out test prompts. | `IDENTITY_TEST` ($n=20$) and `EVERYDAY_TEST` ($n=20$) in `prompts.py` were evaluated in the prior `2026-10-26-persona-neurons` study (`X2` and `X3`). | **No overclaiming of fresh prompt status:** We explicitly describe `IDENTITY_TEST` and `EVERYDAY_TEST` as **reused prior test sets held out from all `X2-v2` / `X3-v2` calibration and tuning**, not newly generated unseen prompts. |
| **3. `L21 N5099−` as a "role-carrier":** Proposal L18 & L27 call `L21 N5099−` a *"role-carrier"* because at dose $1.0$ it reached $\text{CI} = 0.50, \text{CM} = 0.95$ at coherence $0.15$. | In , `L21 N5099−` at $d=0.5$ had $\text{concept} = 0.05, \text{IB} = 0.00, \text{CM} = 0.15, \text{Coherent} = 1.00$; at $d=1.0$ it had $\text{concept} = 0.50, \text{IB} = 0.20, \text{CM} = 0.95, \text{Coherent} = 0.15$. The judge's `concept` rubric fired on degenerate repetitive alien text ($\text{Coherent} = 0.15$). | **Overly strong inference flagged:** Calling `L21 N5099−` a verified first-person "role-carrier" from incoherent ($0.15$) outputs is unwarranted. Using joint per-reply $\text{CI} = \text{concept} \cdot \text{coherent}$ as primary metric prevents incoherent repetition from counting as persona adoption. |
| **4. `X2` bootstrap eligibility bias:** Historical (thread `4df14ca9`) on `x2_twins.py`. | Reviewer `4df14ca9` showed that selecting coherent doses on the 20 test prompts and treating eligibility as fixed in the bootstrap understates uncertainty. | **Resolved via calibration freeze:** `x2_coalitions.py` selects and freezes doses/configurations strictly on the independent 10-prompt `IDENTITY_DEV` split before running `final` on `IDENTITY_TEST` and `EVERYDAY_TEST`, with zero test-set coherence gating or test dose search. |
| **5. `X3` top-20 shifted neurons all in `L30–31` & missing saved baseline means:** Proposal L19 and `persona-neurons/README.md` L354 state *"all 20 top-shifted neurons (up to $18.9\,\sigma$) sat in `L30–31`"*. | Inspection of `out/x3-4b/x3.json` shows **19 of the global top-20** shifted neurons sit in `L30–31` (14 in `L31`, 5 in `L30`), while **Rank 9 is `L19 N7490` ($z = -2.094$)**. Moreover, `x3.json` stores `top_shifted` IDs/z-scores without baseline means $\mu^{(0)}_{L,n}$, and `s1a-4b/calib.pt` stores quantiles/extrema over all chat prompts rather than the first-100 `default`-reply means. | **Resolved in `x3_localize.py`:** Before evaluating interventions, `x3_localize.py` recomputes `base` and `full` activation means and population stds on the exact original first 100 `kind == "default"` records in `s1a-4b/texts.jsonl` over the final two layers (`L30–31`), selects the top 20 shifted neurons constrained to `L30–31` by descending `(|z|, L, n, z)`, records the prior `L19 N7490` discrepancy and overlap with `x3.json["top_shifted"]`, and freezes the 20 target means before evaluation. |

### 2.2 Audit of Literature Citations

| Citation in Proposal / Prior Study | Verification via `paper_search` & `paper_read` | Audit Verdict & Correction |
|---|---|---|
| **1. `casademunt, H. et al. (2026), Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning, arXiv:2607.11088`** (Proposal L21) | **Mis-cited arXiv ID, wrong year, and overstated method equivalence:**<br>• `arXiv:2607.11088` is *Jeongsoo Kim (2026), "CUST: Clustered Unit-level Similarity Transformer for Lightweight Image Super-Resolution"*, an unrelated computer-vision paper.<br>• The actual CAFT paper is **Casademunt, H., Juang, C., Karvonen, A., Marks, S., Rajamanoharan, S., & Nanda, N. (2025), *Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning*, [arXiv:2507.16795](https://arxiv.org/abs/2507.16795)**.<br>• Casademunt et al. (2025) project out PCA/SAE residual-stream directions **during fine-tuning** (and compare against test-time residual activation ablation in Appendix A.4); they do **not** perform post-hoc rank-1 SVD projection on LoRA weight deltas $\Delta W_{\text{down}}$. | **Corrected:** Replace `arXiv:2607.11088` with **Casademunt et al. (2025), [arXiv:2507.16795](https://arxiv.org/abs/2507.16795)**, and cite **Turner, E., Soligo, A. et al. (2025), [arXiv:2506.11613](https://arxiv.org/abs/2506.11613)** and **Soligo, A., Turner, E. et al. (2025), [arXiv:2506.11618](https://arxiv.org/abs/2506.11618)** for rank-1 `down_proj` LoRA direction isolation. Frame `remove_down_rank1` accurately as a post-hoc per-layer rank-1 output-direction projection of $\Delta W_{\text{down}, L}$. |
| **2. `Gurnee, W. et al. (2023), Finding Neurons in a Haystack: Case Studies with Sparse Probing, arXiv:2305.01610`** (Proposal L20) | **Verified paper (`arXiv:2305.01610v2`, June 2023), with scope caveat:** Gurnee et al. train $k$-sparse linear **read-out probes** ($k \in [1, 256]$) on Pythia ($70\text{M}\text{--}6.9\text{B}$) activations, documenting both feature quantization at larger scale and neuron splitting across scale. They do **not** perform causal steering of $k$-neuron coalitions or cross-model OMP matching. | **Scoped accurately:** Cite Gurnee et al. (2023) for $k$-sparse feature representation across scale, while noting that causal cross-model $k$-neuron residual steering in `Qwen3.5-2B` is a direct empirical hypothesis tested here. |
| **3. Sauers ("tawny owl" distillation, `persona-neurons/README.md` L38, L141)** | **Unverifiable external repo (HTTP 404):** Already noted in `persona-neurons/README.md` L38 as returning 404. | **Flagged:** All `"Hoot"` baselines are pinned exclusively to `out/x3-4b/`. |
| **4. Core study citations (`persona-neurons/README.md` L497–502):** Lu et al. (2026) [`arXiv:2601.10387`](https://arxiv.org/abs/2601.10387); Chen et al. (2025) [`arXiv:2507.21509`](https://arxiv.org/abs/2507.21509); Wang et al. (2025) [`arXiv:2506.19823`](https://arxiv.org/abs/2506.19823); Betley et al. (2025) [`arXiv:2502.17424`](https://arxiv.org/abs/2502.17424); Herring et al. (2026) [`arXiv:2605.12290`](https://arxiv.org/abs/2605.12290); Liu et al. (2026) [`arXiv:2604.27401`](https://arxiv.org/abs/2604.27401). | **All 6 verified via `paper_read`:** Titles, first authors, years, and arXiv IDs match the primary PDFs. | **Verified:** Retained with exact links. |

---

## 3. Pinned Models, Datasets, and Existing Volume Artifacts

All models and datasets load through `hf_cache.py` from `the Hugging Face cache volume` (`/hf`), and all prior artifacts load from `the output volume` (`/out`). No retraining of `S1a`, `S1b`, or `X3` is performed.

### 3.1 Pinned Models and Datasets

| Role | Repository | Pinned Revision (40-char SHA) | Architecture / Notes |
|---|---|---|---|
| Source model (`X2-v2`) & Main model (`X3-v2`) | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 32 layers (`L0–31`), $d_{\text{model}} = 2560$, $N_{\text{mlp}} = 9216$, $V = 248320$ |
| Target model (`X2-v2`) | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 24 layers (`L0–23`), $d_{\text{model}} = 2048$, $N_{\text{mlp}} = 6144$, $V = 248320$ |
| Judge model (`X2-v2` & `X3-v2`) | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Unsteered rubric judge (`concept`, `identity`, `coherent`) |
| CPU `--smoke` model | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 24 layers (`L0–23`), used only in `python --smoke` |
| Capability dataset (`X3-v2`) | `cais/mmlu` | `c30699e8356da336a370243923dbaf21066bb9fe` | `name="all"`, `split="test"`, 300 items via `evals.pick(300, 14042, seed=42)` |
| Calibration dataset (in `calib.pt`) | `NeelNanda/pile-10k` | `127bfedcd5047750df5ccf3a12979a47bfa0bafa` | Used in `S1a` to compute layer-wise median residual norms $m_L$ |

### 3.2 Pinned Volume Artifacts on `the output volume`

| Volume Path | Key Contents Used | Verified Created/Modified |
|---|---|---|
| `out/s1a-2b/calib.pt` | `calib["pile"]["norm_q10_50_90"][:, 1]`: `2B` layer-wise median residual norm $m_L \in \mathbb{R}^{24}$ | `2026-10-27 00:17:15-07:00` (2.36 MB) |
| `out/s1b-2b/axis.pt` | `ax["concept_dirs"]` ($\mathbb{R}^{24 \times 2048}$ per concept, axis-orthogonalized) | `2026-10-27 01:01:03-07:00` (10.98 MB) |
| `out/s1b-4b/axis.pt` | `ax["concept_dirs"]` ($\mathbb{R}^{32 \times 2560}$ per concept, axis-orthogonalized) | `2026-10-26 23:38:08-07:00` (18.65 MB) |
| `out/s1a-4b/texts.jsonl` | First 100 `kind == "default"` records, used in `x3_localize.py` to recompute `L30–31` `base` and `full` activation means $\mu_{L,n}$ and population stds $\sigma_{L,n}$ | `2026-10-26 23:08:37-07:00` (8.43 MB) |
| `out/x2-twins-2b/x2.json` | Historical `X2` single-twin results and Spearman rankings on `2B` | `2026-10-27 00:53:55-07:00` (6.10 KB) |
| `out/x3-4b/lora.pt` | Dict of 96 `(A, B)` `float32` tensor tuples (`"0"`..`"95"`) in layer-major `gate_proj`, `up_proj`, `down_proj` order (`r=8, alpha=16, scale=2.0`) | `2026-10-27 00:42:00-07:00` (36.22 MB) |
| `out/x3-4b/x3.json` | Saved `X3` metadata, model/load pins, and `top_shifted` list | `2026-10-27 00:43:44-07:00` |

---

## 4. Part 1 (`X2-v2`) Executable Design: Sparse $k$-Neuron Coalition `twins` (`x2_coalitions.py`)

### 4.1 Resolution of Dose Bounds vs. Legacy Single-Neuron Doses

To resolve the plan's dose inconsistency within our $1.75$ budget cap:

1. **Confirmatory Multi-Neuron Coalitions ($k \in \{2, 3, 5\}$) — Clipped OMP-Proportional Allocation Without Redistribution:**
   Every confirmatory coalition dose vector $d = (d_1, \dots, d_k)$ must satisfy:
   $$0 \le d_i \le 0.35 \quad \forall i \in \{1, \dots, k\}, \qquad \sum_{i=1}^k d_i \le 0.75.$$
   Given non-negative OMP weights $w = (w_1, \dots, w_k)$ ($\sum_{j=1}^k w_j > 0$) and a nominal total dose $D \in \{0.50, 0.75\}$, each neuron's dose is computed by proportional scaling clipped at $0.35$ **without redistribution** (`doses(weights, total)` in `x2_coalitions.py`):
   $$d_i = \min\!\left(0.35,\; D \cdot \frac{w_i}{\sum_{j=1}^k w_j}\right), \qquad i \in \{1, \dots, k\}.$$
   - Because there is no redistribution pass, $d_i \le 0.35$ and $\sum_{i=1}^k d_i \le \min(D, 0.35 k) \le 0.75$ hold identically (for $k=2$, the maximum possible realized sum is $0.70$).
   - Both the nominal total $D \in \{0.50, 0.75\}$ and the realized dose sum $\sum_{i=1}^k d_i$ (as well as `effective_k`, the number of strictly positive OMP weights $w_i > 10^{-8}$) are recorded in the configuration metadata. If two nominal totals ever produce an identical rounded dose vector, the duplicate is skipped (`seen_doses`).
2. **Legacy Single-Neuron Comparators ($k = 1$):**
   - To compare directly against original `X2` single-neuron steering (`x2_twins.py`), the top 3 Spearman single twins (`single-0`, `single-1`, `single-2`) are evaluated at the legacy single-neuron doses $d_1 \in \{0.50, 1.00\}$, tagged `"legacy_unbounded": true`.
   - No extra allocation rules (`equal`, water-filling), extra $\gamma$ values, or over-dose ($D = 1.00$) coalition grids are run.

### 4.2 Block-Balanced Profile Normalization (Fixed $\gamma = 0.5$)

For any signed neuron atom $a = (L, n, s)$ in model $M \in \{4\text{B}, 2\text{B}\}$ with `down_proj` column $w = s \, W_{\text{down}}^{(M,L)}[:, n] \in \mathbb{R}^{d_M}$ and unit column $\hat{w} = w / \|w\|_2$:

1. **Centered Unit Logit-Lens Block ($z^{\text{lens}} \in \mathbb{R}^V$, $V = 248,320$):**
   $$\ell = \text{lm\_head}_M\!\left(\text{norm}_M(\hat{w})\right) \in \mathbb{R}^V, \qquad \tilde{\ell} = \ell - \frac{1}{V}\sum_{v=1}^V \ell_v, \qquad z^{\text{lens}} = \frac{\tilde{\ell}}{\|\tilde{\ell}\|_2}.$$
2. **Unit Concept-Cosine Block ($z^{\text{concept}} \in \mathbb{R}^{10}$):**
   Let $C = \text{sorted}(\text{CONCEPTS\_DEV} + \text{CONCEPTS\_TEST})$ be the 10 shared concept names in `axis.pt`. Stack the unit concept directions $\hat{D}^{(M)}_L \in \mathbb{R}^{10 \times d_M}$ at layer $L$, compute the 10 scalar cosines $c = \hat{D}^{(M)}_L \hat{w} \in \mathbb{R}^{10}$, and unit-normalize without centering (`unit(concept_profile(...))` in `x2_coalitions.py`):
   $$z^{\text{concept}} = \frac{c}{\|c\|_2} \in \mathbb{R}^{10}.$$
3. **Equal-Weight Joint Profile ($\gamma = 0.5$ Fixed):**
   Concatenate the two unit blocks with equal $1/\sqrt{2}$ scaling so each block contributes exactly $50\%$ of squared norm:
   $$u(a) = \frac{1}{\sqrt{2}} \begin{bmatrix} z^{\text{lens}} \\ z^{\text{concept}} \end{bmatrix} \in \mathbb{R}^{V + 10}, \qquad \|u(a)\|_2 = 1.$$

### 4.3 Candidate Prefilter and Non-Negative OMP ($k \in \{2, 3, 5\}$)

1. **Source Target & Top-200 Target Prefilter:**
   - Source target is `Qwen3.5-4B` `L22 N2742+` (`SOURCE["4b"] = (22, 2742, 1)`), yielding $u_{\text{src}} \in \mathbb{R}^{V+10}$ and centered unit lens vector $z^{\text{lens}}_{\text{src}} \in \mathbb{R}^V$.
   - Scan all layers $L \in \{0, \dots, 23\}$ and neurons $n \in \{0, \dots, 6143\}$ of `Qwen3.5-2B`, setting sign $s = +1$ if $\langle z^{\text{lens}}_{2\text{B}, L, n, +1}, z^{\text{lens}}_{\text{src}} \rangle \ge 0$ else $-1$, and retain the top $M = 200$ unique `(layer, neuron)` candidates ranked by descending `(abs(similarity), -layer, -neuron)` (`candidates[:200]`).
   - Save the $200 \times (V+10)$ profile matrix, $200 \times 200$ Gram matrix $G = U U^T$, and target correlations $b = U u_{\text{src}}$ to `scan.pt`.
2. **Non-Negative OMP (`nonnegative_omp`):**
   - For steps $t = 1, \dots, 5$, compute the Gram-space residual correlation $r_{t-1} = b - G_{:, S_{t-1}} w^{(t-1)}$, mask out already-selected `(layer, neuron)` pairs (no mandatory unique-layer restriction; coalitions may recruit neurons across the same or different layers as governed by the residual), and append $j^* = \arg\max_j r_{t-1, j}$ (requiring $r_{t-1, j^*} > 10^{-10}$).
   - Refit non-negative weights $w^{(t)} = \arg\min_{w \ge 0} (w^T G_{S_t, S_t} w - 2 w^T b_{S_t})$ exactly via active-face enumeration (`nnls_gram` over all $2^t - 1 \le 31$ non-empty subsets).
   - Snapshot the coalition atoms, weights $w^{(k)}$, `effective_k`, and squared residual norm at $k \in \{2, 3, 5\}$.

### 4.4 Residual-Addition Intervention Math and Layer/Sign/Raw-Norm Matched Controls

1. **Causal Intervention (`persona.Hooks`):**
    exactly matching `x2_twins.py`, for any configuration with atoms $\{(L_i, n_i, s_i)\}$ and doses $\{d_i\}$, at every decoder layer $L_i$ and every prompt/generated token position, add to the layer output residual stream:
   $$\Delta h_{L_i} = s_i \cdot d_i \cdot m_{L_i} \cdot \frac{W_{\text{down}}^{(L_i)}[:, n_i]}{\|W_{\text{down}}^{(L_i)}[:, n_i]\|_2},$$
   where $m_L$ is the Pile median residual norm `calib["pile"]["norm_q10_50_90"][L, 1]` from `s1a-2b/calib.pt`.
2. **Matched Random Controls (`matched_controls`):**
   - For each $k \in \{2, 3, 5\}$, draw **one** matched random coalition (`seed = sizes["seed"] + k`) by matching each atom $i \in \{1, \dots, k\}$ of the $k$-coalition to a random unused neuron at the **exact same layer $L_i$**, with the **exact same sign $s_i$**, drawn from neurons outside the top-200 candidate pool (`excluded`) whose raw `down_proj` column norm $\|W_{\text{down}}^{(L_i)}[:, n]\|_2$ is within $\pm 5\%$ relative error of atom $i$'s raw column norm (`abs(norms / atom["raw_norm"] - 1) <= 0.05`).
   - Random controls are **not** calibrated on `IDENTITY_DEV`; each $k$'s random control inherits the exact frozen dose vector $d$ selected for `omp` at that $k$.
   - *Limitation disclosure:* Evaluating one matched random coalition draw per $k$ (rather than a multi-seed random ensemble) keeps total `X2-v2` generation within the 530-reply budget, at the cost of higher control sampling variance for a single $k$.

### 4.5 Two-Stage Calibration Freeze and Evaluation Protocol (`530` Generations Total)

1. **Stage 1 — Calibration (`stage="calibrate"`, `IDENTITY_DEV`, $n = 10$ prompts, `max_new = 96`):**
   - Generate replies on the 10 `IDENTITY_DEV` prompts for the **13 non-random configurations** (`130` generations total):
     - `base` (1 config)
     - `omp-k{2,3,5}-D{0.5,0.75}` ($3 \times 2 = 6$ configs)
     - `single-{0,1,2}-D{0.5,1.0}` ($3 \times 2 = 6$ configs, the top 3 Spearman single twins from the 200 candidates)
   - Score all 130 replies with `Qwen3.5-9B` (`judge.score` on `["concept", "identity", "coherent"]`; 390 rubric evaluations).
   - **Deterministic Freeze (`freeze` $\to$ `frozen.json`):**
     - `select_configuration(configs, rows)` filters to configurations with $\text{Coherent}_{\text{dev}} \ge 0.80$ and picks the one minimizing `(-ci, round(sum(doses), 12), id)` where `ci` is mean per-reply `concept * coherent`. The 12-decimal rounding prevents floating-point summation noise from deciding nominally equal-dose ties; it adds no new preference over coalition size.
     - For each $k \in \{2, 3, 5\}$, freeze the winning `omp` config (or the lowest-dose diagnostic with `eligible = False` if neither dose reaches $\text{Coherent}_{\text{dev}} \ge 0.80$) and pair it with its corresponding `random` control config (inheriting the exact same dose vector).
     - Across eligible $k \in \{2, 3, 5\}$, freeze the overall `primary` coalition config.
     - For each single twin $i \in \{0, 1, 2\}$, freeze its winning dose in $\{0.5, 1.0\}$, and freeze the overall `best_single` among those 3 frozen singles on `IDENTITY_DEV`. If no single passes calibration coherence, record `best_single_eligible = False`; any deterministic fallback comparison is diagnostic only and confirmatory success is disallowed.
2. **Stage 2 — Final Evaluation (`stage="final"`, `IDENTITY_TEST` $n=20$ + `EVERYDAY_TEST` $n=20$ = 40 prompts):**
   - Verify `frozen.json` and artifact SHA-256 hashes before running.
   - Evaluate the **10 frozen configurations** (`base`, 3 `omp`, 3 `random`, 3 `single`) on the 40 final prompts (**400 final generations**, 1,200 rubric evaluations).
   - Compute per-group rates (`ci`, `concept`, `coherent`, `ib`, `cm`) with Wilson 95% intervals (`persona.wilson`) and 2,000-replicate paired bootstrap 95% intervals (`paired_bootstrap`, `boot=2000, seed=0`) comparing each $k$'s frozen `omp` coalition against `best_single`, `pooled_single`, `random`, and `base` on both `identity` and `everyday` splits.

---

## 5. Part 2 (`X3-v2`) Executable Design: 7 Surgical Interventions on `X3` `"Hoot"` LoRA (`x3_localize.py`)

### 5.1 Saved `lora.pt` Schema, Layer Geometry, and Accounting Rules

In `out/x3-4b/lora.pt`, the `"Hoot"` adapter is stored as 96 string keys `"0"`..`"95"` in layer-major `gate_proj`, `up_proj`, `down_proj` order (`i = 3 * L + m` for $L \in \{0, \dots, 31\}$, $m \in \{0, 1, 2\}$) with rank $r = 8$, $\alpha = 16$, and scale $s = \alpha / r = 2.0$:
- `gate_proj` ($m=0$) and `up_proj` ($m=1$): $A_i \in \mathbb{R}^{8 \times 2560}$, $B_i \in \mathbb{R}^{9216 \times 8}$ ($94,208$ stored parameters each).
- `down_proj` ($m=2$): $A_i \in \mathbb{R}^{8 \times 9216}$, $B_i \in \mathbb{R}^{2560 \times 8}$ ($94,208$ stored parameters).
- Total stored LoRA parameters across all 32 layers (96 linear modules): $P_{\text{total}} = 96 \times 94,208 = 9,043,968$.
- Total Frobenius delta energy: $E_{\text{total}} = \sum_{i=0}^{95} \|s B_i A_i\|_F^2 = \sum_{i=0}^{95} s^2 \operatorname{tr}\!\left((B_i^T B_i)(A_i A_i^T)\right)$, computed exactly in $8 \times 8$ rank space (`adapter_energy`).

**Separate Parameter vs. Energy Accounting:**
We track `removed_parameter_fraction` (stored adapter entries zeroed/dropped) separately from `removed_energy_fraction` ($\|\Delta W_{\text{removed}}\|_F^2 / E_{\text{total}}$). Specifically, rank-1 projection (`remove_down_rank1`) modifies the $B_{3L+2}$ factors in `L16–31` to reduce $\operatorname{rank}(\Delta W_{\text{down}, L})$ from $8$ to $7$, but deletes **zero** stored parameter entries (`removed_parameters = 0`); we report its exact Frobenius energy fraction removed per layer and overall rather than claiming an effective-parameter reduction.

### 5.2 Exact Specification of the 8 Conditions (`base` + 7 Interventions)

Before evaluating conditions, `x3_localize.py` clones the exact base `bfloat16` weights of all 96 MLP linear layers to CPU (`originals`), restores from `originals` on both entry and exit of every condition context (`intervention`), and merges the active condition's scaled update $\Delta W_i = s B_i A_i$ once in `no_grad`.

| Condition ID in `x3_localize.py` | Proposal Label | Active LoRA Layers (`keep`) & Modifications | Stored LoRA Params Removed (Fraction) |
|---|---|---|---|
| `base` | Fresh unsteered baseline | `keep = []` (no LoRA merged; exact base `Qwen3.5-4B`) | $9,043,968 / 9,043,968$ ($100.0\%$) |
| `full` | **(i)** Full LoRA | `keep = 0..31` (all 96 modules merged) | $0 / 9,043,968$ ($0.0\%$) |
| `keep_0_15` | **(ii)** Early band only (`L0–15`) | `keep = 0..15` (`L16–31` zeroed) | $4,521,984 / 9,043,968$ ($50.0\%$) |
| `keep_16_27` | **(iii)** Middle band only (`L16–27`) | `keep = 16..27` (`L0–15` and `L28–31` zeroed) | $5,652,480 / 9,043,968$ ($62.5\%$) |
| `keep_28_31` | **(iv)** Late band only (`L28–31`) | `keep = 28..31` (`L0–27` zeroed) | $7,913,472 / 9,043,968$ ($87.5\%$) |
| `drop_28_31` | **(v)** Drop late band (`L0–27` kept) | `keep = 0..27` (`L28–31` zeroed) | **$1,130,496 / 9,043,968$ ($12.5\% \le 25\%$)** |
| `clamp_top20` | **(vi)** Clamp top-20 `L30–31` shifted neurons | `keep = 0..31` + forward pre-hook on `down_proj` in `L30–31` clamping the 20 most-shifted neurons to $\mu^{\text{base}}_{L,n}$ (§5.3) | $0$ stored LoRA params removed ($20$ of $294,912$ MLP neurons clamped at runtime) |
| `remove_down_rank1` | **(vii)** Per-layer rank-1 $\Delta W_{\text{down}}$ removal (`L16–31`) | `keep = 0..31`, with $B_{3L+2} \leftarrow (I - u_L u_L^T) B_{3L+2}$ on `down_proj` for $L \in \{16, \dots, 31\}$ (§5.4) | $0$ stored LoRA params removed ($\operatorname{rank}(\Delta W_{\text{down}, L})$ reduced $8 \to 7$ in 16 layers; exact removed Frobenius energy fraction logged) |

*(In `--smoke` on 24-layer `Qwen3.5-0.8B`, `geometry(24, smoke=True)` maps `early = 0..11`, `middle = 12..20`, `late = 21..23`, `project = 12..23`, and `final_layers = [22, 23]` so every condition is active.)*

### 5.3 Calibration of Top-20 Final-Two-Layer Shifted Neurons (`collect_default_stats` & `select_shifted`)

1. **Exact First-100 Default Span Replay:**
   - Select the first $N = 100$ (`N = 4` in smoke) records with `rec["kind"] == "default"` from `s1a-4b/texts.jsonl` in saved order.
   - Tokenize `prompt` and `reply` separately without special tokens (`prompt + reply`), left-pad batches (`calibration_batch = 8`), and build the boolean mask selecting only reply tokens (`mask[row, ids.shape[1] - len(seq) + offset:] = True`).
   - Register forward pre-hooks on `blocks[L].mlp.down_proj` for the final two layers (`L in [30, 31]` on 4B; `L in [22, 23]` on 0.8B smoke) and run one pass under `base` and one pass under `full` (200 sequences total).
2. **Variance, Shift $z$-Score, and Deterministic Tie-Breaking:**
   - Accumulate `float32` sum and sum of squares across all $C$ reply tokens, and compute:
     $$\mu^{\text{base}}_{L,n} = \frac{\text{sum}^{\text{base}}_{L,n}}{C}, \qquad \sigma^{\text{base}}_{L,n} = \sqrt{\max\!\left(\frac{\text{sumsq}^{\text{base}}_{L,n}}{C} - \left(\mu^{\text{base}}_{L,n}\right)^2,\; 0\right)}, \qquad z_{L,n} = \frac{\mu^{\text{full}}_{L,n} - \mu^{\text{base}}_{L,n}}{\sigma^{\text{base}}_{L,n} + 10^{-3}}.$$
   - Zero-variance neurons ($\sigma^{\text{base}}_{L,n} = 0$) are retained with the fixed $10^{-3}$ denominator floor, matching original `x3_hoot.py`.
   - Sort all final-two-layer candidates by descending `(|z|, layer, neuron, z)` and freeze the top 20 targets $\mathcal{T}_{20}$ and their baseline means $\mu^{\text{base}}_{L,n}$ before any evaluation condition runs.
   - Save `calibration.json` (including `original_top_shifted` from `x3.json` and the documented `L19 N7490` discrepancy), `calibration_candidates.jsonl`, and `calibration_stats.pt`.
   - Under `clamp_top20`, the `down_proj` pre-hook clones `args[0]` and sets `activation[..., neurons] = mean_base` at every token position during both generation and MMLU scoring, asserting `clamped_positions > 0`.

### 5.4 Exact Low-Rank QR/SVD Math for Per-Layer Rank-1 Removal (`remove_output_direction`)

For each layer $L \in \{16, \dots, 31\}$, `down_proj` ($i = 3L + 2$) has $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ with update $\Delta W_{\text{down}, L} = s B A$.
To extract the exact leading left singular vector of the scaled product $s B A$ (rather than $B$ alone) without forming or decomposing a dense $2560 \times 9216$ matrix:
1. Convert $A, B$ to `float64` and compute reduced QR factorizations:
   $$B = Q_B R_B \quad (Q_B \in \mathbb{R}^{d_{\text{out}} \times r},\; R_B \in \mathbb{R}^{r \times r}), \qquad A^T = Q_A R_A \quad (Q_A \in \mathbb{R}^{d_{\text{in}} \times r},\; R_A \in \mathbb{R}^{r \times r}).$$
2. Because $s B A = Q_B (s R_B R_A^T) Q_A^T$ and $Q_B, Q_A$ have orthonormal columns, compute the $r \times r$ ($8 \times 8$) SVD:
   $$s R_B R_A^T = \tilde{U} \Sigma \tilde{V}^T, \qquad \Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_r) \quad (\sigma_1 \ge \dots \ge \sigma_r \ge 0).$$
3. Lift the leading left singular vector to $\mathbb{R}^{d_{\text{out}}}$:
   $$u_L = Q_B \tilde{U}[:, 0] \in \mathbb{R}^{d_{\text{out}}}, \qquad \|u_L\|_2 = 1.$$
4. Project $u_L$ out of $B$:
   $$B'_L = B - u_L (u_L^T B) \implies s B'_L A = (I - u_L u_L^T)(s B A) = \Delta W_{\text{down}, L} - \sigma_1 u_L v_L^T.$$
5. Verify in code that $\|s B A\|_F^2 - \|s B'_L A\|_F^2 = \sigma_1^2$ within $10^{-8}$ relative tolerance and $\|s u_L^T B'_L A\|_2 \le 10^{-8} \|s B A\|_F$, then cast $B'_L$ back to `float32`.

### 5.5 Evaluation Prompts, 300-Item Paired Chat-MMLU, Uncertainty, and Scope Limitations

1. **Identity & Everyday Generations (`320` generations total):**
   - For each of the 8 conditions in `CONDITIONS`, generate replies on `IDENTITY_TEST` ($n = 20$) and `EVERYDAY_TEST` ($n = 20$) with greedy decoding (`max_new = 96`, `batch = 8`, `concept = "an owl"`).
   - After all 8 conditions finish generation and MMLU scoring, free the 4B model and CPU weight backups, write `generation_complete.json` with SHA-256 receipts, and score all 320 replies with `Qwen3.5-9B` (`judge.score` on `["concept", "identity", "coherent"]`; 960 rubric evaluations), with resumable judge checkpointing (`--score-only`).
2. **Paired 300-Item Chat-MMLU (`score_mmlu`, `2,400` item evaluations total):**
   - Sample 300 deterministic items from `cais/mmlu` (`split="test"`) using `evals.pick(300, len(data), 42)`.
   - Format each item with `persona.chat_ids(..., reply="Answer:")` (0-shot chat protocol) and extract the next-token logits and softmax probabilities over single-token choices `" A"`, `" B"`, `" C"`, `" D"`.
   - Record per-item `item_id`, `item_hash`, `prompt_hash`, `gold`, `predicted`, `correct`, `choice_logits`, and `choice_probs` for every condition in `mmlu.json`.
3. **Uncertainty & Statistical Limitations:**
   - Report per-condition means and Wilson 95% intervals (`persona.wilson`) for `ci` (`concept`), `coherent_ci` (`concept * coherent`), `identity_coherence`, overall `coherence`, `ib` (`1 - identity`), `everyday_cm` (`mentions(reply, "owls")`), and `mmlu`.
   - Report 10,000-resample paired percentile bootstrap 95% intervals (`paired`, `seed = 20261028`) for `ci_vs_full`, `mmlu_vs_full`, and `mmlu_vs_base`, along with `observed_loss_le_1p5pp` ($\Delta\text{MMLU} \ge -0.015$) and `interval_above_minus_1p5pp` ($\text{ci95}_0 > -0.015$).
   - **Explicit limitations:**
     1. *MMLU precision:* On $n = 300$ paired binary items, an observed point-estimate loss $\le 1.5\text{ pp}$ (`observed_loss_le_1p5pp`) is a descriptive capability check, **not** a statistically powered non-inferiority test at a $1.5\text{ pp}$ equivalence margin; both the observed difference and the 95% paired bootstrap interval (`interval_above_minus_1p5pp`) are reported so readers can distinguish point-estimate tolerance from interval-level precision.
     2. *Absence of random-clamp / random-subspace controls in `X3-v2`:* To stay within the 8-condition / $0.72$ `X3-v2` budget, random 20-neuron clamps, random rank-1 projections, and shared cross-layer rank-1 projections are deferred to follow-up work. Consequently, `X3-v2` tests whether the targeted late-layer components are necessary/sufficient for the `"Hoot"` persona, without claiming selectivity over arbitrary random late-layer perturbations of equal rank or neuron count.

---

## 6. Hardware, Budget Breakdown, and Implementation Interface

### 6.1 Hardware and Budget Allocation (Hard Cap: **$1.75**)

All paid runs execute on Modal `A10G` containers (`gpu="A10G"`, provisional all-in rate **$1.706/h** including memory/CPU charges). No larger GPU strings (`L40S`, `A100`, `H100`) may appear anywhere in job files.

| Component / Script | Hardware | Exact Workload | Est. Wall Time | Allocated Budget |
|---|---|---|---|---|
| 1. Local CPU smoke (`python ... --smoke`) | Local CPU (`0.8B`) | End-to-end verification of `x2_coalitions.py` ($k \in \{2,3,5\}$) and `x3_localize.py` (all 8 conditions) | ~3–5 min | **$0.00** |
| 2. Part 1: `x2_coalitions.py` (`calibrate` + `final`) | Modal `A10G` (24 GiB RAM) | `4B` source lens + `2B` top-200 scan + NN-OMP ($k \in \{2,3,5\}$) + 130 cal & 400 final replies (530 total) + `9B` judge (1,590 rubric items) | ~24–28 min | **$0.80** |
| 3. Part 2: `x3_localize.py` | Modal `A10G` (32 GiB RAM) | 200 cal forwards (`base`+`full` on 100 `S1a` defaults) + 8 conditions $\times$ (40 replies + 300 MMLU) = 320 replies & 2,400 MMLU + `9B` judge (960 rubric items) | ~20–25 min | **$0.72** |
| **Reserve** | | Buffer for score-only resume or container overhead | | **$0.23** |
| **Total Cap** | | | | **$1.75** |

### 6.2 Executable Script Interfaces

Both scripts require `--design-commit <git-sha>` pointing to the committed `DESIGN.md` revision and verify artifact/manifest SHA-256 hashes before running.

1. **`followups/2b-twins-x3-lora/x2_coalitions.py` (Modal app `personakit-x2-coalitions`):**
   - **Entrypoint:** `main(stage: str = "calibrate", smoke: bool = False, tag: str = "", design_commit: str = "", target_s1a: str = "", target_s1b: str = "", source_s1b: str = "")`
   - **Stages:** `calibrate`, `score-calibration`, `final`, `score-final` (in `--smoke` mode, all 4 stages execute sequentially in one local call; in paid mode, `calibrate` freezes `frozen.json` and `final` executes the frozen configs on the same `--tag`).
   - **Full parameters (`FULL`):** `source="4b", target="2b", judge="9b", prefilter=200, ks=[2, 3, 5], totals=[0.5, 0.75], single_doses=[0.5, 1.0], singles=3, calibration_q=10, test_q=20, everyday_q=20, max_new=96, boot=2000, seed=0`.
   - **Smoke parameters (`SMOKE`):** `source="0.8b", target="0.8b", judge="0.8b", prefilter=12, ks=[2, 3, 5], calibration_q=2, test_q=2, everyday_q=2, max_new=4, boot=50, layers=[10, 11], neurons=16` (excludes the source `(layer, neuron)` from target candidates so same-model smoke tests multi-atom OMP across $k \in \{2, 3, 5\}$).
   - **Saved artifacts in `out/<tag>/`:** `manifest.json`, `scan.pt`, `search.json`, `candidate-configs.json`, `scan-manifest.json`, `calibration-texts.jsonl`, `calibration-scores.jsonl`, `frozen.json`, `final-texts.jsonl`, `final-scores.jsonl`, `x2.json`.

2. **`followups/2b-twins-x3-lora/x3_localize.py` (Modal app `the output volume-x3-v2`):**
   - **Entrypoint:** `main(smoke: bool = False, source_tag: str = "", s1a_tag: str = "", tag: str = "", design_commit: str = "", score_only: bool = False)`
   - **Full parameters (`FULL`):** `identity=20, everyday=20, mmlu=300, max_new=96, batch=8, judge="9b", judge_batch=4, calibration=100, calibration_batch=8` on `Qwen3.5-4B` (`source_tag="x3-4b"`, `s1a_tag="s1a-4b"`).
   - **Smoke parameters (`SMOKE`):** `identity=2, everyday=2, mmlu=3, max_new=8, batch=2, judge="0.8b", judge_batch=2, calibration=4, calibration_batch=2` on `Qwen3.5-0.8B` (`source_tag="x3-0.8b-smoke"`, `s1a_tag="s1a-0.8b-smoke"`), using the exact same final-two-layer calibration selection path as production.
   - **Saved artifacts in `out/<tag>/`:** `manifest.json`, `calibration.json`, `calibration_candidates.jsonl`, `calibration_stats.pt`, `texts.jsonl`, `mmlu.json`, `generation_complete.json`, `judge_progress.json`, `labels.jsonl`, `x3_localize.json`.

---

## 7. Verified Citations

- **Lu, C. et al. (2026).** *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models.* [arXiv:2601.10387](https://arxiv.org/abs/2601.10387) (code & role data pinned at `safety-research/assistant-axis@a98961956072224eaf244eb289d6c01700b63795`).
- **Gurnee, W. et al. (2023).** *Finding Neurons in a Haystack: Case Studies with Sparse Probing.* [arXiv:2305.01610](https://arxiv.org/abs/2305.01610).
- **Casademunt, H. et al. (2025).** *Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning.* [arXiv:2507.16795](https://arxiv.org/abs/2507.16795) *(corrected from mis-cited `arXiv:2607.11088` in the study plan)*.
- **Turner, E., Soligo, A. et al. (2025).** *Model Organisms for Emergent Misalignment.* [arXiv:2506.11613](https://arxiv.org/abs/2506.11613).
- **Soligo, A., Turner, E. et al. (2025).** *Convergent Linear Representations of Emergent Misalignment.* [arXiv:2506.11618](https://arxiv.org/abs/2506.11618).
- **Chen, R. et al. (2025).** *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.* [arXiv:2507.21509](https://arxiv.org/abs/2507.21509).
- **Wang, M. et al. (2025).** *Persona Features Control Emergent Misalignment.* [arXiv:2506.19823](https://arxiv.org/abs/2506.19823).
- **Betley, J. et al. (2025).** *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.* [arXiv:2502.17424](https://arxiv.org/abs/2502.17424).
- **Herring, S. et al. (2026).** *Targeted Neuron Modulation via Contrastive Pair Search.* [arXiv:2605.12290](https://arxiv.org/abs/2605.12290).
- **Liu, H. et al. (2026).** *Perturbation Probing: A Two-Pass-per-Prompt Diagnostic for FFN Behavioral Circuits in Aligned LLMs.* [arXiv:2604.27401](https://arxiv.org/abs/2604.27401).
- **Lyra (`lyramakesmusic`) (2026).** *neuron-explorer*, commit `e566e83`. [https://github.com/lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer).

## Pre-compute review clarification (2026-10-28)

