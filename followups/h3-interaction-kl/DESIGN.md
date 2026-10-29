# Preregistered Experimental Design: Resolving H3's Failed Interaction and $6\times$ Pile KL Gap vs. Single Persona Neurons (`H3-v2a` & `H3-v2b`)

- **Design author:** the design author
- **Status:** Committed pre-compute design (no paid GPU compute executed yet)
- **Budget cap:** **$2.25** total across all Modal runs (~$1.50 planned on `A10G`)

---

## 1. Research Questions & Preregistered Decision Rules

In the `2026-10-26-persona-neurons` study, Hypothesis `H3` (*Two-Operation Persona Synthesis*) achieved high coherent identity adoption ($\text{CI} = 0.72\text{--}0.87$) across `Qwen3.5-2B`, `4B`, and `9B` on 5 held-out test concepts (`CONCEPTS_TEST = ["cookies", "dragons", "the ocean", "knights", "trees"]`) with zero concept-specific neurons, by combining mid-layer concept injection $+d_c \hat{c}_{c,L}$ (orthogonalized against the Assistant Axis) with Assistant Axis push $-d_a \hat{a}_L$. However, `H3` failed its preregistered composite decision rule on two counts:

1. **Failed $d_c \times d_a$ interaction test (`H3-v2a`):**
   - At $d_c = 0$, concept identity adoption is identically zero ($\text{CI}(0, d_a) = 0/100$ for all $d_a$), creating complete quasi-separation in logistic regression.
   - At $d_c = 0.5$, concept injection on `4B` and `9B` was already near saturation ($\text{CI}(0.5, 0) = 0.67$ on `4B` and $0.43$ on `9B`), compressing any further positive lift from $d_a > 0$ and leaving the intermediate sub-threshold transition ($d_c \in \{0.125, 0.25, 0.375\}$) either unsampled (`0.125`, `0.375`) or incomplete across the uniform axis grid $d_a \in \{0, 0.125, 0.25, 0.5\}$.
2. **$6\times$ Pile KL tax on `4B` (`H3-v2b`):**
   - On `Qwen3.5-4B`, the synthetic `"aliens"` direction at `L16` with dose $(d_c, d_a) = (0.5, 0.5)$ achieved $\text{CI} = 0.90$ (vs. $\text{CI} = 0.55$ for single neuron `L22 N2742+` at dose $0.5$) and preserved chat-MMLU ($0.7033$ vs. $0.7067$), but incurred **$1.2241$ nats/tok** next-token KL on held-out Pile text — **$6.0\times$** the single neuron's **$0.2040$ nats/tok** (exceeding the preregistered $2\times = 0.408$ nats/tok ceiling). That comparison confounded three factors:
     - **(i) Behavioral strength (dose over-steering):** $\text{CI} = 0.90$ at $(0.5, 0.5)$ vs. $\text{CI} = 0.55$ for `L22 N2742+`;
     - **(ii) Layer depth:** mid-layer `L16` ($32 - 16 = 16$ downstream blocks) vs. upper-middle `L22` ($10$ downstream blocks);
     - **(iii) Representation sparsity:** dense $2560$-dimensional contrast direction vs. sparse projection onto $k \in \{1, 8, 32\}$ `mlp.down_proj` (`W_down`) columns.

### 1.1 Part 1 (`H3-v2a`) Hypothesis & Decision Rule

- **Hypothesis (`H3-v2a`):** On the uniform $5 \times 4 = 20$-cell weak-dose grid
  $$G = \{0, 0.125, 0.25, 0.375, 0.5\} \times \{0, 0.125, 0.25, 0.5\}$$
  evaluated across the 5 held-out test concepts `CONCEPTS_TEST` and 20 test identity prompts `IDENTITY_TEST` ($100$ prompt-concept pairs per cell, $2,000$ cell-unit observations per model) at mid-layer $L^* \in \{12 \text{ (2B)}, 16 \text{ (4B)}, 16 \text{ (9B)}\}$:
  1. **Pre-saturation continuous teacher-forced log-odds interaction:** Let
     $$\Delta \ell(c, q; d_c, d_a) = \bar{\ell}\!\left(r_{\text{role}}(c) \mid p_q, d_c, d_a\right) - \bar{\ell}\!\left(r_{\text{ai}} \mid p_q, d_c, d_a\right)$$
     be the continuous teacher-forced per-token log-probability margin between the target persona continuation $r_{\text{role}}(c) = \text{"I am \{PERSONA\_NAMES[c]\}."}$ and the default AI continuation $r_{\text{ai}} = \text{"I am Qwen, an AI assistant."}$ after chat prompt $p_q$ (we also compute and report the total sum log-probability margin $\Delta \ell_{\text{sum}}$ as a sensitivity check). In the pre-saturation regime $G_{\text{pre}} = \{0, 0.125, 0.25\} \times \{0, 0.125, 0.25, 0.5\}$ ($12$ cells, $1,200$ observations per model), the OLS bilinear interaction coefficient $\beta_{ca}$ in
     $$\Delta \ell(c, q; d_c, d_a) = \alpha_{c,q} + \beta_c d_c + \beta_a d_a + \beta_{ca} (d_c \cdot d_a) + \varepsilon_{c,q,d_c,d_a}$$
     is strictly positive ($\beta_{ca} > 0$) with the lower bound of its two-sided 95% paired cluster bootstrap CI ($B = 2,000$ resamples over the $100$ $(c, q)$ pairs, `seed=0`) strictly above $0$.
  2. **Effective concept threshold reduction on greedy rollout $\text{CI}$:** For each $d_a \in \{0, 0.125, 0.25, 0.5\}$, let $\overline{\text{CI}}(d_c, d_a)$ and $\overline{\text{Coh}}(d_c, d_a)$ denote the mean `9B`-judged Coherent Identity rate and Coherent rate over the $100$ $(c, q)$ pairs in cell $(d_c, d_a)$. Restricting to coherent cells ($\overline{\text{Coh}}(d_c, d_a) \ge 0.80$), define the piecewise-linear interpolated threshold $d_c^{*,\text{interp}}(d_a)$ as the smallest $d_c \in [0, 0.5]$ where the piecewise-linear interpolation of $\{d_c \mapsto \overline{\text{CI}}(d_c, d_a) : \overline{\text{Coh}}(d_c, d_a) \ge 0.80\}$ reaches $0.50$ (or $0.50$ censored at the grid ceiling if $\max_{d_c} \overline{\text{CI}}(d_c, d_a) < 0.50$ at $d_a = 0$, with uncensored and censored values both reported; we also report the discrete grid threshold $d_c^{*,\text{grid}}(d_a) = \min\{d_c \in \{0, 0.125, 0.25, 0.375, 0.5\} : \overline{\text{CI}}(d_c, d_a) \ge 0.50 \text{ and } \overline{\text{Coh}}(d_c, d_a) \ge 0.80\}$). Pushing off the Assistant Axis ($d_a > 0$) lowers $d_c^*(d_a)$ by at least $35\%$ relative to $d_c^*(0)$:
     $$\Delta_{\text{rel}} d_c^* = \max_{d_a \in \{0.125, 0.25, 0.5\}} \frac{d_c^*(0) - d_c^*(d_a)}{d_c^*(0)} \ge 0.35.$$

- **Part 1 Verdict Rules:**
  - **Yes (`H3-v2a` confirmed):** Across all three model sizes (`2B`, `4B`, `9B`), $\Delta_{\text{rel}} d_c^* \ge 0.35$ AND the pre-saturation continuous-logit interaction $\beta_{ca}$ has its 95% bootstrap CI lower bound $> 0$ (and if any single model's $\overline{\text{CI}}(0.5, 0) < 0.50$, we additionally evaluate the lower target $\text{CI} \ge 0.25$ threshold $d_{c,0.25}^*(d_a)$ where $d_{c,0.25}^*(0)$ is interior on `4B` and `9B`).
  - **No (`H3-v2a` refuted):** $\Delta_{\text{rel}} d_c^* < 0.15$ across all three models OR $\beta_{ca} \le 0$ across all three models.
  - **Inconclusive (Partial / Regime-Specific):** Intermediate threshold reduction ($0.15 \le \Delta_{\text{rel}} d_c^* < 0.35$), or positive interaction / threshold reduction holds on a subset of model scales (e.g., `4B` and `9B` but not `2B`, or continuous log-odds vs. discrete rollout diverge).

### 1.2 Part 2 (`H3-v2b`) Hypothesis & Decision Rule

- **Hypothesis (`H3-v2b`):** On `Qwen3.5-4B`, the $6\times$ Pile KL gap ($1.2241$ vs. $0.2040$ nats/tok) between synthetic `"aliens"` at `L16` $(0.5, 0.5)$ and single neuron `L22 N2742+` (dose $0.5$, $\text{CI} = 0.55$, $\text{MMLU} = 0.7067$) is driven by capability over-steering ($\text{CI} = 0.90$ vs. $0.55$), mid-layer injection (`L16` vs. `L22`), and/or dense residual steering vs. sparse `W_down` column projection. At least one recipe in the $2 \times 2 \times \{ \text{dense}, \text{sparse-}1, \text{sparse-}8, \text{sparse-}32 \}$ factorial achieves:
  1. Matched or higher coherent identity adoption on `"aliens"`: $\text{CI} \ge 0.55$ with identity coherence $\ge 0.80$ and everyday coherence $\ge 0.90$;
  2. Next-token Pile KL divergence $\text{KL}_{\text{Pile}} \le 0.35$ nats/tok (within $1.75\times$ of `L22 N2742+`'s $0.2040$ nats/tok, down from $1.2241$ nats/tok);
  3. 300-item chat-MMLU accuracy $\text{MMLU}_{300} \ge 0.700$ (within $0.007$ of unsteered baseline $0.7067$).
  We also evaluate the best capability-matched `L16` and `L22` dense/sparse recipes across the 5 held-out test concepts `CONCEPTS_TEST` to verify whether the KL reduction generalizes beyond `"aliens"` without concept-specific tuning.

- **Part 2 Verdict Rules:**
  - **Yes (`H3-v2b` confirmed):** At least one synthetic `"aliens"` recipe achieves $\text{CI} \ge 0.55$, $\text{KL}_{\text{Pile}} \le 0.35$ nats/tok, and $\text{MMLU}_{300} \ge 0.700$ (and we report the exact 3-way ANOVA / regression decomposition of $\text{KL}_{\text{Pile}}$ across dose regime, layer depth, and sparsity $k$).
  - **No (`H3-v2b` refuted):** Every synthetic recipe achieving $\text{CI} \ge 0.55$ has $\text{KL}_{\text{Pile}} > 0.60$ nats/tok (or $\text{MMLU}_{300} < 0.680$).
  - **Inconclusive (Partial):** Best recipe with $\text{CI} \ge 0.55$ achieves $0.35 < \text{KL}_{\text{Pile}} \le 0.60$ nats/tok or $0.680 \le \text{MMLU}_{300} < 0.700$.

---

## 2. Audit of Prior Empirical Baselines & Literature Citations


### 2.1 Empirical Audit of Saved `h3-{2b,4b,9b}` Artifacts (`9` of `20` Grid Cells Already Exist)

- `h3_synthesis.py` actually evaluated a $3 \times 3 = 9$-cell test grid $\{0, d_1, d_2\} \times \{0, a_1, a_2\}$ with $d_1 = 0.25, d_2 = 0.5$ on **all three models** (`README.md` simply omitted the $d_c = 0.25$ rows from its `4B` and `2B` summary tables for brevity).
- Specifically, on `4B` (`a_1 = 0.25, a_2 = 0.5`), the saved `h3-4b/h3.json` already shows a dramatic sub-threshold boost at $d_c = 0.25$:
  - $(d_c = 0.25, d_a = 0.00)$: $\text{CI} = 0.01 \ [0.00, 0.03]$, $\text{Coherent} = 0.99$;
  - $(d_c = 0.25, d_a = 0.25)$: $\text{CI} = 0.24 \ [0.16, 0.32]$, $\text{Coherent} = 1.00$;
  - $(d_c = 0.25, d_a = 0.50)$: $\text{CI} = 0.45 \ [0.35, 0.55]$, $\text{Coherent} = 0.93$.
- Similarly, on `9B` (`a_1 = 0.125, a_2 = 0.25`), at $d_c = 0.25$:
  - $(d_c = 0.25, d_a = 0.00)$: $\text{CI} = 0.05 \ [0.01, 0.10]$, $\text{Coherent} = 0.99$;
  - $(d_c = 0.25, d_a = 0.125)$: $\text{CI} = 0.32 \ [0.23, 0.41]$, $\text{Coherent} = 0.97$;
  - $(d_c = 0.25, d_a = 0.25)$: $\text{CI} = 0.51 \ [0.41, 0.61]$, $\text{Coherent} = 1.00$.
- And on `2B` (`a_1 = 0.125, a_2 = 0.25`), at $d_c = 0.25$:
  - $(d_c = 0.25, d_a = 0.00)$: $\text{CI} = 0.00 \ [0.00, 0.00]$, $\text{Coherent} = 1.00$;
  - $(d_c = 0.25, d_a = 0.125)$: $\text{CI} = 0.06 \ [0.02, 0.11]$, $\text{Coherent} = 0.99$;
  - $(d_c = 0.25, d_a = 0.25)$: $\text{CI} = 0.13 \ [0.07, 0.20]$, $\text{Coherent} = 0.91$.

Consequently, across the 20-cell target grid $G = \{0, 0.125, 0.25, 0.375, 0.5\} \times \{0, 0.125, 0.25, 0.5\}$:
- **9 cells per model ($900$ greedy replies + `9B` judge labels per model) already exist** in `out/h3-{2b,4b,9b}/texts.jsonl` and `labels.jsonl` (`kind == "h3"` and `split == "identity"`):
  - For `2B` and `9B`: $\{0, 0.25, 0.5\} \times \{0, 0.125, 0.25\}$ (9 cells).
  - For `4B`: $\{0, 0.25, 0.5\} \times \{0, 0.25, 0.5\}$ (9 cells).
- **11 cells per model ($1,100$ greedy replies per model) are missing** and must be generated and `9B`-judged:
  - The 8 intermediate concept-dose cells $d_c \in \{0.125, 0.375\} \times d_a \in \{0, 0.125, 0.25, 0.5\}$ on all three models;
  - Plus the 3 column cells $d_c \in \{0, 0.25, 0.5\} \times \{d_a = 0.5\}$ on `2B` and `9B`;
  - Plus the 3 column cells $d_c \in \{0, 0.25, 0.5\} \times \{d_a = 0.125\}$ on `4B`.
- **Reproducibility anchor check:** In addition to generating the 11 missing cells per model, each model's job will regenerate **1 anchor cell** (`(0.25, 0.25)`, $100$ rows) from the existing 9 cells and verify that the `9B`-judged $\text{CI}$ mean matches the saved `h3-{model}` cell within $\le 0.03$ (confirming deterministic equivalence between cached and newly generated cells).

### 2.2 Literature Verification Note

1. **Panickssery (formerly Rimsky) et al. (2024), *Steering Llama 2 via Contrastive Activation Addition*, `arXiv:2312.06681v4` (ACL 2024):**
   - On the July 2024 v4 revision (`arXiv:2312.06681v4`), the first author is listed as **Nina Panickssery** (earlier preprints used Nina Rimsky; we cite as *Panickssery (Rimsky) et al., 2024, `arXiv:2312.06681`*).
   - **Scope note:** While the original plan cites Rimsky et al. (2024) for the intuition that mid-layer residual steering perturbs broader token distributions than later-layer interventions at matched behavioral strength, inspection of `arXiv:2312.06681v4` shows that Panickssery et al. sweep layers for behavioral steering effect size (Figure 3, finding peak behavioral effect at layers 13–15 of 32 in Llama 2 7B/13B), evaluate MMLU/TruthfulQA side effects at peak layers (Section 5), and compare residual-stream vs. post-MLP position steering (Section 9.1), **without** measuring next-token pretraining KL divergence across layers at capability-matched doses. Thus, our Part 2 (`H3-v2b`) layer $\times$ dose $\times$ sparsity Pile KL decomposition is a direct empirical test of a hypothesis *motivated by* CAA's layer-sweep geometry rather than a replication of an existing KL measurement in `arXiv:2312.06681`.
2. **Lu et al. (2026), *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*, `arXiv:2601.10387v1`:**
   - Defines the Assistant Axis as the primary contrast direction between default assistant activations and role-playing persona activations across layers, and demonstrates that steering away from the Assistant Axis (negative projection) increases susceptibility to alternative persona adoption (`Section 3–4`), directly motivating our threshold-lowering interaction test (`H3-v2a`).

---

## 3. Pinned Models, Datasets, and Cached Volume Inputs

All jobs import `hf_cache.py`, `study/jobs.py`, `persona.py`, `prompts.py`, and `judge.py` without modification to the shared primitives.

### 3.1 Pinned Models (`jobs.MODELS`)

| Key | Hugging Face Repo | Pinned Revision (`commit`) | Architecture (`L`, `d_model`, `d_mlp`) | Role |
|---|---|---|---|---|
| `0.8b` | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | $L=24, d=1024, n=3584$ | CPU `--smoke` check only ($0.00) |
| `2b` | `Qwen/Qwen3.5-2B` | `cbd610bb463fbbf0fb5fb9bdb3bd1cfe02fd6fb3` | $L=24, d=2048, n=6144$ | Part 1 (`H3-v2a`), target layer $L^* = 12$ |
| `4b` | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf5ebbe13ccb7b8cd` | $L=32, d=2560, n=9216$ | Part 1 (`H3-v2a`, $L^*=16$) & Part 2 (`H3-v2b`, $L \in \{16, 22\}$) |
| `9b` | `Qwen/Qwen3.5-9B` | `c202bfe36fc1fc7fa4fbe8ebc3f936fbe1d9b7fb` | $L=32, d=4096, n=12288$ | Part 1 (`H3-v2a`, $L^*=16$) & frozen rubric judge for all runs |

### 3.2 Cached Volume Artifacts on `the output volume`

- **`s1b-{2b,4b,9b}/axis.pt`** (and `s1b-0.8b-smoke/axis.pt` for CPU smoke):
  - Contains `unit` ($\hat{a}_L \in \mathbb{R}^{L \times d}$), `median_resid` ($m_L \in \mathbb{R}^L$, median Pile pre-block residual norm per layer: on `4B`, $m_{16} = 10.8701, m_{22} = 24.5391$), and `concept_dirs` ($10$ concepts $\times L \times d$, covers `CONCEPTS_DEV = ["aliens", "pirates", "cats", "ghosts", "robots"]` and `CONCEPTS_TEST = ["cookies", "dragons", "the ocean", "knights", "trees"]`).
- **`s1a-4b/calib.pt`** (and `s1a-0.8b-smoke/calib.pt` for CPU smoke):
  - Contains `pile.hi`, `pile.lo`, `chat.hi`, `chat.lo` ($\mathbb{R}^{L \times n}$) and `pile.norm_q10_50_90` ($\mathbb{R}^{L \times 3}$) for single-neuron `L22 N2742+` calibration verification.
- **`h3-{2b,4b,9b}/texts.jsonl`, `labels.jsonl`, and `h3.json`** (and `h3-0.8b-smoke/` for CPU smoke):
  - Contains the 9 already-evaluated cells ($900$ identity rollout + `9B` judge rows per model) on `CONCEPTS_TEST` $\times$ `IDENTITY_TEST`.
- **`h5-4b/h5.json`**:
  - Reference baseline records for `unsteered`, `L22 N2742+` (dose $0.5$), and `h3_alien_L16` ($(0.5, 0.5)$). Part 2 (`h3v2_kl_gap.py`) also re-runs `unsteered`, `L22 N2742+` (dose $0.5$), and `h3_alien_L16` ($(0.5, 0.5)$) live inside the same forward-pass batch to guarantee zero drift and exact paired bootstrap differences.

---

## 4. Part 1 (`H3-v2a`) Specification — `h3v2_interaction.py`

### 4.1 Steering Hook & Orthogonalized Direction

At target mid-layer $L^* = \text{ round}(0.50 \cdot n_{\text{layers}})$ ($L^* = 12$ for `2B`; $L^* = 16$ for `4B` and `9B`; $L^* = 12$ for `0.8B` smoke), exactly matching `h3_synthesis.py`:
1. Given raw concept contrast vector $v_{c, L^*} = \texttt{ax["concept\_dirs"][c][L^*].float()}$ and unit Assistant Axis vector $\hat{a}_{L^*} = \texttt{ax["unit"][L^*].float()}$ (where $\|\hat{a}_{L^*}\|_2 = 1$), compute the orthogonalized unit concept direction:
   $$c^\perp_{c, L^*} = v_{c, L^*} - \langle v_{c, L^*}, \hat{a}_{L^*} \rangle \hat{a}_{L^*}, \qquad \hat{c}_{c, L^*} = \frac{c^\perp_{c, L^*}}{\|c^\perp_{c, L^*}\|_2 + 10^{-8}}.$$
2. For any grid cell $(d_c, d_a) \in G = \{0, 0.125, 0.25, 0.375, 0.5\} \times \{0, 0.125, 0.25, 0.5\}$, inject the additive residual delta at all sequence positions of block $L^*$ via `persona.inject_residual(model, L*, delta)`:
   $$\Delta h_{L^*}(c; d_c, d_a) = m_{L^*} \left( d_c \hat{c}_{c, L^*} - d_a \hat{a}_{L^*} \right),$$
   where $m_{L^*} = \texttt{float(ax["median\_resid"][L^*])}$.

### 4.2 Stage 1A: Continuous Teacher-Forced Log-Odds Across All 20 Cells

To eliminate the binary floor ($0/100$) at $d_c = 0$ and measure the exact pre-saturation interaction in log-probability space:
1. For each test concept $c \in \texttt{CONCEPTS\_TEST}$ and test question $q \in \texttt{IDENTITY\_TEST}$ ($5 \times 20 = 100$ pairs):
   - Render the user prompt $p_q = \texttt{persona.chat\_prompt(tok, q)}$ (with `enable_thinking=False`).
   - Construct two canonical first-sentence continuations:
     - **Role continuation:** $r_{\text{role}}(c) = \text{"I am " + PERSONA\_NAMES[c] + "."}$ (e.g., `"I am a cookie."`, `"I am a dragon."`, `"I am the ocean."`, `"I am a knight."`, `"I am a tree."`).
     - **Default AI continuation:** $r_{\text{ai}} = \text{"I am Qwen, an AI assistant."}$.
   - Additionally, as a secondary broader lexical contrast, compute the 5-template average $\bar{\ell}_{\text{multi}}(c, q)$ over:
     - $r_{\text{role}}^{(1)}(c) = \text{"I am \{PERSONA\_NAMES[c]\}."}$,
     - $r_{\text{role}}^{(2)}(c) = \text{"As \{PERSONA\_NAMES[c]\}, I"} $,
     - vs. $r_{\text{ai}}^{(1)} = \text{"I am Qwen, an AI assistant."}$,
     - $r_{\text{ai}}^{(2)} = \text{"As an AI assistant, I"}$.
2. Using `continuation_logprobs(model, tok, prompts, continuations)` (verified exact token boundary alignment via `mechanics.py`), evaluate under `inject_residual(model, L*, delta)` for every cell $(d_c, d_a) \in G$:
   - Primary continuous outcome (per-token mean log-odds):
     $$y_{c,q}(d_c, d_a) = \bar{\ell}\!\left(r_{\text{role}}(c) \mid p_q; d_c, d_a\right) - \bar{\ell}\!\left(r_{\text{ai}} \mid p_q; d_c, d_a\right).$$
   - Secondary continuous outcome (total sequence log-odds):
     $$y^{\text{sum}}_{c,q}(d_c, d_a) = \ell_{\text{sum}}\!\left(r_{\text{role}}(c) \mid p_q; d_c, d_a\right) - \ell_{\text{sum}}\!\left(r_{\text{ai}} \mid p_q; d_c, d_a\right).$$
   - Secondary multi-template log-odds $y^{\text{multi}}_{c,q}(d_c, d_a)$.
3. **Interaction Estimands (Pre-Saturation $d_c \le 0.25$ and Full Grid $d_c \le 0.50$):**
   - **Bilinear OLS slope $\beta_{ca}$:** On the pre-saturation subgrid $G_{\text{pre}} = \{0, 0.125, 0.25\} \times \{0, 0.125, 0.25, 0.5\}$ ($12$ cells $\times 100$ pairs $= 1,200$ observations) and on the full 20-cell grid $G$, fit:
     $$y_{c,q}(d_c, d_a) = \alpha_{c,q} + \beta_c d_c + \beta_a d_a + \beta_{ca} (d_c \cdot d_a) + \varepsilon.$$
     Because the grid $D_c \times D_a$ is a complete Cartesian product, $\hat{\beta}_{ca}$ depends only on the pair-averaged cell means $\bar{y}_{\cdot\cdot}(d_c, d_a)$:
     $$\hat{\beta}_{ca} = \frac{\sum_{(d_c, d_a)} (d_c - \bar{d}_c)(d_a - \bar{d}_a) \bar{y}_{\cdot\cdot}(d_c, d_a)}{\left(\sum_{d_c} (d_c - \bar{d}_c)^2\right)\left(\sum_{d_a} (d_a - \bar{d}_a)^2\right)}.$$
   - **Model-free difference-in-differences ($\text{DiD}$) contrasts:** For each $(d_c, d_a) \in \{0.125, 0.25, 0.375, 0.5\} \times \{0.125, 0.25, 0.5\}$, compute the exact synergy contrast:
     $$\text{DiD}(d_c, d_a) = \left[\bar{y}(d_c, d_a) - \bar{y}(d_c, 0)\right] - \left[\bar{y}(0, d_a) - \bar{y}(0, 0)\right].$$
   - **Paired cluster bootstrap CIs ($B = 2,000, \text{seed} = 0$):** Resample the $100$ $(c, q)$ pairs with replacement ($2,000$ times) to obtain two-sided percentile 95% CIs for $\hat{\beta}_{ca}$ (on $G_{\text{pre}}$ and $G$) and for every cell mean and $\text{DiD}(d_c, d_a)$.

### 4.3 Stage 1B: Complete 20-Cell Greedy Rollout $\text{CI}$ & Effective Threshold $d_c^*(d_a)$

1. **Reusing verified cached cells + generating missing cells:**
   - Load existing `kind == "h3"` (`split == "identity"`) rows and their `9B` judge labels from `/out/h3-{model}/texts.jsonl` and `/out/h3-{model}/labels.jsonl` (verifying 1-to-1 alignment, $900$ rows across the 9 cached cells, and exact question/concept coverage).
   - Generate greedy rollouts (`max_new_tokens=120`, `do_sample=False`, `batch_size=32`) for the **11 missing cells** ($1,100$ rows) plus **1 reproducibility anchor cell** `(0.25, 0.25)` ($100$ rows; $1,200$ generated rows total per model).
   - Unload the subject model (when `model != "9b"`), load the frozen `Qwen3.5-9B` judge once, and score all $1,200$ newly generated replies on the three standard binary rubrics (`identity`, `concept`, `coherent`) from `judge.py`:
     - $\text{IB} = 1 - \text{identity}$,
     - $\text{CM} = \text{concept}$,
     - $\text{Coherent} = \text{coherent}$,
     - $\text{CI} = \mathbb{I}[\text{IB} = 1 \wedge \text{CM} = 1 \wedge \text{Coherent} = 1]$,
     - plus soft probability $\text{CI}_{\text{soft}} = (1 - p_{\text{identity}}) \cdot p_{\text{concept}} \cdot p_{\text{coherent}} \in [0, 1]$ from `labels.jsonl`.
2. **Threshold $d_c^*(d_a)$ and Relative Reduction $\Delta_{\text{rel}} d_c^*$:**
   - For each $d_a \in \{0, 0.125, 0.25, 0.5\}$ and target threshold $\tau \in \{0.50 \text{ (primary)}, 0.25 \text{ (secondary pre-saturation check)}\}$:
     - Let $C(d_a) = \{ d_c \in \{0, 0.125, 0.25, 0.375, 0.5\} : \overline{\text{Coh}}(d_c, d_a) \ge 0.80 \}$ be the coherent concept doses at axis dose $d_a$.
     - **Piecewise-linear interpolated threshold $d_{c,\tau}^{*,\text{interp}}(d_a)$ (primary):** Treating $\{ (d_c, \overline{\text{CI}}(d_c, d_a)) : d_c \in C(d_a) \}$ in ascending order of $d_c$, find the first adjacent pair $(d_{c,j}, d_{c,j+1})$ where $\overline{\text{CI}}(d_{c,j}, d_a) < \tau \le \overline{\text{CI}}(d_{c,j+1}, d_a)$ and linearly interpolate:
       $$d_{c,\tau}^{*,\text{interp}}(d_a) = d_{c,j} + \frac{\tau - \overline{\text{CI}}(d_{c,j}, d_a)}{\overline{\text{CI}}(d_{c,j+1}, d_a) - \overline{\text{CI}}(d_{c,j}, d_a)} \left(d_{c,j+1} - d_{c,j}\right).$$
       If $\overline{\text{CI}}(d_c, d_a) < \tau$ for all $d_c \in C(d_a)$, we record `reached=False` and report both:
       - (a) the **grid-ceiling censored bound** $d_{c,\tau}^{*,\text{cens}}(d_a) = 0.50$ (which yields a conservative lower bound on threshold reduction when $d_a = 0$ fails to reach $\tau$ by $d_c = 0.50$ while $d_a > 0$ reaches $\tau$ at $d_c < 0.325$), and
       - (b) the **linearly extrapolated baseline** $d_{c,\tau}^{*,\text{extrap}}(0)$ from the positive slope on $[0.25, 0.50]$ when $\overline{\text{CI}}(0.5, 0) > \overline{\text{CI}}(0.25, 0)$.
     - **Discrete grid threshold $d_{c,\tau}^{*,\text{grid}}(d_a)$ (secondary):** $\min \{ d_c \in C(d_a) : \overline{\text{CI}}(d_c, d_a) \ge \tau \}$ (or $> 0.50$ if unreached).
     - **Relative threshold reduction:**
       $$\Delta_{\text{rel}} d_{c,\tau}^* = \max_{d_a \in \{0.125, 0.25, 0.5\}} \frac{d_{c,\tau}^*(0) - d_{c,\tau}^*(d_a)}{d_{c,\tau}^*(0)},$$
       with 95% paired cluster bootstrap CIs ($B = 2,000$ resamples over the $100$ $(c, q)$ pairs, `seed=0`).
   - **Rollout interaction tests on $\text{CI}$ and $\text{CI}_{\text{soft}}$:**
     - Compute the pre-saturation ($d_c \in \{0, 0.125, 0.25\}$) and full-grid ($d_c \in \{0, 0.125, 0.25, 0.375, 0.5\}$) linear probability interaction $\beta_{ca}^{\text{CI}}$, soft-probability interaction $\beta_{ca}^{\text{soft}}$, and cell-level rollout $\text{DiD}_{\text{CI}}(d_c, d_a) = [\overline{\text{CI}}(d_c, d_a) - \overline{\text{CI}}(d_c, 0)] - [\overline{\text{CI}}(0, d_a) - \overline{\text{CI}}(0, 0)]$, plus the Firth-penalized / Haldane-smoothed ($+0.5$ pseudocount per cell) log-odds interaction so logistic separation at $d_c = 0$ is also resolved on the rollout table itself.

---

## 5. Part 2 (`H3-v2b`) Specification — `h3v2_kl_gap.py`

### 5.1 Norm-Matched Dense vs. Sparse-$k$ `W_down` Column Recipes on `Qwen3.5-4B`

Let $L \in \{16, 22\}$ be the injection layer on `Qwen3.5-4B`, with median Pile residual norm $m_L = \texttt{float(ax["median\_resid"][L])}$ ($m_{16} = 10.8701$, $m_{22} = 24.5391$).
For any concept $c \in \{\text{"aliens"}\} \cup \texttt{CONCEPTS\_TEST}$ and dose pair $(d_c, d_a)$:

1. **Dense residual direction ($\text{repr} = \text{"dense"}$):**
   - Let $\hat{a}_L = \texttt{ax["unit"][L].float()}$ and let $\hat{c}_{c,L}$ be the unit concept direction at layer $L$ orthogonalized against $\hat{a}_L$.
   - Define the dimensionless combined steering vector:
     $$v_{\text{dense}}(c, L; d_c, d_a) = d_c \hat{c}_{c,L} - d_a \hat{a}_L.$$
     Because $\hat{c}_{c,L} \perp \hat{a}_L$ and both are unit vectors, its Euclidean norm is $\|v_{\text{dense}}\|_2 = \sqrt{d_c^2 + d_a^2}$, and the unit target direction is $\hat{v}(c, L; d_c, d_a) = v_{\text{dense}} / \|v_{\text{dense}}\|_2$.
   - The dense residual injection at block $L$ is:
     $$\Delta h_L^{\text{dense}}(c; d_c, d_a) = m_L v_{\text{dense}}(c, L; d_c, d_a) = m_L \sqrt{d_c^2 + d_a^2} \, \hat{v}(c, L; d_c, d_a).$$

2. **Norm-matched sparse-$k$ `W_down` column projection ($\text{repr} \in \{\text{"sparse\_1"}, \text{"sparse\_8"}, \text{"sparse\_32"}\}$):**
   - Let $W^{(L)} = \texttt{blocks(model)[L].mlp.down\_proj.weight.detach().float()} \in \mathbb{R}^{d \times n}$ ($2560 \times 9216$ on `4B`), and let $\hat{W}^{(L)}_{:, j} = W^{(L)}_{:, j} / (\|W^{(L)}_{:, j}\|_2 + 10^{-8})$ be its unit-normalized columns for $j \in \{0, \dots, n-1\}$.
   - Compute cosine similarities $s_j = \langle \hat{W}^{(L)}_{:, j}, \hat{v}(c, L; d_c, d_a) \rangle$ for all $j$, and select the top-$k$ column indices $S_k = (j_1, \dots, j_k)$ maximizing $|s_j|$ (ties broken deterministically by smaller index $j$).
   - Form the submatrix $U_k = [\hat{W}^{(L)}_{:, j_1}, \dots, \hat{W}^{(L)}_{:, j_k}] \in \mathbb{R}^{d \times k}$, compute its orthonormal column basis $Q_k \in \mathbb{R}^{d \times k}$ via reduced QR decomposition in `torch.float64`, and compute the orthogonal projection of $\hat{v}$ onto $\operatorname{span}(U_k)$:
     $$p_k = Q_k (Q_k^\top \hat{v}), \qquad \hat{p}_k = \frac{p_k}{\|p_k\|_2 + 10^{-12}}.$$
     Record the subspace alignment cosine $\cos \theta_k = \|p_k\|_2 \in (0, 1]$ and the selected signed neuron IDs $\{(j_r, \operatorname{sign}(s_{j_r}))\}_{r=1}^k$.
   - Inject with the **exact same residual Euclidean norm** $m_L \sqrt{d_c^2 + d_a^2}$ as the dense direction:
     $$\Delta h_L^{\text{sparse-}k}(c; d_c, d_a) = m_L \sqrt{d_c^2 + d_a^2} \, \hat{p}_k.$$
   - *Key mathematical bridge:* Because single-neuron steering `inject_residual(model, L, d_unit * m_L * w_hat)` in `persona.py` (L232) also scales the unit `W_down` column $\hat{w}$ by $m_L$, our `sparse_1` recipe at $d_a = 0$ is on the **exact same residual-norm scale** as `H5`'s `L22 N2742+` (where dose $0.5$ injects $0.5 \cdot m_{22} \hat{W}^{(22)}_{:, 2742}$)! Moreover, for $c = \text{"aliens"}$ at $L = 22$, neuron `N2742+` is the #1 cosine-aligned column of `W_down` with $\hat{c}_{\text{aliens}, 22}$ ($\cos = +0.5519$), so `sparse_1` directly tests how much of the gap comes from projecting onto $1, 8, \text{or } 32$ `W_down` columns vs. the full $2560$-dimensional contrast vector.

### 5.2 Stage 2A: Dev-Split Capability-Matched Dose Calibration ($\text{CI} \approx 0.55$)

To compare `L16` and `L22` synthetic directions against `L22 N2742+` ($\text{CI}_{\text{test}} = 0.55$) without dose confounding (and without peeking at `IDENTITY_TEST` during calibration):
1. On `"aliens"`, evaluate a 6-point proportional dose ladder along the balanced ray $d_c = d_a = d$ for
   $$d \in \{0.15, 0.20, 0.25, 0.30, 0.35, 0.50\}$$
   (plus the pure-concept ray $(d_c = 0.35, d_a = 0)$ and $(d_c = 0.50, d_a = 0)$ at `L22` to isolate whether `L22` needs any Assistant Axis push at all) across the 10 dev identity questions `IDENTITY_DEV` for each `(layer, repr)` combination (`L16` and `L22` $\times$ `dense`, `sparse_1`, `sparse_8`, `sparse_32`).
2. Score these dev rollouts with the `9B` judge (and teacher-forced log-odds) and select for each `(layer, repr)` the **capability-matched dose** $(d_c^*, d_a^*) = (d^*, d^*)$ as the smallest $d \in \{0.15, 0.20, 0.25, 0.30, 0.35\}$ achieving $\text{CI}_{\text{dev}} \ge 0.50$ with $\text{Coherent}_{\text{dev}} \ge 0.80$ (breaking ties by minimum $|\text{CI}_{\text{dev}} - 0.55|$, and falling back to $d^* = 0.25$ on `L16` / $d^* = 0.35$ on `L22` if all dev points lie above or below).
   - *Prior check from `h3-4b/texts.jsonl`:* On `h3-4b`'s dev sweep at `L16`, `"aliens"` had $\text{CI}_{\text{dev}}(0.25, 0) = 0.20$, $\text{CI}_{\text{dev}}(0.25, 0.50) = 0.90$, and $\text{CI}_{\text{dev}}(0.50, 0) = 0.90$, confirming that $d^* \in [0.20, 0.30]$ brackets $\text{CI} \approx 0.55$ cleanly.

### 5.3 Stage 2B: Full $2 \times 2 \times 4$ Factorial Evaluation on `"aliens"` + Held-Out 5-Concept Generalization

We evaluate the following conditions on `Qwen3.5-4B` across all four benchmark suites:
- **3 Reference Baselines:**
  1. `unsteered`: no intervention ($\text{KL}_{\text{Pile}} \equiv 0$);
  2. `neuron_L22_N2742_d0.5`: single persona neuron `L22 N2742+` at dose $0.5$ ($\Delta h_{22} = 0.5 m_{22} \hat{W}^{(22)}_{:, 2742}$);
  3. `neuron_L22_N2742_d0.75`: single persona neuron `L22 N2742+` at dose $0.75$ (reference high-dose single neuron).
- **16 Factorial Recipes on `"aliens"` ($2 \text{ dose regimes} \times 2 \text{ layers} \times 4 \text{ representations}$):**
  - **Dose regime:**
    - `matched`: capability-matched dose $(d^*, d^*)$ selected on `IDENTITY_DEV`;
    - `high`: original `H3`/`H5` dose $(0.5, 0.5)$.
  - **Layer:** $L \in \{16, 22\}$.
  - **Representation:** `repr` $\in \{\text{"dense"}, \text{"sparse\_1"}, \text{"sparse\_8"}, \text{"sparse\_32"}\}$.
- **2 Axis-Ablation Diagnostic Recipes on `"aliens"` at `L22`:**
  - `L22_dense_dc0.5_da0` ($(0.5, 0)$, pure concept direction at `L22` without Assistant Axis push) and `L22_dense_dc0.25_da0.25` ($(0.25, 0.25)$, fixed intermediate dose) so that even independent of `IDENTITY_DEV` selection, fixed doses $(0.25, 0.25)$ and $(0.5, 0)$ are always present.
- **4 Held-Out 5-Concept Generalization Recipes (`CONCEPTS_TEST = ["cookies", "dragons", "the ocean", "knights", "trees"]`):**
  - For each of the 5 held-out test concepts, evaluate the 4 key combinations at the median matched dose $\bar{d}^*$ across `"aliens"`:
    1. `(L16, dense, matched)` and `(L16, dense, high=(0.5, 0.5))`;
    2. `(L22, dense, matched)` and `(L22, sparse_8, matched)`.
  - For these $5 \times 4 = 20$ concept-recipe pairs, measure `IDENTITY_TEST` $\text{CI}$ + coherence ($20 \text{ questions} \times 5 \text{ concepts}$) and exact `Pile KL` ($200$ passages) to verify that the KL reduction holds across all 5 held-out concepts.

### 5.4 Benchmark Suites & Statistical Estimands in Part 2

Every condition in the `"aliens"` factorial (and reference baselines) is evaluated on the exact `H5` benchmark protocol from `study/h5_benchmarks.py`:
1. **Identity Adoption (`IDENTITY_TEST`, $n = 20$ questions for `"aliens"`, $n = 100$ for 5 held-out concepts):**
   - Greedy rollout (`max_new_tokens=120`) scored by `Qwen3.5-9B` judge on `identity`, `concept`, `coherent` $\to \text{CI}, \text{IB}, \text{CM}, \text{Coherent}$, plus continuous teacher-forced log-odds $\Delta \ell$.
2. **Everyday Coherence (`EVERYDAY_TEST`, $n = 20$ general assistant questions):**
   - Greedy rollout (`max_new_tokens=120`) scored by `Qwen3.5-9B` judge on `coherent` (and `concept` bleed rate).
3. **300-Item Chat-MMLU (`n_mmlu = 300`, `seed = 0`, 4-way A/B/C/D log-prob choice via `prompts.load_mmlu(300)`):**
   - Exact single-token log-prob argmax over `("A", "B", "C", "D")` after `chat_prompt(tok, mmlu_prompt)`. Report accuracy and paired cluster bootstrap 95% CI of $\Delta \text{MMLU} = \text{MMLU}(\text{recipe}) - \text{MMLU}(\text{unsteered})$ ($B = 2,000, \text{seed} = 0$).
4. **Exact Next-Token Pile KL (`n_pile = 200` passages $\times 128$ tokens from `prompts.load_pile_texts(200)`):**
   - Exact full-vocabulary forward KL divergence in `float32`:
     $$\text{KL}(p_0 \,\|\, p_{\text{steer}}) = \sum_{v \in V} p_0(v) \left(\log p_0(v) - \log p_{\text{steer}}(v)\right) \quad \text{(nats/tok)}$$
     averaged over all valid next-token positions per passage, with 2,000-resample bootstrap 95% CI over the $200$ passages and paired bootstrap difference vs. `L22 N2742+` (dose $0.5$).
5. **Factorial Variance / Effect Decomposition of $\text{KL}_{\text{Pile}}$:**
   - Across the $2 \times 2 \times 4$ `"aliens"` factorial, fit an additive and interaction linear model on passage-level $\text{KL}_{\text{Pile}}$ quantifying the marginal KL reduction ($\Delta \text{KL}$ in nats/tok and % reduction) attributable to:
     - (i) **Capability matching vs. $(0.5, 0.5)$ over-steering**,
     - (ii) **Layer depth (`L22` vs. `L16`)**, and
     - (iii) **Sparsity (`sparse_1`, `sparse_8`, `sparse_32` vs. `dense`)**.

---

## 6. CPU `--smoke` Protocol, Hardware, Budget, and Verification

### 6.1 CPU `--smoke` Mode (`Qwen3.5-0.8B`, $0.00 via `the CPU check runner`)


In `--smoke` mode:

### 6.2 Modal Hardware, Timeouts, and Budget Allocation ($2.25 Cap)

> **CRITICAL `modal run` PRICING GUARD (`RESEARCH.md` L81):** `modal run` estimates its pre-launch budget hold from `timeout * dearest_gpu_rate_in_file` across the entire Python file (including comments and docstrings). Therefore, neither `h3v2_interaction.py` nor `h3v2_kl_gap.py` may contain the strings `L40S`, `A100`, or `H100` anywhere in code, comments, or docstrings, and `jobs.sized` is NOT used. Instead, both scripts decorate their Modal function directly with `gpu="A10G"` ($1.70/hr = $0.000472/s) and `timeout=1800` (pre-launch hold $\approx \$0.85$), keeping every launch well under the $2.25 cap.

| Job | Script & Flags | GPU | Timeout (Hold) | Expected Wall-Clock | Expected Cost |
|---|---|---|---|---|---|
| Smoke 1 | `python .../h3v2_interaction.py --smoke` | Local CPU (`heavy`) | — | ~2 min | **$0.00** |
| Smoke 2 | `python .../h3v2_kl_gap.py --smoke` | Local CPU (`heavy`) | — | ~2 min | **$0.00** |
| Run 1 | `h3v2_interaction.py --model 2b` (`2B` TF + 11 missing cells + `9B` judge) | `A10G` (24 GB) | 1800 s ($0.85) | ~9 min | **~$0.26** |
| Run 2 | `h3v2_interaction.py --model 4b` (`4B` TF + 11 missing cells + `9B` judge) | `A10G` (24 GB) | 1800 s ($0.85) | ~11 min | **~$0.31** |
| Run 3 | `h3v2_interaction.py --model 9b` (`9B` TF + 11 missing cells + `9B` judge) | `A10G` (24 GB) | 2100 s ($0.99) | ~15 min | **~$0.43** |
| Run 4 | `h3v2_kl_gap.py` (`4B` dev calibration + $2\times 2\times 4$ factorial + 5-concept check + `9B` judge) | `A10G` (24 GB) | 2100 s ($0.99) | ~16 min | **~$0.45** |
| **Total** | **All 4 paid Modal runs (`personakit-h3-interaction-kl`)** | **`A10G`** | — | **~51 min** | **~$1.45** (cap **$2.25**) |

### 6.3 Checkpointing & Idempotency

Both jobs save intermediate checkpoints to `the output volume` (`/out/h3v2a-{model}/` and `/out/h3v2b-4b/`) after each stage (`tf_logodds.json`, `generated_texts.jsonl`, `judge_labels.jsonl`, `dev_calibration.json`, `benchmarks.json`) and call `jobs.OUT.commit()` after each stage, so any interrupted run resumes from the last completed stage without re-spending GPU time.
