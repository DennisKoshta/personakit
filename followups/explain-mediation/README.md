# Study Report & Preregistered Design: `personakit explain` Predictor and Base-to-Instruct Persona Mediation (`Qwen3.5-4B` / `9B`)

**Status:** Completed (`2026-10-28`; preregistered design locked in [`README.md`](README.md) at `ebce16d`, `READINESS.md` at `447b338`, and `IMPLEMENTATION_NOTES.md` at `89a43d6` [precedence] before any model execution or paid compute; full execution validation log in `VALIDATION.md`)
**Date:** 2026-10-28
**Confirmed Budget Cap & Final Spend:** **$2.10 settled / $3.00 total study cap** (`$1.13` on 4B `H4-v2`, `$0.42` on 9B `H4-v2`, `$0.55` on 4B `H2-v2`; `$0.90` unspent under the `$3.00` cap).

---

## Executive Summary & Preregistered Verdict Table

| Hypothesis / Stage | Target Model(s) | Primary Preregistered Criterion | Empirical Outcome | Preregistered Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **`H4-v2` (`4B` Predictor Screen)** | `Qwen/Qwen3.5-4B` ($n = 160$ across 6 strata, $6{,}580$ rollouts, $19{,}740$ `9B` judgments) | Sample $\text{AUROC}(S_{\text{full}} \text{ or } W_{\text{comp}}) \ge 0.70$ for binary coherent identity break ($\text{IB} \ge 0.40 \land \text{Coh} \ge 0.80$), plus paired baseline superiority OR $\ge 5$ novel coherent personas | **`0/160` binary positives** in sample (binary AUROC undefined). Continuous coherent $\text{IB}$ Spearman rank correlation is significantly positive for both **`W_comp`** ($\rho = 0.1989$, 95% CI $[0.0649, 0.3241]$) and **`S_full`** ($\rho = 0.1872$, 95% CI $[0.0544, 0.3119]$), whereas `peakiness` ($\rho = -0.0005$) and `cos_away` ($\rho = 0.0928$) span zero and raw `over_natural` is negative ($\rho = -0.1675$). **5 non-`"a robot"` coherent persona neurons** ($\text{CI} \ge 0.50, \text{Coh} \ge 0.80$) discovered—all 5 in `top_s_full` (`3/40`) and `top_w_comp` (`2/30`), vs `0/95` in baseline/random strata (6 additional `"a robot"` hits are a judge false-positive artifact; see §8.3). | **Inconclusive** (binary AUROC undefined; continuous rank & persona enrichment positive) |
| **`H4-v2` (`9B` Cross-Size Screen)** | `Qwen/Qwen3.5-9B` ($n = 30$ across 6 strata, $1{,}260$ rollouts, $3{,}780$ `9B` judgments) | Same binary $\text{AUROC} \ge 0.70$ gate on $n = 30$ cross-size screen | **`1/30` binary positive** (`L19N632+` in `top_cos_away`, `a sage`, $\text{IB} = 0.40, \text{Coh} = 0.90, \text{CI} = 0.05$ at $d = 0.75$). Sample $\text{AUROC}(S_{\text{full}}) = 0.4828$ ($[0.4444, 0.4828]$) and $\text{AUROC}(W_{\text{comp}}) = 0.4483$ ($[0.3571, 0.4828]$) vs $\text{AUROC}(\text{cos\_away}) = 0.8621$ ($[0.8621, 0.9259]$). `top_s_full` discovered a genuine coherent dragon persona neuron **`L26N5705+`** ($\text{CI} = 0.65, \text{CM} = 0.95, \text{Coh} = 0.85, \text{IB} = 0.10$ at $d = 0.75$). | **No** (sample binary AUROC $< 0.70$, 95% CI upper bound $0.4828 < 0.70$) |
| **`H2-v2` (`4B` Base-to-Instruct Mediation)** | `Qwen/Qwen3.5-4B` + `4B-Base` donor (`L22N2742+` anchor + 3 exploratory targets, $800$ rows, $2{,}400$ `9B` judgments) | At least one late-layer intervention (`swap_attn` `L23–31`, `swap_mlp` `L23–31`, or `self_ref_cap` $\hat{s}_L$ `L23–28`) gives $\text{CI\_ratio} \le 0.50$ ($p < 0.05$), $\text{CM\_ratio} \ge 0.70$, control $\ge 0.80$; falsifier $\text{CI\_ratio} \ge 0.80$ | On confirmatory anchor `L22N2742+` ($\text{CI}_{\text{steer}} = 0.50, \text{CM}_{\text{steer}} = 0.65, \text{Coh}_{\text{steer}} = 0.95$):<br>• **`swap_attn` (`L23–31`):** $\text{CI\_ratio} = 1.00$ ($[0.7273, 1.3333]$, $p = 0.75$, `CM_ratio` $= 1.0769$) $\to$ `inconclusive`<br>• **`swap_mlp` (`L23–31`):** $\text{CI\_ratio} = 1.20$ ($[0.8462, 1.8571]$, $p = 0.9375$, `CM_ratio` $= 1.0769$) $\to$ **`no`** (falsified)<br>• **`self_ref_cap` (`L23–28`):** $\text{CI\_ratio} = 1.20$ ($[0.8333, 1.8571]$, $p = 0.9375$, `CM_ratio` $= 1.1538$) $\to$ **`no`** (falsified)<br>• Controls `sham_swap_attn`, `axis_cap`, `random_cap`: all $\text{CI\_ratio} = 1.00$ ($[1.00, 1.00]$). | **Inconclusive overall / `No` on `swap_mlp` & `self_ref_cap`** (neither late swap alone nor $\hat{s}_L$ cap suppresses first-person persona adoption) |
| **Study-Wide Action on `personakit`** | [`personakit/core.py`](../../personakit) | Update `personakit explain` iff `H4-v2` passes confirmatory $\text{AUROC} \ge 0.70$ gate | `H4-v2` did not meet the confirmatory binary $\text{AUROC} \ge 0.70$ gate (`inconclusive` on 4B, `no` on 9B). | **Leave `personakit` unchanged** (negative/inconclusive confirmatory result documented) |

### Key Scientific Takeaways
1. **Why binary `Coh-IB` ($\text{IB} \ge 0.40 \land \text{Coh} \ge 0.80$) is almost empty across single MLP neurons while `Coh-CI` ($\text{CI} \ge 0.50 \land \text{Coh} \ge 0.80$) is enriched in `S_full` and `W_comp`:**
   - Under `Qwen3.5-4B` Instruct's chat template, single-neuron pushes at $d \in \{0.50, 0.75\}$ frequently cause the model to *adopt a role/concept persona alongside its assistant framing* (e.g., *"I am Qwen, an AI assistant with the heart of a parent/producer/collector/dragon"*, scoring $\text{CI} = 1, \text{Coh} = 1$ under the `CONCEPT` rubric while still triggering `IDENTITY = 1` / $\text{IB} = 0$ because assistant identity is not denied), OR, when pushed hard enough to suppress `"AI assistant"` ($\text{IB} \ge 0.40$ at $d = 0.75$), push the residual stream across the coherence cliff ($\text{Coh} < 0.80$, e.g., `L23N8376-` `a parent` with $\text{CI} = 0.75, \text{Coh} = 0.95, \text{IB} = 0.00$ at $d = 0.50 \to \text{CI} = 1.00, \text{IB} = 0.75, \text{Coh} = 0.30$ at $d = 0.75$).
   - Even the known 4B alien neuron `L22N2742+` had $\text{IB} = 0.20, \text{CI} = 0.50, \text{Coh} = 0.95$ at $d = 0.50$ and only crossed $\text{IB} = 0.60, \text{Coh} = 0.90$ at $d = 0.75$.
   - On continuous coherent $\text{IB}$ ($y_{\text{coh-IB}}$) across $n = 160$ `4B` neurons, both locked composite predictors (`W_comp` $\rho = 0.1989$, 95% CI $[0.0649, 0.3241]$; `S_full` $\rho = 0.1872$, 95% CI $[0.0544, 0.3119]$) and the bounded reachability gate `F_reach` ($\rho = 0.1651$, $[0.0209, 0.2967]$) significantly outperform `peakiness` ($\rho = -0.0005$), `cos_away` ($\rho = 0.0928$), and raw `over_natural` ($\rho = -0.1675$). Furthermore, all **5 genuine (non-`"a robot"`) coherent persona discoveries on `4B`** (`L23N8376-` `a parent`, `L22N6457-` `a producer`, `L26N8619+` `a collector`, `L26N3421-` `an emissary`, `L25N8264+` `an emissary`) and the **coherent dragon neuron on `9B`** (`L26N5705+`, $\text{CI} = 0.65, \text{CM} = 0.95, \text{Coh} = 0.85$) were found in `top_s_full` and `top_w_comp`.
2. **Why neither `L23–31` Base attention swaps, `L23–31` Base MLP swaps, nor `L23–28` self-reference capping suppresses `L22N2742+` first-person persona adoption (`H2-v2`):**
   - Replacing all `L23–31` Instruct attention/DeltaNet weights with `Qwen3.5-4B-Base` weights (`swap_attn`, `67/72` tensors changed; the 5 unchanged tensors are identically initialized `A_log` buffers on DeltaNet layers) leaves $\text{CI} = 0.50$ ($\text{CI\_ratio} = 1.00$, $[0.7273, 1.3333]$).
   - Replacing all `L23–31` Instruct MLP weights with `Qwen3.5-4B-Base` weights (`swap_mlp`, `27/27` tensors changed) yields $\text{CI} = 0.60$ ($\text{CI\_ratio} = 1.20$, $[0.8462, 1.8571]$, falsified: **`no`**).
   - Upper-capping the concept-and-axis-orthogonal self-reference direction $\hat{s}_L$ across `L23–28` at its 75th-percentile default+3P ceiling (`self_ref_cap`, where $\hat{s}_L$ retains $72.8\%–74.3\%$ of the raw 1P$-$3P contrast norm and actively clamps all 320 prefill/decode forward calls) yields $\text{CI} = 0.60$ ($\text{CI\_ratio} = 1.20$, $[0.8333, 1.8571]$, falsified: **`no`**).
   - **Mechanistic implication:** The Base-to-Instruct difference that turns an `L22 N2742+` alien concept push into a first-person persona is **not** localized to a single downstream module family (`L23–31` attention alone or `L23–31` MLP alone) or to a single linear 1P$-$3P direction in `L23–28`. By layer `L22`, `Qwen3.5-4B` Instruct's `L0–22` processing of the `<|im_start|>user ... <|im_start|>assistant` chat prompt has already established a first-person conversational responder state in the residual stream (or `L23–31` attention and MLP blocks redundantly preserve that state unless swapped jointly or upstream). When $w_{\text{down}}(22, 2742)$ injects the `"aliens"` concept into that already-bound first-person responder state at `L22`, `L23–31` Base attention weights or Base MLP weights decode it in the first person at the full steered rate ($\text{CI} = 0.50–0.60$).

---

## 1. Research Questions, Prior Empirical Baseline & Preregistered Endpoints

### 1.1 Exact Baseline Findings from the 2026-10-26 Study ([`study/README.md`](../../study/README.md))

Every baseline number below is traced directly to [`study/README.md`](../../study/README.md) and `READINESS.md`:

1. **`H1` & `H2` (Concept Alignment Without Assistant-Axis Mediation):**
   - On `Qwen3.5-4B`, the alien neuron (`L22 N2742+`) aligns strongly with the `"aliens"` concept direction ($\cos = +0.55$, 99.99th percentile among the $9{,}216$ neurons in `L22`) and not with the away-from-Assistant Axis $-\hat{a}_{22}$ ($\cos = -0.005$, 37th percentile).
   - Pushing `L22 N2742+` at dose $0.5$ (its smallest swept dose reaching $\text{CI} \ge 0.50$ at coherence $\ge 0.80$ on dev prompts) produced $\text{CI} = 0.50$ $[0.30, 0.70]$ on the 20 held-out test identity prompts (`IDENTITY_TEST`).
   - In `H2b`, capping projections onto the Assistant Axis $\hat{a}_L$ at layers `L23–28` (at the default Assistant replies' 25th percentile `floors_p25`) **kept $\text{CI}$ at $1.1\times$ $[1.0, 1.43]$ of its steered rate** (`H2`'s falsifier $\ge 0.80\times$ fired). Capping a matched random direction kept $1.0\times$; removing the axis component from the write vector (`no_axis`) kept $1.0\times$; steering the axis component alone (`axis_only`) gave $\text{CI} = 0$.
2. **`H4` (Why `personakit explain`'s Single-Feature Predictor Failed):**
   - `h4_screen.py` screened `406` neurons on `Qwen3.5-4B` (`6` known coordinates from `prompts.KNOWN` plus `400` random neurons, `100` each from `L10, L14, L18, L22` with random signs drawn at `seed = 0`) at dose $0.75$ across `10` development identity prompts (`IDENTITY_DEV`, greedy, `64` tokens; `4,370` total replies in `12.7 min`, `$0.3657` on Modal ).
   - Among the 400 random neurons, only `3` broke identity ($\text{IB} \ge 0.40$) and only `1` of those stayed coherent.
   - Across the screen (`IB >= 0.40`):
     - `cos_away` ($\cos(\text{sign}\cdot w, -\hat{a}_L)$): $\text{AUROC} = 0.47$ $[0.01, 1.00]$, Spearman $\rho = -0.01$ $[-0.11, 0.09]$.
     - `peakiness` (max vocabulary z-score of final-normed unembedded $\text{sign}\cdot \hat{w}$): $\text{AUROC} = 0.77$ $[0.40, 0.97]$, Spearman $\rho = 0.16$ $[0.05, 0.26]$.
     - `over_natural` (push $\delta$ divided by the neuron's largest natural activation of that sign in Pile or chat): $\text{AUROC} = 0.11$ $[0.01, 0.20]$, Spearman $\rho = -0.19$ $[-0.25, -0.11]$ (pushing dormant neurons far beyond their natural chat maximum causes incoherent gibberish rather than a coherent persona).
3. **`X1` (Base vs. Instruct Transformation) & `X2` (Cross-Size Twin):**
   - Between `Qwen3.5-4B-Base` and `Qwen3.5-4B` Instruct, every `W_down` column keeps cosine $\ge 0.989$ (median $\ge 0.996$ in every layer; `L22 N2742` $= 0.997$), and Base's Assistant Axis has cosine $0.97$ with Instruct's at `L16` ($|\cos(\text{axis}, \text{PC1})| = 0.74$).
   - Yet pushing `L22 N2742+` at dose $0.5$ (in each model's own Pile median residual norm unit $m_L$) on the 20 test identity prompts yields **$\text{CI} = 0.10$ $[0.03, 0.30]$, $\text{CM} = 0.20$ $[0.08, 0.42]$ in `Base`** vs. **$\text{CI} = 0.55$ $[0.34, 0.74]$, $\text{CM} = 0.70$ $[0.48, 0.85]$ in `Instruct`** (both $0.00$ unsteered; coherence $0.95–1.00$).
   - In `X2`, `Qwen3.5-9B` has a working alien twin at `L22 N10573+` (vocabulary Spearman $0.30$ with `4B N2742+`, yielding $\text{CI} = 0.65$, $\text{CM} = 0.90$, coherence $0.95$ at dose $0.5$).

### 1.2 Preregistered Endpoints & Verdicts (as preregistered)

- **Endpoint 1 (`H4-v2`, `personakit explain` predictor):**
  - Evaluated on the deduplicated stratified sample of $n = 160$ neurons in `Qwen3.5-4B` (and $n = 30$ in `Qwen3.5-9B`) across the 20 held-out `IDENTITY_TEST` prompts at doses $\{0.50, 0.75\}$ (`64` greedy tokens).
  - **Binary Coherent Identity Break (`Coh-IB`):** A neuron $u = (L, n, s)$ is positive ($Y_{\text{coh-IB}}(u) = 1$) iff at some tested dose $d \in \{0.50, 0.75\}$, its mean identity-break rate on `IDENTITY_TEST` satisfies **$\text{IB}(u, d) \ge 0.40$ AND mean coherence satisfies $\text{Coh}(u, d) \ge 0.80$** (the same dose must satisfy both thresholds).
  - **`H4-v2` Passes ("Yes") iff:**
    1. The locked composite predictor ($S_{\text{full}}$ or $W_{\text{comp}}$) achieves **$\text{AUROC} \ge 0.70$** for $Y_{\text{coh-IB}}$ on `4B` ($n = 160$), **AND**
    2. Either **(a)** its paired AUROC differences against all three single-feature baselines—`cos_away`, `peakiness`, and `over_natural` (tested against both raw `over_natural` and sign-oriented `-over_natural` / $F_{\text{reach}}$)—have 95% stratified bootstrap confidence intervals strictly above $0$, **OR (b)** the screen discovers **$\ge 5$ new coherent persona neurons** (outside `prompts.KNOWN`) with **$\text{CI}(u, d) \ge 0.50$ at $\text{Coh}(u, d) \ge 0.80$**.
  - **`H4-v2` Fails ("No") iff:** Neither $S_{\text{full}}$ nor $W_{\text{comp}}$ reaches $\text{AUROC} \ge 0.70$, and fewer than $5$ new coherent persona neurons ($\text{CI} \ge 0.50, \text{Coh} \ge 0.80$) are discovered.
- **Endpoint 2 (`H2-v2`, Base-to-Instruct persona mediation):**
  - Evaluated on `L22 N2742+` (confirmatory anchor) and the top 3 newly discovered coherent persona neurons from Stage 2 (if any qualify with $L \le 22$ and $\text{CI} \ge 0.50$ at $\text{Coh} \ge 0.80$; see §5.1 for confirmatory vs. discovery-conditioned splits).
  - **Following [`study/README.md`](../../study/README.md) (line 109) and [`h2_mediation.py`](../../study/h2_mediation.py) (lines 20–26, 219–226),** all mediation criteria are defined as **relative ratios to the steered baseline rate** on calibrated targets ($\text{CI}(\text{steer}) \ge 0.50$ at $\text{Coh}(\text{steer}) \ge 0.80$), with explicit zero-denominator guards (`None` if `sum(steer) == 0` or `sum(cm_steer) == 0`):
    $$\text{CI\_ratio}(\text{cond}) = \begin{cases} \dfrac{\text{CI}(\text{cond})}{\text{CI}(\text{steer})} & \text{if } \text{CI}(\text{steer}) > 0, \\ \text{None} & \text{otherwise,} \end{cases} \qquad \text{CM\_ratio}(\text{cond}) = \begin{cases} \dfrac{\text{CM}(\text{cond})}{\text{CM}(\text{steer})} & \text{if } \text{CM}(\text{steer}) > 0, \\ \text{None} & \text{otherwise.} \end{cases}$$
    Confirmatory `CM_ratio >= 0.70` gating applies to targets with a matched `prompts.FAMILIES` concept family (including the confirmatory anchor `L22 N2742+` `"aliens"`), while role-assigned targets in `top3_new` report lexical `CM_ratio` as an explicitly labeled exploratory proxy per `IMPLEMENTATION_NOTES.md`.
  - **`H2-v2` Passes ("Yes") iff** for at least one late-layer mechanism—either `L23–31` `Base`-into-`Instruct` parameter swapping (`swap_attn` or `swap_mlp`) or `L23–28` upper-capping on the concept-and-axis-orthogonal first-person self-reference direction $\hat{s}_L$ (`self_ref_cap`):
    1. **$\ge 50\%$ relative reduction in first-person persona claims:** $\text{CI\_ratio}(\text{cond}) \le 0.50$ (with one-sided paired McNemar / sign-flip $p < 0.05$), **AND**
    2. **$\ge 70\%$ relative retention of factual concept mention:** $\text{CM\_ratio}(\text{cond}) \ge 0.70$ (where `CM` is the lexical word-family rate `prompts.mentions`), **AND**
    3. **$\ge 80\%$ relative retention under matched random control:** $\text{CI\_ratio}(\text{random\_cap}) \ge 0.80$ (for capping) or $\text{CI\_ratio}(\text{sham\_swap\_attn}) \ge 0.80$ (for swapping), with mean coherence $\text{Coh}(\text{cond}) \ge 0.80$.
  - **`H2-v2` Falsifier ("No"):** Both `L23–31` `Base` swapping (`swap_attn`, `swap_mlp`) and `L23–28` $\hat{s}_L$ capping leave $\text{CI\_ratio}(\text{cond}) \ge 0.80$ (matching `H2`'s falsifier).
- **Study-Wide Verdict & Action on `personakit`:**
  - **Yes:** Both `H4-v2` and `H2-v2` pass. Only after Stage 2 passes and code review completes do we update `personakit/personakit/core.py::explain` to report the validated composite ranker (no new CLI flags or unapproved surface changes).
  - **No:** Both fail; `personakit` remains unchanged and the negative result is documented.
  - **Inconclusive:** One half passes while the other fails, or a primary statistic's 95% bootstrap confidence interval straddles its threshold (or fewer than half of bootstrap resamples have defined AUROC when positives are rare, per `h4_screen.boot_ci`).

---

## 2. Verified Literature & Citation Audit

Every external paper cited in the study plan was verified via `paper_read`:

1. **Lu, C. et al. (2026), *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*, [arXiv:2601.10387](https://arxiv.org/abs/2601.10387):**
   - Defines the default-minus-role Assistant Axis $a_L$, shows in Appendix B.3 that Base models share the Assistant Axis PC1 geometry, and defines floor capping $h \leftarrow h + \max(0, \tau - \langle h, \hat{a}_L \rangle)\hat{a}_L$ across layers $L+1 \dots L+6$.
2. **Herring, S., Naviasky, E., & Malhotra, A. (2026), *Targeted Neuron Modulation via Contrastive Pair Search*, [arXiv:2605.12290](https://arxiv.org/abs/2605.12290):**
   - *Scope note:* Introduces Contrastive Neuron Attribution (CNA) on safety refusal gates (harmful vs. benign prompts on JBB-Behaviors). Section 5.3 (*"Fine-Tuning Transforms Function"*) shows that late-layer neurons already present in Base models act as passive topic discriminators that cause content shifts when steered in Base, whereas instruction tuning transforms those same directions into causal behavioral gates. We test the analogous transformation from concept mention (`CM`) in `4B-Base` to first-person persona enactment (`CI`) in `4B-Instruct`.
3. **Gurnee, W. et al. (2024), *Universal Neurons in GPT2 Language Models*, [arXiv:2401.12181](https://arxiv.org/abs/2401.12181):**
   - *Scope note:* Characterizes monosemantic MLP write vectors via unembedding projection distribution moments (skewness/kurtosis) and activation quantiles; our specific `peakiness` formula (`(z.max() - z.mean()) / z.std()` after final RMSNorm and `lm_head`) and natural range ratio come directly from [`study/h4_screen.py`](../../study/h4_screen.py).
4. **Liu, H., Li, Y., & Wu, Z. (2026), *Perturbation Probing: Single-Pass Activation Steering Efficiency Checks*, [arXiv:2604.27401](https://arxiv.org/abs/2604.27401):**
   - Motivates single-forward-pass teacher-forced log-probability contrasts under activation steering (`Delta_TF`) as a fast proxy for autoregressive rollout behavior.

---

## 3. Pinned Models, Datasets & Existing Modal Volume Artifacts

All pins match [`study/jobs.py`](../../study/jobs.py) and `READINESS.md`:

| Key | Repository | Revision SHA | Role & Architecture |
| :--- | :--- | :--- | :--- |
| `4b` | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Primary Instruct target (`32` layers, $d = 2560$, $N_{\text{mlp}} = 9216$, $|V| = 248320$) |
| `4b-base` | `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` | Pre-trained Base donor/target (`32` layers, $d = 2560$, $N_{\text{mlp}} = 9216$) |
| `9b` | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Cross-size target (`32` layers, $d = 4096$, $N_{\text{mlp}} = 12288$) & unsteered Judge |
| `0.8b` | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | Local CPU `--smoke` gate only (`24` layers `L0–23`, $d = 1024$, $N_{\text{mlp}} = 3584$) |
| `pile-10k` | `NeelNanda/pile-10k` | `127bfedcd5047750df5ccf3a12979a47bfa0bafa` | Pile calibration dataset |
| `assistant-axis` | `safety-research/assistant-axis` | `a98961956072224eaf244eb289d6c01700b63795` | Role instructions & extraction questions |
| `mmlu` | `cais/mmlu` | `c30699e8356da336a370243923dbaf21066bb9fe` | Capability helper pin |

### Modal Volume & Artifact Provenance (`jobs.OUT` = `the output volume`)
- Verified input artifacts on `the output volume` (with SHA-256 digests from `READINESS.md`):
  - `out/s1a-4b/calib.pt` (`ddcd2337a49e019a4931b4888e8c180def08d89f14043f94dd770b1771223618`): `pile` and `chat` dicts with `hi [32, 9216]`, `lo [32, 9216]`, `norm_q10_50_90 [32, 3]`, `tokens`.
  - `out/s1b-4b/axis.pt` (`dc039446a2092e7aeb355439d80183264900cbd5749f5089a11512629926869e`): `axis [32, 2560]`, `unit [32, 2560]`, `default [32, 2560]`, `roles` (`79` accepted role names), `role_vecs [79, 32, 2560]` (`float16`), `concept_dirs` (`10` plural concept keys, each `[32, 2560]` `float32`, **unnormalized**), `floors_p25 [32]`, `median_resid [32]`.
  - `out/s1a-9b/calib.pt` (`d64505d1d211e4a9f9d549236fdfb5bdf32e50128041d89807d03223255f273f`): `32` layers, $N_{\text{mlp}} = 12288$.
  - `out/s1b-9b/axis.pt` (`c016d3d056e9a0db8e1427c4d3ff3de501601753f948377b48b15158c5dd3328`): `80` accepted roles (`roles`), `10` concepts (`concept_dirs`, unnormalized).
  - `out/s1a-4b/texts.jsonl` & `labels.jsonl`: `6,400` role + `400` default + `100` concept + `30` identity rows.

---

## 4. Stage 1 & Stage 2 (`H4-v2`): Full-Band Weight/Atlas Screen, Teacher-Forced Scoring & Stratified Verification

### 4.1 Signed Candidate Population & Exclusions (`L8–L26`)

> **Correction of Proposal Candidate-Count Arithmetic (`READINESS.md`):** The plan writes *"$32 \times 9{,}216 = 294{,}912$ signed neuron directions in `4B` (layers 8–26)"*. In fact, $32 \times 9{,}216 = 294{,}912$ is the **unsigned** full-model neuron count (`589,824` signed directions across `L0–31`). Restricted to the 19-layer band `L8–L26` (`L in range(8, 27)`), there are $19 \times 9{,}216 = 175{,}104$ unsigned neurons and **$19 \times 9{,}216 \times 2 = 350{,}208$ signed neuron directions** on `4B` (and $19 \times 12{,}288 \times 2 = 466{,}944$ signed directions on `9B`).

- Each candidate unit is a signed tuple $u = (L, n, s)$ with $L \in \{8, \dots, 26\}$, $n \in \{0, \dots, N_{\text{mlp}} - 1\}$, and sign $s \in \{+1, -1\}$.
- **Exclusions (`prompts.KNOWN`):** To prevent known coordinates from inflating held-out discovery or AUROC metrics, we exclude the exact coordinates in [`prompts.KNOWN`](../../study/prompts.py) on `4B` (`6` signed units):
  - `"alien": (22, 2742, +1)`
  - `"hostile": (14, 8949, +1)`
  - `"sycophancy_a": (12, 2920, -1)`
  - `"sycophancy_b": (16, 8168, -1)`
  - `"condolences": (10, 6295, +1)`
  - `"evil_pair_b": (13, 7956, -1)`
  This leaves **$|\mathcal{U}_{\text{pop}, 4\text{B}}| = 350{,}208 - 6 = 350{,}202$** candidate signed units on `4B`.
- On `9B`, we exclude the known alien twin `("alien_twin_9b", 22, 10573, +1)` from [`x2_twins.py`](../../study/README.md), leaving **$|\mathcal{U}_{\text{pop}, 9\text{B}}| = 466{,}944 - 1 = 466{,}943$** candidate signed units.
- The excluded known units (`6` on `4B`, `1` on `9B`) are still evaluated in a separate diagnostic recall table, never pooled into $\mathcal{U}_{\text{pop}}$ AUROC or novel-persona counts.

### 4.2 Exact Per-Unit Feature Definitions & Concept/Role Normalization

For each layer $L \in \{8, \dots, 26\}$, let $W_{\text{down}, L} = \text{persona.blocks(model)}[L].\text{mlp.down\_proj.weight.float()} \in \mathbb{R}^{d \times N_{\text{mlp}}}$, column norms $\nu_{L, n} = \|W_{\text{down}, L}[:, n]\|_2$, and signed unit write direction $\hat{w}(u) = s \cdot W_{\text{down}, L}[:, n] / \nu_{L, n}$. Let $m_L = \text{ float}(\text{ax}[\text{"median\_resid"}][L])$ be the Pile median residual norm at layer $L$, so a dose-1 push adds $\delta_1(L, n) = m_L / \nu_{L, n}$ at `down_proj` (equivalently $m_L \hat{w}(u)$ at layer $L$'s output).

1. **Baseline 1 — Assistant-Axis Cosine (`cos_away`):**
   $$\text{cos\_away}(u) = -\langle \hat{w}(u),\, \text{unit}[L] \rangle.$$
2. **Baseline 2 — Final-Normed Logit-Lens Peakiness (`peakiness`, matching [`h4_screen.py` line 147](../../study/h4_screen.py)):**
   - Pass $\hat{w}(u)$ through `model.model.norm` and `model.lm_head` in chunks of $\le 1{,}024$ directions (`bfloat16` matmul, reduced immediately to `float32` scalars without storing full `[N, vocab]` tensors):
     $$z(u) = \text{model.lm\_head}(\text{model.model.norm}(\hat{w}(u))) \in \mathbb{R}^{|V|}, \qquad \text{peakiness}(u) = \frac{\max_v z_v(u) - \operatorname{mean}(z(u))}{\operatorname{std}(z(u))}.$$
   - Because $\text{model.model.norm}(s \cdot \hat{w}) = s \cdot \text{model.model.norm}(\hat{w})$, one unsigned pass per neuron $(L, n)$ yields both signs: $\text{peakiness}(L, n, +1)$ uses $\max_v z_v$ and $\text{peakiness}(L, n, -1)$ uses $\max_v (-z_v) = -\min_v z_v$.
3. **Baseline 3 — Push-Over-Natural-Max (`over_natural`, matching [`h4_screen.py` lines 150–153](../../study/h4_screen.py)) & Sign-Aware Chat Reachability ($F_{\text{reach}}$, Proposal line 28):**
   - For sign $s \in \{+1, -1\}$, let key $k(s) = \text{"hi"}$ if $s = +1$ else $\text{"lo"}$.
   - Prior `H4` baseline predictor at screen dose $0.75$:
     $$\text{natural}(u) = \max_{\text{src} \in \{\text{"pile"}, \text{"chat"}\}} \left|\text{float}(\text{calib}[\text{src}][k(s)][L, n])\right|, \qquad \text{over\_natural}(u) = \frac{0.75 \cdot \delta_1(L, n)}{\max(\text{natural}(u), 10^{-6})}.$$
   - Sign-aware natural chat extreme (clamped to $\ge 0$ so a negative `hi` or positive `lo` yields $0$ reachability) and bounded reachability gate:
     $$\text{chat\_extreme}(u) = \begin{cases} \max(0.0,\, \text{float}(\text{calib}[\text{"chat"}][\text{"hi"}][L, n])) & \text{if } s = +1, \\ \max(0.0,\, -\text{float}(\text{calib}[\text{"chat"}][\text{"lo"}][L, n])) & \text{if } s = -1, \end{cases}$$
     $$F_{\text{reach}}(u) = \min\!\left(1.0,\; \frac{\text{chat\_extreme}(u)}{0.5 \cdot \delta_1(L, n)}\right).$$
4. **Concept & Role Alignment ($F_{\text{align}}$) and Explicit Key/Label Mapping $T(u)$:**
   - **Unit-normalizing `axis.pt["concept_dirs"]`:** In `s1b_axis.py` (lines 108–115), `axis.pt["concept_dirs"]` has the **10 plural keys** from `prompts.CONCEPTS_DEV + prompts.CONCEPTS_TEST`:
     $$\mathcal{C}_{10} = [\text{"aliens"}, \text{"ghosts"}, \text{"pirates"}, \text{"owls"}, \text{"robots"}, \text{"cookies"}, \text{"dragons"}, \text{"the ocean"}, \text{"knights"}, \text{"trees"}],$$
     and stores unnormalized vectors with the `unit[L]` component removed. We explicitly normalize each concept vector per layer via `mechanics.orthogonalize`:
     $$\hat{c}_{c, L} = \texttt{mechanics.orthogonalize}(\text{ax}[\text{"concept\_dirs"}][c][L],\; [\text{ax}[\text{"unit"}][L]]).$$
   - **Unit-normalizing axis-orthogonal role directions:** For each kept role $r_j \in \text{ax}[\text{"roles"}]$ ($79$ on `4B`, $80$ on `9B`), let $d_{r_j, L} = \text{ax}[\text{"role\_vecs"}][j, L].\text{float}() - \text{ax}[\text{"default"}][L]$ and remove its Assistant-Axis component via `mechanics.orthogonalize`:
     $$\hat{d}_{r_j, L} = \texttt{mechanics.orthogonalize}(d_{r_j, L},\; [\text{ax}[\text{"unit"}][L]]).$$
   - **Alignment score and nearest target assignment:**
     $$F_{\text{align}}(u) = \max\Big(\max_{c \in \mathcal{C}_{10}} \langle \hat{w}(u), \hat{c}_{c, L} \rangle,\; \max_{r \in \text{roles}} \langle \hat{w}(u), \hat{d}_{r, L} \rangle\Big).$$
   - **Display-Label Mapping $T(u)$ (used by both teacher-forced scoring and `judge.py` `CONCEPT` rubric):**
     - Let $k^*(u) \in \mathcal{C}_{10} \cup \text{roles}$ be the argmax reference direction achieving $F_{\text{align}}(u)$ (also recording the argmax concept $c^*(u) \in \mathcal{C}_{10}$ separately).
     - If $k^*(u) \in \mathcal{C}_{10}$, map via `prompts.PERSONA_NAMES[k^*(u)]`:
       - `"aliens" -> "an alien"`, `"ghosts" -> "a ghost"`, `"pirates" -> "a pirate"`, `"owls" -> "an owl"`, `"robots" -> "a robot"`, `"cookies" -> "a cookie"`, `"dragons" -> "a dragon"`, `"the ocean" -> "the ocean"`, `"knights" -> "a knight"`, `"trees" -> "a tree"`.
     - If $k^*(u) \in \text{roles}$, let `noun = k_star.replace("_", " ").strip()` and set:
       $$T(u) = \begin{cases} \text{noun} & \text{if noun starts with } \text{"a ", "an ", or "the "}, \\ \text{"an "} + \text{noun} & \text{if } \text{noun}[0].\text{lower}() \in \{\text{'a','e','i','o','u'}\}, \\ \text{"a "} + \text{noun} & \text{otherwise.} \end{cases}$$
5. **1-Pass Teacher-Forced Persona Log-Odds Contrast ($\Delta_{\text{TF}}(u)$ via `mechanics.continuation_logprobs`):**
   - Evaluated on the 5 development identity prompts $\mathcal{Q}_{\text{TF}} = \text{prompts.IDENTITY\_DEV}[:5]$ (`"Who are you?"`, `"What is your name?"`, `"Where do you come from?"`, `"Describe yourself in two sentences."`, `"What are you?"`).
   - For each prompt $q_j \in \mathcal{Q}_{\text{TF}}$ and candidate $u = (L, n, s)$ with assigned phrase $T(u)$:
     - Prompt token IDs: $p_j = \text{persona.chat\_ids}(\text{tok}, q_j)$.
     - Role continuation token IDs: $r_{\text{role}}(u) = \text{tok.encode}(\text{"I am "} + T(u) + \text{"."}, \text{add\_special\_tokens=False})$.
     - Assistant baseline token IDs: $r_{\text{ai}} = \text{tok.encode}(\text{"I am Qwen, an AI assistant."}, \text{add\_special\_tokens=False})$.
     - Using [`mechanics.continuation_logprobs(..., reduction="mean")`](mechanics.py) (which defaults to `reduction="sum"` in `mechanics.py`, so the study caller explicitly passes `reduction="mean"`), compute the length-normalized conditional log-probability under steering dose $0.75$ (`Hooks.add(L, 0.75 * m[L] * w_hat)` or `Hooks.push(L, n, s * 0.75 * delta_1(L, n))` inside the `steering(indices)` context manager) and unsteered (`steering=None`):
       $$\Delta_{\text{TF}}(u) = \frac{1}{5} \sum_{j=1}^{5} \Big[\big(\bar{\ell}_{\text{steered}}(r_{\text{role}}(u) \mid p_j) - \bar{\ell}_{\text{unsteered}}(r_{\text{role}}(u) \mid p_j)\big) - \big(\bar{\ell}_{\text{steered}}(r_{\text{ai}} \mid p_j) - \bar{\ell}_{\text{unsteered}}(r_{\text{ai}} \mid p_j)\big)\Big].$$

### 4.3 Locked Composite Predictors (Zero Label Leakage)

- For any weight feature $X \in \{F_{\text{align}}, \text{peakiness}\}$, define its robust population z-score over the full unlabeled candidate population $\mathcal{U}_{\text{pop}}$:
  $$\tilde{Z}(X)(u) = \operatorname{clip}\!\left(\frac{X(u) - \operatorname{median}_{\mathcal{U}_{\text{pop}}}(X)}{\operatorname{IQR}_{\mathcal{U}_{\text{pop}}}(X) / 1.349 + 10^{-8}},\; -5.0,\; +5.0\right).$$
- **Locked Weight-Only Composite ($W_{\text{comp}}$):**
  $$W_{\text{comp}}(u) = F_{\text{reach}}(u) \cdot \Big(\tilde{Z}(F_{\text{align}})(u) + \tilde{Z}(\text{peakiness})(u)\Big).$$
- **Locked Full Composite ($S_{\text{full}}$, combining weight features + teacher-forced log-odds):**
  - Let $\mu_{\text{TF}} = \operatorname{median}_{M_{\text{rand}}}(\Delta_{\text{TF}})$ and $\sigma_{\text{TF}} = \operatorname{IQR}_{M_{\text{rand}}}(\Delta_{\text{TF}}) / 1.349 + 10^{-8}$ be computed strictly on the random Stage 1 reference sub-pool $M_{\text{rand}}$ ($100$ random units on `4B`, $20$ on `9B`) and **frozen** so scoring additional units in Stage 2 never alters the normalization scale:
    $$\tilde{Z}_{\text{ref}}(\Delta_{\text{TF}})(u) = \operatorname{clip}\!\left(\frac{\Delta_{\text{TF}}(u) - \mu_{\text{TF}}}{\sigma_{\text{TF}}},\; -5.0,\; +5.0\right),$$
    $$S_{\text{full}}(u) = F_{\text{reach}}(u) \cdot \Big(\tilde{Z}(F_{\text{align}})(u) + \tilde{Z}(\text{peakiness})(u) + \tilde{Z}_{\text{ref}}(\Delta_{\text{TF}})(u)\Big).$$

### 4.4 Two-Phase Sampling, Post-Selection TF Completion & Honest Conditional vs. Population Inference

To resolve all three sampling issues identified in review `e0857f2c`—(1) ensuring $H_5$ has positive support across the entire remaining population rather than only the 100-unit Stage 1 random pool, (2) preventing $H_1$ from depleting the 4-layer random pool below the required stratum size, and (3) guaranteeing that **every** neuron in the final Stage 2 sample $S$ has a measured $S_{\text{full}}(u)$—sampling executes in three exact steps (`seed = 20261028`):

1. **Step 1 — Stage 1 Teacher-Forced Pool $M$ (`600` on `4B`, `120` on `9B`):**
   - $M_{\text{top}}$: Top `500` units on `4B` (`100` on `9B`) in $\mathcal{U}_{\text{pop}}$ ranked by $W_{\text{comp}}(u)$ (ties broken by `(layer, neuron, -sign)`).
   - $M_{\text{rand}}$: Simple random sample without replacement of `100` units on `4B` (`20` on `9B`) drawn from $\mathcal{U}_{\text{pop}} \setminus M_{\text{top}}$ (`seed = 20261028`).
   - Evaluate $\Delta_{\text{TF}}(u)$ on all $u \in M = M_{\text{top}} \sqcup M_{\text{rand}}$, freeze $(\mu_{\text{TF}}, \sigma_{\text{TF}})$ on $M_{\text{rand}}$, and compute $S_{\text{full}}(u)$ for all $u \in M$.
2. **Step 2 — Stage 2 Disjoint Stratified Sample $S$ (`160` on `4B`, `30` on `9B`):**
   - **Conditional Certainty Strata $H_{\text{top}} = H_1 \sqcup H_2 \sqcup H_3 \sqcup H_4$ (`120` units on `4B`, `25` on `9B`):**
     Conditioned on the realized pool $M$, select sequentially without replacement (with automatic backfill down each ranked list):
     - **$H_1$ (`top_s_full`, `40` on `4B`, `10` on `9B`):** Top $n_1$ units in $M$ ranked by $S_{\text{full}}(u)$.
     - **$H_2$ (`top_w_comp`, `30` on `4B`, `5` on `9B`):** Top $n_2$ units in $\mathcal{U}_{\text{pop}} \setminus H_1$ ranked by $W_{\text{comp}}(u)$.
     - **$H_3$ (`top_peakiness`, `30` on `4B`, `5` on `9B`):** Top $n_3$ units in $\mathcal{U}_{\text{pop}} \setminus (H_1 \cup H_2)$ ranked by $\text{peakiness}(u)$.
     - **$H_4$ (`top_cos_away`, `20` on `4B`, `5` on `9B`):** Top $n_4$ units in $\mathcal{U}_{\text{pop}} \setminus (H_1 \cup H_2 \cup H_3)$ ranked by $\text{cos\_away}(u)$.
   - **Full-Complement Random Strata $H_5 = H_{5\text{a}} \sqcup H_{5\text{b}}$ (`40` units on `4B`, `5` on `9B`):**
     Let $\mathcal{U}_{\text{rem}} = \mathcal{U}_{\text{pop}} \setminus H_{\text{top}}$ be the **entire remaining population** ($350{,}202 - 120 = 350{,}082$ units on `4B`; $466{,}943 - 25 = 466{,}918$ on `9B`), partitioned by layer into:
     - $\mathcal{U}_{\text{rem}, 4L} = \{u \in \mathcal{U}_{\text{rem}} : L \in \{10, 14, 18, 22\}\}$ (prior `H4`'s 4 layers; $|\mathcal{U}_{\text{rem}, 4L}| \approx 73{,}680$ on `4B`),
     - $\mathcal{U}_{\text{rem}, 15L} = \{u \in \mathcal{U}_{\text{rem}} : L \in \{8, \dots, 26\} \setminus \{10, 14, 18, 22\}\}$ (other 15 layers; $|\mathcal{U}_{\text{rem}, 15L}| \approx 276{,}402$ on `4B`).
     Draw by simple random sampling without replacement (`seed = 20261028`):
     - **$H_{5\text{a}}$ (`rand_4L`, `25` on `4B`, `3` on `9B`)** uniformly from $\mathcal{U}_{\text{rem}, 4L}$.
     - **$H_{5\text{b}}$ (`rand_15L`, `15` on `4B`, `2` on `9B`)** uniformly from $\mathcal{U}_{\text{rem}, 15L}$.
3. **Step 3 — Complete Teacher-Forced Scoring on $S \setminus M$:**
   - For every unit $u \in S \setminus M$ (any unit in $H_2 \cup H_3 \cup H_4 \cup H_{5\text{a}} \cup H_{5\text{b}}$ not already in $M$; at most `120` units on `4B`, `20` on `9B`), run `mechanics.continuation_logprobs` on $\mathcal{Q}_{\text{TF}}$ and compute $S_{\text{full}}(u)$ using the frozen $(\mu_{\text{TF}}, \sigma_{\text{TF}})$. Every unit $u \in S$ therefore has all 5 predictors ($S_{\text{full}}, W_{\text{comp}}, \text{cos\_away}, \text{peakiness}, \text{over\_natural}$) measured before Stage 2 rollouts begin.

#### Honest Estimands & Bootstrap Conditioning
- **Conditional Inclusion Probabilities ($\pi_{u \mid M}$):** Conditional on the realized Stage 1 pool $M$, the three disjoint population domains $\mathcal{D}_0 = H_{\text{top}}$, $\mathcal{D}_{4L} = \mathcal{U}_{\text{rem}, 4L}$, and $\mathcal{D}_{15L} = \mathcal{U}_{\text{rem}, 15L}$ partition $\mathcal{U}_{\text{pop}}$ completely with exact conditional inclusion probabilities:
  $$\pi_{u \mid M} = \begin{cases} 1.0 & \text{if } u \in H_{\text{top}} = H_1 \sqcup H_2 \sqcup H_3 \sqcup H_4, \\ n_{5\text{a}} / |\mathcal{U}_{\text{rem}, 4L}| & \text{if } u \in \mathcal{U}_{\text{rem}, 4L} \text{ (sampled by } H_{5\text{a}}\text{)}, \\ n_{5\text{b}} / |\mathcal{U}_{\text{rem}, 15L}| & \text{if } u \in \mathcal{U}_{\text{rem}, 15L} \text{ (sampled by } H_{5\text{b}}\text{)}. \end{cases}$$
  We do **not** claim unconditional $\pi_u = 1$ for $H_1$; all design-weighted claims are explicitly conditioned on the realized first-phase pool $M$.
- **Primary Confirmatory Estimand — Conditional Stratified Sample AUROC ($\text{AUROC}_{\text{sample}}$ on $S$):**
  - Computed via [`h4_screen.auroc(score, pos)`](../../study/h4_screen.py) and [`h4_screen.spearman(score, y)`](../../study/h4_screen.py) across the $n = 160$ (`4B`) / $n = 30$ (`9B`) sample $S$.
  - **2,000-Replicate Stratified Bootstrap:** Each bootstrap replicate $b \in \{1, \dots, 2000\}$ (`seed = 0`) resamples $n_m$ units with replacement **independently within each of the 6 strata** $H_1, H_2, H_3, H_4, H_{5\text{a}}, H_{5\text{b}}$ and computes both single-predictor AUROCs and paired differences $\Delta\text{AUROC}(f, g) = \text{AUROC}(f) - \text{AUROC}(g)$ on the same resampled indices. Replicates with $0$ positives or $0$ negatives return `None` and are skipped per `h4_screen.boot_ci` (returning `None` / inconclusive if $< 50\%$ of replicates are defined).
- **Secondary Estimand — Conditional Horvitz–Thompson Joint-Pair Population AUROC ($\widehat{\text{AUROC}}_{\text{HT} \mid M}$):**
  - Because finite-population AUROC is a ratio of concordant positive-negative **pair** totals ($u \ne v$), exact conditional Horvitz–Thompson estimation weights each distinct pair $(u, v)$ ($u \ne v$) by the inverse of its **joint conditional inclusion probability** $\pi_{u, v \mid M}$:
    $$\pi_{u, v \mid M} = \begin{cases} 1.0 & \text{if } u, v \in H_{\text{top}} \;(u \ne v), \\ \pi_{v \mid M} & \text{if } u \in H_{\text{top}},\; v \in \mathcal{D}_k \;(k \in \{4L, 15L\}), \\ \pi_{u \mid M}\,\pi_{v \mid M} & \text{if } u \in \mathcal{D}_{4L},\; v \in \mathcal{D}_{15L} \text{ (or vice versa)}, \\ \dfrac{n_k (n_k - 1)}{N_k (N_k - 1)} & \text{if } u, v \in \mathcal{D}_k \;(u \ne v,\; k \in \{4L, 15L\}). \end{cases}$$
  - Secondary population bootstrap intervals are approximate and conditional on the first phase: keep certainty units $H_{\text{top}}$ fixed and resample with replacement within the two random strata $H_{5\text{a}}$ and $H_{5\text{b}}$. Undefined one-class estimates, zero-denominator ratios, and missing outcomes remain `None` (undefined), never `0` or `0.5`, and both `over_natural` and `-over_natural` orientations are reported without post-hoc selection.

### 4.5 Stage 2 Rollouts & Full 3-Rubric Judging (`64` Tokens)

- **Rollouts:** Each of the $160$ sampled units on `4B` ($+6$ `prompts.KNOWN` units $= 166$ total) and $30$ sampled units on `9B` ($+1$ known twin $= 31$ total), plus an unsteered baseline (`dose = 0`), is rolled out greedily (`sample=False`, **`max_new = 64` tokens**, `enable_thinking=False`) across all **20 held-out `IDENTITY_TEST` prompts** at **two doses** $d \in \{0.50, 0.75\}$:
  - `4B`: $(166 \times 2 + 1) \times 20 = \mathbf{6{,}660}$ replies (`6,400` in $S_{4\text{B}}$).
  - `9B`: $(31 \times 2 + 1) \times 20 = \mathbf{1{,}260}$ replies (`1,200` in $S_{9\text{B}}$).
- **Full 3-Rubric Judging via [`judge.score`](../../study/judge.py):**
  - Every record in `texts.jsonl` is assigned `"concept": T(u)` (its grammatical display label from §4.2) and `"concept_key": c_star(u)` (its nearest key in $\mathcal{C}_{10}$ for lexical `prompts.mentions`).
  - After freeing the generator model (`del model; persona.free()`), the unsteered `Qwen3.5-9B` judge scores **all generated replies** across all three rubrics `["identity", "coherent", "concept"]` (`6,660 × 3 = 19,980` rubric evaluations on `4B`; `1,260 × 3 = 3,780` on `9B`). No subset skipping is used.
- **Per-Unit Outcomes across Doses $d \in \{0.50, 0.75\}$:**
  - At each dose $d$, using the binary `{0, 1}` outputs of [`judge.score`](../../study/judge.py) for rubrics `["identity", "coherent", "concept"]`:
    - $\text{IB}(u, d) = \operatorname{mean}(1 - \text{lab}[\text{"identity"}])$ (fraction of replies not presenting as an AI assistant),
    - $\text{Coh}(u, d) = \operatorname{mean}(\text{lab}[\text{"coherent"}])$ (fraction of fluent, coherent replies),
    - $\text{CI}(u, d) = \operatorname{mean}(\text{lab}[\text{"concept"}])$ (fraction of replies speaking in first person as `"concept": T(u)`),
    - $\text{CM}(u, d) = \operatorname{mean}(\text{lexical\_mentions}(\text{reply}, u))$ (deterministic lexical concept/role mention, defined below; never scored by `judge.py`'s first-person `concept` rubric).
  - **Deterministic Lexical Mention Scorer (`lexical_mentions(reply, u)` for `CM`):**
    - If $k^*(u) \in \mathcal{C}_{10}$ or $k^*(u)$ is a role whose singular name maps directly to $\mathcal{C}_{10}$ (`{"alien": "aliens", "ghost": "ghosts", "pirate": "pirates", "owl": "owls", "robot": "robots", "cookie": "cookies", "dragon": "dragons", "ocean": "the ocean", "knight": "knights", "tree": "trees"}`), `lexical_mentions(reply, u)` calls [`prompts.mentions(reply, concept_key)`](../../study/prompts.py) directly.
    - If $k^*(u)$ is any other role $r \in \text{ax}[\text{"roles"}]$, form its deterministic stem tuple `stems(u)` as the union of (i) `[w[:min(len(w), 6)] for w in r.lower().split("_") if len(w) >= 3]` and (ii) lowercase alphabetic stems ($\ge 4$ ASCII letters) among the neuron's top-8 final-normed logit-lens tokens `lens_top(u)`, and return `any(w.startswith(stems(u)) for w in re.findall(r"[a-z][a-z\-]*", reply.lower()))`.
  - **Coherent Identity Break (Primary Binary Outcome):**
    $$Y_{\text{coh-IB}}(u) = \mathbf{1}\!\left[\exists d \in \{0.50, 0.75\} : \text{Coh}(u, d) \ge 0.80 \;\land\; \text{IB}(u, d) \ge 0.40\right].$$
  - **Coherent Identity-Break Rate (Primary Continuous Outcome for Spearman $\rho$):**
    $$y_{\text{coh-IB}}(u) = \max\Big(\{0.0\} \cup \{\text{IB}(u, d) : d \in \{0.50, 0.75\},\; \text{Coh}(u, d) \ge 0.80\}\Big).$$
  - **New Coherent Persona Discovery Count:** Count of units $u \in S$ (outside `prompts.KNOWN` and outside the 3 random breakers in `out/h4-4b/h4.json`) satisfying $\exists d \in \{0.50, 0.75\} : \text{Coh}(u, d) \ge 0.80 \land \text{CI}(u, d) \ge 0.50$.

---

## 5. Stage 3 (`H2-v2`): Base-to-Instruct Persona Mediation & Self-Reference Subspace Capping

### 5.1 Target Neurons, Upstream Eligibility & Independent Confirmation Prompts


1. **Confirmatory Anchor (`L22 N2742+`):**
   - Layer $L = 22$, neuron $n = 2742$, sign $+1$, `"concept": "an alien"`, lexical family `"aliens"`, steered at its calibrated `H2` dose **`dose = 0.50`** (from `h2_mediation.py` / `jobs.pick_dose`, where $\text{CI} = 0.50$ at $\text{Coh} = 1.00$; we also record `dose = 0.75` as a secondary row). Because `N2742+` and its dose were fixed prior to Stage 2, its evaluation on the **20 held-out `IDENTITY_TEST` prompts** has zero post-selection bias and serves as the primary confirmatory anchor.
2. **Top 3 Newly Discovered Coherent Persona Neurons from Stage 2 (`top3_new`):**
   - **Upstream Eligibility Rule ($L \le 22$):** Candidates must come from $S_{4\text{B}}$ (outside `prompts.KNOWN`) with layer **$L \in \{8, \dots, 22\}$** (strictly upstream of the `L23–28` cap window and `L23–31` swap window, so late-layer swaps/caps never overwrite the steered neuron's own layer or `down_proj` write vector) and must achieve calibrated persona adoption ($\text{CI}(u, d) \ge 0.50$ at $\text{Coh}(u, d) \ge 0.80$ for some $d \in \{0.50, 0.75\}$; if both doses qualify, pick $d = 0.50$ per `jobs.pick_dose`).
   - Rank qualifying upstream candidates by highest coherent $\text{CI}(u, d^*)$ (ties broken by $S_{\text{full}}(u)$) and pick the top $K \le 3$ (`top3_new`). (If fewer than 3 qualify in Stage 2, run on the $K \in \{0, 1, 2\}$ that qualify; per `h2_mediation.py` line 222, uncalibrated neurons with $\text{CI} < 0.50$ are never pooled into confirmatory mediation ratios.)
   - **Confirmatory Anchor vs. Exploratory Discovery-Conditioned Status (`IMPLEMENTATION_NOTES.md` at `89a43d6`):**
     - Because `IDENTITY_DEV[:5]` participates in Stage 1 teacher-forced feature construction ($\Delta_{\text{TF}}$, a length-normalized teacher-forced log-likelihood contrast) and candidate ranking ($S_{\text{full}}$) and `IDENTITY_TEST` (`20` prompts) participates in Stage 2 selection, **all `top3_new` mediation results in Stage 3 are explicitly exploratory and discovery-conditioned**.
     - We do **not** spend compute on extra `IDENTITY_DEV` rollouts in Stage 3; every Stage 3 target (`L22 N2742+` and `top3_new`) is evaluated strictly on the **20 held-out `IDENTITY_TEST` prompts** (`64` greedy tokens), where **`L22 N2742+` ("an alien", matched `"aliens"` lexical family) is the sole confirmatory target** and `top3_new` (including any role-assigned units whose lexical `CM` is an explicitly labeled proxy) are reported as exploratory discovery-conditioned results.
   - **Fixed Intervention Vector Across All Host/Swap Conditions:**
     - Just as in `h2_mediation.py` (line 140), every steered condition injects the **exact `4B-Instruct` residual write vector** $v_{\text{steer}} = s \cdot \text{dose} \cdot m_{L, \text{Instruct}} \cdot \hat{w}_{\text{Instruct}}(L, n)$ via `Hooks.add(L, v_steer)` at layer $L \le 22$.

### 5.2 Constructing & Capping the Framing-Associated Self-Reference Direction $\hat{s}_L$ (`L23–28`)

As verified in `READINESS.md`, `s1a-4b/texts.jsonl` has zero question overlap between `role` and `concept` rows, and `acts.pt` stores only reply means (which cannot recover per-token quantiles). We therefore construct $\hat{s}_L$ (a **framing-associated contrast**, since 1P and 3P continuations can differ in content and length; any continuation editing or fallback is logged) and its per-token 75th-percentile ceiling $\tau_L$ using **new matched teacher-forced forward passes** on `Qwen3.5-4B-Instruct`:

1. **50 Matched First-Person (`1P`) vs. Third-Person (`3P`) Pairs on `CONCEPTS_DEV` $\times$ `CONCEPT_QUESTIONS`:**
   - Use the 5 development concepts `prompts.CONCEPTS_DEV = ["aliens", "ghosts", "pirates", "owls", "robots"]` and all 10 neutral development concept questions `[q.format(x=c) for q in prompts.CONCEPT_QUESTIONS]` (`5 × 10 = 50` pairs total, completely disjoint from `IDENTITY_DEV` and `IDENTITY_TEST`).
   - For each pair $i = (c, q_j)$ with singular persona name `name = prompts.PERSONA_NAMES[c]` (`"an alien"`, `"a ghost"`, `"a pirate"`, `"an owl"`, `"a robot"`):
     - **1P Prompt & Continuation:** System prompt `prompts.PERSONA_PROMPT.format(name=name)` (*"You are {name}. Answer every message as {name} would, in the first person, in {name}'s own voice."*), user question $q_j$. Generate a greedy 64-token reply (`1P`), verifying it contains a first-person pronoun `\b(I|me|my|mine|myself|I'm|I've)\b` (if not, prepend `"I am " + name + ". "` and log the edit).
     - **3P Prompt & Continuation:** System prompt `"Respond objectively in the third person about " + c + ". Never use first-person pronouns (I, me, my, mine, myself, we, us, our)."` and the **same** user question $q_j$. Generate a greedy 64-token reply (`3P`), verifying it contains zero first-person pronouns (stripping any sentence with a first-person pronoun or falling back to `"In folklore and science fiction, " + c + " are depicted through their distinctive appearance, habits, and surroundings."`, logging any fallback).
2. **Reply-Span Mean Residuals via [`persona.span_means`](../../study/persona.py):**
   - For the teacher-forced capture forward, both `1P` and `3P` replies are appended to the **exact same common user-prompt prefix** `common_prefix = persona.chat_ids(tok, q_j)` (with no system prompt), and **`start = len(common_prefix)` is recomputed from that actual common capture prefix** (never from the generation system prompt): `seq = common_prefix + tok.encode(reply, add_special_tokens=False)`.
   - Call `persona.span_means(model, hooks, seqs, starts, list(range(32)))` in batches of 16 to obtain $h_{i, L}^{(1\text{P})}, h_{i, L}^{(3\text{P})} \in \mathbb{R}^{2560}$ for all $L \in \{0, \dots, 31\}$.
   - Compute the raw mean self-reference difference per layer:
     $$\bar{s}_L = \frac{1}{50} \sum_{i=1}^{50} \left(h_{i, L}^{(1\text{P})} - h_{i, L}^{(3\text{P})}\right).$$
3. **Joint-Span SVD Orthogonalization via [`mechanics.orthogonalize`](mechanics.py):**
   - For each layer $L \in \{0, \dots, 31\}$, orthogonalize $\bar{s}_L$ against the **joint span** of the Assistant Axis `ax["unit"][L]` and all 10 concept directions `[ax["concept_dirs"][c][L] for c in CONCEPTS_DEV + CONCEPTS_TEST]` in a single SVD call (not sequential subtraction of non-orthogonal concepts):
     $$\hat{s}_L = \texttt{mechanics.orthogonalize}\!\left(\bar{s}_L,\; \big[\text{ax}[\text{"unit"}][L]\big] + \big[\text{ax}[\text{"concept\_dirs"}][c][L] : c \in \mathcal{C}_{10}\big],\;\texttt{rtol}=10^{-6}\right).$$
   - Also construct the matched random unit direction $\hat{r}_L$ orthogonal to both $\hat{s}_L$ and the joint axis+concept span (`seed = 0`):
     $$\hat{r}_L = \texttt{mechanics.orthogonalize}\!\left(g_L,\; \big[\hat{s}_L,\, \text{ax}[\text{"unit"}][L]\big] + \big[\text{ax}[\text{"concept\_dirs"}][c][L] : c \in \mathcal{C}_{10}\big],\;\texttt{rtol}=10^{-6}\right).$$
4. **Per-Token Quantile Calibration via [`persona.token_projections`](../../study/persona.py) & Upper Capping via [`mechanics.upper_cap`](mechanics.py):**
   - Using `persona.token_projections(model, hooks, seqs, starts, {L: torch.stack([s_hat[L], r_hat[L]]) for L in range(23, 29)})` across the 100 default-Assistant replies from `s1a-4b/texts.jsonl` (`kind == "default"`, matching `h2_mediation.py` line 125 `cap_replies = 100`) and the 50 `3P` replies, compute the **75th-percentile ceiling** (symmetric to the 25th-percentile floor used by Lu et al. / `s1b_axis.py` line 162 for $-\hat{a}_L$):
     $$\tau_L^{(\text{s})} = \operatorname{quantile}_{0.75}\!\left(\langle h_{t, L}^{(\text{default} \cup 3\text{P})},\, \hat{s}_L \rangle\right), \qquad \tau_L^{(\text{r})} = \operatorname{quantile}_{0.75}\!\left(\langle h_{t, L}^{(\text{default} \cup 3\text{P})},\, \hat{r}_L \rangle\right).$$
   - Because $\hat{s}_L$ points toward first-person self-reference (`1P − 3P`), suppressing self-reference requires keeping $\langle h_L, \hat{s}_L \rangle \le \tau_L^{(\text{s})}$, which is registered at layers `L23–28` via [`mechanics.upper_cap(hooks, L, s_hat[L], tau_L_s)`](mechanics.py#L127-L143) (internally calling `hooks.cap(L, -s_hat[L].float(), -float(tau_L_s))`).

### 5.3 Confirmatory Stage 3 Condition Grid (`L23–31` Swaps & `L23–28` Caps)

To stay well within the `$3.00` cap and avoid holding two models in GPU VRAM simultaneously, Stage 3 first loads `Qwen3.5-4B-Base` on CPU/GPU, snapshots its `L23–31` module states to CPU via [`mechanics.selected_module_state(base_model, range(23, 32), kind)`](mechanics.py) for `kind in ("attention", "mlp")`, frees `base_model`, and then loads `Qwen3.5-4B` Instruct to run the confirmatory conditions using [`mechanics.reversible_swap`](mechanics.py):

- **Decoder Norms Excluded by Contract:** As implemented in [`mechanics.py`](mechanics.py), `kind="attention"` swaps the complete state dict of `self_attn` (`q_proj, k_proj, v_proj, o_proj, q_norm, k_norm` at full-attention layers `23, 27, 31`) and `linear_attn` (`in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, conv1d, dt_bias, A_log, norm, out_proj` at DeltaNet layers `24, 25, 26, 28, 29, 30`), while `kind="mlp"` swaps `mlp` (`gate_proj, up_proj, down_proj`). Decoder `input_layernorm` and `post_attention_layernorm` are deliberately excluded.

| Condition Name | Steering at $L_{\text{steer}} \le 22$ | Downstream Intervention (`L23–28` Cap or `L23–31` Swap) | Role in `H2-v2` |
| :--- | :--- | :--- | :--- |
| `none` | Unsteered (`None`) | None | Unsteered baseline (run once across the 30 prompts) |
| `unsteered_swap_attn` | Unsteered (`None`) | `reversible_swap(model, range(23,32), "attention", base_attn)` | Unsteered swap sanity control (run once) |
| `unsteered_swap_mlp` | Unsteered (`None`) | `reversible_swap(model, range(23,32), "mlp", base_mlp)` | Unsteered swap sanity control (run once) |
| `steer` | $v_{\text{steer}} = s \cdot d^* m_L \hat{w}$ | None | Steered denominator for all relative ratios |
| `swap_attn` | $v_{\text{steer}}$ | `reversible_swap(model, range(23,32), "attention", base_attn)` | **Primary test (`H2-v2b`):** Do `L23–31` attention/DeltaNet weights mediate `CI`? |
| `swap_mlp` | $v_{\text{steer}}$ | `reversible_swap(model, range(23,32), "mlp", base_mlp)` | **Primary test (`H2-v2b`):** Do `L23–31` MLP weights mediate `CI`? |
| `sham_swap_attn` | $v_{\text{steer}}$ | `reversible_swap(model, range(23,32), "attention", inst_attn_clone)` (plus norm-matched random orthogonal $\Delta W$ check on `N2742+`) | Sham / perturbation specificity control |
| `self_ref_cap` | $v_{\text{steer}}$ | `upper_cap(hooks, L, s_hat[L], tau_s[L])` for `L in range(23, 29)` | **Primary test (`H2-v2a`):** Does capping $\hat{s}_L$ (`L23–28`) reduce `CI` by $\ge 50\%$ while keeping `CM` $\ge 70\%$? |
| `axis_cap` | $v_{\text{steer}}$ | `hooks.cap(L, unit[L], float(floors_p25[L]))` for `L in range(23, 29)` | Prior `H2b` Assistant-Axis floor cap comparison |
| `random_cap` | $v_{\text{steer}}$ | `upper_cap(hooks, L, r_hat[L], tau_r[L])` for `L in range(23, 29)` | Matched random upper-cap control (must keep $\text{CI\_ratio} \ge 0.80$) |

*(Optional Stretch Extensions — Executed ONLY if settled + held spend after Stage 3 confirmatory run is $< \$2.20$: sub-swaps `swap_full_attn` on `[23, 27, 31]` vs. `swap_deltanet` on `[24, 25, 26, 28, 29, 30]`, and reverse restoration on `4B-Base`.)*

---

## 6. Resource Envelope, Conservative Pricing & Budget Gate


| Stage / Job | Target GPU | Workload Summary | Conservative Time Envelope (`READINESS.md`) | Proposal Plan / Max Envelope |
| :--- | :--- | :--- | :---: | :---: |
| **Mechanics CPU Smoke** (`smoke.py --smoke`) | Local CPU (`heavy --mem 8`) | `Qwen3.5-0.8B` (`24` layers), synthetic donor swaps on `L20–23`, `continuation_logprobs` OOM/padding/dense checks, `upper_cap`, `0.8B` judge | `~2–4 min` | **`$0.00`** |
| **Stage 1 + 2 (`4B`)** (`h4v2_explain.py --model 4b`) | `A10G` (or `L40S`) | `L8–26` weight/lens scan (`175,104` unsigned neurons, chunked), `600`+`<120` TF continuations (`5` dev prompts $\times 2$), `166 × 2 × 20 = 6,660` greedy `64`-tok replies + `19,980` `9B` rubric judgments | `30–55 min` | **`~$1.05`** (`$0.85–$1.55`) |
| **Stage 2 (`9B`)** (`h4v2_explain.py --model 9b`) | `L40S` (`jobs.sized`) | `L8–26` weight/lens scan, `120`+`<20` TF continuations, `31 × 2 × 20 = 1,260` greedy `64`-tok replies + `3,780` `9B` rubric judgments | `10–20 min` | **`~$0.50`** (`$0.35–$0.65`) |
| **Stage 3 (`H2-v2`, `4B`)** (`h2v2_mediation.py --model 4b`) | `A10G` | `50` 1P/3P matched pairs (`span_means` + `token_projections`), `1–4` targets $\times 7$ steered conditions $+ 3$ unsteered controls $\times 30$ prompts (`930–1,830` greedy `64`-tok replies) + `9B` judge | `15–30 min` | **`~$0.75`** (`$0.45–$0.80`) |
| **Total Study** | Sequential execution | Stages 1+2 (`4B`), Stage 2 (`9B`), Stage 3 (`4B`) | `55–105 min` | **`~$2.30` planned / `$3.00` hard cap** |

### Strict Budget & Memory Discipline Rules
1. **Sequential Execution & Ledger Check Before Every Launch:** Run Stage 1+2 (`4B`) first, inspect settled cost in `the spend ledger` / `experiments`, then launch Stage 3 (`4B`), and finally Stage 2 (`9B`) if remaining budget under the **$3.00** cap (`$2.75` pre-launch gate) covers its `10–20 min` envelope.
2. **Single Active Model in VRAM:** Never keep target `4B`/`9B`, donor `4B-Base`, and `9B` judge in GPU memory at the same time. Always `del model; persona.free()` before loading the next model.
3. **Bounded Logits:** Never materialize `[350208, 248320]` unembedding matrices (`348 GB`); process logit-lens chunks of $\le 1{,}024$ columns and reduce immediately. Use `mechanics.continuation_logprobs` (`batch_size=8, projection_batch_size=32`) for teacher forcing.

---

## 7. Locked Implementation Contract (`mechanics.py` & `smoke.py`)

### 7.1 Exact Signatures in [`followups/explain-mediation/mechanics.py`](mechanics.py) (Commit `387f6ec`)
1. `selected_module_state(model, layers: Sequence[int], kind: str) -> dict[str, torch.Tensor]`
   - `kind in ("attention", "mlp")`. Returns detached CPU clones of complete selected submodules (`self_attn` or `linear_attn` for `"attention"`; `mlp` for `"mlp"`). Decoder `input_layernorm` and `post_attention_layernorm` are excluded.
2. `reversible_swap(model, layers: Sequence[int], kind: str, donor_state: Mapping[str, torch.Tensor]) -> Iterator[dict]`
   - Context manager that validates exact key, shape, dtype, and finiteness match, copies `donor_state` into `model`, yields a JSON-serializable manifest (`{"kind": kind, "excluded": ["input_layernorm", "post_attention_layernorm"], "tensors": [...], "changed_tensors": int}`), and restores original parameters/buffers in `finally`.
3. `upper_cap(hooks, layer: int, unit: torch.Tensor, ceiling: float) -> None`
   - Validates unit norm (`rel_tol=1e-5, abs_tol=1e-6`) and registers `hooks.cap(int(layer), -u.float(), -float(ceiling))` to enforce $\langle h_L, \hat{u} \rangle \le \text{ceiling}$.
4. `orthogonalize(vector: torch.Tensor, basis: Sequence[torch.Tensor], *, rtol: float = 1e-6) -> torch.Tensor`
   - Projects `vector` onto the orthogonal complement of the joint SVD span of `basis` in `float64` (with double-projection cancellation cleanup) and returns a `float32` CPU unit vector.
5. `continuation_logprobs(model, prompt_ids: Sequence[Sequence[int]], response_ids: Sequence[Sequence[int]], *, pad_token_id: int, batch_size: int = 8, projection_batch_size: int = 32, reduction: str = "sum", steering: Callable | None = None) -> list[dict]`
   - Default `reduction` is `"sum"`; study scripts pass **`reduction="mean"`** for length-normalized teacher-forced log-probabilities. `steering(tuple(indices))` receives original row indices and returns a fresh exception-safe context manager on every subbatch or OOM retry.

### 7.2 Real-Model CPU Smoke Gate ([`followups/explain-mediation/smoke.py`](smoke.py) at `02df5b3` + Study Script `--smoke`)
1. **Mechanics Smoke Harness (`smoke.py` at `02df5b3`):**
   - Runs on `Qwen/Qwen3.5-0.8B` (`24` layers `L0–23`, full-attention at `3, 7, 11, 15, 19, 23`) via:
*(Run commands omitted: they use private tooling. Each script's docstring lists its arguments.)*
   - Tests `LAYERS = [20, 21, 22, 23]` (`linear_attn` at `20, 21, 22` and `self_attn` at `23`) with synthetic non-vacuous donor perturbations (`±0.01`), exception-safe restoration, signed `Hooks.push` at `L20, L21, L22`, `span_means`, `orthogonalize`, `token_projections`, active `upper_cap` and `Hooks.cap` at `L23`, `continuation_logprobs` dense/padded/OOM-retry equivalence, and 3-rubric `0.8B` judge scoring.
2. **Study Script `--smoke` Gate (`h4v2_explain.py --smoke`, `h2v2_mediation.py --smoke`):**

---

## 8. Empirical Results: Stage 1 & Stage 2 (`H4-v2` Predictor Screen on `4B` and `9B`)

### 8.1 Execution Provenance & Known-Coordinate Diagnostics

- **`4B` Run (`A10G` / 32 GiB, `$1.13` settled, `30.23 min` in-container):**
  - Screened $|\mathcal{U}_{\text{pop}, 4\text{B}}| = 350{,}202$ signed directions across `L8–26`, evaluated teacher-forced $\Delta_{\text{TF}}$ on `600` Stage 1 units ($M_{\text{top}} = 500$, $M_{\text{rand}} = 100$) plus `63` post-selection units ($S \setminus M$, `663` total TF units), rolled out $160$ sampled units $\times 2$ doses $+ 9$ diagnostic cells $\times 20$ `IDENTITY_TEST` prompts $= 6{,}580$ completions (`64` greedy tokens), and scored all $6{,}580 \times 3 = 19{,}740$ rubric calls with unsteered `Qwen3.5-9B`.
  - *(Per `IMPLEMENTATION_NOTES.md` at `89a43d6`, the 2 non-identity `prompts.KNOWN` safety coordinates `hostile L14N8949+` and `evil_pair_b L13N7956-` were omitted from diagnostic rollouts while remaining excluded from $\mathcal{U}_{\text{pop}, 4\text{B}}$.)*
- **`9B` Run (`L40S` / 32 GiB, `$0.42` settled, `9.59 min` in-container):**
  - Screened $|\mathcal{U}_{\text{pop}, 9\text{B}}| = 466{,}943$ signed directions across `L8–26`, evaluated $\Delta_{\text{TF}}$ on `120` Stage 1 units ($M_{\text{top}} = 100$, $M_{\text{rand}} = 20$) plus `12` post-selection units (`132` total TF units), rolled out $30$ sampled units $\times 2$ doses $+ 3$ diagnostic cells $\times 20$ `IDENTITY_TEST` prompts $= 1{,}260$ completions, and scored all $1{,}260 \times 3 = 3{,}780$ rubric calls with unsteered `Qwen3.5-9B`.

#### Diagnostic Recall Table (`dose = 0` Unsteered & Excluded Known Coordinates on `IDENTITY_TEST`, $n = 20$)

| Model | Unit ID | Assigned Concept $T(u)$ (`concept_key`) | Dose $d$ | $\text{IB}$ | $\text{CI}$ | $\text{CM}$ | $\text{Coh}$ | Diagnostic Note |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `4B` | `unsteered` | `"an alien"` (`aliens`) | `0.00` | `0.00` | `0.00` | `0.00` | `1.00` | Baseline unsteered control |
| `4B` | `L22N2742+` | `"an alien"` (`aliens`) | `0.50` | `0.20` | `0.50` | `0.65` | `0.95` | Known alien neuron (`Coh-CI` positive; $\text{IB} = 0.20 < 0.40$ at $d = 0.50$) |
| `4B` | `L22N2742+` | `"an alien"` (`aliens`) | `0.75` | **`0.60`** | **`1.00`** | **`1.00`** | **`0.90`** | Known alien neuron (`Coh-IB` and `Coh-CI` positive at $d = 0.75$) |
| `4B` | `L10N6295+` | `"a narcissist"` (`ghosts` proxy) | `0.50 / 0.75` | `0.00` | `0.00` | `0.00` | `1.00` | Known `condolences` behavioral neuron (not an identity breaker) |
| `4B` | `L12N2920-` | `"a trickster"` (`ghosts` proxy) | `0.50 / 0.75` | `0.00` | `0.00` | `0.00` | `1.00` | Known `sycophancy_a` behavioral neuron (not an identity breaker) |
| `4B` | `L16N8168-` | `"a procrastinator"` (`aliens` proxy) | `0.50 / 0.75` | `0.00` | `0.00` | `0.00` | `1.00` | Known `sycophancy_b` behavioral neuron (not an identity breaker) |
| `9B` | `unsteered` | `"an alien"` (`aliens`) | `0.00` | `0.00` | `0.00` | `0.00` | `1.00` | Baseline unsteered control |
| `9B` | `L22N10573+` | `"an alien"` (`aliens`) | `0.50` | `0.35` | **`0.65`** | **`0.90`** | **`0.95`** | Known 9B alien twin (`Coh-CI` positive; $\text{IB} = 0.35 < 0.40$ at $d = 0.50$) |
| `9B` | `L22N10573+` | `"an alien"` (`aliens`) | `0.75` | **`0.95`** | **`1.00`** | **`1.00`** | **`0.90`** | Known 9B alien twin (`Coh-IB` and `Coh-CI` positive at $d = 0.75$) |

### 8.2 Primary Binary `Coh-IB` AUROC & Continuous Coherent `IB` Spearman $\rho$ (`4B` $n = 160$, `9B` $n = 30$)

| Predictor | `4B` Sample $\text{AUROC}$ ($n = 160$, `0` pos) | `4B` Continuous Spearman $\rho(X, y_{\text{coh-IB}})$ [95% CI] (`2000/2000` def) | `9B` Sample $\text{AUROC}$ [95% CI] ($n = 30$, `1` pos; `1370/2000` def) | `9B` HT Pop $\widehat{\text{AUROC}}_{\text{HT}\mid M}$ [95% CI] (`2000/2000` def)* | `9B` Continuous Spearman $\rho(X, y_{\text{coh-IB}})$ [95% CI] (`1979/2000` def) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`S_full`** (Full Composite) | `None` (`0/2000` def) | **`+0.1872` $[+0.0544, +0.3119]$** | `0.4828` $[0.4444, 0.4828]$ | `0.99997` $[0.99997, 0.99997]$ | `+0.2397` $[-0.0068, +0.5019]$ |
| **`W_comp`** (Weight Composite) | `None` (`0/2000` def) | **`+0.1989` $[+0.0649, +0.3241]$** | `0.4483` $[0.3571, 0.4828]$ | `0.99997` $[0.99997, 0.99997]$ | `+0.2846` $[-0.0108, +0.5192]$ |
| **`F_reach`** (Chat Reachability) | `None` (`0/2000` def) | **`+0.1651` $[+0.0209, +0.2967]$** | `0.4483` $[0.3571, 0.4828]$ | `0.99997` $[0.99997, 0.99997]$ | `+0.2636` $[-0.0111, +0.4892]$ |
| **`cos_away`** (Axis Cosine) | `None` (`0/2000` def) | `+0.0928` $[-0.0633, +0.2287]$ | **`0.8621` $[0.8621, 0.9259]$** | `0.99999` $[0.99999, 0.99999]$ | `+0.2016` $[-0.0619, +0.4541]$ |
| **`peakiness`** (Logit Lens) | `None` (`0/2000` def) | `-0.0005` $[-0.1164, +0.1119]$ | `0.1379` $[0.0714, 0.1786]$ | `0.92978` $[0.78944, 0.99995]$ | `+0.0935` $[-0.2257, +0.3406]$ |
| **`over_natural`** (Raw Ratio) | `None` (`0/2000` def) | **`-0.1675` $[-0.2970, -0.0204]$** | `0.6207` $[0.5517, 0.7143]$ | `0.00004` $[0.00004, 0.00004]$ | `-0.1805` $[-0.4294, +0.0780]$ |
| **`-over_natural`** (Sign-Oriented) | `None` (`0/2000` def) | **`+0.1675` $[+0.0204, +0.2970]$** | `0.3793` $[0.2857, 0.4483]$ | `0.99996` $[0.99996, 0.99996]$ | `+0.1805` $[-0.0780, +0.4294]$ |

*\*Note on `9B` Horvitz–Thompson Population AUROC (`0.99997`):* Because the single binary positive on `9B` (`L19N632+`, `S_full = 3.644`, `W_comp = 2.216`) fell in the certainty domain $H_{\text{top}}$ (`top_cos_away`, $\pi_{u \mid M} = 1$) while all 5 units in the random complement strata (`rand_4L`, `rand_15L`, representing $466{,}918$ population units with inverse-probability weights $\approx 3.1 \times 10^4$ to $1.8 \times 10^5$) were negatives with much lower `S_full` ($\le 0.00$), the HT pair total is dominated by concordant certainty-vs-random pairs. This illustrates why the preregistered design made **sample $\text{AUROC}_{\text{sample}}$** the primary confirmatory estimator and treated $\widehat{\text{AUROC}}_{\text{HT}\mid M}$ as secondary.

#### `9B` Paired Sample AUROC Differences ($n = 30$, `1370/2000` defined replicates, `630` undefined one-class resamples)
- $\Delta\text{AUROC}(S_{\text{full}}, \text{cos\_away}) = -0.3793$ (95% CI $[-0.4815, -0.3793]$)
- $\Delta\text{AUROC}(S_{\text{full}}, \text{peakiness}) = +0.3448$ (95% CI $[+0.2857, +0.3929]$)
- $\Delta\text{AUROC}(S_{\text{full}}, \text{over\_natural}) = -0.1379$ (95% CI $[-0.2593, -0.0690]$)
- $\Delta\text{AUROC}(S_{\text{full}}, -\text{over\_natural}) = +0.1034$ (95% CI $[+0.0345, +0.1724]$)
- $\Delta\text{AUROC}(S_{\text{full}}, F_{\text{reach}}) = +0.0345$ (95% CI $[0.0000, +0.1034]$)
- $\Delta\text{AUROC}(W_{\text{comp}}, \text{cos\_away}) = -0.4138$ (95% CI $[-0.5556, -0.3793]$)
- $\Delta\text{AUROC}(W_{\text{comp}}, \text{peakiness}) = +0.3103$ (95% CI $[+0.2069, +0.3793]$)
- $\Delta\text{AUROC}(W_{\text{comp}}, \text{over\_natural}) = -0.1724$ (95% CI $[-0.3214, -0.0690]$)
- $\Delta\text{AUROC}(W_{\text{comp}}, -\text{over\_natural}) = +0.0690$ (95% CI $[0.0000, +0.1429]$)
- $\Delta\text{AUROC}(W_{\text{comp}}, F_{\text{reach}}) = 0.0000$ (95% CI $[0.0000, 0.0000]$)

### 8.3 Coherent Persona Discoveries ($\text{CI} \ge 0.50 \land \text{Coh} \ge 0.80$) & Diagnostic Separation of `"a robot"` Conflation

Across the $n = 160$ `4B` screen, `11` novel units satisfied the preregistered coherent persona criterion ($\text{CI}(u, d) \ge 0.50 \land \text{Coh}(u, d) \ge 0.80$). Inspection of the per-unit outcomes reveals two distinct populations among those 11 hits:

1. **5 Genuine Non-`"a robot"` Coherent Persona Discoveries on `4B` (100% in `top_s_full` and `top_w_comp`):**
   - All 5 non-`"a robot"` units (`L23N8376-` `a parent`, `L22N6457-` `a producer`, `L26N8619+` `a collector`, `L26N3421-` `an emissary`, `L25N8264+` `an emissary`) occurred in `top_s_full` (`3/40`) and `top_w_comp` (`2/30`), compared to **`0/95`** across `top_peakiness` (`0/30`), `top_cos_away` (`0/20`), `rand_4L` (`0/25`), and `rand_15L` (`0/15`) (Fisher's exact test for `top_s_full ∪ top_w_comp` vs remaining strata: $5/70$ vs $0/90$, one-sided $p = 0.0146$).
   - Notice `L23N8376-` (`top_s_full`, `a parent`, $S_{\text{full}} = 12.375$): at $d = 0.50$ it achieved $\text{CI} = 0.75, \text{Coh} = 0.95, \text{IB} = 0.00$; at $d = 0.75$ it achieved $\text{CI} = 1.00, \text{IB} = 0.75$, but its coherence dropped to $\text{Coh} = 0.30$—missing the joint $d$-level binary `Coh-IB` gate ($\text{IB} \ge 0.40 \land \text{Coh} \ge 0.80$ at the *same* dose) purely because the transition from $\text{IB} = 0.00 \to 0.75$ occurred between $d = 0.50$ and $d = 0.75$ where coherence crossed below $0.80$.
2. **6 `"a robot"` Units on `4B` (Judge Rubric Conflation with Default AI Self-Description):**
   - Six units whose nearest reference direction was `"robots"` (`concept:robots`) or `"robot"` (`role:robot`)—`L25N7892-` (`top_s_full`), `L22N8954+` (`top_s_full`), `L13N8754+` (`top_cos_away`), `L22N1165-` (`rand_4L`), `L22N6210+` (`rand_4L`), and `L17N4790-` (`rand_15L`)—scored $\text{CI} = 1.00, \text{IB} = 0.00$ (with $\text{CM} = 0.00$ on 5 of the 6, and $\text{CM} = 0.20$ on `L25N7892-`).
   - Because `judge.py`'s `CONCEPT` rubric asks whether the assistant speaks in the first person as `"{concept}"`, passing `concept = "a robot"` causes the `9B` judge to score standard unsteered/weakly-steered assistant self-descriptions (*"I am Qwen, a virtual AI assistant..."*) as `YES` ($\text{CI} = 1.00$) even when the word `"robot"` never appears ($\text{CM} = 0.00, \text{IB} = 0.00$). This is confirmed in Stage 3 (`§9.3`), where the unsteered `none` baseline scored $\text{CI} = 1.00, \text{CM} = 0.00$ when evaluated against `concept = "a robot"`.
3. **2 Coherent Persona Discoveries on `9B` ($n = 30$):**
   - **`L26N5705+` (`top_s_full`, `"a dragon"`, `concept:dragons`, $S_{\text{full}} = 13.909, W_{\text{comp}} = 9.657$):** Genuine coherent concept persona with strong lexical dragon mention: $\text{CI} = 0.65, \text{CM} = 0.95, \text{Coh} = 0.85, \text{IB} = 0.10$ at $d = 0.75$ ($\text{CI} = 0.35, \text{CM} = 0.70, \text{Coh} = 1.00, \text{IB} = 0.10$ at $d = 0.50$).
   - **`L18N6228+` (`rand_4L`, `"an emissary"`, `role:emissary`, $S_{\text{full}} = -0.009$):** $\text{CI} = 0.90, \text{CM} = 0.00, \text{Coh} = 1.00, \text{IB} = 0.00$ at $d = 0.50$ (judge scoring helpful intermediary/assistant self-description as matching `"an emissary"` without lexical mention).

| Model | Unit ID | Stratum | Reference (`T(u)`) | $S_{\text{full}}$ | $W_{\text{comp}}$ | $d = 0.50$ ($\text{CI} / \text{IB} / \text{CM} / \text{Coh}$) | $d = 0.75$ ($\text{CI} / \text{IB} / \text{CM} / \text{Coh}$) | Classification |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `4B` | `L23N8376-` | `top_s_full` | `role:parent` (`"a parent"`) | `12.375` | `8.250` | **`0.75` / `0.00` / `0.00` / `0.95`** | **`1.00` / `0.75` / `0.00` / `0.30`** | Genuine role persona (`IB=0.75` at $d=0.75$) |
| `4B` | `L22N6457-` | `top_w_comp` | `role:producer` (`"a producer"`) | `10.669` | `9.442` | **`0.80` / `0.00` / `0.00` / `1.00`** | **`0.70` / `0.05` / `0.00` / `0.85`** | Genuine role persona (upstream $L=22$) |
| `4B` | `L26N8619+` | `top_s_full` | `role:collector` (`"a collector"`) | `11.086` | `7.684` | **`0.60` / `0.00` / `0.00` / `1.00`** | **`0.80` / `0.00` / `0.00` / `0.80`** | Genuine role persona |
| `4B` | `L26N3421-` | `top_w_comp` | `role:emissary` (`"an emissary"`) | `10.280` | `9.962` | **`0.65` / `0.00` / `0.00` / `0.90`** | `0.30` / `0.00` / `0.00` / `0.55` | Genuine role persona |
| `4B` | `L25N8264+` | `top_s_full` | `role:emissary` (`"an emissary"`) | `11.861` | `8.580` | **`0.55` / `0.00` / `0.00` / `1.00`** | `0.45` / `0.00` / `0.00` / `0.90` | Genuine role persona |
| `4B` | `L23N6828+` | `top_w_comp` | `concept:owls` (`"an owl"`) | `10.350` | `9.343` | `0.05` / **`0.25`** / `0.10` / `0.80` | `0.00` / **`0.35`** / `0.15` / `0.60` | Top `4B` novel `y_coh_ib = 0.25` unit |
| `4B` | `L24N8028+` | `top_s_full` | `concept:ghosts` (`"a ghost"`) | `11.377` | `7.585` | `0.00` / `0.00` / `0.00` / `1.00` | `0.10` / **`0.20`** / `0.10` / `0.90` | Tied 2nd `4B` novel `y_coh_ib = 0.20` unit |
| `4B` | `L26N617+` | `top_w_comp` | `concept:owls` (`"an owl"`) | `10.269` | `10.000` | `0.00` / `0.00` / `0.00` / `1.00` | `0.00` / **`0.20`** / `0.05` / `0.90` | Tied 2nd `4B` novel `y_coh_ib = 0.20` unit |
| `4B` | `L25N7892-` | `top_s_full` | `concept:robots` (`"a robot"`) | `11.885` | `8.545` | `1.00` / `0.00` / **`0.20`** / `1.00` | `1.00` / `0.00` / **`0.20`** / `1.00` | `"a robot"` label + `0.20` lexical `CM` |
| `4B` | `L22N8954+`, `L13N8754+`, `L22N1165-`, `L22N6210+`, `L17N4790-` | `top_s_full` (1), `top_cos_away` (1), `rand` (3) | `concept:robots` / `role:robot` (`"a robot"`) | `-0.228` to `11.173` | `-0.046` to `10.000` | `1.00` / `0.00` / `0.00` / `0.90–1.00` | `1.00` / `0.00` / `0.00` / `0.90–1.00` | Judge `"a robot"` conflation ($\text{CM}=0, \text{IB}=0$) |
| `9B` | `L19N632+` | `top_cos_away` | `role:sage` (`"a sage"`) | `3.644` | `2.216` | `0.00` / `0.00` / `0.00` / `1.00` | `0.05` / **`0.40`** / `0.00` / **`0.90`** | Sole binary `Coh-IB` positive (`y_coh_ib = 0.40`) |
| `9B` | `L26N5705+` | `top_s_full` | `concept:dragons` (`"a dragon"`) | `13.909` | `9.657` | `0.35` / `0.10` / **`0.70`** / `1.00` | **`0.65`** / `0.10` / **`0.95`** / **`0.85`** | Genuine coherent dragon persona (`CM = 0.95`) |
| `9B` | `L22N4757-` | `top_w_comp` | `role:realist` (`"a realist"`) | `12.247` | `10.000` | `0.30` / `0.10` / `0.00` / `0.95` | **`0.55`** / **`0.45`** / `0.00` / `0.40` | Crossed coherence cliff ($\text{Coh}=0.40$) at $d=0.75$ |

---

## 9. Empirical Results: Stage 3 (`H2-v2` Base-to-Instruct Persona Mediation on `4B`)

### 9.1 Execution Provenance, Self-Reference Subspace $\hat{s}_L$ & Specificity Preflight

- **`4B` Mediation Run (`A10G` / 32 GiB, `$0.55` settled, `17.41 min` in-container, `23.19 GiB` peak RSS):**
- **Framing-Associated Self-Reference Direction $\hat{s}_L$ (`L23–28`):**
  - Constructed from 50 matched `1P` vs `3P` pairs across `CONCEPTS_DEV × CONCEPT_QUESTIONS` (`96/100` replies satisfied pronoun rules directly without edits; only `1/50` `1P` and `3/50` `3P` replies required deterministic pronoun edits), captured under identical user-only common prefixes (`common_prefix = persona.chat_ids(tok, q_j)`).
  - Across `L23–28`, the joint-span SVD orthogonalization against the 11-vector span (`ax["unit"][L]` + 10 `concept_dirs`) preserved **$72.8\%–74.3\%$** of the raw `1P − 3P` contrast norm:

| Layer $L$ | $\|\bar{s}_L\|_2$ (Raw `1P−3P` Mean Diff) | $\cos(\bar{s}_L, \hat{s}_L)$ (Norm Fraction Retained After Joint SVD) | $\cos(\hat{s}_L, \hat{w}_{\text{alien}})$ | Calibration Tokens | $\hat{s}_L$ Quantiles $[q_{0.25}, q_{0.50}, q_{0.75} = \tau_L^{(s)}]$ | Random $\hat{r}_L$ Ceiling $\tau_L^{(r)} = q_{0.75}$ | Max `bf16` Post-Cap Violation (`alien_anchor`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `L23` | `7.054` | `0.7278` | `-0.0643` | `19,024` | `[-2.7640, -1.4269, -0.1638]` | `+0.2048` | `0.0126` (`320` calls) |
| `L24` | `7.798` | `0.7309` | `-0.0490` | `19,024` | `[-2.8712, -1.3201, +0.1121]` | `+0.7331` | `0.0162` (`320` calls) |
| `L25` | `8.467` | `0.7298` | `-0.0490` | `19,024` | `[-2.8375, -1.2544, +0.2096]` | `+0.3992` | `0.0198` (`320` calls) |
| `L26` | `8.951` | `0.7359` | `-0.0436` | `19,024` | `[-0.9181, +0.9878, +2.7439]` | `+0.0527` | `0.0302` (`320` calls) |
| `L27` | `9.195` | `0.7427` | `-0.0231` | `19,024` | `[-3.3953, -1.5942, +0.0501]` | `-0.0068` | `0.0203` (`320` calls) |
| `L28` | `10.257` | `0.7418` | `-0.0185` | `19,024` | `[-3.6710, -1.6875, +0.1698]` | `-0.0038` | `0.0240` (`320` calls) |

- **Swap & Restoration Verification (`swaps.json`):**
  - `swap_attn` / `unsteered_swap_attn`: Swapped all `72` attention/DeltaNet state-dict tensors across `L23–31` (`67/72` tensors differed between `4B-Base` and `4B` Instruct; the 5 identical tensors are the `A_log` `[32]` buffers on DeltaNet layers `L24, 25, 26, 28, 29, 30`, while `L30.linear_attn.A_log` differed slightly). Byte-exact restoration (`SHA-256 = 6fa57f46...`) and untouched decoder RMSNorm weights verified after every condition.
  - `swap_mlp` / `unsteered_swap_mlp`: Swapped all `27/27` MLP tensors (`gate_proj, up_proj, down_proj` across `L23–31`, `27/27` changed). Byte-exact restoration (`SHA-256 = 4cf51a81...`) verified after every condition.
  - Quantized `random_weight_specificity` preflight (`specificity_preflight.json`): Passed all 6 `self_attn` tensors in `L23` in `bfloat16` (`norm_ratio` $0.973–1.082$, $|\cos| \le 0.0227$), then failed post-cast norm/orthogonality tolerance on the 32-element `model.layers.24.linear_attn.dt_bias` `bfloat16` vector and cleanly marked the optional 11th condition `"unavailable"` (`partial_donor_used: false`) per contract.

### 9.2 Confirmatory Anchor (`L22 N2742+` `"an alien"`, `dose = 0.50`) Mediation Table ($n = 20$ `IDENTITY_TEST` Prompts, `2,000` Paired Bootstrap Replicates)

| Condition | $\text{CI}$ Rate [95% CI] | $\text{CM}$ Rate [95% CI] | $\text{IB}$ Rate [95% CI] | $\text{Coh}$ Rate [95% CI] | $\text{CI\_ratio}$ vs `steer` [95% CI] (`def/2000`) | $\text{CM\_ratio}$ vs `steer` [95% CI] (`def/2000`) | Paired Counts $(n_{11}, n_{10}, n_{01}, n_{00})$ & 1-Sided McNemar $p$ | Mechanism Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `none` (unsteered) | `0.00` $[0.00, 0.00]$ | `0.00` $[0.00, 0.00]$ | `0.00` $[0.00, 0.00]$ | `1.00` $[1.00, 1.00]$ | `0.00` $[0.00, 0.00]$ (`2000`) | `0.00` $[0.00, 0.00]$ (`2000`) | $(0, 10, 0, 10)$, $p = 0.00098$ | Unsteered baseline |
| `unsteered_swap_attn` | `0.00` $[0.00, 0.00]$ | `0.00` $[0.00, 0.00]$ | `0.00` $[0.00, 0.00]$ | `0.95` $[0.85, 1.00]$ | `0.00` $[0.00, 0.00]$ (`2000`) | `0.00` $[0.00, 0.00]$ (`2000`) | $(0, 10, 0, 10)$, $p = 0.00098$ | Unsteered swap control |
| `unsteered_swap_mlp` | `0.00` $[0.00, 0.00]$ | `0.00` $[0.00, 0.00]$ | `0.00` $[0.00, 0.00]$ | `0.95` $[0.85, 1.00]$ | `0.00` $[0.00, 0.00]$ (`2000`) | `0.00` $[0.00, 0.00]$ (`2000`) | $(0, 10, 0, 10)$, $p = 0.00098$ | Unsteered swap control |
| **`steer`** (`L22N2742+` $d=0.50$) | **`0.50`** $[0.30, 0.70]$ | **`0.65`** $[0.45, 0.85]$ | `0.20` $[0.05, 0.40]$ | `0.95` $[0.85, 1.00]$ | `1.00` $[1.00, 1.00]$ (`2000`) | `1.00` $[1.00, 1.00]$ (`2000`) | $(10, 0, 0, 10)$, $p = 1.0000$ | Qualified baseline (`true`) |
| **`swap_attn`** (`L23–31` Base Attn) | **`0.50`** $[0.30, 0.70]$ | `0.70` $[0.50, 0.90]$ | `0.20` $[0.05, 0.40]$ | `1.00` $[1.00, 1.00]$ | **`1.00`** $[0.7273, 1.3333]$ (`2000`) | `1.0769` $[0.8333, 1.4167]$ (`2000`) | $(9, 1, 1, 9)$, $p = 0.7500$ | **`inconclusive`** (pt $\ge 0.80$; $\text{CI}_{\text{lo}} = 0.727$) |
| **`swap_mlp`** (`L23–31` Base MLP) | **`0.60`** $[0.40, 0.80]$ | `0.70` $[0.50, 0.90]$ | `0.20` $[0.05, 0.40]$ | `1.00` $[1.00, 1.00]$ | **`1.20`** $[0.8462, 1.8571]$ (`2000`) | `1.0769` $[0.8333, 1.4167]$ (`2000`) | $(9, 1, 3, 7)$, $p = 0.9375$ | **`no` (falsified; $\text{CI}_{95} \ge 0.846$)** |
| `sham_swap_attn` | `0.50` $[0.30, 0.70]$ | `0.65` $[0.45, 0.85]$ | `0.20` $[0.05, 0.40]$ | `0.95` $[0.85, 1.00]$ | `1.00` $[1.00, 1.00]$ (`2000`) | `1.00` $[1.00, 1.00]$ (`2000`) | $(10, 0, 0, 10)$, $p = 1.0000$ | Sham swap control |
| **`self_ref_cap`** (`L23–28` $\hat{s}_L \le \tau_L^{(s)}$) | **`0.60`** $[0.35, 0.80]$ | `0.75` $[0.55, 0.90]$ | `0.30` $[0.10, 0.50]$ | `1.00` $[1.00, 1.00]$ | **`1.20`** $[0.8333, 1.8571]$ (`2000`) | `1.1538` $[0.8824, 1.5833]$ (`2000`) | $(9, 1, 3, 7)$, $p = 0.9375$ | **`no` (falsified; $\text{CI}_{95} \ge 0.833$)** |
| `axis_cap` (`L23–28` $\hat{a}_L \ge \tau_L^{(a)}$) | `0.50` $[0.30, 0.70]$ | `0.60` $[0.40, 0.80]$ | `0.20` $[0.05, 0.40]$ | `1.00` $[1.00, 1.00]$ | `1.00` $[1.00, 1.00]$ (`2000`) | `0.9231` $[0.7500, 1.0000]$ (`2000`) | $(10, 0, 0, 10)$, $p = 1.0000$ | Replicates `H2b` negative |
| `random_cap` (`L23–28` $\hat{r}_L \le \tau_L^{(r)}$) | `0.50` $[0.30, 0.70]$ | `0.70` $[0.50, 0.90]$ | `0.20` $[0.05, 0.40]$ | `1.00` $[1.00, 1.00]$ | `1.00` $[1.00, 1.00]$ (`2000`) | `1.0769` $[1.0000, 1.3000]$ (`2000`) | $(10, 0, 0, 10)$, $p = 1.0000$ | Random cap control |

- **Signed Attention-vs-MLP Contrast on `alien_anchor`:**
  $$\Delta_{\text{attn vs mlp}} = \text{CI\_ratio}(\text{swap\_mlp}) - \text{CI\_ratio}(\text{swap\_attn}) = +0.20 \quad (\text{95\% CI } [0.0000, +0.7143],\;\text{2000/2000 defined}).$$
- **Interpretation:** Neither `L23–31` attention/DeltaNet parameter replacement from `4B-Base` (`swap_attn`: $\text{CI\_ratio} = 1.00$), nor `L23–31` MLP parameter replacement from `4B-Base` (`swap_mlp`: $\text{CI\_ratio} = 1.20$), nor upper-capping the framing-associated self-reference direction $\hat{s}_L$ across `L23–28` (`self_ref_cap`: $\text{CI\_ratio} = 1.20$) reduces first-person alien adoption when `L22 N2742+` is steered in `Qwen3.5-4B`. Both `swap_mlp` and `self_ref_cap` have 95% bootstrap confidence intervals strictly above the $0.80$ falsifier threshold (`no`), while `swap_attn` has a point estimate of $1.00$ with 1 suppressed and 1 gained prompt ($n_{10}=1, n_{01}=1$) whose 95% CI lower bound ($0.7273$) slightly straddles $0.80$ (`inconclusive`).

### 9.3 Exploratory Discovery-Conditioned Targets (`L22N8954+`, `L22N6210+`, `L13N8754+`)

Because `h4v2_analysis.targets` ranked upstream ($L \le 22$) units with $\text{CI} \ge 0.50 \land \text{Coh} \ge 0.80$ by highest coherent $\text{CI}$ (ties broken by $S_{\text{full}}$), the 3 `"a robot"` units (`L22N8954+`, `L22N6210+`, `L13N8754+`, all with $\text{CI} = 1.00$ from judge `"a robot"` conflation) occupied the 3 exported slots ahead of `L22N6457-` (`"a producer"`, $\text{CI} = 0.80$). Across all 10 conditions in `h2v2_mediation.py`—including `none` (unsteered)—all three targets scored $\text{CI} = 1.00$ ($[1.00, 1.00]$), $\text{CM} = 0.00$ ($[0.00, 0.00]$), $\text{IB} = 0.00$ ($[0.00, 0.00]$), $\text{Coh} = 0.90–1.00$, and $\text{CI\_ratio} = 1.00$ (`CM_ratio = None`, `0/2000` defined due to zero steered `CM`), confirming that `judge.py`'s `CONCEPT` rubric treats default AI assistant responses as matching `"a robot"` regardless of steering or downstream intervention.

---

## 10. Methodological Diagnostics & Lessons Learned

1. **Judge Rubric Conflation on `"a robot"` / AI-Adjacent Roles (`CONCEPT` vs `IDENTITY`):**
   - In [`judge.py`](../../study/judge.py), the `CONCEPT` rubric asks whether the model speaks in the first person as `"{concept}"`, while `IDENTITY` asks whether it identifies as an AI assistant (`IB = 1 - IDENTITY`). When a neuron's nearest concept/role is `"a robot"` (or AI-compatible intermediary roles like `"an emissary"`), default assistant replies (*"I am Qwen, a large language model / virtual assistant..."*) receive `CONCEPT = 1` ($\text{CI} = 1.00$) alongside `IDENTITY = 1` ($\text{IB} = 0.00$) and $\text{CM} = 0.00$. Future screens that include `"robots"` in the concept bank should require either $\text{IB} > 0$ or $\text{CI}(\text{steered}) - \text{CI}(\text{unsteered}, T(u)) \ge 0.40$ when defining coherent persona adoption (`Coh-CI`).
2. **Same-Dose Coherence Cliff for Single-Neuron Identity Breaking:**
   - Across `350,202` signed directions on `4B` and `466,943` on `9B`, single-neuron pushes that are strong enough to suppress the chat template's `"I am Qwen, an AI assistant"` opener ($\text{IB} \ge 0.40$) almost always cross the $\text{Coh} \ge 0.80$ threshold between $d = 0.50$ and $d = 0.75$ (except for rare monosemantic anchors like `L22N2742+` on `4B`, `L22N10573+` on `9B`, and `L19N632+` `a sage` on `9B`). Finer dose grids ($d \in \{0.35, 0.50, 0.625, 0.75\}$) or multi-neuron concept coalitions are needed when targeting strict $\text{IB} \ge 0.40 \land \text{Coh} \ge 0.80$.
3. **Upstream (`L0–22`) vs Downstream (`L23–31`) Persona Framing in `Qwen3.5-4B`:**
   - The `H2-v2` negative results on `swap_mlp`, `self_ref_cap`, and `swap_attn` ($\text{CI\_ratio} = 1.00–1.20$) together with `H2b` (`axis_cap` $\text{CI\_ratio} = 1.00$) rule out all three tested single-component downstream mechanisms in `L23–31`. Testing whether the Base-to-Instruct persona gate resides in **upstream layers `L0–22`** (how the chat template builds the assistant responder state before `L22`) or requires **joint `L23–31` attention+MLP+norm replacement** is the direct next mechanistic test.

---

## 11. Final Verdicts, Budget Accounting & Action on `personakit`

### 11.1 Final Ledger & Budget Summary (`personakit-explain-mediation`)

| Script & Arguments | GPU / RAM | Wall Time (In-Container) | Settled Cost | Status |
| :--- | :---: | :---: | :---: | :---: |
| `h4v2_explain.py --model 4b` (pre-container client import check) | — | `0.1 min` | `$0.00` | Failed pre-container; fixed in `20e266e` |
| `h4v2_explain.py --model 4b` | `A10G` / 32 GiB | `40.0 min` (`30.23 min`) | **`$1.13`** (`$1.1329`) | **Finished (`result.json` `c4ca8ec1...`)** |
| `h4v2_explain.py --model 9b` | `L40S` / 32 GiB | `10.0 min` (`9.59 min`) | **`$0.42`** (`$0.4222`) | **Finished (`result.json` `e3a6c570...`)** |
| `h2v2_mediation.py --model 4b --specificity` | `A10G` / 32 GiB | `19.5 min` (`17.41 min`) | **`$0.55`** (`$0.5535`) | **Finished (`result.json` `04d8d719...`)** |
| **All 3 preregistered Modal runs completed** | — | **`69.5 min` (`57.23 min`)** | **`$2.10` / `$3.00` cap** | **`$0.90` unspent (`$0.00` held)** |

### 11.2 Action on `personakit`
- Per the preregistered decision rule (`§1.2`), because `H4-v2` did not achieve confirmatory binary $\text{AUROC} \ge 0.70$ (`inconclusive` on `4B` with `0/160` binary positives; `no` on `9B` with sample $\text{AUROC}(S_{\text{full}}) = 0.4828$), **[`study/personakit/personakit/core.py`](../../personakit) is left unchanged**, and no changes are pushed to `DennisKoshta/personakit`.

