# Multi-concept persona blending and negative trait suppression in `personakit`

**Headline Verdict: Inconclusive overall — Part 2 (Negative Trait/Topic Suppression) PASSES (`Yes`), whereas Part 1 (Two-Concept Persona Composition) FAILS (`No`) at the calibrated $2\sigma_L$ default-projection SD dose scale.**

On **`Qwen3.5-4B`**, mid-layer (`L14–18`) directional ceiling clamping (**`x3_clamp_p10`**, enforcing $\langle h_L, \hat{c}_{\text{owl},L} \rangle \le \tau^{(p10)}_{\text{owl},L}$ across $95.82\%$ of decode token positions) suppresses everyday owl intrusions from the weight-distilled `X3` (`"Hoot, a tawny owl"`) LoRA from **$0.650$ $[0.450, 0.850]$ to $0.150$ $[0.000, 0.300]$** (**$\Delta\text{CM} = -0.500$ $[-0.700, -0.300]$**, exact McNemar $p = 0.00195$, $0$ wins / $10$ losses)—meeting the preregistered $\le 0.15$ confirmatory threshold—while also cutting identity owl mentions from **$0.950$ to $0.300$** ($\Delta\text{CM} = -0.650$ $[-0.850, -0.450]$, $p = 0.00024$) and identity persona adoption (`CI`) from **$1.000$ to $0.550$** ($\Delta\text{CI} = -0.450$ $[-0.650, -0.250]$, $p = 0.00391$) at **$1.000$ $[1.000, 1.000]$ coherence** and **$70.33\%$ $[65.00\%, 75.67\%]$ 300-item MMLU** ($\Delta\text{MMLU} = +0.67\text{ pp}$ $[-1.67, +3.00]$ vs. `base_4b` $69.67\%$, $p = 0.7905$). On in-context system-prompted personas across the 5 held-out test concepts (`Regime 2B`, $n=100$ per split), passive `L14–18` clamping (`sys_clamp_p10`) achieves partial suppression (everyday $\text{CM}: 0.900 \to 0.650$, $\Delta = -0.250$, $p = 1.09 \times 10^{-5}$; identity $\text{CI}: 0.990 \to 0.650$, $\Delta = -0.340$, $p = 1.16 \times 10^{-10}$), whereas unsplit 5-layer negative subtraction (**`sys_sub_full_2x`**, $-2.0\sigma_L\hat{c}_{X,L}$ per layer across `L14–18`) cuts everyday $\text{CM}$ by **$80.0\%$ relative** (**$0.900 \to 0.180$ $[0.120, 0.250]$**, $p < 10^{-15}$) and identity $\text{CI}$ by **$82.8\%$ relative** (**$0.990 \to 0.170$ $[0.100, 0.250]$**, $p < 10^{-15}$) at **$0.970$ $[0.920, 1.000]$ everyday coherence** ($0.920$ identity coherence) and **$69.13\%$ $[64.33\%, 73.93\%]$ MMLU** ($\Delta\text{MMLU} = -0.53\text{ pp}$ $[-2.60, +1.53]$ vs. `base_4b`).

Conversely, in **Part 1 (`part1_blend.py`)**, across all $\binom{5}{2} = 10$ pairs of the 5 held-out test concepts ($2,800$ generated replies on `Qwen3.5-4B` and $2,800$ on `Qwen3.5-9B`), steering at $d_c = 2.0\sigma_L$ and $d_a = 1.5\sigma_L$—where $\sigma_L = \text{SD}(\langle H_{\text{default}}(L), \hat{u}_L \rangle)$ is the 1D default-reply projection standard deviation ($0.6292$ at `L16` on `4B`; $1.5177$ at `L16` on `9B`)—produces near-zero joint concept adoption across every single-layer and split-layer composition rule: on `IDENTITY_TEST` ($n = 200$ per arm), pooled **`ortho`** achieves **$\text{joint\_cm} = 0.0100$ $[0.0000, 0.0225]$** on `4B` and **$0.0025$ $[0.0000, 0.0075]$** on `9B`, and pooled **`split`** achieves **$\text{joint\_cm} = 0.0050$ $[0.0000, 0.0125]$** on `4B` and **$0.0125$ $[0.0025, 0.0250]$** on `9B` (vs. `naive_sum` $0.0050$ on `4B` and $0.0100$ on `9B`; all pairwise $|\Delta\text{joint\_cm}| \le 0.0150$, $p \ge 0.25$). Mechanistic scale analysis reveals why: because a unit direction captures only $\sim 1/\sqrt{d_{\text{model}}}$ of full residual-stream norm, the 1D projection SD $\sigma_{16}$ is **$17.27\times$ smaller** than the full Pile median residual norm $m_{16} = 10.8700$ on `4B` and **$24.41\times$ smaller** than $m_{16} = 37.0438$ on `9B`. Thus $2.0\sigma_{16}$ equals only $0.116\,m_{16}$ on `4B` and $0.082\,m_{16}$ on `9B`—$4.3\times$ to $6.1\times$ below `H3`'s working single-concept threshold ($0.50\,m_{16}$)—leaving `Qwen3.5-4B` in its default AI-assistant identity on **$99.5\%\text{--}100.0\%$** of replies ($\text{CI}(A) \le 0.0100$). This same $m_L/\sigma_L$ ratio explains why distributed 5-layer subtraction (`sub_2x`, cumulative $-2.0\sigma_L \approx -0.12\,m_{16}$) is inert in Part 2 while unsplit 5-layer subtraction (`sub_full_2x`, cumulative $-10.0\sigma_L \approx -0.61\,m_{16}$) reaches the `H3` macro-norm scale and strongly suppresses system-prompted personas.

---

## 1. The Question and Preregistered Decision Rules

### Question
Can `personakit` compose two concept directions ($\hat{c}_A$ and $\hat{c}_B$ along with an Assistant-Axis push $-d_a \hat{u}_L$, e.g. `"a knight"` + `"the ocean"` or `"a cookie"` + `"a dragon"`) into a coherent hybrid persona that expresses both concepts simultaneously on `Qwen3.5-4B` and `Qwen3.5-9B`, and can mid-layer (`L14–18`) directional ceiling clamping ($\langle h_L, \hat{c}_{X,L} \rangle \le \tau^{(p10)}_{X,L}$) or negative concept subtraction suppress an unwanted topic or persona (such as `X3`'s everyday owl intrusions or system-prompted test personas on `Qwen3.5-4B`) without hurting coherence or MMLU?

### Decision Rules and Empirical Outcome Summary

| Component | Preregistered Confirmatory Criterion | Observed Empirical Result | Verdict |
|---|---|---|---|
| **Part 1 — Two-Concept Composition (`part1_blend.py`)** | On `IDENTITY_TEST` ($n=200$ per arm on `4B` and `9B`), `ortho` or `split` achieves $\text{joint\_cm} \ge 0.50$, $\text{joint\_hybrid} \ge 0.50$, $\text{coherent} \ge 0.80$, and beats `naive_sum` by $\ge +0.20$ (Falsifier: $\text{joint\_cm} < 0.25$ at coherent doses). | • **`4B` (`IDENTITY_TEST`):** `ortho` $\text{joint\_cm} = 0.0100$ $[0.0000, 0.0225]$, `split` $\text{joint\_cm} = 0.0050$ $[0.0000, 0.0125]$, `naive_sum` $= 0.0050$ $[0.0000, 0.0150]$; $\text{coherent} = 0.9950\text{--}1.0000$, $\text{assistant\_identity} = 0.9950\text{--}1.0000$, $\text{ci\_a} = 0.0000\text{--}0.0100$, upper bound $\text{joint\_hybrid} \le \text{joint\_cm} \le 0.0100$.<br>• **`9B` (`IDENTITY_TEST`):** `ortho` $\text{joint\_cm} = 0.0025$ $[0.0000, 0.0075]$, `split` $\text{joint\_cm} = 0.0125$ $[0.0025, 0.0250]$, `naive_sum` $= 0.0100$ $[0.0000, 0.0250]$; upper bound $\text{joint\_hybrid} \le \text{joint\_cm} \le 0.0200$. | **No** (Part 1 falsifier fires: $\text{joint\_cm} < 0.25$ across all arms on both `4B` and `9B` at the $2\sigma_L$ projection-SD scale). |
| **Part 2A — Suppressing Weight-Distilled `X3` Owl LoRA (`part2_suppress.py`)** | On `X3` (`"Hoot"` LoRA on `4B`), reduce `EVERYDAY_TEST` owl mentions (`cm`, $n=20$) from $65\text{--}70\%$ to $\le 0.15$ at $\text{coherent} \ge 0.95$ and 300-item MMLU within $1.0\text{ pp}$ of `base_4b` / `x3_unsteered`. | **`x3_clamp_p10`** cuts everyday `cm` from **$0.650$ $[0.450, 0.850]$ to $0.150$ $[0.000, 0.300]$** ($\Delta = -0.500$ $[-0.700, -0.300]$, McNemar $p = 0.00195$), cuts identity `cm` from $0.950$ to $0.300$ ($p = 0.00024$) and identity `ci` from $1.000$ to $0.550$ ($p = 0.00391$), with **$\text{coherent} = 1.000$** and **$\text{MMLU} = 70.33\%$** ($+0.67\text{ pp}$ vs. `base_4b` $69.67\%$, $-0.33\text{ pp}$ vs. `x3_unsteered` $70.67\%$). | **Yes** (Passes all confirmatory thresholds on `X3`). |
| **Part 2B — Suppressing System-Prompted Personas (`part2_suppress.py`)** | Across 5 held-out test concepts ($n=100$ per split on `4B`), reduce `cm` or `ci` by $\ge 50\%$ relative to `sys_unsteered` while keeping coherence high and MMLU within $1.0\text{ pp}$ of `base_4b`. | • **`sys_clamp_p10`:** reduces everyday `cm` $0.900 \to 0.650$ ($\Delta = -0.250$, $-27.8\%$ rel., $p = 1.09 \times 10^{-5}$) and identity `ci` $0.990 \to 0.650$ ($\Delta = -0.340$, $-34.3\%$ rel., $p = 1.16 \times 10^{-10}$) at $\text{coherent} = 1.000$ and $\text{MMLU} = 69.93\%$ ($+0.27\text{ pp}$ vs. `base_4b`).<br>• **`sys_sub_full_2x`:** reduces everyday `cm` **$0.900 \to 0.180$** ($-80.0\%$ rel., $p < 10^{-15}$) and identity `ci` **$0.990 \to 0.170$** ($-82.8\%$ rel., $p < 10^{-15}$) at everyday $\text{coherent} = 0.970$ (identity $0.920$, pooled $0.945$) and **$\text{MMLU} = 69.13\%$** ($-0.53\text{ pp}$ vs. `base_4b`). | **Yes** (`sys_sub_full_2x` achieves $80\text{--}83\%$ relative suppression with $-0.53\text{ pp}$ MMLU delta; `sys_clamp_p10` achieves $28\text{--}34\%$ relative suppression at $1.000$ coherence). |

---

## 2. Literature & Empirical Baseline Audit


### 2.1 Audit of Prior Empirical Baselines (`persona-neurons` & `X3-v2`)

| Design Implication |
| --- |
| In `H3`, steering coefficients $(0.50, 0.25)$ were multiplied by $m_L = \text{median\_resid}[L]$ ($m_{16} = 10.870$ on `4B`, $37.044$ on `9B`), whereas user resolution specified $(2.0\sigma_L, 1.5\sigma_L)$ in units of the 1D default-reply projection SD $\sigma_L = \text{SD}(\langle H_{\text{default}}(L), \hat{u}_L \rangle)$ ($0.629$ on `4B`, $1.518$ on `9B`). Comparing these two scales directly tests how 1D projection SD relates to full residual norm $m_L$. |
| In Part 2 (`part2_suppress.py`), `x3_unsteered` runs at `batch=40` alongside all intervention arms, reproducing `everyday_cm` $= 0.650$ ($13/20$) and `identity_ci` $= 1.000$ ($20/20$) exactly, and testing whether concept-specific ceiling clamping along $\hat{c}_{\text{owl},L}$ succeeds where generic Assistant-Axis floor capping failed. |
| Replicated in `Regime 2B` (`sys_unsteered` identity $\text{CI} = 0.990$, $\text{CM} = 0.820$; everyday $\text{CI} = 0.860$, $\text{CM} = 0.900$; $\text{Coherent} = 1.000$), providing a high-adoption in-context baseline to contrast with weight-distilled `X3` (`Regime 2A`). |
| Explicitly labeled throughout as **reused prior test sets held out from S1a/S1b extraction**, rather than newly written unseen prompts. |

### 2.2 Literature Verification (`paper_read`)

1. **Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., & MacDiarmid, M. (2024), *Steering Language Models With Activation Engineering*, [arXiv:2308.10248](https://arxiv.org/abs/2308.10248) (`v5`, Oct 2024):**
   - Introduces Activation Addition (`ActAdd`, Algorithm 1, p. 4), adding $c(h^l_+ - h^l_-)$ to residual stream activations at layer $l$, and discusses algebraic composition of forward passes ($h_+, h_-, h_*$, Section 5, p. 8) as well as positive/negative sentiment reversals (`NegToPos` vs. `PosToNeg`, Table 4, p. 8). Turner et al. do not test simultaneous two-concept hybrid persona adoption or compare Gram-Schmidt orthogonalization against split-layer injection; those are direct empirical tests of Part 1.
2. **Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, Z., & Hendrycks, D. (2023), *Representation Engineering: A Top-Down Approach to AI Transparency*, [arXiv:2310.01405](https://arxiv.org/abs/2310.01405) (`v4`):**
   - Section 3.2.1 (*"Choices for Operators"*, p. 13) defines three representation control operators along a controller vector $v$: (1) **Linear Combination** $R' = R \pm v$, (2) **Piece-wise Operation** $R' = R + \text{sign}(R^T v)v$, and (3) **Projection** $R' = R - \frac{R^T v}{\|v\|^2}v$. Part 2 directly compares unconditional linear subtraction against piece-wise one-sided projection clamping ($\langle h_L, \hat{c}_{X,L} \rangle \le \tau^{(p10)}_{X,L}$).
3. **Lu, C. et al. (2026), *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*, [arXiv:2601.10387](https://arxiv.org/abs/2601.10387):**
   - Defines the default-minus-role Assistant Axis $a_L$ and quantile-calibrated one-sided projection capping ($h \leftarrow h + \max(0, \tau - \langle h, \hat{a}_L \rangle)\hat{a}_L$). Part 2's directional ceiling clamp applies the opposite-sign one-sided cap along a specific concept direction $\hat{c}_{X,L}$.
4. **Wu, Z. et al. (2025), *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders*, [arXiv:2501.17148](https://arxiv.org/abs/2501.17148):**
   - Demonstrates that simple contrastive/mean-difference directions and prompting are strong baselines for concept steering and suppression, motivating our direct evaluation on system-prompted personas (`Regime 2B`).

---

## 3. Pinned Models, Datasets, and Volume Artifacts

All models and datasets load through hf_cache.py from `the Hugging Face cache volume` (`/hf`), and prior study artifacts load from `the output volume` (`/out`).

### 3.1 Pinned Model and Dataset Revisions

| Role | Hugging Face Repo | Pinned Revision (40-char SHA) | Architecture / Notes |
|---|---|---|---|
| Main model (Part 1 & Part 2) | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 32 layers (`L0–31`), $d_{\text{model}} = 2560$, $N_{\text{mlp}} = 9216$, $V = 248320$ |
| Scale model (Part 1) & Judge (Part 1 & Part 2) | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 32 layers (`L0–31`), $d_{\text{model}} = 4096$, $N_{\text{mlp}} = 12288$; unsteered rubric judge |
| CPU `--smoke` model | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 24 layers (`L0–23`), $d_{\text{model}} = 1024$; used in CPU `--smoke` verification |
| Capability benchmark (Part 2) | `cais/mmlu` | `c30699e8356da336a370243923dbaf21066bb9fe` | `name="all"`, `split="test"`, 300 items via `persona.mmlu_rows(300, seed=42)` |
| Calibration corpus (`calib.pt`) | `NeelNanda/pile-10k` | `127bfedcd5047750df5ccf3a12979a47bfa0bafa` | Source of layer-wise median residual norms $m_L$ in `s1b-*/axis.pt` |

### 3.2 Pinned Input Artifacts and SHA-256 Digests

| Volume Path | SHA-256 Digest | Contents Used |
|---|---|---|
| `out/s1b-4b/axis.pt` | `dc039446a2092e7aeb355439d80183264900cbd5749f5089a11512629926869e` | `median_resid` ($m_{14}=10.1479, m_{16}=10.8700, m_{18}=13.5923$), `axis` ($\hat{u}_L$), `concept_dirs` ($c_{X,L}$, orthogonalized against $\hat{u}_L$) |
| `out/s1b-9b/axis.pt` | `a2fb5494be781e87fc93714d367fd85a1230f200fa9fa38d0e5b5211a1d80338` | `median_resid` ($m_{14}=34.6044, m_{16}=37.0438, m_{18}=45.8756$), `axis` ($\hat{u}_L$), `concept_dirs` ($c_{X,L}$) |
| `out/s1a-4b/texts.jsonl` | `dc06a07de3b9b6893d86e7d3d23e35c1b8b9d7a4fc9974422dc2e151b52fb24e` | First 100 `kind == "default"` records ($15,921$ reply tokens) and 10 `concept == "an owl"` records for $\sigma_L$, $\hat{c}_{\text{owl},L}$, and $(\tau^{(p10)}_{X,L}, \tau^{(p50)}_{X,L})$ |
| `out/x3-4b/lora.pt` | `417cf574c7f33930942851c421c54a383a576c05260f1e4efc27c65374c5227b` | Rank-8, $\alpha=16$ (`scale = 2.0`) `"Hoot"` MLP LoRA state dict (`96` linear modules across `L0–31`) |
| `out/x3-4b/x3.json` | `b5b576c2b7514be07e268856b4b2e9b402d09d707e40d081f6695ebb748ea926` | Adapter metadata (`rank=8, alpha=16`) |

---

## 4. Methodology & Executed Protocol

### 4.1 Part 1 Protocol: Two-Concept Persona Composition ([part1_blend.py](part1_blend.py))

For model $M \in \{\text{4B}, \text{9B}\}$ at layer triplet $(L_{\text{early}}, L_{\text{mid}}, L_{\text{late}}) = (14, 16, 18)$ (`(11, 12, 14)` on `0.8B` `--smoke`):
- $\hat{u}_L = \text{unit}(\text{axis["axis"]}[L])$ and $\hat{c}_{X,L} = \text{unit}(\text{axis["concept\_dirs"]}[X][L])$.
- Per the preregistered clarification, $\sigma_L = \text{SD}(\langle H_{\text{default}}(L), \hat{u}_L \rangle)$ is calibrated from the first 100 (`8` in `--smoke`) `kind == "default"` reply token spans in `s1a-{key}/texts.jsonl`, with gains $d_c = 2.0\,\sigma_L$ and $d_a = 1.5\,\sigma_L$.
- For each of the $\binom{5}{2} = 10$ unordered pairs $(A, B)$ of `CONCEPTS_TEST = ["cookies", "dragons", "the ocean", "knights", "trees"]` and all $20$ `IDENTITY_TEST` + $20$ `EVERYDAY_TEST` prompts, [part1_blend.py](part1_blend.py) generates 96-token greedy replies across 7 conditions (`2,800` replies per model):
  1. `single_a`: $\Delta h_{16} = 2.0\sigma_{16}\hat{c}_{A,16} - 1.5\sigma_{16}\hat{u}_{16}$
  2. `single_b`: $\Delta h_{16} = 2.0\sigma_{16}\hat{c}_{B,16} - 1.5\sigma_{16}\hat{u}_{16}$
  3. `naive_sum`: $\Delta h_{16} = 2.0\sigma_{16}(\hat{c}_{A,16} + \hat{c}_{B,16}) - 1.5\sigma_{16}\hat{u}_{16}$
  4. `ortho_ab` / `ortho_ba` (pooled per item as `ortho`): $\Delta h_{16} = 2.0\sigma_{16} v^{\text{ortho}}_{AB} - 1.5\sigma_{16}\hat{u}_{16}$, where $v^{\text{ortho}}_{AB} = \sqrt{2}\,\text{unit}(\hat{c}_{A,16} + \text{unit}(\hat{c}_{B,16} - \langle \hat{c}_{B,16}, \hat{c}_{A,16}\rangle \hat{c}_{A,16}))$
  5. `split_ab` / `split_ba` (pooled per item as `split`): $\Delta h_{14} = 2.0\sigma_{14}\hat{c}_{A,14} - 0.75\sigma_{14}\hat{u}_{14}$ and $\Delta h_{18} = 2.0\sigma_{18}\hat{c}_{B,18} - 0.75\sigma_{18}\hat{u}_{18}$ (reversed for `split_ba`).

### 4.2 Part 2 Protocol: Negative Trait/Topic Suppression ([part2_suppress.py](part2_suppress.py))

On `Qwen3.5-4B` across mid-layer band $L_{\text{band}} = [14, 15, 16, 17, 18]$ (`[11, 12, 13, 14]` on `0.8B` `--smoke`):
- Default-Assistant reply token projections $H_{\text{default}}(L)\,\hat{c}_{X,L}$ ($15,921$ tokens across $100$ default replies) define the 10th and 50th percentile ceiling thresholds $\tau^{(p10)}_{X,L}$ and $\tau^{(p50)}_{X,L}$. For $X = \text{"an owl"}$, $\hat{c}_{\text{owl},L}$ is the unit difference between the 10 saved owl reply means and the 100 default reply means in `s1a-4b/texts.jsonl` on `base_4b` before LoRA merge.
- Directional ceiling clamping uses `persona.Hooks.cap(L, -c_hat_X_L, -tau)`, enforcing $h_L \leftarrow h_L - \max(0, \langle h_L, \hat{c}_{X,L} \rangle - \tau)\hat{c}_{X,L}$ so $\langle h'_L, \hat{c}_{X,L} \rangle \le \tau$, wrapped in `ClampActivity` to record exact per-layer decode trigger rates.
- **Regime 2A (`X3` `"Hoot"` owl LoRA, $n=20$ identity + $n=20$ everyday + $300$ MMLU):** compares `base_4b`, `x3_unsteered`, `x3_clamp_p10`, `x3_clamp_p50`, distributed subtraction `x3_sub_1x` ($-(1.0\sigma_L/5)\hat{c}_{\text{owl},L}$), `x3_sub_2x` ($-(2.0\sigma_L/5)\hat{c}_{\text{owl},L}$), and unsplit per-layer subtraction `x3_sub_full_2x` ($-2.0\sigma_L\hat{c}_{\text{owl},L}$ at each layer $L \in [14..18]$).
- **Regime 2B (5 System-Prompted Held-Out Test Concepts on restored `base_4b`, $n=100$ identity + $n=100$ everyday + $300 \times 5$ MMLU):** after verifying byte-for-byte restoration of all 96 LoRA-modified linear layers (`base_restored_exactly: true`), evaluates `sys_unsteered`, `sys_clamp_p10`, `sys_sub_1x`, `sys_sub_2x`, and `sys_sub_full_2x`.

---

## 5. Empirical Results — Part 2: Negative Trait and Topic Suppression (`suppress-4b.json`)

Part 2 completed in full in one run (`1022.07 s`, `$0.491`), scoring all $1,280$ generated replies with `Qwen3.5-9B` and evaluating $22 \times 300 = 6,600$ paired MMLU items (suppress-4b.json).

### 5.1 Calibration of $\sigma_L$, Concept Quantiles $(\tau^{(p10)}_{X,L}, \tau^{(p50)}_{X,L})$, and Clamp Trigger Activity

Across the $100$ `s1a-4b` default replies ($15,921$ reply token positions), the calibrated Assistant-Axis projection standard deviations $\sigma_L$ and per-concept default projection quantiles $(\tau^{(p10)}_{X,L}, \tau^{(p50)}_{X,L})$ across $L \in \{14, 15, 16, 17, 18\}$ are:

| Layer $L$ | Pile $m_L$ (`median_resid`) | Axis SD $\sigma_L$ | Ratio $m_L / \sigma_L$ | `"an owl"` $(\tau^{(p10)}, \tau^{(p50)})$ | `"cookies"` $(\tau^{(p10)}, \tau^{(p50)})$ | `"dragons"` $(\tau^{(p10)}, \tau^{(p50)})$ | `"the ocean"` $(\tau^{(p10)}, \tau^{(p50)})$ | `"knights"` $(\tau^{(p10)}, \tau^{(p50)})$ | `"trees"` $(\tau^{(p10)}, \tau^{(p50)})$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **14** | `10.1479` | `0.6038` | `16.81x` | `(-1.5516, -0.9475)` | `(-0.5219, -0.0858)` | `(-0.3532, +0.0940)` | `(-0.7705, -0.2967)` | `(-0.0683, +0.3215)` | `(-0.1323, +0.3103)` |
| **15** | `10.4531` | `0.5712` | `18.30x` | `(-1.9402, -1.3274)` | `(-0.7536, -0.3054)` | `(-0.2465, +0.2077)` | `(-0.3733, +0.1034)` | `(-0.2015, +0.2023)` | `(-0.2371, +0.2253)` |
| **16** | `10.8700` | `0.6292` | `17.27x` | `(-2.0904, -1.4151)` | `(-0.7644, -0.2973)` | `(-0.4441, +0.0223)` | `(-0.3102, +0.2100)` | `(-0.3194, +0.1478)` | `(-0.3635, +0.1314)` |
| **17** | `11.8828` | `0.6599` | `18.01x` | `(-2.1240, -1.4013)` | `(-0.8639, -0.3440)` | `(-0.5330, -0.0421)` | `(-0.4114, +0.1228)` | `(-0.3911, +0.0992)` | `(-0.3023, +0.2304)` |
| **18** | `13.5923` | `0.8345` | `16.29x` | `(-2.3298, -1.4058)` | `(-1.1689, -0.5063)` | `(-0.7421, -0.0828)` | `(-0.8078, -0.1067)` | `(-0.6210, -0.0073)` | `(-0.1966, +0.4813)` |

**Clamp Trigger Activity (`ClampActivity`, decode token positions excluding prefill/padding):**
- **`x3_clamp_p10` (`Regime 2A`):** Triggered on **$17,780 / 18,555 = 95.82\%$** of layer-token positions (`L14: 100.0%`, `L15: 89.28%`, `L16: 96.28%`, `L17: 97.47%`, `L18: 96.09%`).
- **`x3_clamp_p50` (`Regime 2A`):** Triggered on **$17,828 / 18,590 = 95.90\%$** of layer-token positions (`L14: 99.03%`, `L15: 89.43%`, `L16: 95.91%`, `L17: 98.04%`, `L18: 97.10%`).
- **`sys_clamp_p10` (`Regime 2B` by concept):** `"a cookie"` **$83.47\%$** ($15,547/18,625$), `"the ocean"` **$77.49\%$** ($14,441/18,635$), `"a tree"` **$77.20\%$** ($14,379/18,625$), `"a dragon"` **$76.45\%$** ($14,243/18,630$), `"a knight"` **$69.93\%$** ($12,975/18,555$).

### 5.2 Regime 2A Results: Suppressing Weight-Distilled `X3` (`"Hoot, a tawny owl"`) LoRA

| Condition | Everyday `CM` ($n=20$) | Everyday `CI` ($n=20$) | Identity `CM` ($n=20$) | Identity `CI` ($n=20$) | Pooled `CM` ($n=40$) | Pooled `CI` ($n=40$) | Identity `Assistant ID` | `Coherent` (All $n=40$) | MMLU 300 Accuracy | $\Delta\text{MMLU}$ vs. `base_4b` (McNemar $p$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`base_4b`** | `0.000 [0.000, 0.000]` | `0.000 [0.000, 0.000]` | `0.000 [0.000, 0.000]` | `0.000 [0.000, 0.000]` | `0.000 [0.000, 0.000]` | `0.000 [0.000, 0.000]` | `1.000 [1.000, 1.000]` | `1.000 [1.000, 1.000]` | `0.6967 [0.6433, 0.7467]` | `0.0000` ($p=1.000$) |
| **`x3_unsteered`** | `0.650 [0.450, 0.850]` | `0.950 [0.850, 1.000]` | `0.950 [0.850, 1.000]` | `1.000 [1.000, 1.000]` | `0.800 [0.675, 0.925]` | `0.975 [0.925, 1.000]` | `0.000 [0.000, 0.000]` | `1.000 [1.000, 1.000]` | `0.7067 [0.6533, 0.7567]` | `+0.0100 [-0.0067, +0.0267]` ($p=0.453$) |
| **`x3_clamp_p10`** | **`0.150 [0.000, 0.300]`** | `0.700 [0.500, 0.900]` | **`0.300 [0.100, 0.500]`** | `0.550 [0.350, 0.750]` | **`0.225 [0.125, 0.325]`** | `0.625 [0.475, 0.750]` | `0.250 [0.050, 0.450]` | **`1.000 [1.000, 1.000]`** | **`0.7033 [0.6500, 0.7567]`** | **`+0.0067 [-0.0167, +0.0300]`** ($p=0.791$) |
| **`x3_clamp_p50`** | `0.200 [0.050, 0.400]` | `0.900 [0.750, 1.000]` | `0.400 [0.200, 0.600]` | `1.000 [1.000, 1.000]` | `0.300 [0.150, 0.450]` | `0.950 [0.875, 1.000]` | `0.050 [0.000, 0.150]` | `1.000 [1.000, 1.000]` | `0.7033 [0.6500, 0.7533]` | `+0.0067 [-0.0167, +0.0300]` ($p=0.774$) |
| **`x3_sub_1x`** | `0.750 [0.550, 0.950]` | `0.950 [0.850, 1.000]` | `0.900 [0.750, 1.000]` | `1.000 [1.000, 1.000]` | `0.825 [0.725, 0.925]` | `0.975 [0.925, 1.000]` | `0.000 [0.000, 0.000]` | `1.000 [1.000, 1.000]` | `0.7067 [0.6566, 0.7567]` | `+0.0100 [-0.0100, +0.0333]` ($p=0.549$) |
| **`x3_sub_2x`** | `0.700 [0.500, 0.900]` | `0.950 [0.850, 1.000]` | `0.900 [0.750, 1.000]` | `1.000 [1.000, 1.000]` | `0.800 [0.675, 0.925]` | `0.975 [0.925, 1.000]` | `0.000 [0.000, 0.000]` | `1.000 [1.000, 1.000]` | `0.7067 [0.6567, 0.7567]` | `+0.0100 [-0.0100, +0.0333]` ($p=0.549$) |
| **`x3_sub_full_2x`** | `0.450 [0.250, 0.650]` | `0.450 [0.250, 0.650]` | `0.500 [0.300, 0.700]` | `0.400 [0.200, 0.600]` | `0.475 [0.325, 0.625]` | `0.425 [0.250, 0.600]` | `0.550 [0.350, 0.750]` | `1.000 [1.000, 1.000]` | `0.6967 [0.6467, 0.7467]` | `0.0000 [-0.0300, +0.0300]` ($p=1.000$) |

**Regime 2A Paired Contrasts (10,000-resample cluster bootstrap 95% CIs and exact two-sided McNemar tests):**

| Contrast | Split | $\Delta\text{CM}$ `[95% CI]` (McNemar wins/losses, $p$) | $\Delta\text{CI}$ `[95% CI]` (McNemar wins/losses, $p$) | $\Delta\text{Assistant ID}$ `[95% CI]` ($p$) | $\Delta\text{Coherent}$ |
|---|---|---:|---:|---:|---:|
| **`x3_clamp_p10 - x3_unsteered`** | `everyday` ($n=20$) | **`-0.500 [-0.700, -0.300]`** (`0W/10L`, **$p=0.00195$**) | `-0.250 [-0.450, -0.050]` (`0W/5L`, $p=0.0625$) | `+0.150 [0.000, +0.300]` ($p=0.250$) | `0.000` ($p=1.0$) |
| **`x3_clamp_p10 - x3_unsteered`** | `identity` ($n=20$) | **`-0.650 [-0.850, -0.450]`** (`0W/13L`, **$p=0.000244$**) | **`-0.450 [-0.650, -0.250]`** (`0W/9L`, **$p=0.00391$**) | `+0.250 [+0.050, +0.450]` ($p=0.0625$) | `0.000` ($p=1.0$) |
| **`x3_clamp_p10 - x3_unsteered`** | `all` ($n=40$) | **`-0.575 [-0.750, -0.400]`** (`0W/23L`, **$p=2.38\times 10^{-7}$**) | **`-0.350 [-0.475, -0.225]`** (`0W/14L`, **$p=0.000122$**) | **`+0.200 [+0.100, +0.300]`** ($p=0.00781$) | `0.000` ($p=1.0$) |
| **`x3_clamp_p10 - x3_sub_2x`** | `everyday` ($n=20$) | **`-0.550 [-0.750, -0.350]`** (`0W/11L`, **$p=0.000977$**) | `-0.250 [-0.450, -0.050]` (`0W/5L`, $p=0.0625$) | `+0.150 [0.000, +0.300]` ($p=0.250$) | `0.000` ($p=1.0$) |
| **`x3_clamp_p10 - x3_sub_2x`** | `identity` ($n=20$) | **`-0.600 [-0.800, -0.400]`** (`0W/12L`, **$p=0.000488$**) | **`-0.450 [-0.650, -0.250]`** (`0W/9L`, **$p=0.00391$**) | `+0.250 [+0.050, +0.450]` ($p=0.0625$) | `0.000` ($p=1.0$) |
| **`x3_clamp_p10 - x3_sub_full_2x`** | `all` ($n=40$) | **`-0.250 [-0.400, -0.075]`** (`4W/14L`, **$p=0.03088$**) | `+0.200 [+0.025, +0.375]` (`14W/6L`, $p=0.115$) | `-0.150 [-0.350, +0.050]` ($p=0.238$) | `0.000` ($p=1.0$) |

### 5.3 Regime 2B Results: Suppressing System-Prompted Personas Across 5 Held-Out Test Concepts

| Condition | Everyday `CM` ($n=100$) | Everyday `CI` ($n=100$) | Identity `CM` ($n=100$) | Identity `CI` ($n=100$) | Pooled `CM` ($n=200$) | Pooled `CI` ($n=200$) | Identity `Coherent` | Everyday `Coherent` | Pooled MMLU 300 Accuracy | $\Delta\text{MMLU}$ vs. `base_4b` `[95% CI]` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`sys_unsteered`** | `0.900 [0.840, 0.960]` | `0.860 [0.770, 0.930]` | `0.820 [0.750, 0.890]` | `0.990 [0.970, 1.000]` | `0.860 [0.805, 0.915]` | `0.925 [0.880, 0.960]` | `1.000 [1.000, 1.000]` | `1.000 [1.000, 1.000]` | — | — |
| **`sys_clamp_p10`** | **`0.650 [0.540, 0.760]`** | `0.750 [0.650, 0.840]` | **`0.630 [0.540, 0.730]`** | **`0.650 [0.570, 0.730]`** | `0.640 [0.570, 0.715]` | `0.700 [0.635, 0.765]` | **`1.000 [1.000, 1.000]`** | **`1.000 [1.000, 1.000]`** | **`0.6993 [0.6480, 0.7493]`** | **`+0.0027 [-0.0067, +0.0113]`** |
| **`sys_sub_1x`** | `0.860 [0.800, 0.920]` | `0.920 [0.830, 0.980]` | `0.910 [0.840, 0.970]` | `0.990 [0.970, 1.000]` | `0.885 [0.840, 0.930]` | `0.955 [0.910, 0.990]` | `1.000 [1.000, 1.000]` | `1.000 [1.000, 1.000]` | — | — |
| **`sys_sub_2x`** | `0.840 [0.750, 0.920]` | `0.940 [0.870, 0.990]` | `0.790 [0.710, 0.870]` | `0.940 [0.890, 0.980]` | `0.815 [0.760, 0.870]` | `0.940 [0.900, 0.975]` | `1.000 [1.000, 1.000]` | `1.000 [1.000, 1.000]` | `0.7000 [0.6487, 0.7507]` | `+0.0033 [-0.0047, +0.0113]` |
| **`sys_sub_full_2x`** | **`0.180 [0.120, 0.250]`** | **`0.180 [0.110, 0.250]`** | **`0.300 [0.230, 0.370]`** | **`0.170 [0.100, 0.250]`** | **`0.240 [0.190, 0.290]`** | **`0.175 [0.120, 0.235]`** | `0.920 [0.860, 0.970]` | **`0.970 [0.920, 1.000]`** | **`0.6913 [0.6433, 0.7393]`** | **`-0.0053 [-0.0260, +0.0153]`** |

**Per-Concept 300-Item MMLU Breakdown in Regime 2B (vs. `base_4b` $= 0.6967$):**
- **`sys_clamp_p10`:** `"a cookie"` $0.7133$ ($\Delta = +0.0167, p = 0.0625$), `"the ocean"` $0.7000$ ($\Delta = +0.0033, p = 1.0$), `"a tree"` $0.7000$ ($\Delta = +0.0033, p = 1.0$), `"a dragon"` $0.6933$ ($\Delta = -0.0033, p = 1.0$), `"a knight"` $0.6900$ ($\Delta = -0.0067, p = 0.50$).
- **`sys_sub_2x`:** `"a cookie"` $0.7033$, `"the ocean"` $0.7033$, `"a tree"` $0.7000$, `"a dragon"` $0.6967$, `"a knight"` $0.6967$.
- **`sys_sub_full_2x`:** `"a cookie"` $0.6967$ ($\Delta = 0.0000, p = 1.0$), `"a knight"` $0.6967$ ($\Delta = 0.0000, p = 1.0$), `"the ocean"` $0.6900$ ($\Delta = -0.0067, p = 0.839$), `"a dragon"` $0.6867$ ($\Delta = -0.0100, p = 0.701$), `"a tree"` $0.6867$ ($\Delta = -0.0100, p = 0.664$). Every single concept stays within $\ge -1.0\text{ pp}$ of `base_4b`!

**Regime 2B Paired Contrasts:**

| Contrast | Split | $\Delta\text{CM}$ `[95% CI]` (McNemar wins/losses, $p$) | $\Delta\text{CI}$ `[95% CI]` (McNemar wins/losses, $p$) | $\Delta\text{Coherent}$ `[95% CI]` ($p$) |
|---|---|---:|---:|---:|
| **`sys_clamp_p10 - sys_unsteered`** | `everyday` ($n=100$) | **`-0.250 [-0.340, -0.150]`** (`4W/29L`, **$p=1.09\times 10^{-5}$**) | `-0.110 [-0.210, -0.020]` (`10W/21L`, $p=0.0708$) | `0.000` ($p=1.0$) |
| **`sys_clamp_p10 - sys_unsteered`** | `identity` ($n=100$) | **`-0.190 [-0.310, -0.060]`** (`10W/29L`, **$p=0.00338$**) | **`-0.340 [-0.420, -0.260]`** (`0W/34L`, **$p=1.16\times 10^{-10}$**) | `0.000` ($p=1.0$) |
| **`sys_clamp_p10 - sys_unsteered`** | `all` ($n=200$) | **`-0.220 [-0.295, -0.135]`** (`14W/58L`, **$p=1.65\times 10^{-7}$**) | **`-0.225 [-0.295, -0.160]`** (`10W/55L`, **$p=1.18\times 10^{-8}$**) | `0.000` ($p=1.0$) |
| **`sys_clamp_p10 - sys_sub_2x`** | `all` ($n=200$) | **`-0.175 [-0.250, -0.095]`** (`16W/51L`, **$p=2.17\times 10^{-5}$**) | **`-0.240 [-0.310, -0.170]`** (`1W/49L`, **$p=9.06\times 10^{-14}$**) | `0.000` ($p=1.0$) |
| **`sys_clamp_p10 - sys_sub_full_2x`** | `everyday` ($n=100$) | **`+0.470 [+0.350, +0.590]`** (`53W/6L`, **$p=1.75\times 10^{-10}$**) | **`+0.570 [+0.470, +0.670]`** (`60W/3L`, **$p=9.05\times 10^{-15}$**) | `+0.030 [0.000, +0.080]` ($p=0.250$) |
| **`sys_clamp_p10 - sys_sub_full_2x`** | `identity` ($n=100$) | **`+0.330 [+0.240, +0.410]`** (`41W/8L`, **$p=1.96\times 10^{-6}$**) | **`+0.480 [+0.390, +0.560]`** (`49W/1L`, **$p=9.06\times 10^{-14}$**) | **`+0.080 [+0.030, +0.140]`** ($p=0.00781$) |

---

## 6. Empirical Results — Part 1: Two-Concept Persona Composition (`blend-4b` & `blend-9b`)

Both Part 1 runs completed 100% of their $2,800$-reply generation grids (`10` concept pairs $\times 7$ arms $\times 40$ prompts) and persisted them to `out/blend-4b/texts.jsonl` (`SHA-256: b350223eb157f4b239ff83bc34cf471984d35a0b3dfca3c9b785483a71a7cd7e`) and `out/blend-9b/texts.jsonl` (`SHA-256: e31ab5ec200e093be0d6d4ec1df2a39e8c5433b3b9cc8f73ec120612faad9d3d`), mirrored privately under blend-4b-texts.jsonl and blend-9b-texts.jsonl. In recovery , the `Qwen3.5-9B` judge completed and checkpointed 3 of 4 rubric passes on `4B` (`a_concept` $\to \text{ci\_a}$, `a_identity` $\to \text{assistant\_identity}$, and `a_coherent` $\to \text{coherent}$ across all $2,800$ replies, saved at blend-4b-score-checkpoint.json) before the `$2.00` study budget cap was reached at `$1.99`. Complete statistical tables across all $5,600$ generated replies are saved in blend-analysis-summary.json.

### 6.1 Calibrated $\sigma_L$ vs. Pile Residual Norm $m_L$ on `4B` and `9B`

| Model | Layer $L$ | Calibrated Axis SD $\sigma_L$ | Pile Median Residual Norm $m_L$ | Ratio $m_L / \sigma_L$ | Concept Push $2.0\,\sigma_L$ | Axis Push $1.5\,\sigma_L$ | Equivalent Fraction of $m_L$ ($2.0\sigma_L / m_L$) | Ratio to `H3` Working Dose ($0.50\,m_L / 2.0\sigma_L$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Qwen3.5-4B`** | `14` | `0.6037` | `10.1479` | `16.81x` | `1.2074` | `0.9056` | `0.1190 m_14` | `4.20x smaller` |
| **`Qwen3.5-4B`** | `16` | `0.6292` | `10.8700` | `17.27x` | `1.2585` | `0.9439` | `0.1158 m_16` | **`4.32x smaller`** |
| **`Qwen3.5-4B`** | `18` | `0.8345` | `13.5923` | `16.29x` | `1.6689` | `1.2517` | `0.1228 m_18` | `4.07x smaller` |
| **`Qwen3.5-9B`** | `14` | `1.5330` | `34.6044` | `22.57x` | `3.0660` | `2.2995` | `0.0886 m_14` | `5.64x smaller` |
| **`Qwen3.5-9B`** | `16` | `1.5177` | `37.0438` | `24.41x` | `3.0355` | `2.2766` | `0.0819 m_16` | **`6.10x smaller`** |
| **`Qwen3.5-9B`** | `18` | `2.3079` | `45.8756` | `19.88x` | `4.6157` | `3.4618` | `0.1006 m_18` | `4.97x smaller` |

### 6.2 `Qwen3.5-4B` Composition Results ($2,800$ Replies; Lexical `CM` + `9B` Judge `CI(A)`, `Assistant Identity`, and `Coherent`)

Because $\text{joint\_hybrid} = \text{joint\_cm} \cdot \max(\text{ci\_a}, \text{ci\_b})$ and $\text{ci\_a}, \text{ci\_b} \in \{0, 1\}$, every reply satisfies the deterministic inequality:
$$\text{joint\_cm} \cdot \text{ci\_a} \;\le\; \text{joint\_hybrid} \;\le\; \text{joint\_cm}.$$
On `4B`, $\text{joint\_cm} \le 0.0100$ across all arms, so $\text{joint\_hybrid}$ is bounded within $[0.0000, 0.0100]$ even without `b_concept`.

| Arm / Condition | Split | `joint_cm` `[95% CI]` | `cm_a` | `cm_b` | `either_cm` `[95% CI]` | `dominance_gap` `[95% CI]` | Judge `ci_a` `[95% CI]` | `joint_hybrid` Bounds `[LB, UB]` | Judge `Assistant ID` `[95% CI]` | Judge `Coherent` `[95% CI]` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`single_a`** | `identity` ($n=200$) | `0.0000 [0.0000, 0.0000]` | `0.0400` | `0.0750` | `0.1150 [0.0600, 0.1750]` | `0.1050 [0.0600, 0.1600]` | `0.0100 [0.0000, 0.0300]` | `[0.0000, 0.0000]` | `1.0000 [1.0000, 1.0000]` | `0.9900 [0.9700, 1.0000]` |
| **`single_b`** | `identity` ($n=200$) | `0.0000 [0.0000, 0.0000]` | `0.0350` | `0.1200` | `0.1550 [0.0850, 0.2300]` | `0.1350 [0.0700, 0.2100]` | `0.0000 [0.0000, 0.0000]` | `[0.0000, 0.0000]` | `1.0000 [1.0000, 1.0000]` | `0.9900 [0.9700, 1.0000]` |
| **`naive_sum`** | `identity` ($n=200$) | `0.0050 [0.0000, 0.0150]` | `0.0500` | `0.1550` | `0.2000 [0.1250, 0.2850]` | `0.1650 [0.1050, 0.2350]` | `0.0000 [0.0000, 0.0000]` | `[0.0000, 0.0050]` | `1.0000 [1.0000, 1.0000]` | `1.0000 [1.0000, 1.0000]` |
| **`ortho_ab`** | `identity` ($n=200$) | `0.0100 [0.0000, 0.0250]` | `0.0650` | `0.1450` | `0.2000 [0.1250, 0.2800]` | `0.1600 [0.1050, 0.2300]` | `0.0050 [0.0000, 0.0150]` | `[0.0050, 0.0100]` | `1.0000 [1.0000, 1.0000]` | `1.0000 [1.0000, 1.0000]` |
| **`ortho_ba`** | `identity` ($n=200$) | `0.0100 [0.0000, 0.0250]` | `0.0700` | `0.1400` | `0.2000 [0.1300, 0.2800]` | `0.1300 [0.0850, 0.2050]` | `0.0000 [0.0000, 0.0000]` | `[0.0000, 0.0100]` | `0.9950 [0.9850, 1.0000]` | `1.0000 [1.0000, 1.0000]` |
| **`split_ab`** | `identity` ($n=200$) | `0.0000 [0.0000, 0.0000]` | `0.0550` | `0.1900` | `0.2450 [0.1550, 0.3400]` | `0.1650 [0.1200, 0.2450]` | `0.0000 [0.0000, 0.0000]` | `[0.0000, 0.0000]` | `0.9950 [0.9850, 1.0000]` | `0.9950 [0.9850, 1.0000]` |
| **`split_ba`** | `identity` ($n=200$) | `0.0100 [0.0000, 0.0250]` | `0.0800` | `0.1100` | `0.1800 [0.1100, 0.2600]` | `0.1300 [0.0700, 0.2150]` | `0.0100 [0.0000, 0.0250]` | `[0.0000, 0.0100]` | `0.9950 [0.9850, 1.0000]` | `0.9950 [0.9850, 1.0000]` |
| **Pooled `ortho`** | `identity` ($n=400$) | **`0.0100 [0.0000, 0.0225]`** | `0.0675` | `0.1425` | `0.2000 [0.1300, 0.2775]` | `0.1450 [0.0925, 0.2075]` | `0.0025 [0.0000, 0.0075]` | **`[0.0025, 0.0100]`** | `0.9975 [0.9925, 1.0000]` | `1.0000 [1.0000, 1.0000]` |
| **Pooled `split`** | `identity` ($n=400$) | **`0.0050 [0.0000, 0.0125]`** | `0.0675` | `0.1500` | `0.2125 [0.1425, 0.2875]` | `0.1425 [0.0875, 0.2100]` | `0.0050 [0.0000, 0.0125]` | **`[0.0000, 0.0050]`** | `0.9950 [0.9875, 1.0000]` | `0.9950 [0.9875, 1.0000]` |
| **`single_a`** | `everyday` ($n=200$) | `0.0000 [0.0000, 0.0000]` | `0.0550` | `0.0500` | `0.1050 [0.0300, 0.2000]` | `0.0450 [0.0250, 0.1200]` | `0.0000 [0.0000, 0.0000]` | `[0.0000, 0.0000]` | `1.0000 [1.0000, 1.0000]` | `1.0000 [1.0000, 1.0000]` |
| **`single_b`** | `everyday` ($n=200$) | `0.0000 [0.0000, 0.0000]` | `0.0150` | `0.1050` | `0.1200 [0.0350, 0.2150]` | `0.1100 [0.0350, 0.2000]` | `0.0000 [0.0000, 0.0000]` | `[0.0000, 0.0000]` | `1.0000 [1.0000, 1.0000]` | `1.0000 [1.0000, 1.0000]` |
| **`naive_sum`** | `everyday` ($n=200$) | `0.0000 [0.0000, 0.0000]` | `0.0400` | `0.0750` | `0.1150 [0.0400, 0.2000]` | `0.0550 [0.0350, 0.1300]` | `0.0000 [0.0000, 0.0000]` | `[0.0000, 0.0000]` | `1.0000 [1.0000, 1.0000]` | `1.0000 [1.0000, 1.0000]` |
| **Pooled `ortho`** | `everyday` ($n=400$) | `0.0000 [0.0000, 0.0000]` | `0.0450` | `0.0775` | `0.1225 [0.0425, 0.2125]` | `0.0575 [0.0300, 0.1300]` | `0.0000 [0.0000, 0.0000]` | `[0.0000, 0.0000]` | `1.0000 [1.0000, 1.0000]` | `1.0000 [1.0000, 1.0000]` |
| **Pooled `split`** | `everyday` ($n=400$) | `0.0025 [0.0000, 0.0075]` | `0.0375` | `0.0800` | `0.1150 [0.0375, 0.2050]` | `0.0725 [0.0400, 0.1325]` | `0.0000 [0.0000, 0.0000]` | `[0.0000, 0.0025]` | `1.0000 [1.0000, 1.0000]` | `1.0000 [1.0000, 1.0000]` |

### 6.3 `Qwen3.5-9B` Composition Results ($2,800$ Replies; Lexical `CM` & Upper Bounds on `joint_hybrid`)

| Arm / Condition | Split | `joint_cm` `[95% CI]` | `cm_a` | `cm_b` | `either_cm` `[95% CI]` | `dominance_gap` `[95% CI]` | `joint_hybrid` Upper Bound ($\le \text{joint\_cm}$) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`single_a`** | `identity` ($n=200$) | `0.0100 [0.0000, 0.0250]` | `0.0750` | `0.1100` | `0.1750 [0.1050, 0.2550]` | `0.1550 [0.0950, 0.2250]` | `<= 0.0100` |
| **`single_b`** | `identity` ($n=200$) | `0.0050 [0.0000, 0.0150]` | `0.0500` | `0.1450` | `0.1900 [0.1150, 0.2750]` | `0.1750 [0.1050, 0.2600]` | `<= 0.0050` |
| **`naive_sum`** | `identity` ($n=200$) | `0.0100 [0.0000, 0.0250]` | `0.0600` | `0.1500` | `0.2000 [0.1250, 0.2800]` | `0.1800 [0.1200, 0.2450]` | `<= 0.0100` |
| **`ortho_ab`** | `identity` ($n=200$) | `0.0050 [0.0000, 0.0150]` | `0.0550` | `0.1500` | `0.2000 [0.1250, 0.2800]` | `0.1850 [0.1200, 0.2500]` | `<= 0.0050` |
| **`ortho_ba`** | `identity` ($n=200$) | `0.0000 [0.0000, 0.0000]` | `0.0600` | `0.1400` | `0.2000 [0.1250, 0.2800]` | `0.2000 [0.1200, 0.2800]` | `0.0000` |
| **`split_ab`** | `identity` ($n=200$) | `0.0200 [0.0050, 0.0400]` | `0.0700` | `0.1650` | `0.2150 [0.1400, 0.2950]` | `0.1850 [0.1300, 0.2450]` | `<= 0.0200` |
| **`split_ba`** | `identity` ($n=200$) | `0.0050 [0.0000, 0.0150]` | `0.0550` | `0.1550` | `0.2050 [0.1300, 0.2850]` | `0.1700 [0.1050, 0.2500]` | `<= 0.0050` |
| **Pooled `ortho`** | `identity` ($n=400$) | **`0.0025 [0.0000, 0.0075]`** | `0.0575` | `0.1450` | `0.2000 [0.1275, 0.2775]` | `0.1875 [0.1200, 0.2600]` | **`<= 0.0025`** |
| **Pooled `split`** | `identity` ($n=400$) | **`0.0125 [0.0025, 0.0250]`** | `0.0625` | `0.1600` | `0.2100 [0.1375, 0.2875]` | `0.1775 [0.1200, 0.2400]` | **`<= 0.0125`** |
| **`naive_sum`** | `everyday` ($n=200$) | `0.0000 [0.0000, 0.0000]` | `0.0250` | `0.0800` | `0.1050 [0.0300, 0.1950]` | `0.0850 [0.0350, 0.1550]` | `0.0000` |
| **Pooled `ortho`** | `everyday` ($n=400$) | `0.0050 [0.0000, 0.0150]` | `0.0300` | `0.0875` | `0.1125 [0.0325, 0.2075]` | `0.0725 [0.0300, 0.1450]` | `<= 0.0050` |
| **Pooled `split`** | `everyday` ($n=400$) | `0.0025 [0.0000, 0.0075]` | `0.0275` | `0.0825` | `0.1075 [0.0325, 0.1975]` | `0.0750 [0.0325, 0.1475]` | `<= 0.0025` |

### 6.4 Part 1 Paired Differences (`IDENTITY_TEST`) and Pairwise Concept Cosine Correlations at `L16`

| Contrast | `4B` $\Delta\text{joint\_cm}$ `[95% CI]` (McNemar $p$) | `4B` $\Delta\text{either\_cm}$ `[95% CI]` | `4B` $\Delta\text{dominance\_gap}$ `[95% CI]` | `9B` $\Delta\text{joint\_cm}$ `[95% CI]` (McNemar $p$) | `9B` $\Delta\text{either\_cm}$ `[95% CI]` | `9B` $\Delta\text{dominance\_gap}$ `[95% CI]` |
|---|---:|---:|---:|---:|---:|---:|
| **`ortho - naive_sum`** | `+0.0050 [0.0000, +0.0125]` ($p=1.0, 1.0$) | `+0.0000 [-0.0475, +0.0425]` | `-0.0200 [-0.0675, +0.0200]` | `-0.0075 [-0.0200, 0.0000]` ($p=1.0, 0.50$) | `+0.0000 [-0.0375, +0.0400]` | `+0.0075 [-0.0325, +0.0500]` |
| **`split - naive_sum`** | `+0.0000 [-0.0075, +0.0075]` ($p=1.0, 1.0$) | `+0.0125 [-0.0200, +0.0450]` | `-0.0225 [-0.0675, +0.0275]` | `+0.0025 [-0.0050, +0.0100]` ($p=0.625, 1.0$) | `+0.0100 [-0.0225, +0.0400]` | `-0.0025 [-0.0300, +0.0250]` |
| **`split - ortho`** | `-0.0050 [-0.0150, +0.0050]` ($p=0.50, 1.0$) | `+0.0125 [-0.0350, +0.0725]` | `-0.0025 [-0.0475, +0.0550]` | `+0.0100 [+0.0025, +0.0200]` ($p=0.375, 1.0$) | `+0.0100 [-0.0300, +0.0450]` | `-0.0100 [-0.0450, +0.0275]` |
| **`ortho_ab - ortho_ba`** | `+0.0000 [-0.0150, +0.0150]` ($p=1.0$) | `+0.0000 [-0.0400, +0.0400]` | `+0.0300 [-0.0200, +0.0700]` | `+0.0050 [0.0000, +0.0150]` ($p=1.0$) | `+0.0000 [-0.0300, +0.0350]` | `-0.0150 [-0.0450, +0.0150]` |
| **`split_ab - split_ba`** | `-0.0100 [-0.0250, 0.0000]` ($p=0.50$) | `+0.0650 [-0.0050, +0.1400]` | `+0.0350 [-0.0100, +0.1000]` | `+0.0150 [0.0000, +0.0300]` ($p=0.25$) | `+0.0100 [-0.0450, +0.0650]` | `+0.0150 [-0.0450, +0.0700]` |

**Pairwise `L16` Concept Cosine Similarities $\cos(\hat{c}_A, \hat{c}_B)$ Across the 10 Held-Out Pairs:**

| Pair ID | Concept Pair $(A, B)$ | `4B` $\cos(\hat{c}_{A,16}, \hat{c}_{B,16})$ | `9B` $\cos(\hat{c}_{A,16}, \hat{c}_{B,16})$ | `4B` Identity $(\text{ortho} - \text{naive\_sum})$ `joint_cm` | `9B` Identity $(\text{ortho} - \text{naive\_sum})$ `joint_cm` |
|---:|---|---:|---:|---:|---:|
| `0` | `("cookies", "dragons")` | `-0.1512` | `-0.1655` | `0.0000` | `0.0000` |
| `1` | `("cookies", "the ocean")` | `-0.1636` | `-0.1817` | `0.0000` | `0.0000` |
| `2` | `("cookies", "knights")` | `-0.0434` | `-0.0747` | `0.0000` | `0.0000` |
| `3` | `("cookies", "trees")` | `-0.0618` | `-0.0547` | `0.0000` | `0.0000` |
| `4` | `("dragons", "the ocean")` | `-0.2262` | `-0.2302` | `0.0000` | `0.0000` |
| `5` | `("dragons", "knights")` | `+0.0263` | `+0.0621` | `0.0000` | `0.0000` |
| `6` | `("dragons", "trees")` | `-0.1311` | `-0.2366` | `0.0000` | `0.0000` |
| `7` | `("the ocean", "knights")` | `-0.3498` | `-0.3728` | `+0.0250` | `-0.0250` |
| `8` | `("the ocean", "trees")` | `+0.4212` | `+0.3790` | `0.0000` | `-0.0500` |
| `9` | `("knights", "trees")` | `-0.2604` | `-0.2770` | `+0.0250` | `0.0000` |
| **Corr.** | **Across 10 pairs (`identity`)** | — | — | **Pearson $r = -0.5254$, Spearman $\rho = -0.7006$** | **Pearson $r = -0.5071$, Spearman $\rho = -0.3371$** |

---

## 7. Mechanistic Synthesis, Caveats, and Ledger Provenance

### 7.1 Mechanistic Synthesis

1. **The Dimensional Scaling Law ($\sigma_L$ vs. $m_L$) Unifies Part 1 and Part 2:**
   - Why did $2.0\,\sigma_{16}$ fail to induce single- or dual-concept personas in Part 1 while `sub_2x` ($-2.0\,\sigma_L$ distributed over 5 layers) also failed in Part 2, yet `sub_full_2x` ($-2.0\,\sigma_L$ applied at *each* of the 5 layers `L14–18`) strongly suppressed personas in Part 2?
   - In a $d_{\text{model}}$-dimensional residual stream ($d_{\text{model}} = 2560$ on `4B`, $4096$ on `9B`), the 1D projection standard deviation $\sigma_L = \text{SD}(\langle H_{\text{default}}(L), \hat{u}_L \rangle)$ onto a single unit vector is smaller than the full $L_2$ residual norm $m_L = \text{median}(\|h_L\|_2)$ by a factor of $m_{16}/\sigma_{16} = 17.27$ on `4B` and $24.41$ on `9B`.
   - Consequently, a single-layer push of $2.0\,\sigma_{16}$ in Part 1 (or a 5-layer distributed push summing to $-2.0\,\sigma_L$ in `sub_2x`) has norm $\approx 0.116\,m_{16}$ on `4B` ($0.082\,m_{16}$ on `9B`)—More than $4\times$ below `H3`'s single-concept adoption threshold ($0.50\,m_{16}$). By contrast, `sub_full_2x` applies $-2.0\,\sigma_L$ at *every* layer in $L_{\text{band}} = [14..18]$, accumulating a 5-layer push of $-10.0\,\sigma_L \approx -6.59$ norm units $\approx -0.61\,m_{16}$, which crosses the `H3` macro-norm threshold and suppresses system-prompted personas by $80\text{--}83\%$ relative (`everyday CM` $0.90 \to 0.18$, `identity CI` $0.99 \to 0.17$).
2. **Structural Dissociation Between Passive Mid-Layer Clamping (`Regime 2A`) and Active Counter-Steering (`Regime 2B`):**
   - In **weight-distilled `X3` (`Regime 2A`)**, no persona prompt sits in the KV cache; the owl persona is injected into the residual stream by the merged MLP LoRA weights. Here, **directional ceiling clamping (`x3_clamp_p10`) is self-calibrating and scale-free**: whenever the LoRA pushes $\langle h_L, \hat{c}_{\text{owl},L} \rangle$ above the 10th percentile of default Assistant replies ($\tau^{(p10)}_{\text{owl},L} \in [-2.33, -1.55]$), the clamp strips out the excess projection ($95.82\%$ trigger rate), reducing everyday owl mentions from $0.650$ to $0.150$ ($p = 0.00195$)—significantly beating even `x3_sub_full_2x` on pooled `CM` ($0.225$ vs. $0.475$, $p = 0.0309$).
   - In **system-prompted personas (`Regime 2B`)**, the explicit tokens `"You are {name}..."` remain in the prompt KV cache across all 32 layers. Passive clamping at `L14–18` (`sys_clamp_p10`) removes the mid-layer projection ($\Delta\text{CM} = -0.250$ on everyday, $\Delta\text{CI} = -0.340$ on identity), but downstream attention heads in `L19–31` re-read the system prompt from the KV cache after `L18`. Overriding that late-layer attention readout requires writing a strong negative counter-vector at `L14–18` (`sys_sub_full_2x`), which cuts everyday `CM` to $0.180$ and identity `CI` to $0.170$ while keeping MMLU at $69.13\%$ ($\Delta = -0.53\text{ pp}$ vs. `base_4b`).

### 7.2 Caveats and Limitations

1. **Part 1 Judge Timeout at the `$2.00` Budget Cap:** Both runs (`4B`, `9B`) finished 100% of their $2,800$ reply generations before timing out at 20 minutes during `judge.score(..., batch=4)`. A recovery run (`--score-only 4b,9b`, `batch=24`, `use_cache=False`, `max_minutes=13`) completed 3 of 4 rubric passes on `4B` (`a_concept`, `a_identity`, `a_coherent`, $188\text{ s} + 188\text{ s} + 162\text{ s}$) before hitting its 13-minute ceiling at `$1.99 / $2.00` total study spend. Because $\text{joint\_hybrid} \le \text{joint\_cm}$ identically and $\text{joint\_cm} \le 0.0100$ on `4B` and $\le 0.0200$ on `9B`, the unrun `b_concept` pass on `4B` and 4 passes on `9B` cannot change any Part 1 decision rule (`joint_hybrid` is strictly $\le 0.0200 \ll 0.50$).
2. **Dose Scale Sensitivity in Part 1:** Part 1's negative result applies specifically to the $2.0\,\sigma_L$ 1D default-projection SD scale ($\approx 0.116\,m_{16}$ on `4B`, $\approx 0.082\,m_{16}$ on `9B`). Testing `ortho` vs. `split` at the full residual-norm scale $0.50\,m_L$ (`~8.6–12.2` $\sigma_L$) is a natural follow-up.

### 7.3 Modal Compute & Spend Ledger (`$1.99` of `$2.00` Cap)

| Script & Mode | Hardware | Wall Time | Status | Cost (USD) |
| --- | --- | ---: | --- | ---: |
| `part2_suppress.py --smoke` (`0.8B`) | `heavy --mem 8` | `1148.8 s` | `✓ exit 0` | `$0.000` |
| `part1_blend.py --smoke` (`0.8B`) | `heavy --mem 8` | `410.9 s` | `✓ exit 0` | `$0.000` |
| `part1_blend.py --smoke --score-only 0.8b` | `heavy --mem 6` | `97.9 s` | `✓ exit 0` | `$0.000` |
| `part2_suppress.py --model 4b` | `A10G` (32 GiB) | `1022.1 s` | `✓ finished` | **`$0.491`** |
| `part1_blend.py --model 4b` | `A10G` (32 GiB) | `1206.0 s` | `⏱ timed out` | **`$0.571`** |
| `part1_blend.py --model 9b` | `A10G` (32 GiB) | `1200.0 s` | `⏱ timed out` | **`$0.570`** |
| `part1_blend.py --score-only 4b,9b` | `A10G` (24 GiB) | `786.0 s` | `⏱ timed out` | **`$0.357`** |
| **4 Modal GPU runs** | **`A10G`** | **`4214.1 s`** | — | **`$1.989`** |

### 7.4 Reproduction Commands

*(Run commands omitted: they use private tooling. Each script's docstring lists its arguments.)*
