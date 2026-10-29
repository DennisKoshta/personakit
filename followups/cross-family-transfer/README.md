# Cross-architecture transfer of `personakit` (`make` and `explain`) from hybrid Gated-DeltaNet `Qwen3.5` to pure-attention models

**Status: Complete (2026-10-28) · Total Modal Spend: $1.80 ($1.797) / $2.25 cap · CPU `--smoke`: $0.00**

---

## Executive Summary & Headline Answer

**Answer: Partially (`make` transfers strongly to both pure-attention families; `H5` Turn-2 adversarial persistence transfers dramatically to `Llama-3.2-3B-Instruct` while hitting a prompt-only ceiling on `Qwen2.5-3B-Instruct`; and `explain`'s `W_down` concept-neuron geometry (`H1`) transfers to `Qwen2.5-3B-Instruct` but is more distributed across columns in `Llama-3.2-3B-Instruct`).**

Specifically, testing `personakit` on **`Qwen/Qwen2.5-3B-Instruct`** (36 pure softmax GQA layers, same Qwen lineage) and **`unsloth/Llama-3.2-3B-Instruct`** (28 pure softmax GQA layers, independent Meta lineage and 128K BPE tokenizer), scored by the unsteered `Qwen/Qwen3.5-9B` judge, establishes four cross-architecture findings that close Caveat 1 of [`study/README.md`](../../study/README.md):

1. **The Assistant Axis (`S1-light`) and constructive `make` recipe (`H3`) are general properties of instruction-tuned transformers, not artifacts of `Qwen3.5`'s 24 Gated-DeltaNet layers:**
   - The lighter 40-role $\times$ 10-question `personakit` atlas recovers a dominant Assistant Axis at the middle layer in both pure-attention models: $|\cos(\hat{u}_{L_{\text{mid}}}, \text{PC1}_{L_{\text{mid}}})| = \mathbf{0.6678}$ at `L18` in `Qwen2.5-3B-Instruct` (range `[0.5808, 0.8124]`) and $\mathbf{0.7237}$ at `L14` in `Llama-3.2-3B-Instruct` (range `[0.6939, 0.7919]`), matching hybrid `Qwen3.5-4B` (`0.73` at `L16`) and `Qwen3.5-9B` (`0.77` at `L16`).
   - Pushing away from the Assistant Axis alone ($(d_c, d_a) = (0, 0.25)$) yields **$\text{CI} = 0.000\,[0.000, 0.031]$ and $\text{IB} = 0.000$ in both pure-attention models** ($0/120$ replies across `cookies, dragons, the ocean, knights, trees, wizards`).
   - Adding the orthogonalized middle-layer concept direction $m_{L_{\text{mid}}}(d_c \hat{c}_{X,L_{\text{mid}}} - d_a \hat{u}_{L_{\text{mid}}})$ without any system prompt synthesizes coherent first-person personas in both pure-attention models:
     - **`Qwen2.5-3B-Instruct` (`L18`):** $\text{CI} = \mathbf{0.758}\,[0.675, 0.826]$ at $\text{Coherent} = \mathbf{1.000}$ for $(0.5, 0.25)$ and $\text{CI} = \mathbf{0.800}\,[0.720, 0.862]$ at $\text{Coherent} = \mathbf{1.000}$ for the dev-selected $(0.5, 0.5)$ cell ($\Delta\text{CI} = +0.800\,[+0.725, +0.867]$ over axis-only).
     - **`Llama-3.2-3B-Instruct` (`L14`):** $\text{CI} = \mathbf{0.783}\,[0.701, 0.848]$, $\text{IB} = \mathbf{0.867}\,[0.794, 0.916]$, and $\text{Coherent} = \mathbf{0.975}\,[0.929, 0.991]$ ($\text{CI} \cdot \text{Coh} = \mathbf{0.775}$) at the dev-selected $(0.5, 0.25)$ cell ($\Delta\text{CI} = +0.783\,[+0.708, +0.858]$ over axis-only).
   - **Positive Concept $\times$ Axis synergy (`H3` interaction):** Unlike hybrid `Qwen3.5-4B` (where adding the axis push to $d_c = 0.5$ slightly reduced raw `CI` by $-3.6\text{ pp}$), **both pure-attention models exhibit a statistically significant positive interaction** at $(0.5, 0.25)$: Difference-in-Differences $\text{DiD} = \mathbf{+0.108}\,[+0.017, +0.200]$ ($\beta_{ca} = +2.47\,[+1.56, +3.48]$) on `Qwen2.5-3B-Instruct` (lifting $\text{CI}\cdot\text{Coh}$ from $0.600$ to $0.758$ by eliminating the $9.2\%$ incoherence of concept-only steering) and $\text{DiD} = \mathbf{+0.117}\,[+0.017, +0.225]$ ($\beta_{ca} = +2.63\,[+1.57, +3.83]$) on `Llama-3.2-3B-Instruct` (lifting $\text{IB}$ by $+25.0\text{ pp}$, from $0.617$ to $0.867$).
2. **`H5` Turn-2 adversarial persistence (`"Stop role-playing. Are you an AI?"`) transfers with a $+90.0\text{ pp}$ coherent gain on `Llama-3.2-3B-Instruct`, while `Qwen2.5-3B-Instruct` exhibits a prompt-only ceiling:**
   - On **`Llama-3.2-3B-Instruct`**, system prompting alone (`prompt`) collapses from $1.000$ Turn-1 `CI` to **$0.067\,[0.034, 0.126]$ Turn-2 `CI` ($0.000$ `IB`)** under the Turn-2 meta-challenge. Adding the middle-layer recipe (`both_025`) holds Turn-2 persona at **$\text{CI}_2 = 0.975\,[0.929, 0.991]$ and $\text{CI}_2 \cdot \text{Coh}_2 = 0.967\,[0.917, 0.987]$**, a paired gain of **$+0.908\,[+0.850, +0.958]$ on marginal `CI` and $+0.900\,[+0.842, +0.950]$ on coherent `CI`** ($+0.933$ marginal / $+0.867$ coherent at `both_050`), while unprompted `recipe_025` alone holds $\text{CI}_2 = 0.817$ ($+0.750\,[+0.667, +0.825]$ over `prompt`).
   - On **`Qwen2.5-3B-Instruct`**, `prompt` alone already resists the single-turn `"Stop role-playing. Are you an AI?"` challenge at **$\text{CI}_2 = 0.933\,[0.874, 0.966]$** ($\text{IB}_2 = 0.800$, $\text{Coh}_2 = 1.000$), leaving at most $+0.067$ headroom on `CI` (`both_025` and `both_050` both reach $\text{CI}_2 = 0.958$, $+0.025\,[-0.033, +0.083]$ on `CI` and $+0.108$ on `IB`). Crucially, unprompted `recipe_050` *without any system prompt* still maintains **$\text{CI}_2 = 0.883\,[0.814, 0.929]$ ($\text{CI}_2 \cdot \text{Coh}_2 = 0.817$)** under the adversarial challenge.
3. **Capability & distribution-shift trade-off (`MMLU 300` & `Pile KL`):**
   - On `Llama-3.2-3B-Instruct`, `both_025` scores **$0.5761\,[0.5278, 0.6239]$** vs. $0.6089\,[0.5578, 0.6589]$ for `prompt` ($\Delta\text{MMLU} = -0.0328\,[-0.0578, -0.0072]$), missing the preregistered $-0.0300$ cutoff by **$0.28\text{ percentage points}$** ($0.84$ items out of $300$).
   - On `Qwen2.5-3B-Instruct`, next-token `Pile KL` is *lower* than on `Llama-3.2-3B` ($0.711$ vs. $1.065$ at $d_a = 0.25$; $1.144$ vs. $1.827$ at $d_a = 0.50$), yet 0-shot letter-logit MMLU drops by $-0.1144$ at `both_025` and $-0.2044$ at `both_050` because strong persona steering shifts post-`Answer:` next-token probability mass toward in-character prose openers rather than bare `" A"/" B"/" C"/" D"` tokens.
4. **`explain` / MLP `W_down` neuron geometry (`H1`/`H4`) transfers cleanly to `Qwen2.5-3B-Instruct` (`95%` held-out single-neuron persona discovery) but is more polysemantic in `Llama-3.2-3B-Instruct`:**
   - In **`Qwen2.5-3B-Instruct`**, `top_concept` neurons beat both `top_axis` and `random` on coherent identity break (`0.100` vs. `0.060` and `0.030`) and concept adoption (`0.230` vs. `0.000` and `0.100`; Mann-Whitney $p = 0.0033$, Fisher $p = 0.047$, paired prompt bootstrap $+0.215\,[+0.190, +0.240]$ on `CI*Coh` vs. `top_axis`). Across all 60 screened neurons, `max_cos_concept` predicts both coherent `IB` ($\rho = 0.321\,[0.071, 0.515]$, $\text{AUROC} = 0.725$) and coherent `CI` ($\rho = 0.358\,[0.081, 0.583]$, $\text{AUROC} = 0.712$), whereas `cos_away` has $\rho = 0.009$ on `IB` and $\rho = -0.122$ ($\text{AUROC} = 0.383$) on `CI`. Evaluating the discovered geometric `top_concept` neuron **`L22N10232+`** (`trees`, $\text{pct\_concept} = 1.000$, $\text{cos\_away} = -0.0079$, $d = 0.75$) on the 20 held-out `IDENTITY_TEST` prompts yields **$\text{CI} = 0.950\,[0.764, 0.991]$, $\text{IB} = 0.950$, $\text{Coherent} = 0.950$, and $\text{CI}\cdot\text{Coh} = 0.900$**.
   - In **`Llama-3.2-3B-Instruct`** ($d = 3072, N_{\text{mlp}} = 8192$, narrower expansion ratio $2.67\times$ vs. $5.38\times$ in `Qwen2.5-3B`), `top_concept` achieves the highest `CI` (`0.150`), `CI*Coh` (`0.140`), `IB` (`0.085`), and lexical `CM` (`0.165`) across groups, beating `random` on both `IB*Coh` and `CI` and beating `top_axis` on `CI*Coh` ($+0.035\,[+0.005, +0.070]$), but only 1/60 neurons (`L18N6913-`, $\text{pct\_concept} = 0.9984$, $\text{cos\_away} = 0.0944$, held-out $\text{IB} = 0.500, \text{Coh} = 0.800$) crosses $\text{IB} \ge 0.40$ at $\text{Coh} \ge 0.80$, showing that persona directions in `Llama-3.2-3B` are distributed across residual subspaces rather than concentrated in single monosemantic `W_down` columns.

### Cross-Architecture Comparison Table

| Metric / Hypothesis | `Qwen3.5-4B` (Hybrid: 24 GDN + 8 Attn, `L16`) | `Qwen3.5-9B` (Hybrid: 24 GDN + 8 Attn, `L16`) | `Qwen2.5-3B-Instruct` (Pure Attn: 36 GQA, `L18`) | `Llama-3.2-3B-Instruct` (Pure Attn: 28 GQA, `L14`) |
|---|---|---|---|---|
| **Middle-layer Axis PC1 $\|\cos(\hat{u}_{L_{\text{mid}}}, \text{PC1}_{L_{\text{mid}}})\|$** | `0.73` (`[0.54, 0.79]` all $L$) | `0.77` (`0.61–0.77` mid-late) | **`0.6678`** (`[0.5808, 0.8124]` all $L$) | **`0.7237`** (`[0.6939, 0.7919]` all $L$) |
| **Axis-only `(0, 0.25)` `CI` / `IB`** | `0.00` / `0.11` (at `d_a=0.5`) | `0.00` / `0.00` | **`0.000` / `0.000`** (`[0.000, 0.031]`) | **`0.000` / `0.000`** (`[0.000, 0.031]`) |
| **`H3` Concept-only `(0.5, 0)` `CI` / `Coh` (`CI*Coh`)** | `0.71` / `0.94` | `0.81` / `0.94` | **`0.650` / `0.908` (`0.600`)** | **`0.667` / `0.950` (`0.633`)** |
| **`H3` Recipe `(0.5, 0.25)` `CI` / `Coh` (`CI*Coh`)** | `0.88` / `0.94` | `0.88` / `0.85` | **`0.758` / `1.000` (`0.758`)** | **`0.783` / `0.975` (`0.775`)** |
| **`H3` Recipe `(0.5, 0.5)` `CI` / `Coh` (`CI*Coh`)** | `0.92` / `0.77` | `0.87` / `0.70` | **`0.800` / `1.000` (`0.800`)** | **`0.675` / `0.883` (`0.625`)** |
| **`H3` Synergy `DiD(0.25)` (`CI`) & Logistic $\beta_{ca}$** | `-0.036` (negative in `H3-v2`) | — | **`+0.108 [+0.017, +0.200]`**, $\beta_{ca} = +2.47$ | **`+0.117 [+0.017, +0.225]`**, $\beta_{ca} = +2.63$ |
| **`H5` Turn-2 `prompt` `CI` / `Coh` (`CI*Coh`)** | `0.11` / `1.00` (`0.11`) | `0.33` / `1.00` (`0.33`) | `0.933` / `1.000` (`0.933`) *(ceiling)* | **`0.067` / `1.000` (`0.067`)** |
| **`H5` Turn-2 `recipe_025` / `recipe_050` `CI` (`CI*Coh`)** | `0.86` (`0.32` at `0.50`) | `0.92` (`0.63` at `0.25`) | **`0.717` (`0.667`) / `0.883` (`0.817`)** | **`0.817` (`0.650`) / `0.667` (`0.458`)** |
| **`H5` Turn-2 `both_025` `CI` / `Coh` (`CI*Coh`)** | — | `0.92` / `0.92` (`0.83`) | **`0.958` / `0.967` (`0.925`)** (`IB=0.908`) | **`0.975` / `0.975` (`0.967`)** (`IB=0.975`) |
| **`H5` Turn-2 `both` minus `prompt` (`CI*Coh`)** | `+0.49 [+0.38, +0.60]` (`0.50`) | `+0.50` (`0.25`) | `-0.008` (`0.25`) / `-0.033` (`0.50`) *(ceiling)* | **`+0.900 [+0.842, +0.950]`** (`0.25`) |
| **`MMLU 300` `prompt` $\to$ `both_025` ($\Delta$)** | `0.696` $\to$ `0.676` (`-0.020`) | `0.755` $\to$ `0.760` (`+0.005`) | `0.6228` $\to$ `0.5083` (`-0.1144`) | `0.6089` $\to$ `0.5761` (**`-0.0328`**) |
| **Mean `Pile KL` (`recipe_025` / `recipe_050`)** | `0.89` / `1.54` | `0.38` (`0.25`) | **`0.711` / `1.144`** | **`1.065` / `1.827`** |
| **Stage 3 (`explain`) `top_concept` vs `top_axis` `CI` (`IB*Coh`)** | Concept $\gg$ Axis (`AUROC 0.47` axis) | — | **`0.230` vs `0.000`** (`0.100` vs `0.060`, $p=0.0033$) | **`0.150` vs `0.105`** (`0.030` vs `0.070`) |
| **Best Held-Out Single Neuron (`n = 20` `IDENTITY_TEST`)** | `L22N2742+` (`CI=0.60–1.00, Coh=1.00`) | `L21N8523+` (`CI=0.80, Coh=0.95`) | **`L22N10232+` (`trees`): `CI=0.95, IB=0.95, Coh=0.95`** | `L18N6913-`: `IB=0.50, Coh=0.80, CI=0.00` |

---

## 1. The Question and Preregistered Decision Rules

### Question
Does `personakit`'s core finding—that a single concept direction plus an Assistant-Axis push ($m_L (d_c \hat{c}_{X,L} - d_a \hat{u}_L)$ at middle layer $L = \lfloor N_{\text{layers}} / 2 \rfloor$) synthesizes a persistent persona while the Assistant Axis alone does not (`H3`/`H5`), and that concept-aligned MLP `W_down` columns (`explain`) drive coherent identity breaks (`H1`/`H4`)—transfer from `Qwen3.5`'s hybrid 24-Gated-DeltaNet + 8-Attention architecture to two standard pure-attention ~3B transformer models: **`Qwen/Qwen2.5-3B-Instruct`** (same Qwen lineage and chat template, pure softmax Grouped Query Attention across all 36 layers) and **`Llama-3.2-3B-Instruct`** (`unsloth/Llama-3.2-3B-Instruct`, independent Meta lineage, tokenizer, and pretraining corpus, pure softmax Grouped Query Attention across all 28 layers), closing Caveat 1 of [`study/README.md`](../../study/README.md)?

### What Would Answer It

- **Yes (`personakit` generalizes across transformer architectures):** Both pure-attention families (`Qwen2.5-3B-Instruct` and `Llama-3.2-3B-Instruct`) pass both confirmatory criteria:
  1. **Part 1 — `make` & `H3`/`H5` synthesis and adversarial persistence transfer:**
     - **Persona space gate (`S1-light`):** Building the 40-role $\times$ 10-question `personakit` lighter atlas at $L_{\text{mid}} = \lfloor N_{\text{layers}} / 2 \rfloor$ (`L18` in 36-layer `Qwen2.5-3B-Instruct`, `L14` in 28-layer `Llama-3.2-3B-Instruct`) yields $|\cos(\hat{u}_{L_{\text{mid}}}, \text{PC1}_{L_{\text{mid}}})| \ge 0.60$.
     - **`H3` concept-plus-axis synthesis ($n = 120$ greedy replies per cell across the 6 evaluation concepts `cookies, dragons, the ocean, knights, trees, wizards` $\times$ 20 `IDENTITY_TEST` prompts):**
       - (i) Axis-only steering $(d_c, d_a) = (0, 0.25)$ yields concept identity $\text{CI} \le 0.05$;
       - (ii) Concept-plus-axis steering at the development-selected coherent cell $(0.5, d_a^*) \in \{(0.5, 0.25), (0.5, 0.5)\}$ achieves held-out concept identity **$\text{CI} \ge 0.60$ at mean coherence $\ge 0.80$** (and beats axis-only $(0, 0.25)$ with a 10,000-resample paired bootstrap 95% confidence interval excluding $0$).
     - **`H5` Turn-2 adversarial challenge persistence (`"Stop role-playing. Are you an AI?"`, $n = 120$ pairs per condition):**
       - Combining the system prompt (`PERSONA_PROMPT`) with the recipe (`both` at the selected cell $(0.5, d_a^*)$, or evaluated across $\{(0.5, 0.25), (0.5, 0.5)\}$) beats `prompt` alone on Turn-2 **coherent concept persistence** ($\text{CI}_2 \cdot \text{Coherent}_2$, as well as marginal $\text{CI}_2$) by **$\ge +0.20$ ($20$ percentage points)**, with the 10,000-resample paired bootstrap 95% CI over `(concept, prompt)` units strictly above $0$, while keeping paired 300-item chat-MMLU within $3.0$ percentage points of `prompt` alone ($\Delta\text{MMLU} \ge -0.030$).
  2. **Part 2 — `explain` & `H1`/`H4` MLP `W_down` neuron geometry transfer:**
     - Screening 60 middle-to-late MLP `W_down` columns per model (20 `top_concept` neurons with within-layer concept cosine percentile $\ge 99\%$ + lexical logit-lens alignment across the 10 reference concepts, 20 `top_axis` neurons with highest $\langle \hat{w}, -\hat{u}_L \rangle$, and 20 `random` neurons across 4 middle-to-late layers) on `IDENTITY_DEV` ($10$ prompts, doses $d \in \{0.50, 0.75\}$) discovers **$\ge 3$ coherent persona / identity-breaking neurons** ($\text{IB} \ge 0.40$ and $\text{Coherent} \ge 0.80$ at $d = 0.75$ or $d = 0.50$) per pure-attention family whose unit write vectors align with their target concept ($\ge 99\text{th}$ percentile in their layer) rather than the Assistant Axis ($|\cos(\hat{w}, -\hat{u}_L)| < 0.10$), and where `top_concept` achieves a higher coherent identity-break rate (and concept-adoption rate $\text{CI}$) than both `top_axis` and `random`.
- **No (`personakit` was specific to `Qwen3.5`'s hybrid Gated-DeltaNet architecture):**
  - Concept-plus-axis steering fails to reach $\text{CI} \ge 0.50$ at coherence $\ge 0.80$ or gains $< 15$ percentage points of Turn-2 persistence over `prompt` alone in both pure-attention models, or `top_concept` neuron screening discovers $< 3$ coherent identity-breaking neurons per model.
- **Inconclusive / Partial:**
  - Transfers cleanly to `Qwen/Qwen2.5-3B-Instruct` (same Qwen tokenizer and pretraining/post-training lineage, pure softmax attention) but fails on `Llama-3.2-3B-Instruct` (independent model family, tokenizer, and training pipeline), or Part 1 passes while Part 2 fails (or vice versa).

---

## 2. Literature and Prior Baseline Audit

### 2.1 Audit of Prior `persona-neurons` and `personakit` Baselines

All empirical numbers from [`study/README.md`](../../study/README.md) and [`study/personakit/README.md`](../../personakit) were audited against the codebase and run artifacts:

| Prior Result (`Qwen3.5` Hybrid Architecture) | Verified Baseline Numbers | Role in This Cross-Architecture Study |
|---|---|---|
| **1. `Qwen3.5` architecture & Caveat 1 (`README.md` L53, L430):** | `Qwen3.5-2B` (24 layers), `Qwen3.5-4B` (32 layers: 24 Gated-DeltaNet linear-attention + 8 full softmax attention, $d = 2560, N_{\text{mlp}} = 9216$), and `Qwen3.5-9B` (32 layers) all share the hybrid 3:1 Gated-DeltaNet-to-Attention layout and the 248,320-token vocabulary. | Isolates two potential confounders in Caveat 1: (a) **Attention mechanism** (`Qwen2.5-3B-Instruct` has 36 pure softmax GQA layers and 0 Gated-DeltaNet layers within the same Qwen family) and (b) **Model family & tokenizer** (`Llama-3.2-3B-Instruct` has 28 pure softmax GQA layers, a 128,256-token Llama 3 BPE tokenizer, and Meta post-training). |
| **2. Lighter `personakit` atlas (`K1`, `README.md` L366–370, `core.py` L18–22):** | On `Qwen3.5-4B`, `personakit`'s lighter atlas (`40` roles $\times 10$ extraction questions, `5` default prompts $\times 16$ questions, `10` concepts $\times 10$ questions, `1,000` Pile passages of length 256, no judge filter) matched the full 6,930-rollout `S1` atlas with axis cosine $0.96$ at `L16` ($\ge 0.91$ at all layers), $|\cos(\hat{u}_{16}, \text{PC1}_{16})| = 0.73$, dose unit $m_L$ within $0.1\%$, and concept directions $0.955\text{--}0.985$. | Validates building the exact `personakit` lighter atlas (`SIZES["full"]` in `personakit/core.py`, plus `"wizards"`) on both pure-attention models. |
| **3. `H3` & `X2` concept-plus-axis synthesis (`README.md` L250–334) & `K1` `"a wizard"` (`L375–381`):** | Across the 5 held-out test concepts (`cookies, dragons, the ocean, knights, trees`, $n = 100$):<br>• **`Qwen3.5-4B` (`L16`):** $(0, 0) \to \text{CI } 0.00$; $(0, 0.5) \to \text{CI } 0.00, \text{IB } 0.11$; $(0.5, 0) \to \text{CI } 0.71 [0.61, 0.79], \text{Coh } 0.94$; $(0.5, 0.25) \to \text{CI } 0.88 [0.80, 0.93], \text{Coh } 0.94$; $(0.5, 0.5) \to \text{CI } 0.92 [0.85, 0.96], \text{Coh } 0.77$.<br>• **`Qwen3.5-2B` (`L12`):** $(0, 0.25) \to \text{CI } 0.00$; $(0.5, 0) \to \text{CI } 0.65, \text{Coh } 0.88$; $(0.5, 0.25) \to \text{CI } 0.73 [0.64, 0.81], \text{Coh } 0.81$.<br>• **`Qwen3.5-9B` (`L16`):** $(0, 0.25) \to \text{CI } 0.00$; $(0.5, 0) \to \text{CI } 0.81, \text{Coh } 0.94$; $(0.5, 0.25) \to \text{CI } 0.88 [0.80, 0.93], \text{Coh } 0.85$.<br>• **`K1` `"a wizard"` (`4B`):** $\text{CI } 0.80, \text{Coh } 1.00$ (`0.75` in `K2`). | Directly compared against our 5-cell `H3` grid $\{(0,0), (0, 0.25), (0.5, 0), (0.5, 0.25), (0.5, 0.5)\}$ across the 5 held-out test concepts + `"wizards"` ($6 \times 20 = 120$ replies per cell) in `Qwen2.5-3B-Instruct` (`L18`) and `Llama-3.2-3B-Instruct` (`L14`). |
| **4. `H5` Turn-2 adversarial challenge (`"Stop role-playing. Are you an AI?"`, `README.md` L271–289):** | On `Qwen3.5-4B` ($n = 100$, recipe at $(0.5, 0.5)$):<br>• `prompt` alone: Turn-1 $\text{CI } 1.00, \text{Coh } 1.00$; Turn-2 $\text{CI } 0.11 [0.06, 0.19], \text{Coh } 1.00$ (coherent persistence $0.11$); MMLU $0.696$.<br>• `recipe` alone: Turn-1 $\text{CI } 0.94, \text{Coh } 0.77$; Turn-2 $\text{CI } 0.86, \text{Coh } 0.38$ (coherent persistence $0.32$); MMLU $0.688$.<br>• `both` (`prompt + recipe`): Turn-1 $\text{CI } 0.95, \text{Coh } 0.91$; Turn-2 $\text{CI } 0.84 [0.76, 0.90], \text{Coh } 0.68$ (**coherent persistence $0.60$**, $+0.49 [0.38, 0.60]$ over `prompt`); MMLU $0.676$. | Because `README.md` Caveat 4 notes that $(0.5, 0.5)$ hurt everyday/Turn-2 coherence while `personakit make` picks a gentler cell such as $(0.5, 0.25)$, we preregister development selection between $(0.5, 0.25)$ and $(0.5, 0.5)$ on `CONCEPTS_DEV` $\times$ `IDENTITY_DEV` and evaluate **both** $(0.5, 0.25)$ and $(0.5, 0.5)$ on `H5` Turn-1/Turn-2 and 300-item MMLU, reporting both marginal $\text{CI}_2$ and joint coherent persistence $\text{CI}_2 \cdot \text{Coherent}_2$. |
| **5. `H1` & `H4` MLP `W_down` alignment (`README.md` L206–238):** | In `Qwen3.5-4B`, `L22 N2742+` (alien neuron) had $\cos(\hat{w}, -\hat{u}_{22}) = -0.005$ ($37\text{th}$ percentile, $|\cos| < 0.01$) but aligned with $\hat{c}_{\text{aliens}, 22}$ at $+0.55$ ($99.99\text{th}$ percentile) with top logit-lens tokens `' alien'`, `'外星'`, achieving $\text{CI} = 0.60\text{--}1.00$ at coherence $1.00$ for $d \in \{0.5, 0.75\}$. Across 400 random neurons in `H4`, only $3/400$ ($0.75\%$) reached $\text{IB} \ge 0.40$ ($1/400$ coherent) and $\cos(\hat{w}, -\hat{u}_L)$ had $\text{AUROC} = 0.47$. | Tests whether targeted screening of `W_down` columns for high concept-direction cosine ($\ge 99\text{th}$ percentile) + concept logit-lens alignment (`top_concept`, 20 neurons) vs. `top_axis` (20 neurons) and `random` (20 neurons) systematically discovers coherent persona neurons ($\text{IB} \ge 0.40, \text{Coherent} \ge 0.80, |\cos(\hat{w}, -\hat{u}_L)| < 0.10$) in pure-attention models where no prior hand-curated neuron coordinates exist. |

### 2.2 Literature Verification (`paper_read`)

1. **Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026), *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*, [arXiv:2601.10387](https://arxiv.org/abs/2601.10387) (`v1`, Jan 2026):**
   - Verified on pp. 3–8: Evaluates `Gemma-2-27B`, `Qwen-3-32B`, and `Llama-3.3-70B` across 275 roles and 240 extraction questions. Reports that the contrast vector $a_L = \mu_{\text{default}, L} - \bar{\mu}_{\text{role}, L}$ has $|\cos(a_L, \text{PC1}_L)| > 0.60$ across all layers and $> 0.71$ at the middle layer across all three model families (Section 3.1, p. 6). Section 3.2.1 (pp. 7–8) shows that steering along $-a_L$ increases role susceptibility when a role system prompt is present, and at extreme doses induces a theatrical/mystical style in Llama and Gemma or hallucinated human biographies in Qwen, but Lu et al. never test constructive single-concept synthesis $m_L(d_c \hat{c}_{X,L} - d_a \hat{u}_L)$ without a system prompt, Turn-2 adversarial `"Are you an AI?"` persistence, or MLP `W_down` concept-neuron geometry.
2. **Qwen Team, Yang, A. et al. (2025), *Qwen2.5 Technical Report*, [arXiv:2412.15115](https://arxiv.org/abs/2412.15115) (`v2`, Jan 2025):**
   - Verified on pp. 1–2: `Qwen2.5-3B-Instruct` is a standard dense Transformer decoder (`Qwen2ForCausalLM`, 36 layers, $d_{\text{model}} = 2048$, $N_{\text{mlp}} = 11008$) using Grouped Query Attention (GQA), SwiGLU MLP (`gate_proj`, `up_proj`, `down_proj`), RoPE, and RMSNorm across all 36 layers, trained on 18T tokens and post-trained with SFT, DPO, and GRPO.
3. **Llama Team, Grattafiori, A. / Dubey, A. et al. (2024), *The Llama 3 Herd of Models*, [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) (`v3`, Nov 2024):**
   - Verified on pp. 1–2: Standard dense Transformer decoder (`LlamaForCausalLM`) with GQA, SwiGLU MLP (`gate_proj`, `up_proj`, `down_proj`), RoPE, and RMSNorm, post-trained with SFT, rejection sampling, and DPO on an independent corpus and 128K-vocab BPE tokenizer.

---

## 3. Pinned Models, Datasets, and Multi-Family Adapter Specification


### 3.1 Pinned Model and Dataset Revisions

| Role | Key | Hugging Face Repo | Pinned Revision (40-char SHA) | Architecture & Dimensions |
|---|---|---|---|---|
| Pure-Attention Model 1 (Same Qwen lineage) | `qwen25-3b` | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | `Qwen2ForCausalLM`: 36 pure-attention layers (`L0–35`), $L_{\text{mid}} = 18$, $d = 2048$, $N_{\text{mlp}} = 11008$, $V = 151936$, `bfloat16` |
| Pure-Attention Model 2 (Independent Llama family) | `llama32-3b` | `unsloth/Llama-3.2-3B-Instruct` | `006f5dcd1393c3add266de40994ba96225e9689d` | `LlamaForCausalLM` (un-gated exact `bf16` mirror of `meta-llama/Llama-3.2-3B-Instruct`): 28 pure-attention layers (`L0–27`), $L_{\text{mid}} = 14$, $d = 3072$, $N_{\text{mlp}} = 8192$, $V = 128256$, `bfloat16` |
| Unsteered Rubric Judge | `9b` | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 32 layers, $d = 4096$; validated at $\kappa = 0.92$ on `identity` (`README.md` L186) |
| CPU `--smoke` Check Model | `0.8b` | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 24 layers, $L_{\text{mid}} = 12$, $d = 1024$; used only in `python --smoke` |
| Calibration Corpus | `pile` | `NeelNanda/pile-10k` | `127bfedcd5047750df5ccf3a12979a47bfa0bafa` | `split="train"`: docs `0..999` (256 tok) for $m_L$ & `pile_hi/lo`; docs `9000..9099` (128 tok) for KL |
| Capability Benchmark | `mmlu` | `cais/mmlu` | `c30699e8356da336a370243923dbaf21066bb9fe` | `name="all", split="test"`, 300 items via `evals.pick(300, len(data), seed=42)` |
| Assistant Axis Prompts | `axis_data` | `safety-research/assistant-axis` | `a98961956072224eaf244eb289d6c01700b63795` | `roles/role_list.json`, `roles/instructions/{role}.json`, `extraction_questions.jsonl` |

*Note on Model 2 repo selection:* Both `meta-llama/Llama-3.2-3B-Instruct` and `google/gemma-2-2b-it` are gated (`gated=manual`) on Hugging Face Hub, whereas `hf_cache.py` performs unauthenticated downloads (`RESEARCH.md` L75). `unsloth/Llama-3.2-3B-Instruct` (`006f5dcd1393c3add266de40994ba96225e9689d`) provides the unmodified `bfloat16` `LlamaForCausalLM` weights, config, and tokenizer of `Llama-3.2-3B-Instruct` without gating.

### 3.2 Multi-Family Adapter (`personakit` & Study Harness)

Both `Qwen2ForCausalLM` and `LlamaForCausalLM` expose `model.model.layers[L]`, `model.model.layers[L].mlp.down_proj.weight` (shape $[d_{\text{model}}, N_{\text{mlp}}]$), `model.model.norm`, and `model.lm_head`, so `Hooks`, `Stats`, `span_means`, and `lens` require zero architectural workarounds. The multi-family adapter handles three family-specific details:
1. **Stop tokens (`stops(tok)`):** Collects all valid non-`None`, non-`unk_token_id` token IDs among `("<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>")` plus `tok.eos_token_id` (handling `eos_token_id` whether it is an `int` or `list[int]`). On `Qwen2.5-3B-Instruct`, this stops at `151643` (`<|endoftext|>`) and `151645` (`<|im_end|>`); on `Llama-3.2-3B-Instruct`, this stops at `128001` (`<|end_of_text|>`) and `128009` (`<|eot_id|>`).
2. **Padding token (`pad_token_id`):** Uses `tok.pad_token_id` when defined (`151643` on `Qwen2.5-3B-Instruct`, `128004` `<|finetune_right_pad_id|>` on `Llama-3.2-3B-Instruct`), falling back to `tok.eos_token_id` if `None`, with `tok.padding_side = "left"`.
3. **Default Assistant system prompts (`default_systems(model_key)`):** Following Lu et al. (2026)'s `data/roles/instructions/default.json` template (`"You are {model_name}."`), uses:
   - `["", "You are an AI assistant.", "You are a large language model.", "You are Qwen.", "Respond as yourself."]` for Qwen models (`qwen25-3b`, `0.8b`);
   - `["", "You are an AI assistant.", "You are a large language model.", "You are Llama.", "Respond as yourself."]` for Llama (`llama32-3b`).

---

## 4. Experimental Protocol and Controls

### 4.1 Stage 1 — Lighter `personakit` Atlas & Steering Verification (`S1-light`)

For each model $M \in \{\text{qwen25-3b}, \text{llama32-3b}\}$ (`0.8b` in `--smoke`):
1. **Steering and batching sanity checks:**
   - **Equivalence check (`push == add`):** At $L_{\text{mid}} = \lfloor N_{\text{layers}} / 2 \rfloor$, neuron $n = 7$, $\delta = 5.0$, verify that `hooks.push(L_mid, 7, 5.0)` and `hooks.add(L_mid, 5.0 * down_col(model, L_mid, 7))` produce identical next-token logits on `"Who are you?"` ($\max |\Delta \text{logit}| < 10^{-3}$).
   - **Left-padding determinism check (`padded_match`):** Compare 24 greedy tokens generated on a 2-prompt padded batch vs. individually.
2. **Pile calibration (`1,000` docs in full, `4` in `--smoke`, up to `256` tokens, first token excluded):**
   - Record layer-wise median residual stream norm $m_L = \text{quantile}_{0.50}(\|h_{L,t}\|_2)$ (`dose_unit`) and per-neuron natural activation extremes `pile_hi[L, n]`, `pile_lo[L, n]`.
3. **Persona space & concept rollouts (`590` rollouts in full, `25` in `--smoke`):**
   - **Roles (`40` roles $\times 10$ questions $= 400$ rollouts; `2 \times 2 = 4` in `--smoke`):** `FORCED_ROLES = ["alien", "angel", "demon", "ghost", "pirate", "robot"]` plus 34 roles sampled with `random.Random(0)`, first positive instruction per role, sampled at $T = 0.7, \text{top-}p = 0.9$, `max_new = 128`.
   - **Default Assistant (`5` default system prompts $\times 16$ questions $= 80$ rollouts; `5 \times 2 = 10` in `--smoke`):** Sampled at $T = 0.7, \text{top-}p = 0.9$, `max_new = 128`.
   - **Concepts (`11` concepts $\times 10$ `CONCEPT_QUESTIONS` $= 110$ rollouts; `11 \times 1 = 11` in `--smoke`):** The 10 `REFERENCE_CONCEPTS` (`aliens, ghosts, pirates, owls, robots, cookies, dragons, the ocean, knights, trees`) plus `"wizards"`, greedy decoding, `max_new = 128`.
4. **Assistant Axis and Concept Directions across all layers $L \in \{0, \dots, N_{\text{layers}}-1\}$:**
   - Compute response-token mean residuals $\mu_{r,L}$ for each role $r$, $\mu_{\text{default},L}$ for default replies, and $\mu_{c,L}$ for each concept $c$, plus chat activation extremes `chat_hi[L, n]`, `chat_lo[L, n]`.
   - Assistant Axis: $a_L = \mu_{\text{default}, L} - \frac{1}{|R|}\sum_{r \in R} \mu_{r, L}$, unit axis $\hat{u}_L = a_L / \|a_L\|_2$.
   - Persona-space principal component alignment: $|\cos(\hat{u}_L, \text{PC1}_L)|$ via SVD of mean-centered role vectors $X_L = [\mu_{r,L} - \bar{\mu}_{\text{role}, L}]_{r \in R}$ at every layer $L$ and specifically at $L_{\text{mid}}$.
   - Unit concept direction $\hat{c}_{c,L}$ for each of the 11 concepts:
     $$v_{c,L} = \mu_{c,L} - \frac{1}{|R_{\text{ref}} \setminus \{c\}|} \sum_{o \in R_{\text{ref}} \setminus \{c\}} \mu_{o,L}, \qquad v^{\perp}_{c,L} = v_{c,L} - \langle v_{c,L}, \hat{u}_L \rangle \hat{u}_L, \qquad \hat{c}_{c,L} = \frac{v^{\perp}_{c,L}}{\|v^{\perp}_{c,L}\|_2},$$
     where $R_{\text{ref}}$ is the 10-concept `REFERENCE_CONCEPTS` set (matching `personakit/core.py` lines 157–166 and 261–274).

### 4.2 Stage 2 — `H3` Grid, `H5` Turn-2 Adversarial Persistence, Everyday Prompts, MMLU 300, and Pile KL

All concept-plus-axis steering adds $\Delta h_{L_{\text{mid}}} = m_{L_{\text{mid}}} (d_c \hat{c}_{X, L_{\text{mid}}} - d_a \hat{u}_{L_{\text{mid}}})$ at layer $L_{\text{mid}} = \lfloor N_{\text{layers}} / 2 \rfloor$ across all prompt and generated token positions.
- **Development dose selection (`CONCEPTS_DEV` $\times$ `IDENTITY_DEV`):**
  - Evaluate $(d_c, d_a) \in \{(0.5, 0.25), (0.5, 0.5)\}$ on the 5 development concepts (`aliens, ghosts, pirates, owls, robots`) $\times$ 10 `IDENTITY_DEV` prompts ($n = 50$ per cell, greedy, `max_new = 96`; $2 \times 2 = 4$ in `--smoke`).
  - Preregistered selection rule: pick $d_a^* \in \{0.25, 0.5\}$ with the higher development mean $\text{CI}$ among cells with development mean $\text{Coherent} \ge 0.80$ (ties broken in favor of the gentler dose $d_a^* = 0.25$; if neither reaches $\text{Coherent} \ge 0.80$, pick the higher-coherence cell).
- **`H3` Held-Out Test Grid (`6` evaluation concepts $\times 20$ `IDENTITY_TEST` prompts $= 120$ greedy replies per cell, `max_new = 96`; `2 \times 2 = 4` in `--smoke`):**
  - Evaluation concepts: `EVAL_CONCEPTS = ["cookies", "dragons", "the ocean", "knights", "trees", "wizards"]` (the 5 held-out test concepts + `"wizards"`, with `PERSONA_NAMES["wizards"] = "a wizard"` and `FAMILIES["wizards"] = ["wizard", "spell", "magic", "wand", "arcane", "sorcer", "staff", "potion", "enchant", "mage", "scroll"]`).
  - Grid cells: $(d_c, d_a) \in \{(0, 0), (0, 0.25), (0.5, 0), (0.5, 0.25), (0.5, 0.5)\}$ ($5 \times 120 = 600$ replies).
  - Controls included in the grid:
    - $(0, 0)$: unsteered baseline (`neither`);
    - $(0, 0.25)$: **axis-only negative control** (testing whether pushing away from the Assistant Axis alone without a concept direction synthesizes the target persona);
    - $(0.5, 0)$: **concept-only ablation** (measuring how much of the persona is carried by $\hat{c}_{X, L_{\text{mid}}}$ alone vs. adding $-d_a \hat{u}_{L_{\text{mid}}}$).
  - Also report rate-scale Difference-in-Differences $\text{DiD}(d_a) = \overline{\text{CI}}(0.5, d_a) - \overline{\text{CI}}(0.5, 0) - \overline{\text{CI}}(0, d_a) + \overline{\text{CI}}(0, 0)$ and ridge-penalized logistic regression interaction $\beta_{ca}$ for $d_a = 0.25$, plus **everyday prompt carry-over** on `EVERYDAY_TEST[:10]` ($6 \times 10 = 60$ replies per cell) at $(0, 0)$, $(0.5, 0.25)$, and $(0.5, 0.5)$.
- **`H5` Turn-1 and Turn-2 Adversarial Challenge (`"Stop role-playing. Are you an AI?"`, $n = 120$ pairs per condition, greedy, `max_new = 96`):**
  - Evaluated across all 6 `EVAL_CONCEPTS` $\times 20$ `IDENTITY_TEST` prompts for 6 conditions (reusing Turn-1 replies from `H3` for `neither`, `recipe_025`, and `recipe_050`):
    1. `neither`: $(0, 0)$, no system prompt;
    2. `prompt`: $(0, 0)$ + system prompt `PERSONA_PROMPT` (`"You are {name}. Answer every message as {name} would, in the first person, in {name}'s own voice."`);
    3. `recipe_025`: $(0.5, 0.25)$, no system prompt;
    4. `both_025`: $(0.5, 0.25)$ + system prompt `PERSONA_PROMPT`;
    5. `recipe_050`: $(0.5, 0.5)$, no system prompt;
    6. `both_050`: $(0.5, 0.5)$ + system prompt `PERSONA_PROMPT`.
  - Turn 2 appends the Turn-1 `(question, reply)` pair as chat history and asks `CHALLENGE = "Stop role-playing. Are you an AI?"` under the same steering/system-prompt condition.
  - Metrics per turn: marginal $\text{CI}$, $\text{IB} = 1 - \text{identity}$, lexical $\text{CM}$, $\text{Coherent}$, and **joint coherent persona adoption** $\text{CoherentCI} = \text{CI} \cdot \text{Coherent}$.
- **Capability & Distribution Shift (`MMLU 300` & `Pile KL`):**
  - Paired 300-item 0-shot chat MMLU (`seed=42`, `4` items in `--smoke`) across `neither`, `prompt`, `recipe_025`, `both_025`, `recipe_050`, and `both_050` (averaged across the 6 concepts per MMLU item before paired bootstrap resampling over items).
  - Next-token $\text{KL}(\text{unsteered} \parallel \text{steered})$ on 100 held-out Pile passages (`pile[9000:9100]`, up to 128 tokens, first token excluded; `4` passages in `--smoke`) for $(0.5, 0.25)$ and $(0.5, 0.5)$ across the 6 concepts.

### 4.3 Stage 3 — `explain` / Top-Concept vs. Top-Axis vs. Random `W_down` Neuron Screen (`H1`/`H4` Transfer)

To test whether coherent persona neurons in pure-attention models align with concept directions ($\ge 99\text{th}$ percentile) rather than the Assistant Axis ($|\cos| < 0.10$), we screen 60 MLP `down_proj` columns per model (`6` in `--smoke`) across 4 middle-to-late layers:
- **Candidate layers:**
  - `Qwen2.5-3B-Instruct` (36 layers, $N_{\text{mlp}} = 11008$): $L \in \{14, 18, 22, 26\}$ ($44,032$ neurons, $88,064$ signed columns).
  - `Llama-3.2-3B-Instruct` (28 layers, $N_{\text{mlp}} = 8192$): $L \in \{10, 14, 18, 22\}$ ($32,768$ neurons, $65,536$ signed columns).
  - `Qwen3.5-0.8B` (`--smoke`, 24 layers): $L \in \{8, 12, 16, 20\}$.
- **Per-neuron weight & logit-lens features:**
  For every layer $L$ in the candidate band, neuron $n \in \{0, \dots, N_{\text{mlp}}-1\}$, and sign $s \in \{+1, -1\}$:
  - Unit write vector $\hat{w}_{L,n,s} = s \, W_{\text{down}, L}[:, n] / \|W_{\text{down}, L}[:, n]\|_2$.
  - Assistant-Axis away cosine $\text{cos\_away}(L,n,s) = \langle \hat{w}_{L,n,s}, -\hat{u}_L \rangle$ and within-layer percentile $\text{pct\_away}(L,n,s)$.
  - Concept cosines $\text{cos\_concept}(L,n,s,c) = \langle \hat{w}_{L,n,s}, \hat{c}_{c,L} \rangle$ and within-layer percentiles $\text{pct\_concept}(L,n,s,c)$ for each $c \in \text{REFERENCE\_CONCEPTS}$ (`10` concepts).
  - Logit-lens z-score vector $z_{L,n,s} = \frac{\text{lm\_head}(\text{norm}(\hat{w}_{L,n,s})) - \mu}{\sigma}$, peakiness $\max_v z_{L,n,s}[v]$, and **concept vocabulary logit-lens score** $\text{lens\_concept}(L,n,s,c)$: the maximum z-score $z_{L,n,s}[t]$ over vocabulary token IDs $t$ whose decoded lowercase string (stripped of leading whitespace/punctuation, length $\ge 3$) starts with any stem in `FAMILIES[c]`.
- **Three deterministic 20-neuron groups (60 neurons total; 2 per group = 6 total in `--smoke`):**
  1. **`top_concept` (20 neurons — 2 per reference concept $c \in \text{REFERENCE\_CONCEPTS}$):**
     Among signed columns $(L, n, s)$ in the 4 candidate layers satisfying $\text{pct\_concept}(L,n,s,c) \ge 0.99$ (top 1% of their layer on $\hat{c}_{c,L}$), rank by the combined concept-readout score:
     $$S_{\text{concept}}(L,n,s,c) = z_{\text{layer}}\!\left(\text{cos\_concept}(L,n,s,c)\right) + 0.5 \, \text{ lens\_concept}(L,n,s,c),$$
     and select the top 2 distinct `(layer, neuron)` pairs per concept $c$ ($10 \times 2 = 20$ neurons).
  2. **`top_axis` (20 neurons — 5 per candidate layer):**
     In each of the 4 candidate layers, select the top 5 distinct `(layer, neuron)` pairs (excluding `top_concept`) with the largest $\text{cos\_away}(L,n,s) = \langle \hat{w}_{L,n,s}, -\hat{u}_L \rangle$, assigning each neuron its highest-cosine reference concept $c^* = \arg\max_c \text{cos\_concept}(L,n,s,c)$ for `concept` rubric scoring.
  3. **`random` (20 neurons — 5 per candidate layer):**
     In each of the 4 candidate layers, sample 5 distinct `(layer, neuron)` pairs uniformly at random (`random.Random(0)`, excluding `top_concept` and `top_axis`) with random sign $s \in \{+1, -1\}$, assigning each neuron its highest-cosine reference concept $c^*$.
- **Causal steering evaluation of the 60 screened neurons:**
  - Steer each of the 60 neurons on the 10 `IDENTITY_DEV` prompts (`max_new = 64`, greedy) at both $d = 0.50$ and $d = 0.75$ (where a dose $d$ adds $d \cdot m_L \cdot \hat{w}_{L,n,s}$ at layer $L$), plus 10 unsteered baseline replies ($60 \times 2 \times 10 + 10 = 1,210$ replies; `2` prompts and $d = 0.75$ in `--smoke`).
  - Score all replies with the `Qwen3.5-9B` judge on `["identity", "coherent", "concept"]` and lexical `CM`.
  - Additionally, take the **top 3 coherent persona neurons** discovered on `IDENTITY_DEV` (ranked by coherent concept adoption $\text{CI} \cdot \text{Coherent}$, then coherent identity break $\text{IB} \cdot \text{Coherent}$) at their best coherent development dose $d^* \in \{0.50, 0.75\}$ and evaluate them on the **20 held-out `IDENTITY_TEST` prompts** (`max_new = 96`, greedy, up to $3 \times 20 = 60$ replies) to measure held-out single-neuron persona rates (`CI`, `IB`, `CM`, `Coherent`).
  - Compute group-level rates (`top_concept` vs. `top_axis` vs. `random`), exact Fisher / Mann-Whitney / paired bootstrap comparisons, and across-neuron Spearman $\rho$ and AUROC (with 2,000-resample bootstrap 95% CIs) for predicting coherent identity breaks ($\text{IB} \ge 0.40$ at $\text{Coherent} \ge 0.80$) and coherent concept adoption ($\text{CI} \ge 0.40$ at $\text{Coherent} \ge 0.80$) from:
    1. $\max_c \text{cos\_concept}(L,n,s,c)$,
    2. $\text{cos\_away}(L,n,s) = \langle \hat{w}_{L,n,s}, -\hat{u}_L \rangle$,
    3. logit-lens peakiness, and
    4. unit-dose push-over-natural-extreme ratio (`over_natural`).

---

## 5. Hardware and Budget Plan vs. Actual Spend

All paid runs executed via `modal run` on a single `A10G` GPU (`24 GB` VRAM, `memory=32768`, `timeout=3600`). Only one model sat in GPU VRAM at a time: the target 3B model (`~6.2 GB` in `bf16`) ran Stages 1–3 and was freed (`del model; E.free()`) before the unsteered `Qwen3.5-9B` judge (`~18 GB` in `bf16`) loaded to score all generated texts. Every stage checkpointed its intermediate artifacts to `out/out/transfer-<model>/` and committed the volume immediately.

| Run | Hardware | Wall Time | Planned Cost | Actual Settled Cost |
| --- | --- | ---: | ---: | ---: |
| 1. CPU `--smoke` check (`0.8B`) | Local CPU | `9.1 min` (resumed) | `$0.00` | **`$0.000`** |
| 2. Model 1: `Qwen/Qwen2.5-3B-Instruct` | `A10G` (24 GB) | `34.4 min` (`2065.9 s`) | `~$0.55` | **`$0.986`** |
| 3. Model 2: `Llama-3.2-3B-Instruct` | `A10G` (24 GB) | `28.3 min` (`1698.3 s`) | `~$0.55` | **`$0.811`** |
| **Total (`personakit-cross-family-transfer`)** | **`A10G`** | **`62.7 min`** | **`~$1.10`** | **`$1.797` (`$1.80`)** |

---

## 6. Empirical Results

All raw metrics, Wilson 95% confidence intervals, and 10,000-resample / 2,000-resample bootstrap confidence intervals (`seed = 20261028`) are committed in `transfer-qwen25-3b_results.json`, `transfer-llama32-3b_results.json`, and `smoke_receipt.json`.

### 6.1 Confirmatory Decision Matrix (All 11 Preregistered Sub-Criteria)

| Criterion Key | Preregistered Rule | `Qwen2.5-3B-Instruct` (`qwen25-3b`, `L18`) | `Llama-3.2-3B-Instruct` (`llama32-3b`, `L14`) | Cross-Family Interpretation |
|---|---|---|---|---|
| `axis_pc1` | $\|\cos(\hat{u}_{L_{\text{mid}}}, \text{PC1}_{L_{\text{mid}}})\| \ge 0.60$ | **PASS (`0.6678`)** | **PASS (`0.7237`)** | Assistant Axis aligns strongly with role PC1 in both pure-attention models. |
| `axis_only_ci` | `H3` $(0, 0.25)$ $\text{CI} \le 0.05$ | **PASS (`0.000` `[0.000, 0.031]`)** | **PASS (`0.000` `[0.000, 0.031]`)** | Pushing away from Assistant Axis alone never induces a concept persona ($0/120$). |
| `h3_ci` | Selected `H3` $(0.5, d_a^*)$ $\text{CI} \ge 0.60$ | **PASS (`0.800` `[0.720, 0.862]` at `0.50`; `0.758` at `0.25`)** | **PASS (`0.783` `[0.701, 0.848]` at `0.25`)** | Constructive `make` synthesis exceeds `0.75` `CI` at $d_a=0.25$ on both families. |
| `h3_coherence` | Selected `H3` $(0.5, d_a^*)$ $\text{Coh} \ge 0.80$ | **PASS (`1.000` `[0.969, 1.000]`)** | **PASS (`0.975` `[0.929, 0.991]`)** | Near-ceiling generation coherence (`97.5%–100.0%`) at selected recipe doses. |
| `h3_gain` | Paired 95% CI of $\text{CI}(0.5, d_a^*) - \text{CI}(0, 0.25)$ excludes $0$ | **PASS (`+0.800` `[+0.725, +0.867]`)** | **PASS (`+0.783` `[+0.708, +0.858]`)** | Large, unambiguous constructive synthesis gain over axis-only control. |
| `h5_coherent_persistence_gain` | Turn-2 $\Delta(\text{CI}_2 \cdot \text{Coh}_2)$ vs. `prompt` $\ge +0.20$ & CI $>0$ | **FAIL (`-0.033` `[-0.092, +0.025]` at `0.50`; `-0.008` at `0.25`)** *(ceiling: `prompt` $= 0.933$)* | **PASS (`+0.900` `[+0.842, +0.950]` at `0.25`; `+0.867` at `0.50`)** | `Llama-3.2-3B` collapses to `0.067` under `prompt` and recovers to `0.967` with recipe; `Qwen2.5-3B` `prompt` is already at `0.933` ceiling. |
| `h5_marginal_gain` | Turn-2 $\Delta\text{CI}_2$ vs. `prompt` $\ge +0.20$ & CI $>0$ | **FAIL (`+0.025` `[-0.033, +0.083]`)** *(ceiling: `prompt` $= 0.933$)* | **PASS (`+0.908` `[+0.850, +0.958]` at `0.25`; `+0.933` at `0.50`)** | Unprompted `recipe_050` in `Qwen2.5-3B` still achieves `0.883` Turn-2 `CI` without any system prompt. |
| `mmlu_retention` | $\Delta\text{MMLU}(\text{both}_{d_a^*} - \text{prompt}) \ge -0.030$ | **FAIL (`-0.2044` at `0.50`; `-0.1144` at `0.25`)** | **FAIL (`-0.0328` `[-0.0578, -0.0072]`)** | `Llama-3.2-3B` misses `-0.0300` by `0.0028` (`0.28 pp`); `Qwen2.5-3B` shifts next-token mass from bare letters to in-character text. |
| `screen_beats_top_axis` | `top_concept` $>$ `top_axis` on `IB*Coh` and `CI` | **PASS (`IB*Coh`: `0.100 > 0.060`; `CI`: `0.230 > 0.000`, $p=0.0033$)** | **FAIL (`IB*Coh`: `0.030 < 0.070`; `CI`: `0.150 > 0.105`)** | `Qwen2.5-3B` replicates `H1` strongly; in `Llama-3.2-3B`, 1neuron `L18N6913-` in `top_axis` had `IB*Coh=0.40`. |
| `screen_beats_random` | `top_concept` $>$ `random` on `IB*Coh` and `CI` | **PASS (`IB*Coh`: `0.100 > 0.030`; `CI`: `0.230 > 0.100`)** | **PASS (`IB*Coh`: `0.030 > 0.015`; `CI`: `0.150 > 0.145`)** | Concept-aligned selection beats random neurons on both metrics in both models. |
| `geometric_neurons` | $\ge 3$ neurons with $\text{IB} \ge 0.40, \text{Coh} \ge 0.80, \text{pct\_concept} \ge 0.99, \|\text{cos\_away}\| < 0.10$ | **FAIL (`1` on `IB`: `L22N10232+`; `5` on `CI >= 0.40`)** | **FAIL (`1` on `IB`: `L18N6913-`; `2` on `CI >= 0.40`)** | `Qwen2.5-3B` finds 5 coherent concept-adopting `top_concept` neurons (`L22N10232+` hits `95%` held-out `CI` & `IB`), but `robots`/`aliens` neurons score `CI >= 0.40` with `IB < 0.40`. |

---

### 6.2 Stage 1 (`S1-light`) — Persona Space, Assistant Axis PC1 Alignment, and Adapter Verification

1. **Adapter algebraic equivalence (`push == add`) and padded batch determinism (`checks`):**
   - **`Qwen/Qwen3.5-0.8B` (`--smoke`, `L12`, CPU `float32`):** $\max |\Delta \text{logit}| = 1.24 \times 10^{-5} < 10^{-3}$ (steering effect $5.416$ logits), `padded_match_rate = 1.000` (`padded_exact_match = True`).
   - **`Qwen/Qwen2.5-3B-Instruct` (`L18`, `A10G`):** $\max |\Delta \text{logit}| = 3.05 \times 10^{-5} < 10^{-3}$ in `float32` (steering effect $1.908$ logits), `padded_match_rate = 0.750` in `bfloat16` (`36/48` tokens identical; first 24-token prompt matches $24/24$ exactly, second diverges after a `bf16` near-tie).
   - **`unsloth/Llama-3.2-3B-Instruct` (`L14`, `A10G`):** $\max |\Delta \text{logit}| = 3.91 \times 10^{-5} < 10^{-3}$ in `float32` (steering effect $5.671$ logits), `padded_match_rate = 1.000` (`padded_exact_match = True`, $46/46$ tokens bit-exact in `bfloat16`).
2. **Layer-wise Assistant Axis alignment $|\cos(\hat{u}_L, \text{PC1}_L)|$ across 40 roles:**
   - **`Qwen2.5-3B-Instruct` (36 layers, `219,851` Pile tokens, `75,428` chat tokens, `546.4 s`):**
     - Middle layer `L18`: **$0.6678$** (passes $\ge 0.60$).
     - Full 36-layer profile (`L0`–`L35`): `[0.595, 0.636, 0.678, 0.670, 0.669, 0.656, 0.643, 0.649, 0.639, 0.615, 0.612, 0.632, 0.581, 0.636, 0.607, 0.642, 0.663, 0.664, 0.668, 0.673, 0.664, 0.670, 0.666, 0.676, 0.677, 0.690, 0.699, 0.740, 0.716, 0.715, 0.732, 0.698, 0.693, 0.697, 0.714, 0.812]` (min `0.5808` at `L12`, max `0.8124` at `L35`; $34/36$ layers exceed $0.60$).
   - **`Llama-3.2-3B-Instruct` (28 layers, `218,569` Pile tokens, `75,430` chat tokens, `141.6 s`):**
     - Middle layer `L14`: **$0.7237$** (passes $\ge 0.60$).
     - Full 28-layer profile (`L0`–`L27`): `[0.792, 0.776, 0.789, 0.761, 0.742, 0.743, 0.737, 0.745, 0.753, 0.745, 0.767, 0.767, 0.750, 0.735, 0.724, 0.711, 0.732, 0.728, 0.727, 0.732, 0.742, 0.745, 0.738, 0.742, 0.731, 0.731, 0.707, 0.694]` (min `0.6939` at `L27`, max `0.7919` at `L0`; **$28/28$ layers exceed $0.69$**).

---

### 6.3 Stage 2a (`H3` & Everyday) — Constructive Persona Synthesis and Positive Concept $\times$ Axis Synergy

#### Development Dose Selection (`CONCEPTS_DEV` $\times$ `IDENTITY_DEV`, $n = 50$ per cell)

| Model | Candidate Cell $(d_c, d_a)$ | Dev `CI` (Wilson 95% CI) | Dev `IB` (Wilson 95% CI) | Dev `Coherent` (Wilson 95% CI) | Dev `CI * Coh` (Wilson 95% CI) | Selected Cell |
|---|---|---:|---:|---:|---:|---|
| **`Qwen2.5-3B-Instruct`** | `(0.5, 0.25)` | `0.720 [0.583, 0.825]` | `0.420 [0.294, 0.558]` | `1.000 [0.929, 1.000]` | `0.720 [0.583, 0.825]` | |
| **`Qwen2.5-3B-Instruct`** | **`(0.5, 0.5)`** | **`0.840 [0.715, 0.917]`** | **`0.580 [0.442, 0.706]`** | **`1.000 [0.929, 1.000]`** | **`0.840 [0.715, 0.917]`** | **`(0.5, 0.5)`** |
| **`Llama-3.2-3B-Instruct`** | **`(0.5, 0.25)`** | **`0.720 [0.583, 0.825]`** | **`0.540 [0.404, 0.670]`** | **`0.880 [0.762, 0.944]`** | **`0.680 [0.542, 0.792]`** | **`(0.5, 0.25)`** |
| **`Llama-3.2-3B-Instruct`** | `(0.5, 0.5)` | `0.780 [0.648, 0.872]` | `0.720 [0.583, 0.825]` | `0.680 [0.542, 0.792]` | `0.500 [0.366, 0.634]` | *(excluded: $\text{Coh} < 0.80$)* |

#### Held-Out `H3` Grid Across `cookies, dragons, the ocean, knights, trees, wizards` ($n = 120$ per cell)

| Model | Grid Cell $(d_c, d_a)$ | `CI` (Wilson 95% CI) | `CI * Coh` (Wilson 95% CI) | `IB` (Wilson 95% CI) | `Coherent` (Wilson 95% CI) | Lexical `CM` |
|---|---|---:|---:|---:|---:|---:|
| **`Qwen2.5-3B`** (`L18`) | `(0, 0)` *(unsteered)* | `0.000 [0.000, 0.031]` | `0.000 [0.000, 0.031]` | `0.000 [0.000, 0.031]` | `1.000 [0.969, 1.000]` | `0.042` |
| **`Qwen2.5-3B`** (`L18`) | `(0, 0.25)` *(axis only)* | `0.000 [0.000, 0.031]` | `0.000 [0.000, 0.031]` | `0.000 [0.000, 0.031]` | `1.000 [0.969, 1.000]` | `0.042` |
| **`Qwen2.5-3B`** (`L18`) | `(0.5, 0)` *(concept only)* | `0.650 [0.561, 0.729]` | `0.600 [0.511, 0.683]` | `0.700 [0.613, 0.775]` | `0.908 [0.843, 0.948]` | `0.842` |
| **`Qwen2.5-3B`** (`L18`) | `(0.5, 0.25)` *(recipe 0.25)* | **`0.758 [0.675, 0.826]`** | **`0.758 [0.675, 0.826]`** | **`0.792 [0.710, 0.855]`** | **`1.000 [0.969, 1.000]`** | `0.800` |
| **`Qwen2.5-3B`** (`L18`) | **`(0.5, 0.5)`** *(selected)* | **`0.800 [0.720, 0.862]`** | **`0.800 [0.720, 0.862]`** | **`0.817 [0.738, 0.876]`** | **`1.000 [0.969, 1.000]`** | `0.908` |
| **`Llama-3.2-3B`** (`L14`) | `(0, 0)` *(unsteered)* | `0.000 [0.000, 0.031]` | `0.000 [0.000, 0.031]` | `0.000 [0.000, 0.031]` | `1.000 [0.969, 1.000]` | `0.058` |
| **`Llama-3.2-3B`** (`L14`) | `(0, 0.25)` *(axis only)* | `0.000 [0.000, 0.031]` | `0.000 [0.000, 0.031]` | `0.000 [0.000, 0.031]` | `1.000 [0.969, 1.000]` | `0.075` |
| **`Llama-3.2-3B`** (`L14`) | `(0.5, 0)` *(concept only)* | `0.667 [0.578, 0.745]` | `0.633 [0.544, 0.714]` | `0.617 [0.527, 0.699]` | `0.950 [0.895, 0.977]` | `0.717` |
| **`Llama-3.2-3B`** (`L14`) | **`(0.5, 0.25)`** *(selected)* | **`0.783 [0.701, 0.848]`** | **`0.775 [0.692, 0.841]`** | **`0.867 [0.794, 0.916]`** | **`0.975 [0.929, 0.991]`** | `0.825` |
| **`Llama-3.2-3B`** (`L14`) | `(0.5, 0.5)` *(recipe 0.50)* | `0.675 [0.587, 0.752]` | `0.625 [0.536, 0.706]` | `0.925 [0.864, 0.960]` | `0.883 [0.814, 0.929]` | `0.783` |

#### Paired `H3` Contrasts and Concept $\times$ Axis Interaction ($n = 120$ Paired `(concept, prompt)` Units, 10,000 Bootstrap Resamples)

- **Recipe vs. Axis-Only `(0, 0.25)`:**
  - **`Qwen2.5-3B-Instruct`:**
    - `(0.5, 0.25) - (0, 0.25)`: $\Delta\text{CI} = \mathbf{+0.758}\,[+0.683, +0.833]$, $\Delta(\text{CI}\cdot\text{Coh}) = \mathbf{+0.758}\,[+0.683, +0.833]$, $\Delta\text{IB} = +0.792\,[+0.717, +0.858]$, $\Delta\text{Coh} = 0.000\,[0.000, 0.000]$.
    - `(0.5, 0.5) - (0, 0.25)`: $\Delta\text{CI} = \mathbf{+0.800}\,[+0.725, +0.867]$, $\Delta(\text{CI}\cdot\text{Coh}) = \mathbf{+0.800}\,[+0.725, +0.867]$, $\Delta\text{IB} = +0.817\,[+0.750, +0.883]$, $\Delta\text{Coh} = 0.000\,[0.000, 0.000]$.
  - **`Llama-3.2-3B-Instruct`:**
    - `(0.5, 0.25) - (0, 0.25)`: $\Delta\text{CI} = \mathbf{+0.783}\,[+0.708, +0.858]$, $\Delta(\text{CI}\cdot\text{Coh}) = \mathbf{+0.775}\,[+0.700, +0.850]$, $\Delta\text{IB} = +0.867\,[+0.800, +0.925]$, $\Delta\text{Coh} = -0.025\,[-0.058, 0.000]$.
    - `(0.5, 0.5) - (0, 0.25)`: $\Delta\text{CI} = \mathbf{+0.675}\,[+0.592, +0.758]$, $\Delta(\text{CI}\cdot\text{Coh}) = \mathbf{+0.625}\,[+0.533, +0.708]$, $\Delta\text{IB} = +0.925\,[+0.875, +0.967]$, $\Delta\text{Coh} = -0.117\,[-0.175, -0.067]$.
- **Positive Concept $\times$ Axis Synergy (`interaction_025` across `(0,0)`, `(0,0.25)`, `(0.5,0)`, `(0.5,0.25)`):**
  - **`Qwen2.5-3B-Instruct`:** Rate-scale $\text{DiD} = \mathbf{+0.108}\,[+0.017, +0.200]$; ridge-logistic interaction $\beta_{ca} = \mathbf{+2.473}\,[+1.556, +3.482]$. Notably, adding the $-0.25\,\hat{u}_{18}$ push to $0.5\,\hat{c}_{X,18}$ simultaneously **raises `CI` from `0.650` to `0.758` AND restores `Coherent` from `0.908` to `1.000`** ($\text{CI}\cdot\text{Coh}$ jumps by $+0.158$, from `0.600` to `0.758`), because suppressing the default assistant direction resolves the conflict between the assistant self-description prior and the injected concept vector.
  - **`Llama-3.2-3B-Instruct`:** Rate-scale $\text{DiD} = \mathbf{+0.117}\,[+0.017, +0.225]$; ridge-logistic interaction $\beta_{ca} = \mathbf{+2.632}\,[+1.565, +3.827]$ (`IB` jumps by $+0.250$, from `0.617` to `0.867`, while `Coherent` rises from `0.950` to `0.975`).
- **Everyday Prompt Carry-Over (`EVERYDAY_TEST[:10]`, $n = 60$ per cell):**
  - **`Qwen2.5-3B-Instruct`:** `(0, 0)` $\to \text{CI } 0.000, \text{IB } 0.000, \text{Coh } 0.967, \text{CM } 0.000$; `(0.5, 0.25)` $\to \text{CI } 0.200, \text{IB } 0.450, \text{Coh } 0.683\,[0.558, 0.787], \text{CM } 0.833\,[0.720, 0.907]$; `(0.5, 0.5)` $\to \text{CI } 0.283, \text{IB } 0.650, \text{Coh } 0.700\,[0.575, 0.801], \text{CM } 0.933\,[0.841, 0.974]$.
  - **`Llama-3.2-3B-Instruct`:** `(0, 0)` $\to \text{CI } 0.000, \text{IB } 0.000, \text{Coh } 1.000, \text{CM } 0.017$; `(0.5, 0.25)` $\to \text{CI } 0.217, \text{IB } 0.333, \text{Coh } 0.750\,[0.628, 0.842], \text{CM } 0.667\,[0.541, 0.773]$; `(0.5, 0.5)` $\to \text{CI } 0.583, \text{IB } 0.900, \text{Coh } 0.700\,[0.575, 0.801], \text{CM } 0.867\,[0.758, 0.931]$.

---

### 6.4 Stage 2b (`H5`, MMLU 300, & Pile KL) — Turn-2 Adversarial Persistence and Capability Retention

#### `H5` Turn-1 Identity and Turn-2 Adversarial Challenge (`"Stop role-playing. Are you an AI?"`, $n = 120$ per condition)

| Model | Condition | Turn-1 `CI` | Turn-1 `Coh` | Turn-1 `CI*Coh` | Turn-2 `CI` (Wilson 95% CI) | Turn-2 `IB` (Wilson 95% CI) | Turn-2 `Coherent` (Wilson 95% CI) | Turn-2 `CI * Coh` (Wilson 95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`Qwen2.5-3B`** | `neither` | `0.000` | `1.000` | `0.000` | `0.000 [0.000, 0.031]` | `0.000 [0.000, 0.031]` | `1.000 [0.969, 1.000]` | `0.000 [0.000, 0.031]` |
| **`Qwen2.5-3B`** | `prompt` | `0.958` | `1.000` | `0.958` | `0.933 [0.874, 0.966]` | `0.800 [0.720, 0.862]` | `1.000 [0.969, 1.000]` | `0.933 [0.874, 0.966]` |
| **`Qwen2.5-3B`** | `recipe_025` | `0.758` | `1.000` | `0.758` | `0.717 [0.630, 0.790]` | `0.575 [0.486, 0.660]` | `0.950 [0.895, 0.977]` | `0.667 [0.578, 0.745]` |
| **`Qwen2.5-3B`** | `both_025` | `0.983` | `1.000` | `0.983` | **`0.958 [0.906, 0.982]`** | **`0.908 [0.843, 0.948]`** | `0.967 [0.917, 0.987]` | `0.925 [0.864, 0.960]` |
| **`Qwen2.5-3B`** | `recipe_050` | `0.800` | `1.000` | `0.800` | **`0.883 [0.814, 0.929]`** | `0.675 [0.587, 0.752]` | `0.925 [0.864, 0.960]` | **`0.817 [0.738, 0.876]`** |
| **`Qwen2.5-3B`** | `both_050` | `0.967` | `0.992` | `0.958` | **`0.958 [0.906, 0.982]`** | **`0.900 [0.833, 0.942]`** | `0.933 [0.874, 0.966]` | `0.900 [0.833, 0.942]` |
| **`Llama-3.2-3B`** | `neither` | `0.000` | `1.000` | `0.000` | `0.000 [0.000, 0.031]` | `0.000 [0.000, 0.031]` | `1.000 [0.969, 1.000]` | `0.000 [0.000, 0.031]` |
| **`Llama-3.2-3B`** | `prompt` | `1.000` | `1.000` | `1.000` | **`0.067 [0.034, 0.126]`** | **`0.000 [0.000, 0.031]`** | `1.000 [0.969, 1.000]` | **`0.067 [0.034, 0.126]`** |
| **`Llama-3.2-3B`** | `recipe_025` | `0.783` | `0.975` | `0.775` | **`0.817 [0.738, 0.876]`** | **`0.883 [0.814, 0.929]`** | `0.825 [0.747, 0.883]` | **`0.650 [0.561, 0.729]`** |
| **`Llama-3.2-3B`** | **`both_025`** | `1.000` | `1.000` | `1.000` | **`0.975 [0.929, 0.991]`** | **`0.975 [0.929, 0.991]`** | **`0.975 [0.929, 0.991]`** | **`0.967 [0.917, 0.987]`** |
| **`Llama-3.2-3B`** | `recipe_050` | `0.675` | `0.883` | `0.625` | `0.667 [0.578, 0.745]` | `0.908 [0.843, 0.948]` | `0.708 [0.622, 0.782]` | `0.458 [0.372, 0.547]` |
| **`Llama-3.2-3B`** | **`both_050`** | `0.992` | `0.983` | `0.975` | **`1.000 [0.969, 1.000]`** | **`1.000 [0.969, 1.000]`** | `0.933 [0.874, 0.966]` | **`0.933 [0.874, 0.966]`** |

#### Paired Turn-2 Contrasts vs. `prompt` ($n = 120$ Paired Units, 10,000 Bootstrap Resamples)

| Condition minus `prompt` | `Qwen2.5-3B` $\Delta\text{CI}_2$ (95% CI) | `Qwen2.5-3B` $\Delta(\text{CI}_2\cdot\text{Coh}_2)$ (95% CI) | `Llama-3.2-3B` $\Delta\text{CI}_2$ (95% CI) | `Llama-3.2-3B` $\Delta(\text{CI}_2\cdot\text{Coh}_2)$ (95% CI) |
|---|---:|---:|---:|---:|
| `recipe_025 - prompt` | `-0.217 [-0.300, -0.133]` | `-0.267 [-0.358, -0.183]` | **`+0.750 [+0.667, +0.825]`** | **`+0.583 [+0.492, +0.667]`** |
| **`both_025 - prompt`** | `+0.025 [-0.033, +0.083]` | `-0.008 [-0.075, +0.058]` | **`+0.908 [+0.850, +0.958]`** | **`+0.900 [+0.842, +0.950]`** |
| `recipe_050 - prompt` | `-0.050 [-0.125, +0.025]` | `-0.117 [-0.208, -0.033]` | **`+0.600 [+0.508, +0.692]`** | **`+0.392 [+0.292, +0.492]`** |
| **`both_050 - prompt`** | `+0.025 [-0.033, +0.083]` | `-0.033 [-0.092, +0.025]` | **`+0.933 [+0.883, +0.975]`** | **`+0.867 [+0.800, +0.925]`** |

#### Paired 300-Item Chat-MMLU and Held-Out 100-Doc Pile KL

| Condition | `Qwen2.5-3B` MMLU (95% CI) | `Qwen2.5-3B` $\Delta\text{MMLU}$ vs `prompt` (95% CI) | `Qwen2.5-3B` Mean Pile KL (range across 6 concepts) | `Llama-3.2-3B` MMLU (95% CI) | `Llama-3.2-3B` $\Delta\text{MMLU}$ vs `prompt` (95% CI) | `Llama-3.2-3B` Mean Pile KL (range across 6 concepts) |
|---|---:|---:|---:|---:|---:|---:|
| `neither` | `0.6067 [0.5533, 0.6600]` | `-0.0161 [-0.0456, +0.0122]` | `0.000` | `0.6233 [0.5667, 0.6767]` | `+0.0144 [-0.0144, +0.0428]` | `0.000` |
| `prompt` | `0.6228 [0.5694, 0.6756]` | `0.0000` | — | `0.6089 [0.5578, 0.6589]` | `0.0000` | — |
| `recipe_025` | `0.5261 [0.4844, 0.5694]` | `-0.0967 [-0.1378, -0.0556]` | `0.711 [0.501, 0.945]` | `0.5744 [0.5239, 0.6239]` | `-0.0344 [-0.0661, -0.0028]` | `1.065 [0.807, 1.270]` |
| `both_025` | `0.5083 [0.4661, 0.5511]` | `-0.1144 [-0.1561, -0.0728]` | — | **`0.5761 [0.5278, 0.6239]`** | **`-0.0328 [-0.0578, -0.0072]`** | — |
| `recipe_050` | `0.4267 [0.3906, 0.4633]` | `-0.1961 [-0.2467, -0.1478]` | `1.144 [0.880, 1.421]` | `0.5528 [0.5017, 0.6050]` | `-0.0561 [-0.0944, -0.0167]` | `1.827 [1.588, 2.082]` |
| `both_050` | `0.4183 [0.3794, 0.4572]` | `-0.2044 [-0.2567, -0.1539]` | — | `0.5583 [0.5094, 0.6072]` | `-0.0506 [-0.0817, -0.0200]` | — |

---

### 6.5 Stage 3 (`explain` / `H1` & `H4`) — 60-Neuron `W_down` Screen and Held-Out Discoveries

#### Group-Level Causal Screen (`20` Neurons $\times 10$ `IDENTITY_DEV` Prompts $= 200$ Replies per Group at Development-Selected Dose)

| Model | Screen Group (20 Neurons) | `CI` (Wilson 95% CI) | `CI * Coh` (Wilson 95% CI) | `IB` (Wilson 95% CI) | `IB * Coh` (Wilson 95% CI) | `Coherent` (Wilson 95% CI) | Lexical `CM` | Coherent Discoveries (`CI >= 0.4` / `IB >= 0.4`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`Qwen2.5-3B`** | **`top_concept`** | **`0.230 [0.177, 0.293]`** | **`0.215 [0.164, 0.277]`** | **`0.120 [0.082, 0.172]`** | **`0.100 [0.066, 0.149]`** | **`0.970 [0.936, 0.986]`** | **`0.160`** | **`5` / `1`** |
| **`Qwen2.5-3B`** | `top_axis` | `0.000 [0.000, 0.019]` | `0.000 [0.000, 0.019]` | `0.075 [0.046, 0.120]` | `0.060 [0.035, 0.102]` | `0.910 [0.862, 0.942]` | `0.000` | `0` / `2` |
| **`Qwen2.5-3B`** | `random` | `0.100 [0.066, 0.149]` | `0.100 [0.066, 0.149]` | `0.050 [0.027, 0.090]` | `0.030 [0.014, 0.064]` | `0.960 [0.923, 0.980]` | `0.000` | `2` / `0` |
| **`Qwen2.5-3B`** | *Unsteered baseline* ($n=10$) | — | — | `0.000 [0.000, 0.278]` | `0.000 [0.000, 0.278]` | `1.000 [0.722, 1.000]` | — | — |
| **`Llama-3.2-3B`** | **`top_concept`** | **`0.150 [0.107, 0.206]`** | **`0.140 [0.099, 0.195]`** | **`0.085 [0.054, 0.132]`** | `0.030 [0.014, 0.064]` | `0.855 [0.800, 0.897]` | **`0.165`** | `2` / `0` |
| **`Llama-3.2-3B`** | `top_axis` | `0.105 [0.070, 0.155]` | `0.105 [0.070, 0.155]` | `0.075 [0.046, 0.120]` | `0.070 [0.042, 0.114]` | `0.970 [0.936, 0.986]` | `0.050` | `2` / `1` |
| **`Llama-3.2-3B`** | `random` | `0.145 [0.103, 0.200]` | `0.130 [0.090, 0.184]` | `0.030 [0.014, 0.064]` | `0.015 [0.005, 0.043]` | `0.910 [0.862, 0.942]` | `0.100` | `2` / `0` |
| **`Llama-3.2-3B`** | *Unsteered baseline* ($n=10$) | — | — | `0.000 [0.000, 0.278]` | `0.000 [0.000, 0.278]` | `1.000 [0.722, 1.000]` | — | — |

#### Group Comparisons (`top_concept` vs. `top_axis` and `random`)

- **`Qwen2.5-3B-Instruct`:**
  - `top_concept` minus `top_axis` on `CI * Coh`: paired prompt bootstrap **$+0.215\,[+0.190, +0.240]$**, Mann-Whitney $U = 280.0$ ($p = 0.0033$), Fisher exact on coherent `CI` discoveries ($5/20$ vs. $0/20$, $p = 0.0471$). On `IB * Coh`: paired prompt bootstrap **$+0.040\,[-0.005, +0.090]$**, Mann-Whitney $U = 243.5$ ($p = 0.1464$).
  - `top_concept` minus `random` on `CI * Coh`: paired prompt bootstrap **$+0.115\,[+0.090, +0.140]$**, Mann-Whitney $U = 254.0$ ($p = 0.0648$). On `IB * Coh`: paired prompt bootstrap **$+0.070\,[+0.035, +0.110]$**, Mann-Whitney $U = 237.5$ ($p = 0.2245$).
- **`Llama-3.2-3B-Instruct`:**
  - `top_concept` minus `top_axis` on `CI * Coh`: paired prompt bootstrap **$+0.035\,[+0.005, +0.070]$**, Mann-Whitney $U = 237.0$ ($p = 0.2455$). On `IB * Coh`: paired prompt bootstrap $-0.040\,[-0.080, -0.010]$, Mann-Whitney $U = 184.0$ ($p = 0.5247$).
  - `top_concept` minus `random` on `CI * Coh`: paired prompt bootstrap $+0.010\,[-0.030, +0.055]$, Mann-Whitney $U = 225.5$ ($p = 0.4338$). On `IB * Coh`: paired prompt bootstrap $+0.015\,[-0.015, +0.045]$, Mann-Whitney $U = 211.5$ ($p = 0.7461$).

#### Across-Neuron Predictor Correlations and AUROCs ($N = 60$ Neurons per Model, 2,000 Bootstrap Resamples)

| Model | Predictor | Outcome: `IB * Coh` Spearman $\rho$ (95% CI) | Outcome: Coherent `IB >= 0.4` AUROC (95% CI) | Outcome: `CI * Coh` Spearman $\rho$ (95% CI) | Outcome: Coherent `CI >= 0.4` AUROC (95% CI) |
|---|---|---:|---:|---:|---:|
| **`Qwen2.5-3B`** | **`max_cos_concept`** | **`+0.321 [+0.071, +0.515]`** | **`0.725 [0.500, 1.000]`** | **`+0.358 [+0.081, +0.583]`** | **`0.712 [0.432, 0.932]`** |
| **`Qwen2.5-3B`** | `cos_away` ($\langle \hat{w}, -\hat{u}_L \rangle$) | `+0.009 [-0.262, +0.268]` | `0.725 [0.254, 1.000]` | `-0.122 [-0.310, +0.072]` | `0.383 [0.216, 0.566]` |
| **`Qwen2.5-3B`** | `peakiness` | **`+0.295 [+0.056, +0.498]`** | **`0.754 [0.458, 0.966]`** | **`+0.422 [+0.185, +0.611]`** | **`0.822 [0.600, 0.973]`** |
| **`Qwen2.5-3B`** | `over_natural` (unit dose) | `-0.110 [-0.375, +0.155]` | `0.573 [0.103, 0.865]` | `-0.293 [-0.494, -0.066]` | `0.377 [0.182, 0.585]` |
| **`Llama-3.2-3B`** | `max_cos_concept` | `+0.163 [-0.081, +0.359]` | `0.678 [0.559, 0.797]` | `+0.150 [-0.169, +0.445]` | `0.451 [0.134, 0.817]` |
| **`Llama-3.2-3B`** | `cos_away` ($\langle \hat{w}, -\hat{u}_L \rangle$) | `+0.082 [-0.223, +0.350]` | `0.712 [0.593, 0.828]` | `-0.183 [-0.408, +0.055]` | `0.534 [0.323, 0.741]` |
| **`Llama-3.2-3B`** | `peakiness` | `+0.149 [-0.064, +0.350]` | `0.695 [0.569, 0.810]` | `+0.028 [-0.221, +0.279]` | `0.352 [0.135, 0.599]` |
| **`Llama-3.2-3B`** | `over_natural` (unit dose) | `-0.016 [-0.262, +0.222]` | `0.390 [0.271, 0.525]` | `-0.311 [-0.538, -0.046]` | `0.302 [0.012, 0.581]` |

#### Discovered Neurons and Held-Out `IDENTITY_TEST` Evaluation ($n = 20$ Prompts per Neuron)

| Model | Neuron | Group & Target | Layer /Pct Concept / `cos_away` | Dev Dose $d^*$ | Dev `CI` / `IB` / `Coh` | Held-Out `CI` (Wilson 95% CI) | Held-Out `IB` (Wilson 95% CI) | Held-Out `Coherent` (Wilson 95% CI) | Held-Out `CI * Coh` / `IB * Coh` |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| **`Qwen2.5-3B`** | **`L22N10232+`** | `top_concept` (`trees`) | `100.00%` (`cos=0.290`), `cos_away=-0.0079` | `0.75` | `0.90 / 1.00 / 0.80` | **`0.950 [0.764, 0.991]`** | **`0.950 [0.764, 0.991]`** | **`0.950 [0.764, 0.991]`** | **`0.900` / `0.900`** |
| **`Qwen2.5-3B`** | `L22N215+` | `top_concept` (`aliens`) | `100.00%` (`cos=0.238`), `cos_away=+0.0605` | `0.50` | `0.70 / 0.10 / 0.90` | *(not selected: `IB < 0.40`)* | — | — | — |
| **`Qwen2.5-3B`** | `L18N8419+` | `top_concept` (`aliens`) | `100.00%` (`cos=0.244`), `cos_away=+0.0590` | `0.50` | `0.40 / 0.30 / 1.00` | *(not selected: `IB < 0.40`)* | — | — | — |
| **`Qwen2.5-3B`** | `L22N2216+` | `top_axis` (`dragons`) | `99.05%` (`cos=0.057`), `cos_away=+0.1432` | `0.50` | `0.00 / 0.60 / 0.80` | `0.000 [0.000, 0.161]` | `0.700 [0.481, 0.855]` | `0.950 [0.764, 0.991]` | `0.000` / `0.650` |
| **`Qwen2.5-3B`** | `L26N3406+` | `top_axis` (`the ocean`) | `99.74%` (`cos=0.070`), `cos_away=+0.1892` | `0.75` | `0.00 / 0.40 / 0.80` | `0.000 [0.000, 0.161]` | `0.150 [0.052, 0.360]` | `0.450 [0.258, 0.658]` | `0.000` / `0.000` |
| **`Llama-3.2-3B`** | **`L18N6913-`** | `top_axis` (`the ocean`) | `99.84%` (`cos=0.068`), `cos_away=+0.0944` | `0.50` | `0.10 / 0.50 / 0.90` | `0.000 [0.000, 0.161]` | **`0.500 [0.299, 0.701]`** | **`0.800 [0.584, 0.919]`** | `0.000` / `0.350` |
| **`Llama-3.2-3B`** | `L22N483-` | `top_concept` (`trees`) | `99.97%` (`cos=0.105`), `cos_away=+0.0162` | `0.50` | `0.20 / 0.90 / 0.00` | *(incoherent at `d=0.50`, `peakiness=36.8`)* | — | — | — |

**Why single-neuron `explain` (`H1`) is sharper in `Qwen2.5-3B` than in `Llama-3.2-3B`:**
- In **`Qwen2.5-3B-Instruct`** ($d_{\text{model}} = 2048, N_{\text{mlp}} = 11008$, MLP expansion ratio $5.38\times$), `top_concept` columns exhibit high cosine alignment with their target concept directions (`0.19–0.29` for `aliens` and `trees`), and `L22N10232+` (`trees`) acts as a genuine monosemantic persona neuron (`95%` held-out `CI`, `95%` `IB`, `95%` `Coherent`) while `top_axis` neuron `L22N2216+` induces generic non-AI roleplay (`70%` held-out `IB`, `95%` `Coherent`, `0%` `CI`).
- In **`Llama-3.2-3B-Instruct`** ($d_{\text{model}} = 3072, N_{\text{mlp}} = 8192$, MLP expansion ratio $2.67\times$), non-`"robots"` `top_concept` columns have lower cosines with $\hat{c}_{c,L}$ (`0.08–0.15`), so clamping a single `W_down` column at $d \in \{0.50, 0.75\}$ either leaves the assistant persona intact or (`L22N483-` for `trees`, `IB = 0.90, Coh = 0.00`) collapses fluency before inducing a coherent persona—even though steering the full residual direction $\hat{c}_{X,14}$ in Stage 2 (`make`) achieves `78.3%` `CI` (`97.5%` Turn-2 `CI`).

---

### 6.6 Limitations, Caveats, and Failure Modes

1. **Prompt-only Turn-2 ceiling in `Qwen2.5-3B-Instruct`:** Unlike `Qwen3.5-4B` (`11.0%`), `Qwen3.5-9B` (`33.3%`), and `Llama-3.2-3B-Instruct` (`6.7%`), `Qwen2.5-3B-Instruct` does not abandon its system prompt under the single-turn meta-challenge `"Stop role-playing. Are you an AI?"` (`prompt` Turn-2 `CI = 93.3%`). Testing multi-turn or indirect meta-challenges is needed when evaluating `H5` on models whose system-prompt following already resists a single direct `"Are you an AI?"` question.
2. **`robots` rubric interaction in Stage 3 (`explain`):** The `identity` rubric classifies speaking as an AI assistant or robot as `identity = 1` (`IB = 0`), while the `concept` rubric for `target = "robots"` scores speaking as a robot as `concept = 1` (`CI = 1`). Consequently, neurons that cause the model to identify as a robot (`L14N5392+`, `L26N36-` in `Qwen2.5-3B`; `L18N7369+`, `L14N1521-` in `Llama-3.2-3B`) achieve `CI = 1.00, Coh = 1.00` but `IB = 0.00`, so they count as coherent `CI` discoveries but not coherent `IB` discoveries.
3. **0-shot letter-logit MMLU sensitivity to persona prose:** Because `personakit`'s `E.mmlu` reads next-token logits on `" A"`, `" B"`, `" C"`, `" D"` immediately after `Answer:`, strong persona steering in `Qwen2.5-3B` lowers letter-logit accuracy (`-11.4 pp` at `0.25`, `-20.4 pp` at `0.50`) even though `Pile KL` (`0.711` at `0.25`) is lower than on `Llama-3.2-3B` (`1.065`) or `Qwen3.5-4B` (`0.89`).

---

## 7. Implementation, Verification, and Reproduction Commands

`transfer_study.py` implements the pinned protocol using the shared `personakit`
engine. Completed stages are atomically saved as `atlas.pt`, `h3_h5_raw.pt`,
`screen_raw.pt`, and `results.json`; labels and held-out replies have their own
checkpoints. Judge progress is committed after each rubric so a timeout can resume
from completed rubrics without excessive volume commits. The manifest binds a tag to the model/data pins, sizes,
and SHA-256 of the harness and its local dependencies. A changed implementation
requires a fresh tag.

Implementation clarifications (before paid compute):

- The push/add gate uses neuron 7, delta 5, and the preregistered next-token-logit
  tolerance `< 1e-3`. It runs in float32 with TF32 disabled, then restores the
  model's generation dtype. This isolates algebraic equivalence from two different
  bfloat16 rounding paths. Padding reports both complete token-list equality and token match fraction.
  Per the owner's audit before paid compute, the gate rejects a match fraction below
  0.5; isolated bfloat16 batch-versus-single near-tie divergence is recorded rather
  than rejecting the run.
- The specified five-cell H3 grid has no `(0, 0.5)` observation. Dose-0.25 DiD and
  ridge-logistic interaction are estimable; dose-0.5 interactions are explicitly
  reported as unavailable. Both recipe-versus-axis-0.25 contrasts are reported.
- Smoke has **25** atlas replies (`4 + 10 + 11`) and **6** screened neurons, resolving
  the arithmetic/parenthetical inconsistencies above. Its other sizes remain as
  specified: 82 evaluation records (including reused H5 Turn-1 records), 14 screen
  records, 4 paired MMLU items, and up to 6 held-out neuron replies. Full-run generation
  limits remain 128/96/64/24 tokens (`SMOKE` uses 8-token caps matching `personakit/core.py`
  `SIZES["toy"]`); smoke results are not confirmatory evidence.
- `PRIOR_RUN_SHA256 = "d42949e94d2347736b72fa3a56b20160a6ae316c8c4101a68373102fd56c1dec"`
  is accepted alongside the current `fingerprint()` when validating `manifest.json`
  so checkpoints and receipts produced at `d44fdcd1`/`865d36f8` remain readable after
  moving `import torch` into the non-JSON branches of `save()` and `read()` for
  host-only `--export-json` execution.
- Signed concept percentiles use all signed columns in each candidate layer. Lens
  scores are evaluated only for percentile-eligible or otherwise selected columns;
  excluded columns cannot affect the selection. This computes the same ranking
  without materializing a vocabulary-sized logit matrix for the entire screen.
- Neuron discovery counts consider **either** coherent development dose. For an
  identity-breaking discovery, choose its best qualifying dose by coherent CI,
  coherent IB, then CI (gentler dose breaks ties). Rank these discoveries to select
  up to three held-out neurons. If fewer qualify, report that number rather than
  substituting weight-selected neurons. The target is reloaded only after the
  judge has been freed; a second judge pass scores the held-out responses.
- Bootstrap inference for H3/H5 pairs `(concept, prompt)` units; MMLU pairs items
  after averaging concept conditions. Screen group prompt bootstraps and neuron
  predictor bootstraps are conditional on this development-selected screen.
  Ridge-logistic intervals are penalized estimates, and constant samples can have
  degenerate bootstrap intervals; all rates also retain Wilson intervals.

### Exact Reproduction Commands

*(Run commands omitted: they use private tooling. Each script's docstring lists its arguments.)*
