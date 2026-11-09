# Week 2 report: entropic OT pairing, InfoNCE actions, retrieval of held-out cells

Builds on Week 1 (`REPORT.md`): same processed data, splits and frozen state embeddings. Nothing here touches the 127 test doubles. Modules: `src/pertact/ot.py`, `infonce.py`, `retrieval.py`; settings under `ot:` and `infonce:` in `config.yaml`.

**Design decisions (agreed before the runs)**
* **OT-partner positives:** each control's positive is its own OT partner, and the other cells of the same perturbation in the batch are negatives.
* **Actions:** one free vector per gene.
* **State spaces:** both PCA-50 (`X_pca_g`) and scVI-30 (`X_scvi30`).

## 1. Method

### Entropic OT pairing (Sinkhorn-Knopp)
* **What:** for each single perturbation and gem group, couple that group's control cells to the perturbation's cells. The coupling minimises total squared displacement in the state space, minus ε × entropy, with uniform weights on both sides.
* **Solver:** Sinkhorn-Knopp scaling in the log domain (dual potentials), stopping when the L1 marginal error is below 1e-6.
* **Splits:** training and validation cells are paired separately, each with its own split's controls.
* **Blur:** ε = 0.05 × the median squared distance between control cells. An earlier version scaled ε by the control-to-perturbed cost, which made strongly perturbed genes blurrier simply because they moved further. It was changed before any real-data run.
* **Output:** for every perturbed cell, its 5 most-coupled controls and their renormalised weights (`results/ot/<space>__<split>.npz`).
* **Label-free variant:** each gem group's controls are coupled to all of its single-perturbed cells at once (`__pooled.npz`), used wherever the perturbation must not be known in advance.

### InfoNCE actions
* **Model:** `s(x_pert) ≈ s(x_ctrl) + f(a)`, with f(a) a free vector per training gene in the state space.
* **Batch:** 32 perturbations × 8 training cells. Each cell takes one control partner, drawn from its OT weights.
* **Score and loss:**
  * Query `q = s(x_ctrl) + f(a)`.
  * Score `−‖q − s(x_pert)‖² / (τ·D₀)`, where D₀ is the mean squared distance between random training cells, so τ means the same in both spaces.
  * Loss: cross-entropy toward the diagonal, rows and columns averaged.
* **Difficult negatives:** the 7 other cells of the same perturbation share f(a), so only the carried-over control state can separate them.
* **Training:** Adam (lr 0.05), validation loss checked every 100 steps, patience 10. τ is chosen from {0.01, 0.03, 0.1, 0.3} by validation instance-retrieval MRR (seed 0): **0.03 for PCA-50, 0.1 for scVI-30**. Then 3 seeds per space.
* **Speed:** training takes seconds, 1,400–3,800 steps.

### Retrieval tests (validation cells: the held-out 20% of every single)
* **Instance retrieval:** the query is the cell's per-perturbation OT partner + f(a), ranked against all 9,444 held-out perturbed cells. Reported: MRR and recall@k of the cell itself, plus `same_pert@10` (the share of the 10 nearest cells from the right perturbation).
* **Identification:** the start is the label-free pooled-OT partner, or the gem group's mean control. All 105 actions plus "no perturbation" (f = 0) are tried (chance top-1 = 1/106). Reported overall and by effect-strength tertile (mean-shift length).
* **References:**
  * **mean shift:** the average OT displacement on training pairs, closed form;
  * **zero:** f = 0;
  * **shuffled:** mean-shift actions permuted across genes.
* **Controls added after the first results:**
  * **InfoNCE calibrated:** all actions × one scalar α, fitted by least squares on training pairs;
  * **InfoNCE direction at mean-shift length**, and **mean-shift direction at InfoNCE length.**

### Mean shift vs InfoNCE: two ways to turn the same OT pairs into an action
Entropic OT does not produce an action. It only says which control cells each perturbed cell most likely came from. The mean shift and InfoNCE are two ways of turning those same pairs into one vector f(a) per perturbation. Same pairs, same state spaces, same splits; only the objective differs. The numbers below come from sections 2 and 5 (validation cells, PCA-50 / scVI-30).

| | mean shift | InfoNCE |
|---|---|---|
| **How it reads the OT pairs** | Averages each perturbation's pair displacements: f(a) = mean over its training cells of (perturbed state − OT-weighted control state). | Uses every pair as a training example, scored against every other perturbed cell in the batch. |
| **Objective** | Pull only. It is the exact least-squares solution of min Σ‖control + f(a) − partner‖². | Pull toward the partner and push away from every other cell (contrastive). |
| **Uses other perturbations?** | No. Each perturbation is solved on its own. | Yes. Other perturbations' cells are the easy negatives. |
| **Training** | None: closed form, deterministic. | Gradient training, temperature τ, 3 seeds (seconds per run). |
| **Length** | The average displacement itself, so right by definition. | Unconstrained: median 2.1× / 2.4× too long, about 4× for weak perturbations. Needs a per-action rescale. |
| **Directions** | Share one large common direction: median pairwise cosine 0.49 / 0.22, about 3 / 6 effective dimensions. | Spread apart: 0.14 / 0.04, about 13 effective dimensions. |
| **Find the held-out cell (MRR)** | 0.120 / 0.133 | 0.181 / 0.174 as trained; 0.114 / 0.125 at the mean shift's lengths. |
| **Name the perturbation (top-1)** | 0.235 / 0.191 | 0.095 / 0.093 as trained; 0.268 / 0.190 with a per-action scale. |
| **Week 1 decoder, direction only** | 0.961 / 0.962 | 0.959 / 0.965 |
| **Action for a gene with no training cells** | Impossible: it needs that gene's cells. | Not as built (one free vector per gene). The loss would train f(a) computed from gene features unchanged. |
| **Action that depends on the starting cell** | No: one average per perturbation. | Possible: f(a, state) trains with the same loss. |

**When to use which**
* **Mean shift:**
  * when the action's length must be right, as when adding actions in the state space (f(a) + f(b) for a double);
  * as the baseline any learned action has to beat;
  * when you want a deterministic, interpretable answer with nothing to tune.
* **InfoNCE:**
  * when the actions must be told apart, as in identification (after rescaling) and held-out-cell retrieval;
  * when the action has to be computed rather than looked up: from gene features, to reach genes with no cells, or from the starting state. The mean shift has no version of these.
* **Both:** InfoNCE's direction times a per-action scale fitted on the OT pairs. That scale is the projection of the mean shift onto f(a) (section 5), so the combination takes its direction from InfoNCE and its length from the data.

**Why we are trying InfoNCE here**
* **The mean shift only pulls.** It is what a pure squared-error ("reconstruction") objective gives for a fixed per-perturbation shift. It keeps everything a perturbation's cells share, including a large response common to many perturbations, and nothing pushes perturbations apart. InfoNCE scores an action by whether it lands on the right cell rather than on other perturbations' cells. That should make it keep what is specific to each perturbation. The direction spread above shows it does spread actions apart.
* **It tests the "state + action" model cell by cell.** The difficult negatives share f(a), so the only way to rank the right cell first is to carry the control's own state over. The mean shift is fitted to averages, so it never has to.
* **It is a learning objective, so it extends where a closed form cannot.** Two such cases are actions computed from gene features, for genes never perturbed in training, and actions that depend on the starting state. A contrastive loss trains either one as it stands. That is the main reason to build it now.
* **Whether it pays off is still open.** At matched lengths it ties the mean shift on every test so far. The composition test on held-out doubles decides.

## 2. Research questions

**Q1. Does entropic OT give meaningful pairs?**
* *Answer: yes.*
* *Checks:*
  * Agrees with the POT library to 1e-8 relative difference.
  * On a synthetic cloud shifted by a known vector, the true source cell is the top partner for 100% of cells.
  * Translating the controls leaves the plan unchanged, as it must for a quadratic cost.
* *Real data:* all 3,348 problems converged (marginal error ≤ 1e-6, 70–140 median iterations). OT partners are about half as far as random same-gem controls:

| space | split | partner cost: OT vs random | controls per cell (perplexity) |
|---|---|---|---|
| PCA-50 | train | 56.6 vs 109.4 | 75 |
| PCA-50 | val | 63.3 vs 104.6 | 33 |
| scVI-30 | train | 14.5 vs 30.6 | 60 |
| scVI-30 | val | 16.7 vs 29.5 | 30 |

* *Note:* each perturbed cell spreads over 30–75 controls, so the coupling is fairly blurry; ε is the knob for sharper pairs. Validation pools are smaller, hence lower perplexity.

**Q2. Do the InfoNCE actions retrieve held-out cells of seen singles?**
* *Answer: yes, better than every reference.*
* *Evidence (validation instance retrieval, 3 seeds):*

| space | actions | MRR | recall@1 | recall@10 | recall@100 | same_pert@10 |
|---|---|---|---|---|---|---|
| PCA-50 | no action | 0.026 | 0.008 | 0.051 | 0.244 | 0.016 |
| PCA-50 | shuffled | 0.023 | 0.007 | 0.047 | 0.228 | 0.016 |
| PCA-50 | mean shift | 0.120 | 0.059 | 0.232 | 0.571 | 0.115 |
| PCA-50 | **InfoNCE** | **0.181 ± 0.002** | **0.101** | **0.342** | **0.670** | **0.255** |
| scVI-30 | no action | 0.031 | 0.009 | 0.064 | 0.312 | 0.017 |
| scVI-30 | shuffled | 0.026 | 0.007 | 0.052 | 0.280 | 0.016 |
| scVI-30 | mean shift | 0.133 | 0.066 | 0.266 | 0.652 | 0.126 |
| scVI-30 | **InfoNCE** | **0.174 ± 0.002** | **0.092** | **0.339** | **0.706** | **0.245** |

* *Generalisation:* the gap from training to held-out cells is small: MRR 0.209 → 0.181 (PCA) and 0.190 → 0.174 (scVI).

**Q3. Does that gain come from better directions or from longer actions?**
* *Answer: mostly length.*
* *Evidence:*
  * InfoNCE actions point where the mean shift does (median cosine 0.80 PCA, 0.86 scVI; 0.88 / 0.95 for strong perturbations), but are 2.1× (PCA) and 2.4× (scVI) longer. Weak perturbations are 3.8–4.2× longer.
  * Giving the mean shift InfoNCE's per-gene lengths reaches same_pert@10 0.227 / 0.249 (InfoNCE 0.255 / 0.245) and MRR 0.135 / 0.166 (InfoNCE 0.181 / 0.174).
  * Giving InfoNCE directions the mean-shift lengths drops MRR to 0.114 / 0.125.
  * So in PCA about 0.05 MRR is genuinely directional; in scVI nearly all of the gain is length.
* *Why:* InfoNCE only fixes relative distances. Stepping further along the action's direction moves the query away from other perturbations' cells, most of which sit near the control cloud, faster than it moves away from its own partner. The loss therefore rewards overshooting.

**Q4. Can the actions identify which perturbation a held-out cell received (label-free start)?**
* *Answer: only once length is fixed; then as well as or better than the mean shift.*
* *Evidence (top-1 / top-5, pooled-OT start, chance 0.009 / 0.047):*

| space | mean shift | InfoNCE | InfoNCE calibrated (α ≈ 0.42) | InfoNCE direction, mean-shift length |
|---|---|---|---|---|
| PCA-50 | 0.235 / 0.413 | 0.095 / 0.433 | **0.248 / 0.469** | **0.255** / 0.451 |
| scVI-30 | **0.191** / 0.385 | 0.093 / 0.337 | 0.188 / 0.386 | 0.187 / **0.391** |

* *Why raw InfoNCE fails:* its overshooting actions leave "no perturbation" as the nearest candidate for 76–80% of held-out cells.
* *Calibration:* one scalar, fitted on training pairs only, fixes that (α = 0.414–0.431 across seeds and spaces).
* *By effect strength:*
  * **Strong perturbations:** calibrated InfoNCE is best (PCA 0.441 vs 0.320; scVI 0.458 vs 0.382).
  * **Weak perturbations:** every method is near chance (≤ 0.10). This matches Week 1's finding that 18 singles barely raise their own target.
* *With the gem-mean start:* the pattern is the same (PCA calibrated 0.246 vs mean shift 0.216; scVI 0.258 vs 0.264).

**Q5. Which state space works better, PCA-50 or scVI-30?**
* *Answer: neither clearly.*
* *Evidence:*
  * InfoNCE instance MRR: 0.181 (PCA) vs 0.174 (scVI).
  * Calibrated identification: 0.248 vs 0.188 with the pooled start, 0.246 vs 0.258 with the gem-mean start.
  * scVI actions agree more closely with the mean shift (cosine 0.86 vs 0.80).
* *Recommendation:* PCA-50 keeps the advantage of decoding linearly back to genes, so keep both for now.

## 3. What this means for composition

The learned actions have good directions and wrong lengths. Composing them as f(a) + f(b), or feeding them to the Week 1 decoder, would carry the 2–4× overshoot into every double. Before that, the actions need a length anchor. The options:
* add a squared-error term `λ‖s(x_ctrl) + f(a) − s(x_pert)‖²` to the InfoNCE loss;
* use the one-scalar calibration;
* use a learnable temperature with the squared-error term.

Section 5 (addendum) tests these options.

## 4. Caveats
* **τ selection:** the selection signal is retrieval MRR, not the loss. At small τ, the validation loss barely drops below chance (PCA τ = 0.03: 5.37–5.47 vs ln 256 = 5.55), because a few far-off cases dominate it while ranking still improves. Larger τ raises same_pert@10 (0.475 at τ = 0.3, PCA) but lowers MRR.
* **Partner sampling:** OT partners are sampled from the top 5 couplings. The full coupling is blurrier (perplexity 30–75), so "the" partner is a soft notion.
* **Seed scope:** only InfoNCE has seeds; the references are deterministic.
* **Outputs:**
  * Retrieval tables: `results/week2_retrieval.csv`.
  * Per-perturbation identification: `results/week2_identification_by_pert.csv`.
  * Effect strengths: `results/week2_effect_strength.csv`.
  * Actions: `embeddings/infonce__<space>__s<seed>.npz`.
  * Training histories: `results/infonce/`.

## 5. Addendum: fixing the action length

Still Week 2 work, using the same OT pairs, OT-partner positive definition, batch construction (32 perturbations × 8 pairs) and splits as above. Nothing here reads a test double. New code:
* `retrieval.per_action_scale` and `length_table`;
* the optional `rho` anchor term in `infonce.infonce_loss`, plus `python -I -m pertact.infonce anchor`;
* `python -I -m pertact.evaluate actions`, the decoder smoke test, which runs on validation only.

### 5.1 Which score does the current InfoNCE use, and what is τ?
* **Score:** negative squared distance after adding the action, not cosine:
  `s(i, j) = −‖z_ctrl,i + f(a_i) − z_j‖² / (τ · D₀)`.
* **D₀:** the mean squared distance between random training cells, so τ is unitless and means the same in both spaces. D₀ is 120.1 for PCA-50 and 33.1 for scVI-30 (mean of 3 seeds).
* **τ:** 0.03 (PCA-50) and 0.1 (scVI-30), chosen by validation instance-retrieval MRR on seed 0.
  * In the form `−‖·‖² / τ′`, the effective temperature is τ′ = τ·D₀ = **3.6 (PCA-50)** and **3.3 (scVI-30)** squared state units.
  * For scale: a training cell's squared distance to its OT partners is 56.6 in PCA-50 and 14.5 in scVI-30.
* **Consequence for option 1:** it does not apply. The overshoot already happens under the squared-distance score, so the as-trained actions are the option-1 result. No retraining was needed.

### 5.2 Variants and the bar they had to clear
| # | variant | what changes | training changed? |
|---|---|---|---|
| 1 | InfoNCE as trained | nothing (already squared distance) | no |
| 2a | + one global scalar α | all actions × α, least squares on training OT pairs: min_α Σ‖start + α f(a) − z_pert‖² | no |
| 2b | + one scalar per action c_a | same least squares, one c per perturbation, on that perturbation's training pairs | no |
| 3 | + length anchor | loss += (ρ/τ) · mean‖q − k_pos‖² / D₀ | yes |
| ref | mean shift | average OT displacement on training pairs (closed form) | no training |

* **The per-action scalar has a closed form.** c_a f(a) is the projection of the mean OT shift onto f(a). Its length is therefore ‖mean shift‖ × cos(f(a), mean shift). It can only undershoot, by exactly that cosine.
* **The bar for "works":**
  * the median length ratio ‖f(a)‖ / ‖mean OT shift‖ falls within 0.8–1.25;
  * identification is no worse than the mean shift without any further rescaling;
  * held-out instance retrieval is no worse than the mean shift.
* **Why the mean shift is the bar:** if a length-fixed InfoNCE action set loses to the closed-form average on held-out cells, InfoNCE training adds nothing. That is why option 3 was run even though option 2 fixes length and identification.

### 5.3 Results (validation cells; InfoNCE 3 seeds, seed sd ≤ 0.014 everywhere)
**Length ratio ‖f(a)‖ / ‖mean OT shift‖, per perturbation.** Values are the median [min–max] over the 105 perturbations (seed-averaged). The last two columns are the share of perturbations within 0.8–1.25.

| variant | PCA-50 | scVI-30 | in range, PCA | in range, scVI |
|---|---|---|---|---|
| 1 InfoNCE as trained | 2.06 [0.89–11.3] | 2.39 [1.08–9.20] | 5% | 2% |
| 2a global α (α = 0.42–0.43) | 0.88 [0.38–4.83] | 1.00 [0.45–3.86] | 30% | 34% |
| 2b per action (c_a median 0.39 / 0.34) | 0.80 [0.38–0.94] | 0.85 [0.33–0.98] | 49% | 57% |
| 3 anchor, ρ = 1 (strong, see below) | 1.17 [0.89–2.71] | 1.30 [0.92–2.80] | 68% | 34% |

* **By effect strength:** raw InfoNCE overshoots weak perturbations most (median 3.7× PCA, 4.2× scVI; strong ones 1.5× / 1.6×).
* **Global α cannot fix that spread.** It leaves weak perturbations 1.6–1.8× too long and strong ones at 0.65–0.69×.
* **Per-action scaling fixes the spread but undershoots weak perturbations** (0.64 / 0.50), because their InfoNCE directions agree least with the mean shift.

**Held-out retrieval and identification:**

| variant | instance MRR (PCA / scVI) | same_pert@10 | identification top-1, pooled-OT start | top-1, strong tertile | top-1, gem-mean start |
|---|---|---|---|---|---|
| mean shift (ref) | 0.120 / 0.133 | 0.115 / 0.126 | 0.235 / 0.191 | 0.320 / 0.382 | 0.216 / 0.264 |
| 1 InfoNCE as trained | **0.181 / 0.174** | **0.255 / 0.245** | 0.095 / 0.093 | 0.248 / 0.250 | 0.160 / 0.198 |
| 2a global α | 0.082 / 0.087 | 0.067 / 0.065 | 0.248 / 0.188 | **0.441 / 0.458** | 0.246 / 0.258 |
| 2b per action | 0.092 / 0.116 | 0.090 / 0.113 | **0.268** / 0.190 | 0.394 / 0.402 | 0.257 / 0.260 |
| 3 anchor, ρ = 1 | 0.136 / 0.154 | 0.138 / 0.158 | 0.206 / 0.156 | 0.296 / 0.311 | 0.212 / 0.245 |
| *InfoNCE direction at mean-shift length (control)* | 0.114 / 0.125 | 0.116 / 0.122 | 0.255 / 0.187 | 0.358 / 0.390 | 0.262 / 0.265 |

Chance top-1 is 1/106. The weak tertile is near chance for every variant (≤ 0.10).

**Verdict per variant:**
* **1 (as trained): fails.** The length is 2–2.4× too long and identification without rescaling collapses (0.09 vs 0.19–0.24), even though instance retrieval is the highest of all.
* **2 (post-hoc rescaling): fixes length and identification, loses retrieval.**
  * Both scalings bring the median length into range.
  * Both match or beat the mean shift on identification: 2b gives PCA 0.268 vs 0.235 and ties in scVI (0.190 vs 0.191); 2a is best on strong perturbations (0.44–0.46 vs 0.32–0.38).
  * Instance retrieval falls below the mean shift in both spaces (PCA 0.082–0.092 vs 0.120; scVI 0.087–0.116 vs 0.133). So option 2 fails the bar.
  * The control row explains why. At exactly the mean-shift length, InfoNCE's directions retrieve slightly *worse* than the mean shift's own (0.114 vs 0.120; 0.125 vs 0.133). InfoNCE's retrieval edge exists only at the overshooting length its directions were trained for.
* **3 (anchor): a weak anchor does not fix the length; one strong enough to do so is no longer weak.** The ρ grid ran on seed 0, with ρ chosen by the training-pair length ratio only:

| ρ (anchor weight as a fraction of InfoNCE's own positive pull) | λ = ρ/τ, PCA / scVI | anchor share of the validation loss, PCA / scVI | median length ratio, PCA / scVI | val instance MRR, PCA / scVI |
|---|---|---|---|---|
| 0 | 0 | 0 / 0 | 2.07 / 2.40 | 0.183 / 0.173 |
| 0.01 | 0.33 / 0.1 | 4% / 2% | 1.98 / 2.30 | 0.182 / 0.174 |
| 0.03 | 1 / 0.3 | 10% / 5% | 1.84 / 2.13 | 0.180 / 0.174 |
| 0.1 | 3.3 / 1 | 24% / 14% | 1.59 / 1.84 | 0.167 / 0.173 |
| 0.3 | 10 / 3 | 46% / 32% | 1.34 / 1.59 | 0.154 / 0.166 |
| **1** | **33 / 10** | **73% / 59%** | **1.17** / 1.30 | 0.136 / 0.153 |

  * **Flagged: the selected anchor is strong.** The rule picked ρ = 1 (λ = 33 in PCA-50, 10 in scVI-30), the only grid value that reaches the target, and only in PCA. scVI-30 stays at 1.30× even at ρ = 1. At this weight the squared-error term is 59–73% of the loss: this is the reconstruction objective with some InfoNCE, not InfoNCE with a weak anchor.
  * **What it gives:** actions that point almost exactly along the mean shift (median cosine 0.97 / 0.96, vs 0.80 / 0.85 unanchored) and are still 17–30% long. They beat the mean shift on retrieval (0.136 vs 0.120; 0.154 vs 0.133), plausibly from that remaining overshoot (Q3). They lose on identification without rescaling (0.206 vs 0.235; 0.156 vs 0.191).
  * **Weak settings:** at ρ ≤ 0.03 (anchor ≤ 10% of the loss) the median overshoot only drops from 2.1× to 1.8× (PCA) and from 2.4× to 2.1× (scVI).
  * **Why the anchor must be so strong:** InfoNCE's positive logit already applies the same squared-error pull, with weight 1/τ. The overshoot comes from the denominator, which pushes each query away from other perturbations' cells. The anchor only adds to the pull, so it shifts the balance only once its weight is comparable to InfoNCE's own (ρ ≈ 1).
  * Because the anchor this strong overlaps with the planned reconstruction ablation, these actions are reported but not carried forward.

**Answers to the three questions:**
* **Length ratio:** see the first table. It is fixed by post-hoc rescaling (2a, 2b), and only approximately by a strong anchor.
* **Retrieval against the mean shift:** only overshooting actions beat it (1, and 3 with its remaining 1.2–1.3× length). Every length-correct InfoNCE variant is below it.
* **Identification without rescaling:** no. The as-trained actions fail (0.09). The strong anchor recovers 82–88% of the mean shift's accuracy, and its least-squares α would still be 0.81–0.88. With rescaling, InfoNCE matches the mean shift (scVI) or beats it (PCA, and every strong-tertile comparison).

**Choice for the decoder smoke test: 2b, per-action scaling, both spaces.**
* **Rule:** among variants that fix the median length and leave InfoNCE training unchanged (this excludes the strong anchor), take the highest validation identification top-1, averaged over spaces. 2b scores 0.229, 2a 0.218.
* **Note:** every embedding goes through the same unit-mean-norm step before the decoder, which removes any global scale. 2a would therefore reach the decoder exactly as the raw actions do. 2b differs because it changes relative lengths between perturbations.

### 5.4 Decoder smoke test (validation singles, not the Week 4 evaluation)
* **Setup:**
  * The per-action-scaled InfoNCE actions (2b) are frozen and used as the perturbation embedding of the Week 1 latent-additive and decoder-only models, in both spaces. InfoNCE seed s is paired with decoder seed s.
  * Week 1 hyperparameters throughout: hidden 1024, latent 128, dropout 0, lr 3e-4, ≤150 epochs, early stopping on validation-single loss, and the same unit-mean-norm step.
* **References:**
  * **One-hot and PCA loadings.** Week 1's saved latent-additive models are reloaded. Week 1 never saved its decoder-only models, so those were retrained with the same seeds. Every reference score reproduces `results/week1.csv` exactly (max difference 0.0).
  * **Mean-shift and raw InfoNCE actions,** run through the same pipeline.
  * **Direction only (follow-up):** InfoNCE and mean-shift actions with every vector divided by its own norm, so all 105 have length 1, as in one-hot. For InfoNCE this is also the direction of every rescaled variant (2a, 2b), since rescaling never changes direction.
* **Test split:** only the 105 validation singles are scored. The runner is built without the test split, so no test cell is loaded.

| embedding | dim | latent additive: Pearson Δ, top-20 DE | target-excluded | weak third | decoder-only: Pearson Δ, top-20 DE | weak third |
|---|---|---|---|---|---|---|
| one-hot | 105 | 0.959 ± 0.005 | 0.949 | 0.904 | 0.967 ± 0.002 | 0.926 |
| PCA loadings | 32 | **0.966 ± 0.002** | **0.952** | **0.923** | **0.968 ± 0.001** | **0.929** |
| **InfoNCE, per-action scaled, PCA-50** | 50 | 0.944 ± 0.013 | 0.938 | 0.859 | 0.955 ± 0.004 | 0.890 |
| **InfoNCE, per-action scaled, scVI-30** | 30 | 0.936 ± 0.013 | 0.930 | 0.834 | 0.956 ± 0.010 | 0.894 |
| InfoNCE as trained, PCA-50 | 50 | 0.948 ± 0.006 | 0.940 | 0.877 | 0.967 ± 0.002 | 0.926 |
| InfoNCE as trained, scVI-30 | 30 | 0.961 ± 0.007 | 0.949 | 0.909 | 0.965 ± 0.001 | 0.921 |
| mean shift, PCA-50 | 50 | 0.924 ± 0.016 | 0.924 | 0.813 | 0.958 ± 0.005 | 0.901 |
| mean shift, scVI-30 | 30 | 0.940 ± 0.012 | 0.936 | 0.853 | 0.961 ± 0.004 | 0.909 |
| **InfoNCE, direction only, PCA-50** | 50 | 0.959 ± 0.005 | 0.947 | 0.904 | 0.967 ± 0.002 | 0.925 |
| **InfoNCE, direction only, scVI-30** | 30 | 0.965 ± 0.004 | 0.951 | 0.919 | 0.965 ± 0.002 | 0.921 |
| mean shift, direction only, PCA-50 | 50 | 0.961 ± 0.004 | 0.949 | 0.911 | 0.966 ± 0.001 | 0.923 |
| mean shift, direction only, scVI-30 | 30 | 0.962 ± 0.001 | 0.949 | 0.915 | 0.966 ± 0.000 | 0.924 |

Mean ± sd over 3 seeds. *Weak third*: the 35 perturbations with the smallest mean OT shift in PCA-50, the same 35 for every row. On the strong and medium thirds every embedding scores 0.973–0.989 with both decoders.

* **The plumbing works.** Frozen actions train, early-stop and predict validation singles at 0.94–0.97 Pearson Δ, the same range as the Week 1 embeddings. As Week 1 Q5 found, seen singles barely separate embeddings.
* **The length-fixed actions sit slightly below one-hot and PCA loadings,** and below the as-trained InfoNCE actions:
  * latent additive: 0.944 / 0.936 vs 0.959 / 0.966;
  * decoder-only: 0.955 / 0.956 vs 0.967 / 0.968.
* **The whole gap is in the weakest third:** for example, latent additive scores 0.834–0.859 there vs 0.904 for one-hot.
* **Why: the uneven lengths of the length-faithful actions.**
  * The decoder sees each embedding after a single global rescale, so relative lengths between genes survive into W·e.
  * Length-faithful actions (the mean shift and per-action scaling) make weak perturbations 3–6× shorter than strong ones. The median weak/strong norm ratio is 0.29–0.30 for the mean shift and 0.16–0.23 for per-action scaling. So the perturbations that are hardest to fit reach the decoder with the smallest inputs.
  * Raw InfoNCE overshoots weak perturbations most (Q3), which makes its norms nearly uniform: weak/strong 0.73–0.76, coefficient of variation 0.14. That is close to one-hot, and it fits weak perturbations about as well.
* **Follow-up: direction only confirms that their length is the cause.** Only the lengths change, the directions are identical.
  * Dividing every action by its own norm removes the whole gap. InfoNCE directions score 0.959 / 0.965 (latent additive) and 0.967 / 0.965 (decoder-only), against one-hot 0.959 / 0.967 and PCA loadings 0.966 / 0.968.
  * The gain is in the weak third, as predicted: latent additive 0.904 / 0.919, up from 0.859 / 0.834 with per-action lengths, level with one-hot (0.904).
  * It is consistent: paired by seed, the weak third improves in 23 of 24 cases (4 length-faithful action sets × 2 decoders × 3 seeds). The one exception is per-action PCA-50, latent additive, seed 2 (0.893 → 0.891).
  * The latent-additive seed sd also drops to 0.004–0.005, the one-hot range. The extra seed noise came from the uneven lengths too.
* **But uneven length is not harmful by itself.** PCA loadings reach the decoder just as uneven: weak/strong norm ratio 0.29, coefficient of variation 1.15, weak-third median norm 0.27 of the mean, as short as per-action scaling (0.27–0.36) and shorter than the mean shift (0.44–0.45). Yet they score best on the weak third (0.923). So these actions' lengths cost the weak third in combination with something else about them.
  * One candidate is that short weak vectors pointing along the strong ones cannot be amplified without also amplifying the strong ones. It fits the mean shift: a median 67% (PCA-50) and 50% (scVI-30) of each weak direction lies in the strong vectors' top-5 subspace, against 14% for PCA loadings. It does not fit per-action InfoNCE, which overlaps only 18% and still loses (0.859 / 0.834).
  * The smoke test therefore shows *that* these actions need equal lengths in the decoder, not why PCA loadings do not. Numbers: `results/week2_decoder/length_overlap.csv`.
* **Direction only does not separate InfoNCE from the mean shift.** The mean shift's own directions do just as well: 0.961 / 0.962 (latent additive) and 0.966 / 0.966 (decoder-only). Every overall difference between the two is 0.003 or less, and every weak-third difference 0.007 or less, smaller than the larger seed sd of each pair (0.004–0.012).
  * This happens even though the two direction sets are shaped very differently (`results/week2_decoder/direction_spread.csv`):

    | unit directions | median pairwise cosine | median nearest-neighbour cosine | participation ratio (effective dimensions) |
    |---|---|---|---|
    | InfoNCE, PCA-50 / scVI-30 | 0.14 / 0.04 | 0.61 / 0.66 | 13.0 / 12.8 |
    | mean shift, PCA-50 / scVI-30 | 0.49 / 0.22 | 0.90 / 0.83 | 3.1 / 6.3 |

  * The mean shifts share one large common direction (the norm of their mean unit vector is 0.59 in PCA-50, against 0.38 for InfoNCE). InfoNCE spreads its actions out, as its denominator rewards. With only 105 seen perturbations, the decoder evidently separates either set equally well, so the spread does not show up in this test. It may matter for unseen combinations, where the decoder cannot memorise each input.
* **Noise:** with effect-scaled embeddings the latent-additive seed sd is 0.012–0.016, against 0.002–0.005 for one-hot and PCA loadings. Differences under about 0.015 between action variants in that decoder are within noise. The decoder-only gap (0.955 vs 0.967) is larger than its sd.
* **Limits:**
  * Validation singles are seen perturbations, so this tests plumbing, not composition.
  * The Week 1 decoder ignores the starting state (Week 1 Q6), so any benefit of state-dependent actions could not show here.
  * Early stopping uses validation loss, as in Week 1, so every number is optimistic by the same small amount.

### 5.5 Where this leaves the actions
* **The overshoot is real under the squared-distance score.** InfoNCE learns actions of nearly constant length: about 4× the mean OT shift for weak perturbations and about 1.5× for strong ones.
* **Length can be fixed after training.** The per-action least-squares scalar (the projection of the mean shift onto f(a)) restores identification. It gives up the retrieval gain, because at the correct length InfoNCE's directions are no better than the mean shift's for retrieval. They are slightly better for identification in PCA-50 (0.268 vs 0.235) and tied in scVI-30.
* **A training-time anchor works only when it dominates the loss.** It is therefore flagged and not carried forward.
* **Length matters in two different ways:**
  * For composing in the state space (f(a) + f(b) landing on a double), the per-perturbation length should be right, and the per-action scalar gives it.
  * As a decoder input, these actions' lengths are a nuisance. Direction-only actions match one-hot (0.959–0.967) and recover the weak third. PCA loadings show that uneven length is not always harmful, so this is a finding about these vectors, not a general rule.
* **Practical split:** carry each action as a direction plus a length fitted on training pairs, and decide per use which one the next stage gets. For the decoder, pass the direction only. For composing in the state space, use direction × fitted length. The composition step is not tested here.
* **Open question:** once lengths are matched, InfoNCE does not yet clearly beat the closed-form mean shift. Retrieval is lower and identification is equal or slightly better. In the decoder the two tie, both with their true lengths (0.944 vs 0.924 PCA-50; 0.936 vs 0.940 scVI-30, within noise) and as directions only (0.959 vs 0.961; 0.965 vs 0.962). InfoNCE's directions are far more spread out (pairwise cosine 0.14 vs 0.49 in PCA-50), but seen singles cannot reward that. The comparison that can settle it is composition on doubles.

### 5.6 Addendum outputs
* **Length and scales:** `results/week2_length_ratio.csv` (per perturbation, variant and seed) and `results/week2_action_scales.csv` (α and c_a).
* **Retrieval:** `results/week2_retrieval.csv` now includes `infonce_scaled_per_action` and `infonce_anchor`. The previous version is kept as `week2_retrieval_v1.csv`.
* **Anchor:**
  * grid: `results/infonce/anchor_grid.csv`;
  * chosen ρ and λ: `results/infonce/anchor.json`;
  * actions: `embeddings/infonce_anchor__<space>__s<seed>.npz`.
* **Decoder smoke test:**
  * scores: `results/week2_decoder/smoke__{main,raw,meanshift,unit_infonce,unit_meanshift}.csv`;
  * diagnostics: `results/week2_decoder/direction_spread.csv` and `length_overlap.csv`, from `scripts/week2_direction_diagnostics.py`;
  * summary tables: `results/week2_decoder/summary.csv` and `paired_length_vs_unit.csv`, from `scripts/week2_decoder_summary.py`;
  * per-perturbation scores: `smoke_per_perturbation__*.csv`;
  * predictions: `results/week2_decoder/predictions/`;
  * models: `models/decoders/<embedding>__<decoder>__s<seed>.pt`.
