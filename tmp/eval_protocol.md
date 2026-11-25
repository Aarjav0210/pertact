# Evaluation protocol

Draft, 2026-11-15. This file lists what is evaluated; `tmp/eval_design.md` describes the code that runs it. Where the
two disagree, this file wins. Items marked **[P]** are pre-registered as primary, and everything else is secondary.

---

## 1. Data and held-out units

**1.1 Data.** Norman et al. 2019, K562 CRISPRa: 105 single-gene perturbations and 127 doubles ("A+B"). The doubles
involve 71 of the 105 genes. 34 genes appear in no double, and four hub genes (PTPN12, KLF1, CEBPE, UBASH3B) each
appear in 10–12 doubles.

**1.2 Gene folds [P].** The 105 genes are partitioned into 5 folds, stratified jointly by effect tier, Norman 2019
program and number of doubles per gene. The four hub genes are placed in different folds. A held-out gene loses
*all* of its cells (current train and val) to the evaluation side. We generate R = 10 partitions with different
seeds. Expensive models are run on partition 0, and cheap models (ridge, nearest-gene, mean shift) on all 10.

**1.3 Seen-ness of doubles.** In each fold, a double is labelled seen2, seen1 or seen0 by how many of its genes
are in the training fold. Per fold the expected counts are about 81 / 41 / 5. One partition makes only about 24
distinct doubles seen0, and ten partitions about 112. A double's seen0 score is its mean over the partitions where it
is seen0, and we report how many doubles that covers. For cheap models we also run leave-pair-genes-out: 127 fits,
each removing both genes of one double.

**1.4 Pair folds.** Models that learn interactions from doubles get a separate 5-fold split over the 127 doubles,
stratified by interaction-strength quartile and hub gene. These models train on all singles plus 4/5 of the doubles.
This is the only seen2 test for them.

**1.5 Controls.** The existing control partition is used (6,781 train / 1,453 val / 1,453 test). Models only see
training controls, and observed shifts are measured against test controls.

**1.6 Leakage rules.**
- Everything fitted is fitted per fold on training-fold cells only: the HVG statistics, the PCA-50 basis and
  `pca_loadings`, OT pairs, actions, decoders, DE lists for training genes, early stopping and tuning.
- Tuning uses an inner leave-genes-out CV inside the training fold.
- scVI-30 stays frozen (it was fit on all training cells, including those of genes later held out). We report one
  sensitivity check: partition 0 rerun with a per-fold refit.
- GenePT and STRING are external and stay fixed. Possible literature leakage into GenePT is noted as a caveat.
- The runner fails if a model's `trained_on_genes()` contains a held-out gene, or a double that contains one.

## 2. Unseen perturbation and unseen combination prediction

**2.1 Outputs.**
- For every held-out single and every double, each model predicts the shift in PCA-50 (and scVI-30). Models with a
  decoder also predict the mean expression profile.
- For seen1 and seen0, the model also predicts each held-out gene as a single. This separates the single error from
  the composition error.

**2.2 Primary metrics [P].**
- *State space:* the normalised miss `||pred − obs||² / ||obs||²` in PCA-50 (`doubles.centre_scores`). A value of 1
  means no better than predicting no change.
- *Expression space:* `pearson_delta_de20_nt`, the Pearson of predicted vs observed change on the top-20 DE genes
  with the target genes removed.
- For held-out singles, the DE genes come from their own held-out cells, compared with test controls.

**2.3 Secondary metrics.**
- Cosine and length ratio of the predicted shift.
- The scVI-30 normalised miss.
- `mse_de20_nt` and `pearson_delta_all`.
- For doubles, `pearson_int_top20`. For seen1/seen0 its additive reference is built from the held-out singles'
  measured effects. Only the evaluator uses it; no model sees it.

**2.4 Reporting.** Singles, seen2, seen1 and seen0 are reported separately, each also by effect tier. A headline
number never pools them.

**2.5 Baselines (every table).**

| condition | baselines |
|---|---|
| unseen single | no change; mean of training shifts ("generic"); **nearest-gene copy** per embedding (k = 1 and similarity-weighted k = 5); ridge from embedding to shift; one-hot floor; split-half noise ceiling |
| seen2 | **additive of measured training singles**; the model's own summed actions; the stronger single alone |
| seen1 / seen0 | additive with nearest-gene copies for unseen genes; the seen single alone (seen1); 2 × generic (seen0); oracle additive of measured held-out singles (composition ceiling) |

## 3. Selecting unseen genes from a fixed candidate pool

**3.1 Pools.**
- *Main pool P_k [P]:* the ~21 held-out genes of fold k plus "no perturbation".
- *Naming pool P_k+:* all 105 genes plus control. Seen genes use their learned actions and unseen genes use
  predicted ones. It is reported because seen candidates are better calibrated and draw picks away from unseen ones.
- *Pair pool Q_k:* the seen1 and seen0 doubles of fold k.
- Each pool is written to `pools/<task>_<fold>.json` with a SHA before any run. Every model scores exactly that list,
  and pools are never mixed within one comparison.

**3.2 Selection tasks.** Each task is defined by how the model's outputs rank the pool, what the truth is, and the
metric.

| task | rule from model output | truth | metric |
|---|---|---|---|
| **S1 target reaching [P]** | rank c by ‖s₀ + f̂(c) − target‖² (s₀ = gem-matched control mean; target = mean state of half of g*'s cells) | g* | rank, top-1/top-5, MRR (chance shown) |
| S2 strongest effect | rank by ‖f̂(c)‖ | measured ‖shift‖ | NDCG@5, precision@5 (top quartile = positive), Spearman |
| S3 program push | rank by u·f̂(c), with u a program axis computed on training cells of this fold | u · measured shift | hit@5 of top decile, regret@k = (best in pool − best picked) / best in pool |
| S4 non-additive pairs | rank pairs by ‖f̂(a,b) − f̂(a) − f̂(b)‖ | `EvalRef.strength` | precision@k, AUPRC (top quartile = positive) |
| S5 coverage | greedy farthest-point on f̂ | measured effects of picks | programs hit; coverage relative to oracle k-medoids |

For S1, weak-effect genes are reported in their own subset, because their targets look like controls.

**3.3 Sequential selection.**
1. Start with 10 genes revealed at random; all other genes are unrevealed.
2. In each round, the model picks 5 genes from the unrevealed pool.
3. Their cells are revealed and the model refits on all revealed genes.
4. Repeat for 10 rounds. The whole run is repeated for at least 20 seed draws.

Unrevealed cells never reach fitting. We report the area under the discovery curve (best true S3 score, or number of
top-decile genes found, against experiments used) and the number of experiments needed to find the first top-decile
gene.

**3.4 Baselines.**
- Random: exact expectation and simulated.
- Embedding similarity to known strong or on-program training genes, one per embedding.
- Nearest-gene copy, from 2.5.
- Hub prior: rank by embedding norm or STRING degree.
- One-hot floor.
- Oracle: ranks by truth (regret 0).
- Split-half oracle: ranks by truth from the other half of the cells (the attainable ceiling).

**3.5 Joint report.** The section 2 metrics and section 3 metrics go side by side in one table per model, by effect
tier. A model shrunk toward the generic mean can score well on MSE and still rank at chance. We also report the
Spearman across models between their prediction rank and their selection rank.

## 4. Gene representations: neighbor structure and how it changes

**4.1 Representations.**
- *Priors:* GenePT, STRING-n2v, pca_loadings. One-hot is a control only, because its kNN graph is all ties.
- *Learned:* InfoNCE (3 seeds) and InfoNCE-anchor in PCA-50 and scVI-30.
- *Decoder:* the perturbation latent `W e`. In `LatentAdditive`, the control-state term is the same for every
  perturbation, so `W e` is the perturbation's own part.
- *Mean shift on training cells* is labelled as a near-ceiling reference, not as a learned representation.
- All are compared on the common gene set G (STRING-n2v covers 93 of the 105 genes, so |G| ≤ 93). |G| is reported.

**4.2 Reference geometry E.** For each gene, the measured effect comes from **validation cells only**: the shift in
PCA-50 and scVI-30, and the log-fold change on a gene set fixed from training data. Validation cells are split into
halves E₁ and E₂ (20 resamples). The agreement between E₁ and E₂ is the noise ceiling for every metric below, and
every score is also given as a fraction of it. Genes with fewer than 10 cells per half are flagged.

**4.3 Neighbor graphs.** Cosine kNN after per-dimension z-scoring, k ∈ {5, **10 [P]**, 20}. We also build per-gene
similarity ranks. Euclidean distance and cosine without z-scoring are run as a sensitivity check.

**4.4 Agreement between two representations X and Y.**
- Jaccard@k, raw and chance-corrected: (J − J₀)/(1 − J₀), with J₀ ≈ k/(|G| − 1).
- Trustworthiness and continuity.
- RSA: Spearman between distance matrices, with a Mantel test (10⁴ joint row/column permutations).
- Program purity of neighbors, compared with label permutation.

**4.5 Rewiring, prior P → learned L [P].**
- Gained edges = kNN_L \ kNN_P; lost edges = kNN_P \ kNN_L.
- Each set gets a precision against the kNN graph of E.
- Headline: Δ = prec(gained) − prec(lost), with a bootstrap CI over genes. Δ > 0 means learning moved the neighbor
  structure toward the measured effect geometry.
- Gained and lost edges are also tested for enrichment in GO-BP slim, CORUM and STRING ≥ 700 co-membership
  (Fisher's test, BH correction). STRING is not used when P is STRING-n2v.
- An edge counts only if it appears in at least 2 of the 3 seeds.

**4.6 Nulls.**
- Label permutation for RSA and purity.
- Degree-preserving edge swaps of L.
- Random rewiring with the same number of gained edges.
- For `W e`, a random Gaussian projection of the prior to the same dimension. Without it, a change caused by the
  linear map alone would look like learning.

**4.7 Held-out genes [P].** Folds from 1.2. Each unseen gene is placed through its prior (ridge from prior to action,
or `W e`). We measure recall@10 of its true effect-neighbors among the training genes, against the same quantity in
the raw prior. This tests whether the rewiring holds for unseen genes or was memorised; in-sample rewiring is
partly circular, because the decoder is trained to reproduce the effects.

**4.8 Figures.**
- *Main figure:*
  - (a) Jaccard@10 to E along the chain prior → InfoNCE → `W e`, with the ceiling band and the null floor.
  - (b) prec(gained) vs prec(lost), against the degree-preserving null.
  - (c) held-out recall@10, prior vs learned.
- *Supplement:* a representation × representation matrix (RSA in the lower triangle, Jaccard@10 in the upper),
  one per state space.

## 5. Statistics (all sections)

**5.1 Replicates.** 3 training seeds × 5 folds × R partitions. Each condition's scores are averaged before
inference, so each condition counts once.

**5.2 CIs.** 95% bootstrap with 2,000 resamples, over genes for singles and over targets or draws for selection.
For doubles we use a gene-cluster bootstrap: resample genes and keep the pairs whose two genes were both drawn,
because hub genes make doubles dependent.

**5.3 Comparisons.** Model vs baseline is tested on per-condition differences (paired bootstrap CI and Wilcoxon
signed-rank), with Holm correction across the comparisons pre-registered in this file. We also report the win rate.
A difference inside the CI is reported as no difference.

**5.4 Pre-registration.** The [P] items, k = 10, PCA-50 and the comparison list are fixed before the first scored
run. Changing them makes a new protocol version.

## 6. Known limitations

- seen0 is small and its doubles are correlated, even with 10 partitions, so seen0 claims are exploratory.
- Norman et al. chose the doubles mostly within programs, so results may not transfer to random pairs.
- The selection pools are small (~21 genes), so chance levels are high (top-5 ≈ 0.25). Chance is shown next to
  every number.
- With 105 (or 93) genes and many weak-effect genes, the strong tier dominates most scores, so everything is
  reported by tier.
- Program axes and labels taken from Norman 2019 may encode held-out genes. Axes used by models are therefore
  recomputed per fold, and the labels are only used for stratification and evaluation.
