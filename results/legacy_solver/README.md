# Week 2 results from the earlier OT solver

Until 2026-11-11 the OT pairs were made by a hand-written Sinkhorn solver. The code now calls the POT library, and the
main Week 2 files were regenerated with it. This folder keeps what the earlier solver produced, so the two can be
compared. The current code does not reproduce these files, and `scripts/check_outputs.py` does not check them.

| here | what it is | regenerated as |
|---|---|---|
| `ot/` | the pairs, their diagnostics, the solver checks and the pair-plot statistics | `results/ot/` |
| `embeddings/` | InfoNCE and anchored action vectors trained on those pairs | `embeddings/infonce*` |
| `infonce/` | temperature tuning, training histories and the anchor grid | `results/infonce/` |
| `week2_*.csv` | scores of both tests, effect strengths and the length tables | `results/week2_*.csv` |
| `week2_decoder/` | the direction diagnostics computed from those vectors | the same two files in `results/week2_decoder/` |
| `pairs_comparison.csv` | one row per pair file: how many cells get different partners under the two solvers | |
| `scores_comparison.csv` | every Week 2 test score under the two solvers | |

How large the difference is:

* A cell's set of 5 partners differs for at most 0.16% of the cells in a file (52 of 37,770).
* Of 1,428 test scores, the largest change is 0.0125. For the validation headline scores (MRR, recall@10, right
  perturbation among the 10 nearest, top-1) the largest change is 0.0050 and the median 0.00015.
* The action vectors are nearly the same: the cosine between old and new has a median of 0.9997 or higher in every
  file but one. The exception is the anchored PCA-50 set of seed 0, whose training now stops after 2,100 steps instead
  of 2,800 (median cosine 0.94).
* The pair figure (`figures/ot_pairs_X_pca_g.png`) is identical under both solvers, pixel for pixel, so no copy is kept.
* The chosen temperatures (0.03 for PCA-50, 0.1 for scVI-30) and the chosen anchor weight (1 in both spaces) are the
  same.

Not rerun: the Week 2 decoder test. The scores in `results/week2_decoder/` (`smoke*`, `summary.csv`,
`paired_length_vs_unit.csv`, `predictions/`) come from decoders trained on the vectors in `embeddings/` here. Those 60
decoders are in `models/decoders_legacy_solver/`, which is not in git.
