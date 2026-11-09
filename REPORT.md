# Week 1 report: Norman 2019 data, frozen encoders, perturbation embeddings, latent additive baseline

Everything below was produced by the code in `src/pertact/` with `config.yaml` (seed 0). Package versions are in `env.txt`.

## 1. Dataset

| item | value |
|---|---|
| file | `NormanWeissman2019_filtered.h5ad` (scPerturb harmonised, raw integer UMI counts) |
| source | Zenodo record 13350497 (scPerturb RNA v1.4; same file and md5 as record 7041849) |
| md5 | `c870e6967d91c017d9da827bab183cd6` (verified after download) |
| raw size | 111,445 cells × 33,694 genes; `obs` has `guide_id`, `gemgroup` (8 groups), `good_coverage`, `number_of_cells`, `perturbation`, `nperts` |

The GEARS `norman` file was **not** used; raw counts were available.

### Label format (inspected, not assumed)

* `guide_id` holds one or more guide-*constructs* joined by `;`, e.g. `KLF1_NegCtrl0;KLF1_NegCtrl0`. Each Norman lentivirus carries two sgRNAs, so a construct is `<geneA>_<geneB>`.
* Non-targeting guides are `NegCtrl0`, `NegCtrl1`, `NegCtrl10`, `NegCtrl11`. A cell is a control when both positions are `NegCtrl*`, a single when one is, a double when neither is.
* Some constructs carry an alternative-sgRNA suffix (`TGFBR2_IGDCC3_2`, `HOXC13_NegCtrl0_1`). The parser strips pure-digit tokens, so they map to the same genes.
* 48 conditions appear under several constructs: both orientations (`BAK1_NegCtrl0` / `NegCtrl0_BAK1`) and four different control constructs. Orientation is ignored, so `A_B` and `B_A` become one condition `A+B`, with genes sorted alphabetically.
* My parsed conditions agree with scPerturb's `perturbation` column for 111,445 / 111,445 cells.
* Three guide symbols are outdated relative to the expression matrix's gene names. They were resolved through HGNC (previous/alias symbol → Ensembl id → `var.ensemble_id`). Embedding tables are always indexed by the guide symbol.

| guide symbol | current HGNC | column in matrix |
|---|---|---|
| C19orf26 | CBARP | CBARP |
| C3orf72 | FOXL2NB | FOXL2NB |
| KIAA1804 | MAP3K21 | RP5-862P8.2 (ENSG00000143674) |

## 2. Cell filtering and QC

| step | cells removed | cells left |
|---|---|---|
| raw | | 111,445 |
| low guide coverage (`good_coverage == False`, i.e. `number_of_cells == 0`) | 6,938 | 104,507 |
| more than one guide identity (`number_of_cells > 1`: droplet multiplets, up to 4+ guides; median UMIs 20.1k vs 13.4k for singlets) | 13,339 | 91,168 |
| ambiguous guide (constructs with different target sets) | 0 | 91,168 |
| total counts < 2,000 | 0 | |
| genes detected < 1,000 | 0 | |
| mitochondrial fraction > 15% | 112 | 91,056 |
| perturbations with < 50 cells (`min_cells_per_perturbation`, configurable) | 189 | **90,867** |

The scPerturb file is already pre-filtered: the minimum is 3,712 UMIs and 1,057 genes per cell, so the count and gene thresholds remove nothing. QC metrics were recomputed from counts (`MT-` genes).

Perturbations dropped for having fewer than 50 cells (all doubles): CEBPA+JUN (49), CBL+UBASH3A (48), CEBPB+JUN (46), C3orf72+FOXL2 (46).

### Counts after QC

| | value | expected (secondary sources) |
|---|---|---|
| total cells | 90,867 | ~105k (−13.5%) |
| control cells | 9,687 | |
| single-gene perturbations | 105 (47,214 cells) | ~105 |
| double perturbations | 127 (33,966 cells) | ~131 (−3%) |
| cells per perturbation, min / median / max | 50 / 286 / 1,625 | |
| cells per single | 95 / 405 / 1,625 | |
| cells per double | 50 / 250 / 1,001 | |

**Why the total is 13.5% below ~105k.** About 104.5k cells have a good-coverage guide call, which matches the "~105k" figure. Of those, 13,339 carry more than one guide identity. Your instruction to drop cells with more than two guide assignments removes them. Keeping only singlets matches the GEARS/Norman analysis (≈91k cells). The gap is within your 20% stop threshold, so I continued.

### Normalisation and genes

* `X` = log1p of counts normalised to 10,000 per cell, using library size over all 33,694 genes before any gene subsetting. Raw counts are in `layers["counts"]`.
* Rarely detected genes: kept genes detected in ≥ 20 **training** cells, which leaves 17,285 genes.
* HVGs: `seurat_v3` on raw counts of **training cells only** (44,551 cells), `batch_key = gemgroup`, 5,000 genes.
* Every perturbed target gene is force-included. 77 of the 105 targets were already HVGs and 28 were added, giving **5,028 genes**. `var` has `highly_variable`, `is_target`, `target_symbol` and `n_cells_train`.
* `gemgroup` is kept in `obs` and used as the batch key throughout.

Output: `data/processed/norman2019.h5ad` (90,867 × 5,028; `obs["split"]`, `obsm` state embeddings).

## 3. Splits (`splits.json`, seed 0)

| split | cells | single cells | double cells | controls | conditions |
|---|---|---|---|---|---|
| train | 44,551 | 37,770 | 0 | 6,781 | 105 singles |
| val | 10,897 | 9,444 | 0 | 1,453 | 105 singles (20% of each, held out per perturbation) |
| test | 35,419 | 0 | 33,966 | 1,453 | 127 doubles |
| excluded doubles | 0 | | | | none |

* Controls are split 70/15/15 within each gem group, so every split has controls from every batch for gem-matched sampling. The test control pool was not in the spec. I added it so that test doubles are compared against held-out controls (a decision; see §7).
* All 127 doubles have both components among the 105 training singles, so the excluded-doubles list is empty. The four doubles removed by the 50-cell filter are listed above.
* Assertions in `splits.py` check that the split is a partition and that no double cell is in train or val. Re-running gives a byte-identical `splits.json`.

## 4. Frozen state encoders g (fit on train cells, applied to all)

| key | model | notes |
|---|---|---|
| `obsm["X_pca_g"]` | PCA, 50 comps, on log-normalised HVGs (centred, unscaled) | 19.4% of variance; model in `models/pca_g.npz` |
| `obsm["X_scvi10"]` | scVI, n_latent 10, 2×128 layers, NB likelihood, batch = gemgroup | posterior means; model in `models/scvi10/` |
| `obsm["X_scvi30"]` | scVI, n_latent 30 | model in `models/scvi30/` |

Both scVI models ran the full 200 epochs on CPU; early stopping on scVI's internal 10% validation split never triggered.

| model | train ELBO | validation ELBO |
|---|---|---|
| scVI-10 | 2,525.6 | 2,540.8 |
| scVI-30 | 2,525.7 | 2,538.0 |

Latent means for all 90,867 cells are stored in `data/processed/state_embeddings.npz` and in the processed h5ad's `obsm`.

UMAP sanity figures (computed on a subsample, see below): `figures/umap_X_pca_g.png`, `figures/umap_X_scvi10.png`, `figures/umap_X_scvi30.png`. Each has three panels: control vs single vs double; CEBPA / KLF1 / CEBPE / SPI1 cells over controls; and gem group.

The UMAPs look right:
* Strong singles separate from controls in all three embeddings. KLF1 cells form their own island (erythroid shift); CEBPE, CEBPA and SPI1 gather at a myeloid edge.
* Controls sit in the main body.
* All 8 gem groups are evenly mixed, so batch does not drive the structure.
* scVI-30 separates the strong perturbations more cleanly than PCA-50.

The UMAPs use umap-learn directly on a 30k-cell subsample plus every cell of the four highlighted perturbations.

## 5. Perturbation embeddings (`embeddings/<method>.npz`)

Each file holds `emb` (rows = covered target genes), `genes` (guide symbols) and `missing`. Details are in `embeddings/coverage.json`.

| method | dim | covered / 105 targets | notes |
|---|---|---|---|
| one-hot | 105 | 105 | identity over the 105 training target genes |
| PCA loadings (Ahlmann-Eltze 2025) | 32 | 105 | pseudobulk = mean log-expression of each training single over **train cells** (105 perts × 5,028 genes), centred per gene, SVD; a gene's embedding is its row of the gene-loading matrix V[:, :32]. K=32 explains 88.5% of pseudobulk variance. Full 5,028 × 32 matrix in `pca_loadings_full.npz` |
| GenePT | 1536 | 105 | `GenePT_gene_embedding_ada_text.pickle` (ada-002, NCBI summaries) from Zenodo 10833191, which the GenePT README points to. KIAA1804 matched via current symbol MAP3K21; C19orf26 and C3orf72 match under their old names. Loaded with a restricted unpickler that accepts only dict/list/ndarray |
| STRING node2vec | 64 | **93** | STRING v12 `9606.protein.links.v12.0` + `protein.info` / `protein.aliases`. combined_score ≥ 700 → 236,930 undirected edges, 16,201 proteins, weight = score/1000. PecanPy SparseOTF walks (p=q=1, 10 walks × length 80) + gensim skip-gram (window 10, 1 epoch, 1 worker), seed 0. Edge list in `data/interim/string_v12_edges_ge700.tsv.gz` |

### Coverage of GenePT and STRING

| | covered | missing |
|---|---|---|
| GenePT | 105 / 105 targets; 105 / 105 val singles; 127 / 127 test doubles | none |
| STRING node2vec | 93 / 105 targets; 93 / 105 val singles; 108 / 127 test doubles fully covered | C19orf26, C3orf72, FEV, GLB1L2, HES7, IER5L, MAP7D1, MIDN, PRTG, ZBTB1, ZBTB10, ZNF318. All 12 exist in STRING but have no edge with score ≥ 700, so they are isolated in the high-confidence graph |

Sanity checks on node2vec:
* Nearest neighbours are biologically sensible: KLF1 → AHSP, HBE1, GATA1; MAPK1 → MAPK3, DUSP6; CDKN1A → CDK6, CDKN2D, CDK2.
* Cosine similarity separates edges from random pairs with AUC 0.988.
* Re-running gives identical walks and vectors.

Missing genes are not imputed. In evaluation, conditions that touch an uncovered training gene are left out of decoder training for that method. At evaluation, an uncovered gene contributes a zero vector (e = 0 means "no perturbation"). Every split is reported overall and on the covered subset (`<split>_covered`).

### Embeddings for every measured gene (for the downstream pipeline)

The decoder runs above use embeddings for the 105 perturbed genes only. For later work, including predicting perturbations of genes never perturbed here, `python -I -m pertact.embeddings all_genes` builds GenePT and STRING node2vec vectors for every gene measured in the Norman K562 matrix. There is no imputation: a gene without a vector is simply absent.

| gene set | genes | GenePT | STRING protein found | STRING node2vec (score ≥ 700 graph) |
|---|---|---|---|---|
| all measured | 33,694 | 22,640 (67%) | 19,507 | 16,165 (48%) |
| expressed (≥ 20 training cells) | 17,285 | 14,416 (83%) | 13,550 | 11,730 (68%) |
| model genes (5,028) | 5,028 | 4,323 (86%) | 4,065 | 3,607 (72%) |
| perturbed targets | 105 | 105 | 105 | 93 |

* **Files.**
  * `embeddings/gene_universe.csv` has one row per measured gene: matrix symbol, Ensembl id, current HGNC symbol, cells detected (all and training), expressed / model-gene / target flags, and the key and route each source matched by.
  * `embeddings/genept_all_genes.npz` (22,640 × 1536) and `embeddings/string_n2v_all_genes.npz` (16,165 × 64) hold the vectors, keyed by matrix symbol with Ensembl ids.
  * Coverage summary: `embeddings/all_genes_coverage.json`.
* **Matching.**
  * GenePT is matched by symbol: matrix symbol (21,156 genes), then the current HGNC symbol for the gene's Ensembl id (1,484).
  * STRING is matched by Ensembl gene id through STRING's alias file (19,337), with symbol or unique-alias fallbacks (170).
  * The rows for the 105 targets are identical to the target files the decoders used (105/105 GenePT, 93/93 STRING).
* **What is missing.**
  * Among expressed genes, 2,869 lack GenePT; 2,653 of them are lncRNA or clone-named loci (RP11-…, AC…, LINC…) with no NCBI summary to embed.
  * 5,555 lack STRING node2vec: 3,557 non-coding loci plus about 2,000 protein-coding genes with no interaction scoring ≥ 700. Lowering the STRING threshold (e.g. 400) would cover many of the latter.
* **GenePT aliases.**
  * GenePT stores aliases as references to one shared vector: 93,800 symbols, 33,230 distinct genes. About 1,000 matrix genes therefore share a vector with another matrix gene.
  * The pickle is now read opcode by opcode into a float32 cache (`data/interim/genept_ada.npz`). Nothing is executed, and memory stays near 0.6 GB instead of several GB of Python floats.

## 6. Decoders and evaluation

### Setup

* **Decoder.** Latent additive `x̂ = dec(enc(g(x_ctrl)) + W·e)` and decoder-only `x̂ = dec(W·e)`. `enc` and `dec` are 2-hidden-layer MLPs with LayerNorm and ReLU. `W` is linear with no bias, d → latent. Doubles use `W·(e_a + e_b)`.
* **State input `g`.** PCA-50 (`X_pca_g`) for every run. The scVI embeddings are built but not yet used as decoder input.
* **Training.**
  * Targets are training single-perturbation cells plus training controls (controls get e = 0). Each target gets a random training control from the same gem group as its input.
  * Loss is MSE on log-normalised expression of the 5,028 genes.
  * AdamW, batch 256, at most 150 epochs, early stopping (patience 8) on validation-single loss. The cap was first 60; nine runs reached it and were re-run at 150 (see Caveats).
  * Embeddings are scaled by one constant to mean row norm 1.
* **Hyperparameters.** Chosen once on validation singles with one-hot and the latent-additive decoder (`results/tuning.csv`, 9 configs): **hidden 1024, latent 128, dropout 0, lr 3e-4**, at Pearson Δ (top-20 DE) 0.965. They are fixed for every embedding and both decoders. Test doubles were never used for any choice.
* **Prediction.** Each perturbation's predicted mean averages 500 control inputs drawn from that split's control pool, gem-matched to the perturbation's own cells.
* **Metrics.** Computed per perturbation, then averaged:
  * MSE of the mean profile on all 5,028 genes and on the top-20 DE genes.
  * Pearson correlation of predicted vs observed change from the control mean, on all genes and on the top-20 DE genes.
  * DE genes come from Wilcoxon tests against controls, ranked by |score|. Singles use training cells; test doubles use their own cells vs test controls.
* **Stricter metrics (added after the first pass).**
  * *Without targets:* the top-20 DE metrics after removing the activated genes, which CRISPRa pushes up and any model that knows the target gets for free. DE lists are stored 50 deep so 20 genes remain.
  * *Interaction:* doubles only. The interaction is profile minus additive reference (test control + Δa + Δb from training singles), i.e. what addition cannot explain. *Interaction r* correlates predicted with observed interaction, on all genes and on the 20 genes with the largest observed interaction.
  * *Noise ceiling:* split each double's cells in half and correlate the halves' profiles (Spearman-Brown, square-rooted). It is the best score any predictor could reach against noisy observed means.
  * *Most non-additive quarter:* the 32 test doubles whose observed interaction is largest relative to their total change on the top-20 DE genes.
  * All of these are evaluation-only and use no fitting on doubles.
* **Seeds.** 3 per run (0, 1, 2). Mean ± std over seeds below. Baselines are deterministic, so they have one value.
* **Raw outputs.** `results/week1.csv` (method, decoder, split, seed, metric, value), per-perturbation values in `results/week1_per_perturbation.csv`, run info in `results/train_info.csv`.

### Test: 127 held-out doubles (composition)

| method | decoder | MSE all | MSE top-20 DE | Pearson Δ all | Pearson Δ top-20 DE |
|---|---|---|---|---|---|
| additive expression (Δa + Δb) | baseline | 0.0023 | 0.0574 | 0.893 | 0.968 |
| control mean (no change) | baseline | 0.0085 | 0.5507 | n/a | n/a |
| one-hot | latent additive | 0.0019 ± 0.0001 | 0.0621 ± 0.0047 | 0.867 ± 0.001 | 0.960 ± 0.001 |
| one-hot | decoder only | 0.0020 ± 0.0001 | 0.0616 ± 0.0022 | 0.875 ± 0.011 | 0.963 ± 0.002 |
| PCA loadings (K=32) | latent additive | 0.0028 ± 0.0001 | 0.1234 ± 0.0031 | 0.824 ± 0.005 | 0.923 ± 0.005 |
| PCA loadings (K=32) | decoder only | 0.0029 ± 0.0001 | 0.1388 ± 0.0077 | 0.825 ± 0.004 | 0.916 ± 0.006 |
| GenePT | latent additive | 0.0024 ± 0.0006 | 0.0720 ± 0.0040 | 0.859 ± 0.016 | 0.950 ± 0.009 |
| GenePT | decoder only | 0.0021 ± 0.0002 | 0.0661 ± 0.0022 | 0.867 ± 0.006 | 0.954 ± 0.005 |
| STRING node2vec | latent additive | 0.0032 ± 0.0002 | 0.1229 ± 0.0076 | 0.781 ± 0.011 | 0.887 ± 0.002 |
| STRING node2vec | decoder only | 0.0029 ± 0.0002 | 0.1395 ± 0.0097 | 0.800 ± 0.015 | 0.895 ± 0.007 |
| STRING node2vec, 108 covered doubles | latent additive | 0.0026 ± 0.0002 | 0.0808 ± 0.0079 | 0.821 ± 0.011 | 0.934 ± 0.002 |
| STRING node2vec, 108 covered doubles | decoder only | 0.0023 ± 0.0002 | 0.1013 ± 0.0103 | 0.845 ± 0.015 | 0.951 ± 0.004 |

### Validation: 105 singles (held-out cells of seen perturbations)

| method | decoder | MSE all | MSE top-20 DE | Pearson Δ all | Pearson Δ top-20 DE |
|---|---|---|---|---|---|
| training-cell mean (Δa) | baseline | 0.0012 | 0.0069 | 0.770 | 0.968 |
| control mean | baseline | 0.0045 | 0.2542 | n/a | n/a |
| one-hot | latent additive | 0.0013 ± 0.0000 | 0.0090 ± 0.0009 | 0.743 ± 0.007 | 0.959 ± 0.005 |
| one-hot | decoder only | 0.0013 ± 0.0001 | 0.0069 ± 0.0003 | 0.756 ± 0.013 | 0.967 ± 0.002 |
| PCA loadings (K=32) | latent additive | 0.0013 ± 0.0000 | 0.0112 ± 0.0010 | 0.758 ± 0.007 | 0.966 ± 0.002 |
| PCA loadings (K=32) | decoder only | 0.0012 ± 0.0000 | 0.0072 ± 0.0004 | 0.765 ± 0.002 | 0.968 ± 0.001 |
| GenePT | latent additive | 0.0013 ± 0.0001 | 0.0103 ± 0.0031 | 0.749 ± 0.012 | 0.952 ± 0.015 |
| GenePT | decoder only | 0.0013 ± 0.0000 | 0.0093 ± 0.0018 | 0.747 ± 0.013 | 0.956 ± 0.009 |
| STRING node2vec | latent additive | 0.0017 ± 0.0000 | 0.0375 ± 0.0004 | 0.666 ± 0.008 | 0.865 ± 0.022 |
| STRING node2vec | decoder only | 0.0017 ± 0.0000 | 0.0358 ± 0.0011 | 0.668 ± 0.011 | 0.844 ± 0.017 |
| STRING node2vec, 93 covered singles | latent additive | 0.0013 ± 0.0000 | 0.0092 ± 0.0010 | 0.744 ± 0.009 | 0.956 ± 0.005 |
| STRING node2vec, 93 covered singles | decoder only | 0.0012 ± 0.0000 | 0.0068 ± 0.0005 | 0.753 ± 0.007 | 0.964 ± 0.004 |

### Beyond additivity: test doubles

| method | decoder | Interaction r, all genes | Interaction r, top-20 int. genes | Pearson Δ top-20 DE, no targets | MSE top-20 DE, no targets |
|---|---|---|---|---|---|
| additive expression (Δa + Δb) | baseline | n/a | n/a | 0.968 | 0.0579 |
| control mean (no change) | baseline | 0.375 | 0.376 | n/a | 0.5241 |
| noise ceiling (split-half) | baseline | 0.890 | 0.981 | 0.997 | n/a |
| one-hot | latent additive | 0.385 ± 0.022 | 0.441 ± 0.052 | 0.959 ± 0.002 | 0.0597 ± 0.0049 |
| one-hot | decoder only | 0.362 ± 0.029 | 0.395 ± 0.018 | 0.962 ± 0.002 | 0.0599 ± 0.0024 |
| PCA loadings (K=32) | latent additive | 0.344 ± 0.012 | 0.292 ± 0.026 | 0.921 ± 0.006 | 0.1220 ± 0.0028 |
| PCA loadings (K=32) | decoder only | 0.364 ± 0.004 | 0.307 ± 0.009 | 0.916 ± 0.006 | 0.1371 ± 0.0074 |
| GenePT | latent additive | 0.342 ± 0.080 | 0.401 ± 0.043 | 0.951 ± 0.007 | 0.0681 ± 0.0040 |
| GenePT | decoder only | 0.344 ± 0.028 | 0.386 ± 0.027 | 0.954 ± 0.004 | 0.0635 ± 0.0017 |
| STRING node2vec | latent additive | 0.297 ± 0.019 | 0.341 ± 0.029 | 0.885 ± 0.002 | 0.1165 ± 0.0076 |
| STRING node2vec | decoder only | 0.348 ± 0.013 | 0.346 ± 0.022 | 0.894 ± 0.008 | 0.1326 ± 0.0093 |
| STRING node2vec, 108 covered doubles | latent additive | 0.320 ± 0.022 | 0.395 ± 0.033 | 0.932 ± 0.003 | 0.0778 ± 0.0078 |
| STRING node2vec, 108 covered doubles | decoder only | 0.381 ± 0.015 | 0.403 ± 0.026 | 0.948 ± 0.005 | 0.0975 ± 0.0099 |

### Most non-additive quarter of test doubles (32)

| method | decoder | Pearson Δ top-20 DE | Interaction r, all genes | Interaction r, top-20 int. genes | MSE top-20 DE |
|---|---|---|---|---|---|
| additive expression (Δa + Δb) | baseline | 0.946 | n/a | n/a | 0.1228 |
| control mean (no change) | baseline | n/a | 0.566 | 0.581 | 0.4386 |
| noise ceiling (split-half) | baseline | 0.995 | 0.919 | 0.991 | n/a |
| one-hot | latent additive | 0.930 ± 0.007 | 0.523 ± 0.017 | 0.588 ± 0.048 | 0.0800 ± 0.0043 |
| one-hot | decoder only | 0.936 ± 0.003 | 0.479 ± 0.049 | 0.562 ± 0.024 | 0.0744 ± 0.0058 |
| PCA loadings (K=32) | latent additive | 0.903 ± 0.006 | 0.464 ± 0.011 | 0.499 ± 0.025 | 0.0950 ± 0.0011 |
| PCA loadings (K=32) | decoder only | 0.907 ± 0.007 | 0.528 ± 0.005 | 0.563 ± 0.020 | 0.1018 ± 0.0068 |
| GenePT | latent additive | 0.925 ± 0.009 | 0.400 ± 0.164 | 0.469 ± 0.167 | 0.0902 ± 0.0204 |
| GenePT | decoder only | 0.925 ± 0.009 | 0.399 ± 0.098 | 0.475 ± 0.101 | 0.0951 ± 0.0136 |
| STRING node2vec | latent additive | 0.856 ± 0.002 | 0.396 ± 0.027 | 0.479 ± 0.005 | 0.1262 ± 0.0070 |
| STRING node2vec | decoder only | 0.882 ± 0.008 | 0.493 ± 0.014 | 0.532 ± 0.015 | 0.1137 ± 0.0068 |

Tables regenerate from `results/week1.csv` with `python -I scripts/report_tables.py`. One-hot, PCA loadings and GenePT cover every target, so their "covered" numbers equal the overall ones; the full covered tables are printed by `python -I -m pertact.evaluate table`.

### Research questions

The project asks whether perturbation embeddings *compose*: can a double be predicted from its two singles as `g(x) + f(a) + f(b)`? Each question below gives the answer, the evidence (from the tables above and the sequential section below) and why. Pairs of numbers are latent additive / decoder-only, 3-seed means; Pearson Δ is on the top-20 DE genes unless stated.

**Q1. Does any embedding predict held-out doubles better than simply adding the two singles' measured effects?**
* *Answer: no.*
* *Evidence:*
  * The additive expression baseline (control + Δa + Δb) scores 0.968.
  * One-hot 0.960 / 0.963, GenePT 0.950 / 0.954, PCA loadings 0.923 / 0.916, STRING 0.887 / 0.895.
  * One-hot beats the baseline on 44 of 127 doubles. It has the lowest all-gene MSE (0.0019 vs 0.0023).
* *Why:* Every test-double gene was seen as a single, and Norman doubles are mostly near-additive, so the measured sum is close to the best pattern available. The decoder adds two learned latent shifts, which at best reproduces that sum. One-hot does so most faithfully because each gene gets a free shift. Its lower MSE comes from damping the sum (Q2).

**Q2. Can any model capture the part of a double that addition cannot explain (the interaction)?**
* *Answer: no.*
* *Evidence:* interaction r on all genes, against "predict no change" at 0.375 and a noise ceiling of 0.890:

  | method | interaction r |
  |---|---|
  | one-hot | 0.385 / 0.362 |
  | GenePT | 0.342 / 0.344 |
  | PCA loadings | 0.344 / 0.364 |
  | STRING | 0.297 / 0.348 |

  On the 32 most non-additive doubles, "no change" scores 0.566 and the models 0.40–0.53. On the top-20 interaction genes only one-hot edges above the reference (0.441 vs 0.376; ceiling 0.981).
* *Why:* Doubles are mostly sub-additive, moving less than Δa + Δb. "No change" therefore earns credit by pure shrinkage, and the models reproduce only that dampening. They never see a double in training, so they cannot learn gene-specific interactions. The ceiling shows the signal is real.

**Q3. Which embeddings compose best, and why do PCA loadings fall behind?**
* *Answer:* one-hot > GenePT > STRING > PCA loadings.
* *Evidence:* change from validation singles to test doubles (latent additive):

  | embedding | val singles → test doubles | drop |
  |---|---|---|
  | one-hot | 0.959 → 0.960 | +0.001 |
  | GenePT | 0.952 → 0.950 | −0.002 |
  | STRING (covered) | 0.956 → 0.934 | −0.022 |
  | PCA loadings | 0.966 → 0.923 | −0.043 |

* *Why:* With 32 dimensions for 105 genes, PCA-loading shifts share directions, so their latent sums land where the decoder never trained. Its single predictions are fine: averaging them in expression space gives 0.961 (Q4). STRING's overall scores (0.865 val, 0.887 test) are lower mainly because 12 genes have no vector.

**Q4. Does applying the perturbations one after the other (sequential composition) work better than adding them?**
* *Answer: no.* It reduces to averaging the two singles.
* *Evidence:*
  * Pearson Δ rises for the weaker embeddings (PCA loadings 0.923 → 0.961), but top-20 MSE roughly doubles to triples (one-hot 0.062 → 0.170).
  * The a→b prediction correlates 0.99 with single b and 0.42 with single a.
  * The predicted change is about half the observed size (0.47–0.50).
* *Why:* The decoder barely uses its starting state (Q6), so the second step overwrites the first. Pearson ignores scale, which is the only reason it improves. Details are in the next section.

**Q5. Do the embeddings differ on new cells of perturbations seen in training?**
* *Answer: no.*
* *Evidence:* Validation singles score: one-hot 0.959 / 0.967, PCA loadings 0.966 / 0.968, GenePT 0.952 / 0.956, STRING on its 93 covered genes 0.956 / 0.964. The training-cell mean scores 0.968.
* *Why:* Seen perturbations only need memorising. This test checks the pipeline; it cannot separate the embeddings.

**Q6. Does the cell's starting state help (latent additive vs decoder-only)?**
* *Answer: not measurably.*
* *Evidence:* Latent additive and decoder-only differ by at most 0.008 on every embedding, in both directions.
* *Why:* Metrics on mean profiles average away cell-to-cell variation, the only thing the state could explain. Distribution-level metrics (energy distance, MMD) are needed.

**Q7. Is the headline metric inflated by the activated genes themselves?**
* *Answer: not for doubles.*
* *Evidence:* Removing the targets from the top-20 DE list: additive 0.968 → 0.968, one-hot 0.960 → 0.959 and 0.963 → 0.962. Val singles drop slightly (training mean 0.968 → 0.955).
* *Why:* The top genes of doubles are dominated by downstream responses.

**Q8. Were results limited by training length?**
* *Answer: no.*
* *Evidence:* Raising the epoch cap from 60 to 150 moved the nine capped runs by at most 0.005. All of them early-stopped by themselves (see Caveats).

### Sequential vs additive composition

**Setup.** `python -I -m pertact.evaluate compose` takes each latent-additive model (4 embeddings × 3 seeds, now saved in `models/decoders/`) and predicts every test double two ways, from the same 500 control inputs:

* **Additive** (as above): `dec(enc(s) + W(e_a + e_b))`, one pass.
* **Sequential**: apply a, re-encode the predicted profile with the frozen PCA `g`, then apply b: `ŷ_a = dec(enc(s) + W e_a)`, `x̂ = dec(enc(g(ŷ_a)) + W e_b)`.
* Sequential is run in both orders and averaged, because simultaneous CRISPRa has no order. Rows are labelled `latent_additive_seq`.
* Decoder-only cannot be run sequentially, because it ignores the starting state.
* The additive predictions reproduce the saved ones exactly (max difference 0).

| embedding | composition | Pearson Δ top-20 DE | Pearson Δ all | MSE top-20 DE | interaction r, all genes | non-additive quarter: Pearson Δ / interaction r |
|---|---|---|---|---|---|---|
| additive expression (Δa + Δb) | baseline | 0.968 | 0.893 | 0.057 | n/a | 0.946 / n/a |
| control mean (no change) | baseline | n/a | n/a | 0.551 | 0.375 | n/a / 0.566 |
| one-hot | additive | 0.960 ± 0.001 | 0.867 ± 0.001 | 0.062 ± 0.005 | 0.385 ± 0.022 | 0.930 / 0.523 |
| one-hot | sequential | 0.957 ± 0.002 | 0.831 ± 0.010 | 0.170 ± 0.004 | 0.385 ± 0.005 | 0.928 / 0.560 |
| PCA loadings | additive | 0.923 ± 0.005 | 0.824 ± 0.005 | 0.123 ± 0.003 | 0.344 ± 0.012 | 0.903 / 0.464 |
| PCA loadings | sequential | 0.961 ± 0.003 | 0.857 ± 0.010 | 0.157 ± 0.006 | 0.390 ± 0.007 | 0.939 / 0.566 |
| GenePT | additive | 0.950 ± 0.009 | 0.859 ± 0.016 | 0.072 ± 0.004 | 0.342 ± 0.080 | 0.925 / 0.400 |
| GenePT | sequential | 0.960 ± 0.001 | 0.855 ± 0.005 | 0.158 ± 0.010 | 0.405 ± 0.010 | 0.934 / 0.576 |
| STRING node2vec (108 covered doubles) | additive | 0.934 ± 0.002 | 0.821 ± 0.011 | 0.081 ± 0.008 | 0.320 ± 0.022 | |
| STRING node2vec (108 covered doubles) | sequential | 0.958 ± 0.004 | 0.840 ± 0.003 | 0.159 ± 0.011 | 0.412 ± 0.003 | |

**What it shows**

1. **Sequential lifts the pattern scores of the weaker embeddings and evens them out.** Pearson Δ top-20 DE: PCA loadings 0.923 → 0.961, GenePT 0.950 → 0.960, STRING (covered) 0.934 → 0.958. One-hot is unchanged (0.960 → 0.957). Every embedding lands near 0.96, still below the additive expression baseline (0.968), and beats that baseline on only 12–19 of 127 doubles.
2. **The size gets worse.** Top-20 MSE roughly doubles to triples (e.g. one-hot 0.062 → 0.170). The predicted change is about half the observed size: median ratio 0.47–0.50, against 0.65–0.82 for additive decoding and 1.03 for the additive expression baseline.
3. **The second perturbation overwrites the first.**
   * The a→b prediction correlates 0.99 with the model's own prediction for single b, and only about 0.42 with single a.
   * Each order alone is poor: Pearson Δ 0.74–0.80, with the better order 0.92 and the worse 0.60.
   * The two orders agree only at r ≈ 0.44.
   * Averaging the orders gives the mean of the two single predictions (r 0.98–0.99, size ratio 1.00).
   * So sequential composition through this decoder is in effect "average the two singles". It gains on Pearson only because Pearson ignores scale, and half of Δa + Δb points the same way as Δa + Δb.
4. **Its interaction gain is the same shrinkage artefact.** Half the sum has interaction −(Δa + Δb)/2, which is proportional to the "no change" baseline's. Sequential interaction r therefore converges to that baseline: 0.38–0.41 vs 0.375 on all doubles, 0.56–0.58 vs 0.566 on the non-additive quarter. Its seed-to-seed spread also collapses (GenePT ±0.080 → ±0.010).
5. **Why the second step overwrites the first.** The decoder barely uses its starting state, which decoder-only ≈ latent additive already showed. So "apply b to a cell already moved by a" gives the same answer as "apply b to a control cell". Sequential composition is only meaningful when the action genuinely depends on the state it acts on.
6. **PCA loadings' additive problem is in the latent sum, not the singles.** Averaging its single predictions in expression space (0.961) beats summing its embeddings in latent space (0.923): `W(e_a + e_b)` lands where the decoder never trained.

Per-double order effects are in `results/sequential_orders.csv`; sequential predictions (both orders and their mean) are in `results/predictions/*__latent_additive_seq__*.npz`.

### Caveats
* **Epoch cap checked: undertraining was not the explanation.** With the first 60-epoch cap, 9 of 24 runs stopped at the cap with their best epoch at 58: all six PCA-loadings runs, one-hot decoder-only seeds 0–1, and GenePT latent-additive seed 1. All nine were re-run with a 150-epoch cap, which is equivalent to re-running all 24, because the other 15 had already early-stopped below 60. Every re-run then stopped on its own, at epochs 67–85.
  * Three re-runs restored the same best epoch (58), so their scores are identical to before.
  * The other six moved by at most 0.005.
  * The 60-epoch results are kept in `results/week1_cap60.csv` and `results/train_info_cap60.csv`.

  | run | test Pearson Δ top-20 DE, cap 60 | cap 150 | epochs (best), cap 150 |
  |---|---|---|---|
  | PCA loadings, latent additive, seeds 0 / 1 / 2 | 0.917 / 0.922 / 0.925 | 0.917 / 0.926 / 0.926 | 67 (58) / 69 (60) / 82 (73) |
  | PCA loadings, decoder only, seeds 0 / 1 / 2 | 0.914 / 0.915 / 0.915 | 0.919 / 0.920 / 0.910 | 85 (76) / 78 (69) / 81 (72) |
  | one-hot, decoder only, seeds 0 / 1 | 0.962 / 0.961 | 0.965 / 0.961 | 85 (76) / 67 (58) |
  | GenePT, latent additive, seed 1 | 0.955 | 0.955 | 67 (58) |

* **Retraining reproduced the earlier runs exactly.** All 24 runs were retrained to compute the new metrics, because the first pass saved scores but not predictions. Every earlier metric came back identical (96 of 96 values). The first-pass file is kept as `results/week1_v1_4metrics.csv`. Predictions are now saved in `results/predictions/`, and `python -I -m pertact.evaluate rescore` recomputes any metric without retraining.
* **MPS is not fully deterministic.** Runs used Apple-silicon GPU (MPS); seeds fix data order and initialisation, but MPS kernels are not bit-deterministic, so re-runs can differ in the third decimal.
* **The test is in-distribution for genes.** All test-double genes are training singles. This test cannot show the main advantage of non-one-hot embeddings, which is generalising to unseen genes.

## 7. Decisions, surprises and problems

**Data and labels**
* **Total is below the expected count.** It comes out at 90,867 cells, not ~105k (−13.5%, inside your 20% threshold). The ~105k figure matches the 104,507 good-coverage cells *before* removing the 13,339 cells with more than one guide identity. Those are droplet multiplets (1.5× the UMIs) that can carry 3–4 guides, so they were dropped as instructed. This matches the singlet-only count used by GEARS (~91k).
* **127 doubles, not 131.** Four doubles fall below 50 cells after the multiplet filter.
* **Every double gene also exists as a single.** The 131 designed doubles involve 73 distinct genes, all of them among the 105 singles. The remaining 32 single genes are never paired. So every one of the 127 kept doubles has both components in training, and the excluded-doubles list is empty.
* **Doubles are concentrated on a few hub genes.** PTPN12 is in 12 doubles, KLF1 and CEBPE in 11, UBASH3B in 10; 20 genes appear in only one. A few hubs will weigh heavily in any average over doubles.
* **Orientation merged.** `A_B` and `B_A` constructs, and the four non-targeting control constructs, were each merged into one condition (48 conditions have more than one construct).
* **Outdated symbols resolved.** Three guide symbols (C19orf26, C3orf72, KIAA1804) don't match the matrix's gene names. They were mapped through HGNC/Ensembl, and embedding tables stay keyed by guide symbol.
* **Weak CRISPRa activation.** The target gene's log-expression in its own single rises by a median of +0.33 over control. For 18 of the 105 singles the rise is under 0.1, and CKS1B (−0.16), CELF2 (−0.03) and ZBTB25 (0.00) show none. These perturbations may carry little signal, so a per-perturbation look is advisable before reading too much into them.

**Splits and evaluation choices**
* **Test control pool added.** The spec names train/val control shares only. I added a 15% test pool, gem-stratified, so test doubles are compared against held-out controls and the decoder draws held-out control inputs.
* **Control cells are decoder targets with e = 0.** This gives the zero vector a defined meaning ("no perturbation"), which is also how uncovered genes are handled in the "overall" numbers.
* **Embedding scaling.** Each table is multiplied by one constant so its mean row norm over training genes is 1. One-hot and GenePT are unchanged by this; PCA loadings and node2vec are rescaled. There is no centering, so `e(a) + e(b)` keeps its meaning.
* **Composition is exactly additive in latent space.** W is linear with no bias, as specified, so `W(e_a + e_b) = W e_a + W e_b`. With both components seen in training, a double's latent shift is fixed by the two single shifts. The embedding then matters only through the rank of `W·e` and the optimiser's implicit bias, which makes one-hot the most flexible option rather than a lower bound. Embeddings should separate more clearly on held-out genes, or with a nonlinear map from e to the latent.
* **Decoder-only is a weak lower bound under mean-level metrics.** It still learns one mean per seen perturbation. Conditioning on the control state mainly helps cell-to-cell variation, which averaging over cells removes.

**Engineering problems (all fixed)**
* **numba × PyTorch segfault.** scanpy's Wilcoxon segfaulted when PyTorch was already loaded, an OpenMP clash on macOS. Fixed by setting `NUMBA_THREADING_LAYER=workqueue` in `pertact/__init__.py`.
* **scanpy × scipy UMAP failure.** scanpy 1.11's `sc.pp.neighbors` calls `csr_array.getnnz`, which scipy 1.17 removed, so UMAPs are computed with umap-learn directly.
* **PecanPy needed a dependency upgrade.** It pulls `nptyping`, which breaks on NumPy 2. Upgrading to `nptyping` 2.5 pinned NumPy to 1.26.
* **PecanPy walks were not reproducible.** They varied run to run under parallel numba, so walks use one numba thread and gensim one worker. Identical output was verified on a test graph.
* **gensim's Cython dot product logged 9 swallowed exceptions** over about 13M training words. The resulting vectors check out (edge-vs-random AUC 0.988, sensible neighbours).
* **The laptop slept with the lid closed for several hours.** Jobs paused and results are unaffected. Wall-clock seconds for two grid configs in `results/tuning.csv` include that sleep.
* **One grid config with dropout 0.1 scored worse.** At 512/64/1e-3 it gave 0.926 vs 0.954 for dropout 0, so dropout was fixed at 0 for the rest of the grid to save compute on battery.
