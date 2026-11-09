# Perturbations as actions

Research code for treating a CRISPRa perturbation as an action: a vector f(a) added to a cell's state, so that
`state(control) + f(a) ≈ state(perturbed)`. The long-term question is composition: do f(a) and f(b), learned from
single perturbations, predict the double a+b? The data is Norman et al. 2019 (K562, CRISPRa).

* **Week 1** (`REPORT.md`): data, splits, frozen state encoders, perturbation embeddings, and the latent additive and decoder-only models.
* **Week 2** (`REPORT_week2.md`): entropic OT pairing, InfoNCE actions, retrieval of held-out cells, and the action-length addendum.
* **Progress site:** `site/index.html`. `scripts/build_site.py` refreshes only the Week 1 results; the Week 2 content is static HTML.

## Setup

Needs Python 3.11 and [uv](https://docs.astral.sh/uv/). The runs used an Apple M1 Pro with PyTorch on MPS.

`pyproject.toml` lists no dependencies, so the exact package versions come from `env.txt`. Its `-e` line points at an
old folder, so install only the pinned lines:

```bash
uv venv .venv --python 3.11
grep -E '^[A-Za-z0-9_.-]+==' env.txt > /tmp/pertact_requirements.txt
uv pip install --python .venv/bin/python --no-deps -r /tmp/pertact_requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

`uv pip check` reports that `zarr` wants numpy 2 or later while numpy is 1.26.4. The original runs had the same pins.

Commands below assume `source .venv/bin/activate`. The `-I` flag ignores the current directory and `PYTHONPATH`, so
the installed `pertact` package is what runs.

## Data

`data/` and `models/` are not in git. Put the raw files at the paths set in `config.yaml`:

| file | path under `data/raw/` | source |
|---|---|---|
| `NormanWeissman2019_filtered.h5ad` | `scperturb/` | scPerturb RNA v1.4, Zenodo 13350497 (md5 `c870e6967d91c017d9da827bab183cd6`) |
| `hgnc_complete_set.txt` | `hgnc/` | HGNC complete set |
| `GenePT_emebdding_v2.zip` | `genept/` | Zenodo 10833191 |
| `9606.protein.links.v12.0.txt.gz`, `.info.`, `.aliases.` | `string/` | STRING v12, human |

## Splits

| split | perturbed cells | control cells | used for |
|---|---|---|---|
| train | 37,770: 80% of each of the 105 singles | 6,781 (70%) | all fitting |
| validation | 9,444: the other 20% of each single | 1,453 (15%) | early stopping, tuning, Week 2 scores |
| test | 33,966: 127 doubles | 1,453 (15%) | final scores only; never used to fit or tune |

Validation contains the same 105 perturbations as training, and choices were made on it, so its scores are slightly
optimistic. `splits.json` is deterministic for seed 0.

## Rules for every experiment

* No leakage: HVGs, PCA, scVI, PCA loadings, DE genes for singles and all hyperparameters come from training or validation data only.
* Never use double-perturbation cells for fitting or tuning, and don't tune anything on test doubles.
* Fixed seeds everywhere; package versions logged to `env.txt`.
* Missing genes are not imputed.

## Reproducing the results

Saved decoders are reloaded when `models/decoders/<method>__<decoder>__s<seed>.pt` exists. Delete a file to retrain that model.

**Week 1**

```bash
python -I -m pertact.data qc                # cell-level filters, all genes
python -I -m pertact.splits                 # splits.json
python -I -m pertact.data hvg               # train-only genes -> processed h5ad
python -I -m pertact.encoders pca
python -I -m pertact.encoders scvi
python -I -m pertact.encoders umap
python -I -m pertact.encoders write_h5ad
python -I -m pertact.embeddings             # onehot pca_loadings genept string_n2v
python -I -m pertact.evaluate de
python -I -m pertact.evaluate baselines
python -I -m pertact.evaluate tune          # resumable
python -I -m pertact.evaluate run           # 4 embeddings x 2 decoders x 3 seeds
python -I -m pertact.evaluate compose       # additive vs sequential doubles
python -I -m pertact.evaluate rescore       # recompute metrics from saved predictions
python -I -m pertact.embeddings all_genes   # GenePT + STRING vectors for all measured genes
python -I -m pertact.evaluate table
python -I scripts/build_site.py
```

Refitting scVI or node2vec may not reproduce the saved values exactly. The cell states used in every result are in
`data/processed/state_embeddings.npz`.

**Week 2**

```bash
python -I -m pertact.ot --check             # POT agreement, synthetic shift recovery
python -I -m pertact.ot                     # OT pairs: both spaces, train and val
python -I -m pertact.infonce tune           # temperature on validation retrieval, seed 0
python -I -m pertact.infonce train          # both spaces x 3 seeds
python -I -m pertact.infonce anchor         # anchor-weight grid on seed 0, then 3 seeds
python -I -m pertact.retrieval              # run after train and anchor
python -I -m pertact.evaluate actions --variants infonce_scaled_per_action --tag main
python -I -m pertact.evaluate actions --variants infonce --references --tag raw
python -I -m pertact.evaluate actions --variants mean_shift --references --tag meanshift
python -I -m pertact.evaluate actions --variants infonce_unit --references --tag unit_infonce
python -I -m pertact.evaluate actions --variants mean_shift_unit --references --tag unit_meanshift
python -I scripts/week2_direction_diagnostics.py
python -I scripts/week2_decoder_summary.py
```

`pertact.retrieval` skips any InfoNCE or anchor file that does not exist, without an error. Check that
`embeddings/` has all 12 `infonce*` files (2 spaces x 3 seeds x 2 variants) before running it.

The `actions` command scores validation singles only and never loads a test cell.

## Layout

| path | contents | in git |
|---|---|---|
| `src/pertact/` | package: data, splits, encoders, embeddings, decoders, evaluation, OT, InfoNCE, retrieval | yes |
| `scripts/` | report tables, site refresh, Week 2 diagnostics and summaries | yes |
| `config.yaml`, `splits.json`, `env.txt` | settings, cell splits, package versions | yes |
| `embeddings/` | perturbation embeddings and learned actions | yes, except `genept_all_genes.npz` |
| `results/` | metrics, OT pairs, tuning, summaries | yes, except the two `predictions/` folders |
| `figures/`, `logs/`, `site/`, `REPORT*.md` | figures, run logs, progress site, write-ups | yes |
| `data/`, `models/` | raw and processed data, trained models | no |
