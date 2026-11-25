# Evaluation design: gene representations, OOD perturbations, selection

Status: proposal for discussion, 2026-11-15. No code has been written yet. The item-by-item protocol (splits,
metrics, baselines, statistics) is in `tmp/eval_protocol.md`, which supersedes the Track A–C details below.

**Purpose.** Settle how we evaluate before more models get built: what is scored, on which held-out data, and how the
code is laid out so that everyone's models are scored the same way and the numbers can be compared.

## 0. Where we are, and the gap

What we have now (`metrics.py`, `runs.py`, `retrieval.py`, `doubles.py`):

| question | where | held-out unit |
|---|---|---|
| predict the expression of a seen single | `EvalRef` on `val` | cells of the 105 training singles |
| predict a double from two seen singles | `EvalRef` on `test`, `doubles.py` | 127 doubles, **both genes seen** |
| does s(ctrl) + f(a) land on the right cell / name the gene | `retrieval.py` | cells of seen singles |

**Gap: no current test holds out a perturbation.** `splits.json` puts all 105 singles in `train_singles`, and
`val_singles` are the same genes (other cells). So every number so far is in-distribution in the perturbation
axis. A one-hot embedding can't say anything about a gene it hasn't seen, which is the case we care about next. It
should score at chance on any OOD test and becomes the "no transfer" floor.

The design below adds three evaluation tracks that share one harness:

* **A. Gene representation (probing)**: what a gene vector (GenePT, STRING-n2v, PCA loadings, learned f(a)) encodes.
* **B. OOD perturbation prediction**: predict the effect of a perturbation whose gene(s) were never trained on.
* **C. OOD perturbation selection**: rank unseen candidates for a goal; later, **D. generation** of an action that
  reaches a target.

## 1. Track A: gene representation probing

Input: any table `gene -> vector`. Nothing about cells. Three probe families, from cheap to closest to our goal.

**A1. Annotation probes (what does the vector know about biology).**
Labels: GO BP slim, Reactome top-level pathways, TF vs non-TF (Lambert 2018 list), CORUM complexes, STRING
neighbours (held out from STRING-n2v's own training graph, so only for non-STRING embeddings).
Probe: logistic regression (L2, C picked by inner CV) and kNN (k = 10, cosine). Both, because a linear probe measures
linearly decodable information and kNN measures the local geometry that OOD transfer by similarity actually relies on.
Gene set: the intersection that every compared embedding covers, from `embeddings/all_genes_coverage.json`
(GenePT ∩ STRING-n2v on expressed genes ≈ 10k). Report the intersection size.
Metrics: AUROC and AUPRC per label, macro-averaged; labels with < 10 positives dropped.

**A2. Effect probes (does the vector predict what the perturbation does).** This connects A to B.
Target: the training mean shift of each of the 105 singles, in PCA-50, in scVI-30, and as top-DE expression change.
Probe: ridge from gene vector to effect, **leave-genes-out CV** over the 105 (5 folds, same folds as Track B).
Metrics: cosine and Pearson of the predicted vs real shift, the normalised miss `||pred - real||² / ||real||²`
(1 = no better than predicting zero), and effect-strength Spearman.
n = 105 is small. Report bootstrap CIs over genes and a permutation null (shuffle gene labels on the vectors).

**A3. Geometry agreement (RSA).** Spearman between the gene–gene similarity matrix of an embedding and the gene–gene
similarity of real effects (105 × 105, from training cells), and ARI between embedding clusters and the Norman 2019
programs (erythroid, granulocyte/megakaryocyte, pro-growth, G1 arrest, …). Cheap, label-free, and a good sanity
plot for the meeting.

Learned actions f(a) only exist for the 105 trained genes, so for them only A2 (in CV) and A3 apply. A1 on 105 genes
is too small to be worth a number.

## 2. Track B: OOD perturbation prediction

**New split files.** These are separate files. `splits.json` stays as it is, so earlier weeks still reproduce.

| split file | held out | used for |
|---|---|---|
| `splits/ood_singles_k5.json` | 5 folds over the 105 genes; fold k's genes are removed from **all** training (cells, DE, actions, decoders) | unseen singles |
| `splits/ood_doubles_k5.json` | same gene folds; each test double labelled `seen2` / `seen1` / `seen0` by how many of its genes are in the training fold | unseen-gene doubles |

Rules for the folds:
* Stratify by effect tier (`actions.effect_tiers`) and by Norman program, so each fold has strong and weak genes and
  no program sits entirely in one fold.
* Optional harder variant: hold out whole programs (`ood_program`). This tests transfer to an unseen biology, not
  just an unseen gene.
* Controls keep the existing train/val/test partition.
* Each split file stores its own SHA-1. Every result row records it.

**What to compute.** Reuse `EvalRef.metrics` unchanged (MSE, Pearson-Δ on all genes, on DE-20 and on DE-20 minus
targets) on the held-out conditions, plus the state-space centre scores from `doubles.centre_scores`. For doubles,
report each of `seen2` / `seen1` / `seen0` separately, and keep the interaction metrics.

**Fixed baselines** (every table):
* control mean (no effect); mean of all training singles' shifts ("generic perturbation"), which is the baseline
  most published OOD models struggle to beat;
* **nearest-gene copy**: copy the effect of the training gene closest in the embedding (one row per embedding). Any
  model that can't beat this has not learned more than the embedding's neighbourhood.
* one-hot through the same model (the floor: should equal the generic baseline);
* noise ceiling (`metrics.noise_ceiling`, split-half Spearman–Brown).

DE genes for held-out singles come from their own test cells (like doubles today), never from training cells.

## 3. Track C: OOD perturbation selection (and D: generation)

The question changes from "predict the effect" to "pick the right perturbation". A model can rank candidates
correctly even when its profiles are off. Selection is scored retrospectively on the Track B folds: the candidates
are the held-out genes (plus all 105 for the naming tasks), and the truth is their measured effect.

| task | goal given to the model | truth | metrics |
|---|---|---|---|
| **C1 strongest effect** | none | measured effect size of held-out genes | Spearman, precision@k, NDCG@k |
| **C2 reach a target state** | a target cell population (mean state or a set of cells) | the gene that produced it | rank of true gene, top-1/5, MRR (Week 2/3 "naming", now with unseen candidates) |
| **C3 move along a program** | a direction (e.g. erythroid score up) | projection of each held-out gene's real shift | Spearman, hit@k of the top decile |
| **C4 non-additive pairs** | candidate pairs | measured interaction strength (`EvalRef.strength`) | precision@k, AUPRC for the top quarter |
| **C5 budgeted / active** (later) | pick k genes per round, see their data, retrain | cumulative best target score vs number of experiments | area under the discovery curve vs random and vs nearest-gene |

C2 is the one we can run first: `doubles.rank_true` and `retrieval.identification` already do the ranking. They only
need candidates whose action comes from the embedding (for unseen genes) instead of from training pairs.

**D. Generation (interface only for now).** The model outputs an action vector or an embedding, not a gene. Score:
(i) the decoded/predicted state's distance to the target, (ii) **realisability**, the distance to and rank of the
nearest real gene in the embedding table, and (iii) whether that nearest gene, looked up in the held-out data,
actually reaches the target (C2 metrics). Without (ii) and (iii), generation scores can't be checked against data.

## 4. Code layout

The rule: **models and evaluation meet only at a small adapter interface.** Evaluation code never imports training
code. Anyone's model is scored by writing an adapter.

```
src/pertact/
  ...existing modules (unchanged; metrics.py stays as the metric source of truth)
  bench/                      # new package; named bench/ so it doesn't clash with evaluate.py
    __init__.py
    interfaces.py             # the 4 adapter protocols below + result row schema
    registry.py               # @register_model("genept_ridge"), @register_task("ood_singles")
    splits_ood.py             # builds splits/ood_*.json; deterministic, hashed
    labels.py                 # GO / Reactome / TF / CORUM / Norman programs -> data/labels/*.parquet (versioned)
    probes.py                 # logistic, kNN, ridge probes with group-aware CV
    tasks/
      probe_annotation.py     # A1
      probe_effect.py         # A2
      geometry.py             # A3
      ood_predict.py          # B (calls metrics.EvalRef, doubles.centre_scores)
      select.py               # C1–C4
      generate.py             # D (stub)
    baselines.py              # control, generic, nearest-gene, one-hot, noise ceiling, wrapped as adapters
    runner.py                 # python -I -m pertact.bench run --task ... --model ... --seeds ...
    report.py                 # results/bench/**  ->  leaderboard tables and figures
configs/bench/                # one yaml per task: split file, gene set, probe settings, metrics
splits/                       # committed: ood_singles_k5.json, ood_doubles_k5.json
results/bench/<task>/<model>/s<seed>__f<fold>.csv + .meta.json
tests/bench/                  # tiny synthetic fixture; each task runs end to end in seconds
```

### 4.1 Adapter interfaces (`interfaces.py`)

```python
class GeneEmbedder(Protocol):                 # Track A, and the input side of B/C
    name: str
    def genes(self) -> list[str]: ...
    def embed(self, genes: list[str]) -> np.ndarray: ...       # NaN rows for uncovered genes; never imputed

class EffectPredictor(Protocol):              # Track B
    name: str
    def fit(self, ctx: FitContext) -> None: ...                # ctx holds ONLY the training fold
    def predict(self, conditions: list[str], ctx: PredictContext) -> dict[str, Prediction]: ...
    # Prediction carries optional .expr (mean profile, n_genes) and/or .state (shift in a named state space);
    # a task scores whichever it needs and errors clearly if missing.

class Ranker(Protocol):                       # Track C
    name: str
    def rank(self, candidates: list[str | tuple[str, str]], goal: Goal, ctx: PredictContext) -> np.ndarray: ...
    # Default: any EffectPredictor becomes a Ranker via `PredictorRanker(pred, goal_score)`.

class Generator(Protocol):                    # Track D
    def generate(self, goal: Goal, ctx: PredictContext, n: int) -> np.ndarray: ...   # action/embedding vectors
```

`FitContext` is built by the runner from the split file. It holds the training cells, the training genes and the
DE lists computed on training data. **The runner checks for leakage**: after `fit`, the adapter reports
`trained_on_genes()`, and the run fails if any held-out gene is in it. This one check guards against the most
likely OOD mistake.

### 4.2 Result schema (same for every task)

Long format, one row per metric:

```
task, model, split_file, split_sha, fold, seed, subset, metric, value, n, ci_lo, ci_hi
```

`subset` is e.g. `seen1`, `strong_tier`, `program=erythroid`, `covered`. The `.meta.json` next to each file stores
git SHA, dirty flag, config hash, embedding fingerprint (`runs.fingerprint`), package versions and wall time. The
same row format is produced by every track, so `report.py` is a single group-by, and anyone can add a model
without changing the reports.

### 4.3 Rules for a shared project

1. **One metric source.** Metrics live in `metrics.py` / `bench/` only. Scripts and notebooks import them and never
   re-implement a metric. A number in a report must come from `results/bench/`.
2. **Package vs scripts vs notebooks.** `bench/` is tested and deterministic. `scripts/` only turns result files into
   tables and figures and never fits anything. Notebooks are for exploration, and their numbers don't go in
   reports.
3. **Frozen evaluation.** Split files, label files and task configs are committed and versioned. Changing one means
   a new file name (`ood_singles_k5_v2.json`), not an edit.
4. **Pre-registration, as in Week 3.** Each task config lists its metrics and subsets before the first scored run.
   Test folds are never used for tuning. Hyper-parameters are tuned with an inner split inside the training fold.
5. **Seeds × folds.** Report mean ± sd over seeds and folds, and bootstrap CIs over genes. With 105 genes, a
   difference smaller than the CI is reported as no difference.
6. **Adding a model** = one adapter file + `@register_model` + one line in a model yaml. Adding a task = one
   file in `tasks/`, a config, and a synthetic test. Review checklist: leakage check passes, synthetic test passes,
   meta file written.
7. **Cheap by default** (shared cluster): tasks cache `FitContext` per fold under `data/cache/bench/`, and the
   runner takes `--dry-run` (prints what would run) and `--max-conditions` for smoke tests. Long runs go through
   SLURM like `scripts/week3_decoder.sbatch`.

## 5. Order of work

1. `splits_ood.py` + the leakage check + the fixed baselines on Track B singles. This shows how hard the OOD test is
   before any model work. *(first)*
2. A2 + A3 for the four existing embeddings and the learned actions (in CV). Small, and it tells us which
   embedding to build OOD models on.
3. B on doubles by `seen2/1/0`, and C2 (naming with unseen candidates), reusing `doubles.py` ranking.
4. A1 annotation probes (needs label downloads).
5. C1, C3, C4. Then C5 and D when there is a selection or generation model.

## 6. Questions for the meeting

* **Fold scheme**: random-stratified 5-fold over genes, held-out programs, or both? (Proposal: both; the 5-fold
  version is primary.)
* **Main OOD metric**: Pearson-Δ on DE-20 without targets, or the state-space normalised miss? (Proposal: report
  both; the state-space one is primary because it doesn't depend on the decoder.)
* **Label sources and versions** for A1 (GO slim release, Reactome version, TF list).
* **Selection goals** worth defining now for C3: which programs or target states matter for the project.
* **Second dataset** for cell-context OOD (e.g. Replogle K562 vs RPE1)? This changes whether `FitContext` needs a
  cell-type axis now. (Proposal: add the field now, with one value.)
* **Naming**: `bench/` vs `eval/`, and whether `metrics.py` moves into it later.
