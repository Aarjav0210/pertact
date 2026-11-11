"""Scoring of predicted expression profiles: DE genes, the fixed references of a split, the metrics and the noise ceiling.

Metrics are computed per perturbation, then averaged over perturbations:
    mse_all / mse_de20                       : predicted vs observed mean expression
    pearson_delta_all / pearson_delta_de20   : predicted vs observed change (mean - control mean)
    mse_de20_nt / pearson_delta_de20_nt      : same on the top-20 DE genes after removing the perturbed target genes,
                                               which CRISPRa pushes up and any model that knows the target gets for free
    pearson_int_all / pearson_int_top20      : doubles only. Interaction = profile - additive reference
                                               (control + Δa + Δb from training singles); correlation of predicted vs
                                               observed interaction on all genes / on the 20 genes with the largest
                                               observed interaction
Each split is reported overall, on the conditions the embedding fully covers ("<split>_covered") and, for test,
on the most non-additive quarter of doubles ("test_nonadditive").
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import get_logger, path, read_json, write_json
from .data import CONTROL, condition_genes

log = get_logger("evaluate")
METRICS = ["mse_all", "mse_de20", "pearson_delta_all", "pearson_delta_de20",
           "mse_de20_nt", "pearson_delta_de20_nt", "pearson_int_all", "pearson_int_top20"]
NONADDITIVE_FRAC = 0.25


# --------------------------------------------------------------------------- data helpers
def group_means(adata, mask, conds) -> dict[str, np.ndarray]:
    """Mean expression of each perturbation's cells among the masked cells."""
    cond = adata.obs["condition"].astype(str).to_numpy()
    return {c: np.asarray(adata.X[mask & (cond == c)].mean(0)).ravel() for c in conds}


def split_mask(adata, name):
    """True for the cells of one split."""
    return (adata.obs["split"] == name).to_numpy()


def eval_conditions(splits, split):
    """What is scored in a split: the singles on validation, the doubles on test."""
    return splits["train_singles"] if split == "val" else splits["test_doubles"]


def train_deltas(adata, splits) -> dict[str, np.ndarray]:
    """Δa = mean(training cells of a) - mean(training controls), for every training single."""
    mu = group_means(adata, split_mask(adata, "train"), splits["train_singles"] + [CONTROL])
    return {c: mu[c] - mu[CONTROL] for c in splits["train_singles"]}


# --------------------------------------------------------------------------- DE genes
def compute_de(cfg, adata, splits):
    """Store a longer ranked list (n_de_store) so target genes can be removed and still leave n_de genes."""
    import scanpy as sc

    n, n_store = cfg["eval"]["n_de"], cfg["eval"].get("n_de_store", 50)
    out = {"n_de": n, "n_stored": n_store, "method": "wilcoxon", "rankby_abs": True}
    for key, split, conds in (("singles", "train", splits["train_singles"]), ("doubles", "test", splits["test_doubles"])):
        m = split_mask(adata, split) & adata.obs["condition"].astype(str).isin(conds + [CONTROL]).to_numpy()
        b = adata[m].copy()
        b.obs["condition"] = b.obs["condition"].astype(str).astype("category")
        t0 = time.time()
        sc.tl.rank_genes_groups(b, "condition", groups=conds, reference=CONTROL, method="wilcoxon",
                                rankby_abs=True, n_genes=n_store, tie_correct=False)
        names = b.uns["rank_genes_groups"]["names"]
        out[key] = {c: [str(g) for g in names[c]] for c in conds}
        log.info("DE %s: %d conditions on %s cells (%.0fs)", key, len(conds), split, time.time() - t0)
    write_json(out, path(cfg, "results_dir") / "de_genes.json")
    return out


def load_de(cfg):
    """The stored ranked DE gene lists."""
    return read_json(path(cfg, "results_dir") / "de_genes.json")


# --------------------------------------------------------------------------- metrics
def pearson(a, b):
    """Pearson correlation of two vectors; nan if either is constant."""
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


class EvalRef:
    """Everything fixed for one evaluation split: observed means, control mean, gene lists, additive reference."""

    def __init__(self, cfg, adata, splits, de, split, deltas):
        self.split = split
        n = cfg["eval"]["n_de"]
        vidx = {g: i for i, g in enumerate(adata.var_names)}
        t2v = dict(adata.uns["target_to_var"])
        self.conds = eval_conditions(splits, split)
        self.obs = group_means(adata, split_mask(adata, split), self.conds + [CONTROL])
        self.ctrl = self.obs[CONTROL]
        de_c = de["singles"] if split == "val" else de["doubles"]
        self.de_idx, self.de_nt_idx = {}, {}
        for c in self.conds:
            targets = {t2v.get(g, g) for g in condition_genes(c)}
            self.de_idx[c] = np.array([vidx[g] for g in de_c[c][:n]])
            nt = [g for g in de_c[c] if g not in targets][:n]
            assert len(nt) == n, f"not enough stored DE genes for {c}"
            self.de_nt_idx[c] = np.array([vidx[g] for g in nt])
        self.add_ref, self.int_idx, self.nonadditive = {}, {}, set()
        if split == "test":
            strength = {}
            for c in self.conds:
                self.add_ref[c] = self.ctrl + sum(deltas[g] for g in condition_genes(c))
                i_obs = self.obs[c] - self.add_ref[c]
                self.int_idx[c] = np.argsort(-np.abs(i_obs))[:n]
                j = self.de_idx[c]
                strength[c] = np.linalg.norm(i_obs[j]) / np.linalg.norm((self.obs[c] - self.ctrl)[j])
            k = int(round(NONADDITIVE_FRAC * len(self.conds)))
            self.nonadditive = set(sorted(strength, key=strength.get, reverse=True)[:k])
            self.strength = strength

    def metrics(self, c, pred) -> dict:
        """Every metric for one perturbation, from its predicted mean profile."""
        obs, ctrl, j, jn = self.obs[c], self.ctrl, self.de_idx[c], self.de_nt_idx[c]
        dp, do = pred - ctrl, obs - ctrl
        out = {"mse_all": float(np.mean((pred - obs) ** 2)), "mse_de20": float(np.mean((pred[j] - obs[j]) ** 2)),
               "pearson_delta_all": pearson(dp, do), "pearson_delta_de20": pearson(dp[j], do[j]),
               "mse_de20_nt": float(np.mean((pred[jn] - obs[jn]) ** 2)), "pearson_delta_de20_nt": pearson(dp[jn], do[jn]),
               "pearson_int_all": np.nan, "pearson_int_top20": np.nan}
        if c in self.add_ref:
            ip, io, k = pred - self.add_ref[c], obs - self.add_ref[c], self.int_idx[c]
            out["pearson_int_all"], out["pearson_int_top20"] = pearson(ip, io), pearson(ip[k], io[k])
        return out

    def score(self, preds: dict, covered: dict | None = None) -> pd.DataFrame:
        """Score a {perturbation: predicted profile} dict. One row per perturbation."""
        return pd.DataFrame([{"condition": c, "covered": True if covered is None else covered[c],
                              "nonadditive": c in self.nonadditive, **self.metrics(c, p)} for c, p in preds.items()])


def summarise(per: pd.DataFrame, method, decoder, split, seed) -> list[dict]:
    """Average the per-perturbation scores: over the split, over the perturbations the embedding covers, and, for
    test, over the most non-additive doubles. One row per subset and metric."""
    subsets = [(split, per), (f"{split}_covered", per[per.covered])]
    if split == "test":
        subsets.append(("test_nonadditive", per[per.nonadditive]))
    out = []
    for sub, df in subsets:
        for m in METRICS:
            vals = df[m].to_numpy(dtype=float) if len(df) else np.array([])
            v = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            out.append({"method": method, "decoder": decoder, "split": sub, "seed": seed, "metric": m,
                        "value": v, "n_conditions": len(df)})
    return out


# --------------------------------------------------------------------------- noise ceiling
def spearman_brown_ceiling(r: float) -> float:
    """From the correlation r between the mean profiles of two random halves of a group of cells, the best
    correlation any predictor can reach against the mean of the whole group: sqrt(2r / (1 + r)).
    2r / (1 + r) is the Spearman-Brown reliability of the full mean; its square root is the expected correlation
    between the true profile and the measured one."""
    return float(np.sqrt(max(0.0, 2 * r / (1 + r)))) if np.isfinite(r) and r > -1 else np.nan


def noise_ceiling(adata, ref: EvalRef, seed: int) -> pd.DataFrame:
    """Best correlation any predictor could reach against the observed (noisy) test means.

    Split each double's cells in half, correlate the two halves' profiles, apply Spearman-Brown to get the
    reliability of the full mean, and take its square root (expected correlation of the true profile with the
    observed one). Gene lists are the same as in evaluation.
    """
    rng = np.random.default_rng(seed)
    cond = adata.obs["condition"].astype(str).to_numpy()
    te = split_mask(adata, "test")
    rows = []
    for c in ref.conds:
        idx = np.flatnonzero(te & (cond == c))
        idx = idx[rng.permutation(len(idx))]
        h1, h2 = (np.asarray(adata.X[np.sort(h)].mean(0)).ravel() for h in np.array_split(idx, 2))
        j, jn, k = ref.de_idx[c], ref.de_nt_idx[c], ref.int_idx[c]
        d1, d2, i1, i2 = h1 - ref.ctrl, h2 - ref.ctrl, h1 - ref.add_ref[c], h2 - ref.add_ref[c]
        rows.append({"condition": c, "covered": True, "nonadditive": c in ref.nonadditive,
                     "mse_all": np.nan, "mse_de20": np.nan, "mse_de20_nt": np.nan,
                     "pearson_delta_all": spearman_brown_ceiling(pearson(d1, d2)),
                     "pearson_delta_de20": spearman_brown_ceiling(pearson(d1[j], d2[j])),
                     "pearson_delta_de20_nt": spearman_brown_ceiling(pearson(d1[jn], d2[jn])),
                     "pearson_int_all": spearman_brown_ceiling(pearson(i1, i2)),
                     "pearson_int_top20": spearman_brown_ceiling(pearson(i1[k], i2[k]))})
    return pd.DataFrame(rows)
