"""Naming the perturbation: one table for every way of building the namer (Week 5).

The test is reverse identification on validation cells of single perturbations. A displacement is a cell minus a start
chosen without the cell's label (the mean control of its gem group, or its partners in the label-free pairing). All 105
genes are ranked for it, and the score is how often the true gene comes first. The same test is run on one cell and on
the average displacement of a set of cells of one perturbation.

Rows of the table:
    mean shift            the training mean shift of each gene, compared by cosine
    mean shift, whitened  the same after the computed projection (actions.whitening)
    centroid, lda, logistic   plain classifiers fitted on the training cells' displacements (scikit-learn defaults)
    stored vectors        every run of the naming table (actions.table_variants): cosine between the gene's vector
                          and the displacement, after the run's own projection if it has one

Nothing is tuned here and no test cell is read.

    python -I -m pertact.naming                      # results/naming_table/naming.csv and a summary in the log
    python -I -m pertact.naming --variant NAME ...   # only these stored sets (the references are always included)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .actions import (actions_path, displacements, load_actions, load_projection, mean_shift_actions, table_variants,
                      whitening)
from .config import get_logger, load_config, path
from .encoders import load_all
from .retrieval import _avg_rank, _unit, identification

log = get_logger("naming")
STARTS = ("ctrl_mean", "pooled_ot")


def rank_summary(scores, classes, labels) -> dict:
    """Top-1, top-5 and mean reciprocal rank of the true gene, from one score per displacement and class (higher is
    better). `top1_per_gene` averages over genes instead of displacements, so a gene with many cells counts no more
    than a rare one."""
    col = {g: i for i, g in enumerate(classes)}
    t = np.array([col[g] for g in labels])
    r = _avg_rank(-scores[np.arange(len(t)), t], -scores)
    per_gene = float(pd.Series(r <= 1).groupby(np.asarray(labels)).mean().mean())
    return {"top1": float((r <= 1).mean()), "top1_per_gene": per_gene, "top5": float((r <= 5).mean()),
            "mrr": float((1 / r).mean()), "n": int(len(r)), "n_candidates": int(scores.shape[1])}


def set_means(X, y, set_size, n_sets, rng):
    """For every gene, `n_sets` sets of `set_size` of its displacements, each averaged into one. The cells of a set
    are distinct whenever the gene has that many. Returns the averaged displacements and their genes."""
    out, lab = [], []
    for g in np.unique(y):
        ix = np.flatnonzero(y == g)
        for _ in range(n_sets):
            out.append(X[rng.choice(ix, size=set_size, replace=len(ix) < set_size)].mean(0))
            lab.append(g)
    return np.stack(out), np.array(lab)


def name_scores(X, F, proj=None) -> np.ndarray:
    """Cosine between every displacement (a row of X) and every gene's vector (a row of F). With a projection the
    displacement is projected first; the vectors are already in the projected space."""
    return _unit(X if proj is None else X @ np.asarray(proj, dtype=np.float64).T) @ _unit(np.asarray(F, dtype=np.float64)).T


def _classifier_scores(model, genes):
    """A fitted classifier as a function from displacements to one score per gene, in the order of `genes`."""
    order = [list(model.classes_).index(g) for g in genes]
    return lambda X: model.decision_function(X)[:, order]


def _centroid_scores(C):
    """Minus the squared distance from every displacement to every class centre (a row of C)."""
    return lambda X: -((X * X).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2 * X @ C.T)


def run(cfg, variants=None):
    """Score every row of the naming table in both state spaces, for both label-free starts, from one validation
    cell and from sets of validation cells. Writes results/naming_table/naming.csv."""
    adata, splits = load_all(cfg)
    genes = sorted(splits["train_singles"])
    variants = list(variants) if variants else table_variants(cfg)
    n_set, n_sets = cfg["infonce"]["set_size"], cfg["infonce"]["naming_sets_per_gene"]
    rows = []
    for space in cfg["ot"]["spaces"]:
        ms = mean_shift_actions(cfg, adata, space)
        Fms = np.stack([ms[g] for g in genes])
        stored = {}
        for name in variants:
            for seed in cfg["eval"]["seeds"]:
                if not actions_path(cfg, space, seed, name).exists():
                    log.warning("%s seed %d: no stored vectors named %s; left out", space, seed, name)
                    continue
                acts = load_actions(cfg, space, seed, name)
                stored[(name, seed)] = (np.stack([acts[g] for g in genes]), load_projection(cfg, space, seed, name))
        for start in STARTS:
            Xtr, ytr = displacements(cfg, adata, space, "train", start)
            Xva, yva = displacements(cfg, adata, space, "val", start)
            W = whitening(cfg, adata, space, start)
            small = int((pd.Series(yva).value_counts() < n_set).sum())
            log.info("%s, %s start: %d validation cells of %d genes; %d genes have fewer than %d cells, so their sets repeat cells",
                     space, start, len(yva), len(np.unique(yva)), small, n_set)
            inputs = {"one cell": (Xva, yva),
                      f"set of {n_set}": set_means(Xva, yva, n_set, n_sets, np.random.default_rng(cfg["seed"]))}
            models = {("mean shift", -1): lambda X: name_scores(X, Fms),
                      ("mean shift, whitened", -1): lambda X: name_scores(X, Fms @ W.T, W),
                      ("centroid", -1): _centroid_scores(np.stack([Xtr[ytr == g].mean(0) for g in genes])),
                      ("lda", -1): _classifier_scores(LinearDiscriminantAnalysis().fit(Xtr, ytr), genes),
                      ("logistic", -1): _classifier_scores(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr), genes)}
            for (name, seed), (F, proj) in stored.items():
                models[(name, seed)] = lambda X, F=F, proj=proj: name_scores(X, F, proj)
            for (model, seed), score in models.items():
                for inp, (X, y) in inputs.items():
                    rows.append({"space": space, "start": start, "input": inp, "model": model, "seed": seed} | rank_summary(score(X), genes, y))
            # the displacements built here must give the reverse test's own answer for the mean shift (that test works
            # in single precision, so a near-tie may fall the other way for a cell or two)
            ref, _ = identification(cfg, adata, space, "val", ms, genes, start, score="cosine")
            mine = next(r for r in rows if (r["space"], r["start"], r["input"], r["model"]) == (space, start, "one cell", "mean shift"))
            assert abs(mine["top1"] - ref["top1"]) < 1e-3, (mine["top1"], ref["top1"])
    df = pd.DataFrame(rows)
    out = path(cfg, "results_dir") / "naming_table"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "naming.csv", index=False)
    order = list(dict.fromkeys(df.model))
    for metric in ("top1_per_gene", "top1"):
        t = df.groupby(["space", "model", "start", "input"])[metric].mean().unstack(["start", "input"])
        t = t.reindex([(sp, mo) for sp in cfg["ot"]["spaces"] for mo in order])
        log.info("naming on validation cells, %s, mean over seeds:\n%s", metric, t.round(3).to_string())
    log.info("wrote %s", out / "naming.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="*", default=None, help="stored sets to score (default: every run of the naming table)")
    ap.add_argument("--spaces", nargs="*", default=None, help="override config ot.spaces")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.spaces:
        cfg["ot"]["spaces"] = a.spaces
    run(cfg, a.variant)


if __name__ == "__main__":
    main()
