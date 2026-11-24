"""Reference for naming the perturbation: plain classifiers that predict the gene from a cell's displacement.

Naming a seen gene from one cell is a classification with 105 classes. The action-vector methods rank the genes by
how well start + f(a) explains the cell; a classifier can be trained to do that ranking directly. This script fits
standard classifiers on the training cells' displacements (cell minus its label-free start, as in the reverse test)
and scores them on validation cells with the same starts, cells and rank statistics:

    centroid       nearest class centre of the training displacements (a mean shift for this start)
    lda            linear discriminant analysis: nearest centre after whitening the spread within genes
    logistic       multinomial logistic regression on the standardised displacement
    logistic_unit  the same on the unit-length displacement, so only its direction is used (as the cosine loss does)

Nothing is tuned: scikit-learn defaults throughout. No test cell is read.

    python -I scripts/naming_classifier_reference.py    # results/naming_classifiers.csv
"""
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from pertact.actions import mean_shift_actions
from pertact.config import get_logger, load_config, path
from pertact.data import CONTROL, cell_labels
from pertact.encoders import load_all, states
from pertact.ot import load_pairs, start_states
from pertact.retrieval import _avg_rank, identification

log = get_logger("naming_classifiers")
STARTS = ("pooled_ot", "ctrl_mean")


def displacements(cfg, adata, space, split, start):
    """Cell minus start for every single-perturbed cell of a split, with its gene. The start is chosen without the
    cell's label, exactly as in the reverse test: its pooled OT partners, or the mean control of its gem group."""
    S = states(adata, space)
    z = load_pairs(cfg, space, split, pooled=True)
    if start == "pooled_ot":
        s0 = start_states(S, z)
    else:
        cond, gem, cell_split = cell_labels(adata.obs, cfg["data"]["batch_key"])
        ctrl = (cell_split == split) & (cond == CONTROL)
        mu = {g: S[ctrl & (gem == g)].mean(0) for g in np.unique(gem)}
        s0 = np.stack([mu[g] for g in gem[z["cell"]]])
    return (S[z["cell"]] - s0).astype(np.float64), z["cond"]


def rank_summary(scores, classes, labels):
    """Top-1, top-5 and mean reciprocal rank of the true gene, from one score per cell and class (higher is better).
    `top1_per_gene` averages over genes instead of cells, so a gene with many cells counts no more than a rare one."""
    col = {g: i for i, g in enumerate(classes)}
    t = np.array([col[g] for g in labels])
    r = _avg_rank(-scores[np.arange(len(t)), t], -scores)
    per_gene = float(pd.Series(r <= 1).groupby(np.asarray(labels)).mean().mean())
    return {"top1": float((r <= 1).mean()), "top1_per_gene": per_gene, "top5": float((r <= 5).mean()),
            "mrr": float((1 / r).mean()), "n_cells": int(len(r)), "n_candidates": int(scores.shape[1])}


def main():
    """Fit every classifier on training displacements and score it on validation displacements, in both state spaces
    and for both starts. The reverse test's own mean-shift row is written next to them for reference."""
    cfg = load_config()
    adata, splits = load_all(cfg)
    genes = sorted(splits["train_singles"])
    rows = []
    for space in cfg["ot"]["spaces"]:
        ms = mean_shift_actions(cfg, adata, space)
        for start in STARTS:
            Xtr, ytr = displacements(cfg, adata, space, "train", start)
            Xva, yva = displacements(cfg, adata, space, "val", start)
            base = {"space": space, "start": start}
            ref, _ = identification(cfg, adata, space, "val", ms, genes, start)
            rows.append(base | {"model": "mean_shift (reverse test, 106 candidates)", **{k: ref[k] for k in ("top1", "top5", "mrr", "n_cells", "n_candidates")}})
            # (that row has no per-gene value: the reverse test's own summary is per cell)
            # check of the displacements built here: the reverse test recomputed from them must give the same answer
            F = np.stack([ms[g] for g in genes] + [np.zeros(Xva.shape[1])])
            d2 = (Xva * Xva).sum(1)[:, None] + (F * F).sum(1)[None, :] - 2 * Xva @ F.T
            again = rank_summary(-d2, genes + [CONTROL], yva)
            assert abs(again["top1"] - ref["top1"]) < 1e-9, (again["top1"], ref["top1"])

            centres = np.stack([Xtr[ytr == g].mean(0) for g in genes])
            dist = (Xva * Xva).sum(1)[:, None] + (centres * centres).sum(1)[None, :] - 2 * Xva @ centres.T
            rows.append(base | {"model": "centroid"} | rank_summary(-dist, genes, yva))
            models = {"lda": (LinearDiscriminantAnalysis(), Xtr, Xva),
                      "logistic": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), Xtr, Xva),
                      "logistic_unit": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), normalize(Xtr), normalize(Xva))}
            for name, (model, A, B) in models.items():
                model.fit(A, ytr)
                rows.append(base | {"model": name} | rank_summary(model.decision_function(B), list(model.classes_), yva))
            log.info("%s, %s start:\n%s", space, start, pd.DataFrame(rows)[lambda d: (d.space == space) & (d.start == start)]
                     [["model", "top1", "top1_per_gene", "top5", "mrr"]].round(3).to_string(index=False))
    out = path(cfg, "results_dir") / "naming_classifiers.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
