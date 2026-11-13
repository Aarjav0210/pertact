"""Week 1's saved predictions of the 127 test doubles, scored with the Week 3 state-space scores. PCA-50 only.

A Week 1 prediction is a mean expression profile that a trained decoder produced from an embedding. PCA-50 is a
linear map of expression, so the profile can be projected to a point in the state space and compared with the
double's real centre exactly as f(a) + f(b) was in pertact.doubles. scVI-30 is not a linear map of expression, so it
is left out. Nothing is trained or predicted here: only saved predictions are read, and no action set is scored again.

Not a like-for-like comparison with the Week 2 action vectors. The rows written here are an embedding plus a trained
decoder (or, for additive expression, a sum of measured changes). The action-vector rows in results/week3_doubles.csv
are vectors added directly in the state space and have not been through a trained decoder.

    python -I scripts/week3_week1_in_state_space.py   # results/week3_doubles_week1_embeddings.csv and ..._by_pair.csv
"""
import numpy as np
import pandas as pd

from pertact.config import get_logger, load_config, path
from pertact.data import CONTROL, cell_labels, condition_genes
from pertact.doubles import _rank_summary, centre_scores, rank_true, real_shifts
from pertact.encoders import load_all, states
from pertact.metrics import EvalRef, load_de, train_deltas

log = get_logger("week1_in_state_space")
SPACE = "X_pca_g"


def score_shifts(shifts, conds, real, v_cells, true_cell):
    """Score one method's predicted shifts, {double: vector in PCA-50}: the centre scores of every double, and how
    often the right double is named among the 127, from each test cell and from each double's centre."""
    per = pd.DataFrame([centre_scores(shifts[c], real.loc[c, "shift"]) for c in conds])
    per.insert(0, "condition", conds)
    F = np.stack([shifts[c] for c in conds]).astype(np.float32)
    cell = _rank_summary(rank_true(v_cells, F, true_cell))
    v_centre = -np.stack([real.loc[c, "shift"] for c in conds]).astype(np.float32)
    centre = _rank_summary(rank_true(v_centre, F, np.arange(len(conds))))
    return per, {"cell_top1": cell["top1"], "cell_top5": cell["top5"], "centre_top1": centre["top1"], "centre_top5": centre["top5"]}


def main():
    """Project the additive expression baseline and every saved Week 1 prediction of the test doubles into PCA-50,
    score them like the summed action vectors, and write one table per double and one per run."""
    cfg = load_config()
    adata, splits = load_all(cfg)
    S = states(adata, SPACE)
    cond, gem, cell_split = cell_labels(adata.obs, cfg["data"]["batch_key"])
    in_test = cell_split == "test"
    conds = sorted(splits["test_doubles"])

    pca = np.load(path(cfg, "models_dir") / "pca_g.npz")
    assert (pca["var_names"] == adata.var_names.to_numpy().astype(str)).all()
    proj = lambda X: (np.asarray(X, dtype=np.float64) - pca["mean"]) @ pca["components"].T
    # the projection is the encoder: the mean profile of the test controls must land on the mean of their states
    ctrl = in_test & (cond == CONTROL)
    gap = float(np.abs(proj(np.asarray(adata.X[ctrl].mean(0)).ravel()) - S[ctrl].mean(0)).max())
    assert gap < 1e-4, gap

    real = real_shifts(S, cond, gem, in_test, conds)
    mu = {g: S[ctrl & (gem == g)].astype(np.float64).mean(0) for g in np.unique(gem[ctrl])}
    start = {}  # the control centre each double is measured from: the gem-group means, weighted by the double's cells
    for c in conds:
        g_c = gem[in_test & (cond == c)]
        start[c] = sum((g_c == g).mean() * mu[g] for g in np.unique(g_c))
    cells = np.flatnonzero(in_test & np.isin(cond, conds))
    v_cells = (np.stack([mu[g] for g in gem[cells]]) - S[cells]).astype(np.float32)
    where = {c: i for i, c in enumerate(conds)}
    true_cell = np.array([where[c] for c in cond[cells]])

    ref = EvalRef(cfg, adata, splits, load_de(cfg), "test", train_deltas(adata, splits))
    runs = [("additive_expr", "baseline", 0, {c: proj(ref.add_ref[c]) - start[c] for c in conds})]
    for method in cfg["eval"]["methods"]:
        for decoder in cfg["eval"]["decoders"]:
            for seed in cfg["eval"]["seeds"]:
                z = np.load(path(cfg, "results_dir") / "predictions" / f"{method}__{decoder}__s{seed}.npz")
                P = dict(zip(z["test_conds"].tolist(), proj(z["test_pred"])))
                runs.append((method, decoder, seed, {c: P[c] - start[c] for c in conds}))
    pers, names = [], []
    for method, decoder, seed, shifts in runs:
        per, name = score_shifts(shifts, conds, real, v_cells, true_cell)
        pers.append(per.assign(method=method, decoder=decoder, seed=seed))
        names.append({"method": method, "decoder": decoder, "seed": seed} | name)
    per = pd.concat(pers, ignore_index=True)

    # paired comparisons with the summed mean shifts (Week 3) and with the additive expression baseline
    w3 = pd.read_csv(path(cfg, "results_dir") / "week3_doubles_by_pair.csv")
    ms = w3[(w3.space == SPACE) & (w3.rule == "sum") & (w3.actions == "mean_shift")].set_index("condition")["sq_miss"]
    add = per[per.method == "additive_expr"].set_index("condition")["sq_miss"]
    string_genes = set(np.load(path(cfg, "embeddings_dir") / "string_n2v.npz")["genes"].tolist())
    covered = [c for c in conds if all(g in string_genes for g in condition_genes(c))]
    rows = []
    for (method, decoder, seed), g in per.groupby(["method", "decoder", "seed"], sort=False):
        g = g.set_index("condition")
        for subset, idx in (("all", conds), ("covered_by_string", covered)):
            h = g.loc[idx]
            rows.append({"method": method, "decoder": decoder, "seed": seed, "subset": subset, "n_conditions": len(h),
                         "median_miss_ratio": h.miss_ratio.median(), "share_miss_ratio_below_1": (h.miss_ratio < 1).mean(),
                         "median_cos": h.cos.median(), "median_len_ratio": h.len_ratio.median(),
                         "share_beats_mean_shift": (h.sq_miss < ms.loc[idx]).mean(),
                         "share_beats_additive_expr": (h.sq_miss < add.loc[idx]).mean()})
    summary = pd.DataFrame(rows).merge(pd.DataFrame(names), on=["method", "decoder", "seed"])
    summary.loc[summary.subset != "all", ["cell_top1", "cell_top5", "centre_top1", "centre_top5"]] = np.nan  # named among all 127 only
    out = path(cfg, "results_dir")
    front = ["method", "decoder", "seed", "condition"]
    per[front + [c for c in per if c not in front]].to_csv(out / "week3_doubles_week1_embeddings_by_pair.csv", index=False)
    summary.to_csv(out / "week3_doubles_week1_embeddings.csv", index=False)
    show = summary[summary.subset == "all"].groupby(["method", "decoder"], sort=False).mean(numeric_only=True)
    log.info("PCA-50, 127 test doubles, mean over seeds (projection check %.1e):\n%s", gap,
             show[["median_miss_ratio", "median_cos", "median_len_ratio", "cell_top1", "centre_top1"]].round(3).to_string())


if __name__ == "__main__":
    main()
