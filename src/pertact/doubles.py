"""Composition of action vectors on the test doubles, scored in the frozen state space (no decoder, nothing fitted).

A double a+b is predicted as start + f(a) + f(b). The action vectors come from training pairs only: the same sets
as the Week 2 tests (retrieval.build_action_sets). The doubles are scored and never used to choose anything. The
list of action sets, references and scores below was fixed before the first run on the doubles.

1. The centre. The real shift of a double is the centre of its test cells minus the centre of the test controls,
   taken per gem group and averaged with the double's cell counts. The predicted shift f(a) + f(b) is compared with
   it: squared miss, the miss as a fraction of the real shift's squared length (1 = no better than adding nothing),
   cosine (direction) and length ratio (length). Two references use one of the two vectors only: the gene with the
   longer training mean shift, and whichever of the two lands closer (an oracle, it looks at the double).
2. Naming the double. Each test cell starts from the mean test control of its gem group. Every candidate
   combination is added to that start and the true one is ranked by how close it lands to the cell. Candidates are
   the 127 measured doubles, or every combination of the 105 genes (no perturbation, 105 singles, 5,460 pairs).
   The same ranking is also done once per double, from its centre.

    python -I -m pertact.doubles               # results/week3_doubles.csv and week3_doubles_by_pair.csv
    python -I -m pertact.doubles --split val   # the same code on validation singles; prints, writes nothing

The validation run is a check of this code: with one gene per condition, naming the perturbation from the gem-mean
start is the Week 2 reverse test, and the numbers must equal those in results/week2_retrieval.csv.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from .actions import effect_tiers, mean_shift_actions
from .config import get_logger, load_config, path
from .data import CONTROL, cell_labels, condition_genes
from .encoders import load_all, states
from .retrieval import _avg_rank, build_action_sets

log = get_logger("doubles")
RULES = ("sum", "stronger_single", "best_single")  # how the vectors of a condition's genes are combined


# --------------------------------------------------------------------------- what really happened
def real_shifts(S, cond, gem, in_split, conditions) -> pd.DataFrame:
    """The measured shift of every condition in one split: the centre of its cells minus the centre of the split's
    controls, per gem group, averaged with the condition's cell counts. Also the number of cells and `noise_sq`, the
    squared miss expected of a perfect prediction because both centres are estimated from a finite number of cells
    (within-gem spread of the condition's cells / n, plus the controls' share)."""
    ctrl = in_split & (cond == CONTROL)
    mu = {g: S[ctrl & (gem == g)].mean(0) for g in np.unique(gem[ctrl])}
    ctrl_noise = {g: float(S[ctrl & (gem == g)].var(0, ddof=1).sum()) / int((ctrl & (gem == g)).sum()) for g in mu}
    rows = []
    for c in conditions:
        idx = np.flatnonzero(in_split & (cond == c))
        g_c = gem[idx]
        shift, within, noise = np.zeros(S.shape[1]), 0.0, 0.0
        for g in np.unique(g_c):
            x = S[idx[g_c == g]].astype(np.float64)
            w = len(x) / len(idx)
            shift += w * (x.mean(0) - mu[g])
            within += float(((x - x.mean(0)) ** 2).sum())
            noise += w * w * ctrl_noise[g]
        dof = len(idx) - len(np.unique(g_c))
        rows.append({"condition": c, "n_cells": len(idx), "shift": shift,
                     "noise_sq": within / dof / len(idx) + noise if dof > 0 else np.nan})
    return pd.DataFrame(rows).set_index("condition")


def centre_scores(pred, real) -> dict:
    """A predicted shift against the measured one: squared miss, miss as a fraction of the squared real length,
    cosine and length ratio."""
    npred, nreal = float(np.linalg.norm(pred)), float(np.linalg.norm(real))
    miss = float(((pred - real) ** 2).sum())
    return {"sq_miss": miss, "miss_ratio": miss / nreal ** 2, "len_ratio": npred / nreal,
            "cos": float(pred @ real / (npred * nreal)) if npred > 0 else np.nan}


# --------------------------------------------------------------------------- candidates and their ranks
def all_combinations(genes) -> list[tuple]:
    """Every combination of up to two genes, as sorted tuples: none, each gene alone, each pair."""
    return [()] + [(g,) for g in genes] + list(combinations(sorted(genes), 2))


def compose(actions, combos, dim) -> np.ndarray:
    """One row per combination: the sum of its genes' action vectors (zero for no gene or an unknown gene)."""
    zero = np.zeros(dim, np.float32)
    return np.stack([np.sum([actions.get(g, zero) for g in c], axis=0) if c else zero for c in combos]).astype(np.float32)


def rank_true(v, F, true, chunk=2048) -> np.ndarray:
    """For every row of v, the rank of candidate true[i] when candidates are ordered by ||v_i + F_c||² (1 = closest,
    ties averaged). With v = start - cell this is the distance from start + F_c to the cell."""
    fn = (F * F).sum(1)
    out = []
    for i in range(0, len(v), chunk):
        vi, ti = v[i:i + chunk], true[i:i + chunk]
        d2 = (vi * vi).sum(1)[:, None] + fn[None, :] + 2 * vi @ F.T
        out.append(_avg_rank(d2[np.arange(len(ti)), ti], d2))
    return np.concatenate(out)


def _rank_summary(r) -> dict:
    """Top-1, top-5 and mean reciprocal rank of an array of ranks."""
    return {"top1": float((r <= 1).mean()), "top5": float((r <= 5).mean()), "mrr": float((1 / r).mean())}


# --------------------------------------------------------------------------- the two scores for every action set
def score_conditions(S, cond, gem, in_split, conditions, sets, ms_len, candidate_sets):
    """Score every action set on the given conditions of one split. `sets` is {(name, seed): {gene: vector}},
    `ms_len` the length of each gene's training mean shift (it picks the stronger gene of a pair), `candidate_sets`
    is {label: list of gene tuples} and each must contain every scored condition. Returns one table with a row per
    action set, condition and rule, and one with a row per action set and rule."""
    dim = S.shape[1]
    real = real_shifts(S, cond, gem, in_split, conditions)
    keys = {c: tuple(sorted(condition_genes(c))) for c in conditions}
    ctrl = in_split & (cond == CONTROL)
    mu = {g: S[ctrl & (gem == g)].mean(0) for g in np.unique(gem[ctrl])}
    cells = np.flatnonzero(in_split & np.isin(cond, conditions))
    v_cells = np.stack([mu[g] for g in gem[cells]]) - S[cells]           # start - cell, for every scored cell
    v_centre = -np.stack([real.loc[c, "shift"] for c in conditions]).astype(np.float32)
    true = {}  # per candidate set: the index of the right candidate, for every scored cell and for every condition
    for lab, cs in candidate_sets.items():
        where = {k: i for i, k in enumerate(cs)}
        true[lab] = (np.array([where[keys[c]] for c in cond[cells]]), np.array([where[keys[c]] for c in conditions]))
    cell_cond = pd.Series(cond[cells])
    pair_rows, sum_rows = [], []
    for (name, seed), acts in sets.items():
        zero = np.zeros(dim, np.float32)
        base = {"actions": name, "seed": seed}
        by_rule = {r: [] for r in RULES}
        for c in conditions:
            vecs = [acts.get(g, zero).astype(np.float64) for g in keys[c]]
            shift = real.loc[c, "shift"]
            singles = [centre_scores(f, shift) for f in vecs]
            strongest = int(np.argmax([ms_len[g] for g in keys[c]]))
            by_rule["sum"].append(centre_scores(np.sum(vecs, axis=0), shift))
            by_rule["stronger_single"].append(singles[strongest])
            by_rule["best_single"].append(min(singles, key=lambda s: s["sq_miss"]))
        ident = {}
        for lab, cs in candidate_sets.items():
            F = compose(acts, cs, dim)
            r_cell = rank_true(v_cells, F, true[lab][0])
            per = pd.DataFrame({"condition": cell_cond, "rank": r_cell}).groupby("condition")["rank"]
            ident[lab] = {"cell": _rank_summary(r_cell), "centre_rank": rank_true(v_centre, F, true[lab][1]),
                          "cell_top1": per.agg(lambda r: (r <= 1).mean()), "cell_mrr": per.agg(lambda r: (1 / r).mean())}
        for rule in RULES:
            df = pd.DataFrame(by_rule[rule])
            df.insert(0, "condition", list(conditions))
            df["n_cells"] = real.loc[list(conditions), "n_cells"].to_numpy()
            df["real_len"] = [float(np.linalg.norm(real.loc[c, "shift"])) for c in conditions]
            df["noise_sq"] = real.loc[list(conditions), "noise_sq"].to_numpy()
            s = base | {"rule": rule, "n_conditions": len(df), "median_sq_miss": float(df.sq_miss.median()),
                        "mean_sq_miss": float(df.sq_miss.mean()), "median_miss_ratio": float(df.miss_ratio.median()),
                        "share_miss_ratio_below_1": float((df.miss_ratio < 1).mean()),
                        "median_cos": float(df.cos.median()) if df.cos.notna().any() else np.nan,
                        "median_len_ratio": float(df.len_ratio.median())}
            if rule == "sum":
                for lab, d in ident.items():
                    df[f"centre_rank__{lab}"] = d["centre_rank"]
                    df[f"cell_top1__{lab}"] = d["cell_top1"].reindex(list(conditions)).to_numpy()
                    df[f"cell_mrr__{lab}"] = d["cell_mrr"].reindex(list(conditions)).to_numpy()
                    s |= {f"cell_{k}__{lab}": x for k, x in d["cell"].items()}
                    s |= {f"centre_{k}__{lab}": x for k, x in _rank_summary(d["centre_rank"]).items()}
                    s[f"n_candidates__{lab}"] = len(candidate_sets[lab])
            pair_rows.append(df.assign(**base, rule=rule))
            sum_rows.append(s)
    by_pair, summary = pd.concat(pair_rows, ignore_index=True), pd.DataFrame(sum_rows)
    # paired comparisons over conditions, on the squared miss of the summed vectors
    miss = by_pair.pivot_table(index="condition", columns=["actions", "seed", "rule"], values="sq_miss")
    for i, s in summary.iterrows():
        if s["rule"] != "sum":
            continue
        own = miss[(s["actions"], s["seed"], "sum")]
        if ("mean_shift", 0, "sum") in miss:
            summary.loc[i, "share_beats_mean_shift"] = float((own < miss[("mean_shift", 0, "sum")]).mean())
        summary.loc[i, "share_sum_beats_stronger_single"] = float((own < miss[(s["actions"], s["seed"], "stronger_single")]).mean())
        summary.loc[i, "share_sum_beats_best_single"] = float((own < miss[(s["actions"], s["seed"], "best_single")]).mean())
    return by_pair, summary


def run(cfg, split="test"):
    """Score every Week 2 action set in every paired state space: on the test doubles (split "test"), or on the
    validation singles as a check of the code (split "val"). Returns the per-condition and the summary table."""
    adata, splits = load_all(cfg)
    genes = sorted(splits["train_singles"])
    cond, gem, cell_split = cell_labels(adata.obs, cfg["data"]["batch_key"])
    if split == "test":
        conditions = sorted(splits["test_doubles"])
        candidate_sets = {"doubles": [tuple(sorted(condition_genes(c))) for c in conditions],
                          "all_combinations": all_combinations(genes)}
    else:
        conditions = genes
        candidate_sets = {"singles": [()] + [(g,) for g in genes]}
    pairs, sums = [], []
    for space in cfg["ot"]["spaces"]:
        ms = mean_shift_actions(cfg, adata, space)
        sets, _ = build_action_sets(cfg, adata, space, genes, ms)
        _, tier = effect_tiers(adata, space, ms)
        by_pair, summary = score_conditions(states(adata, space), cond, gem, cell_split == split, conditions, sets,
                                            {g: float(np.linalg.norm(v)) for g, v in ms.items()}, candidate_sets)
        by_pair["tiers"] = by_pair["condition"].map(lambda c: "+".join(sorted(tier[g] for g in condition_genes(c))))
        pairs.append(by_pair.assign(space=space))
        sums.append(summary.assign(space=space))
        show = summary[summary.rule == "sum"].groupby("actions", sort=False).mean(numeric_only=True)
        cols = ["median_miss_ratio", "median_cos", "median_len_ratio"] + [c for c in show if c.startswith("cell_top1")]
        log.info("%s, %s, summed vectors, mean over seeds:\n%s", space, split, show[cols].round(3).to_string())
    front = ["space", "actions", "seed", "rule"]
    by_pair, summary = pd.concat(pairs, ignore_index=True), pd.concat(sums, ignore_index=True)
    return (by_pair[front + [c for c in by_pair if c not in front]], summary[front + [c for c in summary if c not in front]])


def main():
    """Score the action sets on the test doubles and write the two tables, or check the code on validation singles."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--spaces", nargs="*", default=None, help="override config ot.spaces")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.spaces:
        cfg["ot"]["spaces"] = a.spaces
    by_pair, summary = run(cfg, a.split)
    if a.split == "test":
        out = path(cfg, "results_dir")
        summary.to_csv(out / "week3_doubles.csv", index=False)
        by_pair.to_csv(out / "week3_doubles_by_pair.csv", index=False)
        log.info("wrote %s and %s", out / "week3_doubles.csv", out / "week3_doubles_by_pair.csv")
    else:
        saved = pd.read_csv(path(cfg, "results_dir") / "week2_retrieval.csv")
        saved = saved[(saved.split == "val") & (saved.task == "identify_ctrl_mean") & saved.metric.isin(["top1", "top5", "mrr"])]
        new = summary[summary.rule == "sum"].melt(id_vars=["space", "actions", "seed"], var_name="metric", value_name="new",
                                                  value_vars=[f"cell_{m}__singles" for m in ("top1", "top5", "mrr")])
        new["metric"] = new["metric"].str.replace("cell_", "").str.replace("__singles", "")
        both = saved.merge(new, on=["space", "actions", "seed", "metric"], how="outer", indicator=True)
        log.info("check against the Week 2 reverse test (gem-mean start): %d values compared, %d without a partner, "
                 "largest difference %.2e", int((both["_merge"] == "both").sum()), int((both["_merge"] != "both").sum()),
                 float((both["value"] - both["new"]).abs().max()))


if __name__ == "__main__":
    main()
