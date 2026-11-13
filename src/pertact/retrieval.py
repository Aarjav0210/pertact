"""Retrieval of held-out cells for seen single perturbations, given perturbation actions s(x) + f(a).

1. Instance retrieval (the InfoNCE task). For each held-out perturbed cell j and its OT partner control c(j)
   (per-perturbation OT among validation cells), the query s(c(j)) + f(a_j) is ranked against every held-out
   perturbed cell. Metrics: recall@1/10/100 and MRR of cell j itself; same_pert@10, the share of the 10 nearest
   cells that belong to the right perturbation.
2. Perturbation identification (label-free start). The start state comes from pooled OT (controls matched to all
   held-out cells without their labels) or from the gem group's mean control. Every candidate (the 105 genes plus
   "no perturbation", f = 0) is scored by -||start + f(a) - s(x_j)||². Metrics: top-1 and top-5 accuracy and MRR of
   the true perturbation (chance top-1 = 1/106), also split by effect strength.

Action sets: InfoNCE (each seed); InfoNCE trained with the weak length anchor (infonce_anchor, if present); InfoNCE calibrated (all actions times one global scalar fitted by least squares on
training pairs, fixing length but not direction); InfoNCE scaled per action (one scalar per gene, same least squares,
which gives c_a f(a) = the projection of the mean shift onto f(a), so its length is ||mean shift|| * cos); mean shift (average OT displacement on training pairs, closed
form); zero (f = 0); shuffled mean shift (actions permuted across genes); and two length-vs-direction controls
that swap lengths gene by gene (InfoNCE direction at mean-shift length, mean-shift direction at InfoNCE length).

    python -I -m pertact.retrieval     # results/week2_retrieval.csv, week2_identification_by_pert.csv,
                                       # week2_length_ratio.csv, week2_action_scales.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .actions import (actions_path, calibrate_scale, effect_tiers, length_table, load_actions, mean_shift_actions,
                      per_action_scale)
from .config import get_logger, load_config, path
from .data import CONTROL, cell_labels
from .encoders import load_all, states
from .ot import load_pairs, start_states

log = get_logger("retrieval")
TRAIN_QUERY_CAP = 5000  # the forward test on training cells uses this many random queries; validation uses them all


# --------------------------------------------------------------------------- the two tests
def _avg_rank(d_true, d_all):
    """Rank of the true item (1 = best) with ties averaged, row by row."""
    less = (d_all < d_true[:, None]).sum(1)
    eq = (d_all == d_true[:, None]).sum(1)
    return less + (eq + 1) / 2.0


def instance_retrieval(cfg, adata, space, split, actions, max_queries=None, seed=0, chunk=1024) -> dict:
    """The forward test. For each paired cell the query is its heaviest partner control plus the action vector, and
    every cell of the split is ranked by distance to it. Returns the rank statistics of the right cell."""
    S = states(adata, space)
    z = load_pairs(cfg, space, split)
    keys = S[z["cell"]]
    conds = z["cond"]
    n = len(keys)
    q_idx = np.arange(n)
    if max_queries and n > max_queries:
        q_idx = np.sort(np.random.default_rng(seed).choice(n, max_queries, replace=False))
    D = S.shape[1]
    F = np.stack([actions.get(c, np.zeros(D, np.float32)) for c in conds])
    Q = S[z["partners"][:, 0]] + F
    kn = (keys * keys).sum(1)
    ranks, same10 = [], []
    for i in range(0, len(q_idx), chunk):
        qi = q_idx[i:i + chunk]
        q = Q[qi]
        d2 = (q * q).sum(1)[:, None] + kn[None, :] - 2 * q @ keys.T
        ranks.append(_avg_rank(d2[np.arange(len(qi)), qi], d2))
        top10 = np.argpartition(d2, 10, axis=1)[:, :10]
        same10.append((conds[top10] == conds[qi][:, None]).mean(1))
    r = np.concatenate(ranks)
    return {"n_queries": int(len(q_idx)), "n_candidates": int(n), "mrr": float((1 / r).mean()),
            "recall@1": float((r <= 1).mean()), "recall@10": float((r <= 10).mean()),
            "recall@100": float((r <= 100).mean()), "same_pert@10": float(np.concatenate(same10).mean())}


def identification(cfg, adata, space, split, actions, genes, start="pooled_ot"):
    """The reverse test. From a start state chosen without the cell's label (its pooled OT partners, or the mean
    control of its gem group), every action vector and "no perturbation" is tried, and the true perturbation is
    ranked by how close each lands to the cell. Returns the summary and the rank of every cell."""
    S = states(adata, space)
    z = load_pairs(cfg, space, split, pooled=True)
    x = S[z["cell"]]
    if start == "pooled_ot":
        s0 = start_states(S, z)
    else:  # mean of the split's controls in the same gem group
        cond, gem, cell_split = cell_labels(adata.obs, cfg["data"]["batch_key"])
        ctrl = (cell_split == split) & (cond == CONTROL)
        mu = {g: S[ctrl & (gem == g)].mean(0) for g in np.unique(gem)}
        s0 = np.stack([mu[g] for g in gem[z["cell"]]])
    cand = genes + [CONTROL]
    Fm = np.stack([actions.get(g, np.zeros(S.shape[1], np.float32)) for g in genes] + [np.zeros(S.shape[1], np.float32)])
    v = s0 - x
    d2 = (v * v).sum(1)[:, None] + (Fm * Fm).sum(1)[None, :] + 2 * v @ Fm.T
    ci = {g: i for i, g in enumerate(cand)}
    t = np.array([ci[c] for c in z["cond"]])
    r = _avg_rank(d2[np.arange(len(t)), t], d2)
    per = pd.DataFrame({"cond": z["cond"], "rank": r})
    return {"n_cells": int(len(r)), "n_candidates": len(cand), "top1": float((r <= 1).mean()),
            "top5": float((r <= 5).mean()), "mrr": float((1 / r).mean())}, per


# --------------------------------------------------------------------------- every set of vectors, both tests
def build_action_sets(cfg, adata, space, genes, ms):
    """Every set of action vectors that is tested in one state space, keyed by (name, seed), plus the rescaling
    numbers fitted on training pairs. The references (mean shift, no action, shuffled) come first, then, for each
    seed with saved InfoNCE vectors: as trained, rescaled globally and per gene, the two length swaps, and anchored."""
    scale_rows = []
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(len(genes))
    sets = {("mean_shift", 0): ms, ("zero", 0): {},
            ("shuffled_mean_shift", 0): {g: ms[genes[perm[i]]] for i, g in enumerate(genes)}}
    for seed in cfg["eval"]["seeds"]:
        if actions_path(cfg, space, seed).exists():
            acts = load_actions(cfg, space, seed)
            sets[("infonce", seed)] = acts
            alpha = calibrate_scale(cfg, adata, space, acts)
            sets[("infonce_calibrated", seed)] = {g: alpha * v for g, v in acts.items()}
            c = per_action_scale(cfg, adata, space, acts)
            sets[("infonce_scaled_per_action", seed)] = {g: c[g] * v for g, v in acts.items()}
            scale_rows += [{"space": space, "actions": "infonce", "seed": seed, "gene": g, "c": v, "alpha": alpha}
                           for g, v in c.items()]
            log.info("%s seed %d: per-action scales c_a median %.3f (range %.3f to %.3f)", space, seed,
                     np.median(list(c.values())), min(c.values()), max(c.values()))
            # length vs direction controls: swap the lengths of the two action sets gene by gene
            nrm = lambda v: v / np.linalg.norm(v)
            sets[("infonce_dir_meanshift_len", seed)] = {g: nrm(acts[g]) * np.linalg.norm(ms[g]) for g in genes}
            sets[("meanshift_dir_infonce_len", seed)] = {g: nrm(ms[g]) * np.linalg.norm(acts[g]) for g in genes}
            log.info("%s seed %d: InfoNCE length calibration alpha = %.3f (fitted on training pairs)", space, seed, alpha)
        else:
            log.warning("%s seed %d: no saved InfoNCE vectors (%s); this seed is left out", space, seed,
                        actions_path(cfg, space, seed).name)
        if actions_path(cfg, space, seed, "infonce_anchor").exists():  # trained with the length anchor; scales logged, not applied
            acts = load_actions(cfg, space, seed, "infonce_anchor")
            sets[("infonce_anchor", seed)] = acts
            alpha, c = calibrate_scale(cfg, adata, space, acts), per_action_scale(cfg, adata, space, acts)
            scale_rows += [{"space": space, "actions": "infonce_anchor", "seed": seed, "gene": g, "c": v,
                            "alpha": alpha} for g, v in c.items()]
            log.info("%s seed %d: anchored InfoNCE, least-squares alpha would be %.3f", space, seed, alpha)
    return sets, scale_rows


def run_tests(cfg):
    """Run both tests on every set of action vectors, in every state space, on validation cells and on training
    cells, and write the five Week 2 result tables."""
    adata, splits = load_all(cfg)
    genes = sorted(splits["train_singles"])
    rows, per_rows, strength_rows, scale_rows, len_rows = [], [], [], [], []
    for space in cfg["ot"]["spaces"]:
        ms = mean_shift_actions(cfg, adata, space)
        sets, scales = build_action_sets(cfg, adata, space, genes, ms)
        scale_rows += scales
        strength, tier = effect_tiers(adata, space, ms)
        strength_rows += [{"space": space, "gene": g, "strength": s, "tier": tier[g]} for g, s in strength.items()]
        len_rows.append(length_table({k: v for k, v in sets.items() if k[0] not in ("zero", "shuffled_mean_shift")},
                                     ms, tier, space))
        for (name, seed), acts in sets.items():
            for split in ("val", "train"):
                base = {"space": space, "actions": name, "seed": seed, "split": split}
                r = instance_retrieval(cfg, adata, space, split, acts, max_queries=TRAIN_QUERY_CAP if split == "train" else None)
                rows += [base | {"task": "instance", "metric": k, "value": v} for k, v in r.items()]
                for start in ("pooled_ot", "ctrl_mean"):
                    m, per = identification(cfg, adata, space, split, acts, genes, start)
                    rows += [base | {"task": f"identify_{start}", "metric": k, "value": v} for k, v in m.items()]
                    per["tier"] = per["cond"].map(tier)
                    for t, g in per.groupby("tier"):
                        rows.append(base | {"task": f"identify_{start}", "metric": f"top1_{t}", "value": float((g["rank"] <= 1).mean())})
                    if split == "val":
                        pp = per.groupby("cond")["rank"].agg(top1=lambda r: (r <= 1).mean(), mrr=lambda r: (1 / r).mean(),
                                                             n="size").reset_index()
                        per_rows.append(pp.assign(space=space, actions=name, seed=seed, start=start,
                                                  strength=pp["cond"].map(strength), tier=pp["cond"].map(tier)))
            v = {x["metric"]: x["value"] for x in rows if x["space"] == space and x["actions"] == name
                 and x["seed"] == seed and x["split"] == "val" and x["task"] == "instance"}
            log.info("%-9s %-20s s%d | val instance MRR %.3f R@10 %.3f same@10 %.3f", space, name, seed,
                     v.get("mrr", np.nan), v.get("recall@10", np.nan), v.get("same_pert@10", np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(path(cfg, "results_dir") / "week2_retrieval.csv", index=False)
    pd.concat(per_rows).to_csv(path(cfg, "results_dir") / "week2_identification_by_pert.csv", index=False)
    pd.DataFrame(strength_rows).to_csv(path(cfg, "results_dir") / "week2_effect_strength.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(path(cfg, "results_dir") / "week2_action_scales.csv", index=False)
    pd.concat(len_rows).to_csv(path(cfg, "results_dir") / "week2_length_ratio.csv", index=False)
    return df


def main():
    """Run both tests on every saved set of action vectors."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--spaces", nargs="*", default=None, help="override config ot.spaces")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.spaces:
        cfg["ot"]["spaces"] = a.spaces
    run_tests(cfg)


if __name__ == "__main__":
    main()
