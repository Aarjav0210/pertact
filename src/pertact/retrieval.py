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

from .config import get_logger, load_config, path
from .data import CONTROL
from .ot import load_pairs

log = get_logger("retrieval")


def _states(adata, space):
    return np.asarray(adata.obsm[space], dtype=np.float32)


def mean_shift_actions(cfg, adata, space) -> dict[str, np.ndarray]:
    """f(a) = mean over training OT pairs of s(x_pert) - Σ_k w_k s(x_ctrl_k)."""
    S = _states(adata, space)
    z = load_pairs(cfg, space, "train")
    start = (S[z["partners"]] * z["weights"][..., None]).sum(1)
    disp = S[z["cell"]] - start
    df = pd.DataFrame(disp)
    df["cond"] = z["cond"]
    m = df.groupby("cond").mean()
    return {g: m.loc[g].to_numpy(dtype=np.float32) for g in m.index}


def calibrate_scale(cfg, adata, space, actions) -> float:
    """One scalar alpha for a whole action set, by least squares on training OT pairs:
    min_alpha sum_j || start_j + alpha f(a_j) - s(x_j) ||².  InfoNCE only constrains relative distances, so its
    actions can be right in direction but wrong in length; this fixes the length without touching directions."""
    S = _states(adata, space)
    z = load_pairs(cfg, space, "train")
    start = (S[z["partners"]] * z["weights"][..., None]).sum(1)
    F = np.stack([actions[c] for c in z["cond"]])
    return float((F * (S[z["cell"]] - start)).sum() / (F * F).sum())


def per_action_scale(cfg, adata, space, actions) -> dict[str, float]:
    """One scalar c_a per action, by least squares on that perturbation's training OT pairs:
    min_c sum_{j in a} || start_j + c f(a) - s(x_j) ||²  =>  c_a = f(a)·mean_j(s(x_j) - start_j) / ||f(a)||².
    The solution is the projection of the mean OT shift onto f(a): direction kept, length = ||mean shift|| * cos."""
    S = _states(adata, space)
    z = load_pairs(cfg, space, "train")
    start = (S[z["partners"]] * z["weights"][..., None]).sum(1)
    disp = pd.DataFrame(S[z["cell"]] - start)
    disp["cond"] = z["cond"]
    tot = disp.groupby("cond").sum()
    n = disp.groupby("cond").size()
    return {g: float(actions[g] @ tot.loc[g].to_numpy() / (n[g] * actions[g] @ actions[g])) for g in tot.index if g in actions}


def length_table(sets, ms, tier, space) -> pd.DataFrame:
    """Per perturbation: ||f(a)|| / ||mean OT shift|| and cos(f(a), mean shift) for every action set."""
    rows = []
    for (name, seed), acts in sets.items():
        for g, m in ms.items():
            if g not in acts:
                continue
            f = acts[g]
            nf, nm = float(np.linalg.norm(f)), float(np.linalg.norm(m))
            rows.append({"space": space, "actions": name, "seed": seed, "gene": g, "tier": tier[g],
                         "len_ratio": nf / nm, "cos_mean_shift": float(f @ m / (nf * nm)) if nf > 0 else np.nan})
    return pd.DataFrame(rows)


def _avg_rank(d_true, d_all):
    """Rank of the true item (1 = best) with ties averaged, row by row."""
    less = (d_all < d_true[:, None]).sum(1)
    eq = (d_all == d_true[:, None]).sum(1)
    return less + (eq + 1) / 2.0


def instance_retrieval(cfg, adata, space, split, actions, max_queries=None, seed=0, chunk=1024) -> dict:
    S = _states(adata, space)
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
    S = _states(adata, space)
    z = load_pairs(cfg, space, split, pooled=True)
    x = S[z["cell"]]
    if start == "pooled_ot":
        s0 = (S[z["partners"]] * z["weights"][..., None]).sum(1)
    else:  # mean of the split's controls in the same gem group
        obs = adata.obs
        gem = obs[cfg["data"]["batch_key"]].astype(str).to_numpy()
        ctrl = (obs["split"].astype(str).to_numpy() == split) & (obs["condition"].astype(str).to_numpy() == CONTROL)
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


def evaluate(cfg):
    from .evaluate import load_all
    from .infonce import scale_D0

    adata, splits = load_all(cfg)
    genes = sorted(splits["train_singles"])
    rows, per_rows, strength_rows, scale_rows, len_rows = [], [], [], [], []
    for space in cfg["ot"]["spaces"]:
        ms = mean_shift_actions(cfg, adata, space)
        rng = np.random.default_rng(cfg["seed"])
        perm = rng.permutation(len(genes))
        sets = {("mean_shift", 0): ms, ("zero", 0): {},
                ("shuffled_mean_shift", 0): {g: ms[genes[perm[i]]] for i, g in enumerate(genes)}}
        for seed in cfg["eval"]["seeds"]:
            p = path(cfg, "embeddings_dir") / f"infonce__{space}__s{seed}.npz"
            if p.exists():
                zz = np.load(p)
                acts = dict(zip(zz["genes"].tolist(), zz["emb"]))
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
            pa = path(cfg, "embeddings_dir") / f"infonce_anchor__{space}__s{seed}.npz"
            if pa.exists():  # InfoNCE trained with the weak length anchor (addendum); scales logged, not applied
                za = np.load(pa)
                acts = dict(zip(za["genes"].tolist(), za["emb"]))
                sets[("infonce_anchor", seed)] = acts
                alpha, c = calibrate_scale(cfg, adata, space, acts), per_action_scale(cfg, adata, space, acts)
                scale_rows += [{"space": space, "actions": "infonce_anchor", "seed": seed, "gene": g, "c": v,
                                "alpha": alpha} for g, v in c.items()]
                log.info("%s seed %d: anchored InfoNCE, least-squares alpha would be %.3f", space, seed, alpha)
        D0 = scale_D0(_states(adata, space)[(adata.obs["split"] == "train").to_numpy()])
        strength = {g: float(np.linalg.norm(v) / np.sqrt(D0)) for g, v in ms.items()}
        cuts = np.quantile(list(strength.values()), [1 / 3, 2 / 3])
        tier = {g: ("weak" if s <= cuts[0] else "medium" if s <= cuts[1] else "strong") for g, s in strength.items()}
        strength_rows += [{"space": space, "gene": g, "strength": s, "tier": tier[g]} for g, s in strength.items()]
        len_rows.append(length_table({k: v for k, v in sets.items() if k[0] not in ("zero", "shuffled_mean_shift")},
                                     ms, tier, space))
        for (name, seed), acts in sets.items():
            for split in ("val", "train"):
                base = {"space": space, "actions": name, "seed": seed, "split": split}
                r = instance_retrieval(cfg, adata, space, split, acts, max_queries=5000 if split == "train" else None)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--spaces", nargs="*", default=None, help="override config ot.spaces")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.spaces:
        cfg["ot"]["spaces"] = a.spaces
    evaluate(cfg)


if __name__ == "__main__":
    main()
