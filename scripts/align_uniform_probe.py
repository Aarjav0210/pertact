"""Alignment and uniformity (Wang & Isola 2020) of every action set, on validation OT pairs.

For each validation pair j (perturbed cell x_j, heaviest OT partner control c_j, perturbation a_j):
    q_j = s(c_j) + f(a_j)                  query, as in the retrieval test
    k_m = s(x_m)                           keys: every perturbed validation cell
    alignment  = mean_j ||q_j - k_j||² / D0
    uniformity = mean_j log mean_m exp(-||q_j - k_m||² / (t·D0))      over all keys (t = the space's chosen tau)
                 split into same-perturbation keys (m != j) and other-perturbation keys
alignment / t + uniformity + log N is the InfoNCE row loss with all N validation cells as candidates, so the two terms
are exactly what the trained loss trades. Also logged per set: uniformity of the unit action directions
(log mean_{a != b} exp(-2||u_a - u_b||²)), median length ratio to the mean shift, retrieval MRR, same_pert@10 and
identification top-1 (pooled-OT start).

    python -I scripts/align_uniform_probe.py            # saved action sets  -> results/align_uniform/sets.csv
    python -I scripts/align_uniform_probe.py --grids    # + seed 0 retrained over the tau and rho grids -> grids.csv

Reads the stored state embeddings, not the processed h5ad, so it runs in a few GB.
"""
from __future__ import annotations

import argparse
import os
import types

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

from pertact.actions import mean_shift_actions
from pertact.config import get_logger, load_config, path, read_json
from pertact.encoders import scale_D0
from pertact.infonce import train_actions
from pertact.ot import load_pairs
from pertact.retrieval import build_action_sets, identification, instance_retrieval

torch.set_num_threads(2)
log = get_logger("align_uniform")


def light_adata(cfg):
    """The two fields every function used here reads (obsm states, obs split), without the expression matrix."""
    z = np.load(path(cfg, "processed_h5ad").with_name("state_embeddings.npz"), allow_pickle=False)
    names = z["obs_names"].astype(str)
    splits = read_json(path(cfg, "splits"))
    split = pd.Series("unused", index=names, dtype=object)
    for k in ("train", "val", "test"):
        split[splits["cells"][k]] = k
    obs = pd.DataFrame({"split": pd.Categorical(split.values)}, index=names)
    obsm = {k: z[k].astype(np.float32) for k in z.files if k != "obs_names"}
    return types.SimpleNamespace(obs=obs, obsm=obsm, n_obs=len(names)), splits


def geometry(S, z, actions, D0, t, chunk=512) -> dict:
    """Alignment and the three uniformity terms on one split's per-perturbation OT pairs."""
    keys = S[z["cell"]].astype(np.float64)
    codes = pd.factorize(z["cond"])[0]
    zero = np.zeros(S.shape[1], np.float32)
    F = np.stack([actions.get(c, zero) for c in z["cond"]]).astype(np.float64)
    Q = S[z["partners"][:, 0]].astype(np.float64) + F
    align = float(((Q - keys) ** 2).sum(1).mean() / D0)
    kn = (keys * keys).sum(1)
    u_all, u_same, u_cross = [], [], []
    n = len(keys)
    for i in range(0, n, chunk):
        idx = np.arange(i, min(i + chunk, n))
        q = Q[idx]
        L = -((q * q).sum(1)[:, None] + kn[None, :] - 2 * q @ keys.T) / (t * D0)
        same = codes[idx][:, None] == codes[None, :]
        same[np.arange(len(idx)), idx] = False  # the positive itself is neither
        cross = codes[idx][:, None] != codes[None, :]
        u_all.append(logsumexp(L, axis=1) - np.log(n))
        u_same.append(logsumexp(L, axis=1, b=same) - np.log(same.sum(1)))
        u_cross.append(logsumexp(L, axis=1, b=cross) - np.log(cross.sum(1)))
    u_all, u_same, u_cross = (float(np.concatenate(u).mean()) for u in (u_all, u_same, u_cross))
    return {"alignment": align, "uniformity": u_all, "uniformity_same_pert": u_same,
            "uniformity_cross_pert": u_cross, "infonce_rows": align / t + u_all}


def direction_uniformity(actions, genes) -> float:
    """Wang & Isola uniformity (t = 2) of the unit action directions, over distinct gene pairs."""
    U = np.stack([actions[g] for g in genes]).astype(np.float64)
    nrm = np.linalg.norm(U, axis=1)
    if (nrm == 0).any():
        return float("nan")
    U /= nrm[:, None]
    d2 = ((U[:, None] - U[None]) ** 2).sum(-1)
    off = ~np.eye(len(U), dtype=bool)
    return float(logsumexp(-2 * d2[off]) - np.log(off.sum()))


def score(cfg, adata, space, S, val, actions, genes, ms, D0, t) -> dict:
    """Every number of one action set."""
    lr = [np.linalg.norm(actions[g]) / np.linalg.norm(ms[g]) for g in genes] if actions else [0.0]
    r = instance_retrieval(cfg, adata, space, "val", actions)
    ident, _ = identification(cfg, adata, space, "val", actions, genes)
    return {**geometry(S, val, actions, D0, t), "dir_uniformity": direction_uniformity(actions, genes) if actions else
            float("nan"), "len_ratio_median": float(np.median(lr)), "mrr": r["mrr"],
            "same_pert@10": r["same_pert@10"], "ident_top1": ident["top1"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", action="store_true", help="also retrain seed 0 over the tau and rho grids")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    adata, splits = light_adata(cfg)
    genes = sorted(splits["train_singles"])
    taus = read_json(path(cfg, "results_dir") / "infonce" / "best_tau.json")["tau"]
    out = path(cfg, "results_dir") / "align_uniform"
    out.mkdir(parents=True, exist_ok=True)
    rows, grid_rows = [], []
    for space in cfg["ot"]["spaces"]:
        S = adata.obsm[space]
        D0 = scale_D0(S[(adata.obs["split"] == "train").to_numpy()], cfg["seed"])
        t = taus[space]
        val = load_pairs(cfg, space, "val")
        ms = mean_shift_actions(cfg, adata, space)
        sets, _ = build_action_sets(cfg, adata, space, genes, ms)
        for (name, seed), acts in sets.items():
            rows.append({"space": space, "set": name, "seed": seed, "t": t,
                         **score(cfg, adata, space, S, val, acts, genes, ms, D0, t)})
            log.info("%s %s s%d: align %.3f unif %.3f MRR %.3f top1 %.3f", space, name, seed,
                     rows[-1]["alignment"], rows[-1]["uniformity"], rows[-1]["mrr"], rows[-1]["ident_top1"])
        pd.DataFrame(rows).to_csv(out / "sets.csv", index=False)
        if not a.grids:
            continue
        c = cfg["infonce"]
        for kind, values in (("tau", c["tau_grid"]), ("rho", c["anchor_grid"])):
            for v in values:
                tau, rho = (v, 0.0) if kind == "tau" else (t, v)
                F, g, _, info = train_actions(cfg, adata, splits, space, cfg["seed"], tau, verbose=True, rho=rho)
                acts = dict(zip(g, F))
                grid_rows.append({"space": space, "grid": kind, "tau": tau, "rho": rho, "t": t,
                                  "steps": info["steps"],
                                  **score(cfg, adata, space, S, val, acts, genes, ms, D0, t)})
                log.info("%s %s=%g: align %.3f unif %.3f MRR %.3f top1 %.3f", space, kind, v,
                         grid_rows[-1]["alignment"], grid_rows[-1]["uniformity"], grid_rows[-1]["mrr"],
                         grid_rows[-1]["ident_top1"])
                pd.DataFrame(grid_rows).to_csv(out / "grids.csv", index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
