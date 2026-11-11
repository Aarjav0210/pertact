"""Entropic optimal-transport pairing of control and perturbed cells in a frozen state space (Sinkhorn-Knopp, via POT).

No perturbed cell has a measured "before" state, so each perturbed cell is paired with the control cells it most
plausibly came from: the coupling that moves the control cloud onto the perturbed cloud at least total squared
displacement, blurred by an entropy term (epsilon = eps_scale x median squared distance between control cells,
so the blur does not depend on how far a perturbation moves cells). Problems are solved per perturbation and per gem group, so
pairs never cross batches.

    python -I -m pertact.ot              # every space in config ot.spaces, train and val splits
    python -I -m pertact.ot --check      # recover a known shift on synthetic clouds

Writes results/ot/<space>__<split>.npz with, for every perturbed cell of that split (single perturbations only):
    cell (row in processed h5ad), cond, partners (k control rows), weights (k, sum to 1)
and a label-free variant (<space>__<split>__pooled.npz) where each gem group's controls are matched to all of its
perturbed cells at once, without using the perturbation label (for perturbation classification).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import ot as pot  # the POT library; this module is pertact.ot
import pandas as pd

from .config import get_logger, load_config, path, seed_everything, write_json
from .data import CONTROL, cell_labels
from .encoders import load_all

log = get_logger("ot")
EPS_SUBSAMPLE = 500  # controls sampled to estimate the median squared distance that sets epsilon


# --------------------------------------------------------------------------- entropic OT, solved by POT
def sinkhorn(C, eps, max_iter=10000, tol=1e-9):
    """Entropic OT between two uniform clouds:  min_P <P, C> - eps H(P)  s.t.  P 1 = 1/n,  P^T 1 = 1/m.

    Solved by POT's log-domain Sinkhorn-Knopp, which stays stable for small eps. Returns the coupling P and a dict
    with the iterations used, the remaining row-sum error and the transport cost."""
    n, m = C.shape
    a, b = np.full(n, 1.0 / n), np.full(m, 1.0 / m)
    P, log_ = pot.sinkhorn(a, b, C, eps, method="sinkhorn_log", numItermax=max_iter, stopThr=tol, log=True)
    return P, {"iters": int(log_["niter"]), "marginal_err": float(np.abs(P.sum(1) - a).sum()), "cost": float((P * C).sum())}


def pair(Xc, Xp, eps_scale, k, rng=None):
    """Couple controls Xc (n, d) to perturbed cells Xp (m, d). For each perturbed cell return its k most-coupled
    controls and their renormalised weights, plus diagnostics."""
    C = pot.dist(Xc, Xp)  # squared Euclidean distance, every control against every perturbed cell
    rng = rng or np.random.default_rng(0)
    # epsilon from the spread of the control cloud only: a perturbation's shift must not change the blur
    sub = Xc[rng.choice(len(Xc), min(len(Xc), EPS_SUBSAMPLE), replace=False)]
    med = float(np.median(pot.dist(sub, sub)[np.triu_indices(len(sub), 1)]))
    eps = eps_scale * med
    P, info = sinkhorn(C, eps)
    col = P / P.sum(0, keepdims=True)  # p(control | perturbed cell)
    kk = min(k, C.shape[0])
    top = np.argsort(-col, axis=0)[:kk].T  # (m, kk)
    w = np.take_along_axis(col.T, top, axis=1)
    w = w / w.sum(1, keepdims=True)
    ent = -(col * np.log(col + 1e-300)).sum(0)
    rand = C[rng.integers(0, C.shape[0], C.shape[1]), np.arange(C.shape[1])]
    hard = C[top[:, 0], np.arange(C.shape[1])]
    info.update({"eps": eps, "median_ctrl_sqdist": med, "perplexity": float(np.exp(ent).mean()),
                 "cost_hard_partner": float(hard.mean()), "cost_random_partner": float(rand.mean())})
    return top, w, info


# --------------------------------------------------------------------------- pairing for the dataset
def run_space(cfg, adata, splits, space):
    """Solve and save every pairing problem of one state space, for training cells and for validation cells: first
    per perturbation and gem group, then pooled (all single-perturbed cells of a gem group at once, labels unused)."""
    o = cfg["ot"]
    X = np.asarray(adata.obsm[space], dtype=np.float64)
    cond, gem, split = cell_labels(adata.obs, cfg["data"]["batch_key"])
    rng = np.random.default_rng(cfg["seed"])
    out_dir = path(cfg, "results_dir") / "ot"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = []
    for sp in ("train", "val"):
        t0 = time.time()
        ctrl_by_gem = {g_: np.flatnonzero((split == sp) & (cond == CONTROL) & (gem == g_)) for g_ in np.unique(gem)}
        cells, conds, partners, weights = [], [], [], []
        for a in splits["train_singles"]:
            for g_, ci in ctrl_by_gem.items():
                pi = np.flatnonzero((split == sp) & (cond == a) & (gem == g_))
                if len(pi) == 0:
                    continue
                top, w, info = pair(X[ci], X[pi], o["eps_scale"], o["k"], rng)
                cells.append(pi); conds += [a] * len(pi); partners.append(ci[top]); weights.append(w)
                diag.append({"space": space, "split": sp, "mode": "per_perturbation", "cond": a, "gem": g_,
                             "n_ctrl": len(ci), "n_pert": len(pi)} | info)
        np.savez(out_dir / f"{space}__{sp}.npz", cell=np.concatenate(cells), cond=np.array(conds, dtype=str),
                 partners=np.vstack(partners), weights=np.vstack(weights).astype(np.float32))
        # label-free: each gem's controls against all of its single-perturbed cells together
        cells, conds, partners, weights = [], [], [], []
        is_single = np.isin(cond, splits["train_singles"])
        for g_, ci in ctrl_by_gem.items():
            pi = np.flatnonzero((split == sp) & is_single & (gem == g_))
            top, w, info = pair(X[ci], X[pi], o["eps_scale"], o["k"], rng)
            cells.append(pi); conds += cond[pi].tolist(); partners.append(ci[top]); weights.append(w)
            diag.append({"space": space, "split": sp, "mode": "pooled", "cond": "all", "gem": g_,
                         "n_ctrl": len(ci), "n_pert": len(pi)} | info)
        np.savez(out_dir / f"{space}__{sp}__pooled.npz", cell=np.concatenate(cells), cond=np.array(conds, dtype=str),
                 partners=np.vstack(partners), weights=np.vstack(weights).astype(np.float32))
        d = pd.DataFrame([r for r in diag if r["space"] == space and r["split"] == sp and r["mode"] == "per_perturbation"])
        log.info("%s %s: %d problems in %.0fs | max marginal err %.1e | median iters %d | partner cost OT %.1f vs random %.1f "
                 "| perplexity %.1f", space, sp, len(d), time.time() - t0, d.marginal_err.max(), d.iters.median(),
                 (d.cost_hard_partner * d.n_pert).sum() / d.n_pert.sum(), (d.cost_random_partner * d.n_pert).sum() / d.n_pert.sum(),
                 (d.perplexity * d.n_pert).sum() / d.n_pert.sum())
    return diag


def load_pairs(cfg, space, split, pooled=False):
    """One saved set of pairs: cell, cond, partners (k controls per cell) and weights."""
    z = np.load(path(cfg, "results_dir") / "ot" / f"{space}__{split}{'__pooled' if pooled else ''}.npz")
    return {k: z[k] for k in z.files}


def start_states(S, pairs) -> np.ndarray:
    """The start state of every paired cell: the weighted average of its partner controls' states."""
    return (S[pairs["partners"]] * pairs["weights"][..., None]).sum(1)


# --------------------------------------------------------------------------- checks
def check(cfg):
    """Two checks of the pairing on synthetic clouds: a known shift is recovered, and moving the controls does not
    change the plan."""
    rng = np.random.default_rng(0)
    # perturbed = a random subset of controls shifted by a known vector, plus small noise
    n, m, d = 800, 200, 30
    Xc = rng.normal(size=(n, d)) * np.linspace(3, 0.3, d)
    src = rng.choice(n, m, replace=False)
    shift = rng.normal(size=d) * 2
    Xp = Xc[src] + shift + rng.normal(scale=0.05, size=(m, d))
    top, w, info = pair(Xc, Xp, cfg["ot"]["eps_scale"], 5)
    res = {"synthetic_true_source_is_top1": float((top[:, 0] == src).mean()),
           "synthetic_true_source_in_top5": float((top == src[:, None]).any(1).mean())}
    # a translation adds only row/column constants to the quadratic cost, so the plan must not change
    top_s, _, _ = pair(Xc + shift, Xp, cfg["ot"]["eps_scale"], 5)  # same controls, so same epsilon
    res["plan_unchanged_by_translation"] = bool((top_s == top).all())
    res["synthetic_info"] = {k: info[k] for k in ("iters", "marginal_err", "perplexity")}
    for k, v in res.items():
        log.info("%s: %s", k, v)
    write_json(res, path(cfg, "results_dir") / "ot" / "checks.json")
    return res


def main():
    """Run the checks, or pair every configured state space."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed_everything(cfg["seed"])
    (path(cfg, "results_dir") / "ot").mkdir(parents=True, exist_ok=True)
    if a.check:
        check(cfg)
        return
    adata, splits = load_all(cfg)
    diag = []
    for space in cfg["ot"]["spaces"]:
        diag += run_space(cfg, adata, splits, space)
    pd.DataFrame(diag).to_csv(path(cfg, "results_dir") / "ot" / "diagnostics.csv", index=False)


if __name__ == "__main__":
    main()
