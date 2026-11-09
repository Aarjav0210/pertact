"""Entropic optimal-transport pairing of control and perturbed cells in a frozen state space (Sinkhorn-Knopp).

No perturbed cell has a measured "before" state, so each perturbed cell is paired with the control cells it most
plausibly came from: the coupling that moves the control cloud onto the perturbed cloud at least total squared
displacement, blurred by an entropy term (epsilon = eps_scale x median squared distance between control cells,
so the blur does not depend on how far a perturbation moves cells). Problems are solved per perturbation and per gem group, so
pairs never cross batches.

    python -I -m pertact.ot              # every space in config ot.spaces, train and val splits
    python -I -m pertact.ot --check      # compare with POT and recover a known synthetic shift

Writes results/ot/<space>__<split>.npz with, for every perturbed cell of that split (single perturbations only):
    cell (row in processed h5ad), cond, partners (k control rows), weights (k, sum to 1)
and a label-free variant (<space>__<split>__pooled.npz) where each gem group's controls are matched to all of its
perturbed cells at once, without using the perturbation label (for perturbation classification).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .config import get_logger, load_config, path, read_json, seed_everything, write_json
from .data import CONTROL

log = get_logger("ot")


# --------------------------------------------------------------------------- Sinkhorn-Knopp (log domain)
def sq_dists(X, Y):
    """Pairwise squared Euclidean distances, (n, m)."""
    d = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
    return np.maximum(d, 0.0)


def sinkhorn(C, eps, a=None, b=None, max_iter=5000, tol=1e-6):
    """Entropic OT  min_P <P, C> - eps H(P)  s.t.  P 1 = a, P^T 1 = b,  by Sinkhorn-Knopp scaling in the log domain.

    Alternately rescales rows and columns of the Gibbs kernel K = exp(-C / eps) until both marginals match. Working
    with the dual potentials (f, g) instead of the scalings u = exp(f / eps), v = exp(g / eps) avoids under/overflow
    for small eps. Returns the coupling P and a dict with iterations, final marginal error and transport cost.
    """
    n, m = C.shape
    a = np.full(n, 1.0 / n) if a is None else a
    b = np.full(m, 1.0 / m) if b is None else b
    la, lb = np.log(a), np.log(b)
    f, g = np.zeros(n), np.zeros(m)
    err, it = np.inf, 0
    for it in range(1, max_iter + 1):
        f = eps * (la - logsumexp((g[None, :] - C) / eps, axis=1))  # row scaling: make P 1 = a
        g = eps * (lb - logsumexp((f[:, None] - C) / eps, axis=0))  # column scaling: make P^T 1 = b
        if it % 10 == 0 or it == max_iter:
            logP = (f[:, None] + g[None, :] - C) / eps
            err = np.abs(np.exp(logsumexp(logP, axis=1)) - a).sum()  # columns are exact after the g update
            if err < tol:
                break
    P = np.exp((f[:, None] + g[None, :] - C) / eps)
    return P, {"iters": it, "marginal_err": float(err), "cost": float((P * C).sum())}


def pair(Xc, Xp, eps_scale, k, rng=None):
    """Couple controls Xc (n, d) to perturbed cells Xp (m, d). For each perturbed cell return its k most-coupled
    controls and their renormalised weights, plus diagnostics."""
    C = sq_dists(Xc, Xp)
    rng = rng or np.random.default_rng(0)
    # epsilon from the spread of the control cloud only: a perturbation's shift must not change the blur
    sub = Xc[rng.choice(len(Xc), min(len(Xc), 500), replace=False)]
    med = float(np.median(sq_dists(sub, sub)[np.triu_indices(len(sub), 1)]))
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
    o = cfg["ot"]
    X = np.asarray(adata.obsm[space], dtype=np.float64)
    obs = adata.obs
    cond = obs["condition"].astype(str).to_numpy()
    split = obs["split"].astype(str).to_numpy()
    gem = obs[cfg["data"]["batch_key"]].astype(str).to_numpy()
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
    z = np.load(path(cfg, "results_dir") / "ot" / f"{space}__{split}{'__pooled' if pooled else ''}.npz")
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- checks
def check(cfg):
    """1) agree with POT's Sinkhorn on random problems; 2) recover a known translation on synthetic clouds."""
    import ot as pot

    rng = np.random.default_rng(0)
    res = {}
    diffs = []
    for _ in range(5):
        Xc, Xp = rng.normal(size=(300, 20)), rng.normal(size=(120, 20)) + 0.5
        C = sq_dists(Xc, Xp)
        eps = cfg["ot"]["eps_scale"] * np.median(C)
        P, _ = sinkhorn(C, eps, tol=1e-9)
        Q = pot.sinkhorn(np.full(300, 1 / 300), np.full(120, 1 / 120), C, eps, method="sinkhorn_log",
                         numItermax=20000, stopThr=1e-12)
        diffs.append(float(np.abs(P - Q).max() / Q.max()))
    res["max_relative_diff_vs_POT"] = max(diffs)
    # synthetic: perturbed = a random subset of controls shifted by a known vector, plus small noise
    n, m, d = 800, 200, 30
    Xc = rng.normal(size=(n, d)) * np.linspace(3, 0.3, d)
    src = rng.choice(n, m, replace=False)
    shift = rng.normal(size=d) * 2
    Xp = Xc[src] + shift + rng.normal(scale=0.05, size=(m, d))
    top, w, info = pair(Xc, Xp, cfg["ot"]["eps_scale"], 5)
    res["synthetic_true_source_is_top1"] = float((top[:, 0] == src).mean())
    res["synthetic_true_source_in_top5"] = float((top == src[:, None]).any(1).mean())
    # a translation adds only row/column constants to the quadratic cost, so the plan must not change
    top_s, _, _ = pair(Xc + shift, Xp, cfg["ot"]["eps_scale"], 5)  # same controls, so same epsilon
    res["plan_unchanged_by_translation"] = bool((top_s == top).all())
    res["synthetic_info"] = {k: info[k] for k in ("iters", "marginal_err", "perplexity")}
    for k, v in res.items():
        log.info("%s: %s", k, v)
    write_json(res, path(cfg, "results_dir") / "ot" / "checks.json")
    return res


def main():
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
    from .evaluate import load_all

    adata, splits = load_all(cfg)
    diag = []
    for space in cfg["ot"]["spaces"]:
        diag += run_space(cfg, adata, splits, space)
    pd.DataFrame(diag).to_csv(path(cfg, "results_dir") / "ot" / "diagnostics.csv", index=False)


if __name__ == "__main__":
    main()
