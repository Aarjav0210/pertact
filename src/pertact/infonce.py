"""Perturbation actions learned with InfoNCE on OT-paired cells.

Model, in a frozen state space s = g(x):   s(x_pert) ≈ s(x_ctrl) + f(a),   f(a) a free vector per training gene.

Loss. A batch holds P perturbations × K OT pairs each (control i_n → perturbed cell j_n, perturbation a_n):
    q_n = s(x_ctrl_n) + f(a_n)                     query: where the control should land
    k_m = s(x_pert_m)                              keys: the batch's perturbed cells
    L_nm = -||q_n - k_m||² / (tau · D0)            D0 = mean squared distance between random training cells
    loss = ½ [CE(L, diag) + CE(Lᵀ, diag)]
OT-partner positives: each query's positive is its own OT partner, and the K-1 other cells of the same perturbation are
negatives. f(a) is identical for them, so only carrying the control state over separates them. Each step draws the
control partner of every perturbed cell from its OT weights.

Length anchor (Week 2 addendum, used only because post-hoc rescaling did not hold up): an optional weak
squared-error term on each query and its own OT partner,
    loss += lambda · mean_n ||q_n - k_n||² / D0,      lambda = rho / tau.
The InfoNCE positive logit already pulls q_n toward k_n with weight 1/tau (in D0 units), so rho is the anchor's
weight as a fraction of that built-in pull (rho = 0.1: the positive pull is 10% stronger). rho is the smallest value
on a grid that brings the median length ratio ||f(a)|| / ||mean OT shift|| on training pairs into [0.8, 1.25].

    python -I -m pertact.infonce tune      # temperature grid on validation retrieval, seed 0, each space
    python -I -m pertact.infonce train     # every space × seed with the chosen temperature
    python -I -m pertact.infonce anchor    # rho grid (seed 0, training-pair length ratio), then every space × seed
Writes embeddings/infonce__<space>__s<seed>.npz (emb, genes) and results/infonce/ histories; the anchored actions go
to embeddings/infonce_anchor__<space>__s<seed>.npz.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd
import torch

from .actions import length_ratio, save_actions
from .config import VAL_SEED_OFFSET, get_logger, load_config, path, read_json, seed_everything, write_json
from .encoders import load_all, scale_D0
from .ot import load_pairs
from .retrieval import instance_retrieval

log = get_logger("infonce")


class PairData:
    """OT pairs of one split in one state space, grouped by perturbation."""

    def __init__(self, cfg, S: np.ndarray, space: str, split: str, genes: list[str]):
        z = load_pairs(cfg, space, split)
        self.S = torch.as_tensor(S, dtype=torch.float32)
        self.gidx = {g: i for i, g in enumerate(genes)}
        self.cell = z["cell"]
        self.a = np.array([self.gidx[c] for c in z["cond"]])
        self.partners, self.cum = z["partners"], np.cumsum(z["weights"], axis=1)
        self.by_pert = [np.flatnonzero(self.a == i) for i in range(len(genes))]

    def batch(self, rng, P, K):
        """One batch: P perturbations with K pairs each. For every perturbed cell, one of its partner controls is drawn
        with probability equal to its OT weight. Returns control states, perturbed-cell states and gene indices."""
        perts = rng.choice([i for i, ix in enumerate(self.by_pert) if len(ix)], size=P, replace=False)
        rows = np.concatenate([rng.choice(self.by_pert[p], size=K, replace=len(self.by_pert[p]) < K) for p in perts])
        u = rng.random(len(rows))[:, None]
        pick = (u > self.cum[rows]).sum(1).clip(max=self.partners.shape[1] - 1)  # sample a partner from OT weights
        ctrl = self.partners[rows, pick]
        return (self.S[torch.as_tensor(ctrl)], self.S[torch.as_tensor(self.cell[rows])],
                torch.as_tensor(self.a[rows]))


def infonce_loss(F, s_ctrl, s_pert, a, tau, D0, rho=0.0):
    """The loss of one batch: each query (control state + action vector) against every perturbed cell of the batch,
    scored by negative squared distance, symmetric over rows and columns. rho > 0 adds the length anchor."""
    q = s_ctrl + F[a]
    d2 = (q * q).sum(1, keepdim=True) + (s_pert * s_pert).sum(1)[None, :] - 2 * q @ s_pert.T
    L = -d2 / (tau * D0)
    y = torch.arange(len(a))
    loss = 0.5 * (torch.nn.functional.cross_entropy(L, y) + torch.nn.functional.cross_entropy(L.T, y))
    if rho:  # weak length anchor: extra pull of each query onto its own OT partner (see module docstring)
        loss = loss + (rho / tau) * ((q - s_pert) ** 2).sum(1).mean() / D0
    return loss


def train_actions(cfg, adata, splits, space, seed, tau, verbose=True, rho=0.0):
    """Learn one action vector per gene in one state space. The vectors start at zero and are the only thing trained.
    Training stops when the loss on fixed validation batches has not improved for `patience` checks; the best
    vectors are returned with their genes, the loss history and a summary."""
    c = cfg["infonce"]
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    genes = sorted(splits["train_singles"])
    S = np.asarray(adata.obsm[space], dtype=np.float32)
    tr = PairData(cfg, S, space, "train", genes)
    va = PairData(cfg, S, space, "val", genes)
    D0 = scale_D0(S[(adata.obs["split"] == "train").to_numpy()], seed)
    vrng = np.random.default_rng(VAL_SEED_OFFSET + seed)
    val_batches = [va.batch(vrng, c["P"], c["K"]) for _ in range(c["n_val_batches"])]
    F = torch.zeros(len(genes), S.shape[1], requires_grad=True)
    opt = torch.optim.Adam([F], lr=c["lr"])
    best, best_F, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for step in range(1, c["max_steps"] + 1):
        sc, sp, a = tr.batch(rng, c["P"], c["K"])
        loss = infonce_loss(F, sc, sp, a, tau, D0, rho)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % c["eval_every"] == 0:
            with torch.no_grad():
                vl = float(np.mean([infonce_loss(F, *b, tau, D0, rho).item() for b in val_batches]))
            hist.append({"step": step, "train_loss": loss.item(), "val_loss": vl})
            if vl < best - 1e-4:
                best, best_F, bad = vl, F.detach().clone(), 0
            else:
                bad += 1
            if bad >= c["patience"]:
                break
    if best_F is None:
        raise RuntimeError(f"no validation check ran in {step} steps: max_steps must be at least eval_every ({c['eval_every']})")
    if verbose:
        log.info("%s seed %d tau %.3g rho %.3g: %d steps, best val loss %.4f (chance %.4f), %.0fs", space, seed, tau,
                 rho, step, best, np.log(c["P"] * c["K"]), time.time() - t0)
    with torch.no_grad():  # how much of the final validation loss is the anchor term
        parts = [(infonce_loss(best_F, *b, tau, D0).item(), infonce_loss(best_F, *b, tau, D0, rho).item())
                 for b in val_batches]
    nce, tot = map(np.mean, zip(*parts))
    return best_F.numpy(), genes, pd.DataFrame(hist), {"steps": step, "best_val_loss": best, "D0": D0, "rho": rho,
                                                       "val_infonce_part": float(nce),
                                                       "val_anchor_share": float((tot - nce) / tot)}


def anchor(cfg):
    """Pick rho per space on seed 0 by the training-pair length ratio (no validation or test signal), then train
    every seed. Validation retrieval for every grid value is logged for the record, not used to choose."""
    adata, splits = load_all(cfg)
    c = cfg["infonce"]
    taus = read_json(path(cfg, "results_dir") / "infonce" / "best_tau.json")["tau"]
    lo, hi = c["anchor_target"]
    rows, chosen = [], {}
    for space in cfg["ot"]["spaces"]:
        for rho in [0.0] + list(c["anchor_grid"]):
            F, genes, hist, info = train_actions(cfg, adata, splits, space, cfg["seed"], taus[space], rho=rho)
            lr = length_ratio(cfg, adata, space, F, genes)
            r = instance_retrieval(cfg, adata, space, "val", dict(zip(genes, F)))
            rows.append({"space": space, "tau": taus[space], **info, "len_ratio_median": float(np.median(lr)),
                         "len_ratio_min": float(lr.min()), "len_ratio_max": float(lr.max()), **r})
            log.info("  rho %.3g: length ratio median %.2f (%.2f-%.2f), anchor share of val loss %.3f, val MRR %.3f",
                     rho, np.median(lr), lr.min(), lr.max(), info["val_anchor_share"], r["mrr"])
            if space not in chosen and rho > 0 and lo <= np.median(lr) <= hi:
                chosen[space] = rho
        if space not in chosen:
            chosen[space] = float(max(c["anchor_grid"]))
            log.warning("%s: no rho on the grid reached the length target; using the largest (%.3g)", space, chosen[space])
    d = path(cfg, "results_dir") / "infonce"
    pd.DataFrame(rows).to_csv(d / "anchor_grid.csv", index=False)
    write_json({"selected_by": f"smallest rho with median training-pair length ratio in [{lo}, {hi}], seed 0",
                "rho": chosen, "lambda": {sp: chosen[sp] / taus[sp] for sp in chosen}}, d / "anchor.json")
    log.info("chosen rho: %s", chosen)
    for space in cfg["ot"]["spaces"]:
        for seed in cfg["eval"]["seeds"]:
            F, genes, hist, info = train_actions(cfg, adata, splits, space, seed, taus[space], rho=chosen[space])
            save_actions(cfg, space, seed, F, genes, info, hist, taus[space], name="infonce_anchor")


def tune(cfg):
    """Train once per temperature and state space on seed 0, and keep the temperature with the best forward-test
    score (MRR) on validation cells."""
    adata, splits = load_all(cfg)
    rows = []
    for space, tau in itertools.product(cfg["ot"]["spaces"], cfg["infonce"]["tau_grid"]):
        F, genes, hist, info = train_actions(cfg, adata, splits, space, cfg["seed"], tau)
        r = instance_retrieval(cfg, adata, space, "val", dict(zip(genes, F)))
        rows.append({"space": space, "tau": tau, **info, **r})
        log.info("  val instance retrieval: MRR %.4f, recall@10 %.3f, same-perturbation@10 %.3f",
                 r["mrr"], r["recall@10"], r["same_pert@10"])
    df = pd.DataFrame(rows)
    d = path(cfg, "results_dir") / "infonce"
    d.mkdir(parents=True, exist_ok=True)
    old = d / "tuning.csv"
    if old.exists():  # keep other spaces' rows when tuning a subset
        prev = pd.read_csv(old)
        df = pd.concat([prev[~prev.space.isin(df.space.unique())], df])
    df.to_csv(old, index=False)
    best = {sp: float(g.sort_values("mrr", ascending=False).iloc[0]["tau"]) for sp, g in df.groupby("space")}
    write_json({"selected_by": "max validation instance-retrieval MRR, seed 0", "tau": best}, d / "best_tau.json")
    log.info("best tau: %s", best)


def train_all(cfg):
    """Train the final action vectors: every state space and seed, with the chosen temperature."""
    adata, splits = load_all(cfg)
    best = read_json(path(cfg, "results_dir") / "infonce" / "best_tau.json")["tau"]
    for space in cfg["ot"]["spaces"]:
        for seed in cfg["eval"]["seeds"]:
            F, genes, hist, info = train_actions(cfg, adata, splits, space, seed, best[space])
            save_actions(cfg, space, seed, F, genes, info, hist, best[space])


def main():
    """Dispatch tune, train and anchor."""
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tune", "train", "anchor"])
    ap.add_argument("--spaces", nargs="*", default=None, help="override config ot.spaces")
    ap.add_argument("--max-steps", type=int, default=None, help="override for quick smoke tests")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.spaces:
        cfg["ot"]["spaces"] = a.spaces
    if a.max_steps:
        cfg["infonce"]["max_steps"] = a.max_steps
    {"tune": tune, "train": train_all, "anchor": anchor}[a.what](cfg)


if __name__ == "__main__":
    main()
