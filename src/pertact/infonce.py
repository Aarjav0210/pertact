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

Cosine variant (Week 4, CLIP-style). The squared-distance score lets the length of f(a) act as a temperature, which
is where the overshoot comes from. Here only directions are compared:
    L_nm = cos( f(a_n), s(x_pert_m) - s(x_ctrl_n) ) / tau      action against the displacement control n -> cell m
with the same symmetric cross-entropy. f(a) cannot use its length, so it is stored at unit length, and tau is a
parameter learned with the vectors, as in CLIP (it starts at cosine.tau_init and is kept above cosine.tau_min).

    python -I -m pertact.infonce tune      # temperature grid on validation retrieval, seed 0, each space
    python -I -m pertact.infonce train     # every space × seed with the chosen temperature
    python -I -m pertact.infonce anchor    # rho grid (seed 0, training-pair length ratio), then every space × seed
    python -I -m pertact.infonce cosine    # the cosine, unit-length variant: every space × seed
    python -I -m pertact.infonce soft      # soft-positive labels: both scores × every value of soft_grid × space × seed
    python -I -m pertact.infonce table     # the naming table: the cosine loss with one thing changed at a time

Naming table (Week 5). Three switches of the cosine loss, each a run of actions.TABLE_RUNS, scored by pertact.naming:
the start (a cell's OT partner, or the mean control of its gem group, which needs no pairing), the training example
(one cell, or the average of infonce.set_size cells of a perturbation), and a projection of the displacement (none,
the computed whitening, or a square matrix trained with the vectors from the identity). With a projection the vectors
live in the projected space, so those sets are for naming only and are read by pertact.naming alone.

Soft positives (Week 4). With the labels above, the K-1 other cells of a query's perturbation are wrong answers, exactly
like the cells of other perturbations. `soft_targets` gives three levels instead: 1 for the query's own OT partner, s
for the other cells of the same perturbation, 0 for cells of other perturbations, each row normalised to sum to 1. s = 0
is the labelling above; s = 1 ignores the pairing and keeps only the perturbation. In the stored pairs a control is a
partner of a single cell of a perturbation about 99% of the time, so the query's own partner is the only coupled cell.
The vectors are saved as infonce_soft<s> (distance score) and infonce_cos_soft<s> (cosine score).
Writes embeddings/infonce__<space>__s<seed>.npz (emb, genes) and results/infonce/ histories; the anchored actions go
to embeddings/infonce_anchor__<space>__s<seed>.npz and the cosine ones to embeddings/infonce_cos__<space>__s<seed>.npz.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd
import torch

from .actions import TABLE_RUNS, length_ratio, save_actions, soft_name, table_name, whitening
from .config import VAL_SEED_OFFSET, get_logger, load_config, path, read_json, seed_everything, write_json
from .encoders import load_all, scale_D0
from .ot import gem_mean_states, load_pairs
from .retrieval import instance_retrieval

log = get_logger("infonce")


class PairData:
    """OT pairs of one split in one state space, grouped by perturbation. With `adata` given, the start of a cell is
    the mean control of its gem group instead of one of its OT partners (no pairing is used)."""

    def __init__(self, cfg, S: np.ndarray, space: str, split: str, genes: list[str], adata=None):
        z = load_pairs(cfg, space, split)
        self.S = torch.as_tensor(S, dtype=torch.float32)
        self.gidx = {g: i for i, g in enumerate(genes)}
        self.cell = z["cell"]
        self.a = np.array([self.gidx[c] for c in z["cond"]])
        self.partners, self.cum = z["partners"], np.cumsum(z["weights"], axis=1)
        self.by_pert = [np.flatnonzero(self.a == i) for i in range(len(genes))]
        self.ctrl_mean = None if adata is None else torch.as_tensor(gem_mean_states(cfg, adata, S, split, self.cell), dtype=torch.float32)

    def batch(self, rng, P, K, set_size=1):
        """One batch: P perturbations with K examples each. An example is one perturbed cell with its start: one of
        its partner controls, drawn with probability equal to its OT weight, or the mean control of its gem group.
        With set_size > 1 an example is the average of that many cells of the perturbation and of their starts.
        Returns start states, perturbed states and gene indices."""
        perts = rng.choice([i for i, ix in enumerate(self.by_pert) if len(ix)], size=P, replace=False)
        if set_size == 1:
            rows = np.concatenate([rng.choice(self.by_pert[p], size=K, replace=len(self.by_pert[p]) < K) for p in perts])
        else:  # K sets per perturbation; the cells of one set are distinct whenever the perturbation has enough cells
            rows = np.concatenate([rng.choice(self.by_pert[p], size=set_size, replace=len(self.by_pert[p]) < set_size)
                                   for p in perts for _ in range(K)])
        if self.ctrl_mean is None:
            u = rng.random(len(rows))[:, None]
            pick = (u > self.cum[rows]).sum(1).clip(max=self.partners.shape[1] - 1)  # sample a partner from OT weights
            start = self.S[torch.as_tensor(self.partners[rows, pick])]
        else:
            start = self.ctrl_mean[torch.as_tensor(rows)]
        pert, a = self.S[torch.as_tensor(self.cell[rows])], torch.as_tensor(self.a[rows])
        if set_size > 1:
            start, pert, a = start.view(-1, set_size, start.shape[1]).mean(1), pert.view(-1, set_size, pert.shape[1]).mean(1), a[::set_size]
        return start, pert, a


def soft_targets(a, s):
    """The label table of one batch, one row per query: 1 for the query's own partner (the diagonal), s for the other
    cells of the same perturbation, 0 for cells of other perturbations, with every row scaled to sum to 1. The table
    is the same read by columns, so it also labels the other direction (each cell choosing among the queries)."""
    T = s * (a[:, None] == a[None, :]).float()
    T.fill_diagonal_(1.0)
    return T / T.sum(1, keepdim=True)


def infonce_loss(F, s_ctrl, s_pert, a, tau, D0, rho=0.0, soft=None):
    """The loss of one batch: each query (control state + action vector) against every perturbed cell of the batch,
    scored by negative squared distance, symmetric over rows and columns. rho > 0 adds the length anchor.
    `soft` (a number s) replaces the one right answer per query by the three-level labels of `soft_targets`."""
    q = s_ctrl + F[a]
    d2 = (q * q).sum(1, keepdim=True) + (s_pert * s_pert).sum(1)[None, :] - 2 * q @ s_pert.T
    L = -d2 / (tau * D0)
    y = torch.arange(len(a)) if soft is None else soft_targets(a, soft)
    loss = 0.5 * (torch.nn.functional.cross_entropy(L, y) + torch.nn.functional.cross_entropy(L.T, y))
    if rho:  # weak length anchor: extra pull of each query onto its own OT partner (see module docstring)
        loss = loss + (rho / tau) * ((q - s_pert) ** 2).sum(1).mean() / D0
    return loss


def cosine_infonce_loss(F, s_ctrl, s_pert, a, tau, soft=None, proj=None):
    """The cosine loss of one batch (CLIP-style): each action vector against the displacement from its own control
    to every perturbed cell of the batch, both at unit length, symmetric over rows and columns. The length of F
    has no effect, so only the temperature tau sets how sharp the contrast is. `soft` as in infonce_loss.
    `proj` (a matrix, rows = projected directions) is applied to every displacement before it is compared; projecting
    a displacement is the same as projecting both cells and subtracting. The vectors then live in the projected space."""
    f = torch.nn.functional.normalize(F[a], dim=1)
    disp = s_pert[None, :, :] - s_ctrl[:, None, :]                                          # [n, m]: control n -> cell m
    disp = torch.nn.functional.normalize(disp if proj is None else disp @ proj.T, dim=2)
    L = (disp * f[:, None, :]).sum(2) / tau
    y = torch.arange(len(a)) if soft is None else soft_targets(a, soft)
    return 0.5 * (torch.nn.functional.cross_entropy(L, y) + torch.nn.functional.cross_entropy(L.T, y))


def train_actions(cfg, adata, splits, space, seed, tau, verbose=True, rho=0.0, score="distance", soft=None,
                  start="ot", set_size=1, projection=None):
    """Learn one action vector per gene in one state space. The vectors start at zero and are the only thing trained.
    Training stops when the loss on fixed validation batches has not improved for `patience` checks; the best
    vectors are returned with their genes, the loss history and a summary.

    With score="cosine" the cosine loss is used instead: the vectors start from random directions (zero has no
    direction), `tau` is the starting temperature and is learned with them, and the vectors come back at unit length.
    `soft` (a number s) trains and validates with the three-level labels of `soft_targets` instead of one right answer.

    The naming table changes one thing at a time (cosine score only for the projection):
    `start`      "ot": a cell starts from one of its OT partners; "ctrl_mean": from the mean control of its gem group.
    `set_size`   cells of one perturbation averaged into one training example (1 = single cells).
    `projection` None; "white": the displacement is first rescaled by the fixed, computed `whitening`; "train": by a
                 square matrix that starts as the identity and is learned with the vectors. The projection that
                 goes with the returned vectors is in the summary under "proj"."""
    c = cfg["infonce"]
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    genes = sorted(splits["train_singles"])
    S = np.asarray(adata.obsm[space], dtype=np.float32)
    tr = PairData(cfg, S, space, "train", genes, adata if start == "ctrl_mean" else None)
    va = PairData(cfg, S, space, "val", genes, adata if start == "ctrl_mean" else None)
    D0 = scale_D0(S[(adata.obs["split"] == "train").to_numpy()], seed)
    vrng = np.random.default_rng(VAL_SEED_OFFSET + seed)
    val_batches = [va.batch(vrng, c["P"], c["K"], set_size) for _ in range(c["n_val_batches"])]
    cosine = score == "cosine"
    if projection is not None and not cosine:
        raise ValueError("the projection is defined for the cosine score only")
    proj = None
    if cosine:
        cc = c["cosine"]
        if projection == "white":
            proj = torch.as_tensor(whitening(cfg, adata, space, start))
        elif projection == "train":
            proj = torch.eye(S.shape[1]).requires_grad_()
        F = torch.randn(len(genes), S.shape[1] if proj is None else proj.shape[0]).requires_grad_()
        log_tau = torch.tensor(float(np.log(tau)), requires_grad=True)
        tau_now = lambda: log_tau.exp().clamp(min=cc["tau_min"])
        batch_loss = lambda sc, sp, a: cosine_infonce_loss(F, sc, sp, a, tau_now(), soft, proj)
        groups = [{"params": [F, log_tau] if cc["learn_tau"] else [F]}]
        if projection == "train":
            groups.append({"params": [proj], "lr": c["projection"]["lr"]})
        opt = torch.optim.Adam(groups, lr=c["lr"])
    else:
        F = torch.zeros(len(genes), S.shape[1], requires_grad=True)
        batch_loss = lambda sc, sp, a: infonce_loss(F, sc, sp, a, tau, D0, rho, soft)
        opt = torch.optim.Adam([F], lr=c["lr"])
    best, best_F, best_tau, best_proj, bad, hist = np.inf, None, tau, None, 0, []
    t0 = time.time()
    for step in range(1, c["max_steps"] + 1):
        sc, sp, a = tr.batch(rng, c["P"], c["K"], set_size)
        loss = batch_loss(sc, sp, a)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % c["eval_every"] == 0:
            with torch.no_grad():
                vl = float(np.mean([batch_loss(*b).item() for b in val_batches]))
            hist.append({"step": step, "train_loss": loss.item(), "val_loss": vl} | ({"tau": tau_now().item()} if cosine else {}))
            if vl < best - 1e-4:
                best, best_F, bad = vl, F.detach().clone(), 0
                best_tau = tau_now().item() if cosine else tau
                best_proj = None if proj is None else proj.detach().clone()
            else:
                bad += 1
            if bad >= c["patience"]:
                break
    if best_F is None:
        raise RuntimeError(f"no validation check ran in {step} steps: max_steps must be at least eval_every ({c['eval_every']})")
    if cosine:
        if verbose:
            log.info("%s seed %d cosine%s: %d steps, best val loss %.4f (chance %.4f), temperature %.3g -> %.4f, %.0fs", space, seed,
                     "" if (start, set_size, projection) == ("ot", 1, None) else f" [start {start}, {set_size} cell(s) per example, projection {projection}]",
                     step, best, np.log(c["P"] * c["K"]), tau, best_tau, time.time() - t0)
        unit = torch.nn.functional.normalize(best_F, dim=1).numpy()
        extra = ({} if soft is None else {"soft": soft}) | ({} if (start, set_size, projection) == ("ot", 1, None) else
                                                             {"start": start, "set_size": set_size, "projection": str(projection)})
        return unit, genes, pd.DataFrame(hist), {"steps": step, "best_val_loss": best, "D0": D0, "score": "cosine", "tau_start": tau,
                                                 "tau_learned": best_tau} | extra | ({} if best_proj is None else {"proj": best_proj.numpy()})
    if verbose:
        log.info("%s seed %d tau %.3g rho %.3g: %d steps, best val loss %.4f (chance %.4f), %.0fs", space, seed, tau,
                 rho, step, best, np.log(c["P"] * c["K"]), time.time() - t0)
    with torch.no_grad():  # how much of the final validation loss is the anchor term
        parts = [(infonce_loss(best_F, *b, tau, D0, soft=soft).item(), infonce_loss(best_F, *b, tau, D0, rho, soft).item())
                 for b in val_batches]
    nce, tot = map(np.mean, zip(*parts))
    return best_F.numpy(), genes, pd.DataFrame(hist), {"steps": step, "best_val_loss": best, "D0": D0, "rho": rho,
                                                       "val_infonce_part": float(nce),
                                                       "val_anchor_share": float((tot - nce) / tot)} | ({} if soft is None else {"soft": soft})


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


def train_cosine(cfg):
    """Train the cosine, unit-length action vectors: every state space and seed, saved as "infonce_cos" with the
    temperature each run learned."""
    adata, splits = load_all(cfg)
    for space in cfg["ot"]["spaces"]:
        for seed in cfg["eval"]["seeds"]:
            F, genes, hist, info = train_actions(cfg, adata, splits, space, seed, cfg["infonce"]["cosine"]["tau_init"], score="cosine")
            save_actions(cfg, space, seed, F, genes, info, hist, info["tau_learned"], name="infonce_cos")


def train_soft(cfg, scores=None, values=None):
    """Train with the three-level labels for every score, soft value, state space and seed. The distance score keeps
    the temperature chosen for the one-right-answer labels; the cosine score learns its own, as before."""
    adata, splits = load_all(cfg)
    c = cfg["infonce"]
    taus = read_json(path(cfg, "results_dir") / "infonce" / "best_tau.json")["tau"]
    for score in scores or ("cosine", "distance"):
        for s in c["soft_grid"] if values is None else values:
            for space in cfg["ot"]["spaces"]:
                for seed in cfg["eval"]["seeds"]:
                    tau = c["cosine"]["tau_init"] if score == "cosine" else taus[space]
                    F, genes, hist, info = train_actions(cfg, adata, splits, space, seed, tau, score=score, soft=float(s))
                    save_actions(cfg, space, seed, F, genes, info, hist, info.get("tau_learned", tau), name=soft_name(score, s))


def train_table(cfg, only=None):
    """Train the runs of the naming table (actions.TABLE_RUNS) for every state space and seed: the cosine loss with
    another start, with sets of cells as training examples, and with a computed or a trained projection. `only`
    names the stored sets to train (default all)."""
    adata, splits = load_all(cfg)
    c = cfg["infonce"]
    for start, set_size, projection in TABLE_RUNS:
        name = table_name(cfg, start, set_size, projection)
        if only and name not in only:
            continue
        n = c["set_size"] if set_size == "set" else set_size
        for space in cfg["ot"]["spaces"]:
            for seed in cfg["eval"]["seeds"]:
                F, genes, hist, info = train_actions(cfg, adata, splits, space, seed, c["cosine"]["tau_init"], score="cosine",
                                                     start=start, set_size=n, projection=projection)
                save_actions(cfg, space, seed, F, genes, info, hist, info["tau_learned"], name=name)


def main():
    """Dispatch tune, train, anchor, cosine, soft and table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tune", "train", "anchor", "cosine", "soft", "table"])
    ap.add_argument("--only", nargs="*", default=None, help="table: stored sets to train, by name (default every run of the table)")
    ap.add_argument("--scores", nargs="*", choices=["cosine", "distance"], default=None, help="soft: which scores to train (default both)")
    ap.add_argument("--values", nargs="*", type=float, default=None, help="soft: soft values to train (default config infonce.soft_grid)")
    ap.add_argument("--spaces", nargs="*", default=None, help="override config ot.spaces")
    ap.add_argument("--max-steps", type=int, default=None, help="override for quick smoke tests")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.spaces:
        cfg["ot"]["spaces"] = a.spaces
    if a.max_steps:
        cfg["infonce"]["max_steps"] = a.max_steps
    if a.what == "soft":
        train_soft(cfg, a.scores, a.values)
        return
    if a.what == "table":
        train_table(cfg, a.only)
        return
    {"tune": tune, "train": train_all, "anchor": anchor, "cosine": train_cosine}[a.what](cfg)


if __name__ == "__main__":
    main()
