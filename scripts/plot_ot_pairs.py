"""Draw the OT pairs of a few perturbations in a 2-D slice of a frozen state space (training cells only).

One column per perturbation, one gem group.
  top row     each perturbed cell joined to its k OT partner controls (results/ot/<space>__train.npz)
  bottom row  the same cells joined to k random controls of the same gem group (the Week 1 pairing), as a reference

The slice is chosen per perturbation, from the plotted training cells:
  x = direction of the mean shift (perturbed mean - control mean), origin at the control mean
  y = direction of largest remaining spread among the plotted cells, orthogonal to x
Pairs are computed in every dimension of the space, so the slice shows only part of each pair's length. Each panel
prints the mean squared pair length in the full space and the share of it that is visible in the slice.

Run: .venv/bin/python -I scripts/plot_ot_pairs.py [--space X_pca_g] [--conds CEBPA AHR BCL2L11] [--gem 1]
Writes figures/ot_pairs_<space>.png and results/ot/pair_plot_stats_<space>.csv
"""
import argparse

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from pertact.config import load_config, path
from pertact.data import CONTROL, cell_labels

SURFACE, INK, INK2, MUTED, AXIS, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e1e0d9"
BLUE, CTRL = "#2a78d6", "#b4b2a9"


def slice_axes(Xc, Xp):
    """Origin (control mean), unit shift direction u, and the top remaining-spread direction v (orthogonal to u)."""
    o = Xc.mean(0)
    shift = Xp.mean(0) - o
    u = shift / np.linalg.norm(shift)
    R = np.vstack([Xc, Xp]) - o
    R = R - np.outer(R @ u, u)
    v = np.linalg.svd(R - R.mean(0), full_matrices=False)[2][0]
    return o, u, v, float(np.linalg.norm(shift))


def panel(ax, P2c, P2p, seg_idx, lim, note=None):
    """Draw one panel: controls, perturbed cells, the pair lines and the mean-shift arrow."""
    ax.set_facecolor(SURFACE)
    ax.scatter(P2c[:, 0], P2c[:, 1], s=5, c=CTRL, alpha=0.55, linewidths=0, zorder=1)
    segs = np.stack([P2c[seg_idx[:, 1]], P2p[seg_idx[:, 0]]], axis=1)
    ax.add_collection(LineCollection(segs, colors=BLUE, linewidths=0.6, alpha=0.32, zorder=2))
    ax.scatter(P2p[:, 0], P2p[:, 1], s=30, c=BLUE, edgecolors=SURFACE, linewidths=0.9, zorder=3)
    m = P2p.mean(0)
    ax.annotate("", xy=m, xytext=(0, 0), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, shrinkA=0, shrinkB=0, mutation_scale=12))
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, color=AXIS)
    ax.axhline(0, color=GRID, lw=0.8, zorder=0); ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    if note:
        ax.text(0, 1.025, note, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


def main():
    """Build the slices, draw OT pairs and random pairs, and save the figure and its numbers."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="X_pca_g")
    ap.add_argument("--conds", nargs="+", default=["CEBPA", "AHR", "BCL2L11"])
    ap.add_argument("--gem", default="1")
    a = ap.parse_args()
    cfg = load_config(None)
    rng = np.random.default_rng(cfg["seed"])
    obs = ad.read_h5ad(path(cfg, "processed_h5ad"), backed="r").obs
    X = np.load(path(cfg, "processed_h5ad").parent / "state_embeddings.npz")[a.space].astype(np.float64)
    cond, gem, split = cell_labels(obs, cfg["data"]["batch_key"])
    z = np.load(path(cfg, "results_dir") / "ot" / f"{a.space}__train.npz")
    row_of = {c: i for i, c in enumerate(z["cell"])}
    tiers = pd.read_csv(path(cfg, "results_dir") / "week2_effect_strength.csv").query("space == @a.space").set_index("gene")

    built, stats = [], []
    for c in a.conds:
        ci = np.flatnonzero((split == "train") & (cond == CONTROL) & (gem == a.gem))
        pi = np.flatnonzero((split == "train") & (cond == c) & (gem == a.gem))
        rows = np.array([row_of[i] for i in pi])
        partners, w = z["partners"][rows], z["weights"][rows].astype(np.float64)
        assert np.isin(partners, ci).all(), "OT partners must be training controls of the same gem group"
        k = partners.shape[1]
        pos = {g: j for j, g in enumerate(ci)}
        ot_idx = np.array([[i, pos[g]] for i in range(len(pi)) for g in partners[i]])
        rnd_idx = np.array([[i, j] for i in range(len(pi)) for j in rng.choice(len(ci), k, replace=False)])
        o, u, v, shift = slice_axes(X[ci], X[pi])
        B = np.stack([u, v], axis=1)
        P2c, P2p = (X[ci] - o) @ B, (X[pi] - o) @ B
        out = {"cond": c, "gem": a.gem, "n_ctrl": len(ci), "n_pert": len(pi), "shift_len": shift,
               "tier": tiers.loc[c, "tier"] if c in tiers.index else ""}
        for name, idx, ww in (("ot", ot_idx, w.ravel()), ("random", rnd_idx, np.full(len(rnd_idx), 1 / k))):
            d_full = X[pi][idx[:, 0]] - X[ci][idx[:, 1]]
            d_2d = P2p[idx[:, 0]] - P2c[idx[:, 1]]
            full = float((ww * (d_full ** 2).sum(1)).sum() / len(pi))
            vis = float((ww * (d_2d ** 2).sum(1)).sum() / len(pi))
            out |= {f"{name}_sq_len": full, f"{name}_visible_share": vis / full,
                    f"{name}_abs_dy": float((ww * np.abs(d_2d[:, 1])).sum() / len(pi))}
        stats.append(out)
        built.append((c, P2c, P2p, ot_idx, rnd_idx, out))

    ctrl = np.vstack([b[1] for b in built])
    pert = np.vstack([b[2] for b in built])
    lo = np.minimum(np.percentile(ctrl, 0.1, axis=0), pert.min(0))
    hi = np.maximum(np.percentile(ctrl, 99.9, axis=0), pert.max(0))
    pad = 0.04 * (hi - lo)
    lim = [(lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])]
    ratio = (lim[1][1] - lim[1][0]) / (lim[0][1] - lim[0][0])
    # layout in inches, so equal-aspect panels leave no dead space
    n, pw = len(built), 3.5
    ph = pw * ratio
    left, right, gap, top, mid, bottom = 1.15, 0.25, 0.5, 1.62, 0.9, 1.0
    W, H = left + n * pw + (n - 1) * gap + right, top + 2 * ph + mid + bottom
    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    axes = np.empty((2, n), dtype=object)
    for i in range(2):
        for j in range(n):
            y0 = bottom + (1 - i) * (ph + mid)
            axes[i, j] = fig.add_axes(((left + j * (pw + gap)) / W, y0 / H, pw / W, ph / H))
    for j, (c, P2c, P2p, ot_idx, rnd_idx, s) in enumerate(built):
        panel(axes[0, j], P2c, P2p, ot_idx, lim,
              note=f"Mean squared pair length {s['ot_sq_len']:.0f}, {100 * s['ot_visible_share']:.0f}% visible here")
        panel(axes[1, j], P2c, P2p, rnd_idx, lim,
              note=f"Mean squared pair length {s['random_sq_len']:.0f}, {100 * s['random_visible_share']:.0f}% visible here")
        x0 = (left + j * (pw + gap)) / W
        head = f"{c}: {s['tier']} effect" if s["tier"] else c
        fig.text(x0, 1 - 1.0 / H, head, fontsize=12.5, color=INK, fontweight="bold", va="bottom")
        fig.text(x0, 1 - 1.24 / H, f"{s['n_pert']} perturbed cells, {s['n_ctrl']} controls", fontsize=9.5, color=INK2,
                 va="bottom")
        axes[1, j].set_xlabel("along the mean shift", fontsize=9.5, color=INK2)
    for i, lab in enumerate([f"OT partners, {k} per cell", f"Random controls, {k} per cell"]):
        axes[i, 0].set_ylabel("largest remaining spread", fontsize=9.5, color=INK2)
        fig.text(0.28 / W, (bottom + (1 - i) * (ph + mid) + ph / 2) / H, lab, rotation=90, fontsize=11, color=INK,
                 fontweight="bold", va="center", ha="center")
    handles = [Line2D([], [], marker="o", ls="", ms=4, color=CTRL, label="control cell"),
               Line2D([], [], marker="o", ls="", ms=7, color=BLUE, markeredgecolor=SURFACE, label="perturbed cell"),
               Line2D([], [], color=BLUE, lw=1.2, alpha=0.6, label="pair: control to perturbed cell"),
               Line2D([], [], color=INK, lw=1.6, marker=">", ms=5, label="mean shift: control mean to perturbed mean")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, 0.12 / H))
    name = {"X_pca_g": "PCA-50", "X_scvi30": "scVI-30"}.get(a.space, a.space)
    fig.text(0.28 / W - 0.01, 1 - 0.38 / H, f"OT pairs in the {name} state space, training cells of gem group {a.gem}",
             fontsize=14, color=INK, fontweight="bold", va="center")
    fig.text(0.28 / W - 0.01, 1 - 0.66 / H, "Each panel is a 2-D slice of the space, chosen per perturbation. Same scale in "
             "every panel; axes are in state-space distance units.", fontsize=9.5, color=INK2, va="center")
    out = path(cfg, "figures_dir") / f"ot_pairs_{a.space}.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    sd = pd.DataFrame(stats)
    sd.to_csv(path(cfg, "results_dir") / "ot" / f"pair_plot_stats_{a.space}.csv", index=False)
    print(sd.round(3).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
