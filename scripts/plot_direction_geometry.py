"""Direction geometry of the action sets, after Wang & Isola (2020), with the real mean shift and a held-out target.

Unit directions u(a) = f(a) / ||f(a)|| of 105 training genes. Target: each gene's mean transition on VALIDATION pairs,
m_val(a) = mean_j [s(x_j) - sum_k w_jk s(c_jk)], so no set is scored against itself.
    agreement  = mean_a ||u(a) - m_val(a)/||m_val(a)|| ||²     (= 2 - 2 cos; lower = closer to held-out transitions)
    spread     = log mean_{a != b} exp(-2 ||u(a) - u(b)||²)    (Wang & Isola uniformity; lower = more spread)
Sets: random unit vectors, mean shift (training pairs), InfoNCE (seeds 0-2), InfoNCE direction only (seeds 0-2).
Tiers: tertiles of the training mean-shift norm.

    python -I scripts/plot_direction_geometry.py
-> figures/direction_geometry.{pdf,png,legend.md}, figures/direction_sphere_{pca,scvi}.{...},
   results/direction_geometry/summary.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import research_style as rs
from scipy.special import logsumexp

from align_uniform_probe import light_adata
from pertact.actions import load_actions, mean_shift_actions
from pertact.config import load_config, path
from pertact.ot import load_pairs, start_states

ROOT = Path(__file__).resolve().parents[1]
SPACES = {"X_pca_g": ("PCA-50", "o", "pca"), "X_scvi30": ("scVI-30", "s", "scvi")}
SEEDS = [0, 1, 2]
TIERS = ["strong", "medium", "weak"]


def unit(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def spread(U):
    d2 = ((U[:, None] - U[None]) ** 2).sum(-1)
    off = ~np.eye(len(U), dtype=bool)
    return float(logsumexp(-2 * d2[off]) - np.log(off.sum()))


def val_targets(cfg, adata, space, genes):
    S = adata.obsm[space]
    z = load_pairs(cfg, space, "val")
    disp = pd.DataFrame(S[z["cell"]] - start_states(S, z))
    disp["cond"] = z["cond"]
    m = disp.groupby("cond").mean()
    return np.stack([m.loc[g].to_numpy() for g in genes])


def collect(cfg, adata, genes):
    """Unit directions of every set, per space, plus the held-out targets and tiers."""
    out = {}
    for space in SPACES:
        ms = mean_shift_actions(cfg, adata, space)
        M = np.stack([ms[g] for g in genes])
        norm = np.linalg.norm(M, axis=1)
        q = np.quantile(norm, [1 / 3, 2 / 3])
        tier = np.where(norm >= q[1], "strong", np.where(norm >= q[0], "medium", "weak"))
        sets = {("random", 0): unit(np.random.default_rng(0).standard_normal(M.shape)), ("mean_shift", 0): unit(M)}
        for s in SEEDS:
            nce = load_actions(cfg, space, s)
            sets[("infonce", s)] = unit(np.stack([nce[g] for g in genes]))
            z = np.load(path(cfg, "results_dir") / "angle_length" / f"infonce_dir_only__{space}__s{s}.npz")
            assert list(z["genes"].astype(str)) == genes
            sets[("infonce_dir_only", s)] = unit(z["emb"])
        out[space] = {"sets": sets, "target": unit(val_targets(cfg, adata, space, genes)), "tier": tier}
    return out


def summary(G):
    rows = []
    for space, g in G.items():
        for (name, s), U in g["sets"].items():
            a = ((U - g["target"]) ** 2).sum(1)
            C = U @ U.T
            ev = np.linalg.eigvalsh(U.T @ U)
            rows.append({"space": space, "set": name, "seed": s, "agreement": a.mean(), "spread": spread(U),
                         "pair_cos_median": float(np.median(C[np.triu_indices(len(U), 1)])),
                         "eff_dim": float(ev.sum() ** 2 / (ev ** 2).sum()),
                         **{f"agreement_{t}": a[g["tier"] == t].mean() for t in TIERS}})
    return pd.DataFrame(rows)


COL = {"random": rs.C["target"], "mean_shift": rs.C["control_text"], "infonce": rs.C["ours"],
       "infonce_dir_only": rs.C["variant"]}
LAB = {"random": "random", "mean_shift": "mean shift", "infonce": "InfoNCE", "infonce_dir_only": "InfoNCE, direction only"}


def fig_geometry(df):
    fig, R = rs.rows_page([[("A", rs.flex(1, ml=18)), ("B", rs.flex(1, ml=18))]])
    for k in R:
        R[k] = [R[k][0], R[k][1] + 0.06, R[k][2] - 0.012, R[k][3] - 0.06]
    ax = fig.add_axes(R["A"]); rs.letter(ax, "A")
    for space, (_, mk, _) in SPACES.items():
        for name in COL:
            d = df[(df.space == space) & (df.set == name)]
            ax.scatter(d.spread.mean(), d.agreement.mean(), marker=mk, s=40, color=COL[name], edgecolor=rs.C["ink"],
                       lw=0.4, zorder=3, label=LAB[name] if space == "X_pca_g" else None)
    ax.set_xlabel("spread across genes  (uniformity)")
    ax.set_ylabel("distance to held-out transition")
    ax.legend(fontsize=6.5, frameon=False, loc="center right")
    ax = fig.add_axes(R["B"]); rs.letter(ax, "B")
    w, x = 0.2, np.arange(3)
    for i, (space, (lab, _, _)) in enumerate(SPACES.items()):
        for j, name in enumerate(["mean_shift", "infonce"]):
            d = df[(df.space == space) & (df.set == name)]
            v = [d[f"agreement_{t}"].mean() for t in TIERS]
            ax.bar(x + (2 * i + j - 1.5) * w, v, w, color=COL[name], alpha=1 if i == 0 else 0.5,
                   label=f"{LAB[name]}, {lab}")
    ax.set_xticks(x, TIERS)
    ax.set_xlabel("genes by mean-shift norm")
    ax.set_ylabel("distance to held-out transition")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    return fig


def ring_angles(U):
    """Project onto the set's top-2 plane, renormalise to the circle, rotate the mean direction to angle 0."""
    _, sv, Vt = np.linalg.svd(U, full_matrices=False)
    P = U @ Vt[:2].T
    th = np.arctan2(P[:, 1], P[:, 0])
    th = np.angle(np.exp(1j * (th - np.angle(np.exp(1j * th).mean()))))
    return th, float((sv[:2] ** 2).sum() / (sv ** 2).sum())


def vm_density(th, grid, kappa=8.0):
    return np.exp(kappa * np.cos(grid[:, None] - th[None])).mean(1) / (2 * np.pi * np.i0(kappa))


def fig_sphere(g):
    cols = ["random", "mean_shift", "infonce"]
    fig, R = rs.rows_page([[(f"{r}{c}", rs.flex(1, ml=16)) for c in range(3)] for r in "AB"])
    for k in R:
        R[k] = [R[k][0], R[k][1] + 0.02, R[k][2] - 0.01, R[k][3] - 0.02]
    grid = np.linspace(-np.pi, np.pi, 361)
    keep = {}
    for c, name in enumerate(cols):
        U = g["sets"][(name, 0)]
        dist = np.sqrt(((U - g["target"]) ** 2).sum(1))
        th, keep[name] = ring_angles(U)
        dens = vm_density(th, grid)
        ax = fig.add_axes(R[f"A{c}"])
        if c == 0:
            rs.letter(ax, "A")
        ax.hist(dist, bins=np.linspace(0, 2, 21), color=COL[name])
        ax.set_xlim(0, 2)
        ax.set_title(LAB[name], loc="left")
        ax.set_xlabel(r"$\|u(a) - \hat m_{val}(a)\|$")
        if c == 0:
            ax.set_ylabel("genes")
        ax = fig.add_axes(R[f"B{c}"])
        if c == 0:
            rs.letter(ax, "B")
        r = 1 + 0.6 * dens / dens.max()
        ax.fill(np.r_[np.cos(grid), np.cos(grid[::-1]) * r[::-1]], np.r_[np.sin(grid), np.sin(grid[::-1]) * r[::-1]],
                color=COL[name], alpha=0.5, lw=0)
        ax.plot(np.cos(grid), np.sin(grid), color=rs.C["mute"], lw=0.6)
        ax.scatter(np.cos(th), np.sin(th), s=4, color=rs.C["ink"], zorder=3)
        ax.set_aspect("equal"); ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7); ax.axis("off")
    return fig, keep


def main():
    rs.apply()
    cfg = load_config(None)
    adata, splits = light_adata(cfg)
    genes = sorted(splits["train_singles"])
    G = collect(cfg, adata, genes)
    df = summary(G)
    out = path(cfg, "results_dir") / "direction_geometry"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)
    m = df.groupby(["space", "set"])[["agreement", "spread", "pair_cos_median", "eff_dim"]].mean()
    print(df.groupby(["space", "set"]).mean(numeric_only=True).round(3).to_string())
    nums = "; ".join(f"{SPACES[sp][0]} {LAB[n]}: distance {r.agreement:.2f}, spread {r.spread:.2f}"
                     for (sp, n), r in m.iterrows())
    fig = fig_geometry(df)
    rs.save(fig, ROOT / "figures" / "direction_geometry", legend=f"""**Action directions: spread across genes against
distance to held-out transitions.** u(a) = f(a)/‖f(a)‖ for the 105 training genes. Target m̂_val(a): the gene's mean
transition on validation OT pairs (perturbed state minus OT-weighted control states), normalised; no set is scored
against the data it was fitted on. Distance = mean ‖u(a) − m̂_val(a)‖² = 2 − 2 cos (lower = closer). Spread = log
mean over gene pairs of exp(−2‖u(a) − u(b)‖²), Wang & Isola's uniformity (lower = more spread). Random = 105 Gaussian
unit vectors. InfoNCE and InfoNCE direction only (trained with length held at the mean-shift length): mean of seeds
0–2. **A.** Circles PCA-50, squares scVI-30. **B.** Distance by tier of the training mean-shift norm (tertiles); dark
PCA-50, light scVI-30. Values: {nums}. Source: scripts/plot_direction_geometry.py.
""")
    plt.close(fig)
    for space, (lab, _, short) in SPACES.items():
        fig, keep = fig_sphere(G[space])
        dims = {"X_pca_g": 50, "X_scvi30": 30}[space]
        rs.save(fig, ROOT / "figures" / f"direction_sphere_{short}", legend=f"""**Action directions on the sphere,
{lab}, after Wang & Isola (2020) Fig. 2.** Columns: random unit vectors in {dims}-D, the mean shift (training pairs),
InfoNCE as trained (seed 0). **A.** Distance from each gene's direction to its own held-out mean transition,
‖u(a) − m̂_val(a)‖ = √(2 − 2 cos); 0 = same direction, √2 = orthogonal. **B.** Directions projected onto each set's
own top-2 plane, renormalised to the unit circle and rotated so their mean sits at angle 0; dots = genes, band =
von Mises density (κ = 8). The plane keeps {keep['random']:.0%}, {keep['mean_shift']:.0%} and {keep['infonce']:.0%}
of the squared norm, so B is a 2-D view; spread is computed in full {dims}-D (values in direction_geometry).
Source: scripts/plot_direction_geometry.py.
""")
        plt.close(fig)


if __name__ == "__main__":
    main()
