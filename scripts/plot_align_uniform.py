"""Figure for the alignment-uniformity probe: results/align_uniform/{sets,grids}.csv -> figures/align_uniform_plane.

    python -I scripts/plot_align_uniform.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import research_style as rs

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "align_uniform"
C = rs.C
SPACES = {"X_pca_g": ("PCA-50", "o"), "X_scvi30": ("scVI-30", "s")}
ALIGN_MAX = 0.8  # axes stop here; the high-temperature grid points lie beyond (values in the legend)
# set -> (label or None, colour by role, label offset in points). Gray references, blue InfoNCE, yellow variants.
SETS = {
    "zero": ("zero", C["control"], (4, 3)),
    "shuffled_mean_shift": ("shuffled", C["control"], (4, 3)),
    "mean_shift": ("mean shift", C["control_text"], (-2, 9)),
    "infonce": ("InfoNCE", C["ours"], (5, -2)),
    "infonce_calibrated": (None, C["variant"], (0, 0)),
    "infonce_scaled_per_action": (None, C["variant"], (0, 0)),
    "infonce_dir_meanshift_len": ("InfoNCE dir, MS length", C["variant"], (6, -7)),
    "meanshift_dir_infonce_len": ("MS dir, InfoNCE length", C["variant"], (5, -2)),
    "infonce_anchor": (None, C["variant"], (0, 0)),
}
GRID = {"tau": (C["ours"], r"$\tau$"), "rho": (C["variant"], r"$\rho$")}


def per_set(s, space, cols):
    return s[s.space == space].groupby("set", sort=False)[cols].mean()


def plane(ax, s, g, space, ylim):
    d = per_set(s, space, ["alignment", "uniformity"])
    for kind, (col, sym) in GRID.items():
        k = g[(g.space == space) & (g.grid == kind)].sort_values(kind)
        ax.plot(k.alignment, k.uniformity, "-", color=col, lw=0.8, zorder=1, clip_on=True)
        kin = k[k.alignment <= ALIGN_MAX]
        ax.scatter(kin.alignment, kin.uniformity, s=8, color=col, zorder=2)
    for name, r in d.iterrows():
        lab, col, off = SETS[name]
        ax.scatter(r.alignment, r.uniformity, s=24, color=col, edgecolor=C["ink"], lw=0.4, zorder=3)
        if lab:
            ax.annotate(lab, (r.alignment, r.uniformity), xytext=off, textcoords="offset points", fontsize=6.5,
                        color=C["ink"], ha="right" if off[0] < 0 else "left", va="center")
    ax.set_xlim(0.38, ALIGN_MAX)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"alignment  $\|q - k^{+}\|^2 / D_0$")
    ax.set_ylabel("uniformity")
    ax.set_title(SPACES[space][0], loc="left")


def relative(s, g):
    """Every set and grid point relative to its space's mean shift: alignment ratio and uniformity difference."""
    ms = s[s.set == "mean_shift"].set_index("space")
    out = []
    for df, src in ((s, "set"), (g, "grid")):
        d = df.copy()
        d["align_rel"] = d.alignment / d.space.map(ms.alignment)
        d["unif_rel"] = d.uniformity - d.space.map(ms.uniformity)
        d["src"] = src
        out.append(d)
    return out


def trade(ax, s, g, x, y):
    for space, (_, mk) in SPACES.items():
        d = s[s.space == space].groupby("set", sort=False)[[x, y]].mean()
        for name, r in d.iterrows():
            ax.scatter(r[x], r[y], marker=mk, s=18, color=SETS[name][1], edgecolor=C["ink"], lw=0.4, zorder=3)
        for kind, (col, _) in GRID.items():
            k = g[(g.space == space) & (g.grid == kind)].sort_values(kind)
            ax.plot(k[x], k[y], "-", marker=mk, ms=2.5, color=col, lw=0.8, zorder=2)


def main():
    rs.apply()
    s = pd.read_csv(RES / "sets.csv")
    g = pd.read_csv(RES / "grids.csv")
    fig, R = rs.rows_page([[("A", rs.flex(1, ml=18)), ("B", rs.flex(1, ml=18))],
                           [("C", rs.flex(1, ml=18)), ("D", rs.flex(1, ml=18))]])
    for k in R:  # room under the bottom row for its axis labels
        R[k] = [R[k][0], R[k][1] + 0.025, R[k][2] - 0.012, R[k][3] - 0.025]
    ylims = {"X_pca_g": (-23, -12), "X_scvi30": (-10, -6)}
    for k, space in zip("AB", SPACES):
        ax = fig.add_axes(R[k]); rs.letter(ax, k); plane(ax, s, g, space, ylims[space])
    sr, gr = relative(s, g)
    ax = fig.add_axes(R["C"]); rs.letter(ax, "C")
    trade(ax, sr, gr, "align_rel", "ident_top1")
    ax.set_xlim(0.95, ALIGN_MAX / 0.40); ax.set_ylim(-0.01, 0.29)
    ax.set_xlabel("alignment / mean-shift alignment"); ax.set_ylabel("identification top-1")
    ax = fig.add_axes(R["D"]); rs.letter(ax, "D")
    trade(ax, sr[sr.set != "zero"], gr[gr.align_rel <= ALIGN_MAX / 0.40], "unif_rel", "mrr")
    ax.set_xlim(-8, 1); ax.set_ylim(0, 0.2)
    ax.set_xlabel("uniformity − mean-shift uniformity"); ax.set_ylabel("retrieval MRR")
    off = g[g.alignment > ALIGN_MAX]
    beyond = "; ".join(f"{SPACES[r.space][0]} τ = {r.tau:g}: alignment {r.alignment:.2f}, uniformity "
                       f"{r.uniformity:.1f}, MRR {r.mrr:.3f}, top-1 {r.ident_top1:.3f}" for r in off.itertuples())
    legend = f"""**Alignment and uniformity of every action set (validation cells).** Query q = s(c) + f(a), with c
the heaviest OT partner control; keys = all 9,444 held-out perturbed cells; k⁺ = the query's own cell.
Alignment = mean ‖q − k⁺‖² / D₀ (D₀ = mean squared distance between random training cells). Uniformity = mean
over queries of log mean over keys of exp(−‖q − k‖² / (t·D₀)), t = the space's chosen temperature (PCA-50 0.03,
scVI-30 0.1); lower = more spread. Alignment / t + uniformity + log N is the InfoNCE row loss with all N = 9,444 validation cells as
a negative. Large markers: saved action sets (InfoNCE sets averaged over seeds 0–2). Gray = references (zero,
shuffled mean shift, mean shift); blue = InfoNCE as trained; yellow = InfoNCE rescaled (one global α, per-action
c_a), length-swapped (InfoNCE direction at mean-shift length, mean-shift direction at InfoNCE length) and the
anchored model (unlabelled yellow points sit between mean shift and InfoNCE dir, MS length). Lines: seed 0
retrained over τ ∈ {{0.01, 0.03, 0.1, 0.3}} (blue) and over the anchor weight ρ ∈ {{0.01, 0.03, 0.1, 0.3, 1}} at the
chosen τ (yellow, running from InfoNCE toward mean shift as ρ grows). Grid points beyond the axes: {beyond}.
**A, B.** The plane per state space. **C.** Identification top-1 (pooled-OT start, 106 candidates) against
alignment relative to the mean shift. **D.** Retrieval MRR against uniformity relative to the mean shift (zero
action omitted). C, D: circles PCA-50, squares scVI-30. Source: scripts/align_uniform_probe.py,
scripts/plot_align_uniform.py.
"""
    rs.save(fig, ROOT / "figures" / "align_uniform_plane", legend=legend)
    plt.close(fig)


if __name__ == "__main__":
    main()
