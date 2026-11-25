"""Figure for the angle-length split: results/angle_length/scan.csv -> figures/angle_length_split.

    python -I scripts/plot_angle_length.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import research_style as rs

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "angle_length"
C = rs.C
SPACES = {"X_pca_g": "PCA-50", "X_scvi30": "scVI-30"}
DIRS = {"shuffled": ("shuffled", C["control"]), "mean_shift": ("mean shift", C["control_text"]),
        "infonce": ("InfoNCE", C["ours"]), "infonce_dir_only": ("InfoNCE, direction only", C["variant"])}
LENS = {"ms_len": "-", "ls_len": "--"}
METRICS = [("mrr", "retrieval MRR"), ("ident_top1", "identification top-1")]


def curves(ax, d, y):
    for name, (lab, col) in DIRS.items():
        for pol, ls in LENS.items():
            k = d[(d.dir == name) & (d.len == pol)]
            if k.empty:
                continue
            m = k.groupby("c")[["len_ratio_median", y]].mean().reset_index()
            sd = k.groupby("c")[y].std().fillna(0).to_numpy()
            ax.plot(m.len_ratio_median, m[y], ls, marker="o", ms=2.5, lw=1, color=col,
                    label=lab if pol == "ms_len" else None)
            if k.seed.nunique() > 1:
                ax.fill_between(m.len_ratio_median, m[y] - sd, m[y] + sd, color=col, alpha=0.15, lw=0)
    ax.axvline(1, color=C["control"], lw=0.5, zorder=0)


def main():
    rs.apply()
    d = pd.read_csv(RES / "scan.csv")
    fig, R = rs.rows_page([[("A", rs.flex(1, ml=14)), ("B", rs.flex(1, ml=14))],
                           [("C", rs.flex(1, ml=14)), ("D", rs.flex(1, ml=14))]])
    for k in R:
        R[k] = [R[k][0], R[k][1] + 0.025, R[k][2], R[k][3] - 0.025]
    for row, (y, ylab) in zip(("AB", "CD"), METRICS):
        for k, space in zip(row, SPACES):
            ax = fig.add_axes(R[k]); rs.letter(ax, k)
            curves(ax, d[d.space == space], y)
            ax.set_xlabel(r"median $\|f(a)\| \,/\, \|m_a\|$")
            ax.set_ylabel(ylab)
            ax.set_title(SPACES[space], loc="left")
            if k == "D":
                ax.set_ylim(0, 0.27); ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    legend = """**Action directions compared at matched length (validation cells).** Every action is f(a) = c·ℓ_a·u_a with
‖u_a‖ = 1; c ∈ {0.5, 0.75, 1, 1.25, 1.5, 2, 2.5} is one global factor and the x-axis is the resulting median
length relative to the mean shift m_a (mean OT displacement on training pairs). Directions u: mean shift; shuffled
(mean-shift directions permuted across genes); InfoNCE (direction of the freely trained action); InfoNCE, direction
only (trained with its length held at ‖m_a‖, random start). Solid: ℓ_a = ‖m_a‖. Dashed: ℓ_a = u_a·m_a, the
least-squares length on the line u_a. InfoNCE curves are the mean of seeds 0–2 with ±1 sd bands. Vertical line:
mean-shift length. **A, B.** Instance retrieval MRR (query = heaviest OT partner control + f(a), candidates = all
validation perturbed cells). **C, D.** Identification top-1 over 106 candidates from the pooled-OT start. Source:
scripts/angle_length_split.py, scripts/plot_angle_length.py.
"""
    rs.save(fig, ROOT / "figures" / "angle_length_split", legend=legend)
    plt.close(fig)


if __name__ == "__main__":
    main()
