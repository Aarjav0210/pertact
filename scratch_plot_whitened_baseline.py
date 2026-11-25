"""Figure: closed-form whitened mean shift vs cosine InfoNCE, identification on validation singles (Norman)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import research_style as rs

ROOT = Path(__file__).resolve().parent
tok = json.loads((Path.home() / "research-style/tokens.json").read_text())["colors"]
df = pd.read_csv(ROOT / "results/whitened_baseline/summary.csv")
df["grp"] = df["set"].str.replace(r"_s\d$", "", regex=True)
g = df.groupby(["space", "grp"]).mean(numeric_only=True)

SETS = [("mean_shift", "mean shift", tok["control"]),
        ("white_sd", "÷ sd", tok["fill_variant"]),
        ("white_diag", "÷ var", tok["variant"]),
        ("white_full_shr", "LDA, shrunk", tok["variant_text"]),
        ("infonce_cos", "cosine InfoNCE", tok["ours"])]
SPACES = [("X_pca_g", "PCA-50"), ("X_scvi30", "scVI-30")]
TIERS = ["weak", "medium", "strong"]

rs.apply()
fig, rects = rs.rows_page([[("A", 1), ("B", 1.4)]], title="In PCA-50, whitening the mean shift recovers most of cosine InfoNCE's naming gain")
w = 0.16
pad = lambda r: [r[0] + 0.07, r[1], r[2] - 0.07, r[3]]
ax = fig.add_axes(pad(rects["A"])); rs.letter(ax, "A")
for i, (k, lab, c) in enumerate(SETS):
    x = np.arange(len(SPACES)) + (i - 2) * w
    ax.bar(x, [g.loc[(s, k), "cos_ctrl_mean_top1"] for s, _ in SPACES], w, color=c, label=lab)
ax.set_xticks(range(len(SPACES)), [n for _, n in SPACES])
ax.set_ylabel("top-1, 106 candidates")
ax.set_ylim(0, 0.65)
ax.set_title("all genes")


ax = fig.add_axes(pad(rects["B"])); rs.letter(ax, "B")
for i, (k, lab, c) in enumerate(SETS):
    x = np.arange(len(TIERS)) + (i - 2) * w
    ax.bar(x, [g.loc[("X_pca_g", k), f"cos_ctrl_mean_top1_{t}"] for t in TIERS], w, color=c, label=lab)
ax.set_xticks(range(len(TIERS)), TIERS)
ax.set_ylim(0, 0.65)
ax.set_title("PCA-50, by effect strength")
ax.legend(frameon=False, loc="upper left", fontsize=7, handlelength=1, labelspacing=0.3)

rs.save(fig, ROOT / "figures/whitened_baseline", legend=(
    "**Closed-form whitened mean shift vs cosine InfoNCE (Norman validation singles, label-free start).** "
    "Each held-out perturbed cell starts from its gem group's mean validation control; the 105 training genes are "
    "ranked by cosine between the action direction and the displacement start → cell (identify_cos_ctrl_mean, the "
    "same test as results/infonce_cos/retrieval.csv; chance ≈ 0.01). m_a = OT mean shift on training pairs. "
    "÷ sd, ÷ var: m_a divided per dimension by the sd or variance of gem-centred training controls (÷ var = diagonal "
    "LDA). LDA, shrunk: (Σ + λI)⁻¹ m_a, λ = mean eigenvalue of Σ. Cosine InfoNCE: mean of seeds 0–2. No training "
    "for any whitened set. Full LDA without shrinkage (not shown) matches ÷ var in PCA-50 (0.354) and fails in "
    "scVI-30 (0.158; control covariance condition number 1.3e5). **A.** All 105 genes. **B.** PCA-50 by tertile of "
    "training mean-shift norm. Source: scratch_whitened_baseline.py, scratch_plot_whitened_baseline.py; "
    "results/whitened_baseline/summary.csv (job 30157572)."))
