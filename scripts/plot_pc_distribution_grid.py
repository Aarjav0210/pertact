"""Grid of control vs perturbed distributions on selected PCs of X_pca_g (train split).

Rows = perturbations, columns = PCs. Each panel: Gaussian KDE of control (gray) and perturbed (berry) cells.
Columns share the x range (0.5th-99.5th percentile of control + shown perturbations on that PC); each panel's
density axis is free. Colors follow research-logbook/theme v1.2.0 (control = gray, measured = berry).
"""
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
THEME = Path.home() / "research-logbook/theme"
plt.style.use(THEME / "inference.mplstyle")
tok = json.loads((THEME / "tokens.json").read_text())["plot_palette"]

PERTS = ["CEBPE", "SPI1", "KLF1", "IKZF3", "TP73", "SET", "HOXA13", "BAK1"]
PCS = [1, 3, 4, 9, 15, 32, 13, 19]  # 1-based

obs = ad.read_h5ad(ROOT / "data/processed/norman2019.h5ad", backed="r").obs
emb = np.load(ROOT / "data/processed/state_embeddings.npz")
assert (emb["obs_names"] == obs.index.to_numpy()).all()
X = emb["X_pca_g"]
tr = (obs["split"] == "train").to_numpy()
cond = obs["condition"].astype(str).to_numpy()
Xc = X[tr & (cond == "control")]
Xp = {p: X[tr & (cond == p)] for p in PERTS}
sd0 = Xc.std(0)

fig, axes = plt.subplots(len(PERTS), len(PCS), figsize=(2.0 * len(PCS), 1.35 * len(PERTS)), sharex="col")
for j, pc in enumerate(PCS):
    k = pc - 1
    allv = np.concatenate([Xc[:, k]] + [Xp[p][:, k] for p in PERTS])
    lo, hi = np.quantile(allv, [0.005, 0.995])
    grid = np.linspace(lo, hi, 300)
    yc = gaussian_kde(Xc[:, k])(grid)
    for i, p in enumerate(PERTS):
        ax = axes[i, j]
        yp = gaussian_kde(Xp[p][:, k])(grid)
        ax.fill_between(grid, yc, color=tok["gray"], alpha=0.25, lw=0)
        ax.plot(grid, yc, color=tok["gray"], lw=1.2, label="control")
        ax.fill_between(grid, yp, color=tok["berry"], alpha=0.25, lw=0)
        ax.plot(grid, yp, color=tok["berry"], lw=1.4, label="perturbed")
        d = (Xp[p][:, k].mean() - Xc[:, k].mean()) / sd0[k]
        ax.text(0.97, 0.92, f"{d:+.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color=tok["blue"] if d > 0 else tok["berry"])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(lo, hi)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_title(f"PC{pc}", fontsize=11)
        if j == 0:
            ax.set_ylabel(f"{p}\nn={len(Xp[p])}", rotation=0, ha="right", va="center", fontsize=10)
axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
fig.tight_layout(h_pad=0.4, w_pad=0.6)

out = ROOT / "figures/pc_distribution_grid_X_pca_g_train"
fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{out}.pdf", bbox_inches="tight")
print(f"saved {out}.png  n_ctrl={len(Xc)}")
