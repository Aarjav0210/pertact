"""Heatmap of per-perturbation mean shift on each PC of X_pca_g (rows clustered, PCs in order; single perturbations, train split).

Effect d[p, k] = (mean_k(perturbed p) - mean_k(control)) / sd_k(control), all from train-split cells.
Rows (perturbations) are ordered by Ward hierarchical clustering on d; columns are PC1..PC50 in order.
Row labels are colored by the X_pca_g effect-strength tertile in results/week2_effect_strength.csv.
Colors follow research-logbook/theme (v1.2.0): signed map berry (-) -> white -> blue (+).
"""
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

ROOT = Path(__file__).resolve().parents[1]
THEME = Path.home() / "research-logbook/theme"
plt.style.use(THEME / "inference.mplstyle")
tok = json.loads((THEME / "tokens.json").read_text())["plot_palette"]
SIGNED = LinearSegmentedColormap.from_list("berry_white_blue", [tok["berry"], "#FFFFFF", tok["blue"]])

obs = ad.read_h5ad(ROOT / "data/processed/norman2019.h5ad", backed="r").obs
emb = np.load(ROOT / "data/processed/state_embeddings.npz")
assert (emb["obs_names"] == obs.index.to_numpy()).all()
X = emb["X_pca_g"]

tr = (obs["split"] == "train").to_numpy()
cond = obs["condition"].astype(str).to_numpy()
ctrl = tr & (cond == "control")
mu0, sd0 = X[ctrl].mean(0), X[ctrl].std(0)
perts = sorted(set(cond[tr & (obs["n_targets"].to_numpy() == 1)]))
D = np.stack([(X[tr & (cond == p)].mean(0) - mu0) / sd0 for p in perts])
pcs = [f"PC{k + 1}" for k in range(X.shape[1])]

Lr = linkage(D, "ward")
ro, co = leaves_list(Lr), np.arange(D.shape[1])
import pandas as pd
es = pd.read_csv(ROOT / "results/week2_effect_strength.csv").query("space == 'X_pca_g'").set_index("gene")["tier"]
TIER_COLOR = {"strong": tok["berry"], "medium": "#171717", "weak": tok["gray"]}
np.savez(ROOT / "results/pc_shift_train.npz", D=D, perts=np.array(perts), pcs=np.array(pcs), row_order=ro, col_order=co)

vmax = float(np.quantile(np.abs(D), 0.99))
fig = plt.figure(figsize=(13, 17))
gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 12, 0.3], wspace=0.02)
axl, ax, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
kw = dict(color_threshold=0, above_threshold_color=tok["gray"], no_labels=True)
dendrogram(Lr, ax=axl, orientation="left", **kw)
axl.invert_yaxis()
axl.axis("off")
im = ax.imshow(D[ro][:, co], aspect="auto", cmap=SIGNED, vmin=-vmax, vmax=vmax, interpolation="nearest")
ax.set_xticks(range(len(co)), [pcs[i] for i in co], rotation=90, fontsize=7)
ax.set_yticks(range(len(ro)), [perts[i] for i in ro], fontsize=6.5)
ax.yaxis.tick_right()
for t in ax.get_yticklabels():
    tier = es[t.get_text()]
    t.set_color(TIER_COLOR[tier])
    t.set_fontweight("bold" if tier == "strong" else "normal")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], ls="none", marker="s", color=c, label=f"{k} ({(es == k).sum()})") for k, c in TIER_COLOR.items()],
          title="effect strength tertile", loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, fontsize=10, title_fontsize=10)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, cax=cax, extend="both")
cb.set_label("mean shift vs control (control SD units)\n+ blue: perturbed > control   − berry: perturbed < control")
cb.outline.set_visible(False)
cax.yaxis.set_label_position("left")
fig.subplots_adjust(right=0.86)
# leave room for right-side perturbation labels: move colorbar further right
pos = cax.get_position()
cax.set_position([0.93, pos.y0, 0.012, pos.height])

out = ROOT / "figures/pc_shift_rowcluster_X_pca_g_train"
fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{out}.pdf", bbox_inches="tight")
print(f"saved {out}.png  perts={len(perts)} pcs={len(pcs)} vmax={vmax:.2f} n_ctrl={ctrl.sum()}")
