"""Frozen cell-state encoders g, fitted on training cells only and applied to every cell.

    python -I -m pertact.encoders pca      # obsm["X_pca_g"] (50-d), models/pca_g.npz
    python -I -m pertact.encoders scvi     # obsm["X_scvi10"], obsm["X_scvi30"], models/scvi10/, models/scvi30/
    python -I -m pertact.encoders umap     # figures/umap_<key>.png sanity checks

State embeddings are written to data/processed/state_embeddings.npz (keyed by obsm name, rows in
processed-h5ad order) and also into the processed h5ad's obsm.
"""
from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import scipy.sparse as sp

from .config import get_logger, load_config, path, seed_everything
from .data import CONTROL

log = get_logger("encoders")


def train_mask(adata: ad.AnnData) -> np.ndarray:
    return (adata.obs["split"] == "train").to_numpy()


def state_store(cfg):
    return path(cfg, "processed_h5ad").with_name("state_embeddings.npz")


def save_state(cfg, adata, keys):
    p = state_store(cfg)
    old = dict(np.load(p, allow_pickle=False)) if p.exists() else {}
    if "obs_names" in old:
        assert (old["obs_names"] == adata.obs_names.to_numpy().astype(str)).all()
    old.update({k: adata.obsm[k].astype(np.float32) for k in keys})
    old["obs_names"] = adata.obs_names.to_numpy().astype(str)
    np.savez(p, **old)
    log.info("saved %s -> %s", keys, p)


def attach_state(cfg, adata) -> ad.AnnData:
    """Load stored state embeddings into adata.obsm (checks row order)."""
    st = np.load(state_store(cfg), allow_pickle=False)
    assert (st["obs_names"] == adata.obs_names.to_numpy().astype(str)).all()
    for k in st.files:
        if k != "obs_names":
            adata.obsm[k] = st[k]
    return adata


# --------------------------------------------------------------------------- PCA
def fit_pca(cfg, adata):
    from sklearn.decomposition import PCA

    n = cfg["encoders"]["pca"]["n_comps"]
    tr = train_mask(adata)
    X = adata.X
    Xtr = X[tr].toarray() if sp.issparse(X) else np.asarray(X[tr])
    pca = PCA(n_components=n, svd_solver="randomized", random_state=cfg["seed"]).fit(Xtr)
    mean = pca.mean_.astype(np.float32)
    comps = pca.components_.astype(np.float32)
    Z = np.vstack([((X[i:i + 20000].toarray() if sp.issparse(X) else X[i:i + 20000]) - mean) @ comps.T
                   for i in range(0, adata.n_obs, 20000)])
    adata.obsm["X_pca_g"] = Z.astype(np.float32)
    out = path(cfg, "models_dir") / "pca_g.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mean=mean, components=comps, explained_variance_ratio=pca.explained_variance_ratio_,
             var_names=adata.var_names.to_numpy().astype(str))
    log.info("PCA fit on %d train cells; explained variance (50 PCs) = %.3f", tr.sum(),
             pca.explained_variance_ratio_.sum())
    return adata


# --------------------------------------------------------------------------- scVI
def fit_scvi(cfg, adata, latent_dims=None):
    import scvi
    import torch

    c = cfg["encoders"]["scvi"]
    bkey = cfg["data"].get("batch_key")
    tr = train_mask(adata)
    torch.set_float32_matmul_precision("high")
    for d in latent_dims or c["latent_dims"]:
        seed_everything(cfg["seed"])
        scvi.settings.seed = cfg["seed"]
        a_tr = ad.AnnData(X=adata.layers["counts"][tr].copy(), obs=adata.obs.loc[tr, [bkey]].copy(),
                          var=adata.var[[]].copy())
        a_tr.layers["counts"] = a_tr.X
        scvi.model.SCVI.setup_anndata(a_tr, layer="counts", batch_key=bkey)
        m = scvi.model.SCVI(a_tr, n_latent=d, n_hidden=c["n_hidden"], n_layers=c["n_layers"],
                            gene_likelihood=c["gene_likelihood"])
        m.train(max_epochs=c["max_epochs"], early_stopping=c["early_stopping"], batch_size=c["batch_size"],
                accelerator=c.get("accelerator", "auto"), enable_progress_bar=False)
        hist = m.history
        n_ep = len(hist["elbo_train"])
        log.info("scVI-%d: %d epochs, final train ELBO %.1f, val ELBO %.1f", d, n_ep,
                 float(hist["elbo_train"].iloc[-1, 0]),
                 float(hist["elbo_validation"].iloc[-1, 0]) if "elbo_validation" in hist else float("nan"))
        a_all = ad.AnnData(X=adata.layers["counts"].copy(), obs=adata.obs[[bkey]].copy(), var=adata.var[[]].copy())
        a_all.layers["counts"] = a_all.X
        scvi.model.SCVI.setup_anndata(a_all, layer="counts", batch_key=bkey)
        assert list(a_all.obs[bkey].cat.categories) == list(a_tr.obs[bkey].cat.categories)
        adata.obsm[f"X_scvi{d}"] = m.get_latent_representation(a_all).astype(np.float32)  # posterior means
        out = path(cfg, "models_dir") / f"scvi{d}"
        m.save(str(out), overwrite=True, save_anndata=False)
        hist_df = {k: v.iloc[:, 0].to_numpy() for k, v in hist.items() if hasattr(v, "iloc")}
        np.savez(out / "history.npz", **hist_df)
    return adata


# --------------------------------------------------------------------------- UMAP sanity check
def umap_figures(cfg, adata, keys):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    u = cfg["encoders"]["umap"]
    rng = np.random.default_rng(cfg["seed"])
    cond = adata.obs["condition"].astype(str)
    hl = [g for g in u["highlight"] if g in set(cond)]
    # subsample, but keep every cell of the highlighted perturbations
    must = np.flatnonzero(cond.isin(hl).to_numpy())
    rest = np.setdiff1d(np.arange(adata.n_obs), must)
    idx = np.sort(np.concatenate([must, rng.choice(rest, size=min(len(rest), u["n_subsample"]), replace=False)]))
    figdir = path(cfg, "figures_dir")
    figdir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        b = ad.AnnData(obs=adata.obs.iloc[idx][["condition", "n_targets", "gemgroup", "split"]].copy())
        # umap-learn directly: scanpy 1.11's neighbors() calls csr_array.getnnz, removed in scipy 1.17
        E = umap.UMAP(n_neighbors=u["n_neighbors"], random_state=cfg["seed"]).fit_transform(adata.obsm[key][idx])
        c = b.obs["condition"].astype(str).to_numpy()
        nt = b.obs["n_targets"].to_numpy()
        fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
        for lab, m, col in (("double", nt == 2, "#c9c9c9"), ("single", nt == 1, "#8fb3d9"), ("control", c == CONTROL, "#d62728")):
            ax[0].scatter(E[m, 0], E[m, 1], s=1, c=col, label=f"{lab} (n={m.sum()})", rasterized=True)
        ax[0].set_title(f"{key}: control vs perturbed")
        ax[0].legend(markerscale=8, frameon=False, loc="best")
        ax[1].scatter(E[:, 0], E[:, 1], s=1, c="#dddddd", rasterized=True)
        cols = plt.cm.tab10(np.arange(len(hl)))
        for g, col in zip(hl, cols):
            m = c == g
            ax[1].scatter(E[m, 0], E[m, 1], s=3, color=col, label=f"{g} (n={m.sum()})", rasterized=True)
        m = c == CONTROL
        ax[1].scatter(E[m, 0], E[m, 1], s=1, c="black", alpha=0.25, label="control", rasterized=True)
        ax[1].set_title(f"{key}: strong single perturbations")
        ax[1].legend(markerscale=6, frameon=False, loc="best")
        gg = b.obs["gemgroup"].astype(str).to_numpy()
        for i, g in enumerate(sorted(set(gg))):
            m = gg == g
            ax[2].scatter(E[m, 0], E[m, 1], s=1, color=plt.cm.tab10(i), label=f"gem {g}", rasterized=True)
        ax[2].set_title(f"{key}: gem group (batch)")
        ax[2].legend(markerscale=8, frameon=False, loc="best", ncol=2)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.tight_layout()
        fig.savefig(figdir / f"umap_{key}.png", dpi=130)
        plt.close(fig)
        log.info("figure %s", figdir / f"umap_{key}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pca", "scvi", "umap", "write_h5ad"])
    ap.add_argument("--dims", type=int, nargs="*", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed_everything(cfg["seed"])
    adata = ad.read_h5ad(path(cfg, "processed_h5ad"))
    if a.what == "pca":
        save_state(cfg, fit_pca(cfg, adata), ["X_pca_g"])
    elif a.what == "scvi":
        dims = a.dims or cfg["encoders"]["scvi"]["latent_dims"]
        save_state(cfg, fit_scvi(cfg, adata, dims), [f"X_scvi{d}" for d in dims])
    elif a.what == "umap":
        attach_state(cfg, adata)
        umap_figures(cfg, adata, [k for k in ("X_pca_g", "X_scvi10", "X_scvi30") if k in adata.obsm])
    else:  # copy the stored state embeddings into the processed h5ad
        attach_state(cfg, adata)
        adata.write_h5ad(path(cfg, "processed_h5ad"), compression="gzip")
        log.info("obsm now: %s", list(adata.obsm))


if __name__ == "__main__":
    main()
