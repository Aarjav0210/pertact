"""Week 2 addendum diagnostics for the decoder smoke test (training singles only; reads no test cells).

Writes
  results/week2_decoder/direction_spread.csv  how spread out the unit action directions are
  results/week2_decoder/length_overlap.csv    per embedding, as the decoder sees it after unit-mean-norm:
                                              weak/strong norm ratio, weak-third median norm, and how much
                                              each weak direction lies inside the strong vectors' top-5 subspace

Effect-strength thirds are the PCA-50 ones from results/week2_effect_strength.csv, for every embedding.
Run: .venv/bin/python -I scripts/week2_direction_diagnostics.py
"""
import numpy as np
import pandas as pd

from pertact.config import load_config, path
from pertact.decoder_test import SPACE_SHORT, action_embedding
from pertact.embeddings import load_embedding
from pertact.encoders import load_all

ACTIONS = ["infonce", "infonce_scaled_per_action", "mean_shift", "infonce_unit", "mean_shift_unit"]


def main():
    """Measure direction spread and length differences for every vector set, and write the two tables."""
    cfg = load_config(None)
    adata, splits = load_all(cfg)
    rd = path(cfg, "results_dir")
    tiers = pd.read_csv(rd / "week2_effect_strength.csv").query("space == 'X_pca_g'").set_index("gene")["tier"]
    train = set(splits["train_singles"])

    def embeddings():
        for m in ["onehot", "pca_loadings"]:
            g, M = load_embedding(cfg, m)
            keep = [i for i, x in enumerate(g) if x in train]
            yield m, "", 0, [g[i] for i in keep], M[keep]
        for sp in cfg["ot"]["spaces"]:
            for v in ACTIONS:
                for s in cfg["eval"]["seeds"]:
                    g, M = action_embedding(cfg, adata, splits, sp, s, v)
                    yield f"{v}_{SPACE_SHORT[sp]}", sp, s, g, M

    spread, overlap = [], []
    for method, space, seed, genes, M in embeddings():
        M = M / np.linalg.norm(M, axis=1).mean()  # what the decoder sees
        n = np.linalg.norm(M, axis=1)
        t = np.array([tiers[g] for g in genes])
        U = M / n[:, None]
        if method.endswith(("_unit_pca", "_unit_scvi")):
            C = U @ U.T
            off = C[~np.eye(len(genes), dtype=bool)]
            nn = np.sort(C - 2 * np.eye(len(genes)), axis=1)[:, -1]
            sv2 = np.linalg.svd(U, compute_uv=False) ** 2
            spread.append(dict(space=space, actions=method, seed=seed, median_pairwise_cos=np.median(off),
                               median_nearest_neighbour_cos=np.median(nn),
                               participation_ratio=sv2.sum() ** 2 / (sv2 ** 2).sum(),
                               norm_of_mean_direction=np.linalg.norm(U.mean(0))))
        S = M[t == "strong"]
        Uw, Us = U[t == "weak"], U[t == "strong"]
        _, _, Vt = np.linalg.svd(S, full_matrices=False)
        overlap.append(dict(method=method, seed=seed, dim=M.shape[1],
                            weak_strong_norm_ratio=np.median(n[t == "weak"]) / np.median(n[t == "strong"]),
                            weak_median_norm=np.median(n[t == "weak"]), norm_cv=n.std() / n.mean(),
                            weak_share_in_strong_top5=np.median(((Uw @ Vt[:5].T) ** 2).sum(1)),
                            weak_max_cos_to_strong=np.median((Uw @ Us.T).max(1))))
    out = rd / "week2_decoder"
    pd.DataFrame(spread).to_csv(out / "direction_spread.csv", index=False)
    ov = pd.DataFrame(overlap)
    ov.to_csv(out / "length_overlap.csv", index=False)
    print(pd.DataFrame(spread).groupby(["space", "actions"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string())
    print(ov.groupby("method").mean(numeric_only=True).drop(columns="seed").round(3).to_string())


if __name__ == "__main__":
    main()
