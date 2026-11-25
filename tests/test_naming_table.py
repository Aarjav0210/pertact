"""The naming table on a small planted data set: other starts, sets of cells, and a projection of the displacement.

The toy state space has two directions in which cells differ a lot for reasons unrelated to the perturbation, and six
narrow directions that carry the gene effects. Compared as they are, displacements are dominated by the two wide
directions; a projection that shrinks them makes the genes easy to name.
"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from pertact import infonce, naming
from pertact.actions import (TABLE_RUNS, displacements, load_projection, mean_shift_actions, table_name, table_variants,
                             whitening)
from pertact.infonce import PairData, cosine_infonce_loss, train_actions

SPACE, GENES, D = "X_toy", [f"G{i}" for i in range(6)], 8


@pytest.fixture()
def toy(tmp_path):
    """A config, an AnnData and saved pairs for 6 genes in 2 gem groups, train and validation cells."""
    rng = np.random.default_rng(0)
    spread = np.array([6.0, 6.0] + [1.0] * (D - 2))
    shifts = np.zeros((len(GENES), D))
    shifts[np.arange(len(GENES)), 2 + np.arange(len(GENES))] = 2.0          # each gene moves one narrow direction
    obs, S = [], []
    for split, n_ctrl, n_pert in (("train", 300, 80), ("val", 150, 40)):
        for gem in ("1", "2"):
            offset = 3.0 if gem == "2" else 0.0
            S.append(rng.normal(size=(n_ctrl // 2, D)) * spread + offset)
            obs += [(split, "control", gem)] * (n_ctrl // 2)
            for g, sh in zip(GENES, shifts):
                S.append(rng.normal(size=(n_pert // 2, D)) * spread + offset + sh)
                obs += [(split, g, gem)] * (n_pert // 2)
    obs = pd.DataFrame(obs, columns=["split", "condition", "gemgroup"])
    obs.index = obs.index.astype(str)
    S = np.concatenate(S).astype(np.float32)
    adata = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs, obsm={SPACE: S})
    cfg = {"seed": 0, "paths": {"results_dir": str(tmp_path / "results"), "embeddings_dir": str(tmp_path / "embeddings")},
           "data": {"batch_key": "gemgroup"}, "ot": {"spaces": [SPACE]}, "eval": {"seeds": [0]},
           "infonce": {"P": 4, "K": 4, "lr": 0.05, "max_steps": 300, "eval_every": 50, "patience": 4, "n_val_batches": 4,
                       "cosine": {"tau_init": 0.07, "tau_min": 0.01, "learn_tau": True}, "set_size": 8,
                       "projection": {"lr": 0.005}, "naming_sets_per_gene": 5}}
    (tmp_path / "results" / "ot").mkdir(parents=True)
    (tmp_path / "embeddings").mkdir()
    for split in ("train", "val"):
        pert = np.flatnonzero((obs.split == split) & (obs.condition != "control"))
        for tag in ("", "__pooled"):   # 3 partner controls from the cell's own split and gem group
            partners = np.stack([rng.choice(np.flatnonzero((obs.split == split) & (obs.condition == "control") & (obs.gemgroup == obs.gemgroup.iloc[i])), 3, replace=False) for i in pert])
            np.savez(tmp_path / "results" / "ot" / f"{SPACE}__{split}{tag}.npz", cell=pert, cond=obs.condition.to_numpy()[pert].astype(str),
                     partners=partners, weights=np.tile([0.5, 0.3, 0.2], (len(pert), 1)))
    return cfg, adata, {"train_singles": GENES}


def test_names_of_the_table_runs():
    cfg = {"infonce": {"set_size": 64}}
    assert table_name(cfg, "ot", 1, None) == "infonce_cos"
    assert table_name(cfg, "ctrl_mean", "set", "train") == "infonce_cos_cm_set64_proj"
    assert table_name(cfg, "ot", 1, "white") == "infonce_cos_white"
    assert len(set(table_variants(cfg))) == len(TABLE_RUNS) + 1


def test_identity_projection_leaves_the_cosine_loss_unchanged():
    torch.manual_seed(0)
    F, ctrl, pert, a = torch.randn(3, 5), torch.randn(6, 5), torch.randn(6, 5), torch.tensor([0, 0, 1, 1, 2, 2])
    plain = cosine_infonce_loss(F, ctrl, pert, a, 0.1)
    assert torch.isclose(plain, cosine_infonce_loss(F, ctrl, pert, a, 0.1, proj=torch.eye(5)))
    assert not torch.isclose(plain, cosine_infonce_loss(F, ctrl, pert, a, 0.1, proj=torch.diag(torch.tensor([5.0, 1, 1, 1, 0.2]))))


def test_batches_single_cells_as_before_and_sets_and_gem_mean_starts(toy):
    cfg, adata, _ = toy
    S = adata.obsm[SPACE]
    data = PairData(cfg, S, SPACE, "train", GENES)
    start, pert, a = data.batch(np.random.default_rng(3), 4, 4)
    # single cells from OT partners: the same draws as before sets and other starts were added
    rng = np.random.default_rng(3)
    perts = rng.choice(len(GENES), size=4, replace=False)
    rows = np.concatenate([rng.choice(data.by_pert[p], size=4, replace=False) for p in perts])
    pick = (rng.random(len(rows))[:, None] > data.cum[rows]).sum(1).clip(max=2)
    assert torch.equal(start, data.S[torch.as_tensor(data.partners[rows, pick])]) and torch.equal(pert, data.S[torch.as_tensor(data.cell[rows])])
    assert a.tolist() == np.repeat(perts, 4).tolist()
    # gem-mean start: every start is the mean training control of the cell's gem group
    cm = PairData(cfg, S, SPACE, "train", GENES, adata)
    obs = adata.obs
    mu = {g: S[((obs.split == "train") & (obs.condition == "control") & (obs.gemgroup == g)).to_numpy()].mean(0) for g in ("1", "2")}
    assert np.allclose(cm.ctrl_mean.numpy(), np.stack([mu[g] for g in obs.gemgroup.to_numpy()[cm.cell]]), atol=1e-5)
    # sets: 16 examples, each the average of 8 cells of one gene, so their wide directions vary much less
    s1, p1, a1 = cm.batch(np.random.default_rng(1), 4, 4)
    s8, p8, a8 = cm.batch(np.random.default_rng(1), 4, 4, set_size=8)
    assert p8.shape == (16, D) and s8.shape == (16, D) and len(a8) == 16 and (a8.view(4, 4) == a8.view(4, 4)[:, :1]).all()
    assert p8[:, 0].std() < 0.7 * p1[:, 0].std()


def test_fixed_vectors_need_the_projection_and_trained_vectors_mostly_do_not(toy):
    cfg, adata, splits = toy
    Xva, yva = displacements(cfg, adata, SPACE, "val", "ctrl_mean")
    top1 = lambda F, proj=None: naming.rank_summary(naming.name_scores(Xva, F, proj), GENES, yva)["top1"]
    W = whitening(cfg, adata, SPACE, "ctrl_mean")
    assert np.abs(W[:, :2]).max() < 0.5 * np.abs(W[:, 2:]).max()           # the computed projection shrinks the wide directions
    ms = mean_shift_actions(cfg, adata, SPACE)
    Fms = np.stack([ms[g] for g in GENES])
    fixed, fixed_white = top1(Fms), top1(Fms @ W.T, W)
    assert fixed_white > fixed + 0.15                                      # a fixed mean shift gains a lot from it
    trained = {}
    for projection in (None, "white", "train"):
        F, genes, hist, info = train_actions(cfg, adata, splits, SPACE, 0, 0.07, verbose=False, score="cosine",
                                             start="ctrl_mean", projection=projection)
        assert genes == GENES and (projection is None) == ("proj" not in info)
        trained[projection] = top1(F, info.get("proj"))
    # trained vectors can point away from the wide directions by themselves: most of the gain is there without a
    # projection, and a projection changes little
    assert trained[None] > fixed + 0.1, (fixed, trained)
    assert min(trained["white"], trained["train"]) > trained[None] - 0.05 and max(trained.values()) < fixed_white + 0.1, (fixed_white, trained)


def test_the_whole_table_trains_and_is_scored(toy, monkeypatch):
    cfg, adata, splits = toy
    cfg["infonce"]["max_steps"] = 100
    monkeypatch.setattr(infonce, "load_all", lambda cfg: (adata, splits))
    monkeypatch.setattr(naming, "load_all", lambda cfg: (adata, splits))
    infonce.train_cosine(cfg)
    infonce.train_table(cfg)
    assert load_projection(cfg, SPACE, 0, "infonce_cos") is None
    assert load_projection(cfg, SPACE, 0, "infonce_cos_proj").shape == (D, D)
    assert load_projection(cfg, SPACE, 0, "infonce_cos_white").shape[1] == D
    naming.run(cfg)
    df = pd.read_csv(naming.path(cfg, "results_dir") / "naming_table" / "naming.csv")
    assert set(df.model) == {"mean shift", "mean shift, whitened", "centroid", "lda", "logistic"} | set(table_variants(cfg))
    assert set(df.input) == {"one cell", "set of 8"} and set(df.start) == set(naming.STARTS)
    assert len(df) == len(set(df.model)) * 4 and df.top1.between(0, 1).all() and (df.n_candidates == len(GENES)).all()
    one = df[(df.input == "one cell") & (df.start == "ctrl_mean")].set_index("model").top1
    sets = df[(df.input == "set of 8") & (df.start == "ctrl_mean")].set_index("model").top1
    assert sets["lda"] > one["lda"] and one["lda"] > one["mean shift"]       # averaging cells and whitening both help here
    assert one["mean shift, whitened"] > one["mean shift"] + 0.15
