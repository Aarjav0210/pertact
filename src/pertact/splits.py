"""Fixed train / validation / test split.

train : 80% of the cells of every single perturbation + a share of controls
val   : the other 20% of every single perturbation (held out per perturbation) + a share of controls
test  : every double whose two genes are both training singles + a share of controls
Doubles with an unseen component are listed separately and never used for fitting or tuning.

    python -I -m pertact.splits
"""
from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd

from .config import get_logger, load_config, path, write_json
from .data import CONTROL, condition_genes

log = get_logger("splits")


def make_splits(obs: pd.DataFrame, cfg: dict) -> dict:
    """Assign every cell to train, val, test or excluded. 20% of each single's cells go to val; controls are split
    within each gem group; doubles whose two genes are both singles go to test. Asserts that the result is a
    partition and that no double cell is in train or val."""
    s = cfg["splits"]
    rng = np.random.default_rng(cfg["seed"])
    cond = obs["condition"].astype(str)
    nt = obs["n_targets"]
    singles = sorted(cond[nt == 1].unique())
    doubles = sorted(cond[nt == 2].unique())
    cells = {"train": [], "val": [], "test": []}

    for p in singles:
        idx = np.sort(obs.index[cond == p].to_numpy())
        idx = idx[rng.permutation(len(idx))]
        n_val = int(round(s["val_frac_per_single"] * len(idx)))
        cells["val"] += idx[:n_val].tolist()
        cells["train"] += idx[n_val:].tolist()

    fr = s["control_fracs"]
    assert abs(sum(fr.values()) - 1) < 1e-9
    ctrl = obs[cond == CONTROL]
    for _, grp in ctrl.groupby(cfg["data"]["batch_key"], observed=True):
        idx = np.sort(grp.index.to_numpy())
        idx = idx[rng.permutation(len(idx))]
        n_tr, n_va = int(round(fr["train"] * len(idx))), int(round(fr["val"] * len(idx)))
        cells["train"] += idx[:n_tr].tolist()
        cells["val"] += idx[n_tr:n_tr + n_va].tolist()
        cells["test"] += idx[n_tr + n_va:].tolist()

    train_singles = set(singles)
    test_doubles = [d for d in doubles if all(g in train_singles for g in condition_genes(d))]
    excluded = [d for d in doubles if d not in test_doubles]
    cells["test"] += obs.index[cond.isin(test_doubles)].tolist()
    excluded_cells = obs.index[cond.isin(excluded)].tolist()

    all_ids = sum(cells.values(), []) + excluded_cells
    assert len(all_ids) == len(set(all_ids)) == len(obs), "split is not a partition of the cells"
    dbl = set(obs.index[nt == 2])
    assert not dbl & set(cells["train"]) and not dbl & set(cells["val"]), "double cells leaked into train/val"

    def summary(ids):
        c = cond.loc[ids]
        return {"n_cells": len(ids), "n_control": int((c == CONTROL).sum()),
                "n_single_cells": int(nt.loc[ids].eq(1).sum()), "n_double_cells": int(nt.loc[ids].eq(2).sum()),
                "n_conditions": int(c[c != CONTROL].nunique())}

    return {
        "seed": cfg["seed"],
        "config": s,
        "train_singles": singles,
        "val_singles": singles,
        "test_doubles": test_doubles,
        "excluded_doubles": excluded,
        "summary": {k: summary(v) for k, v in cells.items()} | {"excluded": summary(excluded_cells)},
        "cells": {k: sorted(v) for k, v in cells.items()} | {"excluded_doubles": sorted(excluded_cells)},
    }


def main():
    """Build the split and write splits.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    obs = ad.read_h5ad(path(cfg, "qc_h5ad"), backed="r").obs.copy()
    sp = make_splits(obs, cfg)
    write_json(sp, path(cfg, "splits"))
    log.info("wrote %s: %s", path(cfg, "splits"), sp["summary"])
    log.info("train singles %d | test doubles %d | excluded doubles %d %s", len(sp["train_singles"]),
             len(sp["test_doubles"]), len(sp["excluded_doubles"]), sp["excluded_doubles"])


if __name__ == "__main__":
    main()
