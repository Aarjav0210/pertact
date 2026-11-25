"""Angle-length split: compare action directions under the same length policies, with a global length scan.

Every action is written f(a) = c · ell_a · u_a, ||u_a|| = 1. Directions u:
    mean_shift        u = m_a / ||m_a||, m_a the mean OT displacement on training pairs
    infonce           direction of the free InfoNCE action (seeds 0-2, saved)
    infonce_dir_only  InfoNCE trained on directions only, length held at ||m_a|| (seeds 0-2, random start)
    shuffled          mean-shift directions permuted across genes (floor)
Length policies ell_a:
    ms_len            ||m_a||, the mean-shift length
    ls_len            u_a · m_a, the least-squares length on the line u_a (training pairs); = ms_len for mean_shift
c scans one global factor. Retrieval MRR rewards overshoot (c > 1), so directions are compared as curves over c, at
c = 1 and at each curve's best c (best c is picked on validation, the same optimism for every direction).

    python -I scripts/angle_length_split.py      # -> results/angle_length/{scan.csv, dirs.csv}
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from align_uniform_probe import direction_uniformity, geometry, light_adata
from pertact.actions import load_actions, mean_shift_actions
from pertact.config import get_logger, load_config, path, read_json
from pertact.encoders import scale_D0
from pertact.infonce import train_actions
from pertact.ot import load_pairs
from pertact.retrieval import identification, instance_retrieval

torch.set_num_threads(2)
log = get_logger("angle_length")
CS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
SEEDS = [0, 1, 2]


def unit(v):
    return v / np.linalg.norm(v)


def main():
    cfg = load_config(None)
    adata, splits = light_adata(cfg)
    genes = sorted(splits["train_singles"])
    taus = read_json(path(cfg, "results_dir") / "infonce" / "best_tau.json")["tau"]
    out = path(cfg, "results_dir") / "angle_length"
    out.mkdir(parents=True, exist_ok=True)
    rows, drows = [], []
    for space in cfg["ot"]["spaces"]:
        S = adata.obsm[space]
        D0 = scale_D0(S[(adata.obs["split"] == "train").to_numpy()], cfg["seed"])
        t = taus[space]
        val = load_pairs(cfg, space, "val")
        ms = mean_shift_actions(cfg, adata, space)
        ms_len = np.array([np.linalg.norm(ms[g]) for g in genes])
        perm = np.random.default_rng(cfg["seed"]).permutation(len(genes))
        dirs = {("mean_shift", 0): {g: unit(ms[g]) for g in genes},
                ("shuffled", 0): {g: unit(ms[genes[perm[i]]]) for i, g in enumerate(genes)}}
        for s in SEEDS:
            nce = load_actions(cfg, space, s)
            dirs[("infonce", s)] = {g: unit(nce[g]) for g in genes}
            F, g_, _, info = train_actions(cfg, adata, splits, space, s, t, verbose=True, length=ms_len)
            assert list(g_) == genes
            dirs[("infonce_dir_only", s)] = {g: unit(f) for g, f in zip(genes, F)}
            np.savez(out / f"infonce_dir_only__{space}__s{s}.npz", emb=F, genes=np.array(genes))
        for (name, s), U in dirs.items():
            cos = np.array([U[g] @ unit(ms[g]) for g in genes])
            drows.append({"space": space, "dir": name, "seed": s, "dir_uniformity": direction_uniformity(U, genes),
                          "cos_to_ms_median": float(np.median(cos)), "cos_to_ms_q10": float(np.quantile(cos, 0.1))})
            pols = {"ms_len": dict(zip(genes, ms_len))}
            if name not in ("mean_shift", "shuffled"):
                pols["ls_len"] = {g: max(float(U[g] @ ms[g]), 0.0) for g in genes}
            for pol, ell in pols.items():
                for c in CS:
                    acts = {g: (c * ell[g] * U[g]).astype(np.float32) for g in genes}
                    t0 = time.time()
                    r = instance_retrieval(cfg, adata, space, "val", acts)
                    ident, _ = identification(cfg, adata, space, "val", acts, genes)
                    rows.append({"space": space, "dir": name, "seed": s, "len": pol, "c": c,
                                 "len_ratio_median": float(np.median([c * ell[g] / ms_len[i]
                                                                      for i, g in enumerate(genes)])),
                                 **geometry(S, val, acts, D0, t), "mrr": r["mrr"],
                                 "same_pert@10": r["same_pert@10"], "ident_top1": ident["top1"]})
                    log.info("%s %s s%d %s c=%.2f: MRR %.3f top1 %.3f align %.3f (%.0fs)", space, name, s, pol, c,
                             rows[-1]["mrr"], rows[-1]["ident_top1"], rows[-1]["alignment"], time.time() - t0)
                pd.DataFrame(rows).to_csv(out / "scan.csv", index=False)
        pd.DataFrame(drows).to_csv(out / "dirs.csv", index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
