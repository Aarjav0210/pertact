"""Closed-form whitened mean shift vs cosine InfoNCE: identification on validation singles only.

Directions (all from training cells only):
  mean_shift        m_a (OT mean shift, the Week 2 reference)
  white_sd          m_a / sd      (per-dim sd of training controls, gem-centred)
  white_diag        m_a / var     (diagonal LDA)
  white_full        Sigma^-1 m_a  (LDA; Sigma = covariance of gem-centred training controls)
  white_full_shr    (Sigma + lam I)^-1 m_a, lam = mean eigenvalue (shrinkage LDA)
Each is scored by direction (identify_cos) and, at the mean-shift length, by distance (identify).
Also landing cosine with the validation mean shift (direction quality for prediction).
No test cell is read: identification() is called with split="val" only.
"""
import sys
import numpy as np
import pandas as pd

from pertact.actions import effect_tiers, load_actions, mean_shift_actions
from pertact.config import load_config
from pertact.data import CONTROL, cell_labels
from pertact.encoders import load_all, states
from pertact.retrieval import identification, instance_retrieval

cfg = load_config(None)
adata, splits = load_all(cfg)
genes = sorted(splits["train_singles"])
cond, gem, cell_split = cell_labels(adata.obs, cfg["data"]["batch_key"])
unit = lambda v: v / np.linalg.norm(v)
rows = []
for space in cfg["ot"]["spaces"]:
    S = states(adata, space)
    ms = mean_shift_actions(cfg, adata, space)
    _, tier = effect_tiers(adata, space, ms)
    ctrl = (cell_split == "train") & (cond == CONTROL)
    Xc = S[ctrl].astype(np.float64)
    gc = gem[ctrl]
    for g in np.unique(gc):  # remove gem-group means so batch offsets do not count as noise
        Xc[gc == g] -= Xc[gc == g].mean(0)
    C = np.cov(Xc.T)
    ev = np.linalg.eigvalsh(C)
    print(space, "control cov eigenvalues: max %.3g min %.3g cond %.3g" % (ev.max(), ev.min(), ev.max() / ev.min()), flush=True)
    sd, var = np.sqrt(np.diag(C)), np.diag(C)
    Ci = np.linalg.inv(C)
    Cs = np.linalg.inv(C + ev.mean() * np.eye(len(C)))
    dirs = {
        "mean_shift": {g: unit(ms[g]) for g in genes},
        "white_sd": {g: unit(ms[g] / sd) for g in genes},
        "white_diag": {g: unit(ms[g] / var) for g in genes},
        "white_full": {g: unit(Ci @ ms[g]) for g in genes},
        "white_full_shr": {g: unit(Cs @ ms[g]) for g in genes},
    }
    for s in cfg["eval"]["seeds"]:
        a = load_actions(cfg, space, s, "infonce_cos")
        dirs[f"infonce_cos_s{s}"] = {g: unit(a[g]) for g in genes}
    cos_u = np.stack([dirs[f"infonce_cos_s0"][g] for g in genes])
    for name, d in dirs.items():
        at_ms_len = {g: d[g] * np.linalg.norm(ms[g]) for g in genes}
        r = {"space": space, "set": name,
             "cos_to_raw_shift": float(np.median([d[g] @ unit(ms[g]) for g in genes])),
             "cos_to_infonce_cos_s0": float(np.median([d[g] @ cos_u[i] for i, g in enumerate(genes)]))}
        for start in ("ctrl_mean", "pooled_ot"):
            m, per = identification(cfg, adata, space, "val", d, genes, start, score="cosine")
            r[f"cos_{start}_top1"] = m["top1"]
            per["tier"] = per["cond"].map(tier)
            for t, g_ in per.groupby("tier"):
                r[f"cos_{start}_top1_{t}"] = float((g_["rank"] <= 1).mean())
            m, _ = identification(cfg, adata, space, "val", at_ms_len, genes, start)
            r[f"dist_{start}_top1_mslen"] = m["top1"]
        ir = instance_retrieval(cfg, adata, space, "val", at_ms_len)
        r["instance_mrr_mslen"] = ir["mrr"]
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(sys.argv[1], index=False)
