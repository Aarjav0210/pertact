"""Sets of action vectors: one vector f(a) per gene, living in a frozen state space.

Storage (trained sets are saved to embeddings/<name>__<space>__s<seed>.npz), the closed-form reference (the mean
shift), the two rescalings fitted after training, and length comparisons. Training is in infonce.py; the tests of a
set of vectors are in retrieval.py. Everything here uses training pairs only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import path, write_json
from .encoders import states
from .ot import load_pairs, start_states


# --------------------------------------------------------------------------- storage
def actions_path(cfg, space, seed, name="infonce"):
    """File of one saved set: name is "infonce" or "infonce_anchor"."""
    return path(cfg, "embeddings_dir") / f"{name}__{space}__s{seed}.npz"


def load_actions(cfg, space, seed, name="infonce") -> dict[str, np.ndarray]:
    """One saved set of action vectors as {gene: vector}."""
    z = np.load(actions_path(cfg, space, seed, name))
    return dict(zip(z["genes"].tolist(), z["emb"]))


def save_actions(cfg, space, seed, F, genes, info, hist, tau, name="infonce"):
    """Write one trained set of action vectors, its training history and its settings."""
    np.savez(actions_path(cfg, space, seed, name), emb=F.astype(np.float32), genes=np.array(genes, dtype=str), missing=np.array([], dtype=str))
    d = path(cfg, "results_dir") / "infonce"
    d.mkdir(parents=True, exist_ok=True)
    tag = "" if name == "infonce" else f"{name}__"
    hist.to_csv(d / f"history__{tag}{space}__s{seed}.csv", index=False)
    write_json(info | {"tau": tau, "space": space, "seed": seed}, d / f"info__{tag}{space}__s{seed}.json")


# --------------------------------------------------------------------------- the reference and the rescalings
def mean_shift_actions(cfg, adata, space) -> dict[str, np.ndarray]:
    """f(a) = mean over training OT pairs of s(x_pert) - Σ_k w_k s(x_ctrl_k)."""
    S = states(adata, space)
    z = load_pairs(cfg, space, "train")
    disp = S[z["cell"]] - start_states(S, z)
    df = pd.DataFrame(disp)
    df["cond"] = z["cond"]
    m = df.groupby("cond").mean()
    return {g: m.loc[g].to_numpy(dtype=np.float32) for g in m.index}


def calibrate_scale(cfg, adata, space, actions) -> float:
    """One scalar alpha for a whole action set, by least squares on training OT pairs:
    min_alpha sum_j || start_j + alpha f(a_j) - s(x_j) ||².  InfoNCE only constrains relative distances, so its
    actions can be right in direction but wrong in length; this fixes the length without touching directions."""
    S = states(adata, space)
    z = load_pairs(cfg, space, "train")
    F = np.stack([actions[c] for c in z["cond"]])
    return float((F * (S[z["cell"]] - start_states(S, z))).sum() / (F * F).sum())


def per_action_scale(cfg, adata, space, actions) -> dict[str, float]:
    """One scalar c_a per action, by least squares on that perturbation's training OT pairs:
    min_c sum_{j in a} || start_j + c f(a) - s(x_j) ||²  =>  c_a = f(a)·mean_j(s(x_j) - start_j) / ||f(a)||².
    The solution is the projection of the mean OT shift onto f(a): direction kept, length = ||mean shift|| * cos."""
    S = states(adata, space)
    z = load_pairs(cfg, space, "train")
    disp = pd.DataFrame(S[z["cell"]] - start_states(S, z))
    disp["cond"] = z["cond"]
    tot = disp.groupby("cond").sum()
    n = disp.groupby("cond").size()
    return {g: float(actions[g] @ tot.loc[g].to_numpy() / (n[g] * actions[g] @ actions[g])) for g in tot.index if g in actions}


# --------------------------------------------------------------------------- length against the mean shift
def length_ratio(cfg, adata, space, F, genes) -> np.ndarray:
    """||f(a)|| / ||mean OT shift on training pairs||, one value per gene."""
    ms = mean_shift_actions(cfg, adata, space)
    return np.array([np.linalg.norm(f) / np.linalg.norm(ms[g]) for g, f in zip(genes, F)])


def length_table(sets, ms, tier, space) -> pd.DataFrame:
    """Per perturbation: ||f(a)|| / ||mean OT shift|| and cos(f(a), mean shift) for every action set."""
    rows = []
    for (name, seed), acts in sets.items():
        for g, m in ms.items():
            if g not in acts:
                continue
            f = acts[g]
            nf, nm = float(np.linalg.norm(f)), float(np.linalg.norm(m))
            rows.append({"space": space, "actions": name, "seed": seed, "gene": g, "tier": tier[g],
                         "len_ratio": nf / nm, "cos_mean_shift": float(f @ m / (nf * nm)) if nf > 0 else np.nan})
    return pd.DataFrame(rows)
