"""Sets of action vectors: one vector f(a) per gene, living in a frozen state space.

Storage (trained sets are saved to embeddings/<name>__<space>__s<seed>.npz), the closed-form reference (the mean
shift), the two rescalings fitted after training, and length comparisons. Training is in infonce.py; the tests of a
set of vectors are in retrieval.py. Everything here uses training pairs only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import path, write_json
from .encoders import scale_D0, states
from .ot import gem_mean_states, load_pairs, start_states

# The naming table (Week 5): the cosine loss with one thing changed at a time. Each run is (start, cells per training
# example, projection). start: "ot" = the cell's own OT partners, "ctrl_mean" = the mean control of its gem group.
# projection: None, "white" (computed from the training cells, not trained) or "train" (learned with the vectors).
# ("ot", 1, None) is the cosine run itself, stored as "infonce_cos"; it is read, not trained again.
TABLE_RUNS = [("ot", 1, "white"), ("ot", 1, "train"), ("ctrl_mean", 1, None), ("ctrl_mean", 1, "train"),
              ("ctrl_mean", "set", None), ("ctrl_mean", "set", "train")]


# --------------------------------------------------------------------------- storage
def actions_path(cfg, space, seed, name="infonce"):
    """File of one saved set: name is "infonce" or "infonce_anchor"."""
    return path(cfg, "embeddings_dir") / f"{name}__{space}__s{seed}.npz"


def soft_name(score, s) -> str:
    """Name under which the vectors trained with soft value s are stored: "infonce_soft<s>" for the distance score,
    "infonce_cos_soft<s>" for the cosine score."""
    return f"{'infonce_cos' if score == 'cosine' else 'infonce'}_soft{s:g}"


def soft_variants(cfg) -> list[str]:
    """Every stored set of the soft-positive experiment: the cosine vectors with one right answer, then each score at
    each value of the configured grid."""
    return ["infonce_cos"] + [soft_name(score, s) for score in ("cosine", "distance") for s in cfg["infonce"]["soft_grid"]]


def table_name(cfg, start, set_size, projection) -> str:
    """Name under which one run of the naming table is stored, for example "infonce_cos_cm_set64_proj": "_cm" for the
    gem-group mean start, "_set<n>" for training examples that average n cells, "_white" / "_proj" for the computed
    and the trained projection. `set_size` is a number or "set", which stands for config infonce.set_size."""
    n = cfg["infonce"]["set_size"] if set_size == "set" else set_size
    return ("infonce_cos" + ("_cm" if start == "ctrl_mean" else "") + (f"_set{n}" if n > 1 else "")
            + {None: "", "white": "_white", "train": "_proj"}[projection])


def table_variants(cfg) -> list[str]:
    """Every stored set of the naming table: the cosine vectors themselves, then each run of TABLE_RUNS."""
    return ["infonce_cos"] + [table_name(cfg, *run) for run in TABLE_RUNS]


def load_actions(cfg, space, seed, name="infonce") -> dict[str, np.ndarray]:
    """One saved set of action vectors as {gene: vector}."""
    z = np.load(actions_path(cfg, space, seed, name))
    return dict(zip(z["genes"].tolist(), z["emb"]))


def load_projection(cfg, space, seed, name):
    """The projection stored with one set of vectors (rows = projected directions), or None if it has none. Vectors
    stored with a projection live in the projected space and are compared with the projected displacement."""
    z = np.load(actions_path(cfg, space, seed, name))
    return z["proj"] if "proj" in z.files else None


def save_actions(cfg, space, seed, F, genes, info, hist, tau, name="infonce"):
    """Write one trained set of action vectors, its training history and its settings. A projection returned by the
    training (info["proj"]) is stored next to the vectors."""
    info = dict(info)
    proj = info.pop("proj", None)
    extra = {} if proj is None else {"proj": np.asarray(proj, dtype=np.float32)}
    np.savez(actions_path(cfg, space, seed, name), emb=F.astype(np.float32), genes=np.array(genes, dtype=str),
             missing=np.array([], dtype=str), **extra)
    d = path(cfg, "results_dir") / "infonce"
    d.mkdir(parents=True, exist_ok=True)
    tag = "" if name == "infonce" else f"{name}__"
    hist.to_csv(d / f"history__{tag}{space}__s{seed}.csv", index=False)
    write_json(info | {"tau": tau, "space": space, "seed": seed}, d / f"info__{tag}{space}__s{seed}.json")


# --------------------------------------------------------------------------- displacements and the computed projection
def displacements(cfg, adata, space, split, start):
    """Cell minus start for every single-perturbed cell of a split, with its gene. The start is "ot" (the cell's own
    OT partners; those pairs were made per perturbation, so this start uses the label and is for training cells),
    "pooled_ot" (partners from the pairing made without labels) or "ctrl_mean" (the mean control of its gem group)."""
    S = states(adata, space)
    z = load_pairs(cfg, space, split, pooled=start != "ot")
    s0 = gem_mean_states(cfg, adata, S, split, z["cell"]) if start == "ctrl_mean" else start_states(S, z)
    return (S[z["cell"]] - s0).astype(np.float64), z["cond"]


def whitening(cfg, adata, space, start) -> np.ndarray:
    """The computed projection: the rescaling linear discriminant analysis applies before it compares a displacement
    with the gene centres. It shrinks the directions in which cells of one gene are spread widely. Fitted with
    scikit-learn on the training cells' displacements only; rows are the projected directions."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X, y = displacements(cfg, adata, space, "train", start)
    return LinearDiscriminantAnalysis().fit(X, y).scalings_.T.astype(np.float32)


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


def effect_tiers(adata, space, ms) -> tuple[dict[str, float], dict[str, str]]:
    """Each gene's effect strength (the length of its training mean shift, in units of the typical distance between
    two training cells) and which third of the genes it falls in: weak, medium or strong."""
    D0 = scale_D0(states(adata, space)[(adata.obs["split"] == "train").to_numpy()])
    strength = {g: float(np.linalg.norm(v) / np.sqrt(D0)) for g, v in ms.items()}
    cuts = np.quantile(list(strength.values()), [1 / 3, 2 / 3])
    tier = {g: ("weak" if s <= cuts[0] else "medium" if s <= cuts[1] else "strong") for g, s in strength.items()}
    return strength, tier


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
