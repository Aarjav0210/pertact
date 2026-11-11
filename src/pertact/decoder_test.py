"""The Week 2 decoder test: frozen action vectors as the perturbation embedding of the Week 1 decoders.

Validation singles only; the runner is built without the test split, so no test cell is read. Results go to
results/week2_decoder/smoke__<tag>.csv. The command is `python -I -m pertact.evaluate actions`.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .actions import calibrate_scale, load_actions, mean_shift_actions, per_action_scale
from .config import path, read_json
from .decoders import PertEmbedding
from .runs import Runner

SPACE_SHORT = {"X_pca_g": "pca", "X_scvi30": "scvi"}
ACTION_VARIANTS = ("infonce", "infonce_calibrated", "infonce_scaled_per_action", "infonce_anchor", "mean_shift",
                   "infonce_unit", "mean_shift_unit")


def action_embedding(cfg, adata, splits, space, seed, variant) -> tuple[list[str], np.ndarray]:
    """Frozen Week 2 action vectors f(a) for the training singles. InfoNCE seed = decoder seed; the mean shift is
    deterministic. Global rescaling is undone by the unit_mean_norm step every embedding goes through, so
    infonce_calibrated reaches the decoder exactly as raw infonce does; per-action scaling changes relative lengths.
    A "_unit" suffix keeps only the direction: every vector is divided by its own norm (all lengths equal, as one-hot)."""
    if variant.endswith("_unit"):
        genes, M = action_embedding(cfg, adata, splits, space, seed, variant[:-len("_unit")])
        return genes, M / np.linalg.norm(M, axis=1, keepdims=True)
    genes = sorted(splits["train_singles"])
    if variant == "mean_shift":
        acts = mean_shift_actions(cfg, adata, space)
    else:
        acts = load_actions(cfg, space, seed, "infonce_anchor" if variant == "infonce_anchor" else "infonce")
        if variant == "infonce_calibrated":
            alpha = calibrate_scale(cfg, adata, space, acts)
            acts = {g: alpha * v for g, v in acts.items()}
        elif variant == "infonce_scaled_per_action":
            c = per_action_scale(cfg, adata, space, acts)
            acts = {g: c[g] * v for g, v in acts.items()}
        elif variant not in ("infonce", "infonce_anchor"):
            raise ValueError(variant)
    return genes, np.stack([acts[g] for g in genes])


def action_smoke(cfg, variants, spaces, decoders, seeds, references=("onehot", "pca_loadings"), tag="main"):
    """Validation singles only. References reuse the saved Week 1 models (same hyperparameters and seeds)."""
    bp = path(cfg, "results_dir") / "best_hparams.json"
    runner = Runner(cfg, read_json(bp)["hparams"], eval_splits=("val",))
    rd = path(cfg, "results_dir")
    out = rd / "week2_decoder"  # one file set per invocation (tag), so runs can go in parallel
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows, per, infos = [], [], []
    jobs = [(m, None, None) for m in references]
    jobs += [(f"{v}_{SPACE_SHORT[sp]}", v, sp) for v in variants for sp in spaces]
    for (method, variant, space), decoder, seed in itertools.product(jobs, decoders, seeds):
        if variant is not None:
            genes, M = action_embedding(cfg, runner.adata, runner.splits, space, seed, variant)
            runner.custom_emb[(method, seed)] = PertEmbedding.from_arrays(
                method, genes, M, cfg["embeddings"]["normalize"], ref_genes=runner.splits["train_singles"])
        r, p, info = runner.fit_eval(method, decoder, seed, splits=("val",),
                                     pred_file=pred_dir / f"{method}__{decoder}__s{seed}.npz")
        rows += r
        per += p
        infos.append({"method": method, "decoder": decoder, "seed": seed, "variant": variant, "space": space} | info)
        pd.DataFrame(rows).to_csv(out / f"smoke__{tag}.csv", index=False)
        pd.concat(per).to_csv(out / f"smoke_per_perturbation__{tag}.csv", index=False)
        pd.DataFrame(infos).to_csv(out / f"smoke_train_info__{tag}.csv", index=False)
    return pd.DataFrame(rows)
