"""DE genes, metrics, baselines, hyperparameter selection and the week-1 runs.

    python -I -m pertact.evaluate de          # results/de_genes.json (singles: train cells; doubles: own cells)
    python -I -m pertact.evaluate tune        # one-hot + latent additive grid on validation singles
    python -I -m pertact.evaluate baselines   # additive expression, train-mean, control mean, noise ceiling
    python -I -m pertact.evaluate run [--methods ...] [--decoders ...] [--seeds ...]
    python -I -m pertact.evaluate rescore     # recompute metrics from saved predictions (no retraining)
    python -I -m pertact.evaluate compose [--methods ...] [--seeds ...]   # additive vs sequential doubles
    python -I -m pertact.evaluate actions --variants ... [--spaces ...] [--tag t]  # Week 2 smoke test: frozen action
                         # vectors through both decoders, validation singles only (no test doubles);
                         # results/week2_decoder/smoke__<tag>.csv
    python -I -m pertact.evaluate actions --doubles --variants ... [--tag t]       # Week 3: the same decoders also
                         # scored on the 127 test doubles; results/week3_decoder/doubles__<tag>.csv

The metrics are defined in metrics.py; the tables for the report are printed by scripts/report_tables.py.
"""
from __future__ import annotations

import argparse

from .config import load_config, seed_everything
from .decoder_test import ACTION_VARIANTS, action_smoke
from .encoders import load_all
from .metrics import compute_de, load_de
from .runs import compose, rescore, run, run_baselines, tune


def main():
    """Dispatch the evaluation commands."""
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["de", "tune", "baselines", "run", "rescore", "compose", "actions"])
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--decoders", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=None, help="actions: " + ", ".join(ACTION_VARIANTS))
    ap.add_argument("--spaces", nargs="*", default=None, help="actions: state spaces of the action vectors")
    ap.add_argument("--references", nargs="*", default=["onehot", "pca_loadings"], help="actions: Week 1 embeddings")
    ap.add_argument("--tag", default="main", help="actions: output file tag")
    ap.add_argument("--doubles", action="store_true", help="actions: also score the 127 test doubles (results/week3_decoder/)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed_everything(cfg["seed"])
    if a.what == "de":
        adata, splits = load_all(cfg)
        compute_de(cfg, adata, splits)
    elif a.what == "baselines":
        adata, splits = load_all(cfg)
        run_baselines(cfg, adata, splits, load_de(cfg))
    elif a.what == "tune":
        tune(cfg)
    elif a.what == "rescore":
        rescore(cfg)
    elif a.what == "compose":
        compose(cfg, a.methods or cfg["eval"]["methods"], a.seeds if a.seeds is not None else cfg["eval"]["seeds"])
    elif a.what == "actions":
        action_smoke(cfg, a.variants or ["infonce_scaled_per_action"], a.spaces or cfg["ot"]["spaces"],
                     a.decoders or cfg["eval"]["decoders"], a.seeds if a.seeds is not None else cfg["eval"]["seeds"],
                     references=a.references, tag=a.tag, doubles=a.doubles)
    else:
        run(cfg, a.methods or cfg["eval"]["methods"], a.decoders or cfg["eval"]["decoders"],
            a.seeds if a.seeds is not None else cfg["eval"]["seeds"])


if __name__ == "__main__":
    main()
