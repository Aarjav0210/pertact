"""Summarise the Week 3 decoder test: action vectors as decoder inputs, scored on the test doubles, next to Week 1.

    python -I scripts/week3_decoder_summary.py   # results/week3_decoder/summary.csv and paired.csv

Reads the six results/week3_decoder/doubles__<tag>.csv files written by `pertact.evaluate actions --doubles`, their
per-perturbation files, and the Week 1 tables. The one-hot and PCA-loading decoders in the `main` run are the saved
Week 1 models, so their rows must equal Week 1's; the script stops if they do not.
"""
import pandas as pd

from pertact.config import get_logger, load_config, path

log = get_logger("week3_decoder_summary")
WEEK1_METHODS = ["additive_expr", "control_mean", "noise_ceiling", "onehot", "genept", "pca_loadings", "string_n2v"]
DECODERS = ["baseline", "latent_additive", "decoder_only"]
HEADLINE = "pearson_delta_de20"


def load(cfg):
    """The Week 3 score rows of the action-vector sets (references checked against Week 1 and dropped), the Week 1
    rows to compare with, and the per-double headline score of every run."""
    res = path(cfg, "results_dir")
    d = pd.concat([pd.read_csv(f) for f in sorted((res / "week3_decoder").glob("doubles__*.csv"))], ignore_index=True)
    per = pd.concat([pd.read_csv(f) for f in sorted((res / "week3_decoder").glob("doubles_per_perturbation__*.csv"))], ignore_index=True)
    w1 = pd.read_csv(res / "week1.csv")
    w1 = w1[w1.method.isin(WEEK1_METHODS) & w1.decoder.isin(DECODERS)]
    is_ref = d.method.isin(WEEK1_METHODS)
    both = d[is_ref].merge(w1, on=["method", "decoder", "split", "seed", "metric"], suffixes=("", "_week1"))
    gap = float((both.value - both.value_week1).abs().max())
    assert len(both) == is_ref.sum() and gap < 1e-5, f"the reloaded Week 1 decoders do not reproduce Week 1 (largest difference {gap:.2e})"
    log.info("references: %d rows equal Week 1 (largest difference %.1e)", len(both), gap)
    w1per = pd.read_csv(res / "week1_per_perturbation.csv")
    w1per = w1per[w1per.method.isin(WEEK1_METHODS) & w1per.decoder.isin(DECODERS)]
    per = pd.concat([per[~per.method.isin(WEEK1_METHODS)], w1per], ignore_index=True)
    return d[~is_ref].assign(source="week3"), w1.assign(source="week1"), per[per.split == "test"]


def reference_set(method):
    """The mean-shift set an InfoNCE set is compared with: the same state space, and direction only against
    direction only. Empty for anything that is not an InfoNCE set."""
    if not method.startswith("infonce"):
        return ""
    space = method.rsplit("_", 1)[1]
    return f"mean_shift_unit_{space}" if "_unit_" in method else f"mean_shift_{space}"


def main():
    """Write the mean and seed spread of every score, and for each run how many of the 127 doubles it predicts better
    than adding the two singles, than one-hot in the same decoder, and than the matching mean-shift set."""
    cfg = load_config()
    new, w1, per = load(cfg)
    rows = pd.concat([new, w1], ignore_index=True)
    g = rows.groupby(["source", "method", "decoder", "split", "metric"], sort=False)
    summary = g.agg(mean=("value", "mean"), sd=("value", "std"), n_seeds=("value", "size"), n_conditions=("n_conditions", "first")).reset_index()
    out = path(cfg, "results_dir") / "week3_decoder"
    summary.to_csv(out / "summary.csv", index=False)

    # one headline score per double and run, averaged over seeds, then compared double by double
    score = per.groupby(["method", "decoder", "condition"])[HEADLINE].mean().unstack(["method", "decoder"])
    additive = score[("additive_expr", "baseline")]
    paired = []
    for method, decoder in score.columns:
        if decoder == "baseline":
            continue
        own, ref = score[(method, decoder)], reference_set(method)
        ok = own.notna()  # STRING has no vector for 12 genes; its uncovered doubles are still predicted, with a zero vector
        paired.append({"method": method, "decoder": decoder, "n_doubles": int(ok.sum()),
                       "beats_additive_expr": float((own > additive)[ok].mean()),
                       "beats_onehot": float((own > score[("onehot", decoder)])[ok].mean()) if method != "onehot" else float("nan"),
                       "compared_with": ref,
                       "beats_compared": float((own > score[(ref, decoder)])[ok].mean()) if ref else float("nan"),
                       "mean_minus_compared": float((own - score[(ref, decoder)])[ok].mean()) if ref else float("nan")})
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    show = summary[(summary.split == "test") & (summary.metric == HEADLINE)].pivot(index="method", columns="decoder", values="mean")
    log.info("test doubles, %s, mean over seeds:\n%s", HEADLINE, show.sort_values("latent_additive", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
