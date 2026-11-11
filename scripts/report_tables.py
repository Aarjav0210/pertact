"""Markdown results tables for REPORT.md, generated from results/week1.csv (mean ± std over seeds).

    python -I scripts/report_tables.py                      # print the two tables
    python -I scripts/report_tables.py --compare results/week1_cap60.csv   # also print re-run vs earlier values
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["mse_all", "mse_de20", "pearson_delta_all", "pearson_delta_de20"]
NAMES = {"onehot": "one-hot", "pca_loadings": "PCA loadings (K=32)", "genept": "GenePT", "string_n2v": "STRING node2vec"}
DEC = {"latent_additive": "latent additive", "decoder_only": "decoder only"}


def summary(df):
    """Mean, standard deviation and count over seeds."""
    g = df.groupby(["method", "decoder", "split", "metric"])["value"]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "n": g.size()}).reset_index()


def cell(s, method, decoder, split, metric):
    """One formatted table cell: the mean, or mean ± sd when there are several seeds."""
    r = s[(s.method == method) & (s.decoder == decoder) & (s.split == split) & (s.metric == metric)]
    if r.empty or pd.isna(r["mean"].iloc[0]):
        return "n/a"
    m, sd, n = r["mean"].iloc[0], r["std"].iloc[0], r["n"].iloc[0]
    f = "{:.4f}" if metric.startswith("mse") else "{:.3f}"
    return f.format(m) if n == 1 else f"{f.format(m)} ± {f.format(sd)}"


NEW_METRICS = ["pearson_int_all", "pearson_int_top20", "pearson_delta_de20_nt", "mse_de20_nt"]


def table(s, split, baselines, covered_label, metrics=None):
    """One Markdown table for a split: the baselines first, then every embedding with both decoders."""
    metrics = metrics or METRICS
    names = {"mse_all": "MSE all", "mse_de20": "MSE top-20 DE", "pearson_delta_all": "Pearson Δ all",
             "pearson_delta_de20": "Pearson Δ top-20 DE", "pearson_int_all": "Interaction r, all genes",
             "pearson_int_top20": "Interaction r, top-20 int. genes", "pearson_delta_de20_nt": "Pearson Δ top-20 DE, no targets",
             "mse_de20_nt": "MSE top-20 DE, no targets"}
    head = "| method | decoder | " + " | ".join(names[m] for m in metrics) + " |\n|" + "---|" * (len(metrics) + 2)
    rows = [f"| {name} | baseline | " + " | ".join(cell(s, b, "baseline", split, m) for m in metrics) + " |"
            for b, name in baselines]
    for meth, name in NAMES.items():
        for dec, dname in DEC.items():
            rows.append(f"| {name} | {dname} | " + " | ".join(cell(s, meth, dec, split, m) for m in metrics) + " |")
    if covered_label:  # only STRING has a covered subset different from the full split
        for dec, dname in DEC.items():
            rows.append(f"| STRING node2vec, {covered_label} | {dname} | "
                        + " | ".join(cell(s, "string_n2v", dec, f"{split}_covered", m) for m in metrics) + " |")
    return head + "\n" + "\n".join(rows)


def main():
    """Print the report's tables."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/week1.csv")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    s = summary(pd.read_csv(ROOT / a.results))
    print("### Test: 127 held-out doubles (composition)\n")
    print(table(s, "test", [("additive_expr", "additive expression (Δa + Δb)"), ("control_mean", "control mean (no change)")],
                "108 covered doubles"))
    print("\n### Validation: 105 singles (held-out cells of seen perturbations)\n")
    print(table(s, "val", [("train_mean", "training-cell mean (Δa)"), ("control_mean", "control mean")], "93 covered singles"))
    base_t = [("additive_expr", "additive expression (Δa + Δb)"), ("control_mean", "control mean (no change)"),
              ("noise_ceiling", "noise ceiling (split-half)")]
    print("\n### Beyond additivity: test doubles\n")
    print(table(s, "test", base_t, "108 covered doubles", NEW_METRICS))
    print("\n### Most non-additive quarter of test doubles (32)\n")
    print(table(s, "test_nonadditive", base_t, None, ["pearson_delta_de20", "pearson_int_all", "pearson_int_top20", "mse_de20"]))
    if a.compare:
        old = pd.read_csv(ROOT / a.compare)
        new = pd.read_csv(ROOT / a.results)
        k = ["method", "decoder", "split", "seed", "metric"]
        m = new.merge(old, on=k, suffixes=("_new", "_old"))
        m = m[(m.decoder != "baseline") & (m.metric == "pearson_delta_de20") & m.split.isin(["val", "test"])
              & (m.value_new != m.value_old)]
        print("\n### Re-run vs earlier (Pearson Δ top-20 DE, changed runs only)\n")
        print(m.pivot_table(index=["method", "decoder", "seed"], columns="split", values=["value_old", "value_new"])
              .round(3).to_string())


if __name__ == "__main__":
    main()
