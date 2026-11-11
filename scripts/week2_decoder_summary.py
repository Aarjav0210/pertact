"""Week 2 decoder smoke test: the summary numbers in REPORT_week2.md section 5.4 (validation singles only).

Reads every results/week2_decoder/smoke_per_perturbation__*.csv and results/week2_effect_strength.csv.

Writes
  results/week2_decoder/summary.csv                 per embedding and decoder: mean and sd over seeds, overall,
                                                    target-excluded and by effect-strength third
  results/week2_decoder/paired_length_vs_unit.csv   per action set, decoder and seed: weak-third score with fitted
                                                    lengths and as direction only (unit length)

Effect-strength thirds are the PCA-50 ones for every row, so each row is scored on the same 35 weak perturbations.
Run: .venv/bin/python -I scripts/week2_decoder_summary.py
"""
import glob

import pandas as pd

from pertact.config import load_config, path

SCORE, SCORE_NT = "pearson_delta_de20", "pearson_delta_de20_nt"
TIERS = ["weak", "medium", "strong"]
# Action sets with fitted lengths, paired with the same directions at unit length.
PAIRS = [("infonce_scaled_per_action_pca", "infonce_unit_pca"), ("infonce_scaled_per_action_scvi", "infonce_unit_scvi"),
         ("mean_shift_pca", "mean_shift_unit_pca"), ("mean_shift_scvi", "mean_shift_unit_scvi")]


def main():
    """Summarise the decoder test and write the two summary files."""
    cfg = load_config(None)
    rd = path(cfg, "results_dir")
    out = rd / "week2_decoder"
    per = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(out / "smoke_per_perturbation__*.csv")))])
    per = per[per.split == "val"].drop_duplicates(["method", "decoder", "seed", "condition"])
    tiers = pd.read_csv(rd / "week2_effect_strength.csv").query("space == 'X_pca_g'").set_index("gene")["tier"]
    per["tier"] = per["condition"].map(tiers)

    keys = ["method", "decoder", "seed"]
    by_seed = per.groupby(keys).agg(all_105=(SCORE, "mean"), target_excluded=(SCORE_NT, "mean"))
    for t in TIERS:
        by_seed[t] = per[per.tier == t].groupby(keys)[SCORE].mean()

    grouped = by_seed.groupby(["method", "decoder"])
    summary = grouped.mean().join(grouped.std(), rsuffix="_sd").join(grouped.size().rename("n_seeds"))
    summary.to_csv(out / "summary.csv")

    paired = []
    for fitted, unit in PAIRS:
        for decoder in ("latent_additive", "decoder_only"):
            for seed in cfg["eval"]["seeds"]:
                a, b = by_seed.loc[(fitted, decoder, seed), "weak"], by_seed.loc[(unit, decoder, seed), "weak"]
                paired.append({"fitted_lengths": fitted, "unit_length": unit, "decoder": decoder, "seed": seed,
                               "weak_fitted": a, "weak_unit": b, "improved": b > a})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired_length_vs_unit.csv", index=False)

    pd.set_option("display.width", 220)
    print(summary[["all_105", "all_105_sd", "target_excluded", "weak", "weak_sd", "medium", "strong", "n_seeds"]].round(3).to_string())
    print(f"\nWeak third improves at unit length in {int(paired.improved.sum())} of {len(paired)} seed-paired comparisons.")
    print(paired[~paired.improved].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
