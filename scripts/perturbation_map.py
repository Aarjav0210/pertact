"""Describe the perturbation design: singles, designed doubles and how each double maps to its singles.

Writes results/perturbation_map.json (used by the Data tab of the project site).

    python -I scripts/perturbation_map.py
"""
from collections import Counter

import anndata as ad

from pertact.config import load_config, path, read_json, write_json
from pertact.data import CONTROL, cell_labels, condition_genes

EXAMPLE_CELLS = ["AGACGTTGTCTAGCGC", "CAGTAACCAGCCTTGG", "CCTTCCCTCCGTCATC", "AACTCAGAGACTGGGT"]
SNIPPET_GENES = ["KLF1", "SET", "CEBPE", "HBG2", "HBZ", "GATA1"]
# condition averages: two doubles next to their own singles, so the table reads as a + b vs a+b
MEAN_CONDITIONS = [CONTROL, "KLF1", "SET", "KLF1+SET", "CEBPE", "RUNX1T1", "CEBPE+RUNX1T1"]
MEAN_GENES = ["KLF1", "SET", "CEBPE", "RUNX1T1", "HBG2", "HBZ", "GATA1"]


def main():
    """Collect the design, the check of each guide on its own gene, example cells and condition averages into
    results/perturbation_map.json."""
    cfg = load_config()
    adata = ad.read_h5ad(path(cfg, "processed_h5ad"))
    splits = read_json(path(cfg, "splits"))
    qc = read_json(path(cfg, "qc_stats"))
    obs = adata.obs
    cond, _, split = cell_labels(obs, cfg["data"]["batch_key"])
    vidx = {g: i for i, g in enumerate(adata.var_names)}
    t2v = dict(adata.uns["target_to_var"])

    dropped = qc["dropped_small_perturbations"]
    designed = splits["test_doubles"] + splits["excluded_doubles"] + sorted(dropped)
    deg = Counter(g for d in designed for g in condition_genes(d))
    n = Counter(cond)

    # CRISPRa check on training cells: target log-expression in its own single minus training controls
    tr_ctrl = (split == "train") & (cond == CONTROL)
    singles = []
    for g in splits["train_singles"]:
        j = vidx[t2v[g]]
        m = (split == "train") & (cond == g)
        delta = float(adata.X[m][:, j].mean() - adata.X[tr_ctrl][:, j].mean())
        singles.append({"gene": g, "n": n[g], "train": int(m.sum()), "val": int(((split == "val") & (cond == g)).sum()),
                        "n_doubles": deg.get(g, 0), "target_delta": round(delta, 3)})
    doubles = []
    for d in designed:
        a, b = condition_genes(d)
        doubles.append({"name": d, "a": a, "b": b, "n": n.get(d, dropped.get(d, 0)),
                        "status": "test" if d in splits["test_doubles"] else
                        "excluded" if d in splits["excluded_doubles"] else "dropped"})
    examples = []
    for bc in EXAMPLE_CELLS:
        i = obs.index.get_loc(bc)
        r = obs.iloc[i]
        examples.append({"barcode": bc, "guide_id": str(r.guide_id).split(";")[0], "condition": str(r.condition),
                         "gemgroup": str(r.gemgroup), "split": str(r.split), "umis": int(r.total_counts),
                         "genes": int(r.n_genes_by_counts),
                         "expr": {g: [int(adata.layers["counts"][i, vidx[g]]), round(float(adata.X[i, vidx[g]]), 2)]
                                  for g in SNIPPET_GENES}})
    means = []
    for c in MEAN_CONDITIONS:
        m = cond == c
        means.append({"condition": c, "n_cells": int(m.sum()),
                      "mean": {g: round(float(adata.X[m][:, vidx[t2v.get(g, g)]].mean()), 3) for g in MEAN_GENES}})
    single_genes = set(splits["train_singles"])
    out = {
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "n_singles": len(singles), "n_designed_doubles": len(designed), "n_test_doubles": len(splits["test_doubles"]),
        "genes_in_doubles": len(deg), "double_genes_without_single": sorted(set(deg) - single_genes),
        "singles": singles, "doubles": doubles, "examples": examples, "snippet_genes": SNIPPET_GENES,
        "condition_means": means, "mean_genes": MEAN_GENES,
    }
    write_json(out, path(cfg, "results_dir") / "perturbation_map.json")
    print(f"{len(singles)} singles, {len(designed)} designed doubles over {len(deg)} genes; "
          f"double genes without a single: {out['double_genes_without_single']}")


if __name__ == "__main__":
    main()
