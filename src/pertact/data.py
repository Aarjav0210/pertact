"""Norman 2019 loading, guide parsing, QC, normalisation and train-only HVG selection.

Two stages, with the split built in between so that every gene-level choice
(gene detection filter, HVGs) uses training cells only:

    python -I -m pertact.data qc         # stage 1: cell-level filters -> interim h5ad (all genes, counts)
    python -I -m pertact.splits          # splits.json
    python -I -m pertact.data hvg        # stage 2: gene filter + HVGs on train cells -> processed h5ad
"""
from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from .config import get_logger, load_config, path, read_json, write_json

log = get_logger("data")

CONTROL = "control"


# --------------------------------------------------------------------------- guide parsing
def construct_genes(construct: str, control_prefix: str) -> tuple[str, ...]:
    """'KLF1_NegCtrl0' -> ('KLF1',);  'TGFBR2_IGDCC3_2' -> ('IGDCC3', 'TGFBR2');  'NegCtrl1_NegCtrl0' -> ()."""
    toks = [t for t in construct.split("_") if t and not t.isdigit()]  # '_1', '_2' = alternative sgRNA
    return tuple(sorted(t for t in toks if not t.startswith(control_prefix)))


def parse_guide_id(guide_id: str, control_prefix: str) -> tuple[tuple[str, ...] | None, str]:
    """Return (sorted target genes, status). status in {ok, ambiguous, too_many}."""
    sets = {construct_genes(c, control_prefix) for c in str(guide_id).split(";") if c}
    if len(sets) != 1:
        return None, "ambiguous"
    genes = next(iter(sets))
    if len(set(genes)) != len(genes):
        return None, "ambiguous"
    if len(genes) > 2:
        return None, "too_many"
    return genes, "ok"


def condition_name(genes: tuple[str, ...]) -> str:
    """Label of a perturbation: 'control' for no genes, else its genes in alphabetical order joined by '+'."""
    return CONTROL if len(genes) == 0 else "+".join(sorted(genes))


def condition_genes(cond: str) -> list[str]:
    """The genes of a perturbation label; an empty list for control."""
    return [] if cond == CONTROL else cond.split("+")


def cell_labels(obs: pd.DataFrame, batch_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each cell's perturbation, gem group and split (train / val / test / unused), as plain string arrays."""
    return (obs["condition"].astype(str).to_numpy(), obs[batch_key].astype(str).to_numpy(),
            obs["split"].astype(str).to_numpy())


# --------------------------------------------------------------------------- gene symbols
class SymbolResolver:
    """Map guide/target symbols to HGNC records (current symbol, previous symbols, aliases, Ensembl id)."""

    def __init__(self, hgnc_path):
        h = pd.read_csv(hgnc_path, sep="\t", dtype=str,
                        usecols=["symbol", "prev_symbol", "alias_symbol", "ensembl_gene_id", "status"])
        h = h[h.status == "Approved"].fillna("")
        self.current = {}
        self.ensembl = {}
        prev, alias = {}, {}
        for r in h.itertuples(index=False):
            self.current[r.symbol] = r.symbol
            self.ensembl[r.symbol] = r.ensembl_gene_id
            for p in filter(None, r.prev_symbol.split("|")):
                prev.setdefault(p, set()).add(r.symbol)
            for a in filter(None, r.alias_symbol.split("|")):
                alias.setdefault(a, set()).add(r.symbol)
        self.prev, self.alias = prev, alias
        self.synonyms = {}
        for r in h.itertuples(index=False):
            self.synonyms[r.symbol] = [r.symbol] + [s for s in (r.prev_symbol + "|" + r.alias_symbol).split("|") if s]

    def to_current(self, sym: str) -> tuple[str | None, str]:
        """Return (current HGNC symbol, how). Unambiguous previous/alias symbols only."""
        if sym in self.current:
            return sym, "current"
        for table, how in ((self.prev, "previous"), (self.alias, "alias")):
            hits = table.get(sym, set())
            if len(hits) == 1:
                return next(iter(hits)), how
            if len(hits) > 1:
                return None, f"ambiguous_{how}"
        return None, "unknown"

    def candidates(self, sym: str) -> list[str]:
        """sym itself, then its current symbol, then all synonyms of the current symbol."""
        out = [sym]
        cur, _ = self.to_current(sym)
        if cur:
            out += [s for s in self.synonyms.get(cur, [cur]) if s not in out]
        return out


def resolve_targets_to_var(targets, var: pd.DataFrame, resolver: SymbolResolver) -> dict[str, dict]:
    """target symbol -> {var_name, how}. Uses exact name, then HGNC Ensembl id, then synonyms."""
    ens2var = dict(zip(var["ensemble_id"], var.index))
    out = {}
    for t in targets:
        if t in var.index:
            out[t] = {"var_name": t, "how": "exact"}
            continue
        cur, how = resolver.to_current(t)
        e = resolver.ensembl.get(cur, "") if cur else ""
        if e in ens2var:
            out[t] = {"var_name": ens2var[e], "how": f"hgnc_{how}->ensembl {e}"}
            continue
        hit = next((c for c in resolver.candidates(t) if c in var.index), None)
        out[t] = {"var_name": hit, "how": "hgnc_synonym" if hit else "missing"}
    return out


# --------------------------------------------------------------------------- stage 1
def annotate(adata: ad.AnnData, cfg: dict) -> pd.DataFrame:
    """Parse guide ids into conditions and per-cell drop reasons."""
    d = cfg["data"]
    obs = adata.obs
    parsed = obs["guide_id"].astype(str).map(lambda g: parse_guide_id(g, d["control_prefix"]))
    genes = parsed.map(lambda x: x[0])
    status = parsed.map(lambda x: x[1])
    reason = pd.Series("", index=obs.index, dtype=object)
    if d["require_good_coverage"]:
        reason[~obs["good_coverage"].astype(bool)] = "low_guide_coverage"
    reason[(reason == "") & (obs["number_of_cells"] > d["max_guide_identities"])] = "multiple_guide_identities"
    reason[(reason == "") & (status != "ok")] = status[(reason == "") & (status != "ok")].map(lambda s: f"guide_{s}")
    cond = genes.map(lambda g: condition_name(g) if g is not None else "")
    return pd.DataFrame({"condition": cond, "drop_reason": reason,
                         "n_targets": genes.map(lambda g: len(g) if g is not None else -1)})


def cross_check_labels(obs: pd.DataFrame, ann: pd.DataFrame) -> dict:
    """Compare our parsed conditions with scPerturb's `perturbation` column (A_B order-dependent)."""
    theirs = obs["perturbation"].astype(str).map(
        lambda s: CONTROL if s == CONTROL else "+".join(sorted(s.split("_"))))
    ok = ann["condition"] != ""
    agree = (theirs[ok] == ann.loc[ok, "condition"])
    return {"n_parsed": int(ok.sum()), "n_agree_with_scperturb": int(agree.sum()),
            "disagreements": theirs[ok][~agree].value_counts().head(10).to_dict()}


def build_qc(cfg: dict) -> None:
    """Stage 1. Read the raw file, work out each cell's perturbation, apply the guide filter, the cell quality filters
    and the minimum-cells rule, and write the QC'd file with all genes and raw counts. No gene is chosen here, so
    this stage runs before the split exists."""
    d = cfg["data"]
    stats: dict = {"source": str(path(cfg, "raw_h5ad"))}
    log.info("reading %s", path(cfg, "raw_h5ad"))
    adata = ad.read_h5ad(path(cfg, "raw_h5ad"))
    adata.X = sp.csr_matrix(adata.X)
    stats["raw"] = {"n_cells": adata.n_obs, "n_genes": adata.n_vars}
    stats["raw_label_counts"] = {
        "control_cells": int((adata.obs["perturbation"] == CONTROL).sum()),
        "n_single_labels": int((adata.obs.loc[adata.obs.nperts == 1, "perturbation"]).nunique()),
        "n_double_labels": int((adata.obs.loc[adata.obs.nperts == 2, "perturbation"]).nunique()),
    }

    # orientation: constructs A_B and B_A both exist for some pairs -> merged into one condition
    cons = adata.obs["guide_id"].astype(str).str.split(";").str[0]
    orient = pd.DataFrame({"c": cons, "g": cons.map(lambda c: construct_genes(c, d["control_prefix"]))})
    orient = orient.drop_duplicates("c").groupby("g")["c"].apply(sorted)
    stats["constructs_per_condition"] = {
        "n_conditions_with_multiple_constructs": int((orient.map(len) > 1).sum()),
        "examples": {condition_name(k): v for k, v in list(orient[orient.map(len) > 1].items())[:5]},
    }

    ann = annotate(adata, cfg)
    stats["label_cross_check"] = cross_check_labels(adata.obs, ann)
    adata.obs["condition"] = ann["condition"].values
    adata.obs["n_targets"] = ann["n_targets"].values
    stats["guide_filter"] = {k: int(v) for k, v in ann["drop_reason"].value_counts().items() if k}
    adata = adata[ann["drop_reason"].values == ""].copy()
    stats["after_guide_filter"] = adata.n_obs

    # cell QC, recomputed from counts
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    qc_fail = {
        f"total_counts<{d['min_counts']}": adata.obs.total_counts < d["min_counts"],
        f"n_genes<{d['min_genes']}": adata.obs.n_genes_by_counts < d["min_genes"],
        f"pct_mito>{d['max_pct_mito']}": adata.obs.pct_counts_mt > d["max_pct_mito"],
    }
    stats["cell_qc_fail_counts"] = {k: int(v.sum()) for k, v in qc_fail.items()}
    bad = np.logical_or.reduce([v.values for v in qc_fail.values()])
    stats["cell_qc_removed"] = int(bad.sum())
    adata = adata[~bad].copy()
    stats["after_cell_qc"] = adata.n_obs

    # perturbation size filter
    counts = adata.obs["condition"].value_counts()
    small = counts[(counts < d["min_cells_per_perturbation"]) & (counts.index != CONTROL)]
    stats["dropped_small_perturbations"] = {k: int(v) for k, v in small.items()}
    adata = adata[~adata.obs["condition"].isin(small.index)].copy()
    stats["after_min_cells_filter"] = adata.n_obs

    adata.obs["condition"] = adata.obs["condition"].astype("category")
    adata.obs["perturbed"] = adata.obs["n_targets"] > 0
    adata.obs["target_1"] = adata.obs["condition"].astype(str).map(lambda c: (condition_genes(c) + ["", ""])[0])
    adata.obs["target_2"] = adata.obs["condition"].astype(str).map(lambda c: (condition_genes(c) + ["", ""])[1])
    keep_obs = ["guide_id", "condition", "n_targets", "perturbed", "target_1", "target_2", "gemgroup",
                "perturbation", "UMI_count", "total_counts", "n_genes_by_counts", "pct_counts_mt"]
    adata.obs = adata.obs[keep_obs].rename(columns={"perturbation": "scperturb_label", "UMI_count": "guide_umis"})
    adata.obs["gemgroup"] = adata.obs["gemgroup"].astype(str).astype("category")
    adata.var = adata.var[["ensemble_id"]].copy()
    adata.X = adata.X.astype(np.float32)

    stats.update(summarise_conditions(adata.obs))
    resolver = SymbolResolver(path(cfg, "hgnc"))
    targets = sorted({g for c in adata.obs["condition"].cat.categories for g in condition_genes(c)})
    stats["target_to_var"] = resolve_targets_to_var(targets, adata.var, resolver)
    adata.uns["target_to_var"] = {t: (v["var_name"] or "") for t, v in stats["target_to_var"].items()}
    write_json(stats, path(cfg, "qc_stats"))
    out = path(cfg, "qc_h5ad")
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("writing %s  (%d cells x %d genes)", out, adata.n_obs, adata.n_vars)
    adata.write_h5ad(out, compression="gzip")
    log.info("summary: %s", {k: stats[k] for k in ("after_guide_filter", "after_cell_qc", "after_min_cells_filter",
                                                     "n_control_cells", "n_singles", "n_doubles")})


def summarise_conditions(obs: pd.DataFrame) -> dict:
    """Counts for the statistics file: control, single and double cells, and the smallest, median and largest group."""
    vc = obs["condition"].value_counts()
    nt = obs.drop_duplicates("condition").set_index("condition")["n_targets"]
    singles, doubles = vc[nt.reindex(vc.index) == 1], vc[nt.reindex(vc.index) == 2]

    def mmm(s):
        return {"min": int(s.min()), "median": float(s.median()), "max": int(s.max())} if len(s) else {}

    return {"n_cells": int(len(obs)), "n_control_cells": int(vc.get(CONTROL, 0)),
            "n_singles": int(len(singles)), "n_doubles": int(len(doubles)),
            "n_single_cells": int(singles.sum()), "n_double_cells": int(doubles.sum()),
            "cells_per_single": mmm(singles), "cells_per_double": mmm(doubles),
            "cells_per_perturbation": mmm(pd.concat([singles, doubles]))}


# --------------------------------------------------------------------------- stage 2
def build_processed(cfg: dict) -> None:
    """Stage 2. Label each cell with its split, choose the genes from training cells only (detected genes, then highly
    variable genes, plus every perturbed gene), normalise each cell, and write the model matrix."""
    d = cfg["data"]
    splits = read_json(path(cfg, "splits"))
    log.info("reading %s", path(cfg, "qc_h5ad"))
    adata = ad.read_h5ad(path(cfg, "qc_h5ad"))
    adata.layers["counts"] = adata.X.copy()
    split = pd.Series("unused", index=adata.obs_names, dtype=object)
    for name in ("train", "val", "test"):
        split[splits["cells"][name]] = name
    adata.obs["split"] = pd.Categorical(split.values)
    train = adata.obs["split"].values == "train"
    log.info("train cells used for gene selection: %d", train.sum())

    # rarely-detected genes, on training cells
    ncells_train = np.asarray((adata.X[train] > 0).sum(0)).ravel()
    gene_ok = ncells_train >= d["min_cells_per_gene"]
    t2v = {t: v for t, v in adata.uns["target_to_var"].items() if v}
    target_var = set(t2v.values())
    is_target = adata.var_names.isin(list(target_var))
    stats = {"genes_total": int(adata.n_vars), "genes_detected_in_train": int(gene_ok.sum()),
             "min_cells_per_gene": d["min_cells_per_gene"]}

    # normalise per cell on all genes (library size), then restrict genes
    sc.pp.normalize_total(adata, target_sum=d["target_sum"])
    sc.pp.log1p(adata)

    cand = adata[:, gene_ok | is_target]
    tr = cand[train].copy()
    hvg_kw = dict(n_top_genes=d["n_hvg"], flavor=d["hvg_flavor"])
    if d["hvg_flavor"].startswith("seurat_v3"):
        hvg_kw["layer"] = "counts"  # seurat_v3 expects raw counts; other flavours use log-normalised X
    if d.get("batch_key"):
        hvg_kw["batch_key"] = d["batch_key"]
    sc.pp.highly_variable_genes(tr, **hvg_kw)
    hvg = tr.var["highly_variable"].values
    forced = cand.var_names.isin(list(target_var)) & ~hvg
    keep = cand.var_names[hvg | cand.var_names.isin(list(target_var))]
    stats.update({"n_hvg": int(hvg.sum()), "n_targets_forced_in": int(forced.sum()),
                  "targets_forced_in": cand.var_names[forced].tolist(), "n_genes_final": int(len(keep)),
                  "hvg_flavor": d["hvg_flavor"], "hvg_batch_key": d.get("batch_key")})

    out = adata[:, keep].copy()
    out.var["highly_variable"] = out.var_names.isin(cand.var_names[hvg])
    v2t = {}
    for t, v in t2v.items():
        v2t.setdefault(v, []).append(t)
    out.var["is_target"] = out.var_names.isin(list(target_var))
    out.var["target_symbol"] = [",".join(v2t.get(g, [])) for g in out.var_names]
    out.var["n_cells_train"] = ncells_train[adata.var_names.get_indexer(keep)]
    out.X = sp.csr_matrix(out.X, dtype=np.float32)
    out.layers["counts"] = sp.csr_matrix(out.layers["counts"], dtype=np.float32)
    out.uns["hvg_stats"] = {k: v for k, v in stats.items() if not isinstance(v, list)}
    out.uns["log1p"] = {"base": None}
    write_json(stats, path(cfg, "qc_stats").with_name("hvg_stats.json"))
    p = path(cfg, "processed_h5ad")
    p.parent.mkdir(parents=True, exist_ok=True)
    log.info("writing %s (%d x %d); %s", p, out.n_obs, out.n_vars, {k: stats[k] for k in ("n_hvg", "n_targets_forced_in")})
    out.write_h5ad(p, compression="gzip")


def main():
    """Run one stage."""
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["qc", "hvg"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    build_qc(cfg) if a.stage == "qc" else build_processed(cfg)


if __name__ == "__main__":
    main()
