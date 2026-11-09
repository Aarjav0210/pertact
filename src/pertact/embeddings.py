"""Perturbation embeddings, one matrix per method, rows indexed by target gene symbol.

Each method writes embeddings/<method>.npz with
    emb   : (n_genes, d) float32, rows for the covered target genes only (no imputation)
    genes : target symbols (the guide's symbol, e.g. 'C19orf26', not the current HGNC 'CBARP')
    missing : target symbols the method could not cover
and a coverage record in embeddings/coverage.json.

    python -I -m pertact.embeddings [onehot pca_loadings genept string_n2v]
"""
from __future__ import annotations

import argparse
import gzip
import pickle

import anndata as ad
import numpy as np
import pandas as pd

from .config import ROOT, get_logger, load_config, path, read_json, seed_everything, write_json
from .data import SymbolResolver, condition_genes

log = get_logger("embeddings")
METHODS = ["onehot", "pca_loadings", "genept", "string_n2v"]


# --------------------------------------------------------------------------- io
def emb_path(cfg, method):
    return path(cfg, "embeddings_dir") / f"{method}.npz"


def save_embedding(cfg, method, genes, emb, missing, how=None, extra=None):
    p = emb_path(cfg, method)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, emb=np.asarray(emb, dtype=np.float32), genes=np.array(genes, dtype=str),
             missing=np.array(missing, dtype=str))
    cov_p = path(cfg, "embeddings_dir") / "coverage.json"
    cov = read_json(cov_p) if cov_p.exists() else {}
    cov[method] = {"dim": int(np.asarray(emb).shape[1]), "n_covered": len(genes), "n_missing": len(missing),
                   "missing": list(missing), "match": how or {}, **(extra or {})}
    write_json(cov, cov_p)
    log.info("%s: %d genes x %d dims, missing %d %s", method, len(genes), np.asarray(emb).shape[1],
             len(missing), list(missing))


def load_embedding(cfg, method) -> tuple[list[str], np.ndarray]:
    z = np.load(emb_path(cfg, method), allow_pickle=False)
    return z["genes"].tolist(), z["emb"]


def all_targets(splits) -> list[str]:
    conds = splits["train_singles"] + splits["test_doubles"] + splits["excluded_doubles"]
    return sorted({g for c in conds for g in condition_genes(c)})


# --------------------------------------------------------------------------- one-hot
def build_onehot(cfg, splits, **_):
    genes = sorted(splits["train_singles"])
    missing = [g for g in all_targets(splits) if g not in genes]
    save_embedding(cfg, "onehot", genes, np.eye(len(genes)), missing)


# --------------------------------------------------------------------------- PCA gene loadings
def build_pca_loadings(cfg, splits, adata=None, **_):
    """Ahlmann-Eltze et al. 2025: PCA of the (perturbation x gene) pseudobulk of training singles;
    a gene's embedding is its row of the gene-loading matrix."""
    k = cfg["embeddings"]["pca_loadings"]["k"]
    adata = adata if adata is not None else ad.read_h5ad(path(cfg, "processed_h5ad"))
    tr = (adata.obs["split"] == "train").to_numpy()
    cond = adata.obs["condition"].astype(str).to_numpy()
    perts = sorted(splits["train_singles"])
    Y = np.vstack([np.asarray(adata.X[tr & (cond == p)].mean(0)).ravel() for p in perts])  # perts x genes
    assert k <= len(perts) - 1, f"K={k} must be < number of training perturbations ({len(perts)})"
    Yc = Y - Y.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Yc, full_matrices=False)
    V = Vt[:k].T  # genes x K
    V *= np.sign(V[np.abs(V).argmax(0), np.arange(k)])  # deterministic sign
    var_names = adata.var_names.to_numpy()
    t2v = dict(adata.uns["target_to_var"])
    row = {g: i for i, g in enumerate(var_names)}
    targets = all_targets(splits)
    genes = [t for t in targets if t2v.get(t) in row]
    missing = [t for t in targets if t not in genes]
    emb = V[[row[t2v[t]] for t in genes]]
    full = path(cfg, "embeddings_dir") / "pca_loadings_full.npz"
    np.savez(full, emb=V.astype(np.float32), genes=var_names.astype(str),
             singular_values=s[:k], explained_variance_ratio=(s[:k] ** 2) / (s ** 2).sum())
    save_embedding(cfg, "pca_loadings", genes, emb, missing,
                   how={t: t2v[t] for t in genes if t2v[t] != t},
                   extra={"k": k, "n_train_perturbations": len(perts),
                          "explained_variance_ratio_k": float((s[:k] ** 2).sum() / (s ** 2).sum())})


# --------------------------------------------------------------------------- GenePT
class _SafeUnpickler(pickle.Unpickler):
    """The GenePT file is a dict {symbol: list/ndarray of floats}; refuse anything else."""

    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
               ("numpy._core.multiarray", "scalar"), ("builtins", "list"), ("builtins", "dict")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name} in GenePT pickle")


def load_genept(cfg) -> dict[str, np.ndarray]:
    """GenePT vectors by symbol. The pickle unpacks to Python lists (several GB), so the first call writes a
    float32 cache (data/interim/genept_ada.npz) and later calls read that instead."""
    import zipfile

    cache = path(cfg, "qc_h5ad").parent / "genept_ada.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return dict(zip(z["genes"].tolist(), z["emb"]))

    member = cfg["embeddings"]["genept"]["member"]
    extracted = path(cfg, "genept_zip").parent / "extracted" / member
    if not extracted.exists():
        with zipfile.ZipFile(path(cfg, "genept_zip")) as z:
            name = next(n for n in z.namelist() if n.endswith(member))
            extracted.parent.mkdir(parents=True, exist_ok=True)
            extracted.write_bytes(z.read(name))
    keys, emb = _stream_genept_pickle(extracted)
    np.savez(cache, genes=np.array(keys, dtype=str), emb=emb)
    return dict(zip(keys, emb))


def _stream_genept_pickle(fp):
    """Read the {symbol: [float, ...]} pickle opcode by opcode. Nothing is constructed or executed, and memory stays
    at the size of the float32 matrix instead of several GB of Python float objects.

    Aliases share one list object in the pickle (a memo reference), so 93,800 symbols map onto ~33k distinct vectors;
    each symbol is returned with the row of its shared vector.
    """
    import pickletools

    rows, cur, row_of, memo = [], None, {}, []
    key, last = None, None
    with open(fp, "rb") as f:
        for op, arg, _ in pickletools.genops(f):
            n = op.name
            if n in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE"):
                key, last = str(arg), ("str", str(arg))
            elif n == "EMPTY_LIST":
                cur = []
                rows.append(cur)
                row_of[key] = len(rows) - 1
                last = ("list", len(rows) - 1)
            elif n == "BINFLOAT":
                cur.append(arg)
            elif n == "MEMOIZE":
                memo.append(last)
            elif n in ("BINGET", "LONG_BINGET"):
                last = memo[arg]
                if last[0] == "list":
                    row_of[key] = last[1]
            elif n in ("GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX"):
                raise pickle.UnpicklingError(f"unexpected opcode {n} in GenePT pickle")
    assert len({len(r) for r in rows}) == 1, "GenePT vectors differ in length"
    emb = np.asarray(rows, dtype=np.float32)
    keys = list(row_of)
    return keys, emb[[row_of[k] for k in keys]]


def match_symbols(targets, vocab: set, resolver: SymbolResolver, t2v: dict):
    """target -> key in vocab, trying the guide symbol, its var name, the current HGNC symbol and synonyms."""
    out, how = {}, {}
    for t in targets:
        cands = [t] + ([t2v[t]] if t2v.get(t) else []) + resolver.candidates(t)
        for i, c in enumerate(dict.fromkeys(cands)):
            if c in vocab:
                out[t] = c
                if c != t:
                    how[t] = c
                break
    return out, how


def build_genept(cfg, splits, adata=None, **_):
    d = load_genept(cfg)
    dims = {v.shape for v in d.values()}
    assert len(dims) == 1, dims
    resolver = SymbolResolver(path(cfg, "hgnc"))
    t2v = dict((adata if adata is not None else ad.read_h5ad(path(cfg, "processed_h5ad"), backed="r")).uns["target_to_var"])
    targets = all_targets(splits)
    m, how = match_symbols(targets, set(d), resolver, t2v)
    genes = [t for t in targets if t in m]
    emb = np.vstack([d[m[t]] for t in genes])
    save_embedding(cfg, "genept", genes, emb, [t for t in targets if t not in m], how=how,
                   extra={"source": cfg["embeddings"]["genept"]["member"], "vocab_size": len(d),
                          "row_norm_mean": float(np.linalg.norm(emb, axis=1).mean())})


# --------------------------------------------------------------------------- STRING node2vec
def string_graph(cfg):
    """Undirected STRING v12 graph with combined_score >= threshold; nodes are ENSP ids."""
    s = cfg["embeddings"]["string"]
    links = pd.read_csv(path(cfg, "string_links"), sep=" ", dtype={"combined_score": np.int32})
    n_raw = len(links)
    links = links[links.combined_score >= s["min_combined_score"]]
    a, b = links.protein1.to_numpy(), links.protein2.to_numpy()
    und = links[a < b]  # file lists both directions; keep one
    edges = pd.DataFrame({"u": und.protein1.str.replace("9606.", "", regex=False),
                          "v": und.protein2.str.replace("9606.", "", regex=False),
                          "w": und.combined_score / 1000.0})
    meta = {"n_directed_rows_raw": int(n_raw), "n_directed_rows_kept": int(len(links)),
            "n_undirected_edges": int(len(edges)), "n_nodes": int(pd.unique(edges[["u", "v"]].values.ravel()).size),
            "min_combined_score": s["min_combined_score"], "edge_weight": "combined_score / 1000"}
    return edges, meta


def string_symbol_maps(cfg):
    info = pd.read_csv(path(cfg, "string_info"), sep="\t")
    info.columns = [c.lstrip("#") for c in info.columns]
    info["id"] = info.string_protein_id.str.replace("9606.", "", regex=False)
    pref = dict(zip(info.preferred_name, info.id))
    al = pd.read_csv(path(cfg, "string_aliases"), sep="\t")
    al.columns = [c.lstrip("#") for c in al.columns]
    al["id"] = al.string_protein_id.str.replace("9606.", "", regex=False)
    al = al.drop_duplicates(["alias", "id"])
    uniq = al.groupby("alias")["id"].nunique()
    al = al[al.alias.isin(uniq.index[uniq == 1])]
    alias = dict(zip(al.alias, al.id))
    return pref, alias


def node2vec_embed(edges: pd.DataFrame, s: dict, seed: int, tmp_edg):
    """node2vec: PecanPy (SparseOTF) walks on the weighted undirected graph + gensim skip-gram."""
    import numba
    from gensim.models import Word2Vec
    from pecanpy import pecanpy as pp

    numba.set_num_threads(s["workers"])  # PecanPy walks are only reproducible on one numba thread
    edges = edges.sort_values(["u", "v"])
    edges[["u", "v", "w"]].to_csv(tmp_edg, sep="\t", header=False, index=False)
    g = pp.SparseOTF(p=s["p"], q=s["q"], workers=1, verbose=False, random_state=seed)
    g.read_edg(str(tmp_edg), weighted=True, directed=False, delimiter="\t")
    nodes = np.array(sorted(g.nodes))
    walks = g.simulate_walks(num_walks=s["num_walks"], walk_length=s["walk_length"])
    w2v = Word2Vec(walks, vector_size=s["dim"], window=s["window"], min_count=0, sg=1, hs=0, negative=5,
                   workers=s["workers"], epochs=s["epochs"], seed=seed)
    E = np.vstack([w2v.wv[n] for n in nodes])
    return nodes, E, walks


def build_string_n2v(cfg, splits, adata=None, **_):
    s = cfg["embeddings"]["string"]
    seed_everything(cfg["seed"])
    edges, meta = string_graph(cfg)
    edge_out = ROOT / "data" / "interim" / f"string_v12_edges_ge{s['min_combined_score']}.tsv.gz"
    edges.to_csv(edge_out, sep="\t", index=False, compression="gzip")
    log.info("STRING graph: %s", meta)
    nodes, E, _ = node2vec_embed(edges, s, cfg["seed"], edge_out.with_suffix("").with_suffix(".edg"))
    np.savez(path(cfg, "embeddings_dir") / "string_n2v_full.npz", emb=E.astype(np.float32), nodes=nodes.astype(str))

    pref, alias = string_symbol_maps(cfg)
    resolver = SymbolResolver(path(cfg, "hgnc"))
    t2v = dict((adata if adata is not None else ad.read_h5ad(path(cfg, "processed_h5ad"), backed="r")).uns["target_to_var"])
    in_graph = set(nodes)
    row = {n: i for i, n in enumerate(nodes)}
    targets = all_targets(splits)
    genes, rows, how, missing, reasons = [], [], {}, [], {}
    for t in targets:
        cands = list(dict.fromkeys([t] + ([t2v[t]] if t2v.get(t) else []) + resolver.candidates(t)))
        ensp = next((pref[c] for c in cands if c in pref), None)
        via = "preferred_name"
        if ensp is None:
            ensp, via = next((alias[c] for c in cands if c in alias), None), "alias"
        if ensp is None:
            missing.append(t); reasons[t] = "no STRING protein"
            continue
        if ensp not in in_graph:
            missing.append(t); reasons[t] = f"{ensp} has no edge with score >= {s['min_combined_score']}"
            continue
        genes.append(t); rows.append(row[ensp])
        if via != "preferred_name" or t not in pref:
            how[t] = f"{via}:{ensp}"
    save_embedding(cfg, "string_n2v", genes, E[rows], missing, how=how,
                   extra={"graph": meta, "node2vec": {k: s[k] for k in ("dim", "walk_length", "num_walks", "window",
                                                                         "p", "q", "epochs", "workers")},
                          "seed": cfg["seed"], "missing_reason": reasons,
                          "edge_list": str(edge_out.relative_to(ROOT))})


# --------------------------------------------------------------------------- every measured gene
def string_ensembl_map(cfg) -> dict[str, str]:
    """ENSG -> ENSP from STRING's alias file (source Ensembl_gene); genes with several proteins are dropped."""
    al = pd.read_csv(path(cfg, "string_aliases"), sep="\t")
    al.columns = [c.lstrip("#") for c in al.columns]
    al = al[al.source == "Ensembl_gene"]
    al = al.assign(id=al.string_protein_id.str.replace("9606.", "", regex=False)).drop_duplicates(["alias", "id"])
    n = al.groupby("alias")["id"].nunique()
    al = al[al.alias.isin(n.index[n == 1])]
    return dict(zip(al.alias, al.id))


def build_all_genes(cfg, splits, **_):
    """GenePT and STRING node2vec vectors for every gene measured in the Norman K562 matrix (no imputation).

    Writes
        embeddings/gene_universe.csv        one row per measured gene: ids, detection, flags, how each source matched
        embeddings/genept_all_genes.npz     emb (n_covered x 1536), genes (matrix symbols), ensembl
        embeddings/string_n2v_all_genes.npz emb (n_covered x 64),   genes, ensembl
    Matching: GenePT by symbol (matrix symbol, then current HGNC symbol for the Ensembl id, then HGNC synonyms);
    STRING by Ensembl gene id first, then by symbol / unique alias. Detection counts come from QC'd cells only.
    """
    import scipy.sparse as sp

    qc = ad.read_h5ad(path(cfg, "qc_h5ad"), backed="r")
    var = qc.var.copy()
    train = qc.obs_names.isin(splits["cells"]["train"])
    det = np.zeros(qc.n_vars, dtype=np.int64)
    det_tr = np.zeros(qc.n_vars, dtype=np.int64)
    for i in range(0, qc.n_obs, 10000):  # chunked: the full count matrix does not need to be in memory
        X = sp.csr_matrix(qc.X[i:i + 10000])
        X.data = (X.data > 0).astype(np.int8)
        det += np.asarray(X.sum(0)).ravel()
        det_tr += np.asarray(X[train[i:i + 10000]].sum(0)).ravel()
    hvg = set(ad.read_h5ad(path(cfg, "processed_h5ad"), backed="r").var_names)
    t2v = dict(qc.uns["target_to_var"])
    v2t = {}
    for t, v in t2v.items():
        v2t.setdefault(v, []).append(t)
    del X

    resolver = SymbolResolver(path(cfg, "hgnc"))
    ens2sym = {e: s for s, e in resolver.ensembl.items() if e}
    genept = load_genept(cfg)
    pref, alias = string_symbol_maps(cfg)
    ens2ensp = string_ensembl_map(cfg)
    z = np.load(path(cfg, "embeddings_dir") / "string_n2v_full.npz")
    n2v_emb, n2v_row = z["emb"], {n: i for i, n in enumerate(z["nodes"])}  # read once: NpzFile re-reads on every access

    rows, gp_rows, st_rows = [], [], []
    min_cells = cfg["data"]["min_cells_per_gene"]
    for i, (g, e) in enumerate(zip(var.index, var["ensemble_id"])):
        cur = ens2sym.get(e) or resolver.to_current(g)[0]
        cands = list(dict.fromkeys([g] + ([cur] if cur else []) + v2t.get(g, []) + resolver.candidates(cur or g)))
        gk = next((c for c in cands if c in genept), None)
        ensp, how = ens2ensp.get(e), "ensembl_gene"
        if ensp is None:
            ensp, how = next((pref[c] for c in cands if c in pref), None), "preferred_name"
        if ensp is None:
            ensp, how = next((alias[c] for c in cands if c in alias), None), "alias"
        in_graph = ensp in n2v_row if ensp else False
        rows.append({"gene": g, "ensembl_id": e, "hgnc_symbol": cur or "", "n_cells": int(det[i]),
                     "n_cells_train": int(det_tr[i]), "expressed": bool(det_tr[i] >= min_cells), "in_model_genes": g in hvg,
                     "is_target": g in v2t, "target_symbol": ",".join(v2t.get(g, [])),
                     "genept_key": gk or "", "genept_how": ("symbol" if gk == g else "hgnc" if gk else "missing"),
                     "string_protein": ensp or "", "string_how": how if ensp else "missing", "string_in_graph": bool(in_graph)})
        if gk:
            gp_rows.append((g, e, genept[gk]))
        if in_graph:
            st_rows.append((g, e, n2v_emb[n2v_row[ensp]]))
    uni = pd.DataFrame(rows)
    out_dir = path(cfg, "embeddings_dir")
    uni.to_csv(out_dir / "gene_universe.csv", index=False)
    for name, rr in (("genept_all_genes", gp_rows), ("string_n2v_all_genes", st_rows)):
        np.savez(out_dir / f"{name}.npz", emb=np.vstack([r[2] for r in rr]).astype(np.float32),
                 genes=np.array([r[0] for r in rr], dtype=str), ensembl=np.array([r[1] for r in rr], dtype=str))

    tiers = {"all measured": np.ones(len(uni), bool), f"expressed (>= {min_cells} training cells)": uni.expressed.to_numpy(),
             "model genes (5,028)": uni.in_model_genes.to_numpy(), "perturbed targets": uni.is_target.to_numpy()}
    cov = {k: {"n_genes": int(m.sum()), "genept": int((uni.genept_how[m] != "missing").sum()),
               "string_protein": int((uni.string_how[m] != "missing").sum()),
               "string_n2v": int(uni.string_in_graph[m].sum())} for k, m in tiers.items()}
    write_json({"tiers": cov, "genept_match": uni.genept_how.value_counts().to_dict(),
                "string_match": uni.string_how.value_counts().to_dict(),
                "string_graph": f"STRING v12 combined_score >= {cfg['embeddings']['string']['min_combined_score']}"},
               out_dir / "all_genes_coverage.json")
    for k, v in cov.items():
        log.info("%-36s %6d genes | GenePT %6d | STRING protein %6d | node2vec %6d", k, v["n_genes"], v["genept"],
                 v["string_protein"], v["string_n2v"])


BUILDERS = {"onehot": build_onehot, "pca_loadings": build_pca_loadings, "genept": build_genept,
            "string_n2v": build_string_n2v, "all_genes": build_all_genes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("methods", nargs="*", default=METHODS)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    splits = read_json(path(cfg, "splits"))
    adata = ad.read_h5ad(path(cfg, "processed_h5ad")) if "pca_loadings" in a.methods else None
    for m in a.methods:
        BUILDERS[m](cfg, splits, adata=adata)


if __name__ == "__main__":
    main()
