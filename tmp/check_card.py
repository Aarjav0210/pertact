"""Check a dataset CARD.md against the h5ad next to it, and write the result back into the card.

A card is one markdown table: | 字段 | 值 | 状态 |. The text in backticks in 值 is what gets checked; the rest is
prose. The script fills 状态 and rewrites the "## 证据" section below the table. It reads obs only (backed mode),
never the expression matrix.

    python -I tmp/check_card.py <dataset_dir>/raw/scperturb/CARD.md
    python -I tmp/check_card.py <dataset_dir>/processed/pertact/CARD.md
    python -I tmp/check_card.py <dataset_dir>/raw/s3/CARD.md        # Marson: obs read remotely, see remote_obs()

Status marks:  ✓ checked against the data   📄 from the paper / external source, the data cannot show it
               ? not checked yet              ✗ the data disagrees with the card
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

CONTROL_PREFIX = "NegCtrl"


# --------------------------------------------------------------------------- card io
def read_card(p: Path):
    text = p.read_text()
    meta = dict(re.findall(r"^(\w+):\s*(.+)$", text.split("|", 1)[0], flags=re.M))
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] not in ("字段", "") and not set(cells[0]) <= set("-:"):
            rows[cells[0]] = {"value": cells[1], "spec": re.findall(r"`([^`]+)`", cells[1]), "status": cells[2]}
    return text, meta, rows


def write_card(p: Path, text: str, results: dict, header: str):
    out = []
    for line in text.split("\n## 证据")[0].rstrip().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] in results:
            line = f"| {cells[0]} | {cells[1]} | {results[cells[0]][0]} |"
        out.append(line)
    out += ["", "## 证据", "", header, ""]
    out += [f"- **{f}** {mark}: {ev}" for f, (mark, ev) in results.items()]
    p.write_text("\n".join(out) + "\n")


# --------------------------------------------------------------------------- remote obs (files too large to download)
def remote_obs(url: str, cols: list[str]) -> pd.DataFrame:
    """Selected obs columns of a remote h5ad, read with HTTP range requests: only those columns are transferred."""
    import fsspec
    import h5py

    out = {}
    with fsspec.open(url, "rb", block_size=4 * 2**20, cache_type="readahead") as fh, h5py.File(fh, "r") as f:
        for c in cols:
            g = f["obs"][c]
            if isinstance(g, h5py.Group):
                cats = g["categories"][:].astype(str)
                out[c] = pd.Categorical.from_codes(g["codes"][:].astype(np.int32), categories=cats)
            else:
                out[c] = g[:]
    return pd.DataFrame(out)


def card_files(meta) -> list[tuple[str, str, str]]:
    """(donor, condition, file name) for a card whose context is spread over one file per donor x condition."""
    return [(d, c, meta["file_pattern"].format(donor=d, condition=c))
            for d in meta["donors"].split(",") for c in meta["conditions"].split(",")]


def load_remote(meta, cols) -> pd.DataFrame:
    parts = []
    for d, c, fn in card_files(meta):
        df = remote_obs(meta["url"].rstrip("/") + "/" + fn, cols)
        df["_donor"], df["_condition"], df["_file"] = d, c, fn
        parts.append(df)
        print(f"  read {fn}: {len(df):,} cells", flush=True)
    obs = pd.concat(parts, ignore_index=True)
    for c in cols + ["_donor", "_condition", "_file"]:
        if obs[c].dtype == object:
            obs[c] = obs[c].astype("category")
    return obs


# --------------------------------------------------------------------------- checks (one per layer x field)
def guide_genes(gid: str) -> tuple[str, ...] | None:
    """'KLF1_NegCtrl0;KLF1_NegCtrl0' -> ('KLF1',); None if the guides disagree. Same rule as pertact.data."""
    sets = set()
    for c in str(gid).split(";"):
        toks = [t for t in c.split("_") if t and not t.isdigit()]
        sets.add(tuple(sorted(t for t in toks if not t.startswith(CONTROL_PREFIX))))
    return next(iter(sets)) if len(sets) == 1 else None


def col_check(obs, col):
    if col not in obs:
        return "✗", f"obs has no column `{col}`"
    return None


def raw_units(obs, spec, **_):
    col = spec[0]
    if (bad := col_check(obs, col)):
        return bad
    g = obs[col].astype(str).astype(object).map(guide_genes)
    n = g.map(lambda t: -1 if t is None else len(t)).value_counts().to_dict()
    ex = next(t for t in g if t and len(t) == 2)
    return "✓", (f"parsed `{col}`: {n.get(1, 0)} single-gene cells, {n.get(2, 0)} two-gene, {n.get(0, 0)} control, "
                 f"{n.get(-1, 0)} with conflicting guides, {sum(v for k, v in n.items() if k > 2)} with >2 genes "
                 f"(the last two are dropped by QC). Example {ex}")


def raw_control(obs, spec, **_):
    g = obs["guide_id"].astype(str).astype(object).map(guide_genes)
    ctrl = g.map(lambda t: t == ())
    toks = {t for gid in obs.loc[ctrl, "guide_id"] for c in str(gid).split(";") for t in c.split("_") if t}
    ok = ctrl.sum() > 0 and all(t.startswith(spec[0]) or t.isdigit() for t in toks)
    return ("✓" if ok else "✗"), f"{int(ctrl.sum())} cells whose guides are all `{spec[0]}*` ({len(toks)} distinct tokens)"


def literal_col(col):
    def f(obs, spec, **_):
        want = spec[0].split("=", 1)[-1]
        if (bad := col_check(obs, col)):
            return bad
        vals = obs[col].astype(str).value_counts().to_dict()
        return ("✓" if set(vals) == {want} else "✗"), f"`{col}` values: {vals}"
    return f


def batch(obs, spec, **_):
    col = spec[0]
    if (bad := col_check(obs, col)):
        return bad
    vc = obs[col].astype(str).value_counts().sort_index().to_dict()
    ev = f"`{col}`: {len(vc)} levels, cells per level {vc}"
    if "split" in obs and "condition" in obs:
        c = obs[obs["condition"].astype(str) == "control"]
        miss = [(s, b) for s in ("train", "val", "test") for b in vc
                if not ((c["split"] == s) & (c[col].astype(str) == b)).any()]
        ev += "; every split has controls in every level" if not miss else f"; missing controls for {miss}"
        return ("✓" if not miss else "✗"), ev
    return "✓", ev


def modality_raw(obs, spec, **_):
    have = obs["perturbation_type"].astype(str).unique().tolist() if "perturbation_type" in obs else []
    return "📄", (f"the data only says perturbation_type={have}; activation vs interference is not in obs. "
                  f"`{spec[0]}` is from the paper (Norman et al. 2019, Science)")


def proc_units(obs, spec, **_):
    col, sep = spec[0], spec[1] if len(spec) > 1 else "+"
    if (bad := col_check(obs, col)):
        return bad
    cond = obs[col].astype(str)
    n = cond.map(lambda c: 0 if c == "control" else len(c.split(sep)))
    agree = (n == obs["n_targets"].astype(int)).all()
    t1 = (cond[n >= 1].str.split(sep).str[0] == obs.loc[n >= 1, "target_1"].astype(str)).all()
    k = cond[n > 0].groupby(n[n > 0]).nunique().to_dict()
    ex = cond[n == 2].iloc[0]
    return ("✓" if agree and t1 else "✗"), (f"`{col}` split on `{sep}` gives n_targets for every cell ({agree}); "
                                           f"target_1 agrees ({t1}); distinct conditions by size {k}; example {ex!r}")


def proc_control(obs, spec, **_):
    c = obs["condition"].astype(str) == spec[0]
    ok = (c == (obs["n_targets"] == 0)).all() and (c == ~obs["perturbed"].astype(bool)).all()
    return ("✓" if ok else "✗"), f"{int(c.sum())} cells with condition == `{spec[0]}`; same set as n_targets == 0 and perturbed == False ({ok})"


def inherited(field):
    """No obs column in the processed file: take the status the raw card recorded for this field."""
    def f(obs, spec, parent=None, **_):
        if parent is None:
            return "?", "processed card has no `derived_from` line"
        row = parent.get(field)
        val = lambda sp: sp[0].split("=", 1)[-1] if sp else None   # `cell_line=K562` and `K562` name the same value
        if row is None or val(row["spec"]) != val(spec):
            return "✗", f"raw card says {row and row['value']!r}, processed card says `{spec[0]}`"
        return row["status"], f"`{spec[0]}` (no obs column), inherited from the raw card, where it is {row['status']}"
    return f


def proc_split(obs, spec, **_):
    col = spec[0]
    if (bad := col_check(obs, col)):
        return bad
    vc = obs[col].astype(str).value_counts().to_dict()
    dbl = obs["n_targets"] == 2
    leak = int((dbl & obs[col].isin(["train", "val"])).sum())
    return ("✓" if leak == 0 else "✗"), f"`{col}` values {vc}; two-gene cells in train/val: {leak}"


# --------------------------------------------------------------------------- Marson CD4 (cell-level, one file per donor x condition)
MARSON_COLS = ["guide_id", "guide_type", "guide_group", "perturbed_gene_name", "lane_id", "low_quality"]


def m_units(obs, spec, **_):
    col = spec[0]
    single = obs["guide_group"] == "targeting single sgRNA"
    tgt = single & (obs["guide_type"] == "targeting")
    names = obs.loc[tgt, col]
    multi_gid = int((obs.loc[single, "guide_id"] == "multi-guide").sum())
    groups = obs["guide_group"].value_counts(dropna=False).to_dict()
    ok = names.notna().all() and multi_gid == 0
    return ("✓" if ok else "✗"), (f"guide_group counts {groups}. Single-guide targeting cells: {int(tgt.sum()):,}, "
                                 f"{names.nunique():,} distinct `{col}`, {int(names.isna().sum())} without a gene name; "
                                 f"'multi-guide' guide_id inside the single group: {multi_gid}. One gene per cell: "
                                 f"there are no combinations in this dataset")


def m_control(obs, spec, **_):
    want = spec[0].split("=", 1)[-1]
    c = obs["guide_type"] == want
    gids = obs.loc[c, "guide_id"].astype(str)
    prefix_ok = gids.str.startswith("NTC").all()
    by_group = obs.loc[c, "guide_group"].value_counts().to_dict()
    ok = c.sum() > 0 and prefix_ok
    return ("✓" if ok else "✗"), (f"{int(c.sum()):,} cells with guide_type == `{want}`; all their guide_id start with "
                                 f"NTC ({prefix_ok}); {gids.nunique()} distinct NTC guides; their guide_group: {by_group} "
                                 f"(non-targeting singles sit in 'targeting single sgRNA', so select controls by "
                                 f"guide_type, not by guide_group)")


def m_context(obs, spec, card_dir=None, meta=None, **_):
    files = {(d, c) for d, c, _ in card_files(meta)}
    sm = pd.read_csv(card_dir / "suppl_tables" / "sample_metadata.suppl_table.csv")
    sm_keys = {tuple(s.split("_")[2:4]) for s in sm["cell_sample_id"]}
    n = obs.groupby(["_donor", "_condition"], observed=True).size()
    ok = files == sm_keys and (n > 0).all() and len(n) == len(files)
    return ("✓" if ok else "✗"), (f"{len(files)} files = donors {sorted({d for d, _ in files})} x conditions "
                                 f"{sorted({c for _, c in files})}; same set as sample_metadata ({files == sm_keys}); "
                                 f"no obs column, taken from the file name. Cells per context (M): "
                                 f"{ {f'{d}_{c}': round(v / 1e6, 2) for (d, c), v in n.items()} }")


def m_batch(obs, spec, card_dir=None, **_):
    col = spec[0]
    files_per_lane = obs.groupby(col, observed=True)["_file"].nunique()
    shared = files_per_lane[files_per_lane > 1]
    sm = pd.read_csv(card_dir / "suppl_tables" / "sample_metadata.suppl_table.csv")
    run_of = {tuple(s.split("_")[2:4]): r.split("_")[-1] for s, r in zip(sm["cell_sample_id"], sm["10xrun_id"])}
    lane_run = obs.groupby(["_donor", "_condition"], observed=True)[col].agg(
        lambda x: sorted({str(v).split("_")[1][:2] for v in x.dropna().unique()}))
    agree = all(lane_run[k] == [run_of[k]] for k in lane_run.index)
    r1 = sorted(f"{d}_{c}" for (d, c), r in run_of.items() if r == "R1")
    ok = agree
    return ("✓" if ok else "✗"), (f"`{col}`: {obs[col].nunique()} lanes, {obs.groupby('_file', observed=True)[col].nunique().min()}"
                                 f"–{obs.groupby('_file', observed=True)[col].nunique().max()} per file; "
                                 + ("each lane id belongs to one file. " if len(shared) == 0 else
                                    f"{len(shared)} lane ids appear in >1 file, so a batch is `_file` + lane_id. ")
                                 + f"Lane prefix "
                                 f"agrees with the 10xrun_id of sample_metadata ({agree}). Confound: run R1 holds only {r1}; "
                                 f"run R2 holds the rest, including every Stim48hr")


def m_modality(obs, spec, **_):
    return "📄", (f"`{spec[0]}`: obs has guide_type (targeting / non-targeting) but not the CRISPR mode; from the paper "
                  f"(Zhu, Dann et al. 2026, Cell, genome-scale CRISPRi)")


# --------------------------------------------------------------------------- Marson CD4 pseudobulk (one row per guide x donor x condition)
def _sample_meta(card_dir):
    sm = pd.read_csv(card_dir / "suppl_tables" / "sample_metadata.suppl_table.csv")
    sm["D"] = sm["cell_sample_id"].str.split("_").str[2]
    sm["run"] = sm["10xrun_id"].str.split("_").str[-1]
    return sm


def _cell_level_counts(card_dir):
    """Single-guide targeting / NTC cell counts recorded in the cell-level card's evidence."""
    ev = (card_dir / "CARD.md").read_text()
    tgt = re.search(r"Single-guide targeting cells: ([\d,]+)", ev)
    ntc = re.search(r"([\d,]+) cells with guide_type == `non-targeting`", ev)
    num = lambda m: int(m.group(1).replace(",", "")) if m else None
    return num(tgt), num(ntc)


def pb_units(obs, spec, card_dir=None, **_):
    col = spec[0]
    tgt = obs["guide_type"] == "targeting"
    key = ["guide_id", "donor_id", "culture_condition"]
    dup = int(obs.duplicated(key).sum())
    g2gene = obs.loc[tgt].groupby("guide_id", observed=True)[col].nunique()
    multi_gene_guides = int((g2gene > 1).sum())
    n_tgt = int(obs.loc[tgt, "n_cells"].sum())
    want_tgt, _ = _cell_level_counts(card_dir)
    keep = {c: int(obs[c].astype(bool).sum()) for c in obs.columns if c.startswith("keep_")}
    genes_de = obs.loc[tgt & obs["keep_for_DE"].astype(bool), col].nunique()
    ok = dup == 0 and multi_gene_guides == 0 and obs.loc[tgt, col].notna().all()
    return ("✓" if ok else "✗"), (
        f"{obs.shape[0]:,} rows, one per {' x '.join(key)} (duplicate keys: {dup}). Targeting rows: {int(tgt.sum()):,}, "
        f"{obs.loc[tgt, 'guide_id'].nunique():,} guides on {obs.loc[tgt, col].nunique():,} distinct `{col}`; "
        f"guides mapped to >1 gene: {multi_gene_guides}. Sum of n_cells over targeting rows {n_tgt:,} vs "
        f"{want_tgt:,} single-guide targeting cells in the cell-level files ({n_tgt / want_tgt:.1%}). "
        f"QC flags (rows True): {keep}; genes with ≥1 keep_for_DE row: {genes_de:,}")


def pb_control(obs, spec, card_dir=None, **_):
    want = spec[0].split("=", 1)[-1]
    c = obs["guide_type"] == want
    gids = obs.loc[c, "guide_id"].astype(str)
    prefix_ok = gids.str.startswith("NTC").all()
    gene = obs.loc[c, "perturbed_gene_name"].astype(str).value_counts().head(3).to_dict()
    n = int(obs.loc[c, "n_cells"].sum())
    _, want_ntc = _cell_level_counts(card_dir)
    ok = c.sum() > 0 and prefix_ok
    return ("✓" if ok else "✗"), (
        f"{int(c.sum()):,} rows with guide_type == `{want}`, {gids.nunique()} distinct guides, all start with NTC "
        f"({prefix_ok}); their perturbed_gene_name: {gene}. Sum of n_cells {n:,} vs {want_ntc:,} NTC cells in the "
        f"cell-level files ({n / want_ntc:.1%})")


def pb_context(obs, spec, card_dir=None, **_):
    sm = _sample_meta(card_dir)
    ce2d = dict(zip(sm["donor_id"], sm["D"]))
    one_to_one = sm.groupby("donor_id")["D"].nunique().max() == 1 and sm.groupby("D")["donor_id"].nunique().max() == 1
    n = obs.groupby(["donor_id", "culture_condition"], observed=True).size()
    pairs = {(ce2d.get(d), c) for d, c in n.index}
    sm_pairs = set(zip(sm["D"], sm["culture_condition"]))
    ok = one_to_one and pairs == sm_pairs and None not in {p[0] for p in pairs}
    return ("✓" if ok else "✗"), (
        f"{len(n)} donor_id x culture_condition pairs; CE id → file donor via sample_metadata "
        f"{dict(sorted(ce2d.items(), key=lambda kv: kv[1]))} (one-to-one: {one_to_one}); mapped pairs equal the 12 "
        f"cell-level files ({pairs == sm_pairs}). Rows per context (k): "
        f"{ {f'{ce2d[d]}_{c}': round(v / 1e3, 1) for (d, c), v in n.items()} }")


def pb_batch(obs, spec, card_dir=None, **_):
    col = spec[0]
    sm = _sample_meta(card_dir)
    run_of = {(d, c): r for d, c, r in zip(sm["donor_id"], sm["culture_condition"], sm["run"])}
    got = obs.groupby(["donor_id", "culture_condition"], observed=True)[col].agg(lambda x: sorted({str(v).split("_")[-1] for v in x.unique()}))
    agree = all(got[k] == [run_of[k]] for k in got.index)
    ce2d = dict(zip(sm["donor_id"], sm["D"]))
    r1 = sorted(f"{ce2d[d]}_{c}" for (d, c), r in run_of.items() if r == "R1")
    lane = [c for c in obs.columns if "lane" in c.lower()]
    return ("✓" if agree else "✗"), (
        f"`{col}`: {obs[col].astype(str).value_counts().to_dict()}; each donor x condition sits in one run, matching "
        f"sample_metadata ({agree}). Lane columns in obs: {lane or 'none'} (lanes are summed away, so the run is the "
        f"finest batch left). Same confound as the cell-level data: R1 holds only {r1}; every Stim48hr is in R2")


# --------------------------------------------------------------------------- Jin whole brain (in vivo KO, one file per 10x channel)
def _targets(obs, col):
    gt = obs[col].astype(str)
    return gt, gt.str.contains(r"\|", regex=True)


def j_units(obs, spec, **_):
    col = spec[0]
    gt, multi = _targets(obs, col)
    neg, ntc = gt == "Negative", gt == "Non_target"
    single = ~multi & ~neg & ~ntc
    ng = obs["num_guides"].astype(int)
    agree = bool((ng[neg] == 0).all() and (ng[single | ntc] >= 1).all() and (ng[multi] >= 2).all())
    k = gt[multi].str.count(r"\|") + 1
    qc = obs["passes_qc"].astype(str) == "True"
    ok = agree and single.sum() > 0
    return ("✓" if ok else "✗"), (
        f"`{col}`: {int(single.sum()):,} single-gene cells on {gt[single].nunique():,} genes, {int(multi.sum()):,} cells "
        f"with ≥2 genes (`|`-joined; k=2: {int((k == 2).sum()):,}, k≥3: {int((k >= 3).sum()):,}), {int(neg.sum()):,} "
        f"`Negative` (no guide detected), {int(ntc.sum()):,} `Non_target`. num_guides agrees (0 for Negative, ≥1 for "
        f"single/NTC, ≥2 for multi): {agree}. Multi-gene cells are co-infections (MOI>1), not designed combinations. "
        f"passes_qc=True: {int(qc.sum()):,}; single-gene cells passing QC: {int((single & qc).sum()):,} on "
        f"{gt[single & qc].nunique():,} genes")


def j_control(obs, spec, **_):
    want = spec[0].split("=", 1)[-1]
    gt, multi = _targets(obs, "gene_target")
    c = gt == want
    guides = obs.loc[c, "guide_call"].astype(str)
    one = bool((~guides.str.contains(r"\|", regex=True)).all())
    in_multi = int(gt[multi].str.split("|").apply(lambda xs: want in xs).sum())
    qc = obs["passes_qc"].astype(str) == "True"
    return ("✓" if c.sum() > 0 and one else "✗"), (
        f"{int(c.sum()):,} cells with gene_target == `{want}` ({int((c & qc).sum()):,} pass QC), {guides.nunique()} "
        f"distinct guides, each cell one guide: {one}; examples {sorted(guides.unique())[:3]}. Non_target also appears "
        f"inside {in_multi:,} multi-gene cells; `Negative` (no guide) is not a control")


def j_context(obs, spec, **_):
    cols = [s for s in spec if s in obs.columns]
    missing = [s for s in spec if s not in obs.columns]
    n = {c: obs[c].nunique() for c in cols}
    per_mouse = obs.groupby("source", observed=True)["predicted_group"].nunique()
    sex = obs.groupby("source", observed=True)["sex"].nunique().max()
    return ("✓" if not missing else "✗"), (
        f"columns {n}" + (f"; missing {missing}" if missing else "") + f". {obs['source'].nunique()} mice "
        f"({obs.groupby('source', observed=True)['sex'].first().value_counts().to_dict()}; one sex per mouse: {sex == 1}), "
        f"{obs['sample_label'].nunique()} samples (hemispheres); cell-type groups per mouse "
        f"{per_mouse.min()}–{per_mouse.max()}. Top groups: "
        f"{ {k: round(v / 1e6, 2) for k, v in obs['predicted_group'].astype(str).value_counts().head(4).items()} } (M cells). "
        f"Regions: {obs['region_level1'].astype(str).value_counts().to_dict()}")


def j_batch(obs, spec, **_):
    b, ch = spec[0], spec[1]
    ch_per_file = obs.groupby("_file", observed=True)[ch].nunique()
    file_is_ch = bool((ch_per_file == 1).all() and (obs[ch].astype(str) == obs["_file"]).all())
    nested = bool((obs.groupby(ch, observed=True)[b].nunique() == 1).all())
    mice_per_b = obs.groupby(b, observed=True)["source"].nunique().to_dict()
    pools_per_mouse = obs.groupby("source", observed=True)[b].nunique()
    pools_per_sample = int(obs.groupby("sample_label", observed=True)[b].nunique().max())
    ok = file_is_ch and nested and pools_per_sample == 1
    return ("✓" if ok else "✗"), (
        f"`{b}`: {obs[b].nunique()} pools {obs[b].astype(str).value_counts().to_dict()}; `{ch}`: {obs[ch].nunique()} "
        f"10x channels, one per file ({file_is_ch}), each inside one pool ({nested}). Mice per pool {mice_per_b}; "
        f"each hemisphere sample (`sample_label`) sits in {pools_per_sample} pool; {int((pools_per_mouse == 1).sum())} "
        f"mice have both hemispheres in one pool, {int((pools_per_mouse > 1).sum())} have L and R in different pools, "
        f"so mouse is only partly nested in pool")


def j_modality(obs, spec, **_):
    return "📄", (f"`{spec[0]}`: obs has gene_target / guide_call but not the CRISPR mode; from the paper "
                  f"(Shi … Jin 2026, bioRxiv; the HF README describes gene_target as the gene(s) knocked out)")


CHECKS = {
    "raw": {"units": raw_units, "is_control": raw_control, "context": literal_col("cell_line"), "batch": batch,
            "modality": modality_raw},
    "processed": {"units": proc_units, "is_control": proc_control, "context": inherited("context"), "batch": batch,
                  "modality": inherited("modality"), "split": proc_split},
    "marson_raw": {"units": m_units, "is_control": m_control, "context": m_context, "batch": m_batch,
                   "modality": m_modality},
    "marson_pseudobulk": {"units": pb_units, "is_control": pb_control, "context": pb_context, "batch": pb_batch,
                          "modality": inherited("modality")},
    "jin_raw": {"units": j_units, "is_control": j_control, "context": j_context, "batch": j_batch,
                "modality": j_modality},
}


def main(card: str):
    p = Path(card)
    text, meta, rows = read_card(p)
    if "url" in meta:   # one remote file per donor x condition; only the obs columns are read
        obs = load_remote(meta, MARSON_COLS)
        rel, sha = f"{len(card_files(meta))} remote files under {meta['url']}", None
    elif "file_glob" in meta:   # one local file per 10x channel; obs of all files, concatenated
        fs = sorted(p.parent.glob(meta["file_glob"]))
        obs = pd.concat([ad.read_h5ad(f, backed="r").obs.assign(_file=f.stem) for f in fs])
        sums = p.parents[2] / "SHA256SUMS"
        listed = sums.read_text() if sums.exists() else ""
        n_ok = sum(str(f.relative_to(p.parents[2])) in listed for f in fs)
        rel, sha = f"{len(fs)} files {meta['file_glob']}", f"{n_ok}/{len(fs)} files"
    else:
        h5 = p.parent / meta["file"]
        obs = ad.read_h5ad(h5, backed="r").obs
        sums = p.parents[2] / "SHA256SUMS"
        rel = str(h5.relative_to(p.parents[2]))
        sha = next((l.split()[0] for l in sums.read_text().splitlines() if l.endswith(rel)), "not listed") \
            if sums.exists() else "no SHA256SUMS"
    parent = read_card(p.parents[2] / meta["derived_from"])[2] if "derived_from" in meta else None
    results = {}
    for field, row in rows.items():
        fn = CHECKS[meta.get("checks", meta["layer"])].get(field)
        if fn is None:
            results[field] = ("?", "no check written for this field")
        else:
            results[field] = fn(obs, row["spec"], parent=parent, card_dir=p.parent, meta=meta)
    header = (f"Checked {dt.date.today()} by `tmp/check_card.py` on `{rel}` "
              f"({obs.shape[0]:,} " + ("rows" if "pseudobulk" in meta.get("checks", "") else "cells") + (f"; sha256 of {sha} listed in SHA256SUMS)." if sha and sha.endswith("files") else f"; sha256 {sha[:12]}… from SHA256SUMS)." if sha else "; obs only, read remotely)."))
    write_card(p, text, results, header)
    for f, (m, ev) in results.items():
        print(f"{m} {f}: {ev}")
    return 0 if all(m != "✗" for m, _ in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
