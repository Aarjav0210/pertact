"""Rerun the cheap, deterministic steps into a temporary folder and compare them with the saved results.

    python -I scripts/check_outputs.py                  # the fast steps
    python -I scripts/check_outputs.py --steps ot       # the OT pairing, about 15 minutes
    python -I scripts/check_outputs.py --steps all
    python -I scripts/check_outputs.py --list

Nothing under results/, embeddings/ or figures/ is written: every step gets its own temporary copy of the config
that points those folders somewhere else. Steps that train models with unrepeatable results (scVI, node2vec, the
decoders) are not rerun. The exit code is 1 if any file differs from the saved one by more than --tol.

results/legacy_solver/ is not checked: it holds what an earlier hand-written OT solver produced and the current
code does not reproduce it (see the README in that folder).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable, "-I"]
WEEK1 = ["week1.csv", "week1_per_perturbation.csv"]
WEEK2 = ["week2_retrieval.csv", "week2_identification_by_pert.csv", "week2_effect_strength.csv", "week2_action_scales.csv",
         "week2_length_ratio.csv"]
OT_FILES = [f"ot/{sp}__{split}{pooled}.npz" for sp in ("X_pca_g", "X_scvi30") for split in ("train", "val") for pooled in ("", "__pooled")]

# name -> what to run, what to stage and what to compare. Paths are relative to results/ unless they start with
# "embeddings/" or are "splits.json". `link` is read-only input (a symlink; never a folder the step writes into),
# `copy` is input the step rewrites or a folder it writes into.
STEPS = {
    "splits": dict(run=["-m", "pertact.splits"], out=["splits.json"]),
    "baselines": dict(run=["-m", "pertact.evaluate", "baselines"], link=["de_genes.json"], copy=WEEK1, out=WEEK1),
    "rescore": dict(run=["-m", "pertact.evaluate", "rescore"], link=["de_genes.json", "predictions"], copy=WEEK1, out=WEEK1),
    "embeddings": dict(run=["-m", "pertact.embeddings", "onehot", "pca_loadings"], embeddings=True,
                       out=["embeddings/onehot.npz", "embeddings/pca_loadings.npz", "embeddings/pca_loadings_full.npz"]),
    "infonce": dict(run=["-m", "pertact.infonce", "train", "--spaces", "X_pca_g"], link=["ot"], copy=["infonce/best_tau.json"], embeddings=True,
                    out=[f"embeddings/infonce__X_pca_g__s{s}.npz" for s in (0, 1, 2)] + [f"infonce/history__X_pca_g__s{s}.csv" for s in (0, 1, 2)]),
    "retrieval": dict(run=["-m", "pertact.retrieval"], link=["ot"], out=WEEK2),
    "doubles": dict(run=["-m", "pertact.doubles"], link=["ot"], out=["week3_doubles.csv", "week3_doubles_by_pair.csv"]),
    "week1_in_state_space": dict(run=["scripts/week3_week1_in_state_space.py"], link=["predictions", "de_genes.json", "week3_doubles_by_pair.csv"],
                                 out=["week3_doubles_week1_embeddings.csv", "week3_doubles_week1_embeddings_by_pair.csv"]),
    "decoder_summary": dict(run=["scripts/week2_decoder_summary.py"], link=["week2_effect_strength.csv"], copy_glob=["week2_decoder/smoke_per_perturbation__*.csv"],
                            out=["week2_decoder/summary.csv", "week2_decoder/paired_length_vs_unit.csv"]),
    "direction_diagnostics": dict(run=["scripts/week2_direction_diagnostics.py"], link=["ot", "week2_effect_strength.csv"], mkdir=["week2_decoder"],
                                  out=["week2_decoder/direction_spread.csv", "week2_decoder/length_overlap.csv"]),
    "perturbation_map": dict(run=["scripts/perturbation_map.py"], out=["perturbation_map.json"]),
    "plot_ot_pairs": dict(run=["scripts/plot_ot_pairs.py"], link=["week2_effect_strength.csv"], copy=["ot/X_pca_g__train.npz"],
                          out=["ot/pair_plot_stats_X_pca_g.csv"]),
    "ot": dict(run=["-m", "pertact.ot"], out=OT_FILES + ["ot/diagnostics.csv"], slow=True),
}


def saved(rel, ref="results"):
    """The saved file that a regenerated file is compared with. `ref` is the folder that holds a step's reference."""
    return ROOT / rel if rel.startswith("embeddings/") or rel == "splits.json" else ROOT / ref / rel


def flat(o, prefix=""):
    """Every leaf of a nested JSON value, with its path."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{prefix}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flat(v, f"{prefix}/{i}")
    else:
        yield prefix, o


def num_diff(a, b):
    """Largest absolute difference between two numeric arrays. Two NaNs in the same place count as equal."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.abs(np.where(both_nan, 0.0, a - b))
    return float(np.nan_to_num(d, nan=np.inf).max()) if d.size else 0.0


def compare(new: Path, old: Path):
    """(largest numeric difference, description of any other mismatch or '')."""
    if new.suffix == ".csv":
        a, b = pd.read_csv(new), pd.read_csv(old)
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            return np.inf, f"shape {a.shape} vs {b.shape}"
        worst, bad = 0.0, []
        for c in a.columns:
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                worst = max(worst, num_diff(a[c], b[c]))
            elif not (a[c].astype(str) == b[c].astype(str)).all():
                bad.append(c)
        return worst, f"text differs in {bad}" if bad else ""
    if new.suffix == ".npz":
        a, b = np.load(new), np.load(old)
        if sorted(a.files) != sorted(b.files):
            return np.inf, f"keys {sorted(a.files)} vs {sorted(b.files)}"
        worst, bad = 0.0, []
        for k in a.files:
            if a[k].shape != b[k].shape:
                return np.inf, f"{k}: shape {a[k].shape} vs {b[k].shape}"
            if a[k].dtype.kind in "fc":
                worst = max(worst, num_diff(a[k], b[k]))
            elif not np.array_equal(a[k], b[k]):
                bad.append(f"{k} ({int((a[k] != b[k]).sum())} of {a[k].size} entries)")
        return worst, f"differs in {bad}" if bad else ""
    a, b = dict(flat(json.loads(new.read_text()))), dict(flat(json.loads(old.read_text())))
    if a.keys() != b.keys():
        return np.inf, f"{len(a.keys() ^ b.keys())} keys differ"
    worst, bad = 0.0, 0
    for k in a:
        if isinstance(a[k], float) or isinstance(b[k], float):
            worst = max(worst, num_diff([a[k]], [b[k]]))
        elif a[k] != b[k]:
            bad += 1
    return worst, f"{bad} values differ" if bad else ""


def run_step(name, spec, tol, keep):
    """Run one step in its own temporary folders and compare what it wrote with the saved files."""
    tmp = Path(tempfile.mkdtemp(prefix=f"pertact_check_{name}_"))
    res = tmp / "results"
    for d in (res, tmp / "figures", tmp / "embeddings"):
        d.mkdir()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["paths"] |= {"results_dir": str(res), "figures_dir": str(tmp / "figures")}
    if spec.get("embeddings"):
        cfg["paths"]["embeddings_dir"] = str(tmp / "embeddings")
    if name == "splits":
        cfg["paths"]["splits"] = str(tmp / "splits.json")
    for rel in spec.get("link", []):
        (res / rel).parent.mkdir(parents=True, exist_ok=True)
        (res / rel).symlink_to(ROOT / "results" / rel)
    copies = spec.get("copy", []) + [str(p.relative_to(ROOT / "results")) for g in spec.get("copy_glob", []) for p in (ROOT / "results").glob(g)]
    for rel in copies:
        (res / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "results" / rel, res / rel)
    for rel in spec.get("mkdir", []):
        (res / rel).mkdir(parents=True, exist_ok=True)
    cfg_file = tmp / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
    cmd = PY + spec["run"] + (["--config", str(cfg_file)] if spec["run"][0] == "-m" else [])
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=os.environ | {"PERTACT_CONFIG": str(cfg_file)}, capture_output=True, text=True)
    secs = time.time() - t0
    rows, ok = [], proc.returncode == 0
    if not ok:
        rows.append((name, "(the step failed)", "FAILED", proc.stderr.strip().splitlines()[-1][:200] if proc.stderr.strip() else ""))
    else:
        for rel in spec["out"]:
            new = tmp / rel if rel.startswith("embeddings/") or rel == "splits.json" else res / rel
            if not new.exists():
                rows.append((name, rel, "MISSING", "the step did not write it"))
                ok = False
                continue
            worst, note = compare(new, saved(rel, spec.get("ref", "results")))
            good = not note and worst <= tol
            ok &= good
            rows.append((name, rel, "identical" if good and worst == 0 else "same" if good else "DIFFERS", note or f"largest difference {worst:.2e}"))
    if keep or not ok:
        print(f"  kept {tmp}")
    else:
        shutil.rmtree(tmp)
    return rows, ok, secs


def main():
    """Run the chosen steps and report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="*", default=None, help="step names, 'fast' (default) or 'all'")
    ap.add_argument("--tol", type=float, default=0.0, help="largest numeric difference that still counts as the same (default 0)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folders")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in STEPS.items():
            print(f"{k:22s} {'slow' if v.get('slow') else 'fast'}  {' '.join(v['run'])}")
        return
    names = [k for k, v in STEPS.items() if not v.get("slow")] if not a.steps or a.steps == ["fast"] else list(STEPS) if a.steps == ["all"] else a.steps
    all_ok = True
    for name in names:
        print(f"[{name}] running ...", flush=True)
        rows, ok, secs = run_step(name, STEPS[name], a.tol, a.keep)
        all_ok &= ok
        for _, rel, status, note in rows:
            print(f"  {status:9s} {rel:55s} {note}")
        print(f"  {secs:.0f}s", flush=True)
    print("ALL SAME" if all_ok else "SOMETHING DIFFERS")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
