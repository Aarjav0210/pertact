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

Metrics are computed per perturbation, then averaged over perturbations:
    mse_all / mse_de20                       : predicted vs observed mean expression
    pearson_delta_all / pearson_delta_de20   : predicted vs observed change (mean - control mean)
    mse_de20_nt / pearson_delta_de20_nt      : same on the top-20 DE genes after removing the perturbed target genes,
                                               which CRISPRa pushes up and any model that knows the target gets for free
    pearson_int_all / pearson_int_top20      : doubles only. Interaction = profile - additive reference
                                               (control + Δa + Δb from training singles); correlation of predicted vs
                                               observed interaction on all genes / on the 20 genes with the largest
                                               observed interaction
Each split is reported overall, on the conditions the embedding fully covers ("<split>_covered") and, for test,
on the most non-additive quarter of doubles ("test_nonadditive").
"""
from __future__ import annotations

import argparse
import itertools
import time

import anndata as ad
import numpy as np
import pandas as pd

from .config import get_logger, load_config, path, read_json, seed_everything, write_json
from .data import CONTROL, condition_genes
import torch

from .decoders import (LatentAdditive, PertEmbedding, build_split, pick_device, predict_means, predict_means_compose,
                       train_decoder)
from .embeddings import load_embedding
from .encoders import attach_state

log = get_logger("evaluate")
METRICS = ["mse_all", "mse_de20", "pearson_delta_all", "pearson_delta_de20",
           "mse_de20_nt", "pearson_delta_de20_nt", "pearson_int_all", "pearson_int_top20"]
NONADDITIVE_FRAC = 0.25


# --------------------------------------------------------------------------- data helpers
def load_all(cfg):
    adata = ad.read_h5ad(path(cfg, "processed_h5ad"))
    attach_state(cfg, adata)
    return adata, read_json(path(cfg, "splits"))


def group_means(adata, mask, conds) -> dict[str, np.ndarray]:
    cond = adata.obs["condition"].astype(str).to_numpy()
    return {c: np.asarray(adata.X[mask & (cond == c)].mean(0)).ravel() for c in conds}


def split_mask(adata, name):
    return (adata.obs["split"] == name).to_numpy()


def eval_conditions(splits, split):
    return splits["train_singles"] if split == "val" else splits["test_doubles"]


def train_deltas(adata, splits) -> dict[str, np.ndarray]:
    """Δa = mean(training cells of a) - mean(training controls), for every training single."""
    mu = group_means(adata, split_mask(adata, "train"), splits["train_singles"] + [CONTROL])
    return {c: mu[c] - mu[CONTROL] for c in splits["train_singles"]}


# --------------------------------------------------------------------------- DE genes
def compute_de(cfg, adata, splits):
    """Store a longer ranked list (n_de_store) so target genes can be removed and still leave n_de genes."""
    import scanpy as sc

    n, n_store = cfg["eval"]["n_de"], cfg["eval"].get("n_de_store", 50)
    out = {"n_de": n, "n_stored": n_store, "method": "wilcoxon", "rankby_abs": True}
    for key, split, conds in (("singles", "train", splits["train_singles"]), ("doubles", "test", splits["test_doubles"])):
        m = split_mask(adata, split) & adata.obs["condition"].astype(str).isin(conds + [CONTROL]).to_numpy()
        b = adata[m].copy()
        b.obs["condition"] = b.obs["condition"].astype(str).astype("category")
        t0 = time.time()
        sc.tl.rank_genes_groups(b, "condition", groups=conds, reference=CONTROL, method="wilcoxon",
                                rankby_abs=True, n_genes=n_store, tie_correct=False)
        names = b.uns["rank_genes_groups"]["names"]
        out[key] = {c: [str(g) for g in names[c]] for c in conds}
        log.info("DE %s: %d conditions on %s cells (%.0fs)", key, len(conds), split, time.time() - t0)
    write_json(out, path(cfg, "results_dir") / "de_genes.json")
    return out


def load_de(cfg):
    return read_json(path(cfg, "results_dir") / "de_genes.json")


# --------------------------------------------------------------------------- metrics
def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


class EvalRef:
    """Everything fixed for one evaluation split: observed means, control mean, gene lists, additive reference."""

    def __init__(self, cfg, adata, splits, de, split, deltas):
        self.split = split
        n = cfg["eval"]["n_de"]
        vidx = {g: i for i, g in enumerate(adata.var_names)}
        t2v = dict(adata.uns["target_to_var"])
        self.conds = eval_conditions(splits, split)
        self.obs = group_means(adata, split_mask(adata, split), self.conds + [CONTROL])
        self.ctrl = self.obs[CONTROL]
        de_c = de["singles"] if split == "val" else de["doubles"]
        self.de_idx, self.de_nt_idx = {}, {}
        for c in self.conds:
            targets = {t2v.get(g, g) for g in condition_genes(c)}
            self.de_idx[c] = np.array([vidx[g] for g in de_c[c][:n]])
            nt = [g for g in de_c[c] if g not in targets][:n]
            assert len(nt) == n, f"not enough stored DE genes for {c}"
            self.de_nt_idx[c] = np.array([vidx[g] for g in nt])
        self.add_ref, self.int_idx, self.nonadditive = {}, {}, set()
        if split == "test":
            strength = {}
            for c in self.conds:
                self.add_ref[c] = self.ctrl + sum(deltas[g] for g in condition_genes(c))
                i_obs = self.obs[c] - self.add_ref[c]
                self.int_idx[c] = np.argsort(-np.abs(i_obs))[:n]
                j = self.de_idx[c]
                strength[c] = np.linalg.norm(i_obs[j]) / np.linalg.norm((self.obs[c] - self.ctrl)[j])
            k = int(round(NONADDITIVE_FRAC * len(self.conds)))
            self.nonadditive = set(sorted(strength, key=strength.get, reverse=True)[:k])
            self.strength = strength

    def metrics(self, c, pred) -> dict:
        obs, ctrl, j, jn = self.obs[c], self.ctrl, self.de_idx[c], self.de_nt_idx[c]
        dp, do = pred - ctrl, obs - ctrl
        out = {"mse_all": float(np.mean((pred - obs) ** 2)), "mse_de20": float(np.mean((pred[j] - obs[j]) ** 2)),
               "pearson_delta_all": pearson(dp, do), "pearson_delta_de20": pearson(dp[j], do[j]),
               "mse_de20_nt": float(np.mean((pred[jn] - obs[jn]) ** 2)), "pearson_delta_de20_nt": pearson(dp[jn], do[jn]),
               "pearson_int_all": np.nan, "pearson_int_top20": np.nan}
        if c in self.add_ref:
            ip, io, k = pred - self.add_ref[c], obs - self.add_ref[c], self.int_idx[c]
            out["pearson_int_all"], out["pearson_int_top20"] = pearson(ip, io), pearson(ip[k], io[k])
        return out

    def score(self, preds: dict, covered: dict | None = None) -> pd.DataFrame:
        return pd.DataFrame([{"condition": c, "covered": True if covered is None else covered[c],
                              "nonadditive": c in self.nonadditive, **self.metrics(c, p)} for c, p in preds.items()])


def summarise(per: pd.DataFrame, method, decoder, split, seed) -> list[dict]:
    subsets = [(split, per), (f"{split}_covered", per[per.covered])]
    if split == "test":
        subsets.append(("test_nonadditive", per[per.nonadditive]))
    out = []
    for sub, df in subsets:
        for m in METRICS:
            vals = df[m].to_numpy(dtype=float) if len(df) else np.array([])
            v = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            out.append({"method": method, "decoder": decoder, "split": sub, "seed": seed, "metric": m,
                        "value": v, "n_conditions": len(df)})
    return out


def write_results(cfg, rows: list[dict], per_rows: list[pd.DataFrame], keys):
    """Replace rows of results/week1.csv whose (method, decoder, seed) were just recomputed."""
    rd = path(cfg, "results_dir")
    rd.mkdir(parents=True, exist_ok=True)
    for fname, new in (("week1.csv", pd.DataFrame(rows)),
                       ("week1_per_perturbation.csv", pd.concat(per_rows) if per_rows else pd.DataFrame())):
        p = rd / fname
        if new.empty:
            continue
        if p.exists():
            old = pd.read_csv(p)
            k = old[["method", "decoder", "seed"]].apply(tuple, axis=1).isin(keys)
            new = pd.concat([old[~k], new])
        new.sort_values(["method", "decoder", "split", "seed"] + (["metric"] if "metric" in new else ["condition"])
                        ).to_csv(p, index=False)


# --------------------------------------------------------------------------- baselines and noise ceiling
def noise_ceiling(adata, ref: EvalRef, seed: int) -> pd.DataFrame:
    """Best correlation any predictor could reach against the observed (noisy) test means.

    Split each double's cells in half, correlate the two halves' profiles, apply Spearman-Brown to get the
    reliability of the full mean, and take its square root (expected correlation of the true profile with the
    observed one). Gene lists are the same as in evaluation.
    """
    rng = np.random.default_rng(seed)
    cond = adata.obs["condition"].astype(str).to_numpy()
    te = split_mask(adata, "test")
    rows = []
    ceil = lambda r: float(np.sqrt(max(0.0, 2 * r / (1 + r)))) if np.isfinite(r) and r > -1 else np.nan
    for c in ref.conds:
        idx = np.flatnonzero(te & (cond == c))
        idx = idx[rng.permutation(len(idx))]
        h1, h2 = (np.asarray(adata.X[np.sort(h)].mean(0)).ravel() for h in np.array_split(idx, 2))
        j, jn, k = ref.de_idx[c], ref.de_nt_idx[c], ref.int_idx[c]
        d1, d2, i1, i2 = h1 - ref.ctrl, h2 - ref.ctrl, h1 - ref.add_ref[c], h2 - ref.add_ref[c]
        rows.append({"condition": c, "covered": True, "nonadditive": c in ref.nonadditive,
                     "mse_all": np.nan, "mse_de20": np.nan, "mse_de20_nt": np.nan,
                     "pearson_delta_all": ceil(pearson(d1, d2)), "pearson_delta_de20": ceil(pearson(d1[j], d2[j])),
                     "pearson_delta_de20_nt": ceil(pearson(d1[jn], d2[jn])),
                     "pearson_int_all": ceil(pearson(i1, i2)), "pearson_int_top20": ceil(pearson(i1[k], i2[k]))})
    return pd.DataFrame(rows)


def run_baselines(cfg, adata, splits, de):
    deltas = train_deltas(adata, splits)
    rows, per = [], []
    for split in ("val", "test"):
        ref = EvalRef(cfg, adata, splits, de, split, deltas)
        preds = {"control_mean": {c: ref.ctrl for c in ref.conds}}
        if split == "val":
            preds["train_mean"] = {c: ref.ctrl + deltas[c] for c in ref.conds}
        else:
            preds["additive_expr"] = {c: ref.add_ref[c] for c in ref.conds}
        for name, pm in preds.items():
            df = ref.score(pm)
            rows += summarise(df, name, "baseline", split, 0)
            per.append(df.assign(method=name, decoder="baseline", split=split, seed=0))
        if split == "test":
            df = noise_ceiling(adata, ref, cfg["seed"])
            rows += summarise(df, "noise_ceiling", "baseline", split, 0)
            per.append(df.assign(method="noise_ceiling", decoder="baseline", split=split, seed=0))
            log.info("most non-additive doubles (top %d): %s", len(ref.nonadditive),
                     sorted(ref.nonadditive, key=ref.strength.get, reverse=True)[:10])
    keys = {(m, "baseline", 0) for m in ("control_mean", "train_mean", "additive_expr", "noise_ceiling")}
    write_results(cfg, rows, per, keys)
    log.info("baselines:\n%s", pd.DataFrame(rows).query("split in ['val', 'test', 'test_nonadditive']")
             .pivot_table(index=["method", "split"], columns="metric", values="value").round(3).to_string())


# --------------------------------------------------------------------------- decoder runs
def pred_path(cfg, method, decoder, seed):
    return path(cfg, "results_dir") / "predictions" / f"{method}__{decoder}__s{seed}.npz"


def model_path(cfg, method, decoder, seed):
    return path(cfg, "models_dir") / "decoders" / f"{method}__{decoder}__s{seed}.pt"


class Runner:
    def __init__(self, cfg, hp_override=None, eval_splits=("val", "test")):
        self.cfg = cfg
        self.hp = dict(cfg["decoder"]) | (hp_override or {})
        self.custom_emb = {}  # (method, seed) -> PertEmbedding built in memory (Week 2 action vectors)
        self.device = pick_device(self.hp.get("device", "auto"))
        self.adata, self.splits = load_all(cfg)
        a, k, gk = self.adata, self.hp["state_key"], cfg["data"]["batch_key"]
        inc = self.hp["include_control_targets"]
        self.train = build_split(a, "train", k, gk, inc, self.device)
        self.val = build_split(a, "val", k, gk, inc, self.device)
        if "test" in eval_splits:  # val-only runners never read test cells
            self.test_pool = build_split(a, "test", k, gk, False, self.device, target_conds=[])
        S = np.asarray(a.obsm[k][split_mask(a, "train")])
        self.state_stats = (S.mean(0), S.std(0) + 1e-6)
        deltas = train_deltas(a, self.splits)
        de = load_de(cfg)
        self.refs = {s: EvalRef(cfg, a, self.splits, de, s, deltas) for s in eval_splits}
        cond = a.obs["condition"].astype(str).to_numpy()
        gem = a.obs[gk].astype("category").cat.codes.to_numpy().astype(np.int64)
        self.cond_gems = {s: {c: gem[split_mask(a, s) & (cond == c)] for c in self.refs[s].conds} for s in eval_splits}
        log.info("device %s | train targets %d | val targets %d | hp %s", self.device, len(self.train.cond),
                 len(self.val.cond), {x: self.hp[x] for x in ("latent_dim", "hidden_dim", "n_layers", "dropout", "lr")})

    def embedding(self, method) -> PertEmbedding:
        genes, M = load_embedding(self.cfg, method)
        return PertEmbedding.from_arrays(method, genes, M, self.cfg["embeddings"]["normalize"],
                                         ref_genes=self.splits["train_singles"])

    def train_or_load(self, method, decoder, seed, save=True):
        """Load models/decoders/<method>__<decoder>__s<seed>.pt if present, else train (and save) it."""
        emb = self.custom_emb[(method, seed)] if (method, seed) in self.custom_emb else self.embedding(method)
        mp = model_path(self.cfg, method, decoder, seed)
        if mp.exists():
            ck = torch.load(mp, map_location=self.device)
            assert ck["hp"] == {k: self.hp[k] for k in ck["hp"]}, "saved model was trained with other hyperparameters"
            model = LatentAdditive(self.train.X.shape[1], emb.dim, self.train.ctrl_state.shape[1], self.hp["latent_dim"],
                                   self.hp["hidden_dim"], self.hp["n_layers"], self.hp["dropout"], ck["use_state"],
                                   *self.state_stats).to(self.device)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            return model, emb, ck["info"]
        model, info, _ = train_decoder(emb, self.train, self.val, self.hp, decoder, seed, self.device,
                                       self.state_stats, verbose=False)
        if save:
            mp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "use_state": model.use_state, "info": info,
                        "hp": {k: self.hp[k] for k in ("latent_dim", "hidden_dim", "n_layers", "dropout", "lr",
                                                       "weight_decay", "batch_size", "max_epochs", "patience")}}, mp)
        return model, emb, info

    def fit_eval(self, method, decoder, seed, splits=("val", "test"), save=True, pred_file=None):
        model, emb, info = self.train_or_load(method, decoder, seed, save=save)
        rows, per, saved = [], [], {}
        for split in splits:
            ref = self.refs[split]
            pool = self.val if split == "val" else self.test_pool
            pm = predict_means(model, emb, ref.conds, pool, self.cond_gems[split],
                               self.cfg["eval"]["n_ctrl_samples_per_pert"], seed, self.device)
            saved[f"{split}_conds"], saved[f"{split}_pred"] = np.array(ref.conds, dtype=str), pm.astype(np.float32)
            df = ref.score(dict(zip(ref.conds, pm)), covered={c: emb.covers(c) for c in ref.conds})
            rows += summarise(df, method, decoder, split, seed)
            per.append(df.assign(method=method, decoder=decoder, split=split, seed=seed))
        if save:
            p = pred_file or pred_path(self.cfg, method, decoder, seed)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, **saved)
        log.info("%s/%s seed %d: %d epochs (best %d), %.0fs | %s", method, decoder, seed, info["epochs"],
                 info["best_epoch"], info["seconds"],
                 " ".join(f"{r['split']}:{r['metric']}={r['value']:.3f}" for r in rows
                          if r["metric"] in ("pearson_delta_de20", "pearson_int_all") and r["split"] in ("val", "test")
                          and np.isfinite(r["value"])))
        return rows, per, info


def compose(cfg, methods, seeds):
    """Score the same latent-additive models with additive and with sequential composition of test doubles.

    Rows are written with decoder "latent_additive_seq" (mean of both orders). Additive rows stay those of
    `run`, which these models reproduce. Per-double order effects go to results/sequential_orders.csv.
    """
    hp = read_json(path(cfg, "results_dir") / "best_hparams.json")["hparams"]
    runner = Runner(cfg, hp)
    assert runner.hp["state_key"] == "X_pca_g", "sequential re-encoding uses the frozen PCA state encoder"
    pca = np.load(path(cfg, "models_dir") / "pca_g.npz")
    ref = runner.refs["test"]
    orders = []
    for method, seed in itertools.product(methods, seeds):
        model, emb, info = runner.train_or_load(method, "latent_additive", seed)
        out = predict_means_compose(model, emb, ref.conds, runner.test_pool, runner.cond_gems["test"],
                                    cfg["eval"]["n_ctrl_samples_per_pert"], seed, runner.device,
                                    pca["mean"], pca["components"])
        covered = {c: emb.covers(c) for c in ref.conds}
        old = np.load(pred_path(cfg, method, "latent_additive", seed))
        drift = float(np.abs(old["test_pred"] - out["additive"]).max())
        df = ref.score(dict(zip(ref.conds, out["seq_avg"])), covered=covered)
        rows = summarise(df, method, "latent_additive_seq", "test", seed)
        write_results(cfg, rows, [df.assign(method=method, decoder="latent_additive_seq", split="test", seed=seed)],
                      {(method, "latent_additive_seq", seed)})
        p = pred_path(cfg, method, "latent_additive_seq", seed)
        np.savez(p, test_conds=np.array(ref.conds, dtype=str), test_pred=out["seq_avg"],
                 test_pred_ab=out["seq_ab"], test_pred_ba=out["seq_ba"])
        for k, c in enumerate(ref.conds):
            j = ref.de_idx[c]
            dab, dba = out["seq_ab"][k] - ref.ctrl, out["seq_ba"][k] - ref.ctrl
            mab, mba = ref.metrics(c, out["seq_ab"][k]), ref.metrics(c, out["seq_ba"][k])
            orders.append({"method": method, "seed": seed, "condition": c, "covered": covered[c],
                           "order_agreement_de20": pearson(dab[j], dba[j]), "order_agreement_all": pearson(dab, dba),
                           "pearson_delta_de20_ab": mab["pearson_delta_de20"], "pearson_delta_de20_ba": mba["pearson_delta_de20"],
                           "pearson_int_all_ab": mab["pearson_int_all"], "pearson_int_all_ba": mba["pearson_int_all"]})
        r = {x["metric"]: x["value"] for x in rows if x["split"] == "test"}
        log.info("%s seed %d | additive reproduces saved predictions (max |diff| %.2e) | sequential: pearson_delta_de20 %.3f, "
                 "pearson_int_all %.3f", method, seed, drift, r["pearson_delta_de20"], r["pearson_int_all"])
    pd.DataFrame(orders).to_csv(path(cfg, "results_dir") / "sequential_orders.csv", index=False)


def rescore(cfg):
    """Recompute every metric from saved predictions; no retraining."""
    adata, splits = load_all(cfg)
    deltas, de = train_deltas(adata, splits), load_de(cfg)
    refs = {s: EvalRef(cfg, adata, splits, de, s, deltas) for s in ("val", "test")}
    files = sorted((path(cfg, "results_dir") / "predictions").glob("*.npz"))
    for f in files:
        method, decoder, s = f.stem.split("__")
        seed = int(s[1:])
        genes, _ = load_embedding(cfg, method)
        emb = PertEmbedding(method, genes, np.zeros((len(genes), 1), np.float32))
        z = np.load(f)
        rows, per = [], []
        for split, ref in refs.items():
            if f"{split}_conds" not in z.files:  # e.g. sequential predictions exist for test doubles only
                continue
            conds = z[f"{split}_conds"].tolist()
            df = ref.score(dict(zip(conds, z[f"{split}_pred"])), covered={c: emb.covers(c) for c in conds})
            rows += summarise(df, method, decoder, split, seed)
            per.append(df.assign(method=method, decoder=decoder, split=split, seed=seed))
        write_results(cfg, rows, per, {(method, decoder, seed)})
    log.info("rescored %d runs from saved predictions", len(files))


def tune(cfg):
    grid = cfg["decoder"]["tune_grid"]
    keys = list(grid)
    out_p = path(cfg, "results_dir") / "tuning.csv"
    done = pd.read_csv(out_p) if out_p.exists() else pd.DataFrame(columns=keys)
    res = done.to_dict("records")
    have = {tuple(r[k] for k in keys) for r in res}
    runner = Runner(cfg)
    for vals in itertools.product(*(grid[k] for k in keys)):
        hp = dict(zip(keys, vals))
        if tuple(vals) in have:  # resumable: configs already in tuning.csv are not re-run
            continue
        runner.hp = dict(cfg["decoder"]) | hp
        rows, _, info = runner.fit_eval("onehot", "latent_additive", cfg["seed"], splits=("val",), save=False)
        r = {m["metric"]: m["value"] for m in rows if m["split"] == "val"}
        res.append(hp | r | {"best_val_mse": info["best_val_mse"], "epochs": info["epochs"], "seconds": info["seconds"]})
        pd.DataFrame(res).to_csv(out_p, index=False)
        log.info("tune %s -> %s", hp, {k: round(v, 4) for k, v in r.items() if np.isfinite(v)})
    df = pd.DataFrame(res)
    in_grid = np.logical_and.reduce([df[k].isin(grid[k]) for k in keys])
    df = df[in_grid]
    metric = cfg["decoder"]["tune_metric"]
    best = df.sort_values([metric, "best_val_mse"], ascending=[False, True]).iloc[0]
    best_hp = {k: (type(grid[k][0])(best[k])) for k in keys}
    write_json({"selected_by": f"max mean {metric} on validation singles (onehot, latent_additive, seed {cfg['seed']})",
                "hparams": best_hp}, path(cfg, "results_dir") / "best_hparams.json")
    log.info("best: %s", best_hp)


def run(cfg, methods, decoders, seeds):
    bp = path(cfg, "results_dir") / "best_hparams.json"
    hp = read_json(bp)["hparams"] if bp.exists() else {}
    if not hp:
        log.warning("no best_hparams.json; using config decoder hyperparameters")
    runner = Runner(cfg, hp)
    infos = []
    for method, decoder, seed in itertools.product(methods, decoders, seeds):
        r, p, info = runner.fit_eval(method, decoder, seed)
        infos.append({"method": method, "decoder": decoder, "seed": seed} | info)
        write_results(cfg, r, p, {(method, decoder, seed)})
    info_p = path(cfg, "results_dir") / "train_info.csv"
    new = pd.DataFrame(infos)
    if info_p.exists():
        old = pd.read_csv(info_p)
        k = old[["method", "decoder", "seed"]].apply(tuple, axis=1).isin(set(new[["method", "decoder", "seed"]].apply(tuple, axis=1)))
        new = pd.concat([old[~k], new])
    new.to_csv(info_p, index=False)


# --------------------------------------------------------------------------- Week 2: action vectors in the decoder
SPACE_SHORT = {"X_pca_g": "pca", "X_scvi30": "scvi"}
ACTION_VARIANTS = ("infonce", "infonce_calibrated", "infonce_scaled_per_action", "infonce_anchor", "mean_shift",
                   "infonce_unit", "mean_shift_unit")


def action_embedding(cfg, adata, splits, space, seed, variant) -> tuple[list[str], np.ndarray]:
    """Frozen Week 2 action vectors f(a) for the training singles. InfoNCE seed = decoder seed; the mean shift is
    deterministic. Global rescaling is undone by the unit_mean_norm step every embedding goes through, so
    infonce_calibrated reaches the decoder exactly as raw infonce does; per-action scaling changes relative lengths.
    A "_unit" suffix keeps only the direction: every vector is divided by its own norm (all lengths equal, as one-hot)."""
    from .retrieval import calibrate_scale, mean_shift_actions, per_action_scale

    if variant.endswith("_unit"):
        genes, M = action_embedding(cfg, adata, splits, space, seed, variant[:-len("_unit")])
        return genes, M / np.linalg.norm(M, axis=1, keepdims=True)
    genes = sorted(splits["train_singles"])
    if variant == "mean_shift":
        acts = mean_shift_actions(cfg, adata, space)
    else:
        base = "infonce_anchor" if variant == "infonce_anchor" else "infonce"
        z = np.load(path(cfg, "embeddings_dir") / f"{base}__{space}__s{seed}.npz")
        acts = dict(zip(z["genes"].tolist(), z["emb"]))
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
    pred_dir = rd / "week2_decoder" / "predictions"
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
        out = rd / "week2_decoder"  # one file set per invocation, so runs can go in parallel
        pd.DataFrame(rows).to_csv(out / f"smoke__{tag}.csv", index=False)
        pd.concat(per).to_csv(out / f"smoke_per_perturbation__{tag}.csv", index=False)
        pd.DataFrame(infos).to_csv(out / f"smoke_train_info__{tag}.csv", index=False)
    return pd.DataFrame(rows)


def results_table(cfg, splits=("val", "test", "test_nonadditive", "val_covered", "test_covered")) -> str:
    """Markdown table: method x decoder x split, mean ± std over seeds."""
    df = pd.read_csv(path(cfg, "results_dir") / "week1.csv")
    g = df.groupby(["split", "method", "decoder", "metric"])["value"]
    s = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.size()}).reset_index()
    s["cell"] = [("n/a" if pd.isna(m) else f"{m:.4f}" if n == 1 else f"{m:.4f} ± {sd:.4f}") if "mse" in k else
                 ("n/a" if pd.isna(m) else f"{m:.3f}" if n == 1 else f"{m:.3f} ± {sd:.3f}")
                 for m, sd, n, k in zip(s["mean"], s["std"].fillna(0), s["n"], s["metric"])]
    nc = df.groupby(["split", "method", "decoder"])["n_conditions"].first()
    out = []
    order = {m: i for i, m in enumerate(cfg["eval"]["methods"] + ["additive_expr", "train_mean", "control_mean", "noise_ceiling"])}
    for sp in splits:
        t = s[s.split == sp].pivot_table(index=["method", "decoder"], columns="metric", values="cell", aggfunc="first")
        if t.empty:
            continue
        t = t.reindex(columns=METRICS).fillna("n/a").reset_index()
        t["n_cond"] = [int(nc.get((sp, m, d), 0)) for m, d in zip(t.method, t.decoder)]
        t["seeds"] = [int(s[(s.split == sp) & (s.method == m) & (s.decoder == d)]["n"].max()) for m, d in zip(t.method, t.decoder)]
        t = t.sort_values(["decoder", "method"], key=lambda c: c.map(order) if c.name == "method" else c)
        out.append(f"\n**{sp}**\n\n| method | decoder | " + " | ".join(METRICS) + " | n_cond | seeds |\n|" + "---|" * (len(METRICS) + 4))
        out += ["| " + " | ".join(str(v) for v in r) + " |" for r in t[["method", "decoder"] + METRICS + ["n_cond", "seeds"]].values]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["de", "tune", "baselines", "run", "rescore", "compose", "table", "actions"])
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--decoders", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=None, help="actions: " + ", ".join(ACTION_VARIANTS))
    ap.add_argument("--spaces", nargs="*", default=None, help="actions: state spaces of the action vectors")
    ap.add_argument("--references", nargs="*", default=["onehot", "pca_loadings"], help="actions: Week 1 embeddings")
    ap.add_argument("--tag", default="main", help="actions: output file tag")
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
    elif a.what == "table":
        print(results_table(cfg))
    elif a.what == "actions":
        action_smoke(cfg, a.variants or ["infonce_scaled_per_action"], a.spaces or cfg["ot"]["spaces"],
                     a.decoders or cfg["eval"]["decoders"], a.seeds if a.seeds is not None else cfg["eval"]["seeds"],
                     references=a.references, tag=a.tag)
    else:
        run(cfg, a.methods or cfg["eval"]["methods"], a.decoders or cfg["eval"]["decoders"],
            a.seeds if a.seeds is not None else cfg["eval"]["seeds"])


if __name__ == "__main__":
    main()
