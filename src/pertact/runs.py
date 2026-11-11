"""The Week 1 experiment: baselines, decoder tuning, the runs, sequential composition and rescoring.

Decoders are saved to models/decoders/<method>__<decoder>__s<seed>.pt and reloaded when the file exists; predictions
go to results/predictions/ and scores to results/week1.csv. Commands are in evaluate.py.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd
import torch

from .config import get_logger, path, read_json, write_json
from .decoders import (LatentAdditive, PertEmbedding, build_split, pick_device, predict_means, predict_means_compose,
                       train_decoder)
from .embeddings import load_embedding
from .encoders import load_all
from .metrics import EvalRef, load_de, noise_ceiling, pearson, split_mask, summarise, train_deltas

log = get_logger("evaluate")


# --------------------------------------------------------------------------- result files
def replace_runs(old: pd.DataFrame, new: pd.DataFrame, keys) -> pd.DataFrame:
    """`old` without the rows of the (method, decoder, seed) runs in `keys`, followed by `new`."""
    k = old[["method", "decoder", "seed"]].apply(tuple, axis=1).isin(keys)
    return pd.concat([old[~k], new])


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
            new = replace_runs(pd.read_csv(p), new, keys)
        new.sort_values(["method", "decoder", "split", "seed"] + (["metric"] if "metric" in new else ["condition"])
                        ).to_csv(p, index=False)


def fingerprint(emb: PertEmbedding) -> str:
    """A short hash of an embedding table (its genes and values), stored with every decoder trained on it."""
    h = hashlib.sha1("\n".join(emb.genes).encode())
    h.update(np.ascontiguousarray(emb.matrix, dtype=np.float32).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- baselines
def run_baselines(cfg, adata, splits, de):
    """Score the references that need no model: no change, the training mean (validation), adding the two singles'
    changes (test), and the noise ceiling (test)."""
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
    """File of one run's saved predictions."""
    return path(cfg, "results_dir") / "predictions" / f"{method}__{decoder}__s{seed}.npz"


def model_path(cfg, method, decoder, seed):
    """File of one run's saved decoder."""
    return path(cfg, "models_dir") / "decoders" / f"{method}__{decoder}__s{seed}.pt"


class Runner:
    """Holds the data on the device and trains, reloads, predicts and scores decoders. A runner built with
    eval_splits=("val",) never builds the test controls or the test references."""

    def __init__(self, cfg, hp_override=None, eval_splits=("val", "test")):
        self.cfg = cfg
        self._warned_unverified = False
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
        """A saved Week 1 embedding, scaled the way every embedding is before it reaches the decoder."""
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
            if "embedding" not in ck:  # saved before embeddings were fingerprinted: cannot be checked
                if not self._warned_unverified:
                    log.warning("reloading decoders saved without an embedding fingerprint; delete a file in %s to retrain it", mp.parent)
                    self._warned_unverified = True
            elif ck["embedding"] != fingerprint(emb):
                raise RuntimeError(f"{mp.name} was trained on a different embedding table; delete the file to retrain it")
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
            torch.save({"state_dict": model.state_dict(), "use_state": model.use_state, "info": info, "embedding": fingerprint(emb),
                        "hp": {k: self.hp[k] for k in ("latent_dim", "hidden_dim", "n_layers", "dropout", "lr",
                                                       "weight_decay", "batch_size", "max_epochs", "patience")}}, mp)
        return model, emb, info

    def fit_eval(self, method, decoder, seed, splits=("val", "test"), save=True, pred_file=None):
        """Train or reload one decoder, predict the mean profile of every scored perturbation, save and score."""
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
    """Try the decoder grid with the one-hot embedding and the latent additive decoder on seed 0, and keep the setting
    with the best validation score. Settings already in tuning.csv are skipped, so it can be resumed."""
    grid = cfg["decoder"]["tune_grid"]
    keys = list(grid)
    out_p = path(cfg, "results_dir") / "tuning.csv"
    done = pd.read_csv(out_p) if out_p.exists() else pd.DataFrame(columns=keys)
    res = done.to_dict("records")
    have = {tuple(r[k] for k in keys) for r in res}
    runner = Runner(cfg, eval_splits=("val",))  # tuning never touches the test split
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
    """The Week 1 experiment: every embedding, decoder and seed, with the tuned hyperparameters."""
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
        new = replace_runs(pd.read_csv(info_p), new, set(new[["method", "decoder", "seed"]].apply(tuple, axis=1)))
    new.to_csv(info_p, index=False)
