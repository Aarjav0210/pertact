"""Evaluation decoders. The perturbation embedding matrix is the only thing that varies between runs.

Latent additive (PerturBench, Wu et al. 2025):
    x_hat = dec( enc(g(x_ctrl)) + W e(a) ),       W: Linear(d_emb -> latent, bias=False)
    doubles: e(a) + e(b)
Decoder only (lower bound):
    x_hat = dec( W e(a) )

W has the same output width for every embedding, so capacity downstream of the embedding is matched.
Controls are (optionally) training targets with e = 0, so e = 0 means "no perturbation".
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from .config import get_logger, seed_everything
from .data import CONTROL, condition_genes

log = get_logger("decoders")


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- perturbation embeddings
@dataclass
class PertEmbedding:
    """Any (gene -> vector) table. Conditions are embedded as the sum of their genes' rows;
    genes the table does not cover contribute nothing (no imputation)."""

    name: str
    genes: list[str]
    matrix: np.ndarray
    scale: float = 1.0
    index: dict = field(init=False)

    def __post_init__(self):
        self.index = {g: i for i, g in enumerate(self.genes)}

    @classmethod
    def from_arrays(cls, name, genes, matrix, normalize="unit_mean_norm", ref_genes=None):
        m = np.asarray(matrix, dtype=np.float32)
        scale = 1.0
        if normalize == "unit_mean_norm":
            ref = [i for i, g in enumerate(genes) if ref_genes is None or g in set(ref_genes)]
            scale = float(1.0 / np.linalg.norm(m[ref], axis=1).mean())
        elif normalize not in (None, "none"):
            raise ValueError(normalize)
        return cls(name, list(genes), m * scale, scale)

    @property
    def dim(self) -> int:
        return self.matrix.shape[1]

    def covers(self, cond: str) -> bool:
        return all(g in self.index for g in condition_genes(cond))

    def condition_matrix(self, conds) -> np.ndarray:
        out = np.zeros((len(conds), self.dim), dtype=np.float32)
        for i, c in enumerate(conds):
            for g in condition_genes(c):
                if g in self.index:
                    out[i] += self.matrix[self.index[g]]
        return out


# --------------------------------------------------------------------------- data
@dataclass
class SplitData:
    X: torch.Tensor          # (n_targets, G) log-normalised targets (on device)
    cond: np.ndarray         # condition per target
    gem: np.ndarray          # int gem code per target
    ctrl_state: torch.Tensor  # (n_ctrl, d_state) frozen g(x) of this split's control pool (on device)
    ctrl_gem: np.ndarray
    _order: np.ndarray = field(init=False)
    _off: np.ndarray = field(init=False)
    _cnt: np.ndarray = field(init=False)

    def __post_init__(self):
        self._order = np.argsort(self.ctrl_gem, kind="stable")
        n_gem = int(max(self.ctrl_gem.max(), self.gem.max() if len(self.gem) else 0)) + 1
        self._cnt = np.bincount(self.ctrl_gem, minlength=n_gem)
        self._off = np.r_[0, np.cumsum(self._cnt)[:-1]]

    def sample_controls(self, gem_codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One random control from the same gem group for every entry of gem_codes."""
        cnt = self._cnt[gem_codes]
        assert (cnt > 0).all(), "a gem group has no controls in this split"
        return self._order[self._off[gem_codes] + (rng.random(len(gem_codes)) * cnt).astype(np.int64)]


def dense(X) -> np.ndarray:
    return X.toarray().astype(np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)


def build_split(adata: ad.AnnData, split: str, state_key: str, gem_key: str, include_control_targets: bool,
                device, target_conds=None) -> SplitData:
    obs = adata.obs
    in_split = (obs["split"] == split).to_numpy()
    cond = obs["condition"].astype(str).to_numpy()
    gem_codes = obs[gem_key].astype("category").cat.codes.to_numpy().astype(np.int64)
    is_ctrl = cond == CONTROL
    tgt = in_split & (~is_ctrl | include_control_targets)
    if target_conds is not None:
        tgt &= np.isin(cond, list(target_conds) + ([CONTROL] if include_control_targets else []))
    pool = in_split & is_ctrl
    return SplitData(
        X=torch.from_numpy(dense(adata.X[tgt])).to(device),
        cond=cond[tgt], gem=gem_codes[tgt],
        ctrl_state=torch.from_numpy(np.asarray(adata.obsm[state_key][pool], dtype=np.float32)).to(device),
        ctrl_gem=gem_codes[pool],
    )


# --------------------------------------------------------------------------- model
def mlp(sizes, dropout):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(b), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class LatentAdditive(nn.Module):
    def __init__(self, n_genes, emb_dim, state_dim, latent_dim=128, hidden_dim=512, n_layers=2, dropout=0.1,
                 use_state=True, state_mean=None, state_std=None):
        super().__init__()
        self.use_state = use_state
        if use_state:
            self.enc = mlp([state_dim] + [hidden_dim] * n_layers + [latent_dim], dropout)
            self.register_buffer("s_mu", torch.as_tensor(state_mean, dtype=torch.float32))
            self.register_buffer("s_sd", torch.as_tensor(state_std, dtype=torch.float32))
        self.W = nn.Linear(emb_dim, latent_dim, bias=False)
        self.dec = mlp([latent_dim] + [hidden_dim] * n_layers + [n_genes], dropout)

    def latent(self, e, s=None):
        z = self.W(e)
        if self.use_state:
            z = z + self.enc((s - self.s_mu) / self.s_sd)
        return z

    def forward(self, e, s=None):
        return self.dec(self.latent(e, s))


DECODERS = {"latent_additive": True, "decoder_only": False}  # name -> uses control state


# --------------------------------------------------------------------------- training
def train_decoder(emb: PertEmbedding, train: SplitData, val: SplitData, hp: dict, decoder: str, seed: int,
                  device, state_stats, verbose=True):
    """Fit on train targets (covered conditions only), early-stop on validation-single loss."""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    use_state = DECODERS[decoder]
    conds = sorted(set(train.cond) | set(val.cond))
    cidx = {c: i for i, c in enumerate(conds)}
    C = torch.from_numpy(emb.condition_matrix(conds)).to(device)

    def targets(split):
        ok = np.array([emb.covers(c) for c in split.cond])
        return np.flatnonzero(ok), np.array([cidx[c] for c in split.cond])

    tr_idx, tr_c = targets(train)
    va_idx, va_c = targets(val)
    va_ctrl = val.sample_controls(val.gem[va_idx], np.random.default_rng(10_000 + seed))  # fixed val pairs

    model = LatentAdditive(train.X.shape[1], emb.dim, train.ctrl_state.shape[1], hp["latent_dim"], hp["hidden_dim"],
                           hp["n_layers"], hp["dropout"], use_state, *state_stats).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    bs = hp["batch_size"]

    def val_loss():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(va_idx), 4096):
                j = va_idx[i:i + 4096]
                e = C[torch.as_tensor(va_c[j], device=device)]
                s = val.ctrl_state[torch.as_tensor(va_ctrl[i:i + 4096], device=device)]
                tot += nn.functional.mse_loss(model(e, s), val.X[torch.as_tensor(j, device=device)],
                                              reduction="sum").item()
        return tot / (len(va_idx) * val.X.shape[1])

    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(hp["max_epochs"]):
        model.train()
        perm = tr_idx[rng.permutation(len(tr_idx))]
        ctrl = train.sample_controls(train.gem[perm], rng)
        run = 0.0
        for i in range(0, len(perm), bs):
            j = torch.as_tensor(perm[i:i + bs], device=device)
            e = C[torch.as_tensor(tr_c[perm[i:i + bs]], device=device)]
            s = train.ctrl_state[torch.as_tensor(ctrl[i:i + bs], device=device)]
            loss = nn.functional.mse_loss(model(e, s), train.X[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item() * len(j)
        vl = val_loss()
        hist.append({"epoch": ep, "train_mse": run / len(perm), "val_mse": vl})
        if vl < best - 1e-6:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and (ep % 10 == 0 or bad == 0):
            log.info("[%s/%s s%d] ep %d train %.5f val %.5f%s", emb.name, decoder, seed, ep, run / len(perm), vl,
                     " *" if bad == 0 else "")
        if bad >= hp["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    info = {"best_val_mse": best, "epochs": len(hist), "best_epoch": int(np.argmin([h["val_mse"] for h in hist])),
            "seconds": time.time() - t0, "n_train_targets": int(len(tr_idx)),
            "n_train_conditions_dropped": int(len({c for c in train.cond if not emb.covers(c)}))}
    return model, info, hist


@torch.no_grad()
def predict_means(model: LatentAdditive, emb: PertEmbedding, conds, pool: SplitData, cond_gems: dict, n_samples: int,
                  seed: int, device) -> np.ndarray:
    """Mean predicted expression per condition. Control inputs are drawn from the evaluation split's
    control pool, matched to the gem-group mix of that condition's observed cells."""
    rng = np.random.default_rng(seed)
    C = torch.from_numpy(emb.condition_matrix(conds)).to(device)
    out = np.zeros((len(conds), model.dec[-1].out_features), dtype=np.float32)
    for i, c in enumerate(conds):
        gems = rng.choice(cond_gems[c], size=n_samples, replace=True)
        s = pool.ctrl_state[torch.as_tensor(pool.sample_controls(gems, rng), device=device)]
        e = C[i:i + 1].expand(n_samples, -1)
        out[i] = model(e, s).mean(0).cpu().numpy()
    return out


@torch.no_grad()
def predict_means_compose(model: LatentAdditive, emb: PertEmbedding, doubles, pool: SplitData, cond_gems: dict,
                          n_samples: int, seed: int, device, pca_mean, pca_components) -> dict[str, np.ndarray]:
    """Two ways to compose a double a+b from a model trained on singles, on the same control inputs:

    additive   : x̂ = dec(enc(s) + W(e_a + e_b))                       (one pass)
    seq_ab     : ŷ_a = dec(enc(s) + W e_a);  x̂ = dec(enc(g(ŷ_a)) + W e_b)  (apply a, re-encode with frozen PCA g, apply b)
    seq_ba     : the same with b first
    seq_avg    : mean of seq_ab and seq_ba (simultaneous CRISPRa has no order)

    The control sampling matches predict_means draw for draw, so `additive` equals its output.
    """
    assert model.use_state, "sequential composition needs a decoder that reads the cell state"
    rng = np.random.default_rng(seed)
    mu = torch.as_tensor(pca_mean, dtype=torch.float32, device=device)
    P = torch.as_tensor(pca_components, dtype=torch.float32, device=device)
    g = lambda y: (y - mu) @ P.T  # frozen PCA state encoder, same as obsm["X_pca_g"]
    G = model.dec[-1].out_features
    out = {k: np.zeros((len(doubles), G), dtype=np.float32) for k in ("additive", "seq_ab", "seq_ba", "seq_avg")}
    for i, c in enumerate(doubles):
        a, b = condition_genes(c)
        Ea, Eb = (torch.from_numpy(emb.condition_matrix([x])).to(device).expand(n_samples, -1) for x in (a, b))
        gems = rng.choice(cond_gems[c], size=n_samples, replace=True)
        s = pool.ctrl_state[torch.as_tensor(pool.sample_controls(gems, rng), device=device)]
        add = model(Ea + Eb, s)
        ab = model(Eb, g(model(Ea, s)))
        ba = model(Ea, g(model(Eb, s)))
        out["additive"][i] = add.mean(0).cpu().numpy()
        out["seq_ab"][i] = ab.mean(0).cpu().numpy()
        out["seq_ba"][i] = ba.mean(0).cpu().numpy()
        out["seq_avg"][i] = 0.5 * (out["seq_ab"][i] + out["seq_ba"][i])
    return out
