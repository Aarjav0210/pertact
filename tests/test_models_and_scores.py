"""The embedding table, the decoder's two structural properties, the InfoNCE loss and the scoring helpers."""
import numpy as np
import pandas as pd
import torch

from pertact.decoders import LatentAdditive, PertEmbedding, SplitData
from pertact.actions import soft_name, soft_variants
from pertact.infonce import cosine_infonce_loss, infonce_loss, soft_targets
from pertact.metrics import pearson, spearman_brown_ceiling
from pertact.runs import fingerprint, replace_runs


def table():
    return PertEmbedding.from_arrays("toy", ["A", "B"], np.array([[3.0, 0.0], [0.0, 1.0]]), ref_genes=["A", "B"])


def test_embedding_table_scales_once_and_sums_a_double():
    emb = table()
    assert np.isclose(np.linalg.norm(emb.matrix, axis=1).mean(), 1.0)
    assert np.isclose(emb.matrix[0, 0] / emb.matrix[1, 1], 3.0)                    # relative lengths are kept
    m = emb.condition_matrix(["A", "A+B", "control", "A+Z"])
    assert np.allclose(m[1], emb.matrix[0] + emb.matrix[1]) and np.allclose(m[2], 0)
    assert np.allclose(m[3], emb.matrix[0])                                         # an uncovered gene adds nothing
    assert emb.covers("A+B") and not emb.covers("A+Z")


def test_decoder_treats_zero_as_no_perturbation_and_adds_in_the_latent():
    torch.manual_seed(0)
    model = LatentAdditive(n_genes=7, emb_dim=2, state_dim=3, latent_dim=4, hidden_dim=8, n_layers=1, dropout=0.0,
                           state_mean=np.zeros(3), state_std=np.ones(3)).eval()
    s, ea, eb = torch.randn(5, 3), torch.tensor([[1.0, 0.0]]).expand(5, -1), torch.tensor([[0.0, 2.0]]).expand(5, -1)
    base = model.latent(torch.zeros(5, 2), s)
    assert torch.allclose(base, model.enc((s - model.s_mu) / model.s_sd))           # e = 0 leaves the control's latent
    assert torch.allclose(model.latent(ea + eb, s) - base, (model.latent(ea, s) - base) + (model.latent(eb, s) - base), atol=1e-6)
    only = LatentAdditive(7, 2, 3, 4, 8, 1, 0.0, use_state=False).eval()
    assert torch.allclose(only(ea, s), only(ea, torch.randn(5, 3)))                 # decoder-only ignores the state


def test_controls_are_sampled_from_the_same_gem_group():
    data = SplitData(X=torch.zeros(4, 2), cond=np.array(["A"] * 4), gem=np.array([0, 1, 1, 0]),
                     ctrl_state=torch.zeros(6, 3), ctrl_gem=np.array([0, 0, 1, 1, 1, 0]))
    picks = data.sample_controls(np.array([0, 1, 1, 0] * 50), np.random.default_rng(0))
    assert (data.ctrl_gem[picks] == np.array([0, 1, 1, 0] * 50)).all()


def test_infonce_loss_prefers_the_true_shift_and_the_anchor_only_adds():
    torch.manual_seed(0)
    shifts = torch.tensor([[6.0, 0.0], [0.0, 6.0]])
    a = torch.tensor([0, 0, 1, 1])
    ctrl = torch.randn(4, 2)
    pert = ctrl + shifts[a]
    right, zero = infonce_loss(shifts, ctrl, pert, a, 0.1, 10.0), infonce_loss(torch.zeros(2, 2), ctrl, pert, a, 0.1, 10.0)
    assert right < zero
    assert infonce_loss(torch.zeros(2, 2), ctrl, pert, a, 0.1, 10.0, rho=0.5) > zero
    assert torch.isclose(infonce_loss(shifts, ctrl, pert, a, 0.1, 10.0, rho=0.5), right)   # the anchor is zero at the partner


def test_cosine_loss_ignores_length_and_only_turns_the_vectors():
    torch.manual_seed(0)
    shifts = torch.tensor([[6.0, 0.0], [0.0, 6.0]])
    a = torch.tensor([0, 0, 1, 1])
    ctrl = torch.randn(4, 2)
    pert = ctrl + shifts[a] + 0.1 * torch.randn(4, 2)
    F = shifts.clone().requires_grad_()
    right = cosine_infonce_loss(F, ctrl, pert, a, 0.1)
    assert torch.isclose(right, cosine_infonce_loss(5 * F, ctrl, pert, a, 0.1))            # the length of f has no effect
    assert right < cosine_infonce_loss(torch.flip(shifts, [0]), ctrl, pert, a, 0.1)        # the wrong directions score worse
    assert cosine_infonce_loss(F, ctrl, pert, a, 0.05) < right                             # a lower temperature sharpens a right answer
    right.backward()
    assert torch.allclose((F.grad * F).sum(1), torch.zeros(2), atol=1e-5)                  # the gradient is at right angles to f


def test_soft_positive_labels_and_what_they_do_to_the_losses():
    a = torch.tensor([0, 0, 1, 1])
    T = soft_targets(a, 0.5)
    assert torch.allclose(T.sum(1), torch.ones(4)) and torch.allclose(T, T.T)
    assert torch.allclose(T[0], torch.tensor([1.0, 0.5, 0.0, 0.0]) / 1.5)          # partner 1, same perturbation s, others 0
    assert torch.equal(soft_targets(a, 0.0), torch.eye(4))                         # s = 0 is one right answer per query
    assert soft_name("cosine", 0.03) == "infonce_cos_soft0.03" and soft_name("distance", 1.0) == "infonce_soft1"
    assert soft_variants({"infonce": {"soft_grid": [0, 0.1]}}) == ["infonce_cos", "infonce_cos_soft0", "infonce_cos_soft0.1", "infonce_soft0", "infonce_soft0.1"]
    torch.manual_seed(0)
    F, ctrl = torch.randn(2, 3), torch.randn(4, 3)
    pert = ctrl + F[a] + 0.3 * torch.randn(4, 3)
    for hard, soft0 in ((cosine_infonce_loss(F, ctrl, pert, a, 0.1), cosine_infonce_loss(F, ctrl, pert, a, 0.1, soft=0.0)),
                        (infonce_loss(F, ctrl, pert, a, 0.1, 10.0), infonce_loss(F, ctrl, pert, a, 0.1, 10.0, soft=0.0))):
        assert torch.isclose(hard, soft0, atol=1e-6)                               # the new path at s = 0 is the old loss
    swap = torch.tensor([1, 0, 3, 2])                                              # exchange the two cells of each perturbation
    for loss in (lambda p, s: cosine_infonce_loss(F, ctrl, p, a, 0.1, soft=s), lambda p, s: infonce_loss(F, ctrl, p, a, 0.1, 10.0, soft=s)):
        assert torch.isclose(loss(pert, 1.0), loss(pert[swap], 1.0), atol=1e-6)    # at s = 1 the pairing no longer matters
        assert not torch.isclose(loss(pert, 0.0), loss(pert[swap], 0.0), atol=1e-4)   # at s = 0 it does


def test_pearson_and_the_noise_ceiling_formula():
    x, y = np.array([1.0, 2.0, 4.0, 3.0]), np.array([2.0, 1.0, 5.0, 4.0])
    assert np.isclose(pearson(x, y), np.corrcoef(x, y)[0, 1])
    assert np.isnan(pearson(x, np.ones(4)))
    assert spearman_brown_ceiling(1.0) == 1.0 and spearman_brown_ceiling(0.0) == 0.0
    assert np.isclose(spearman_brown_ceiling(0.5), np.sqrt(2 / 3))
    assert np.isnan(spearman_brown_ceiling(float("nan")))


def test_replace_runs_and_fingerprint():
    old = pd.DataFrame({"method": ["a", "b"], "decoder": ["d", "d"], "seed": [0, 0], "value": [1.0, 2.0]})
    new = pd.DataFrame({"method": ["b"], "decoder": ["d"], "seed": [0], "value": [9.0]})
    assert replace_runs(old, new, {("b", "d", 0)})["value"].tolist() == [1.0, 9.0]
    emb, other = table(), table()
    other.matrix[0, 0] += 1e-3
    assert fingerprint(emb) == fingerprint(table()) and fingerprint(emb) != fingerprint(other)
