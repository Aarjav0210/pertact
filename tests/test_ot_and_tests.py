"""The OT pairing, the action-vector helpers and the rank used by both tests, on synthetic clouds."""
import numpy as np

from pertact.ot import pair, sinkhorn, start_states
from pertact.retrieval import _avg_rank

import ot as pot


def test_sinkhorn_plan_has_uniform_row_and_column_sums():
    rng = np.random.default_rng(0)
    C = pot.dist(rng.normal(size=(60, 5)), rng.normal(size=(20, 5)) + 0.5)
    P, info = sinkhorn(C, 0.05 * np.median(C))
    assert np.allclose(P.sum(1), 1 / 60, atol=1e-8) and np.allclose(P.sum(0), 1 / 20, atol=1e-8)
    assert info["marginal_err"] < 1e-8 and info["cost"] > 0


def test_pair_recovers_the_source_of_a_shifted_cloud():
    """The setting of `pertact.ot --check`: perturbed cells are controls moved by one known vector, plus small noise."""
    rng = np.random.default_rng(0)
    Xc = rng.normal(size=(800, 30)) * np.linspace(3, 0.3, 30)
    src = rng.choice(800, 200, replace=False)
    shift = rng.normal(size=30) * 2
    Xp = Xc[src] + shift + rng.normal(scale=0.05, size=(200, 30))
    top, w, info = pair(Xc, Xp, 0.05, 5)
    assert top.shape == (200, 5) and np.allclose(w.sum(1), 1)
    assert (top[:, 0] == src).all()                            # the true source is the heaviest partner
    assert (np.diff(w, axis=1) <= 1e-12).all()                 # weights are in decreasing order
    moved, _, _ = pair(Xc + shift, Xp, 0.05, 5)                # moving the controls must not change the pairs
    assert (moved == top).all()


def test_start_state_is_the_weighted_average_of_the_partners():
    S = np.arange(12, dtype=float).reshape(6, 2)
    pairs = {"partners": np.array([[0, 2], [4, 5]]), "weights": np.array([[0.75, 0.25], [0.5, 0.5]])}
    assert np.allclose(start_states(S, pairs), [0.75 * S[0] + 0.25 * S[2], 0.5 * S[4] + 0.5 * S[5]])


def test_avg_rank_counts_better_items_and_averages_ties():
    d_all = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 5.0]])
    assert _avg_rank(np.array([3.0, 2.0]), d_all).tolist() == [3.0, 1.5]
