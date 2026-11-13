"""The scores of the doubles test, on synthetic cells where the right answer is known."""
import numpy as np

from pertact.doubles import all_combinations, centre_scores, compose, rank_true, real_shifts, score_conditions

F = {"A": np.array([4.0, 0, 0, 0], np.float32), "B": np.array([0, 3.0, 0, 0], np.float32),
     "C": np.array([0, 0, 2.0, 0], np.float32)}
EXTRA = np.array([0, 0, 0, 5.0], np.float32)  # what A+C does beyond the sum of A and C


def toy(seed=0, n=200, noise=0.05):
    """Two gem groups with different control centres. A+B is exactly additive; A+C also moves along a fourth axis."""
    rng = np.random.default_rng(seed)
    S, cond, gem = [], [], []
    for g, centre in (("1", np.zeros(4)), ("2", np.array([10.0, 10, 10, 10]))):
        for c, shift in (("control", 0), ("A+B", F["A"] + F["B"]), ("A+C", F["A"] + F["C"] + EXTRA)):
            S.append(centre + shift + rng.normal(scale=noise, size=(n, 4)))
            cond += [c] * n
            gem += [g] * n
    return np.vstack(S).astype(np.float32), np.array(cond), np.array(gem)


def test_real_shift_is_measured_within_each_gem_group():
    S, cond, gem = toy()
    real = real_shifts(S, cond, gem, np.ones(len(S), bool), ["A+B", "A+C"])
    assert np.allclose(real.loc["A+B", "shift"], F["A"] + F["B"], atol=0.02)       # the gem offset of 10 cancels
    assert np.allclose(real.loc["A+C", "shift"], F["A"] + F["C"] + EXTRA, atol=0.02)
    assert real.loc["A+B", "n_cells"] == 400
    # both centres come from 400 cells with per-axis variance 0.05²: 4 axes x 0.0025 x (1/400 + 1/400)
    assert np.isclose(real.loc["A+B", "noise_sq"], 4 * 0.0025 * 2 / 400, rtol=0.2)


def test_centre_scores_separate_direction_from_length():
    s = centre_scores(np.array([2.0, 0.0]), np.array([1.0, 0.0]))
    assert s == {"sq_miss": 1.0, "miss_ratio": 1.0, "len_ratio": 2.0, "cos": 1.0}   # right direction, twice too long
    assert np.isnan(centre_scores(np.zeros(2), np.array([1.0, 0.0]))["cos"])


def test_combinations_and_their_sums():
    combos = all_combinations(["B", "A", "C"])
    assert len(combos) == 1 + 3 + 3 and ("A", "B") in combos and ("B", "A") not in combos
    M = compose(F, [(), ("A",), ("A", "B"), ("A", "Z")], 4)
    assert np.allclose(M[0], 0) and np.allclose(M[2], F["A"] + F["B"]) and np.allclose(M[3], F["A"])


def test_rank_true_orders_candidates_by_where_they_land():
    v = np.array([[-1.0, 0.0], [0.0, -2.0]], np.float32)          # start - cell: the cells sit at (1, 0) and (0, 2)
    cand = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]], np.float32)
    assert rank_true(v, cand, np.array([0, 1])).tolist() == [1.0, 1.0]
    assert rank_true(v, cand, np.array([1, 0])).tolist() == [3.0, 3.0]
    assert rank_true(v, np.zeros((3, 2), np.float32), np.array([0, 0])).tolist() == [2.0, 2.0]   # all tied


def test_score_conditions_on_an_additive_and_a_non_additive_double():
    S, cond, gem = toy()
    sets = {("mean_shift", 0): F, ("zero", 0): {}, ("double_length", 0): {g: 2 * v for g, v in F.items()}}
    cands = {"doubles": [("A", "B"), ("A", "C")], "all_combinations": all_combinations(list(F))}
    by_pair, summary = score_conditions(S, cond, gem, np.ones(len(S), bool), ["A+B", "A+C"], sets,
                                        {g: float(np.linalg.norm(v)) for g, v in F.items()}, cands)
    row = lambda actions, rule, c: by_pair[(by_pair.actions == actions) & (by_pair.rule == rule) & (by_pair.condition == c)].iloc[0]
    ab, ac = row("mean_shift", "sum", "A+B"), row("mean_shift", "sum", "A+C")
    assert ab.miss_ratio < 1e-4 and np.isclose(ab.cos, 1, atol=1e-4) and np.isclose(ab.len_ratio, 1, atol=0.01)
    assert np.isclose(ac.sq_miss, 25, rtol=0.02) and ac.cos < 0.7             # the sum misses what addition cannot explain
    assert ab["cell_top1__doubles"] == 1.0 and ab["cell_top1__all_combinations"] == 1.0 and ab["centre_rank__all_combinations"] == 1.0
    # one vector only: A is the longer one; alone it misses by exactly |B|² = 9
    assert np.isclose(row("mean_shift", "stronger_single", "A+B").sq_miss, 9, rtol=0.02)
    assert row("mean_shift", "best_single", "A+B").sq_miss <= row("mean_shift", "stronger_single", "A+B").sq_miss
    # adding nothing: the miss is the whole shift, and every candidate ties
    zero = row("zero", "sum", "A+B")
    assert np.isclose(zero.miss_ratio, 1) and zero["centre_rank__doubles"] == 1.5
    # twice too long: same direction, length ratio 2, and it loses to the right length
    long = row("double_length", "sum", "A+B")
    assert np.isclose(long.cos, 1, atol=1e-4) and np.isclose(long.len_ratio, 2, atol=0.02) and np.isclose(long.miss_ratio, 1, atol=0.02)
    s = summary.set_index(["actions", "rule"])
    assert s.loc[("double_length", "sum"), "share_beats_mean_shift"] == 0.0
    assert s.loc[("mean_shift", "sum"), "share_sum_beats_best_single"] == 1.0
    assert s.loc[("mean_shift", "sum"), "n_candidates__all_combinations"] == 7
