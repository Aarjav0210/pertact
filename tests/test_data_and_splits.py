"""Guide parsing, perturbation labels and the split, on tiny hand-made inputs."""
import numpy as np
import pandas as pd

from pertact.data import CONTROL, cell_labels, condition_genes, condition_name, construct_genes, parse_guide_id
from pertact.splits import make_splits


def test_construct_genes_drops_controls_and_suffixes_and_sorts():
    assert construct_genes("KLF1_NegCtrl0", "NegCtrl") == ("KLF1",)
    assert construct_genes("NegCtrl0_KLF1", "NegCtrl") == ("KLF1",)
    assert construct_genes("TGFBR2_IGDCC3_2", "NegCtrl") == ("IGDCC3", "TGFBR2")
    assert construct_genes("NegCtrl10_NegCtrl0", "NegCtrl") == ()


def test_parse_guide_id_accepts_agreeing_constructs_only():
    assert parse_guide_id("KLF1_NegCtrl0;NegCtrl0_KLF1", "NegCtrl") == (("KLF1",), "ok")
    assert parse_guide_id("KLF1_NegCtrl0;SET_NegCtrl0", "NegCtrl") == (None, "ambiguous")
    assert parse_guide_id("KLF1_KLF1", "NegCtrl") == (None, "ambiguous")


def test_condition_labels_round_trip():
    assert condition_name(()) == CONTROL and condition_genes(CONTROL) == []
    assert condition_name(("TGFBR2", "IGDCC3")) == "IGDCC3+TGFBR2"
    assert condition_genes("IGDCC3+TGFBR2") == ["IGDCC3", "TGFBR2"]


def toy_obs():
    rows = []
    for cond, n_targets, n in ((CONTROL, 0, 80), ("A", 1, 40), ("B", 1, 40), ("A+B", 2, 20), ("A+Z", 2, 20)):
        rows += [{"condition": cond, "n_targets": n_targets, "gemgroup": str(i % 2 + 1)} for i in range(n)]
    return pd.DataFrame(rows, index=[f"cell{i}" for i in range(len(rows))])


def test_split_is_a_partition_with_no_double_in_train_or_val():
    obs = toy_obs()
    cfg = {"seed": 0, "data": {"batch_key": "gemgroup"},
           "splits": {"val_frac_per_single": 0.2, "control_fracs": {"train": 0.7, "val": 0.15, "test": 0.15}}}
    sp = make_splits(obs, cfg)
    cells = sp["cells"]
    everything = cells["train"] + cells["val"] + cells["test"] + cells["excluded_doubles"]
    assert sorted(everything) == sorted(obs.index)
    doubles = set(obs.index[obs.n_targets == 2])
    assert not doubles & set(cells["train"]) and not doubles & set(cells["val"])
    assert sp["test_doubles"] == ["A+B"] and sp["excluded_doubles"] == ["A+Z"]          # Z was never perturbed alone
    assert sum(obs.loc[cells["val"], "condition"] == "A") == 8                            # 20% of 40
    assert make_splits(obs, cfg)["cells"] == cells                                        # same seed, same split


def test_cell_labels_are_plain_strings():
    obs = toy_obs().assign(split="train")
    cond, gem, split = cell_labels(obs, "gemgroup")
    assert cond.dtype.kind in "OU" and set(gem) == {"1", "2"} and (split == "train").all()
