from __future__ import annotations

import numpy as np
import pytest

from neuroscan_ai.data.split import SPLITS, SplitCase, make_patient_split


def synthetic_cases(n=250, n_neg=3, n_thick=54, seed=0):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        neg = i < n_neg
        cases.append(
            SplitCase(
                case_id=f"case{i:04d}",
                has_lesion=not neg,
                lesion_volume_ml=0.0 if neg else float(rng.lognormal(1.8, 1.6)),
                thick_slice=n_neg <= i < n_neg + n_thick,
            )
        )
    return cases


def test_split_overlap_zero_and_full_coverage():
    cases = synthetic_cases()
    s = make_patient_split(cases)["splits"]
    sets = [set(s[k]) for k in SPLITS]
    assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1 :])
    assert set().union(*sets) == {c.case_id for c in cases}
    assert sum(len(x) for x in sets) == len(cases)


def test_split_sizes_70_15_15():
    s = make_patient_split(synthetic_cases())["splits"]
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (176, 37, 37)


def test_split_is_deterministic_and_order_independent():
    cases = synthetic_cases()
    a = make_patient_split(cases, seed=42)
    b = make_patient_split(list(reversed(cases)), seed=42)
    assert a["splits"] == b["splits"]
    assert make_patient_split(cases, seed=7)["splits"] != a["splits"]


def test_negatives_are_spread_over_splits():
    s = make_patient_split(synthetic_cases(n_neg=3))
    assert [len(s["stats"][k]["negatives"]) for k in SPLITS] == [1, 1, 1]


def test_strata_are_spread_proportionally():
    s = make_patient_split(synthetic_cases())
    thick = {k: s["stats"][k]["thick_slice"] for k in SPLITS}
    assert sum(thick.values()) == 54
    assert 6 <= thick["val"] <= 10 and 6 <= thick["test"] <= 10


def test_duplicate_ids_rejected():
    c = synthetic_cases(n=10)
    with pytest.raises(ValueError):
        make_patient_split([*c, c[0]])
