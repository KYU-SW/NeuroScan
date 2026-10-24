"""Phase 3 integration tests on real data — train split only (validation/test never opened)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuroscan_ai.data.split_guard import SealedSplitAccessError, SplitGuard
from neuroscan_ai.training.config import load_config
from neuroscan_ai.training.dataset import SliceDataset, load_split_cases, norm_specs
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.data

CFG = load_config(REPO_ROOT / "configs" / "2d_baseline.yaml")
PROCESSED = REPO_ROOT / CFG["data"]["processed_root"]
SPLIT = REPO_ROOT / CFG["data"]["split_file"]
METADATA = REPO_ROOT / CFG["data"]["case_metadata"]


@pytest.fixture(scope="module")
def guard():
    if not (PROCESSED / "cases.csv").exists() or not METADATA.exists():
        pytest.skip("processed/v1 or Phase 2 case metadata not available")
    return SplitGuard(SPLIT)


@pytest.fixture(scope="module")
def train_cases(guard):
    md = pd.read_csv(METADATA).set_index("case_id")
    train = guard.cases("train")
    picks = ["sub-strokecase0046"]
    for group in sorted(md.loc[train, "adc_unit_group"].unique()):
        picks += [c for c in train if md.loc[c, "adc_unit_group"] == group and c not in picks][:2]
    cases = load_split_cases(
        guard,
        "train",
        PROCESSED,
        CFG["data"]["modalities"],
        norm_specs(CFG["data"]["normalization"]),
        picks,
    )
    return cases, md


def test_real_split_test_access_is_sealed(guard):
    with pytest.raises(SealedSplitAccessError):
        guard.cases("test")
    with pytest.raises(SealedSplitAccessError):
        load_split_cases(guard, "test", PROCESSED, ["dwi", "adc"], {})


def test_normalised_real_volumes_are_finite_without_extremes(train_cases):
    cases, _ = train_cases
    for c in cases:
        assert c.image.dtype == np.float32 and np.isfinite(c.image).all(), c.case_id
        assert np.abs(c.image).max() < 10, c.case_id  # clip p0.5-p99.5 bounds the z-scores


def test_adc_unit_groups_are_harmonised(train_cases):
    cases, md = train_cases
    med = {}
    for c in cases:
        brain = c.image[0] != 0
        med.setdefault(md.loc[c.case_id, "adc_unit_group"], []).append(
            float(np.median(c.image[1][brain]))
        )
    assert len(med) == 3
    group_medians = [np.mean(v) for v in med.values()]
    assert max(group_medians) - min(group_medians) < 0.2  # raw medians differ by ~10^6


def test_real_slices_align_with_processed_mask(train_cases):
    cases, _ = train_cases
    ds = SliceDataset(cases, context=0, spatial_size=CFG["data"]["train_spatial_size"])
    stats = ds.stats()
    assert stats["positive_slices"] > 0 and stats["negative_slices"] > 0
    for i in np.flatnonzero(ds.slice_positive)[:20]:
        s = ds[int(i)]
        assert s["image"].shape == (2, 256, 256) and s["label"].sum() > 0
        assert s["slice_has_lesion"]


def test_negative_patient_brain_slices_are_in_training_dataset(guard):
    cases = load_split_cases(
        guard,
        "train",
        PROCESSED,
        CFG["data"]["modalities"],
        norm_specs(CFG["data"]["normalization"]),
        ["sub-strokecase0150"],
        CFG["data"]["brain_mask_policy"],
    )
    assert not cases[0].has_lesion
    ds = SliceDataset(cases, exclude_brain_free_slices=CFG["data"]["exclude_brain_free_slices"])
    brain_slices = int(cases[0].brain.any(axis=(0, 1)).sum())
    assert len(ds) == brain_slices > 0  # every brain-positive slice of the negative patient
    assert ds.n_brain_free_excluded == cases[0].n_slices - brain_slices
