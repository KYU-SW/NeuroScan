"""Phase 5B end-to-end API smoke test on validation patient sub-strokecase0086 with the real
Phase 4 checkpoint: upload -> run (single GPU worker) -> result -> mask download.

The API mask must equal the CLI service mask and the official Phase 4 prediction (D12 tolerance:
<= 0.1 % voxels on CUDA), and the Dice against the GT must match validation_metrics.csv within
0.001. The test split is never touched (SplitGuard access report checked)."""

from __future__ import annotations

import gzip
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from tests.conftest import REPO_ROOT

pytestmark = [
    pytest.mark.api,
    pytest.mark.data,
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]
pytest.importorskip("fastapi")

from neuroscan_ai.data.split_guard import SplitGuard  # noqa: E402
from neuroscan_ai.evaluation.segmentation import evaluate_case  # noqa: E402
from neuroscan_ai.inference import run_independent_inference  # noqa: E402
from neuroscan_ai.preprocessing.mask import binarize  # noqa: E402
from neuroscan_api.worker.loop import default_engine  # noqa: E402
from tests.api_helpers import API, client_for, make_settings, make_worker  # noqa: E402

CASE = "sub-strokecase0086"
PHASE4_RUN = REPO_ROOT / "results/phase4/25d_baseline_20261024-192617_34d18df104da"
CKPT = REPO_ROOT / "models/phase4/25d_baseline_20261024-192617_34d18df104da/best.pt"
META = REPO_ROOT / "data/processed/v1" / CASE / "meta.json"

if not (CKPT.is_file() and PHASE4_RUN.is_dir() and META.is_file()):
    pytest.skip("Phase 4 checkpoint/results or ISLES data not available", allow_module_level=True)


def test_validation_case_end_to_end_through_the_api(tmp_path):
    guard = SplitGuard(REPO_ROOT / "data/splits/split_v1_seed42.json")
    assert CASE in guard.cases("val")
    src = {k: REPO_ROOT / v for k, v in json.loads(META.read_text())["source"].items()}
    settings = make_settings(tmp_path, worker={"device": "auto"})  # real inference_v1 config
    with client_for(settings) as c:
        files = {
            "dwi": (src["dwi"].name, src["dwi"].read_bytes(), "application/gzip"),
            "adc": (src["adc"].name, src["adc"].read_bytes(), "application/gzip"),
        }
        r = c.post(f"{API}/cases", files=files)
        assert r.status_code == 201, r.text
        cid = r.json()["case_id"]
        rid = c.post(f"{API}/cases/{cid}/runs").json()["run_id"]
        w = make_worker(c, default_engine())
        try:
            assert w.load_count == 1
            assert c.get(f"{API}/models/current").json()["config_hash"] == "34d18df104da"
            assert w.run_once()
        finally:
            w.stop()
        st = c.get(f"{API}/runs/{rid}").json()
        assert st["status"] == "succeeded", st
        res = c.get(f"{API}/runs/{rid}/result").json()
        assert CASE not in json.dumps(res) and "strokecase" not in json.dumps(res)
        mask_url = next(v["url"] for v in res["viewer"]["volumes"] if v["id"] == "prediction_mask")
        api_img = nib.Nifti1Image.from_bytes(gzip.decompress(c.get(mask_url).content))
    api_mask = np.asanyarray(api_img.dataobj)
    dwi = nib.load(src["dwi"])
    assert api_mask.shape == dwi.shape and np.allclose(api_img.affine, dwi.affine, atol=1e-5)

    ref = np.asanyarray(nib.load(PHASE4_RUN / "predictions" / f"{CASE}_pred.nii.gz").dataobj)
    mismatch = float((api_mask != ref).mean())
    assert mismatch <= 1e-3, f"API vs Phase 4 mismatch {mismatch:.5%}"

    cli = run_independent_inference(src["dwi"], src["adc"], tmp_path / "cli", device="auto")
    cli_mask = np.asanyarray(nib.load(os.path.join(cli.run_dir, "prediction_mask.nii.gz")).dataobj)
    assert float((api_mask != cli_mask).mean()) <= 1e-3
    assert res["lesion"]["total_volume_ml"] == pytest.approx(
        cli.quantification["total_lesion_volume_ml"], abs=1e-3
    )

    gt = nib.load(src["label"])
    dice = evaluate_case(api_mask, binarize(gt.get_fdata()), gt.header.get_zooms()[:3], CASE)[
        "dice"
    ]
    phase4 = (
        pd.read_csv(PHASE4_RUN / "validation_metrics.csv").set_index("case_id").loc[CASE, "dice"]
    )
    assert abs(dice - phase4) <= 1e-3
    assert guard.access_report()["test_cases_accessed"] == []
    print(
        f"\n[api-smoke] mismatch_vs_phase4={int((api_mask != ref).sum())} voxels "
        f"mismatch_vs_cli={int((api_mask != cli_mask).sum())} dice={dice:.5f} phase4={phase4:.5f} "
        f"volume_ml={res['lesion']['total_volume_ml']} device={st['model']}"
    )
