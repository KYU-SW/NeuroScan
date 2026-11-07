"""Mock validation / inference runner (TEST ONLY: synthetic arrays, no model, no image input).

It exists to exercise the controlled run flow. Its outputs imitate the shape of real outputs and
must never be used as clinical or performance results. It refuses to start without an approved,
unchanged pre-run manifest. ``faults`` injects failures for tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neuroscan_ai.reproducibility import case_keys
from neuroscan_ai.run_control.gate import GateResult, verify_approval
from neuroscan_ai.run_control.outputs import (
    AGGREGATE_KIND,
    METRIC_KIND,
    aggregate_path,
    case_path,
    save_image,
)
from neuroscan_ai.run_control.plan import RunPlan

SHAPE = (16, 16, 8)
AFFINE = np.diag([1.0, 1.0, 2.0, 1.0])
FAULTS = {
    "skip_probability_maps",
    "skip_ungated_predictions",
    "skip_brain_masks",
    "skip_gated_predictions",
    "skip_patient_metrics",
    "skip_aggregate_results",
    "nan_probability",
    "non_binary_mask",
    "shape_mismatch",
    "affine_mismatch",
    "raise",
}


@dataclass
class MockRunner:
    threshold: float = 0.5
    seed: int = 0
    faults: set[str] = field(default_factory=set)
    kind: str = "mock_synthetic_runner"

    def __post_init__(self) -> None:
        unknown = set(self.faults) - FAULTS
        if unknown:
            raise ValueError(f"unknown faults: {sorted(unknown)}")

    def _arrays(self, index: int):
        rng = np.random.default_rng(self.seed + index)
        prob = rng.random(SHAPE).astype(np.float32)
        g = np.indices(SHAPE).astype(float)
        c = (np.array(SHAPE) - 1) / 2
        mask = (((g[0] - c[0]) / 6) ** 2 + ((g[1] - c[1]) / 6) ** 2 + ((g[2] - c[2]) / 3) ** 2) <= 1
        return prob, mask

    def run(self, gate: GateResult, plan: RunPlan) -> None:
        verify_approval(gate)
        if "raise" in self.faults:
            raise RuntimeError("mock runner failure")
        run_dir = gate.run_dir
        keys = sorted(case_keys(plan.aliases).values())
        totals = {"ungated_voxels": 0, "gated_voxels": 0}
        for i, key in enumerate(keys):
            prob, mask = self._arrays(i)
            ung = prob >= self.threshold
            gat = ung & mask
            totals["ungated_voxels"] += int(ung.sum())
            totals["gated_voxels"] += int(gat.sum())
            if "nan_probability" in self.faults and i == 0:
                prob = prob.copy()
                prob[0, 0, 0] = np.nan
            mask_out = mask.astype(np.uint8)
            if "non_binary_mask" in self.faults and i == 0:
                mask_out = mask_out.copy()
                mask_out[0, 0, 0] = 2
            aff_gated = AFFINE.copy()
            if "affine_mismatch" in self.faults and i == 0:
                aff_gated[0, 3] = 5.0
            gated_out = gat.astype(np.uint8)
            if "shape_mismatch" in self.faults and i == 0:
                gated_out = gated_out[:, :, :-1]
            outs = {
                "probability_maps": (prob, AFFINE),
                "ungated_predictions": (ung.astype(np.uint8), AFFINE),
                "brain_masks": (mask_out, AFFINE),
                "gated_predictions": (gated_out, aff_gated),
            }
            for kind, (data, aff) in outs.items():
                if f"skip_{kind}" not in self.faults:
                    save_image(case_path(run_dir, kind, key), data, aff)
            if "skip_patient_metrics" not in self.faults:
                _write_json(
                    case_path(run_dir, METRIC_KIND, key),
                    {
                        "case_key": key,
                        "synthetic": True,
                        "ungated_voxels": int(ung.sum()),
                        "gated_voxels": int(gat.sum()),
                    },
                )
        if "skip_aggregate_results" not in self.faults:
            _write_json(
                aggregate_path(run_dir),
                {
                    "synthetic": True,
                    "cases": len(keys),
                    "threshold": self.threshold,
                    "note": "mock runner output; not a clinical or performance result",
                    **totals,
                },
            )


def _write_json(path: Path, obj: dict) -> None:
    if path.exists():
        raise FileExistsError("output exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


__all__ = ["AGGREGATE_KIND", "FAULTS", "MockRunner"]
