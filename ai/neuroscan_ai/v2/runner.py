"""v2 2D smoke training runner for the run-control lifecycle.

``run(gate, plan)`` first verifies the approved, unchanged pre-run manifest; only then are cases
read and the model built. It trains on ``plan.aliases`` (train split), predicts the validation
cases ``plan.validation_aliases`` (val split) and writes every required output under case keys.
Smoke outputs are not performance results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from neuroscan_ai.reproducibility import case_keys
from neuroscan_ai.reproducibility.environment import capture_determinism
from neuroscan_ai.run_control.gate import GateResult, verify_approval
from neuroscan_ai.run_control.plan import RunPlan
from neuroscan_ai.v2.dataset import BRAIN_MASK_POLICY, PREPROCESS_POLICY, CaseFiles, load_case
from neuroscan_ai.v2.inference import dice, predict_probability
from neuroscan_ai.v2.model_2d import build_model, describe
from neuroscan_ai.v2.outputs import (
    save_checkpoint,
    write_aggregate,
    write_case_outputs,
    write_training_record,
)
from neuroscan_ai.v2.training import resolve_device, set_determinism, train_epochs

THRESHOLD_SOURCE = "smoke_test_fixed_not_validated"


@dataclass
class V2TwoDRunner:
    config: Mapping[str, Any]
    case_files: Mapping[str, CaseFiles]  # alias -> files (never written to any output)
    kind: str = "v2_2d_training_runner"

    @property
    def seed(self) -> int:
        return int(self.config["training"]["seed"])

    @property
    def threshold(self) -> float:
        return float(self.config["inference"]["threshold"])

    def apply_determinism(self) -> dict[str, Any]:
        """Apply the fixed smoke settings and return them as recorded by the manifests."""
        set_determinism(self.seed)
        tr = self.config["training"]
        return capture_determinism(
            torch,
            seed=self.seed,
            precision="fp32",
            amp=False,
            workers=int(tr.get("dataloader_workers", 0)),
            batch_size=int(tr["batch_size"]),
        )

    def run(self, gate: GateResult, plan: RunPlan) -> None:
        verify_approval(gate)  # before any data or model is touched
        cfg = self.config
        tr, inf = cfg["training"], cfg["inference"]
        set_determinism(self.seed)
        device = resolve_device(cfg["device"])
        train_cases = [load_case(self.case_files[a]) for a in plan.aliases]
        model = build_model(self.seed, int(cfg["model"]["base_channels"]))
        info = describe(model, self.seed, int(cfg["model"]["base_channels"]))
        history = train_epochs(
            model,
            train_cases,
            epochs=int(tr["epochs"]),
            batch_size=int(tr["batch_size"]),
            lr=float(tr["learning_rate"]),
            weight_decay=float(tr["weight_decay"]),
            seed=self.seed,
            pad_multiple=int(inf["pad_multiple"]),
            device=device,
        )
        run_dir = gate.run_dir
        meta = {**info, "epochs": int(tr["epochs"]), "optimizer": tr["optimizer"]}
        save_checkpoint(run_dir, model, meta)
        write_training_record(
            run_dir,
            "training_history",
            {
                "model": info,
                "device_type": device.type,
                "train_cases": len(train_cases),
                "optimizer": tr["optimizer"],
                "learning_rate": float(tr["learning_rate"]),
                "weight_decay": float(tr["weight_decay"]),
                "batch_size": int(tr["batch_size"]),
                "loss": "bce_with_logits + soft_dice",
                "history": history,
            },
        )
        write_training_record(
            run_dir,
            "threshold_record",
            {
                "threshold": self.threshold,
                "source": THRESHOLD_SOURCE,
                "optimized": False,
            },
        )

        keys = case_keys(plan.all_aliases)
        per_case = {}
        for alias in sorted(plan.validation_aliases, key=lambda a: keys[a]):
            key = keys[alias]
            case = load_case(self.case_files[alias])
            prob = predict_probability(
                model,
                case,
                device=device,
                pad_multiple=int(inf["pad_multiple"]),
                batch_size=int(inf["batch_size"]),
            )
            ungated = prob >= self.threshold
            gated = ungated & case.brain
            metric = {
                "case_key": key,
                "smoke_test_only": True,
                "gt_voxels": int(case.label.sum()),
                "ungated_voxels": int(ungated.sum()),
                "gated_voxels": int(gated.sum()),
                "ungated_outside_brain_voxels": int((ungated & ~case.brain).sum()),
                "dice_ungated": dice(ungated, case.label),
                "dice_gated": dice(gated, case.label),
                "probability_max": float(prob.max()),
            }
            write_case_outputs(
                run_dir,
                key,
                probability=prob,
                ungated=ungated,
                brain=case.brain,
                gated=gated,
                affine=case.affine,
                metric=metric,
            )
            per_case[key] = metric
        gated_d = [m["dice_gated"] for m in per_case.values() if m["dice_gated"] is not None]
        summary = {
            "smoke_test_only": True,
            "note": "smoke test reference values; not a performance result",
            "validation_cases": len(per_case),
            "threshold": self.threshold,
            "threshold_source": THRESHOLD_SOURCE,
            "dice_gated_mean": float(np.mean(gated_d)) if gated_d else None,
            "policies": {"preprocessing": PREPROCESS_POLICY, "brain_mask": BRAIN_MASK_POLICY},
        }
        write_training_record(run_dir, "validation_metrics", {"per_case": per_case, **summary})
        write_aggregate(run_dir, summary)
