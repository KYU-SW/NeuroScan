"""NeuroScan v2 controlled run flow: no run without an approved pre-run manifest.

plan (RunPlan, split access control) -> gate (pre-run checks and manifest) -> runner (only with an
approved, unchanged pre-run manifest) -> outputs (presence and relation checks) -> lifecycle
(post-run manifest, also for failures). Only a synthetic MockRunner exists; connecting a real v2
runner is a separate step. Nothing here imports the v1 runners, PyTorch or MONAI.
"""

from neuroscan_ai.run_control.gate import APPROVED, REJECTED, GateError, GateResult, pre_run_gate
from neuroscan_ai.run_control.lifecycle import (
    COMPLETED,
    FAILED_INCOMPLETE,
    FAILED_MANIFEST,
    FAILED_RUNNER,
    FAILED_VALIDATION,
    load_post_run,
    run_controlled,
    verify_run,
)
from neuroscan_ai.run_control.plan import RunPlan, RunPlanError, build_plan
from neuroscan_ai.run_control.runner import MockRunner

__all__ = [
    "APPROVED",
    "COMPLETED",
    "FAILED_INCOMPLETE",
    "FAILED_MANIFEST",
    "FAILED_RUNNER",
    "FAILED_VALIDATION",
    "REJECTED",
    "GateError",
    "GateResult",
    "MockRunner",
    "RunPlan",
    "RunPlanError",
    "build_plan",
    "load_post_run",
    "pre_run_gate",
    "run_controlled",
    "verify_run",
]
