"""RunPlan: what a v2 run is allowed to do. Refused at construction when it would touch a split the
stage may not use, the test / final-test split, or a SOOP pilot case."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STAGES = ("train", "validation", "inference", "external_validation")
DEFAULT_SPLITS = {"train": {"train"}, "validation": {"val"}}
ALWAYS_REFUSED_SPLITS = {"test", "final_test"}
SOOP_PILOT = re.compile(r"^soop-pilot-\d+$")
POLICY_KEYS = ("preprocessing", "inference", "threshold", "brain_mask", "combination")
VALIDATION_OUTPUTS = (
    "probability_maps",
    "ungated_predictions",
    "brain_masks",
    "gated_predictions",
    "patient_metrics",
    "aggregate_results",
)
TRAIN_OUTPUTS = ("checkpoint", "training_history", "validation_metrics", "selected_threshold")


class RunPlanError(ValueError):
    pass


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    purpose: str
    stage: str
    dataset_id: str
    split_name: str
    aliases: tuple[str, ...]
    allowed_case_count: int
    model_id: str
    checkpoint: Path
    config: Path
    split_file: Path
    policies: dict[str, str]
    output_root: Path
    approved_splits: tuple[str, ...] = ()  # inference / external validation: explicit approval
    expected: dict[str, Any] = field(default_factory=dict)  # hashes / versions the gate must see
    excluded_aliases: tuple[str, ...] = ()  # e.g. permanently excluded cases

    @property
    def required_outputs(self) -> tuple[str, ...]:
        return TRAIN_OUTPUTS if self.stage == "train" else VALIDATION_OUTPUTS

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_id


def _split_members(split_file: Path) -> dict[str, set[str]]:
    data = json.loads(Path(split_file).read_text(encoding="utf-8"))
    splits = data.get("splits", data)
    return {k: set(v) for k, v in splits.items() if isinstance(v, list)}


def build_plan(**kw: Any) -> RunPlan:
    """Validated RunPlan; raises RunPlanError for any forbidden access."""
    kw["aliases"] = tuple(kw.get("aliases", ()))
    kw["approved_splits"] = tuple(kw.get("approved_splits", ()))
    kw["excluded_aliases"] = tuple(kw.get("excluded_aliases", ()))
    plan = RunPlan(**kw)
    if not re.match(r"^[a-z0-9][a-z0-9._-]{2,63}$", plan.run_id):
        raise RunPlanError("run_id must be lower-case [a-z0-9._-], 3-64 chars")
    if plan.stage not in STAGES:
        raise RunPlanError(f"unknown stage '{plan.stage}'")
    if plan.split_name in ALWAYS_REFUSED_SPLITS:
        raise RunPlanError("test and final-test splits are refused by default")
    allowed = DEFAULT_SPLITS.get(plan.stage) or set(plan.approved_splits) - ALWAYS_REFUSED_SPLITS
    if plan.split_name not in allowed:
        raise RunPlanError(f"stage '{plan.stage}' may not access split '{plan.split_name}'")
    missing = [k for k in POLICY_KEYS if not plan.policies.get(k)]
    if missing:
        raise RunPlanError(f"missing policy IDs: {missing}")
    if not plan.aliases or len(plan.aliases) != plan.allowed_case_count:
        raise RunPlanError("case list size differs from allowed_case_count")
    if len(set(plan.aliases)) != len(plan.aliases):
        raise RunPlanError("duplicate case aliases")
    blocked = [a for a in plan.aliases if SOOP_PILOT.match(a) or a in plan.excluded_aliases]
    if blocked:
        raise RunPlanError(f"{len(blocked)} permanently excluded case(s) in the plan")
    members = _split_members(plan.split_file)
    if plan.split_name not in members:
        raise RunPlanError(f"split '{plan.split_name}' not found in the split file")
    if set(plan.aliases) - members[plan.split_name]:
        raise RunPlanError("plan contains cases outside the declared split")
    for other, ids in members.items():
        if other != plan.split_name and set(plan.aliases) & ids:
            raise RunPlanError("plan contains cases that also belong to another split")
    return plan
