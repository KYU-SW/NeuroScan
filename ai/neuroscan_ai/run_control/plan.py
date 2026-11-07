"""RunPlan: what a v2 run is allowed to do. Refused at construction when it would touch a split the
stage may not use, the test / final-test split, or a SOOP pilot case.

A ``train`` run may also declare validation cases from the ``val`` split (``validation_aliases``);
its per-case outputs are then produced for those validation cases only.
"""

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
TRAIN_OUTPUTS = ("checkpoint", "training_history", "validation_metrics", "threshold_record")
VALIDATION_SPLIT = "val"


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
    checkpoint: Path | None  # input checkpoint; None only for a train stage
    config: Path
    split_file: Path
    policies: dict[str, str]
    output_root: Path
    approved_splits: tuple[str, ...] = ()  # inference / external validation: explicit approval
    expected: dict[str, Any] = field(default_factory=dict)  # hashes / versions the gate must see
    excluded_aliases: tuple[str, ...] = ()  # e.g. permanently excluded cases
    validation_aliases: tuple[str, ...] = ()  # train stage only: cases from the val split
    # alias -> role -> input file; only SHA-256 hashes keyed by case key reach the manifest
    input_files: dict[str, dict[str, Path]] = field(default_factory=dict)

    @property
    def required_outputs(self) -> tuple[str, ...]:
        return TRAIN_OUTPUTS + VALIDATION_OUTPUTS if self.stage == "train" else VALIDATION_OUTPUTS

    @property
    def all_aliases(self) -> tuple[str, ...]:
        """Every case the run may read (manifest alias list and case keys)."""
        return self.aliases + self.validation_aliases

    @property
    def output_aliases(self) -> tuple[str, ...]:
        """Cases that get per-case outputs."""
        return self.validation_aliases if self.stage == "train" else self.aliases

    @property
    def split_counts(self) -> dict[str, int]:
        counts = {self.split_name: len(self.aliases)}
        if self.validation_aliases:
            counts[VALIDATION_SPLIT] = len(self.validation_aliases)
        return counts

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
    kw["validation_aliases"] = tuple(kw.get("validation_aliases", ()))
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
    if plan.checkpoint is None and plan.stage != "train":
        raise RunPlanError("an input checkpoint is required outside the train stage")
    if plan.validation_aliases and plan.stage != "train":
        raise RunPlanError("validation_aliases are only allowed for the train stage")
    if plan.stage == "train" and not plan.validation_aliases:
        raise RunPlanError("a train run must declare validation cases from the val split")
    if len(set(plan.all_aliases)) != len(plan.all_aliases):
        raise RunPlanError("duplicate case aliases")
    if set(plan.input_files) - set(plan.all_aliases):
        raise RunPlanError("input files given for cases outside the plan")
    blocked = [a for a in plan.all_aliases if SOOP_PILOT.match(a) or a in plan.excluded_aliases]
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
    if plan.validation_aliases:
        if (
            VALIDATION_SPLIT not in members
            or set(plan.validation_aliases) - members[VALIDATION_SPLIT]
        ):
            raise RunPlanError("validation cases must belong to the val split")
        for other, ids in members.items():
            if other != VALIDATION_SPLIT and set(plan.validation_aliases) & ids:
                raise RunPlanError("validation cases also belong to another split")
    return plan
