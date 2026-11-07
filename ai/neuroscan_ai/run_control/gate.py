"""Pre-run gate: every check passes -> pre-run manifest written and frozen -> ``approved``.

Any failure -> ``rejected`` and the runner is never called. A rejection is recorded in a new run
folder (audit trail) unless the folder already exists, which is itself a rejection reason.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuroscan_ai.component_analysis.records import is_git_ignored
from neuroscan_ai.reproducibility import (
    CASE_KEY_POLICY,
    ManifestError,
    PathRoots,
    RunSpec,
    build_pre_run,
    check_schema,
    write_manifest,
)
from neuroscan_ai.reproducibility.hashing import (
    alias_list_sha256,
    canonical_json_sha256,
    sha256_file,
    sha256_text_lf,
)
from neuroscan_ai.run_control.plan import RunPlan

PRE_RUN_NAME = "pre_run_manifest.json"
REJECTION_NAME = "gate_rejection.json"
APPROVED, REJECTED = "approved", "rejected"
REQUIRED_EXPECTED = (
    "config_sha256_lf",
    "split_sha256_lf",
    "checkpoint_sha256",
    "package_lock_sha256_lf",
    "alias_list_sha256",
)


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class GateResult:
    status: str
    reasons: tuple[str, ...]
    run_dir: Path
    pre_run_sha256: str | None = None
    pre_run: Mapping[str, Any] | None = None

    @property
    def approved(self) -> bool:
        return self.status == APPROVED


def _get(d: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = d
    for k in dotted.split("."):
        if not isinstance(cur, Mapping) or k not in cur:
            return None
        cur = cur[k]
    return cur


def pre_run_gate(
    plan: RunPlan,
    *,
    repo_root: Path,
    roots: PathRoots,
    source_control: Mapping[str, Any],
    environment: Mapping[str, Any],
    determinism: Mapping[str, Any],
    forbidden_ids: Iterable[str] = (),
) -> GateResult:
    reasons: list[str] = []
    exp = plan.expected
    if not source_control.get("git_commit"):
        reasons.append("git_commit_unknown")
    if source_control.get("working_tree_clean") is not True:
        reasons.append("working_tree_not_clean")
    required = [
        k for k in REQUIRED_EXPECTED if k != "checkpoint_sha256" or plan.checkpoint is not None
    ]
    missing = [k for k in required if not exp.get(k)]
    if missing:
        reasons.append("expected_hashes_not_declared:" + ",".join(missing))
    actual = {
        "config_sha256_lf": sha256_text_lf(plan.config),
        "split_sha256_lf": sha256_text_lf(plan.split_file),
        "checkpoint_sha256": sha256_file(plan.checkpoint) if plan.checkpoint else None,
        "package_lock_sha256_lf": environment.get("package_lock_sha256_lf"),
        "alias_list_sha256": alias_list_sha256(plan.all_aliases, CASE_KEY_POLICY),
    }
    for k in required:
        if exp.get(k) and exp[k] != actual[k]:
            reasons.append(f"{k}_mismatch")
    if len(plan.aliases) != plan.allowed_case_count:
        reasons.append("case_count_mismatch")
    for dotted, value in sorted(exp.get("environment", {}).items()):
        if _get(environment, dotted) != value:
            reasons.append(f"environment_mismatch:{dotted}")
    for key, value in sorted(exp.get("determinism", {}).items()):
        if determinism.get(key) != value:
            reasons.append(f"determinism_mismatch:{key}")
    run_dir = plan.run_dir
    if not is_git_ignored(run_dir / "probe", repo_root):
        reasons.append("output_dir_not_git_ignored")
    existed = run_dir.exists()
    if existed:
        reasons.append("output_dir_exists")

    pre = None
    if not reasons:
        spec = RunSpec(
            run_id=plan.run_id,
            stage=plan.stage,
            purpose=plan.purpose,
            dataset_id=plan.dataset_id,
            split_name=plan.split_name,
            aliases=list(plan.all_aliases),
            config=plan.config,
            split=plan.split_file,
            checkpoint=plan.checkpoint,
            policies=dict(plan.policies),
            model={"architecture_id": plan.model_id},
            input_files={a: dict(roles) for a, roles in plan.input_files.items()},
            access={
                **{f"{s}_cases": n for s, n in plan.split_counts.items()},
                "allowed_splits": sorted(plan.split_counts),
            },
        )
        try:
            pre = build_pre_run(
                spec,
                roots=roots,
                source_control=source_control,
                environment=environment,
                determinism=determinism,
                forbidden_ids=forbidden_ids,
            )
        except ManifestError as e:
            reasons.append(f"manifest_refused:{e}")
    if reasons:
        if not existed and "output_dir_not_git_ignored" not in reasons:
            run_dir.mkdir(parents=True)
            body = {"run_id": plan.run_id, "pre_run_status": REJECTED, "reasons": reasons}
            (run_dir / REJECTION_NAME).write_text(json.dumps(body, indent=2), encoding="utf-8")
        return GateResult(REJECTED, tuple(reasons), run_dir)
    run_dir.mkdir(parents=True)
    write_manifest(run_dir / PRE_RUN_NAME, pre, forbidden_ids, plan.all_aliases)
    return GateResult(APPROVED, (), run_dir, canonical_json_sha256(pre), pre)


def verify_approval(gate: GateResult | None) -> Mapping[str, Any]:
    """The runner's precondition: an approved gate whose frozen pre-run manifest is unchanged."""
    if gate is None or not gate.approved or gate.pre_run_sha256 is None:
        raise GateError("runner refused: no approved pre-run manifest")
    path = gate.run_dir / PRE_RUN_NAME
    if not path.is_file():
        raise GateError("runner refused: pre-run manifest file missing")
    stored = json.loads(path.read_text(encoding="utf-8"))
    check_schema(stored, "pre_run")
    if canonical_json_sha256(stored) != gate.pre_run_sha256:
        raise GateError("runner refused: pre-run manifest changed after approval")
    return stored
