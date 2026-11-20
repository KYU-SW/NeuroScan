"""Controlled run lifecycle (v2): plan -> pre-run gate -> runner -> output checks -> post-run.

1 plan validated (RunPlan) 2 pre-run manifest built 3 its hash frozen 4 runner called only with an
approved gate 5 required outputs present 6 output hashes recorded 7 post-run manifest written
8 pre/post link verified 9 final status. Failed runs also get a post-run manifest and their run
folder is kept; a run_id is never reused.

Interruption and resume (runners that support them, e.g. the v2 2D baseline):
- A runner stops at a safe point by raising ``RunInterrupted`` (status ``interrupted`` or
  ``interrupted_resource_guard``) after writing its own resume state. The lifecycle then writes a
  SEGMENT post-run manifest and seal into the runner's segment folder
  (``segments/segment_NN/``), never the run's ``post_run_manifest.json``: the run is not
  finalised and may be resumed.
- ``resume_controlled`` continues such a run under the SAME run ID only when explicitly called:
  the frozen pre-run manifest must exist, the run must not be finalised (no
  ``post_run_manifest.json``: completed and failed runs are never resumed), and the git commit,
  clean tree, config / split / alias-list / every input file hash, package lock, full
  environment and determinism must equal the pre-run manifest; the runner then checks its own
  resume state (``resume_preflight``). Any difference refuses the resume before anything is
  written. A finished resumed run gets the normal post-run manifest and seal.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from neuroscan_ai.reproducibility import (
    CASE_KEY_POLICY,
    ManifestError,
    PathRoots,
    build_post_run,
    case_keys,
    check_schema,
    is_linked,
    write_manifest,
)
from neuroscan_ai.reproducibility.hashing import (
    alias_list_sha256,
    canonical_json_sha256,
    sha256_file,
    sha256_text_lf,
)
from neuroscan_ai.reproducibility.manifest import guard_public
from neuroscan_ai.run_control.gate import APPROVED, PRE_RUN_NAME, GateResult, pre_run_gate
from neuroscan_ai.run_control.outputs import (
    AGGREGATE_KIND,
    AGGREGATE_LABEL,
    IMAGE_KINDS,
    METRIC_KIND,
    OPTIONAL_RUN_FILES,
    aggregate_path,
    case_path,
    check_case,
    run_output_path,
    training_output_labels,
)
from neuroscan_ai.run_control.plan import VALIDATION_OUTPUTS, RunPlan

POST_RUN_NAME = "post_run_manifest.json"
SEAL_NAME = "post_run_manifest.sha256"  # canonical JSON SHA-256 of the post-run manifest
COMPLETED = "completed"
FAILED_INCOMPLETE = "failed_incomplete_outputs"
FAILED_VALIDATION = "failed_output_validation"
FAILED_RUNNER = "failed_runner_error"
FAILED_MANIFEST = "failed_manifest"
INTERRUPTED = "interrupted"
INTERRUPTED_RESOURCE_GUARD = "interrupted_resource_guard"
INTERRUPTED_STATUSES = (INTERRUPTED, INTERRUPTED_RESOURCE_GUARD)
INTERRUPTED_EXIT_CODE = 3
SEGMENTS_DIR = "segments"


class Runner(Protocol):
    threshold: float

    def run(self, gate: GateResult, plan: RunPlan) -> None: ...


class RunInterrupted(Exception):  # noqa: N818 - a controlled stop, not an error
    """Raised by a runner that stopped at a safe point (its resume state is already written)."""

    def __init__(self, status: str, segment_dir: Path, record: Mapping[str, Any]):
        if status not in INTERRUPTED_STATUSES:
            raise ValueError(f"unknown interruption status '{status}'")
        self.status, self.segment_dir, self.record = status, Path(segment_dir), dict(record)
        super().__init__(status)


class ResumeRefusedError(RuntimeError):
    """A resume condition failed; nothing was written."""

    def __init__(self, reasons: Iterable[str]):
        self.reasons = tuple(reasons)
        super().__init__("resume refused: " + ", ".join(self.reasons))


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _safe_message(msg: str, aliases: Iterable[str], forbidden_ids: Iterable[str]) -> str:
    try:
        guard_public(msg, forbidden_ids, aliases)
    except ManifestError:
        return "[redacted]"
    return msg[:200]


def _required_outputs(plan: RunPlan, training_labels: tuple[str, ...]) -> tuple[str, ...]:
    """Plan outputs with the train-stage labels taken from the runner's output profile."""
    if plan.stage != "train":
        return plan.required_outputs
    return training_labels + VALIDATION_OUTPUTS


def _segment_count(run_dir: Path) -> int:
    d = Path(run_dir) / SEGMENTS_DIR
    return sum(1 for p in d.iterdir() if p.is_dir()) if d.is_dir() else 0


def run_controlled(
    plan: RunPlan,
    runner: Runner,
    *,
    repo_root: Path,
    roots: PathRoots,
    source_control: Mapping[str, Any],
    environment: Mapping[str, Any],
    determinism: Mapping[str, Any],
    forbidden_ids: Iterable[str] = (),
) -> dict[str, Any]:
    forbidden_ids = tuple(forbidden_ids)
    # required training outputs are fixed before the gate (unknown profile -> refused)
    training_labels = training_output_labels(getattr(runner, "training_outputs", "smoke_v1"))
    # the seed actually used by the runner is what both manifests record
    runner_seed = getattr(runner, "seed", None)
    if determinism.get("seed") is not None and determinism.get("seed") != runner_seed:
        raise ValueError("declared determinism seed differs from the runner seed")
    if getattr(plan, "seed", None) is not None and plan.seed != runner_seed:
        raise ValueError("run plan seed differs from the runner seed")  # before the gate
    determinism = {**determinism, "seed": runner_seed}
    # a runner that controls framework settings applies them now, so both manifests record the
    # state the run actually uses (not the state before the runner configured it)
    if hasattr(runner, "apply_determinism"):
        determinism = {**determinism, **runner.apply_determinism(), "seed": runner_seed}
    gate = pre_run_gate(
        plan,
        repo_root=repo_root,
        roots=roots,
        source_control=source_control,
        environment=environment,
        determinism=determinism,
        forbidden_ids=forbidden_ids,
    )
    if not gate.approved:
        return {"run_id": plan.run_id, "pre_run_status": "rejected", "reasons": list(gate.reasons)}
    return _execute(
        plan,
        runner,
        gate,
        runner.run,
        training_labels,
        repo_root=repo_root,
        environment=environment,
        determinism=determinism,
        forbidden_ids=forbidden_ids,
    )


def _execute(
    plan: RunPlan,
    runner: Runner,
    gate: GateResult,
    call: Callable[[GateResult, RunPlan], None],
    training_labels: tuple[str, ...],
    *,
    repo_root: Path,
    environment: Mapping[str, Any],
    determinism: Mapping[str, Any],
    forbidden_ids: tuple[str, ...],
    resumed: bool = False,
) -> dict[str, Any]:
    started = _now()
    failed_stage = error_type = error_message = None
    interrupted: RunInterrupted | None = None
    try:
        call(gate, plan)
    except RunInterrupted as e:  # safe stop: segment manifest, the run is not finalised
        interrupted = e
    except Exception as e:  # noqa: BLE001 - recorded, never re-raised past the audit trail
        failed_stage, error_type = "runner", type(e).__name__
        error_message = _safe_message(str(e), plan.all_aliases, forbidden_ids)
    ended = _now()

    keys = case_keys(plan.all_aliases)
    generated: dict[str, dict[str, Path]] = {k: {} for k in (*IMAGE_KINDS, METRIC_KIND)}
    missing: list[str] = []
    check_errors: list[str] = []
    case_ok = 0
    for alias, key in sorted(((a, keys[a]) for a in plan.output_aliases), key=lambda kv: kv[1]):
        present = True
        for kind in (*IMAGE_KINDS, METRIC_KIND):
            p = case_path(gate.run_dir, kind, key)
            if p.is_file():
                generated[kind][alias] = p
            else:
                missing.append(f"{key}:{kind}")
                present = False
        errs = check_case(gate.run_dir, key, runner.threshold) if present else []
        check_errors += [f"{key}:{e}" for e in errs]
        case_ok += int(present and not errs)
    agg = aggregate_path(gate.run_dir)
    outputs: dict[str, dict[str, Path]] = {k: v for k, v in generated.items() if v}
    if agg.is_file():
        outputs[AGGREGATE_KIND] = {AGGREGATE_LABEL: agg}
    else:
        missing.append(f"run:{AGGREGATE_KIND}")
    if plan.stage == "train":  # checkpoint, history, validation metrics, threshold record
        for label in training_labels:
            p = run_output_path(gate.run_dir, label)
            if p.is_file():
                outputs.setdefault(AGGREGATE_KIND, {})[label] = p
            else:
                missing.append(f"run:{label}")
    for label in OPTIONAL_RUN_FILES:  # e.g. a failing runner's failure record
        p = run_output_path(gate.run_dir, label)
        if p.is_file():
            outputs.setdefault(AGGREGATE_KIND, {})[label] = p

    if interrupted is not None:
        status, failed_stage = interrupted.status, "interrupted"
    elif failed_stage:
        status = FAILED_RUNNER
    elif missing:
        status, failed_stage = FAILED_INCOMPLETE, "required_outputs"
    elif check_errors:
        status, failed_stage = FAILED_VALIDATION, "output_relations"
    else:
        status = COMPLETED
    record = {
        "pre_run_status": "approved",
        "failed_stage": failed_stage,
        "error_type": error_type,
        "error_message": error_message,
        "missing_outputs": missing,
        "output_check_errors": check_errors,
        "generated_output_files": sum(len(v) for v in outputs.values()),
        "accessed_splits": [plan.split_name],
        "runner_kind": getattr(runner, "kind", type(runner).__name__),
        "required_outputs": list(_required_outputs(plan, training_labels)),
        "resumed": resumed,
        "interrupted_segments": _segment_count(gate.run_dir),
    }
    if interrupted is not None:
        record["interruption"] = interrupted.record
    access = dict(gate.pre_run["access"]) if gate.pre_run else {}
    exit_code = 0 if status == COMPLETED else INTERRUPTED_EXIT_CODE if interrupted else 1
    post = build_post_run(
        gate.pre_run,
        aliases=plan.all_aliases,
        started_utc=started,
        ended_utc=ended,
        status=status,
        exit_code=exit_code,
        cases_succeeded=case_ok,
        cases_failed=len(plan.output_aliases) - case_ok,
        outputs=outputs,
        environment=environment,
        determinism=determinism,
        access=access,
        reproducibility=record,
        forbidden_ids=forbidden_ids,
    )
    if not is_linked(gate.pre_run, post):
        status = FAILED_MANIFEST
    target = interrupted.segment_dir if interrupted is not None else Path(gate.run_dir)
    if not target.resolve().is_relative_to(Path(gate.run_dir).resolve()):
        raise ValueError("segment folder outside the run folder")
    seal = target / SEAL_NAME
    if (target / POST_RUN_NAME).exists() or seal.exists():
        raise FileExistsError("post-run manifest or seal exists; refusing to overwrite")
    target.mkdir(parents=True, exist_ok=True)
    write_manifest(target / POST_RUN_NAME, post, forbidden_ids, plan.all_aliases)
    seal.write_text(canonical_json_sha256(post), encoding="utf-8")
    return {
        "run_id": plan.run_id,
        "pre_run_status": "approved",
        "status": status,
        "exit_code": post["run"]["exit_code"],
        "missing_outputs": missing,
        "output_check_errors": check_errors,
        "pre_run_sha256": gate.pre_run_sha256,
        "post_run_path": (target / POST_RUN_NAME).relative_to(repo_root).as_posix(),
        "resumed": resumed,
    }


def resume_controlled(
    plan: RunPlan,
    runner: Runner,
    *,
    repo_root: Path,
    source_control: Mapping[str, Any],
    environment: Mapping[str, Any],
    determinism: Mapping[str, Any],
    forbidden_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Explicitly requested resume of an interrupted (not finalised) run; see the module
    docstring. Raises ``ResumeRefusedError`` (nothing written) when any condition fails."""
    forbidden_ids = tuple(forbidden_ids)
    training_labels = training_output_labels(getattr(runner, "training_outputs", "smoke_v1"))
    run_dir = plan.run_dir
    pre_path = run_dir / PRE_RUN_NAME
    if not hasattr(runner, "resume") or not hasattr(runner, "resume_preflight"):
        raise ResumeRefusedError(["runner_does_not_support_resume"])
    if not pre_path.is_file():
        raise ResumeRefusedError(["pre_run_manifest_missing"])
    if (run_dir / POST_RUN_NAME).exists() or (run_dir / SEAL_NAME).exists():
        raise ResumeRefusedError(["run_already_finalised"])
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    check_schema(pre, "pre_run")
    runner_seed = getattr(runner, "seed", None)
    determinism = {**determinism, "seed": runner_seed}
    if hasattr(runner, "apply_determinism"):
        determinism = {**determinism, **runner.apply_determinism(), "seed": runner_seed}
    reasons: list[str] = []
    if getattr(plan, "seed", None) is not None and plan.seed != runner_seed:
        reasons.append("seed_plan_runner_mismatch")
    if pre.get("determinism", {}).get("seed") != runner_seed:
        reasons.append("seed_mismatch")  # resume must use the seed the run started with
    ident = pre.get("identity", {})
    if ident.get("run_id") != plan.run_id:
        reasons.append("run_id_mismatch")
    for name, value in (
        ("stage", plan.stage),
        ("dataset_id", plan.dataset_id),
        ("split_name", plan.split_name),
    ):
        if ident.get(name) != value:
            reasons.append(f"{name}_mismatch")
    if source_control.get("working_tree_clean") is not True:
        reasons.append("working_tree_not_clean")
    if source_control.get("git_commit") != pre.get("source_control", {}).get("git_commit"):
        reasons.append("git_commit_mismatch")
    conf = pre.get("configuration", {})
    if sha256_text_lf(plan.config) != conf.get("config", {}).get("sha256_lf"):
        reasons.append("config_hash_mismatch")
    if sha256_text_lf(plan.split_file) != conf.get("split", {}).get("sha256_lf"):
        reasons.append("split_hash_mismatch")
    inputs = pre.get("inputs", {})
    if alias_list_sha256(plan.all_aliases, CASE_KEY_POLICY) != inputs.get("alias_list_sha256"):
        reasons.append("alias_list_hash_mismatch")
    keys = case_keys(plan.all_aliases)
    current_inputs = {
        keys[a]: {f"{role}_sha256": sha256_file(p) for role, p in sorted(roles.items())}
        for a, roles in plan.input_files.items()
    }
    if dict(sorted(current_inputs.items())) != inputs.get("file_sha256"):
        reasons.append("input_hash_mismatch")
    if environment.get("package_lock_sha256_lf") != pre.get("environment", {}).get(
        "package_lock_sha256_lf"
    ):
        reasons.append("package_lock_hash_mismatch")
    if dict(environment) != pre.get("environment"):
        reasons.append("environment_mismatch")
    if dict(determinism) != pre.get("determinism"):
        reasons.append("determinism_mismatch")
    if reasons:
        raise ResumeRefusedError(reasons)
    gate = GateResult(APPROVED, (), run_dir, canonical_json_sha256(pre), pre)
    reasons = list(runner.resume_preflight(gate, plan))
    if reasons:
        raise ResumeRefusedError(reasons)
    return _execute(
        plan,
        runner,
        gate,
        runner.resume,
        training_labels,
        repo_root=repo_root,
        environment=environment,
        determinism=determinism,
        forbidden_ids=forbidden_ids,
        resumed=True,
    )


def load_post_run(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / POST_RUN_NAME).read_text(encoding="utf-8"))


def verify_run(run_dir: Path, segment: str | None = None) -> dict[str, Any]:
    """Re-check a finished run (or, with ``segment``, an interrupted segment's manifest under
    ``segments/<segment>/``): both manifests valid and linked, every recorded output hash equal
    to the file on disk. Any edit to a manifest or an output makes ``ok`` false."""
    run_dir = Path(run_dir)
    post_dir = run_dir / SEGMENTS_DIR / segment if segment else run_dir
    problems: list[str] = []
    try:
        pre = json.loads((run_dir / PRE_RUN_NAME).read_text(encoding="utf-8"))
        post = load_post_run(post_dir)
        if not is_linked(pre, post):
            problems.append("pre_post_link_broken")
        seal = (post_dir / SEAL_NAME).read_text(encoding="utf-8").strip()
        if seal != canonical_json_sha256(post):
            problems.append("post_run_seal_mismatch")
        for field in ("identity", "access"):
            if post.get(field) != pre.get(field):
                problems.append(f"pre_post_{field}_differs")
        if post.get("alias_list_sha256") != pre.get("inputs", {}).get("alias_list_sha256"):
            problems.append("pre_post_alias_list_differs")
    except (OSError, ValueError) as e:
        return {"ok": False, "problems": [f"manifest_unreadable:{type(e).__name__}"]}
    for kind, entries in post.get("outputs", {}).items():
        for key, recorded in entries.items():
            if kind == AGGREGATE_KIND:
                path = run_output_path(run_dir, key)
            else:
                path = case_path(run_dir, kind, key)
            if not path.is_file():
                problems.append(f"output_missing:{kind}:{key}")
            elif sha256_file(path) != recorded:
                problems.append(f"output_hash_mismatch:{kind}:{key}")
    return {"ok": not problems, "problems": problems}
