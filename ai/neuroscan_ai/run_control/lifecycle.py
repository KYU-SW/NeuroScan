"""Controlled run lifecycle (v2): plan -> pre-run gate -> runner -> output checks -> post-run.

1 plan validated (RunPlan) 2 pre-run manifest built 3 its hash frozen 4 runner called only with an
approved gate 5 required outputs present 6 output hashes recorded 7 post-run manifest written
8 pre/post link verified 9 final status. Failed runs also get a post-run manifest and their run
folder is kept; a run_id is never reused.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from neuroscan_ai.reproducibility import (
    ManifestError,
    PathRoots,
    build_post_run,
    case_keys,
    is_linked,
    write_manifest,
)
from neuroscan_ai.reproducibility.hashing import canonical_json_sha256, sha256_file
from neuroscan_ai.reproducibility.manifest import guard_public
from neuroscan_ai.run_control.gate import PRE_RUN_NAME, GateResult, pre_run_gate
from neuroscan_ai.run_control.outputs import (
    AGGREGATE_KIND,
    AGGREGATE_LABEL,
    IMAGE_KINDS,
    METRIC_KIND,
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


class Runner(Protocol):
    threshold: float

    def run(self, gate: GateResult, plan: RunPlan) -> None: ...


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

    started = _now()
    failed_stage = error_type = error_message = None
    try:
        runner.run(gate, plan)
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

    if failed_stage:
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
    }
    access = dict(gate.pre_run["access"]) if gate.pre_run else {}
    post = build_post_run(
        gate.pre_run,
        aliases=plan.all_aliases,
        started_utc=started,
        ended_utc=ended,
        status=status,
        exit_code=0 if status == COMPLETED else 1,
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
    write_manifest(gate.run_dir / POST_RUN_NAME, post, forbidden_ids, plan.all_aliases)
    seal = gate.run_dir / SEAL_NAME
    if seal.exists():
        raise FileExistsError("seal exists; refusing to overwrite")
    seal.write_text(canonical_json_sha256(post), encoding="utf-8")
    return {
        "run_id": plan.run_id,
        "pre_run_status": "approved",
        "status": status,
        "exit_code": post["run"]["exit_code"],
        "missing_outputs": missing,
        "output_check_errors": check_errors,
        "pre_run_sha256": gate.pre_run_sha256,
        "post_run_path": (gate.run_dir / POST_RUN_NAME).relative_to(repo_root).as_posix(),
    }


def load_post_run(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / POST_RUN_NAME).read_text(encoding="utf-8"))


def verify_run(run_dir: Path) -> dict[str, Any]:
    """Re-check a finished run: both manifests valid and linked, every recorded output hash equal
    to the file on disk. Any edit to a manifest or an output makes ``ok`` false."""
    run_dir = Path(run_dir)
    problems: list[str] = []
    try:
        pre = json.loads((run_dir / PRE_RUN_NAME).read_text(encoding="utf-8"))
        post = load_post_run(run_dir)
        if not is_linked(pre, post):
            problems.append("pre_post_link_broken")
        seal = (run_dir / SEAL_NAME).read_text(encoding="utf-8").strip()
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
