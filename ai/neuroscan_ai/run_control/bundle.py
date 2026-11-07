"""Run-bundle reproducibility comparison: two independent runs, each = pre-run + post-run manifest.

``compare_manifests()`` (reproducibility package) compares two manifests of the same kind and
cannot judge a run on its own post-run manifest (that manifest carries no source control or
config). The official reproducibility verdict for two runs is ``compare_run_bundles()``:

1. integrity of each run: schemas, pre/post link, seal, every output hash vs the file
   (``verify_run``); any failure -> ``mismatch``
2. field comparison of the two bundles, excluding audit fields that must differ between
   independent runs (run_id, timestamps, the manifest hashes and seal themselves, folder names)

Field applicability: a train run has no input checkpoint; when both runs are train runs with no
input checkpoint the field is ``not_applicable`` on both sides and counts as equal. For every
other stage the input checkpoint is required. Audit fields are never removed from manifests.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neuroscan_ai.reproducibility.manifest import SCHEMA
from neuroscan_ai.reproducibility.validation import COMPATIBLE, EXACT, MISMATCH, UNAVAILABLE
from neuroscan_ai.run_control.gate import PRE_RUN_NAME
from neuroscan_ai.run_control.lifecycle import COMPLETED, POST_RUN_NAME, verify_run

NOT_APPLICABLE = "not_applicable"
RESULT, ENVIRONMENT = "result_defining", "environment"
EXCLUDED_AUDIT_FIELDS = (
    "pre.identity.run_id",
    "pre.identity.created_at_utc",
    "post.identity.run_id",
    "post.identity.created_at_utc",
    "post.run.started_utc",
    "post.run.ended_utc",
    "post.pre_run_manifest_sha256",
    "post_run_manifest.sha256 (seal value)",
    "run folder name",
)
# (side, dotted path, class); "checkpoint" is handled by stage applicability
PRE_FIELDS = (
    ("schema", RESULT),
    ("identity.stage", RESULT),
    ("identity.dataset_id", RESULT),
    ("identity.split_name", RESULT),
    ("identity.case_count", RESULT),
    ("identity.purpose", ENVIRONMENT),
    ("inputs.alias_list_sha256", RESULT),
    ("inputs.case_count", RESULT),
    ("inputs.case_key_policy", RESULT),
    ("inputs.file_sha256", RESULT),
    ("inputs.input_manifest_sha256", RESULT),
    ("configuration.config.sha256_lf", RESULT),
    ("configuration.split.sha256_lf", RESULT),
    ("configuration.referenced_configs", RESULT),
    ("configuration.policies", RESULT),
    ("model.architecture_id", RESULT),
    ("access", RESULT),
    ("source_control.git_commit", ENVIRONMENT),
    ("environment", ENVIRONMENT),
    ("determinism", ENVIRONMENT),
)
POST_FIELDS = (
    ("schema", RESULT),
    ("run.status", RESULT),
    ("run.exit_code", RESULT),
    ("run.cases_succeeded", RESULT),
    ("run.cases_failed", RESULT),
    ("access", RESULT),
    ("reproducibility.accessed_splits", RESULT),
    ("reproducibility.runner_kind", RESULT),
    ("outputs", RESULT),  # every per-case and run-level output hash incl. training outputs
    ("environment", ENVIRONMENT),
    ("determinism", ENVIRONMENT),
)
NULLABLE = {"inputs.input_manifest_sha256"}  # optional by design; equal when null on both sides
_MISSING = object()


def _get(m: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = m
    for k in dotted.split("."):
        if not isinstance(cur, Mapping) or k not in cur:
            return _MISSING
        cur = cur[k]
    return cur


def input_checkpoint_state(pre: Mapping[str, Any]) -> str | None:
    """Input checkpoint hash, ``not_applicable`` (train run without one) or None (missing)."""
    ck = _get(pre, "model.checkpoint")
    if isinstance(ck, Mapping) and ck.get("sha256"):
        return ck["sha256"]
    if ck is None and _get(pre, "identity.stage") == "train":
        return NOT_APPLICABLE
    return None


def compare_bundle_contents(
    pre_a: Mapping[str, Any],
    post_a: Mapping[str, Any],
    pre_b: Mapping[str, Any],
    post_b: Mapping[str, Any],
) -> dict[str, Any]:
    """Field comparison of two run bundles (integrity is checked by ``compare_run_bundles``)."""
    diffs: list[dict[str, str]] = []
    missing: list[str] = []
    not_applicable: list[str] = []

    def check(side: str, a: Mapping, b: Mapping, path: str, cls: str) -> None:
        va, vb = _get(a, path), _get(b, path)
        name = f"{side}.{path}"
        if (
            va is _MISSING
            or vb is _MISSING
            or ((va is None or vb is None) and path not in NULLABLE)
        ):
            if not (va is None and vb is None and path in NULLABLE):
                missing.append(name)
            return
        if va != vb:
            diffs.append({"field": name, "class": cls})

    for path, cls in PRE_FIELDS:
        check("pre", pre_a, pre_b, path, cls)
    # input data hashes must actually have been recorded (an empty map proves nothing)
    for side_name, pre in (("a", pre_a), ("b", pre_b)):
        files = _get(pre, "inputs.file_sha256")
        if (_get(pre, "inputs.case_count") or 0) > 0 and isinstance(files, Mapping) and not files:
            missing.append(f"pre.inputs.file_sha256 (not recorded in run {side_name})")
    for path, cls in POST_FIELDS:
        check("post", post_a, post_b, path, cls)
    ca, cb = input_checkpoint_state(pre_a), input_checkpoint_state(pre_b)
    if ca is None or cb is None:
        missing.append("pre.model.checkpoint")
    elif ca == NOT_APPLICABLE and cb == NOT_APPLICABLE:
        not_applicable.append("pre.model.checkpoint")
    elif ca != cb:  # includes not_applicable on one side only
        diffs.append({"field": "pre.model.checkpoint", "class": RESULT})
    dirty = [
        s
        for s, p in (("a", pre_a), ("b", pre_b))
        if _get(p, "source_control.working_tree_clean") is not True
    ]
    not_completed = [
        s for s, p in (("a", post_a), ("b", post_b)) if _get(p, "run.status") != COMPLETED
    ]

    if any(d["class"] == RESULT for d in diffs) or not_completed:
        level = MISMATCH
    elif diffs or dirty:
        level = COMPATIBLE
    elif missing:
        level = UNAVAILABLE
    else:
        level = EXACT
    return {
        "level": level,
        "differences": diffs,
        "missing": missing,
        "not_applicable": not_applicable,
        "working_tree_not_clean": dirty,
        "runs_not_completed": not_completed,
        "excluded_audit_fields": list(EXCLUDED_AUDIT_FIELDS),
    }


def _load(run_dir: Path, name: str) -> dict[str, Any]:
    return json.loads((Path(run_dir) / name).read_text(encoding="utf-8"))


def compare_run_bundles(run_dir_a: Path, run_dir_b: Path) -> dict[str, Any]:
    """Official reproducibility verdict for two independent runs (read-only)."""
    integrity = {"a": verify_run(run_dir_a), "b": verify_run(run_dir_b)}
    try:
        pre_a, post_a = _load(run_dir_a, PRE_RUN_NAME), _load(run_dir_a, POST_RUN_NAME)
        pre_b, post_b = _load(run_dir_b, PRE_RUN_NAME), _load(run_dir_b, POST_RUN_NAME)
    except (OSError, ValueError) as e:
        return {
            "level": MISMATCH,
            "integrity": integrity,
            "reason": f"manifest_unreadable:{type(e).__name__}",
        }
    schemas_ok = all(m.get("schema") == SCHEMA for m in (pre_a, post_a, pre_b, post_b))
    res = compare_bundle_contents(pre_a, post_a, pre_b, post_b)
    res["integrity"] = integrity
    res["schemas_ok"] = schemas_ok
    if not (integrity["a"]["ok"] and integrity["b"]["ok"] and schemas_ok):
        res["level"] = MISMATCH
        res["reason"] = "run_integrity_failed"
    res["comparison"] = "run_bundle"
    return res
