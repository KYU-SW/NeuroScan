"""Compare two run manifests (or a saved manifest with the current state).

Levels, from best to worst:
- ``exact_match``: every compared field is present and equal, both working trees clean.
- ``compatible_but_not_exact``: result-defining fields equal, but code / environment /
  determinism differ (git commit, PyTorch, CUDA, cuDNN, driver, GPU, package lock, determinism,
  dirty tree).
- ``unavailable``: no difference found, but a required field is missing or null on either side.
- ``mismatch``: a result-defining field differs (checkpoint, config, split, inputs, output hashes).

Any difference in git commit, checkpoint, config, split, inputs, PyTorch, CUDA, cuDNN, driver,
GPU, package lock, determinism or output hash prevents ``exact_match``. Text files (config,
split, package lock) are compared by their LF-canonical SHA-256. Both manifests must carry the
schema ``neuroscan-run-manifest-v1``; anything else is refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neuroscan_ai.reproducibility.manifest import ManifestError, check_schema

EXACT = "exact_match"
COMPATIBLE = "compatible_but_not_exact"
MISMATCH = "mismatch"
UNAVAILABLE = "unavailable"

RESULT_FIELDS = (
    ("configuration", "config", "sha256_lf"),
    ("configuration", "split", "sha256_lf"),
    ("model", "checkpoint", "sha256"),
    ("inputs", "alias_list_sha256"),
    ("inputs", "file_sha256"),
)
ENVIRONMENT_FIELDS = (
    ("source_control", "git_commit"),
    ("environment", "packages", "torch"),
    ("environment", "cuda_runtime"),
    ("environment", "cudnn"),
    ("environment", "gpu_driver_version"),
    ("environment", "gpu_name"),
    ("environment", "package_lock_sha256_lf"),
    ("determinism",),
)
OUTPUT_FIELDS = (
    ("outputs", "probability_maps"),
    ("outputs", "ungated_predictions"),
    ("outputs", "gated_predictions"),
)
OPTIONAL_NONE_OK = {
    ("environment", "cuda_runtime"),
    ("environment", "cudnn"),
    ("environment", "gpu_driver_version"),
    ("environment", "gpu_name"),
}


def _get(m: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = m
    for k in path:
        if not isinstance(cur, Mapping) or k not in cur:
            return _MISSING
        cur = cur[k]
    return cur


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def compare_manifests(
    a: Mapping[str, Any], b: Mapping[str, Any], *, cpu_only_ok: bool = False
) -> dict[str, Any]:
    """Field-by-field comparison.

    ``cpu_only_ok`` treats absent CUDA / driver information on both sides as equal.
    """
    check_schema(a)
    check_schema(b)
    if {a.get("kind"), b.get("kind")} - {"pre_run", "post_run"}:
        raise ManifestError("manifest kind must be 'pre_run' or 'post_run'")
    diffs, missing = [], []

    def check(path: tuple[str, ...], kind: str) -> None:
        va, vb = _get(a, path), _get(b, path)
        name = ".".join(path)
        none_ok = cpu_only_ok and path in OPTIONAL_NONE_OK
        if va is _MISSING or vb is _MISSING:
            missing.append(name)
            return
        if va is None or vb is None:  # unknown on one or both sides -> not comparable
            if not (none_ok and va is None and vb is None):
                missing.append(name)
            return
        if va != vb:
            diffs.append({"field": name, "kind": kind})

    for p in RESULT_FIELDS:
        check(p, "result_defining")
    for p in ENVIRONMENT_FIELDS:
        check(p, "environment")
    both_have_outputs = a.get("kind") == "post_run" and b.get("kind") == "post_run"
    if both_have_outputs:
        for p in OUTPUT_FIELDS:
            check(p, "result_defining")
    dirty = [
        s
        for s, m in (("a", a), ("b", b))
        if _get(m, ("source_control", "working_tree_clean")) is not True
    ]
    if any(d["kind"] == "result_defining" for d in diffs):
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
        "working_tree_not_clean": dirty,
        "outputs_compared": both_have_outputs,
    }
