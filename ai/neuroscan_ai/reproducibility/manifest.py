"""Pre-run / post-run manifests: build, sanitise, guard, link and write (never overwrite).

Public manifests contain token paths, hashes, versions, policy IDs and position-based case keys
only. Case aliases are never written in plain text: the sorted alias list is recorded as one
SHA-256 (``alias_list_sha256``) and per-case hashes are keyed ``case_0001``, ``case_0002``, ...
by position in that sorted list (``case_key_policy = "sorted_alias_position_v1"``). Host paths
become ``<WORKDIR>``, ``<CHECKPOINT>`` or ``<DATASET>`` plus a relative POSIX path (an alias inside
a path is replaced by its case key); original subject IDs, account names, credentials and secret
environment values are refused. The alias -> case key map may only be written to the git-ignored
``results/reproducibility/_internal/``.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neuroscan_ai.brain_extraction.audit import find_host_paths, find_identifiers
from neuroscan_ai.reproducibility.hashing import (
    alias_list_sha256,
    canonical_json_sha256,
    file_hashes,
    sha256_file,
)

SCHEMA = "neuroscan-run-manifest-v1"
CASE_KEY_POLICY = "sorted_alias_position_v1"
ACCESS_SPLITS = ("train", "val", "test")
PER_CASE_OUTPUTS = (
    "probability_maps",
    "ungated_predictions",
    "gated_predictions",
    "brain_masks",
    "patient_metrics",
)
AGGREGATE_OUTPUTS = "aggregate_results"
INTERNAL_MAP_DIR = Path("results/reproducibility/_internal")
_LABEL = re.compile(r"^[a-z0-9_]{1,64}$")
_CREDENTIAL = re.compile(
    r"(ghp_[A-Za-z0-9]{20,}|github_pat_\w{20,}|AKIA[0-9A-Z]{12,}|sk-[A-Za-z0-9]{20,}|xox[abp]-\w{10,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY|(?i:bearer\s+[A-Za-z0-9._-]{16,})"
    r"|(?i:(password|passwd|secret|api[_-]?key|token)\s*[=:]\s*\S+))"
)
_SECRET_ENV = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|CREDENTIAL|PRIVATE)", re.I)


class ManifestError(ValueError):
    pass


def check_schema(manifest: Mapping[str, Any], kind: str | None = None) -> None:
    """Refuse a missing or different schema (and, if given, a different manifest kind)."""
    if not isinstance(manifest, Mapping) or manifest.get("schema") != SCHEMA:
        raise ManifestError(f"manifest schema must be '{SCHEMA}'")
    if kind is not None and manifest.get("kind") != kind:
        raise ManifestError(f"expected a '{kind}' manifest")


@dataclass(frozen=True)
class PathRoots:
    """Host roots replaced by tokens (most specific first)."""

    workdir: Path
    checkpoint: Path | None = None
    dataset: Path | None = None

    def ordered(self) -> list[tuple[str, Path]]:
        roots = [
            ("<CHECKPOINT>", self.checkpoint),
            ("<DATASET>", self.dataset),
            ("<WORKDIR>", self.workdir),
        ]
        return [(t, Path(p).resolve()) for t, p in roots if p is not None]


def sanitize_path(path: str | Path, roots: PathRoots) -> str:
    p = Path(path).resolve()
    for token, root in roots.ordered():
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        return f"{token}/{rel.as_posix()}" if rel.parts else token
    raise ManifestError("path outside the declared roots; refusing to record a host path")


def case_keys(aliases: Iterable[str]) -> dict[str, str]:
    """alias -> ``case_NNNN`` by position in the sorted alias list.

    Policy ``sorted_alias_position_v1``.
    """
    ordered = sorted(set(aliases))
    if len(ordered) > 9999:
        raise ManifestError("more than 9999 cases is not supported by the case key policy")
    return {a: f"case_{i:04d}" for i, a in enumerate(ordered, 1)}


def _replace_aliases(text: str, keys: Mapping[str, str]) -> str:
    for alias in sorted(keys, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", keys[alias], text)
    return text


def find_aliases(text: str, aliases: Iterable[str]) -> list[str]:
    return [
        a
        for a in aliases
        if a and re.search(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])", text)
    ]


def find_credentials(text: str) -> list[str]:
    hits = [m.group(0)[:12] + "..." for m in _CREDENTIAL.finditer(text)]
    for k, v in os.environ.items():
        if _SECRET_ENV.search(k) and v and len(v) >= 6 and v in text:
            hits.append(f"env:{k}")
    return hits


def guard_public(obj: Any, forbidden_ids: Iterable[str] = (), aliases: Iterable[str] = ()) -> None:
    text = json.dumps(obj, default=str)
    if find_identifiers(text, tuple(forbidden_ids)):
        raise ManifestError("manifest would contain an original subject identifier")
    if find_aliases(text, aliases):
        raise ManifestError("manifest would contain a case alias in plain text")
    if find_host_paths(text):
        raise ManifestError("manifest would contain a host path or account name")
    if find_credentials(text):
        raise ManifestError("manifest would contain a credential or secret value")


@dataclass
class RunSpec:
    """What a run is about to do (all paths are host paths; they are sanitised on recording)."""

    run_id: str
    stage: str
    purpose: str
    dataset_id: str
    split_name: str
    aliases: list[str]
    config: Path
    split: Path
    checkpoint: Path | None = None
    referenced_configs: list[Path] = field(default_factory=list)
    policies: dict[str, str | None] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    checkpoint_metadata: dict[str, Any] = field(default_factory=dict)
    input_files: dict[str, dict[str, Path]] = field(default_factory=dict)  # alias -> role -> path
    input_manifest: Path | None = None
    access: dict[str, Any] = field(default_factory=dict)


POLICY_KEYS = ("preprocessing", "inference", "threshold", "brain_mask", "combination")


def _access(spec: RunSpec) -> dict[str, Any]:
    a = {f"{s}_cases": int(spec.access.get(f"{s}_cases", 0)) for s in ACCESS_SPLITS}
    a["final_test_accessed"] = bool(spec.access.get("final_test_accessed", False))
    allowed = set(spec.access.get("allowed_splits", [spec.split_name]))
    a["allowed_splits"] = sorted(allowed)
    a["disallowed_split_access"] = any(
        a[f"{s}_cases"] > 0 for s in ACCESS_SPLITS if s not in allowed
    ) or (a["final_test_accessed"] and "final_test" not in allowed)
    return a


def _check_aliases(aliases: Iterable[str], forbidden_ids: Iterable[str]) -> None:
    for alias in aliases:
        if find_identifiers(str(alias), tuple(forbidden_ids)):
            raise ManifestError("case alias looks like an original subject identifier")


def build_pre_run(
    spec: RunSpec,
    *,
    roots: PathRoots,
    source_control: Mapping[str, Any],
    environment: Mapping[str, Any],
    determinism: Mapping[str, Any],
    forbidden_ids: Iterable[str] = (),
    now: datetime.datetime | None = None,
) -> dict[str, Any]:
    _check_aliases([*spec.aliases, *spec.input_files], forbidden_ids)
    keys = case_keys(spec.aliases)
    unknown = set(spec.input_files) - set(keys)
    if unknown:
        raise ManifestError("input files given for cases outside the alias list")

    def entry(p: Path) -> dict[str, str]:
        return {"path": _replace_aliases(sanitize_path(p, roots), keys), **file_hashes(p)}

    inputs = {
        keys[alias]: {f"{role}_sha256": sha256_file(p) for role, p in sorted(roles.items())}
        for alias, roles in spec.input_files.items()
    }
    manifest = {
        "schema": SCHEMA,
        "kind": "pre_run",
        "identity": {
            "run_id": spec.run_id,
            "stage": spec.stage,
            "created_at_utc": (now or datetime.datetime.now(datetime.UTC)).isoformat(),
            "purpose": spec.purpose,
            "dataset_id": spec.dataset_id,
            "split_name": spec.split_name,
            "case_count": len(keys),
        },
        "source_control": dict(source_control),
        "configuration": {
            "config": entry(spec.config),
            "referenced_configs": [entry(p) for p in spec.referenced_configs],
            "split": entry(spec.split),
            "policies": {f"{k}_policy_id": spec.policies.get(k) for k in POLICY_KEYS},
        },
        "model": {
            "architecture_id": spec.model.get("architecture_id"),
            "checkpoint": entry(spec.checkpoint) if spec.checkpoint else None,
            "checkpoint_metadata": dict(spec.checkpoint_metadata),
            "input_channel_policy": spec.model.get("input_channel_policy"),
            "output_definition": spec.model.get("output_definition"),
            "activation": spec.model.get("activation"),
            "fixed_threshold": spec.model.get("fixed_threshold"),
        },
        "environment": dict(environment),
        "determinism": dict(determinism),
        "inputs": {
            "case_count": len(keys),
            "case_key_policy": CASE_KEY_POLICY,
            "alias_list_sha256": alias_list_sha256(spec.aliases, CASE_KEY_POLICY),
            "input_manifest_sha256": sha256_file(spec.input_manifest)
            if spec.input_manifest
            else None,
            "file_sha256": dict(sorted(inputs.items())),
        },
        "access": _access(spec),
    }
    if manifest["access"]["disallowed_split_access"]:
        raise ManifestError("run plan accesses a split that is not allowed")
    guard_public(manifest, forbidden_ids, keys)
    return manifest


def build_post_run(
    pre: Mapping[str, Any],
    *,
    aliases: Iterable[str],
    started_utc: str,
    ended_utc: str,
    status: str,
    exit_code: int,
    cases_succeeded: int,
    cases_failed: int,
    outputs: Mapping[str, Mapping[str, Path]],  # per-case kind -> alias -> path; aggregate -> label
    environment: Mapping[str, Any],
    determinism: Mapping[str, Any],
    access: Mapping[str, Any],
    reproducibility: Mapping[str, Any] | None = None,
    forbidden_ids: Iterable[str] = (),
) -> dict[str, Any]:
    check_schema(pre, "pre_run")
    aliases = list(aliases)
    _check_aliases(aliases, forbidden_ids)
    if alias_list_sha256(aliases, CASE_KEY_POLICY) != pre["inputs"]["alias_list_sha256"]:
        raise ManifestError("alias list differs from the pre-run manifest")
    keys = case_keys(aliases)
    unknown = set(outputs) - {*PER_CASE_OUTPUTS, AGGREGATE_OUTPUTS}
    if unknown:
        raise ManifestError(f"unknown output kinds: {sorted(unknown)}")
    out: dict[str, dict[str, str]] = {}
    for kind in PER_CASE_OUTPUTS:
        items = outputs.get(kind, {})
        if set(items) - set(keys):
            raise ManifestError(f"{kind}: outputs for cases outside the alias list")
        out[kind] = dict(sorted((keys[a], sha256_file(p)) for a, p in items.items()))
    agg = outputs.get(AGGREGATE_OUTPUTS, {})
    for label in agg:
        if not _LABEL.match(label) or label in keys:
            raise ManifestError("aggregate output labels must be generic [a-z0-9_] names")
    out[AGGREGATE_OUTPUTS] = {label: sha256_file(p) for label, p in sorted(agg.items())}
    post = {
        "schema": SCHEMA,
        "kind": "post_run",
        "pre_run_manifest_sha256": canonical_json_sha256(pre),
        "identity": dict(pre["identity"]),
        "run": {
            "started_utc": started_utc,
            "ended_utc": ended_utc,
            "status": status,
            "exit_code": exit_code,
            "cases_succeeded": cases_succeeded,
            "cases_failed": cases_failed,
        },
        "case_key_policy": CASE_KEY_POLICY,
        "alias_list_sha256": pre["inputs"]["alias_list_sha256"],
        "outputs": out,
        "environment": dict(environment),
        "determinism": dict(determinism),
        "access": dict(access),
        "reproducibility": dict(reproducibility or {}),
    }
    guard_public(post, forbidden_ids, keys)
    return post


def is_linked(pre: Mapping[str, Any], post: Mapping[str, Any]) -> bool:
    check_schema(pre, "pre_run")
    check_schema(post, "post_run")
    return post.get("pre_run_manifest_sha256") == canonical_json_sha256(pre) and post.get(
        "identity", {}
    ).get("run_id") == pre.get("identity", {}).get("run_id")


def write_manifest(
    path: Path,
    manifest: Mapping[str, Any],
    forbidden_ids: Iterable[str] = (),
    aliases: Iterable[str] = (),
) -> Path:
    check_schema(manifest)
    guard_public(manifest, forbidden_ids, aliases)
    path = Path(path)
    if path.exists():
        raise ManifestError("manifest exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def write_case_key_map(path: Path, aliases: Iterable[str], repo_root: Path) -> Path:
    """Internal alias -> case key map.

    Only inside the git-ignored ``results/reproducibility/_internal/``.
    """
    path = Path(path).resolve()
    internal = (Path(repo_root) / INTERNAL_MAP_DIR).resolve()
    if internal not in path.parents:
        raise ManifestError(
            "the case key map may only be written to results/reproducibility/_internal/"
        )
    rel = path.relative_to(Path(repo_root).resolve()).as_posix()
    ignored = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", "-q", rel], capture_output=True
    ).returncode
    if ignored != 0:
        raise ManifestError("the case key map path is not git-ignored")
    if path.exists():
        raise ManifestError("case key map exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"case_key_policy": CASE_KEY_POLICY, "map": case_keys(aliases)}
    path.write_text(json.dumps(body, indent=1), encoding="utf-8")
    return path
