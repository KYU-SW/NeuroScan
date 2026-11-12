"""Disk-backed preprocessed cache ``disk_backed_preprocessed_cache_v1`` (v2 2D baseline).

Why: holding every prepared Train / Validation volume in RAM (the earlier runner) grows with the
case count; the cache keeps the prepared model-grid volumes on disk and the dataset reads only the
slices a batch needs through ``numpy.load(path, mmap_mode="r")``.

Layout (``results/v2_cache/<version>/``, git-ignored; built once by
``scripts/prepare_v2_2d_cache.py``, never overwritten; another policy or input -> another version):

  case_NNNN/ dwi.npy adc.npy gt.npy brain_mask.npy valid_region_mask.npy
             original_gt.npy original_brain_mask.npy cache_manifest.json cache_manifest.sha256
  cache_index.json  cache_index.sha256  cache_build_summary.json  _internal/

Arrays are the ``preprocess_v2_isles_1mm_inplane_v1`` result on the 1 mm model grid (X / Y 1.0 mm,
Z kept; DWI / ADC linear then per-case z-score over the model-grid brain mask; GT / brain mask
nearest; GT decode ``binary_mask_decode_v1_atol_1e_6``), full volumes BEFORE padding (slice
padding is applied by the dataset). ``valid_region_mask`` marks the real image region (all 1 on
the unpadded grid). ``original_gt`` / ``original_brain_mask`` (original grid) are stored too: the
validation metrics and the gated output are computed on the original grid.
Axis order (``cache_slice_first_zxy_to_logical_xyz_v1``, every array the same): storage
``[z, x, y]`` (one slice is one contiguous block), logical ``[x, y, z]`` (slice axis 2); readers
transpose storage -> logical without copying. Affines apply to logical voxel indices (x, y, z).
dtypes: dwi / adc float32, masks uint8. Source files are only read.

Affines: ``affine_exact`` is the float64 affine used for every computation and for the restore to
the original grid (JSON floats round-trip float64 exactly; never rounded or altered);
``affine_display`` / ``spacing_mm_display`` are rounded copies for reading only, never used in a
computation (the gate checks they equal the rounded exact values).

Commit binding: a cache records the git commit it was built at (clean tree required) and a
formal run may use it only at that same commit (``check_cache_commit``, before the run gate:
nothing is created and the run ID stays unused on a refusal). Re-using a cache across commits
needs a separately approved preprocessing-code-hash policy.

Manifests carry case keys, hashes, policy IDs, grids and counts only: no alias, subject ID, file
name, host path or user name. ``cache_manifest.sha256`` / ``cache_index.sha256`` are the canonical
JSON SHA-256 seals. ``verify_cache`` (the cache integrity gate) must pass for every case before a
run reads any array; one failing case refuses the run.

RAM policy: ``CachedCase`` keeps small metadata only (case key, slice indices, grids, counts).
``CacheStore`` keeps at most 2 cases' memory maps open (fixed; least-recently-used closed first;
never changed automatically). A batch copies only its slices into tensors.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from neuroscan_ai.reproducibility import CASE_KEY_POLICY, case_keys
from neuroscan_ai.reproducibility.hashing import (
    alias_list_sha256,
    canonical_json_sha256,
    sha256_file,
)
from neuroscan_ai.v2.baseline_preprocess import (
    BRAIN_MASK_POLICY,
    PREPROCESS_POLICY,
    PreparedCase,
)
from neuroscan_ai.v2.baseline_progress import check_public_text, utc_now
from neuroscan_ai.v2.dataset import CaseFiles
from neuroscan_ai.v2.geometry import BINARY_DECODE_POLICY, Grid
from neuroscan_ai.v2.safety_metrics import GT_CASE_STATUS, GT_LOST, gt_case_status

CACHE_POLICY_ID = "disk_backed_preprocessed_cache_v1"
CACHE_POLICY: dict[str, Any] = {
    "policy_id": CACHE_POLICY_ID,
    "mmap_mode": "r",
    "max_active_cases": 2,
    "overwrite": False,
}
TARGET_GRID_POLICY = "inplane_1mm_z_kept_integer_snap_otherwise_ceil_v1"
CASE_SCHEMA = "neuroscan-v2-baseline-cache-case-v1"
INDEX_SCHEMA = "neuroscan-v2-baseline-cache-index-v1"
SUMMARY_SCHEMA = "neuroscan-v2-baseline-cache-summary-v1"
STORAGE_AXIS_ORDER = ["z", "x", "y"]
LOGICAL_AXIS_ORDER = ["x", "y", "z"]
SLICE_AXIS = 2
TRANSPOSE_POLICY = "cache_slice_first_zxy_to_logical_xyz_v1"
AXIS_FIELDS = {
    "storage_axis_order": STORAGE_AXIS_ORDER,
    "logical_axis_order": LOGICAL_AXIS_ORDER,
    "slice_axis": SLICE_AXIS,
    "transpose_policy_id": TRANSPOSE_POLICY,
}
DISPLAY_DECIMALS = 8
INDEX_NAME, INDEX_SEAL = "cache_index.json", "cache_index.sha256"
MANIFEST_NAME, MANIFEST_SEAL = "cache_manifest.json", "cache_manifest.sha256"
SUMMARY_NAME = "cache_build_summary.json"
# name -> (dtype, grid, kind)
ARRAYS: dict[str, tuple[str, str, str]] = {
    "dwi": ("float32", "model", "image"),
    "adc": ("float32", "model", "image"),
    "gt": ("uint8", "model", "binary"),
    "brain_mask": ("uint8", "model", "binary"),
    "valid_region_mask": ("uint8", "model", "all_one"),
    "original_gt": ("uint8", "original", "binary"),
    "original_brain_mask": ("uint8", "original", "binary"),
}
SOURCE_ROLES = {"dwi": "dwi_sha256", "adc": "adc_sha256", "gt": "gt_sha256"}
SPLITS = ("train", "val")


class CacheError(RuntimeError):
    """Cache missing, refused or failing the integrity gate (no run starts)."""

    stage = "cache_integrity"

    def __init__(self, reasons: Iterable[str]):
        self.reasons = tuple(reasons)
        shown = ", ".join(self.reasons[:20])
        more = f" (+{len(self.reasons) - 20} more)" if len(self.reasons) > 20 else ""
        super().__init__(f"cache refused: {shown}{more}")


class CacheCommitError(CacheError):
    """Cache commit binding failed (raised before the run gate: no folder, run ID unused)."""

    stage = "cache_commit"


# ---------------------------------------------------------------- helpers
def to_storage(a: np.ndarray, dtype: str) -> np.ndarray:
    """Logical (x, y, z) -> storage (z, x, y), contiguous (``TRANSPOSE_POLICY``)."""
    return np.ascontiguousarray(np.moveaxis(np.asarray(a), SLICE_AXIS, 0)).astype(dtype, copy=False)


def to_logical(a: np.ndarray) -> np.ndarray:
    """Storage (z, x, y) -> logical (x, y, z) view (no copy)."""
    return np.moveaxis(a, 0, SLICE_AXIS)


def _display(affine: np.ndarray) -> list[list[float]]:
    return [[round(float(v), DISPLAY_DECIMALS) for v in row] for row in affine]


def _grid_record(g: Grid) -> dict[str, Any]:
    """Exact float64 affine (computation / restore) plus rounded display copies."""
    return {
        "shape": list(g.shape),
        "affine_exact": [[float(v) for v in row] for row in np.asarray(g.affine, np.float64)],
        "affine_display": _display(g.affine),
        "spacing_mm_display": [round(v, DISPLAY_DECIMALS) for v in g.spacing],
        "axcodes": list(g.axcodes),
    }


def _grid(rec: Mapping[str, Any]) -> Grid:
    """Grid from ``affine_exact`` only (display values are never used)."""
    return Grid(tuple(int(s) for s in rec["shape"]), np.asarray(rec["affine_exact"], np.float64))


def _commit_ok(commit: Any) -> bool:
    return (
        isinstance(commit, str)
        and len(commit) == 40
        and all(c in "0123456789abcdef" for c in commit)
    )


def split_canonical_sha256(split_file: Path) -> str:
    return canonical_json_sha256(json.loads(Path(split_file).read_text(encoding="utf-8")))


def environment_summary() -> dict[str, Any]:
    def ver(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "packages": {p: ver(p) for p in ("numpy", "scipy", "nibabel", "torch")},
    }


def _write_json(path: Path, obj: Any) -> None:
    if path.exists():
        raise CacheError(["cache_file_exists_refusing_overwrite"])
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


# ---------------------------------------------------------------- build
def build_cache(
    out_dir: Path,
    *,
    config: Mapping[str, Any],
    split_file: Path,
    files: Mapping[str, Mapping[str, CaseFiles]],
    git_commit: str | None,
    package_lock_sha256_lf: str | None,
    forbidden_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a new cache version at ``out_dir`` (must not exist). ``files``: split -> alias ->
    files (``train`` / ``val``). Cases are prepared one at a time (one case in RAM); a refused
    case (e.g. positive GT lost) stops the build and nothing is published. The cache is written
    to ``.<name>.staging`` and renamed into place only when complete."""
    from neuroscan_ai.v2.baseline_dataset import load_case  # noqa: PLC0415 (no import cycle)

    out_dir = Path(out_dir)
    staging = out_dir.with_name(f".{out_dir.name}.staging")
    if out_dir.exists() or staging.exists():
        raise CacheError(["cache_version_exists_refusing_overwrite"])
    if not _commit_ok(git_commit):
        raise CacheCommitError(["cache_git_commit_missing"])
    if set(files) != set(SPLITS):
        raise CacheError(["cache_build_needs_train_and_val"])
    aliases = [a for s in SPLITS for a in files[s]]
    if len(set(aliases)) != len(aliases):
        raise CacheError(["duplicate_alias_across_splits"])
    keys = case_keys(aliases)
    needles = (*aliases, *forbidden_ids)
    common = {
        "config_canonical_sha256": canonical_json_sha256(dict(config)),
        "split_canonical_sha256": split_canonical_sha256(split_file),
        "alias_list_sha256": alias_list_sha256(aliases, CASE_KEY_POLICY),
    }
    built = {
        "git_commit": git_commit,
        "package_lock_sha256_lf": package_lock_sha256_lf,
        "environment": environment_summary(),
    }
    staging.mkdir(parents=True)
    entries: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for split in SPLITS:
        for alias in sorted(files[split], key=lambda a: keys[a]):
            key, f = keys[alias], files[split][alias]
            case = load_case(key, f)  # read-only; raises on a refused case
            p = case.prepared
            arrays = {
                "dwi": p.image[0],
                "adc": p.image[1],
                "gt": p.label,
                "brain_mask": p.brain,
                "valid_region_mask": np.ones(p.model_grid.shape, np.uint8),
                "original_gt": case.original_gt,
                "original_brain_mask": p.original_brain,
            }
            cdir = staging / key
            cdir.mkdir()
            arec = {}
            for name, (dtype, _, _) in ARRAYS.items():
                path = cdir / f"{name}.npy"
                stored = to_storage(arrays[name], dtype)
                np.save(path, stored, allow_pickle=False)
                arec[name] = {
                    "file": f"{name}.npy",
                    "dtype": dtype,
                    "storage_axis_order": STORAGE_AXIS_ORDER,
                    "stored_shape": list(stored.shape),
                    "sha256": sha256_file(path),
                }
                total_bytes += path.stat().st_size
            manifest = {
                "schema": CASE_SCHEMA,
                "cache_policy_id": CACHE_POLICY_ID,
                "case_key": key,
                "split": split,
                "source_sha256": {
                    r: sha256_file(getattr(f, attr))
                    for r, attr in (("dwi", "dwi"), ("adc", "adc"), ("gt", "label"))
                },
                **common,
                "preprocessing_policy": PREPROCESS_POLICY,
                "gt_decode_policy": BINARY_DECODE_POLICY,
                "brain_mask_policy": BRAIN_MASK_POLICY,
                "target_grid_policy": TARGET_GRID_POLICY,
                "gt_case_status_policy": GT_CASE_STATUS["policy_id"],
                **AXIS_FIELDS,
                "original_grid": _grid_record(p.original_grid),
                "model_grid": _grid_record(p.model_grid),
                "ground_truth_status": case.ground_truth_status,
                "gt_voxels_original": int(case.original_gt.sum()),
                "gt_voxels_model_grid": int(p.label.sum()),
                "gt_components_lost": int(case.gt_components_lost),
                "gt_snapped_voxels": int(p.record["gt_decode"]["snapped_voxels"]),
                "brain_slices": [int(z) for z in case.brain_slices],
                "positive_slices": [int(z) for z in case.positive_slices],
                "preprocessing_record": case.record,
                "arrays": arec,
                **built,
            }
            check_public_text(json.dumps(manifest), needles)
            _write_json(cdir / MANIFEST_NAME, manifest)
            seal = canonical_json_sha256(manifest)
            (cdir / MANIFEST_SEAL).write_text(seal, encoding="utf-8")
            entries[key] = {
                "split": split,
                "manifest_sha256": seal,
                "ground_truth_status": case.ground_truth_status,
            }
            del case, p, arrays  # one case in RAM at a time
    index = {
        "schema": INDEX_SCHEMA,
        "cache_policy": CACHE_POLICY,
        "cache_version": out_dir.name,
        "case_key_policy": CASE_KEY_POLICY,
        **common,
        "counts": {s: sum(e["split"] == s for e in entries.values()) for s in SPLITS},
        "cases": dict(sorted(entries.items())),
        **built,
    }
    _write_json(staging / INDEX_NAME, index)
    (staging / INDEX_SEAL).write_text(canonical_json_sha256(index), encoding="utf-8")
    statuses = [e["ground_truth_status"] for e in entries.values()]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "cache_policy_id": CACHE_POLICY_ID,
        "cache_version": out_dir.name,
        "cases": len(entries),
        "counts": index["counts"],
        "ground_truth_status": {s: statuses.count(s) for s in sorted(set(statuses))},
        "arrays_per_case": list(ARRAYS),
        **AXIS_FIELDS,
        "total_bytes": total_bytes,
        "cache_index_sha256": canonical_json_sha256(index),
        "created_at_utc": utc_now(),
        "source_files_modified": False,
    }
    _write_json(staging / SUMMARY_NAME, summary)
    internal = staging / "_internal"
    internal.mkdir()
    _write_json(internal / "case_key_map.json", {"alias_to_case_key": dict(sorted(keys.items()))})
    os.replace(staging, out_dir)  # publish only a complete cache
    return summary


# ---------------------------------------------------------------- commit binding
def commit_reasons(cache_commit: Any, current_commit: Any) -> list[str]:
    if not _commit_ok(current_commit):
        return ["current_git_commit_unknown"]
    if not _commit_ok(cache_commit):
        return ["cache_git_commit_missing"]
    return [] if cache_commit == current_commit else ["cache_git_commit_mismatch"]


def check_cache_commit(cache_dir: Path, source_control: Mapping[str, Any]) -> list[str]:
    """Cheap check BEFORE the run gate: clean tree, sealed index present, cache built at the
    current commit. Reason codes (empty list = allowed); reads the index only."""
    reasons = []
    if source_control.get("working_tree_clean") is not True:
        reasons.append("working_tree_not_clean")
    root = Path(cache_dir)
    if not (root / INDEX_NAME).is_file() or not (root / INDEX_SEAL).is_file():
        return [*reasons, "cache_index_missing"]
    index = json.loads((root / INDEX_NAME).read_text(encoding="utf-8"))
    if (root / INDEX_SEAL).read_text(encoding="utf-8").strip() != canonical_json_sha256(index):
        return [*reasons, "cache_index_seal_mismatch"]
    return reasons + commit_reasons(index.get("git_commit"), source_control.get("git_commit"))


# ---------------------------------------------------------------- integrity gate
@dataclass
class VerifiedCache:
    root: Path
    index: dict[str, Any]
    index_sha256: str
    manifests: dict[str, dict[str, Any]]


def _check_array(arr: np.ndarray, kind: str) -> str | None:
    if kind == "image":
        return None if np.isfinite(arr).all() else "non_finite"
    if kind == "all_one":
        return None if (arr == 1).all() else "not_all_one"
    return None if ((arr == 0) | (arr == 1)).all() else "not_binary"


def verify_cache(
    cache_dir: Path,
    *,
    config: Mapping[str, Any],
    split_file: Path,
    train_aliases: Iterable[str],
    val_aliases: Iterable[str],
    input_sha256: Mapping[str, Mapping[str, str]],
    package_lock_sha256_lf: str | None,
    git_commit: str | None,
) -> VerifiedCache:
    """Cache integrity gate (reads every array once, one case at a time). ``input_sha256`` is
    the pre-run manifest's ``inputs.file_sha256`` (case key -> role hash). Raises ``CacheError``
    listing every failure; nothing is repaired."""
    root = Path(cache_dir)
    train, val = list(train_aliases), list(val_aliases)
    keys = case_keys(train + val)
    expected = {keys[a]: "train" for a in train} | {keys[a]: "val" for a in val}
    if not (root / INDEX_NAME).is_file() or not (root / INDEX_SEAL).is_file():
        raise CacheError(["cache_index_missing"])
    index = json.loads((root / INDEX_NAME).read_text(encoding="utf-8"))
    index_sha = canonical_json_sha256(index)
    errs: list[str] = []
    if (root / INDEX_SEAL).read_text(encoding="utf-8").strip() != index_sha:
        errs.append("cache_index_seal_mismatch")
    if index.get("schema") != INDEX_SCHEMA:
        errs.append("cache_index_schema")
    if index.get("cache_policy") != CACHE_POLICY or dict(config.get("cache") or {}) != CACHE_POLICY:
        errs.append("cache_policy_mismatch")
    current = {
        "config_canonical_sha256": canonical_json_sha256(dict(config)),
        "split_canonical_sha256": split_canonical_sha256(split_file),
        "alias_list_sha256": alias_list_sha256(train + val, CASE_KEY_POLICY),
        "package_lock_sha256_lf": package_lock_sha256_lf,
    }
    errs += [f"{k}_mismatch" for k, v in current.items() if index.get(k) != v]
    errs += commit_reasons(index.get("git_commit"), git_commit)
    cases = index.get("cases") or {}
    if index.get("counts") != {"train": len(train), "val": len(val)}:
        errs.append("case_count_mismatch")
    if set(cases) != set(expected):
        missing = sorted(set(expected) - set(cases))
        errs.append(f"cache_cases_missing:{len(missing)}")
        errs.append(f"cache_cases_unexpected:{len(set(cases) - set(expected))}")
    dirs = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("case_"))
    if dirs != sorted(cases):
        errs.append("cache_case_folders_differ_from_index")
    if errs:
        raise CacheError(errs)
    manifests: dict[str, dict[str, Any]] = {}
    for key in sorted(expected):
        e = _verify_case(root / key, key, expected[key], cases[key], index, input_sha256)
        if isinstance(e, list):
            errs += [f"{key}:{x}" for x in e]
        else:
            manifests[key] = e
    if errs:
        raise CacheError(errs)
    return VerifiedCache(root, index, index_sha, manifests)


def _verify_case(cdir, key, split, entry, index, input_sha256) -> dict[str, Any] | list[str]:
    errs: list[str] = []
    mp, sp = cdir / MANIFEST_NAME, cdir / MANIFEST_SEAL
    if not mp.is_file() or not sp.is_file():
        return ["cache_manifest_missing"]
    m = json.loads(mp.read_text(encoding="utf-8"))
    seal = canonical_json_sha256(m)
    if sp.read_text(encoding="utf-8").strip() != seal or entry.get("manifest_sha256") != seal:
        errs.append("cache_manifest_seal_mismatch")
    expect = {
        "schema": CASE_SCHEMA,
        "cache_policy_id": CACHE_POLICY_ID,
        "case_key": key,
        "split": split,
        "preprocessing_policy": PREPROCESS_POLICY,
        "gt_decode_policy": BINARY_DECODE_POLICY,
        "brain_mask_policy": BRAIN_MASK_POLICY,
        "target_grid_policy": TARGET_GRID_POLICY,
        "gt_case_status_policy": GT_CASE_STATUS["policy_id"],
        **AXIS_FIELDS,
        **{
            k: index.get(k)
            for k in (
                "config_canonical_sha256",
                "split_canonical_sha256",
                "alias_list_sha256",
                "git_commit",
            )
        },
    }
    errs += [f"{k}_mismatch" for k, v in expect.items() if m.get(k) != v]
    for g in ("original_grid", "model_grid"):
        rec = m.get(g) or {}
        try:
            exact = np.asarray(rec["affine_exact"], np.float64)
            if exact.shape != (4, 4) or rec.get("affine_display") != _display(exact):
                errs.append(f"{g}_affine_display_mismatch")
        except (KeyError, TypeError, ValueError):
            errs.append(f"{g}_affine_invalid")
    src = input_sha256.get(key) or {}
    if not src or {r: src.get(h) for r, h in SOURCE_ROLES.items()} != m.get("source_sha256"):
        errs.append("input_hash_mismatch")
    if set(m.get("arrays") or {}) != set(ARRAYS):
        return errs + ["cache_array_set_differs"]
    try:
        grids = {"model": _grid(m["model_grid"]), "original": _grid(m["original_grid"])}
    except (KeyError, TypeError, ValueError):
        return errs + ["cache_grid_record_invalid"]
    sums: dict[str, int] = {}
    slices: dict[str, list[int]] = {}
    for name, (dtype, grid, kind) in ARRAYS.items():
        rec, path = m["arrays"][name], cdir / f"{name}.npy"
        if not path.is_file():
            errs.append(f"cache_array_missing:{name}")
            continue
        if sha256_file(path) != rec.get("sha256"):
            errs.append(f"cache_array_hash_mismatch:{name}")
            continue
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        shp = grids[grid].shape
        if rec.get("storage_axis_order") != STORAGE_AXIS_ORDER:
            errs.append(f"cache_array_axis_order:{name}")
        elif str(arr.dtype) != dtype or rec.get("dtype") != dtype:
            errs.append(f"cache_array_dtype:{name}")
        elif tuple(arr.shape) != (shp[2], shp[0], shp[1]):
            errs.append(f"cache_array_shape:{name}")
        elif (bad := _check_array(arr, kind)) is not None:
            errs.append(f"cache_array_{bad}:{name}")
        elif name in ("gt", "original_gt"):
            sums[name] = int(arr.sum(dtype=np.int64))
            if name == "gt":
                slices["positive"] = [int(z) for z in np.flatnonzero(arr.any(axis=(1, 2)))]
        elif name == "brain_mask":
            slices["brain"] = [int(z) for z in np.flatnonzero(arr.any(axis=(1, 2)))]
        del arr
    if errs:
        return errs
    brain = slices["brain"]
    positive = [z for z in slices["positive"] if z in set(brain)]
    if m.get("brain_slices") != brain or m.get("positive_slices") != positive:
        errs.append("slice_index_mismatch")
    status = gt_case_status(sums["original_gt"], sums["gt"], int(m.get("gt_components_lost", -1)))
    if (
        status != m.get("ground_truth_status")
        or status != entry.get("ground_truth_status")
        or status == GT_LOST
        or sums["original_gt"] != m.get("gt_voxels_original")
        or sums["gt"] != m.get("gt_voxels_model_grid")
    ):
        errs.append("gt_status_mismatch")
    return errs or m


# ---------------------------------------------------------------- memory-mapped access
class CacheStore:
    """Memory maps of at most ``max_active_cases`` cases (fixed 2, LRU). Arrays are opened
    lazily with ``numpy.load(mmap_mode="r")`` and dropped when their case is evicted."""

    def __init__(self, root: Path, mmap_mode: str = "r", max_active_cases: int = 2):
        if mmap_mode != CACHE_POLICY["mmap_mode"]:
            raise CacheError(["mmap_mode_must_be_r"])
        if max_active_cases != CACHE_POLICY["max_active_cases"]:
            raise CacheError(["max_active_cases_is_fixed"])
        self.root = Path(root)
        self.mmap_mode = mmap_mode
        self.max_active_cases = max_active_cases
        self._open: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.opens = 0  # np.load calls (audit / tests)
        self.peak_active_cases = 0

    @property
    def active_cases(self) -> tuple[str, ...]:
        return tuple(self._open)

    def array(self, key: str, name: str) -> np.ndarray:
        """Stored (Z, X, Y) memory map of one array."""
        if name not in ARRAYS:
            raise KeyError(name)
        arrays = self._open.get(key)
        if arrays is None:
            arrays = self._open[key] = {}
            while len(self._open) > self.max_active_cases:
                self._open.popitem(last=False)
        self._open.move_to_end(key)
        self.peak_active_cases = max(self.peak_active_cases, len(self._open))
        if name not in arrays:
            arrays[name] = np.load(
                self.root / key / f"{name}.npy", mmap_mode=self.mmap_mode, allow_pickle=False
            )
            self.opens += 1
        return arrays[name]

    def volume(self, key: str, name: str) -> np.ndarray:
        """Logical (X, Y, Z) view of a stored (Z, X, Y) array (no copy)."""
        return to_logical(self.array(key, name))

    def close(self) -> None:
        self._open.clear()


class ChannelView:
    """Read-only (2, X, Y, Z) view of a cached case's DWI / ADC; ``view[:, :, :, z]`` reads one
    slice from the memory maps (used by slice-wise inference)."""

    ndim = 4

    def __init__(self, store: CacheStore, key: str, shape_xyz: tuple[int, int, int]):
        self._store, self._key = store, key
        self.shape = (2, *shape_xyz)

    def __getitem__(self, idx):
        if (
            not isinstance(idx, tuple)
            or len(idx) != 4
            or any(s != slice(None) for s in idx[:3])
            or not isinstance(idx[3], (int, np.integer))
        ):
            raise TypeError("ChannelView supports [:, :, :, z] only")
        z = int(idx[3])
        return np.stack(
            [self._store.array(self._key, "dwi")[z], self._store.array(self._key, "adc")[z]]
        )


@dataclass(eq=False)
class CachedCase:
    """Disk-backed case with the ``BaselineCase`` interface. Holds metadata only; arrays come
    from the shared ``CacheStore`` when accessed."""

    case_key: str
    split: str
    store: CacheStore = field(repr=False)
    brain_slices: list[int]
    positive_slices: list[int]
    ground_truth_status: str
    gt_components_lost: int
    model_grid: Grid
    original_grid: Grid
    record: dict[str, Any] = field(repr=False)

    @property
    def negative_only_case(self) -> bool:
        from neuroscan_ai.v2.safety_metrics import NEGATIVE_CASE  # noqa: PLC0415

        return self.ground_truth_status == NEGATIVE_CASE

    @property
    def model_grid_shape(self) -> tuple[int, int, int]:
        return self.model_grid.shape

    @property
    def model_grid_spacing(self) -> tuple[float, float, float]:
        return self.model_grid.spacing

    def slice_arrays(self, z: int):
        a = self.store.array
        k = self.case_key
        image = np.stack([a(k, "dwi")[z], a(k, "adc")[z]])
        return image, a(k, "gt")[z], a(k, "brain_mask")[z], a(k, "valid_region_mask")[z]

    @property
    def original_gt(self) -> np.ndarray:
        return self.store.volume(self.case_key, "original_gt")

    @property
    def prepared(self) -> PreparedCase:
        """``PreparedCase`` over memory-mapped views (nothing is loaded until read)."""
        v = self.store.volume
        k = self.case_key
        return PreparedCase(
            image=ChannelView(self.store, k, self.model_grid.shape),
            label=v(k, "gt"),
            brain=v(k, "brain_mask"),
            model_grid=self.model_grid,
            original_grid=self.original_grid,
            original_brain=v(k, "original_brain_mask"),
            record=self.record,
        )


def open_cases(
    verified: VerifiedCache, store: CacheStore
) -> tuple[list[CachedCase], list[CachedCase]]:
    """Train and validation cases in case-key order (metadata only)."""
    out: dict[str, list[CachedCase]] = {s: [] for s in SPLITS}
    for key in sorted(verified.manifests):
        m = verified.manifests[key]
        out[m["split"]].append(
            CachedCase(
                case_key=key,
                split=m["split"],
                store=store,
                brain_slices=list(m["brain_slices"]),
                positive_slices=list(m["positive_slices"]),
                ground_truth_status=m["ground_truth_status"],
                gt_components_lost=int(m["gt_components_lost"]),
                model_grid=_grid(m["model_grid"]),
                original_grid=_grid(m["original_grid"]),
                record=m["preprocessing_record"],
            )
        )
    return out["train"], out["val"]
