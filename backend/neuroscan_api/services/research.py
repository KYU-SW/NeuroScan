"""Phase 7-B research validation cases: registration (CLI only) and read access (API).

Isolation from general data:
- own tables (research_cases, research_files) and own storage area ``research/``; general cases,
  runs and file_resources are never read or written here, and no general route returns a
  research file id;
- ``/files/{id}`` serves a research file only while ``research_enabled`` is true; the research
  routes themselves are not registered otherwise (plain 404);
- public responses carry the alias only. The alias -> source case mapping is a local file
  (``research/_internal/aliases.json``) that is never registered as a servable file.

Registration is all-or-nothing: the bundle is built in ``research/.staging/<uuid>``, verified,
renamed to ``research/<alias>`` and recorded in one database transaction; the alias map is
written last. Any failure rolls the directory and the rows back.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select

from neuroscan_api import API_PREFIX
from neuroscan_api.context import AppContext, audit, iso
from neuroscan_api.db.models import ResearchCase, ResearchFile
from neuroscan_api.db.session import Database
from neuroscan_api.errors import ApiError
from neuroscan_api.safety import violations
from neuroscan_api.schemas import (
    Component,
    GroundTruthComponent,
    GroundTruthMatch,
    LegendItem,
    MatchGroup,
    PredictionMatch,
    Reproducibility,
    ResearchCaseDetail,
    ResearchCaseList,
    ResearchCaseSummary,
    ResearchMatching,
    ResearchMetrics,
    ResearchModel,
    ResearchVolume,
    SliceRange,
)
from neuroscan_api.storage import Storage, StorageError

LOGGER = logging.getLogger("neuroscan_api")
ALIAS = re.compile(r"^val-\d{3}$")
SCHEMA = "neuroscan.research_case/1"
INTERNAL_DIR = "_internal"
STAGING_DIR = ".staging"
VOLUME_ROLES = ("dwi", "adc", "prediction_mask", "prediction_components", "difference")
DOCUMENTS = ("metrics.json", "components.json", "matching.json")
# research responses legitimately contain evaluation keys (dice, gt ...); every other safety
# rule of the general contract still applies, plus: no ISLES case IDs anywhere
CASE_KEY_PATTERN = re.compile(r"sub-strokecase\d+", re.IGNORECASE)
RESEARCH_ALLOWED_KEYS = frozenset({"dice", "iou", "sensitivity", "gt_volume_ml", "ground_truth"})


class ResearchRegistrationError(RuntimeError):
    pass


def research_not_found() -> ApiError:
    return ApiError(404, "RESEARCH_CASE_NOT_FOUND", "Research case not found.")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


# --- registration (CLI) ------------------------------------------------------------------------


def staging_dir(storage: Storage) -> Path:
    path = storage.area("research") / STAGING_DIR / str(uuid.uuid4())
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def alias_exists(db: Database, alias: str) -> bool:
    with db.read() as s:
        return s.scalar(select(ResearchCase.id).where(ResearchCase.alias == alias)) is not None


def _verify_bundle(bundle: Path, alias: str) -> dict[str, Any]:
    manifest = json.loads((bundle / "research_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA or manifest.get("alias") != alias:
        raise ResearchRegistrationError("bundle manifest does not match")
    if manifest.get("split") != "val":
        raise ResearchRegistrationError("only validation research cases can be registered")
    for vid in VOLUME_ROLES:
        v = manifest["volumes"][vid]
        if _sha(bundle / v["file"]) != v["sha256"]:
            raise ResearchRegistrationError(f"bundle file hash mismatch: {vid}")
    for name in DOCUMENTS:
        if _sha(bundle / name) != manifest["documents"][name]:
            raise ResearchRegistrationError(f"bundle document hash mismatch: {name}")
    text = "".join((bundle / n).read_text(encoding="utf-8") for n in (*DOCUMENTS,))
    if CASE_KEY_PATTERN.search(text + json.dumps(manifest)):
        raise ResearchRegistrationError("bundle contains a source case ID")
    return manifest


def _write_alias_map(storage: Storage, alias: str, entry: dict[str, str]) -> None:
    internal = storage.area("research") / INTERNAL_DIR
    internal.mkdir(parents=True, exist_ok=True)
    path = internal / "aliases.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[alias] = entry
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def register_bundle(
    db: Database, storage: Storage, staging: Path, alias: str, source_entry: dict[str, str]
) -> str:
    """Atomically move a verified staging bundle to research/<alias> and record it."""
    if not ALIAS.fullmatch(alias):
        raise ResearchRegistrationError("invalid alias")
    if alias_exists(db, alias):
        raise ResearchRegistrationError("alias already registered")
    manifest = _verify_bundle(staging, alias)
    final = storage.area("research") / alias
    if final.exists():
        raise ResearchRegistrationError("research folder for alias already exists")
    rid = str(uuid.uuid4())
    moved = False
    try:
        with db.write() as s:
            s.add(
                ResearchCase(
                    id=rid,
                    alias=alias,
                    dataset_id=manifest["dataset_id"],
                    split=manifest["split"],
                    source_model=manifest["model"]["model_name"],
                    checkpoint_sha256=manifest["model"]["checkpoint_sha256"],
                    config_hash=manifest["model"]["config_hash"],
                    threshold=float(manifest["model"]["threshold"]),
                    preprocessing_version=manifest["model"]["preprocessing_version"],
                    bundle_rel=alias,
                    metrics_path=f"{alias}/metrics.json",
                    component_matching_path=f"{alias}/matching.json",
                    has_ground_truth=bool(manifest["has_ground_truth"]),
                )
            )
            s.flush()
            for vid in VOLUME_ROLES:
                v = manifest["volumes"][vid]
                s.add(
                    ResearchFile(
                        id=str(uuid.uuid4()),
                        research_case_id=rid,
                        role=vid,
                        storage_area="research",
                        rel_path=f"{alias}/{v['file']}",
                        mime_type="application/gzip",
                        size_bytes=int(v["size_bytes"]),
                        sha256=v["sha256"],
                    )
                )
            audit(s, "research_case_registered", alias=alias)
            s.flush()
            os.rename(staging, final)  # same volume: atomic
            moved = True
            _write_alias_map(storage, alias, source_entry)
    except Exception:
        if moved and final.exists() and not staging.exists():
            os.rename(final, staging)
        raise
    return rid


def discard_staging(staging: Path) -> None:
    shutil.rmtree(staging, ignore_errors=True)


# --- read access (API) -------------------------------------------------------------------------


def _case(ctx: AppContext, alias: str) -> tuple[ResearchCase, dict[str, ResearchFile]]:
    if not ALIAS.fullmatch(alias):
        raise research_not_found()
    with ctx.db.read() as s:
        row = s.scalar(select(ResearchCase).where(ResearchCase.alias == alias))
        if row is None:
            raise research_not_found()
        files = {
            f.role: f
            for f in s.scalars(
                select(ResearchFile).where(ResearchFile.research_case_id == row.id)
            ).all()
        }
    return row, files


def _load(ctx: AppContext, rel: str) -> dict[str, Any]:
    try:
        path = ctx.storage.resolve("research", rel)
    except StorageError:
        raise ApiError(500, "RESULT_INTEGRITY_ERROR", "Research case is incomplete.") from None
    return json.loads(path.read_text(encoding="utf-8"))


def _research_violations(obj: Any) -> list[str]:
    bad = [
        v
        for v in violations(obj)
        if not (v.startswith("forbidden key") and v.rsplit(".", 1)[-1] in RESEARCH_ALLOWED_KEYS)
    ]
    if CASE_KEY_PATTERN.search(json.dumps(obj)):
        bad.append("source case ID in research response")
    return bad


def list_cases(ctx: AppContext) -> ResearchCaseList:
    with ctx.db.read() as s:
        rows = s.scalars(select(ResearchCase).order_by(ResearchCase.alias)).all()
    return ResearchCaseList(
        cases=[
            ResearchCaseSummary(
                alias=r.alias,
                dataset_id=r.dataset_id,
                split="val",
                has_ground_truth=r.has_ground_truth,
                created_at=iso(r.created_at) or "",
            )
            for r in rows
            if r.split == "val"
        ]
    )


def get_case(ctx: AppContext, alias: str) -> ResearchCaseDetail:
    row, files = _case(ctx, alias)
    if row.split != "val":
        raise research_not_found()
    manifest = _load(ctx, f"{row.bundle_rel}/research_manifest.json")
    metrics = _load(ctx, row.metrics_path)["metrics"]
    comps = _load(ctx, f"{row.bundle_rel}/components.json")
    matching = _load(ctx, row.component_matching_path)
    volumes = []
    for vid in VOLUME_ROLES:
        f = files.get(vid)
        v = manifest["volumes"][vid]
        if f is None or f.sha256 != v["sha256"]:
            raise ApiError(500, "RESULT_INTEGRITY_ERROR", "Research case is incomplete.")
        volumes.append(
            ResearchVolume(
                id=vid,
                role=v["role"],
                file_id=f.id,
                url=f"{API_PREFIX}/files/{f.id}",
                mime_type=f.mime_type,
                sha256=f.sha256,
                shape=v["shape"],
                spacing_mm=v["spacing_mm"],
                affine=v["affine"],
                orientation=v["orientation"],
                dtype=v["dtype"],
            )
        )
    gt_by_rank = {g["size_rank"]: g for g in comps["ground_truth_components"]}
    detail = ResearchCaseDetail(
        alias=row.alias,
        dataset_id=row.dataset_id,
        split="val",
        created_at=iso(row.created_at) or "",
        model=ResearchModel(**manifest["model"]),
        metrics=ResearchMetrics(**metrics),
        prediction_components=[
            Component(
                label=c["label"],
                voxel_count=c["voxel_count"],
                volume_ml=c["volume_ml"],
                centroid_voxel=c["centroid_voxel"],
                centroid_world_mm=c["centroid_world_mm"],
                bbox_min_voxel=c["bbox_min_voxel"],
                bbox_max_voxel=c["bbox_max_voxel"],
                hemisphere_estimate=c["hemisphere_estimate"],
                display_label=c["display_label"],
                size_rank=c["size_rank"],
                representative_voxel=c["representative_voxel"],
                representative_world_mm=c["representative_world_mm"],
                slice_range=SliceRange(**c["slice_range"]),
            )
            for c in comps["prediction_components"]
        ],
        ground_truth_components=[
            GroundTruthComponent(
                size_rank=g["size_rank"],
                voxel_count=g["voxel_count"],
                volume_ml=g["volume_ml"],
                centroid_world_mm=g["centroid_world_mm"],
                representative_voxel=g["representative_voxel"],
                representative_world_mm=g["representative_world_mm"],
                slice_range=SliceRange(**g["slice_range"]),
            )
            for g in sorted(gt_by_rank.values(), key=lambda g: g["size_rank"])
        ],
        matching=ResearchMatching(
            method=matching["method"],
            groups=[MatchGroup(**g) for g in matching["groups"]],
            prediction=[PredictionMatch(**p) for p in matching["prediction_components"]],
            ground_truth=[
                GroundTruthMatch(
                    size_rank=g["size_rank"],
                    detected_voxels=g["detected_voxels"],
                    missed_voxels=g["missed_voxels"],
                    group_id=g["group_id"],
                    group_type=g["group_type"],
                    prediction_component_ids=g["prediction_component_ids"],
                )
                for g in matching["ground_truth_components"]
            ],
        ),
        volumes=volumes,
        views=manifest["views"],
        legend=[LegendItem(**item) for item in manifest["legend"]],
        gt_only_color="#CC79A7",
        reproducibility=Reproducibility(
            reference=manifest["reproducibility"].get("reference"),
            identical=manifest["reproducibility"].get("identical"),
            mismatched_voxels=manifest["reproducibility"].get("mismatched_voxels"),
        ),
    )
    bad = _research_violations(detail.model_dump(mode="json"))
    if bad:
        LOGGER.error("research response failed safety checks: %s", "; ".join(bad))
        raise ApiError(500, "RESULT_INTEGRITY_ERROR", "Research case failed safety checks.")
    return detail


def research_file(ctx: AppContext, file_id: str) -> ResearchFile | None:
    """Research file row for /files/{id}, or None (also when research mode is off)."""
    if not ctx.settings.research_enabled:
        return None
    with ctx.db.read() as s:
        return s.get(ResearchFile, file_id)
