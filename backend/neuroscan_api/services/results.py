"""Explicit allow-list mapping of Phase 5A quantification.json + manifest.json to the API result.

Nothing is copied wholesale: each field is named here. Manifest URIs (run-relative paths) are
replaced by file_id URLs from the registry; an URI without a registered file is an integrity
error. The final response is scanned with ``safety.violations`` before it is returned.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from neuroscan_api import API_PREFIX
from neuroscan_api.context import AppContext, iso
from neuroscan_api.db.models import FileResource, InferenceRun
from neuroscan_api.errors import ApiError, engine_retryable
from neuroscan_api.safety import NO_LESION_TEXT, violations
from neuroscan_api.schemas import (
    Component,
    Hemisphere,
    Lesion,
    Location,
    PredictedRegionScores,
    ResearchScores,
    RunModel,
    RunResultResponse,
    Viewer,
    ViewerVolume,
    WarningItem,
)
from neuroscan_api.services.files import load_run_json
from neuroscan_api.services.runs import mark_stale_runs, run_not_found

LOGGER = logging.getLogger("neuroscan_api")
MAX_COMPONENTS = 100
VOLUME_ROLES = {
    "dwi": "input_dwi_deid",
    "adc": "input_adc_deid",
    "prediction_mask": "prediction_mask",
}


def _integrity(what: str) -> ApiError:
    LOGGER.error("result integrity check failed: %s", what)
    return ApiError(500, "RESULT_INTEGRITY_ERROR", "Result failed integrity verification.")


def _viewer(manifest: dict, run: InferenceRun, files: dict[str, FileResource]) -> Viewer:
    volumes = []
    by_rel = {f.rel_path: f for f in files.values()}
    for v in manifest.get("volumes", []):
        role = VOLUME_ROLES.get(v.get("id"))
        row = by_rel.get(f"{run.run_dir_rel}/{v.get('uri', '')}")
        if role is None or row is None or row.role != role or not row.servable:
            raise _integrity("manifest volume without registered file")
        if row.sha256 != v.get("sha256"):
            raise _integrity("manifest sha mismatch")
        volumes.append(
            ViewerVolume(
                id=v["id"],
                role=v["role"],
                file_id=row.id,
                url=f"{API_PREFIX}/files/{row.id}",
                mime_type=row.mime_type,
                sha256=row.sha256,
                shape=v["shape"],
                spacing_mm=v["spacing_mm"],
                affine=v["affine"],
                orientation=v["orientation"],
                dtype=v["dtype"],
                shareable=row.shareable,
                labels=v.get("labels"),
                same_grid_as=v.get("same_grid_as"),
            )
        )
    render = manifest.get("render_3d", {})
    return Viewer(
        coordinate_system={
            "world": manifest["coordinate_system"]["world"],
            "note": manifest["coordinate_system"]["note"],
        },
        volumes=volumes,
        views={
            name: {
                "array_axis": int(view["array_axis"]),
                "count": int(view["count"]),
                "axis_increases_toward": str(view["axis_increases_toward"]),
            }
            for name, view in manifest.get("views", {}).items()
        },
        render_3d={
            "mask_available": bool(render.get("mask_available")),
            "empty_mask": bool(render.get("empty_mask")),
            "suggested": str(render.get("suggested", "")),
        },
    )


def map_result(run: InferenceRun, quant: dict, manifest: dict, files) -> RunResultResponse:
    h = quant["hemisphere"]
    scores = quant["research_score_summary"]
    region = scores.get("predicted_region")
    comps = quant.get("components", [])
    return RunResultResponse(
        run_id=run.id,
        case_id=run.case_id,
        engine_run_id=run.engine_run_id,
        completed_at=iso(run.completed_at),
        model=RunModel(
            model_name=run.model_name,
            checkpoint_sha256=run.checkpoint_sha256,
            config_hash=run.config_hash,
        ),
        lesion=Lesion(
            detected=bool(quant["lesion_detected"]),
            total_volume_ml=float(quant["total_lesion_volume_ml"]),
            voxel_count=int(quant["voxel_count"]),
            component_count=int(quant["component_count"]),
            largest_component_volume_ml=float(quant["largest_component_volume_ml"]),
            affected_slice_count=int(quant["affected_slice_count"]),
            prediction_threshold=float(quant["prediction_threshold"]),
            space=str(quant["space"]),
            coordinate_system=str(quant["coordinate_system"]),
            display_text=None if quant["lesion_detected"] else NO_LESION_TEXT,
        ),
        location=Location(
            bounding_box_voxel=quant.get("bounding_box_voxel"),
            bounding_box_world_mm=quant.get("bounding_box_world_mm"),
            center_of_mass_voxel=quant.get("center_of_mass_voxel"),
            center_of_mass_world_mm=quant.get("center_of_mass_world_mm"),
            hemisphere=Hemisphere(
                status=h["status"],
                estimated=bool(h["estimated"]),
                method=h.get("method"),
                left_volume_ml=h.get("left_volume_ml"),
                right_volume_ml=h.get("right_volume_ml"),
                midline_band_volume_ml=h.get("midline_band_volume_ml"),
                bilateral=h.get("bilateral"),
                reason=h.get("reason"),
                caveat=h["caveat"],
            ),
        ),
        components=[
            Component(
                label=c["label"],
                voxel_count=c["voxel_count"],
                volume_ml=c["volume_ml"],
                centroid_voxel=c["centroid_voxel"],
                centroid_world_mm=c["centroid_world_mm"],
                bbox_min_voxel=c["bbox_min_voxel"],
                bbox_max_voxel=c["bbox_max_voxel"],
                hemisphere_estimate=c["hemisphere_estimate"],
            )
            for c in comps[:MAX_COMPONENTS]
        ],
        components_truncated=len(comps) > MAX_COMPONENTS,
        research_score_summary=ResearchScores(
            kind=scores["kind"],
            calibrated=bool(scores["calibrated"]),
            note=scores["note"],
            max_score_in_brain=scores.get("max_score_in_brain"),
            uncertain_fraction_of_brain=scores.get("uncertain_fraction_of_brain"),
            predicted_region=None
            if region is None
            else PredictedRegionScores(
                mean_score=region["mean_score"],
                median_score=region["median_score"],
                max_score=region["max_score"],
                low_score_fraction=region["low_score_fraction"],
            ),
        ),
        vascular_territory_note=str(quant.get("vascular_territory_note", "")),
        flags=[str(f) for f in quant.get("flags", [])],
        warnings=[WarningItem(code=w["code"], message=w["message"]) for w in run.warnings or []],
        viewer=_viewer(manifest, run, files),
    )


def get_result(ctx: AppContext, run_id: str) -> RunResultResponse:
    mark_stale_runs(ctx)
    with ctx.db.read() as s:
        run = s.get(InferenceRun, run_id)
        if run is None:
            raise run_not_found()
        if run.status in ("queued", "running"):
            raise ApiError(
                409,
                "RUN_NOT_FINISHED",
                "The run has not finished yet.",
                details={"status": run.status},
            )
        if run.status == "cancelled":
            raise ApiError(409, "RUN_CANCELLED", "The run was cancelled.")
        if run.status == "failed":
            raise ApiError(
                409,
                "RUN_FAILED",
                "The run failed.",
                details={"engine_code": run.error_code},
                retryable=engine_retryable(run.error_code),
            )
        rows = s.scalars(select(FileResource).where(FileResource.run_id == run_id)).all()
    files = {r.role: r for r in rows}
    quant = load_run_json(ctx, files.get("quantification_json"))
    manifest = load_run_json(ctx, files.get("manifest_json"))
    try:
        result = map_result(run, quant, manifest, files)
    except (KeyError, TypeError, ValueError) as exc:
        raise _integrity(f"mapping {type(exc).__name__}") from None
    bad = violations(result.model_dump(mode="json"))
    if bad:
        raise _integrity("; ".join(bad))
    return result
