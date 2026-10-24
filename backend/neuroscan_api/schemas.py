"""Pydantic response models. Every field is explicit (extra="forbid"); nothing from the engine
JSON is passed through wholesale. No paths, file names, header text, tracebacks, ground truth or
probability-of-disease semantics."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from neuroscan_api.safety import DISCLAIMER


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorInfo(Strict):
    code: str
    message: str
    request_id: str | None = None
    retryable: bool = False
    details: dict[str, Any] = {}


class ErrorResponse(Strict):
    error: ErrorInfo


class WarningItem(Strict):
    code: str
    message: str


class Disclaimer(Strict):
    en: str = DISCLAIMER["en"]
    ko: str = DISCLAIMER["ko"]


class HealthResponse(Strict):
    status: Literal["ok", "degraded"]
    database: Literal["ok"]
    worker: Literal["online", "offline", "stale", "degraded", "loading"]
    queue_depth: int
    api_version: str
    disclaimer: Disclaimer = Disclaimer()


class ModelInfoResponse(Strict):
    model_name: str | None
    model_type: str | None
    training_run_id: str | None
    checkpoint_sha256: str | None
    config_hash: str | None
    input_channels: int | None
    context_slices: int | None
    threshold: float | None
    device: str | None
    validation_positive_mean_dice: float | None
    validation_note: str = "internal validation used for checkpoint selection (optimistic)"
    loaded_at: str | None
    worker_state: str


class CaseInput(Strict):
    role: Literal["dwi", "adc"]
    sha256: str
    size_bytes: int
    raw_retained: bool
    downloadable: bool = False


class RunSummary(Strict):
    run_id: str
    status: str
    created_at: str
    completed_at: str | None


class CaseResponse(Strict):
    case_id: str
    created_at: str
    status: str
    inputs: list[CaseInput]
    runs: list[RunSummary]
    links: dict[str, str]


class CaseDeleteResponse(Strict):
    case_id: str
    deleted: bool
    runs_deleted: int
    files_deleted: int


class RunCreateRequest(Strict):
    allow_rerun: bool = False


class RunModel(Strict):
    model_name: str | None
    checkpoint_sha256: str | None
    config_hash: str | None


class RunError(Strict):
    code: str
    message: str
    retryable: bool


class RunStatusResponse(Strict):
    run_id: str
    case_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    current_stage: str | None
    progress: float = Field(ge=0.0, le=1.0)
    queue_position: int | None
    created_at: str
    started_at: str | None
    completed_at: str | None
    warning_count: int
    warnings: list[WarningItem]
    error: RunError | None
    model: RunModel
    links: dict[str, str]


class RunCreateResponse(Strict):
    run_id: str
    case_id: str
    status: Literal["queued"]
    queue_position: int
    worker_online: bool
    links: dict[str, str]


class Hemisphere(Strict):
    status: str
    estimated: bool
    method: str | None
    left_volume_ml: float | None
    right_volume_ml: float | None
    midline_band_volume_ml: float | None
    bilateral: bool | None
    reason: str | None
    caveat: str


class Component(Strict):
    label: int
    voxel_count: int
    volume_ml: float
    centroid_voxel: list[float]
    centroid_world_mm: list[float]
    bbox_min_voxel: list[int]
    bbox_max_voxel: list[int]
    hemisphere_estimate: str


class Lesion(Strict):
    detected: bool
    total_volume_ml: float
    voxel_count: int
    component_count: int
    largest_component_volume_ml: float
    affected_slice_count: int
    prediction_threshold: float
    space: str
    coordinate_system: str
    display_text: str | None


class Location(Strict):
    bounding_box_voxel: dict[str, list[int]] | None
    bounding_box_world_mm: dict[str, list[float]] | None
    center_of_mass_voxel: list[float] | None
    center_of_mass_world_mm: list[float] | None
    hemisphere: Hemisphere


class PredictedRegionScores(Strict):
    mean_score: float
    median_score: float
    max_score: float
    low_score_fraction: float


class ResearchScores(Strict):
    kind: str
    calibrated: bool
    note: str
    max_score_in_brain: float | None
    uncertain_fraction_of_brain: float | None
    predicted_region: PredictedRegionScores | None


class ViewerVolume(Strict):
    id: str
    role: str
    file_id: str
    url: str
    mime_type: str
    sha256: str
    shape: list[int]
    spacing_mm: list[float]
    affine: list[list[float]]
    orientation: str
    dtype: str
    shareable: bool
    labels: dict[str, str] | None = None
    same_grid_as: str | None = None


class Viewer(Strict):
    coordinate_system: dict[str, str]
    volumes: list[ViewerVolume]
    views: dict[str, dict[str, Any]]
    render_3d: dict[str, Any]


class RunResultResponse(Strict):
    run_id: str
    case_id: str
    engine_run_id: str
    completed_at: str | None
    model: RunModel
    lesion: Lesion
    location: Location
    components: list[Component]
    components_truncated: bool
    research_score_summary: ResearchScores
    vascular_territory: None = None
    vascular_territory_note: str
    flags: list[str]
    warnings: list[WarningItem]
    viewer: Viewer
    disclaimer: Disclaimer = Disclaimer()
