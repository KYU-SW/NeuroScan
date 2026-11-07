"""NeuroScan v2 component safety analysis (stage 1: static analysis only).

Splits an existing final lesion prediction into 26-connected components on the original grid and
records physical, intensity, brain-mask-boundary and 2D/2.5D agreement measurements plus
provisional clinical safety warnings. It never deletes, cuts or edits a prediction voxel, never
multiplies the prediction by the brain mask, never transforms ADC values, never runs a model and
never imports the v1 inference / training / preprocessing code. No verifier, CAM, attention,
occlusion or UI exists at this stage.
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np

from neuroscan_ai.component_analysis.agreement import ModelMasks, component_agreement
from neuroscan_ai.component_analysis.boundary import BrainBoundary
from neuroscan_ai.component_analysis.extraction import (
    ComponentAnalysisError,
    as_binary,
    extract_components,
    identity,
    reconstructs,
    spacing_from_affine,
)
from neuroscan_ai.component_analysis.measurements import adc_stats, adc_status, dwi_stats
from neuroscan_ai.component_analysis.records import (
    SCHEMA,
    check_alias,
    component_record,
    guard_public,
)
from neuroscan_ai.component_analysis.warnings import (
    ACTIVE_CODES,
    DEFERRED_CODES,
    PROVISIONAL_THRESHOLDS,
    case_warnings,
    component_warnings,
    slice_info,
)


def _check_prob(prob: np.ndarray | None, shape: tuple[int, ...], name: str) -> np.ndarray | None:
    if prob is None:
        return None
    p = np.asarray(prob)
    if p.shape != shape:
        raise ComponentAnalysisError(f"{name} shape differs from the prediction")
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ComponentAnalysisError(f"{name} must be finite and within [0, 1]")
    return p


def analyze_case(
    *,
    alias: str,
    prediction: np.ndarray,
    probability: np.ndarray,
    dwi: np.ndarray,
    adc: np.ndarray,
    affine: np.ndarray,
    brain_mask: np.ndarray,
    brain_mask_policy_id: str,
    adc_storage_status: str | None,
    adc_content_status: str | None = None,
    adc_unit_status: str | None = None,
    adc_background_code: float | None = None,
    slice_thickness_mm: float | None = None,
    spacing_between_slices_mm: float | None = None,
    pred_2d: np.ndarray | None = None,
    pred_25d: np.ndarray | None = None,
    prob_2d: np.ndarray | None = None,
    prob_25d: np.ndarray | None = None,
    model_2d: dict[str, str] | None = None,
    model_25d: dict[str, str] | None = None,
    combination_policy: str | None = None,
    threshold_id: str | None = None,
    gt: np.ndarray | None = None,
    forbidden_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Alias-only record ``{"case": ..., "components": [...]}``. Inputs are only read."""
    check_alias(alias, forbidden_ids)
    pred = as_binary(prediction, "prediction")
    shape = pred.shape
    for name, arr in (("dwi", dwi), ("adc", adc), ("brain_mask", brain_mask)):
        if np.asarray(arr).shape != shape:
            raise ComponentAnalysisError(f"{name} shape differs from the prediction")
    prob = _check_prob(probability, shape, "probability")
    brain = as_binary(brain_mask, "brain_mask")
    spacing = spacing_from_affine(affine)
    p2 = as_binary(pred_2d, "pred_2d") if pred_2d is not None else None
    p25 = as_binary(pred_25d, "pred_25d") if pred_25d is not None else None
    for name, arr in (("pred_2d", p2), ("pred_25d", p25)):
        if arr is not None and arr.shape != shape:
            raise ComponentAnalysisError(f"{name} shape differs from the prediction")
    models = ModelMasks(
        p2, p25, _check_prob(prob_2d, shape, "prob_2d"), _check_prob(prob_25d, shape, "prob_25d")
    )
    gt_b = as_binary(gt, "gt") if gt is not None else None
    if gt_b is not None and gt_b.shape != shape:
        raise ComponentAnalysisError("gt shape differs from the prediction")

    labels, n = extract_components(pred)
    if not reconstructs(labels, pred):  # defensive: union of components must equal the prediction
        raise ComponentAnalysisError("component union differs from the prediction")
    bnd = BrainBoundary(brain, spacing)
    status = adc_status(adc_storage_status, adc_content_status, adc_unit_status)
    slices = slice_info(spacing[2], slice_thickness_mm, spacing_between_slices_mm)
    case_level = case_warnings(slices, status)

    comps = []
    for cid in range(1, n + 1):
        comp = labels == cid
        ident = identity(labels, cid, affine, spacing)
        ident["probability_mean"] = float(np.asarray(prob, dtype=np.float64)[comp].mean())
        ident["probability_max"] = float(np.asarray(prob, dtype=np.float64)[comp].max())
        b = bnd.metrics(comp)
        agr = component_agreement(comp, models)
        warns = component_warnings(ident, b, agr, case_level)
        rec = component_record(
            ident,
            dwi_stats(dwi, comp),
            adc_stats(adc, comp, adc_background_code),
            status,
            b,
            agr,
            warns,
        )
        if gt_b is not None:  # development / validation only; clinical output never needs GT
            rec["development_only_gt"] = {
                "gt_overlap_voxels": int((comp & gt_b).sum()),
                "matches_gt": bool((comp & gt_b).any()),
            }
        comps.append(rec)

    counts: dict[str, int] = {c: 0 for c in ACTIVE_CODES}
    for rec in comps:
        for code in rec["warning_codes"]:
            counts[code] += 1
    case = {
        "schema": SCHEMA,
        "alias": alias,
        "shape": list(shape),
        "spacing_mm": [round(s, 6) for s in spacing],
        **slices,
        "brain_mask_policy_id": brain_mask_policy_id,
        "brain_mask_use": "measurement and normalisation only; prediction not cut by the mask",
        "model_2d": model_2d,
        "model_25d": model_25d,
        "combination_policy": combination_policy,
        "threshold_id": threshold_id,
        **{k: v for k, v in status.items()},
        "component_count": n,
        "prediction_voxels": int(pred.sum()),
        "case_warnings": case_level,
        "warnings_summary": {
            "components_with_code": counts,
            "case_level_codes": [w["code"] for w in case_level],
        },
        "provisional_thresholds": PROVISIONAL_THRESHOLDS,
        "deferred_warning_codes": list(DEFERRED_CODES),
        "gt_used": gt_b is not None,
        "not_performed": [
            "prediction_editing",
            "component_removal",
            "brain_mask_cut",
            "resampling",
            "adc_transformation",
            "model_inference",
            "verifier",
            "attribution_maps",
        ],
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    record = {"case": case, "components": comps}
    guard_public(record, forbidden_ids)
    return record


__all__ = ["ComponentAnalysisError", "analyze_case", "extract_components"]
