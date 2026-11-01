"""Lesion quantification of the restored prediction mask (original DWI grid).

Volumes, components and coordinates reuse ``analysis.lesion_summary`` / ``analysis.components``
(26-connectivity, nothing removed). World coordinates are NIfTI RAS+ scanner millimetres — not
atlas coordinates.

Hemisphere (D3, *estimated*): the midline is the world-x centroid of the brain mask, accepted only
when it agrees with the brain bounding-box centre and the brain is left/right balanced around it;
voxels within +-band of it count as "midline". RAS+ means world +x = patient right. Never derived
from array indices. Oblique images or failed checks -> ``not_available``. No vascular territory
(MCA/ACA/PCA) is reported without atlas registration.

Score summaries are *uncalibrated model output scores* on the processed grid (after gating) —
technical research indicators, not probabilities of disease.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import nibabel as nib
import numpy as np

from neuroscan_ai.analysis.components import component_analysis
from neuroscan_ai.analysis.lesion import lesion_summary
from neuroscan_ai.inference.errors import Code, Issue
from neuroscan_ai.inference.predict import Prediction
from neuroscan_ai.inference.preprocess import PreparedCase
from neuroscan_ai.inference.validation import obliquity_deg
from neuroscan_ai.io import nifti

SCORE_NOTE = (
    "Uncalibrated model output scores (sigmoid) for research/technical review. "
    "They are not probabilities of disease and not clinical confidence."
)
HEMISPHERE_CAVEAT = (
    "Estimated from the brain-mask midline in scanner (RAS+) coordinates; not registered to an "
    "atlas; head rotation is not corrected."
)


def _r(x: float, nd: int = 6) -> float:
    return round(float(x), nd)


def _world(affine: np.ndarray, ijk) -> list[float]:
    return [_r(v, 4) for v in (affine @ np.array([*ijk, 1.0]))[:3]]


def _world_x(affine: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return idx @ affine[0, :3] + affine[0, 3]


def hemisphere_estimate(
    mask: np.ndarray,
    affine: np.ndarray,
    spacing,
    brain: np.ndarray,
    brain_affine: np.ndarray,
    components: list[dict],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, str]]:
    out: dict[str, Any] = {
        "status": "not_available",
        "estimated": True,
        "method": "brain_mask_midline_world_x",
        "caveat": HEMISPHERE_CAVEAT,
        "left_volume_ml": None,
        "right_volume_ml": None,
        "midline_band_volume_ml": None,
        "bilateral": None,
        "reason": None,
    }
    labels = {c["label"]: "not_available" for c in components}
    if obliquity_deg(affine) > cfg["max_obliquity_deg"]:
        out["reason"] = "image axes too oblique"
        return out, labels
    if nifti.orientation(brain_affine) != "RAS" or set(nib.aff2axcodes(affine)) & {None}:
        out["reason"] = "orientation could not be verified"
        return out, labels
    bx = _world_x(brain_affine, np.argwhere(brain))
    lo, hi = float(bx.min()), float(bx.max())
    mid = float(bx.mean())
    width = hi - lo
    asym = abs((mid - lo) - (hi - mid)) / width if width > 0 else 1.0
    if (
        abs(mid - (lo + hi) / 2) > cfg["max_midline_disagreement_mm"]
        or asym > cfg["max_brain_asymmetry"]
    ):
        out["reason"] = "brain midline could not be verified"
        return out, labels
    band = cfg["midline_band_mm"]
    vox_ml = float(np.prod(spacing[:3])) / 1000.0
    lx = _world_x(affine, np.argwhere(mask > 0))
    right = int((lx > mid + band).sum())  # RAS+: +x = patient right
    left = int((lx < mid - band).sum())
    centre = int(lx.size - right - left)
    out.update(
        status="estimated",
        reason=None,
        midline_world_x_mm=_r(mid, 3),
        midline_band_mm=band,
        left_volume_ml=_r(left * vox_ml),
        right_volume_ml=_r(right * vox_ml),
        midline_band_volume_ml=_r(centre * vox_ml),
        bilateral=bool(
            left * vox_ml >= cfg["bilateral_min_ml"] and right * vox_ml >= cfg["bilateral_min_ml"]
        ),
    )
    for c in components:
        x = c["centroid_world_mm"][0]
        labels[c["label"]] = "right" if x > mid + band else "left" if x < mid - band else "midline"
    return out, labels


def score_summary(prediction: Prediction, prepared: PreparedCase, cfg: dict[str, Any]) -> dict:
    prob = prediction.probability_processed.data
    lesion = prediction.binary_processed.data > 0
    brain = prepared.case.brain
    lo_low, hi_low = cfg["low_score_band"]
    lo_unc, hi_unc = cfg["uncertain_band"]
    p = prob[lesion]
    in_brain = prob[brain]
    unc = int(((in_brain > lo_unc) & (in_brain < hi_unc)).sum())
    summary: dict[str, Any] = {
        "kind": "uncalibrated_model_output_score",
        "calibrated": False,
        "space": "processed",
        "note": SCORE_NOTE,
        "max_score_in_brain": _r(in_brain.max()) if in_brain.size else None,
        "uncertain_band": [lo_unc, hi_unc],
        "uncertain_voxels_in_brain": unc,
        "uncertain_fraction_of_brain": _r(unc / in_brain.size) if in_brain.size else None,
        "predicted_region": None,
    }
    if p.size:
        summary["predicted_region"] = {
            "mean_score": _r(p.mean()),
            "median_score": _r(np.median(p)),
            "max_score": _r(p.max()),
            "low_score_band": [lo_low, hi_low],
            "low_score_fraction": _r(((p >= lo_low) & (p < hi_low)).mean()),
            "fraction_0.7_to_0.9": _r(((p >= 0.7) & (p < 0.9)).mean()),
            "fraction_at_least_0.9": _r((p >= 0.9).mean()),
        }
    return summary


def quantify(
    prediction: Prediction, prepared: PreparedCase, inference_cfg: dict[str, Any], threshold: float
) -> tuple[dict[str, Any], list[Issue]]:
    q, issues, _ = quantify_with_labelmap(prediction, prepared, inference_cfg, threshold)
    return q, issues


def quantify_with_labelmap(
    prediction: Prediction, prepared: PreparedCase, inference_cfg: dict[str, Any], threshold: float
) -> tuple[dict[str, Any], list[Issue], np.ndarray]:
    """``quantify`` plus the uint8 component display labelmap (Phase 7-A) from the same single
    component labelling, so labelmap labels and the JSON ``display_label`` always agree."""
    vol = prediction.binary_original
    mask = vol.data
    affine = vol.geometry.affine
    spacing = nifti.spacing_from_affine(affine)
    orient = nifti.orientation(affine)
    slice_axis = next(i for i, c in enumerate(orient) if c in "SI")
    summary = lesion_summary(mask, spacing, affine, slice_axis=slice_axis, with_components=False)
    labelmap, comps = component_analysis(mask, spacing, affine, slice_axis)
    qcfg = inference_cfg["quantification"]
    hemi, comp_side = hemisphere_estimate(
        mask,
        affine,
        spacing,
        prepared.case.brain,
        prepared.processed.affine,
        comps,
        qcfg["hemisphere"],
    )
    detected = summary["voxel_count"] > 0
    q: dict[str, Any] = {
        "space": "original_dwi_grid",
        "coordinate_system": "RAS+ scanner world coordinates (mm), voxel-centre convention",
        "prediction_threshold": threshold,
        "lesion_detected": detected,
        "voxel_count": summary["voxel_count"],
        "total_lesion_volume_ml": _r(summary["volume_ml"]),
        "component_count": len(comps),
        "largest_component_volume_ml": _r(comps[0]["volume_ml"]) if comps else 0.0,
        "affected_slice_count": summary["positive_slice_count"],
        "slice_axis": slice_axis,
        "bounding_box_voxel": None,
        "bounding_box_world_mm": None,
        "center_of_mass_voxel": None,
        "center_of_mass_world_mm": None,
        "hemisphere": hemi,
        "components": [
            {
                "label": c["label"],
                "voxel_count": c["voxel_count"],
                "volume_ml": _r(c["volume_ml"]),
                "bbox_min_voxel": c["bbox_min"],
                "bbox_max_voxel": c["bbox_max"],
                "centroid_voxel": [_r(v, 3) for v in c["centroid_voxel"]],
                "centroid_world_mm": [_r(v, 4) for v in c["centroid_world_mm"]],
                "hemisphere_estimate": comp_side[c["label"]],
                "display_label": c["display_label"],
                "size_rank": c["size_rank"],
                "representative_voxel": c["representative_voxel"],
                "representative_world_mm": [_r(v, 4) for v in c["representative_world_mm"]],
                "slice_range": c["slice_range"],
            }
            for c in comps
        ],
        "research_score_summary": score_summary(prediction, prepared, qcfg),
        "vascular_territory": None,
        "vascular_territory_note": "Not estimated: requires atlas registration (not performed).",
    }
    if detected:
        idx = np.argwhere(mask > 0)
        bmin, bmax = idx.min(axis=0), idx.max(axis=0)
        corners = np.array([_world(affine, c) for c in product(*zip(bmin, bmax, strict=True))])
        com = idx.mean(axis=0)
        q["bounding_box_voxel"] = {"min": bmin.tolist(), "max": bmax.tolist()}
        q["bounding_box_world_mm"] = {
            "min": [_r(v, 4) for v in corners.min(axis=0)],
            "max": [_r(v, 4) for v in corners.max(axis=0)],
        }
        q["center_of_mass_voxel"] = [_r(v, 3) for v in com]
        q["center_of_mass_world_mm"] = _world(affine, com)

    issues: list[Issue] = []
    w = inference_cfg["warnings"]
    brain_ml = float(prepared.case.brain.sum()) * float(np.prod(prepared.processed.spacing)) / 1000
    total = summary["volume_ml"]
    if total > w["large_lesion_ml"] or (
        brain_ml > 0 and total > w["large_lesion_brain_fraction"] * brain_ml
    ):
        issues.append(
            Issue(
                Code.W_LARGE_LESION,
                "Predicted region is unusually large; please review the images.",
                f"{total:.1f} mL predicted, brain {brain_ml:.0f} mL (technical threshold)",
            )
        )
    if len(comps) > w["many_components"]:
        issues.append(
            Issue(
                Code.W_MANY_COMPONENTS,
                "Predicted region is split into many separate parts; please review the images.",
                f"{len(comps)} components (technical threshold)",
            )
        )
    q["flags"] = [i.code.value for i in issues]
    return q, issues, labelmap
