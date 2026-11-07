"""Clinical safety warning codes. Every threshold here is PROVISIONAL (design candidate, not
validated); thresholds are fixed later on validation data only. Warnings add information and
never delete or change a prediction."""

from __future__ import annotations

from typing import Any

from neuroscan_ai.component_analysis.agreement import LOW_OVERLAP_DICE
from neuroscan_ai.component_analysis.boundary import CONTACT_MM

BOUNDARY_CONTACT = "BOUNDARY_CONTACT"
EXTENDS_OUTSIDE_BRAINMASK = "EXTENDS_OUTSIDE_BRAINMASK"
MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
SMALL_LESION = "SMALL_LESION"
THICK_SLICE = "THICK_SLICE"
ADC_STORAGE_UNKNOWN = "ADC_STORAGE_UNKNOWN"
ACTIVE_CODES = (
    BOUNDARY_CONTACT,
    EXTENDS_OUTSIDE_BRAINMASK,
    MODEL_DISAGREEMENT,
    SMALL_LESION,
    THICK_SLICE,
    ADC_STORAGE_UNKNOWN,
)
DEFERRED_CODES = (
    "HIGH_UNCERTAINTY",
    "ATTRIBUTION_MISMATCH",
)  # no uncertainty / attribution model yet

DESIGN_SOURCE = "design_candidate_not_validated"
SMALL_LESION_ML = 1.0
THICK_Z_SPACING_MM = 5.0
THICK_SLICE_THICKNESS_MM = 5.0
PROVISIONAL_THRESHOLDS = {
    "boundary_contact_mm": CONTACT_MM,
    "boundary_contact_in_plane_voxels": 1,
    "small_lesion_ml_below": SMALL_LESION_ML,
    "thick_slice_effective_z_spacing_mm_above": THICK_Z_SPACING_MM,
    "thick_slice_thickness_mm_at_least": THICK_SLICE_THICKNESS_MM,
    "model_disagreement_low_dice_below": LOW_OVERLAP_DICE,
    "threshold_source": DESIGN_SOURCE,
}


def _w(code: str, detail: dict[str, Any], provisional: bool = True) -> dict[str, Any]:
    return {
        "code": code,
        "provisional_threshold": provisional,
        "threshold_source": DESIGN_SOURCE if provisional else None,
        "detail": detail,
    }


def slice_info(
    effective_z_mm: float, slice_thickness: float | None, spacing_between_slices: float | None
) -> dict[str, Any]:
    return {
        "effective_z_spacing_mm": round(float(effective_z_mm), 6),
        "slice_thickness_mm": slice_thickness,
        "spacing_between_slices_mm": spacing_between_slices,
    }


def case_warnings(slices: dict[str, Any], adc: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    z, th = slices["effective_z_spacing_mm"], slices["slice_thickness_mm"]
    if z > THICK_Z_SPACING_MM or (th is not None and th >= THICK_SLICE_THICKNESS_MM):
        out.append(_w(THICK_SLICE, dict(slices)))
    if adc["adc_status_unsettled"]:
        out.append(
            _w(
                ADC_STORAGE_UNKNOWN,
                {
                    k: adc[k]
                    for k in ("adc_storage_status", "adc_content_status", "adc_unit_status")
                },
                provisional=False,
            )
        )
    return out


def component_warnings(
    ident: dict[str, Any],
    boundary: dict[str, Any],
    agreement: dict[str, Any],
    case_level: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    if boundary["touches_boundary_1_inplane_voxel"] or boundary["touches_boundary_3mm"]:
        out.append(
            _w(
                BOUNDARY_CONTACT,
                {
                    "min_distance_to_brain_boundary_mm": boundary[
                        "min_distance_to_brain_boundary_mm"
                    ],
                    "touches_boundary_1_inplane_voxel": boundary[
                        "touches_boundary_1_inplane_voxel"
                    ],
                    "touches_boundary_3mm": boundary["touches_boundary_3mm"],
                },
            )
        )
    if boundary["voxels_outside_brain_mask"] > 0:
        out.append(
            _w(
                EXTENDS_OUTSIDE_BRAINMASK,
                {
                    "voxels_outside_brain_mask": boundary["voxels_outside_brain_mask"],
                    "outside_brain_mask_ratio": boundary["outside_brain_mask_ratio"],
                },
                provisional=False,
            )
        )
    st = agreement["status"]
    if agreement["available"]:
        if st in ("2d_only", "25d_only"):
            out.append(_w(MODEL_DISAGREEMENT, {"reason": st}, provisional=False))
        elif (
            st == "both"
            and agreement["dice_2d_25d"] is not None
            and agreement["dice_2d_25d"] < LOW_OVERLAP_DICE
        ):
            out.append(
                _w(
                    MODEL_DISAGREEMENT,
                    {"reason": "low_overlap", "dice_2d_25d": agreement["dice_2d_25d"]},
                )
            )
    if ident["volume_ml"] < SMALL_LESION_ML:
        out.append(_w(SMALL_LESION, {"volume_ml": ident["volume_ml"]}))
    out += [
        dict(w) for w in case_level
    ]  # THICK_SLICE / ADC_STORAGE_UNKNOWN apply to every component
    return out
