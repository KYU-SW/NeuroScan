"""Brain-mask QA: geometry, values, structure, lesion preservation, baseline comparison, status.

Masks are only measured. Nothing here resamples, dilates, fills or edits a mask. Numeric review
thresholds are deliberately not fixed yet (``ReviewThresholds`` defaults to None = not evaluated);
they are set after the pilot distribution and the visual QA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from neuroscan_ai.brain_extraction.interfaces import STATUS_FAIL, STATUS_PASS, STATUS_REVIEW

AFFINE_ATOL_MM = 1e-3
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


# ------------------------------------------------------------------------------------- geometry
def check_geometry(trace: nib.Nifti1Image, mask: nib.Nifti1Image) -> dict[str, Any]:
    """Compare the mask grid with the TRACE grid. Any mismatch is a failure (no resampling)."""
    t_sp = np.linalg.norm(trace.affine[:3, :3], axis=0)
    m_sp = np.linalg.norm(mask.affine[:3, :3], axis=0)
    diff = float(np.abs(trace.affine - mask.affine).max())
    res = {
        "shape_trace": [int(s) for s in trace.shape],
        "shape_mask": [int(s) for s in mask.shape],
        "shape_equal": tuple(trace.shape) == tuple(mask.shape),
        "max_affine_absdiff": diff,
        "affine_equal": diff <= AFFINE_ATOL_MM,
        "orientation_trace": "".join(nib.aff2axcodes(trace.affine)),
        "orientation_mask": "".join(nib.aff2axcodes(mask.affine)),
        "spacing_equal": bool(np.allclose(t_sp, m_sp, atol=AFFINE_ATOL_MM)),
        "qform_code": [int(trace.header["qform_code"]), int(mask.header["qform_code"])],
        "sform_code": [int(trace.header["sform_code"]), int(mask.header["sform_code"])],
    }
    res["orientation_equal"] = res["orientation_trace"] == res["orientation_mask"]
    res["codes_equal"] = res["qform_code"][0] == res["qform_code"][1] and (
        res["sform_code"][0] == res["sform_code"][1]
    )
    res["qform_sform_close"] = bool(
        np.allclose(trace.get_qform(), mask.get_qform(), atol=AFFINE_ATOL_MM)
        and np.allclose(trace.get_sform(), mask.get_sform(), atol=AFFINE_ATOL_MM)
    )
    # Grid mismatches are failures. Differing qform/sform *codes* (or a quaternion qform that only
    # approximates an oblique affine) do not move voxels; they are reported for review.
    keys = ("shape_equal", "affine_equal", "orientation_equal", "spacing_equal")
    res["failures"] = [k for k in keys if not res[k]]
    res["review"] = [k for k in ("codes_equal", "qform_sform_close") if not res[k]]
    return res


# --------------------------------------------------------------------------------------- values
def check_values(data: np.ndarray) -> dict[str, Any]:
    """Mask values must be exactly {0, 1}; non-finite, empty or all-one masks fail."""
    finite = np.isfinite(data)
    vals = np.unique(data[finite]) if finite.any() else np.array([])
    n_one = int((data == 1).sum())
    res = {
        "n_nonfinite": int((~finite).sum()),
        "distinct_values": [float(v) for v in vals[:10]],
        "n_distinct": int(vals.size),
        "binary": bool(finite.all() and set(vals.tolist()) <= {0.0, 1.0}),
        "voxels": n_one,
        "empty": n_one == 0,
        "all_one": n_one == data.size,
    }
    res["failures"] = [
        k
        for k, bad in (
            ("non_finite", res["n_nonfinite"] > 0),
            ("not_binary", not res["binary"]),
            ("empty_mask", res["empty"]),
            ("all_one_mask", res["all_one"]),
        )
        if bad
    ]
    return res


# ------------------------------------------------------------------------------------ structure
def components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    lab, n = ndimage.label(mask, structure=CONNECTIVITY_26)
    return lab, int(n)


def boundary_contact(mask: np.ndarray) -> float:
    """Share of mask voxels lying on the outer faces of the volume."""
    if not mask.any():
        return 0.0
    shell = np.zeros(mask.shape, dtype=bool)
    for ax in range(mask.ndim):
        sl = [slice(None)] * mask.ndim
        sl[ax] = 0
        shell[tuple(sl)] = True
        sl[ax] = -1
        shell[tuple(sl)] = True
    return float((mask & shell).sum() / mask.sum())


def bounding_box(mask: np.ndarray) -> dict[str, Any]:
    """Voxel bounding box and, per axis, the index range and number of slices that contain mask."""
    if not mask.any():
        return {"bbox_min": None, "bbox_max": None, "slices_with_mask": [0, 0, 0]}
    idx = np.argwhere(mask)
    lo, hi = idx.min(axis=0), idx.max(axis=0)
    per_axis = [
        int(np.any(mask, axis=tuple(a for a in range(3) if a != ax)).sum()) for ax in range(3)
    ]
    return {"bbox_min": lo.tolist(), "bbox_max": hi.tolist(), "slices_with_mask": per_axis}


def structure(mask: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    lab, n = components(mask)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([], dtype=int)
    vox_ml = float(np.prod(np.linalg.norm(affine[:3, :3], axis=0))) / 1000.0
    total = int(mask.sum())
    return {
        "voxels": total,
        "volume_ml": round(total * vox_ml, 6),
        "fraction_of_image": round(total / mask.size, 6),
        "n_components_26": n,
        "largest_component_fraction": round(float(sizes.max() / total), 6) if total else 0.0,
        "boundary_contact_fraction": round(boundary_contact(mask), 6),
        **bounding_box(mask),
    }


# --------------------------------------------------------------------------- lesion preservation
def gt_preservation(mask: np.ndarray, gt: np.ndarray | None) -> dict[str, Any]:
    """Acute GT voxels outside the brain mask (in total and per 26-connected GT component).

    A single GT voxel outside the mask sets ``gt_preservation_failure``. The mask is never
    enlarged or corrected to make it pass.
    """
    if gt is None:
        return {"gt_available": False, "gt_preservation_failure": None}
    g = gt > 0
    outside = g & ~mask
    lab, n = components(g)
    per = []
    for i in range(1, n + 1):
        comp = lab == i
        per.append(
            {"component": i, "voxels": int(comp.sum()), "outside": int((comp & ~mask).sum())}
        )
    n_gt = int(g.sum())
    return {
        "gt_available": True,
        "gt_voxels": n_gt,
        "gt_outside_voxels": int(outside.sum()),
        "gt_outside_fraction": round(float(outside.sum() / n_gt), 6) if n_gt else 0.0,
        "gt_components": n,
        "gt_components_with_outside": sum(1 for c in per if c["outside"]),
        "per_component": per,
        "gt_preservation_failure": bool(outside.any()),
    }


# ----------------------------------------------------------------------------- baseline compare
def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    s = int(a.sum()) + int(b.sum())
    return None if s == 0 else round(2.0 * float((a & b).sum()) / s, 6)


def compare_to_baseline(mask: np.ndarray, baseline: np.ndarray | None) -> dict[str, Any]:
    """Volume ratio, Dice and per-axis slice-coverage change against the v1 background mask."""
    if baseline is None or baseline.shape != mask.shape:
        return {"baseline_available": False}
    b_slices = bounding_box(baseline)["slices_with_mask"]
    m_slices = bounding_box(mask)["slices_with_mask"]
    lost = [
        int((np.any(baseline, axis=o) & ~np.any(mask, axis=o)).sum())
        for o in ((1, 2), (0, 2), (0, 1))
    ]
    return {
        "baseline_available": True,
        "dice_vs_baseline": dice(mask, baseline),
        "volume_ratio_vs_baseline": round(float(mask.sum() / baseline.sum()), 6)
        if baseline.any()
        else None,
        "slices_with_mask_baseline": b_slices,
        "slices_with_mask": m_slices,
        "slices_lost_vs_baseline": lost,
    }


# ---------------------------------------------------------------------------------------- status
@dataclass(frozen=True)
class ReviewThresholds:
    """Review triggers; None = not evaluated (thresholds are fixed after the pilot)."""

    min_fraction_of_image: float | None = None
    max_fraction_of_image: float | None = None
    max_components: int | None = None
    min_largest_component_fraction: float | None = None
    max_boundary_contact_fraction: float | None = None
    max_slices_lost_vs_baseline: int | None = None
    min_volume_ratio_vs_baseline: float | None = None
    max_volume_ratio_vs_baseline: float | None = None


def classify(qa: dict[str, Any], thresholds: ReviewThresholds | None = None) -> dict[str, Any]:
    """pass / review_required / fail. Failures are fixed rules; review rules need thresholds."""
    th = thresholds or ReviewThresholds()
    fail = []
    if qa.get("execution_error"):
        fail.append("execution_failure")
    elif "values" not in qa:
        fail.append("no_mask")
    fail += [f"geometry:{k}" for k in qa.get("geometry", {}).get("failures", [])]
    fail += [f"values:{k}" for k in qa.get("values", {}).get("failures", [])]
    if qa.get("lesion", {}).get("gt_preservation_failure"):
        fail.append("gt_preservation_failure")
    review, not_evaluated = [], []
    if not fail and qa.get("geometry", {}).get("review"):
        review.append("header_codes_differ")
    s, b = qa.get("structure", {}), qa.get("baseline", {})

    def rule(name, value, limit, bad):
        if limit is None:
            not_evaluated.append(name)
        elif value is not None and bad(value, limit):
            review.append(name)

    if s:
        rule("mask_fraction_low", s.get("fraction_of_image"), th.min_fraction_of_image,
             lambda v, t: v < t)  # fmt: skip
        rule("mask_fraction_high", s.get("fraction_of_image"), th.max_fraction_of_image,
             lambda v, t: v > t)  # fmt: skip
        rule("fragmented", s.get("n_components_26"), th.max_components, lambda v, t: v > t)
        rule("largest_component_low", s.get("largest_component_fraction"),
             th.min_largest_component_fraction, lambda v, t: v < t)  # fmt: skip
        rule("boundary_contact_high", s.get("boundary_contact_fraction"),
             th.max_boundary_contact_fraction, lambda v, t: v > t)  # fmt: skip
    if b.get("baseline_available"):
        rule("extent_drop_vs_baseline", max(b.get("slices_lost_vs_baseline", [0])),
             th.max_slices_lost_vs_baseline, lambda v, t: v > t)  # fmt: skip
        rule("volume_low_vs_baseline", b.get("volume_ratio_vs_baseline"),
             th.min_volume_ratio_vs_baseline, lambda v, t: v < t)  # fmt: skip
        rule("volume_high_vs_baseline", b.get("volume_ratio_vs_baseline"),
             th.max_volume_ratio_vs_baseline, lambda v, t: v > t)  # fmt: skip
    status = STATUS_FAIL if fail else STATUS_REVIEW if review else STATUS_PASS
    return {
        "status": status,
        "fail_reasons": fail,
        "review_reasons": review,
        "review_rules_not_evaluated": not_evaluated,
        "thresholds": asdict(th),
    }


def run_qa(
    trace: nib.Nifti1Image,
    mask_img: nib.Nifti1Image | None,
    gt: np.ndarray | None,
    baseline: np.ndarray | None = None,
    thresholds: ReviewThresholds | None = None,
    execution_error: str | None = None,
) -> dict[str, Any]:
    qa: dict[str, Any] = {"execution_error": execution_error}
    if mask_img is not None:
        qa["geometry"] = check_geometry(trace, mask_img)
        data = np.asarray(mask_img.dataobj, dtype=np.float64)
        qa["values"] = check_values(data)
        usable = not qa["geometry"]["failures"] and qa["values"]["binary"]
        if usable:
            m = data == 1
            qa["structure"] = structure(m, trace.affine)
            qa["lesion"] = gt_preservation(m, gt)
            qa["baseline"] = compare_to_baseline(m, baseline)
    qa["classification"] = classify(qa, thresholds)
    return qa
