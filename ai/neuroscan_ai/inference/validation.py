"""Input validation for one DWI + ADC pair (original files are only read, never modified).

Errors stop the run before preprocessing; warnings are recorded and the run continues. Header
text fields are checked for presence only — their content is never logged or written anywhere.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neuroscan_ai.inference.errors import Code, InferenceError, Issue
from neuroscan_ai.io import nifti
from neuroscan_ai.training.brain_gating import BrainMaskError, compute_brain_mask

BRAIN_POLICY = "skull_stripped_background"  # the policy the Phase 4 model was trained with
HEADER_TEXT_FIELDS = ("descrip", "aux_file", "db_name", "intent_name")
NIFTI_HEADER_SIZES = (348, 540)  # NIfTI-1, NIfTI-2


@dataclass
class LoadedInput:
    role: str  # "dwi" | "adc"
    affine: np.ndarray
    data: np.ndarray  # float32, 3-D, original grid (same read as preprocessing.pipeline)
    sha256: str
    size_bytes: int
    format: str
    basename: str  # kept in memory; written only with --record-input-names
    header_text_fields: list[str]
    zooms: tuple[float, ...]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.data.shape)


@dataclass
class ValidationReport:
    dwi: LoadedInput | None = None
    adc: LoadedInput | None = None
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def error(self, code: Code, message: str, detail: str | None = None) -> None:
        self.errors.append(Issue(code, message, detail))

    def warn(self, code: Code, message: str, detail: str | None = None) -> None:
        self.warnings.append(Issue(code, message, detail))


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def obliquity_deg(affine: np.ndarray) -> float:
    """Angle (degrees) between the voxel axes and the nearest axis-aligned (RAS-permuted) frame."""
    m = np.asarray(affine, float)[:3, :3]
    rot = m / np.linalg.norm(m, axis=0)
    ornt = nib.orientations.io_orientation(affine)
    nearest = np.zeros((3, 3))
    for i, (world_axis, flip) in enumerate(ornt):
        nearest[int(world_axis), i] = flip
    cos = np.clip((np.trace(nearest.T @ rot) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.degrees(math.acos(cos)))


def _format_of(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return "nifti+gzip"
    if name.endswith(".nii"):
        return "nifti"
    return None


def _header_size_ok(path: Path, fmt: str) -> bool:
    opener = gzip.open if fmt == "nifti+gzip" else open
    try:
        with opener(path, "rb") as f:
            raw = f.read(4)
    except (OSError, EOFError):
        return False
    if len(raw) < 4:
        return False
    return any(int.from_bytes(raw, order) in NIFTI_HEADER_SIZES for order in ("little", "big"))


def _load_one(path: Path, role: str, report: ValidationReport) -> LoadedInput | None:
    label = role.upper()
    if not path.is_file():
        report.error(Code.INPUT_NOT_FOUND, f"{label} file was not found.")
        return None
    fmt = _format_of(path)
    if fmt is None:
        report.error(Code.INPUT_FORMAT, f"{label} must be a .nii or .nii.gz file.")
        return None
    if not _header_size_ok(path, fmt):
        report.error(
            Code.INPUT_FORMAT, f"{label} is not a readable NIfTI file.", "bad NIfTI header size"
        )
        return None
    try:
        img = nib.load(str(path))
        if not isinstance(img, nib.Nifti1Image | nib.Nifti2Image):
            report.error(Code.INPUT_FORMAT, f"{label} is not a NIfTI image.")
            return None
        shape = img.shape
        if len(shape) == 4 and shape[3] == 1:
            data = img.get_fdata(dtype=np.float32)[..., 0]
        elif len(shape) == 3:
            data = img.get_fdata(dtype=np.float32)  # identical read to preprocessing.pipeline
        else:
            report.error(Code.INPUT_NOT_3D, f"{label} must be a 3-D image.", f"shape {shape}")
            return None
    except Exception as exc:  # truncated / corrupt file; message may contain the path: drop it
        report.error(
            Code.INPUT_CORRUPT,
            f"{label} could not be read (damaged or invalid NIfTI).",
            type(exc).__name__,
        )
        return None
    hdr = img.header
    text_fields = []
    for name in HEADER_TEXT_FIELDS:
        try:
            value = bytes(np.asarray(hdr[name]).tobytes()).strip(b"\x00 ")
        except (KeyError, ValueError):
            continue
        if value:
            text_fields.append(name)
    if len(getattr(hdr, "extensions", [])):
        text_fields.append("extensions")
    return LoadedInput(
        role=role,
        affine=np.asarray(img.affine, float),
        data=np.ascontiguousarray(data),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        format=fmt,
        basename=path.name,
        header_text_fields=text_fields,
        zooms=tuple(float(z) for z in hdr.get_zooms()[:3]),
    )


def _in_any(value: float, ranges) -> bool:
    return any(lo <= value <= hi for lo, hi in ranges)


def validate_inputs(dwi_path, adc_path, cfg: dict[str, Any]) -> ValidationReport:
    """Run every input check; never raises for bad input (see ``ValidationReport.errors``)."""
    report = ValidationReport()
    report.dwi = _load_one(Path(dwi_path), "dwi", report)
    report.adc = _load_one(Path(adc_path), "adc", report)
    if report.errors:
        return report
    dwi, adc = report.dwi, report.adc

    for item in (dwi, adc):
        label = item.role.upper()
        if not np.isfinite(item.data).all():
            report.error(Code.NONFINITE, f"{label} contains NaN or infinite values.")
        elif float(item.data.min()) == float(item.data.max()):
            report.error(Code.EMPTY_IMAGE, f"{label} is empty (constant image).")
        if item.header_text_fields:
            report.warn(
                Code.W_HEADER_TEXT_PRESENT,
                f"{label} header contains text fields; they are not copied to any output.",
                "non-empty fields: " + ", ".join(item.header_text_fields),
            )
    if report.errors:
        return report

    # grid compatibility (no resampling of ADC onto DWI in 5A)
    if dwi.shape != adc.shape:
        report.error(Code.GRID_MISMATCH, "DWI and ADC have different image grids.", "shape differs")
    elif not np.allclose(dwi.affine, adc.affine, atol=nifti.AFFINE_ATOL_MM):
        detail = (
            "orientation differs"
            if nifti.orientation(dwi.affine) != nifti.orientation(adc.affine)
            else "affine differs"
        )
        report.error(Code.GRID_MISMATCH, "DWI and ADC have different image grids.", detail)
    if report.errors:
        return report

    v = cfg["validation"]
    spacing = nifti.spacing_from_affine(dwi.affine)
    orient = nifti.orientation(dwi.affine)
    if (
        not all(math.isfinite(s) and s > 0 for s in spacing)
        or abs(np.linalg.det(dwi.affine)) < 1e-9
    ):
        report.error(Code.SPACING_INVALID, "Voxel spacing is invalid.")
        return report
    if any(
        abs(z - s) > v["zooms_affine_tolerance_mm"] for z, s in zip(dwi.zooms, spacing, strict=True)
    ):
        report.error(
            Code.SPACING_INVALID,
            "Header voxel size and affine disagree.",
            "pixdim vs affine column norms",
        )
    # slice axis = array axis pointing superior/inferior
    slice_axis = next(i for i, c in enumerate(orient) if c in "SI")
    inplane = [s for i, s in enumerate(spacing) if i != slice_axis]
    slice_sp = spacing[slice_axis]
    if (
        min(spacing) < v["min_spacing_mm"]
        or max(inplane) > v["max_inplane_spacing_mm"]
        or slice_sp > v["max_slice_spacing_mm"]
    ):
        report.error(Code.SPACING_INVALID, "Voxel spacing is outside the supported range.")
    else:
        tol = v["spacing_tolerance_mm"]
        lo_xy, hi_xy = v["train_inplane_spacing_mm"]
        lo_z, hi_z = v["train_slice_spacing_mm"]
        if not all(lo_xy - tol <= s <= hi_xy + tol for s in inplane) or not (
            lo_z - tol <= slice_sp <= hi_z + tol
        ):
            report.warn(
                Code.W_SPACING_OUT_OF_TRAIN_RANGE,
                "Voxel spacing is outside the range seen in training; "
                "results may be less reliable.",
            )
    shape = dwi.shape
    inplane_n = [n for i, n in enumerate(shape) if i != slice_axis]
    if min(inplane_n) < v["min_inplane_voxels"] or shape[slice_axis] < v["min_slices"]:
        report.error(Code.SHAPE_TOO_SMALL, "Image is too small for analysis.", f"shape {shape}")
    elif shape[slice_axis] < v["warn_slices_below"]:
        report.warn(Code.W_FEW_SLICES, "Image has very few slices; results may be less reliable.")
    if orient != "LAS":
        report.warn(
            Code.W_ORIENTATION_UNUSUAL,
            "Orientation differs from the training data; it is converted losslessly.",
            f"orientation {orient}",
        )
    obl = obliquity_deg(dwi.affine)
    if obl > v["oblique_warn_deg"]:
        report.warn(Code.W_OBLIQUE, "Image axes are oblique.", f"{obl:.1f} deg")
    if report.errors:
        return report

    # skull-stripped brain (the model's brain-mask policy) on the original DWI grid
    try:
        brain = compute_brain_mask(BRAIN_POLICY, dwi.data)
    except BrainMaskError as exc:
        report.error(
            Code.BRAIN_MASK,
            "Only skull-stripped images are supported (brain region could not be identified).",
            str(exc),
        )
        return report
    frac = float(brain.mean())
    lo_b, hi_b = v["train_brain_fraction"]
    if not lo_b <= frac <= hi_b:
        report.warn(
            Code.W_BRAIN_FRACTION_UNUSUAL,
            "Brain region size differs from the training data.",
            f"brain fraction {frac:.3f}",
        )
    dwi_b, adc_b = dwi.data[brain], adc.data[brain]
    if float(dwi_b.std()) == 0.0 or float(adc_b.std()) == 0.0:
        report.error(Code.EMPTY_IMAGE, "Image has no contrast inside the brain region.")
        return report
    dwi_med, adc_med = float(np.median(dwi_b)), float(np.median(adc_b))
    dwi_range, adc_ranges = [v["dwi_brain_median_range"]], v["adc_brain_median_ranges"]
    normal_fit = _in_any(dwi_med, dwi_range) and _in_any(adc_med, adc_ranges)
    swapped_fit = _in_any(adc_med, dwi_range) and _in_any(dwi_med, adc_ranges)
    if swapped_fit and not normal_fit:
        report.warn(
            Code.W_POSSIBLE_SWAP,
            "DWI and ADC may have been given in the wrong order (inputs were not changed).",
            "intensity ranges match the swapped assignment",
        )
    elif not _in_any(adc_med, adc_ranges):
        report.warn(
            Code.W_ADC_UNIT_UNUSUAL,
            "ADC value range differs from the training data.",
            f"ADC brain median {adc_med:.6g}",
        )
    neg = float((adc_b < 0).mean())
    if neg > v["adc_negative_fraction_warn"]:
        report.warn(Code.W_ADC_NEGATIVE, "ADC has many negative values in the brain region.")
    report.stats = {
        "shape": list(shape),
        "spacing_mm": [round(s, 6) for s in spacing],
        "orientation": orient,
        "obliquity_deg": round(obl, 3),
        "slice_axis": slice_axis,
        "brain_fraction": round(frac, 4),
        "dwi_brain_median": round(dwi_med, 6),
        "adc_brain_median": round(adc_med, 9),
        "adc_negative_fraction_in_brain": round(neg, 5),
    }
    return report


def raise_first_error(report: ValidationReport) -> None:
    if report.errors:
        first = report.errors[0]
        raise InferenceError(first.code, first.message, first.detail)
