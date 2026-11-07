"""SOOP (OpenNeuro ds004889) format adapter — mechanical format conversion only.

Scope (docs: SOOP compatibility pilot, K3):

* BIDS discovery of TRACE / ADC / acute lesion mask (``desc-lesionAcute`` is the primary GT).
* Removal of a trailing singleton axis ``(X, Y, Z, 1) -> (X, Y, Z)``; nothing else.
* Geometry, binary-GT and header checks; ADC storage-group *labelling* from the sidecar.
* Derived files are written so that the analysis value of every voxel is bit-identical to the
  source: the on-disk dtype, the stored (raw) integers and the NIfTI ``scl_slope`` /
  ``scl_inter`` are kept, and the file is re-read and compared before it is accepted.

This module never resamples, skull-strips, normalises, converts ADC units, changes the
background code or binarises / edits a mask. It does not import any preprocessing, brain-mask
or model code. Output folders are keyed by alias only; original subject IDs stay in the local
alias table (``_internal/aliases.json``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

DATASET = "soop"
FORMAT_VERSION = "soop_format_v1"
AFFINE_ATOL_MM = 1e-3  # same tolerance as neuroscan_ai.io.nifti.same_grid

REQUIRED = ("trace_nii", "trace_json", "adc_nii", "adc_json", "mask_acute")
OPTIONAL = ("mask_lesion", "mask_chronic")

# Pilot-observed ADC storage signatures (5 cases). Labels describe how values are stored, they
# are NOT a physical-unit determination; any other combination is "unknown".
ADC_GROUP_RULE_VERSION = "pilot-observed-v2"
ADC_GROUP_NOTE = (
    "P/S/N are storage patterns observed in the 5-case SOOP pilot; a case gets a label only if "
    "every signature field matches exactly. This is NOT a physical ADC unit determination."
)
ADC_EVIDENCE_FIELDS = (
    "SeriesDescription",
    "ProtocolName",
    "PhilipsRescaleSlope",
    "PhilipsRescaleIntercept",
    "PhilipsScaleSlope",
    "PhilipsRWVSlope",
    "PhilipsRWVIntercept",
    "SoftwareVersions",
    "ConversionSoftwareVersion",
    "UsePhilipsFloatNotDisplayScaling",
)
# Exact pilot signatures. PhilipsScaleSlope differs per acquisition even inside one pilot group,
# so it is required to be present, numeric and > 0 rather than equal to a constant.
_PILOT_SW_NEW, _PILOT_SW_OLD = "5.6.1\\5.6.1.0", "3.2.3"
_ADC_SIGNATURES: dict[str, dict[str, Any]] = {
    "P": {
        "SeriesDescription": "ddADC",
        "ProtocolName": "ddADC",
        "PhilipsRescaleSlope": 9.66995e-04,
        "PhilipsRescaleIntercept": 0.0,
        "PhilipsRWVSlope": 9.66995e-07,
        "PhilipsRWVIntercept": 0.0,
        "SoftwareVersions": _PILOT_SW_NEW,
        "UsePhilipsFloatNotDisplayScaling": 1,
    },
    "S": {
        "SeriesDescription": "dADC",
        "ProtocolName": "dADC",
        "PhilipsRescaleSlope": 0.966995,
        "PhilipsRescaleIntercept": 0.0,
        "PhilipsRWVSlope": 0.966995,
        "PhilipsRWVIntercept": 0.0,
        "SoftwareVersions": _PILOT_SW_OLD,
        "UsePhilipsFloatNotDisplayScaling": 1,
    },
    "N": {
        "SeriesDescription": "dADC",
        "ProtocolName": "dADC",
        "PhilipsRescaleSlope": 1.27773e-06,
        "PhilipsRescaleIntercept": -0.00193634,
        "PhilipsRWVSlope": 1.27773e-06,
        "PhilipsRWVIntercept": -0.00193634,
        "SoftwareVersions": _PILOT_SW_NEW,
        "UsePhilipsFloatNotDisplayScaling": 1,
    },
}

# ADC content status (sidecar only, conservative). Not a data change: excluded content is never
# replaced by another image.
ADC_CONTENT_RULE_VERSION = "sidecar-content-v1"
ADC_CONTENT_STATUSES = ("confirmed_adc", "probable_adc", "suspicious_non_adc", "unknown")
# v2 data-pool policy by content status (conservative). Only confirmed_adc is a default
# candidate; probable_adc needs a later image-value QA before it may be promoted; the other two
# are excluded automatically. Pilot cases are excluded from every split and performance count.
ADC_CONTENT_ELIGIBLE = ("confirmed_adc",)
POOL_CANDIDATE, POOL_REVIEW, POOL_EXCLUDED, POOL_PILOT = (
    "eligible_candidate",
    "review_required",
    "auto_excluded",
    "pilot_permanently_excluded",
)
ADC_CONTENT_POOL_POLICY = {
    "confirmed_adc": POOL_CANDIDATE,
    "probable_adc": POOL_REVIEW,
    "suspicious_non_adc": POOL_EXCLUDED,
    "unknown": POOL_EXCLUDED,
}

# header fields that define the spatial frame; compared exactly between source and derived
_FRAME_FIELDS = (
    "qform_code",
    "sform_code",
    "quatern_b",
    "quatern_c",
    "quatern_d",
    "qoffset_x",
    "qoffset_y",
    "qoffset_z",
    "srow_x",
    "srow_y",
    "srow_z",
    "xyzt_units",
)


class SoopFormatError(ValueError):
    """Input rejected by the format adapter (message never contains a subject ID)."""


# ----------------------------------------------------------------------------------- discovery
def subject_paths(subject_id: str) -> dict[str, str]:
    """BIDS-relative paths (posix) of every file the adapter may use for one subject."""
    s = subject_id
    mask = f"derivatives/lesion_masks/{s}/dwi/{s}_space-TRACE_desc-"
    return {
        "trace_nii": f"{s}/dwi/{s}_rec-TRACE_dwi.nii.gz",
        "trace_json": f"{s}/dwi/{s}_rec-TRACE_dwi.json",
        "adc_nii": f"{s}/dwi/{s}_rec-ADC_dwi.nii.gz",
        "adc_json": f"{s}/dwi/{s}_rec-ADC_dwi.json",
        "mask_acute": f"{mask}lesionAcute_mask.nii.gz",
        "mask_lesion": f"{mask}lesion_mask.nii.gz",
        "mask_chronic": f"{mask}lesionChronic_mask.nii.gz",
    }


DATASET_LEVEL_OPTIONAL = ("dwi.bval", "dwi.bvec")


@dataclass(frozen=True)
class SoopCase:
    subject_id: str
    files: dict[str, Path]  # required + present optional files
    dataset_files: dict[str, Path] = field(default_factory=dict)  # dwi.bval / dwi.bvec if present


def eligibility(available: set[str], subject_id: str) -> tuple[bool, list[str]]:
    """Eligible when every required file exists. A missing acute mask is *not* a negative case;
    it is an exclusion with the reason recorded."""
    paths = subject_paths(subject_id)
    missing = [k for k in REQUIRED if paths[k] not in available]
    return not missing, missing


def discover_soop(root: Path) -> tuple[list[SoopCase], dict[str, list[str]]]:
    """Return eligible cases (sorted by subject folder name) and {subject_id: missing kinds}."""
    root = Path(root)
    dataset_files = {k: root / k for k in DATASET_LEVEL_OPTIONAL if (root / k).is_file()}
    cases, excluded = [], {}
    for sub in sorted(p for p in root.glob("sub-*") if p.is_dir()):
        sid = sub.name
        paths = subject_paths(sid)
        present = {k for k, rel in paths.items() if (root / rel).is_file()}
        ok, missing = eligibility({paths[k] for k in present}, sid)
        if not ok:
            excluded[sid] = missing
            continue
        files = {k: root / paths[k] for k in (*REQUIRED, *OPTIONAL) if k in present}
        cases.append(SoopCase(sid, files, dataset_files))
    return cases, excluded


# --------------------------------------------------------------------------- singleton handling
def _header_dims(img: nib.Nifti1Image) -> tuple[int, ...]:
    dim = img.header["dim"]
    return tuple(int(d) for d in dim[1 : int(dim[0]) + 1])


def singleton_plan(img: nib.Nifti1Image) -> str:
    """'keep' for 3D, 'squeeze_last' for (X, Y, Z, 1); raises for anything else."""
    shape = tuple(int(s) for s in img.shape)
    arr_shape = tuple(int(s) for s in img.dataobj.shape)
    if _header_dims(img) != shape or arr_shape != shape:
        raise SoopFormatError(f"shape {shape} does not match the header dims")
    ndim = int(img.header["dim"][0])
    if ndim == 3 and len(shape) == 3:
        if 1 in shape:
            raise SoopFormatError("a spatial axis has size 1")
        return "keep"
    if ndim == 4 and len(shape) == 4:
        if shape[3] != 1:
            raise SoopFormatError(f"last axis has size {shape[3]} (only 1 may be removed)")
        if 1 in shape[:3]:
            raise SoopFormatError("a spatial axis has size 1 (not removable)")
        return "squeeze_last"
    raise SoopFormatError(f"unsupported dimensionality {ndim}")


def _frame(h: nib.Nifti1Header) -> dict[str, Any]:
    out = {k: np.asarray(h[k]).tolist() for k in _FRAME_FIELDS}
    out["pixdim_0_3"] = np.asarray(h["pixdim"][:4]).tolist()
    return out


def _slope_inter(img: nib.Nifti1Image) -> tuple[float, float]:
    """Scaling actually applied by NiBabel when reading (header copy may show NaN)."""
    return float(img.dataobj.slope), float(img.dataobj.inter)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_format_copy(src: Path, dest: Path) -> dict[str, Any]:
    """Write ``dest`` as the 3D version of ``src`` and verify value/frame preservation.

    3D input: byte copy (identical file). (X, Y, Z, 1): new header from a copy of the source
    header, raw stored values of the single volume, original dtype and original scl_slope /
    scl_inter (method "raw+original-scaling"). The written file is re-read; the stored values,
    the scaled analysis values (float64 and float32), dtype, scaling, affine and frame fields must
    be identical, otherwise the file is removed and the case is refused.
    """
    img = nib.load(str(src))
    plan = singleton_plan(img)
    if not dest.name.endswith(".nii.gz"):
        raise SoopFormatError("derived file must be .nii.gz")
    tmp = dest.with_name(dest.name[: -len(".nii.gz")] + ".part.nii.gz")  # NiBabel needs the suffix
    if dest.exists() or tmp.exists():
        raise SoopFormatError("derived file already exists")
    if plan == "keep":
        shutil.copyfile(src, tmp)
    else:
        raw = np.asarray(img.dataobj.get_unscaled())[..., 0]
        hdr = img.header.copy()
        hdr.set_data_shape(raw.shape)
        out = nib.Nifti1Image(raw, None, hdr)
        slope, inter = _slope_inter(img)
        out.header.set_slope_inter(slope, inter)
        nib.save(out, str(tmp))
    try:
        check = verify_preserved(img, nib.load(str(tmp)), plan)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dest)
    check.update(plan=plan, method="byte-copy" if plan == "keep" else "raw+original-scaling")
    return check


def verify_preserved(src: nib.Nifti1Image, out: nib.Nifti1Image, plan: str) -> dict[str, Any]:
    sel = (lambda a: a[..., 0]) if plan == "squeeze_last" else (lambda a: a)
    res = {
        "shape_in": [int(s) for s in src.shape],
        "shape_out": [int(s) for s in out.shape],
        "spatial_shape_equal": tuple(src.shape[:3]) == tuple(out.shape),
        "dtype_equal": src.get_data_dtype() == out.get_data_dtype(),
        "scaling_equal": _slope_inter(src) == _slope_inter(out),
        "raw_equal": bool(
            np.array_equal(sel(np.asarray(src.dataobj.get_unscaled())), out.dataobj.get_unscaled())
        ),
        "scaled_f64_equal": bool(
            np.array_equal(
                sel(np.asarray(src.dataobj, dtype=np.float64)),
                np.asarray(out.dataobj, dtype=np.float64),
            )
        ),
        "scaled_f32_equal": bool(
            np.array_equal(sel(src.get_fdata(dtype=np.float32)), out.get_fdata(dtype=np.float32))
        ),
        "affine_equal": bool(np.array_equal(src.affine, out.affine)),
        "frame_equal": _frame(src.header) == _frame(out.header),
        "orientation_equal": nib.aff2axcodes(src.affine) == nib.aff2axcodes(out.affine),
        "dtype": str(out.get_data_dtype()),
        "scl_slope": _slope_inter(out)[0],
        "scl_inter": _slope_inter(out)[1],
    }
    if src.ndim == 4:
        res["dropped_pixdim4"] = float(src.header["pixdim"][4])
    failed = [k for k, v in res.items() if k.endswith("_equal") and v is not True]
    if failed:
        raise SoopFormatError(f"value or frame not preserved: {failed}")
    return res


# ----------------------------------------------------------------------------- geometry & GT
def geometry(img: nib.Nifti1Image) -> dict[str, Any]:
    aff = img.affine
    spacing = [float(v) for v in np.linalg.norm(aff[:3, :3], axis=0)]
    return {
        "shape": [int(s) for s in img.shape[:3]],
        "spacing_mm": [round(s, 6) for s in spacing],
        "orientation": "".join(nib.aff2axcodes(aff)),
        "obliquity_deg": round(float(np.degrees(nib.affines.obliquity(aff)).max()), 4),
        "affine": np.round(aff, 6).tolist(),
        "qform_code": int(img.header["qform_code"]),
        "sform_code": int(img.header["sform_code"]),
    }


def check_same_grid(ref: nib.Nifti1Image, other: nib.Nifti1Image, name: str) -> dict[str, Any]:
    if tuple(ref.shape[:3]) != tuple(other.shape[:3]):
        raise SoopFormatError(f"{name}: shape differs from TRACE")
    diff = float(np.abs(ref.affine - other.affine).max())
    if diff > AFFINE_ATOL_MM:
        raise SoopFormatError(f"{name}: affine differs from TRACE ({diff:.4g})")
    sp_r = np.linalg.norm(ref.affine[:3, :3], axis=0)
    sp_o = np.linalg.norm(other.affine[:3, :3], axis=0)
    if not np.allclose(sp_r, sp_o, atol=AFFINE_ATOL_MM):
        raise SoopFormatError(f"{name}: spacing differs from TRACE")
    if nib.aff2axcodes(ref.affine) != nib.aff2axcodes(other.affine):
        raise SoopFormatError(f"{name}: orientation differs from TRACE")
    return {"max_affine_absdiff": diff, "same_grid": True}


def check_binary_mask(img: nib.Nifti1Image, name: str) -> dict[str, Any]:
    """Values must already be exactly {0, 1}; the mask is never binarised or edited."""
    data = np.asarray(img.dataobj, dtype=np.float64)
    vals = np.unique(data)
    if not set(vals.tolist()) <= {0.0, 1.0}:
        raise SoopFormatError(f"{name}: not binary ({vals.size} distinct values)")
    vox = float(np.prod(np.linalg.norm(img.affine[:3, :3], axis=0)))
    n = int((data == 1).sum())
    return {"values": [float(v) for v in vals], "voxels": n, "volume_ml": round(n * vox / 1000, 6)}


# --------------------------------------------------------------------------- ADC storage group
def _is_number(a: Any) -> bool:
    return isinstance(a, int | float) and not isinstance(a, bool) and bool(np.isfinite(a))


def _matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        return isinstance(value, str) and value == expected
    if not _is_number(value):
        return False
    return bool(np.isclose(float(value), float(expected), rtol=1e-6, atol=1e-12))


def adc_storage_group(sidecar: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """('P'|'S'|'N'|'unknown', evidence). Pilot-observed storage label, never a unit.

    Every signature field must be present and equal (strings exactly, numbers to 1e-6 relative)
    and PhilipsScaleSlope must be a positive number; otherwise 'unknown' (no guessing).
    """
    ev = {k: sidecar.get(k) for k in ADC_EVIDENCE_FIELDS}
    ss = sidecar.get("PhilipsScaleSlope")
    if _is_number(ss) and ss > 0:
        for group, sig in _ADC_SIGNATURES.items():
            if all(_matches(sidecar.get(k), v) for k, v in sig.items()):
                return group, ev
    return "unknown", ev


def _adc_like_name(name: str) -> bool:
    n = name.lower()
    return "adc" in n or "apparent diffusion coefficient" in n


def _non_adc_name(name: str) -> bool:
    n = name.lower()
    return bool(re.search(r"(?<![0-9])b ?0(?![0-9])|b ?\d{3,4}|dwi|diffusion", n))


def adc_content_status(sidecar: dict[str, Any]) -> tuple[str, str]:
    """Conservative content check of the file labelled ``rec-ADC`` from its sidecar only.

    confirmed_adc: ImageType contains ADC and SeriesDescription and ProtocolName are both
        ADC-named, with no B0 / b-value / DWI wording.
    probable_adc: ImageType contains ADC, neither name is ADC-named nor names a B0 image
        (e.g. a derived series named after its DWI acquisition).
    suspicious_non_adc: ImageType has no ADC token and the names look like a B0 / DWI series.
    unknown: missing fields or contradictory evidence.
    """
    names = [sidecar.get("SeriesDescription"), sidecar.get("ProtocolName")]
    image_type = sidecar.get("ImageType")
    if not all(isinstance(n, str) and n for n in names) or not isinstance(image_type, list):
        return "unknown", "missing SeriesDescription / ProtocolName / ImageType"
    has_adc = "ADC" in {str(t).upper() for t in image_type}
    adc_named = [_adc_like_name(n) for n in names]
    b0_named = any(re.search(r"(?<![0-9])b ?0(?![0-9])", n.lower()) for n in names)
    non_adc = [_non_adc_name(n) and not a for n, a in zip(names, adc_named, strict=True)]
    if has_adc and all(adc_named) and not b0_named:
        return "confirmed_adc", "ImageType ADC + ADC series names"
    if has_adc and not any(adc_named) and not b0_named:
        return "probable_adc", "ImageType ADC, series named after the DWI acquisition"
    if not has_adc and not any(adc_named) and all(non_adc):
        return "suspicious_non_adc", "no ADC in ImageType, B0 / DWI series names"
    return "unknown", "contradictory or insufficient sidecar evidence"


def background_code(img: nib.Nifti1Image) -> float | None:
    """Value shared by all 8 volume corners (scaled), else None. Reported, never altered."""
    data = np.asarray(img.dataobj, dtype=np.float64)
    corners = data[np.ix_(*[[0, s - 1] for s in data.shape[:3]])].ravel()
    vals = np.unique(corners)
    return float(vals[0]) if vals.size == 1 else None


# ---------------------------------------------------------------------------------- build case
def _sidecar(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def v2_pool_status(content_status: str, *, pilot: bool) -> str:
    """Pool label only (no split is created here). Pilot cases are always excluded."""
    if pilot:
        return POOL_PILOT
    return ADC_CONTENT_POOL_POLICY.get(content_status, POOL_EXCLUDED)


def build_case(
    case: SoopCase, alias: str, out_root: Path, *, pilot: bool = False
) -> dict[str, Any]:
    """Write ``out_root/<alias>/{dwi,adc,label}.nii.gz, metadata.json, audit.json``.

    Writes into ``<alias>.part`` and renames at the end; refuses an existing alias folder.
    Metadata and audit hold the alias and file hashes only (no subject ID, no source path).
    """
    out_root = Path(out_root)
    final, work = out_root / alias, out_root / f"{alias}.part"
    if final.exists() or work.exists():
        raise SoopFormatError(f"{alias}: output folder already exists")
    src_hash_before = {k: sha256_file(p) for k, p in case.files.items()}
    work.mkdir(parents=True)
    try:
        trace = nib.load(str(case.files["trace_nii"]))
        adc = nib.load(str(case.files["adc_nii"]))
        acute = nib.load(str(case.files["mask_acute"]))
        conv = {
            "dwi": write_format_copy(case.files["trace_nii"], work / "dwi.nii.gz"),
            "adc": write_format_copy(case.files["adc_nii"], work / "adc.nii.gz"),
            "label": write_format_copy(case.files["mask_acute"], work / "label.nii.gz"),
        }
        d_trace = nib.load(str(work / "dwi.nii.gz"))
        d_adc = nib.load(str(work / "adc.nii.gz"))
        d_label = nib.load(str(work / "label.nii.gz"))
        grid = {
            "adc": check_same_grid(d_trace, d_adc, "ADC"),
            "label": check_same_grid(d_trace, d_label, "acute mask"),
        }
        masks = {"acute": check_binary_mask(acute, "acute mask")}
        for k, name in (("mask_lesion", "lesion"), ("mask_chronic", "chronic")):
            if k in case.files:
                m = nib.load(str(case.files[k]))
                grid[name] = check_same_grid(d_trace, m, f"{name} mask")
                masks[name] = check_binary_mask(m, f"{name} mask")
        adc_json, trace_json = _sidecar(case.files["adc_json"]), _sidecar(case.files["trace_json"])
        group, evidence = adc_storage_group(adc_json)
        content, content_reason = adc_content_status(adc_json)
        metadata = {
            "alias": alias,
            "dataset": DATASET,
            "format_version": FORMAT_VERSION,
            "primary_ground_truth": "desc-lesionAcute",
            "geometry": geometry(d_trace),
            "source_ndim": {"dwi": trace.ndim, "adc": adc.ndim, "label": acute.ndim},
            "masks": masks,
            "optional_masks_present": sorted(m for m in masks if m != "acute"),
            "adc_storage_group": group,
            "adc_storage_rule_version": ADC_GROUP_RULE_VERSION,
            "adc_storage_note": ADC_GROUP_NOTE,
            "adc_storage_evidence": evidence,
            "adc_content_status": content,
            "adc_content_reason": content_reason,
            "adc_content_rule_version": ADC_CONTENT_RULE_VERSION,
            "is_pilot_case": pilot,
            "v2_pool_status": v2_pool_status(content, pilot=pilot),
            "background_code": background_code(d_adc),
            "raw_dtype": str(adc.get_data_dtype()),
            "nifti_scl_slope": _slope_inter(adc)[0],
            "nifti_scl_inter": _slope_inter(adc)[1],
            "trace_sidecar": {
                k: trace_json.get(k)
                for k in (
                    "SeriesDescription",
                    "Manufacturer",
                    "ManufacturersModelName",
                    "MagneticFieldStrength",
                    "SliceThickness",
                    "SpacingBetweenSlices",
                    "SoftwareVersions",
                )
            },
            "dataset_level_files_present": sorted(case.dataset_files),
        }
        src_hash_after = {k: sha256_file(p) for k, p in case.files.items()}
        if src_hash_after != src_hash_before:
            raise SoopFormatError(f"{alias}: a source file changed during conversion")
        audit = {
            "alias": alias,
            "format_version": FORMAT_VERSION,
            "operations": {k: v["plan"] for k, v in conv.items()},
            "methods": {k: v["method"] for k, v in conv.items()},
            "preservation_checks": conv,
            "grid_checks": grid,
            "source_sha256": src_hash_before,
            "source_unchanged": True,
            "derived_sha256": {
                k: sha256_file(work / f"{k}.nii.gz") for k in ("dwi", "adc", "label")
            },
            "not_performed": [
                "resampling",
                "skull_stripping",
                "brain_extraction",
                "intensity_normalisation",
                "adc_value_conversion",
                "background_code_change",
                "mask_binarisation_or_edit",
                "component_removal",
            ],
            "software": {"nibabel": nib.__version__, "numpy": np.__version__},
        }
        (work / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (work / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    os.replace(work, final)
    return {"alias": alias, "metadata": metadata, "audit": audit}
