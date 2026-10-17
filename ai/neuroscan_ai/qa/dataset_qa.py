"""Dataset-agnostic QA over ``CaseRecord`` lists (read-only; never writes next to raw data)."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from neuroscan_ai.datasets.base import CaseRecord
from neuroscan_ai.io import nifti
from neuroscan_ai.preprocessing.mask import binarize, lesion_volume_ml

# Slices thicker than this (mm) are counted as the "thick-slice" acquisition group.
THICK_SLICE_MM = 3.0


def _intensity_stats(data: np.ndarray) -> dict[str, float]:
    finite = data[np.isfinite(data)]
    nonzero = finite[finite != 0]
    if nonzero.size == 0:
        return {"nonzero_frac": 0.0, "p1": 0.0, "p50": 0.0, "p99": 0.0, "nan_count": 0}
    p1, p50, p99 = np.percentile(nonzero, [1, 50, 99])
    return {
        "nonzero_frac": round(nonzero.size / data.size, 4),
        "p1": round(float(p1), 3),
        "p50": round(float(p50), 3),
        "p99": round(float(p99), 3),
        "nan_count": int(data.size - finite.size),
    }


def qa_case(record: CaseRecord) -> dict[str, Any]:
    """One flat row of QA facts for a case. Errors are recorded, not raised."""
    row: dict[str, Any] = {"dataset": record.dataset, "case_id": record.case_id}
    paths = record.all_paths()
    missing = [k for k, p in paths.items() if not p.exists()]
    row["missing_files"] = ";".join(missing)
    if missing:
        row["error"] = "missing files"
        return row

    try:
        imgs = {k: nifti.load(p) for k, p in paths.items()}
        ref_key = next(iter(record.images))
        ref = imgs[ref_key]

        for key, img in imgs.items():
            g = nifti.geometry(img)
            row[f"{key}_shape"] = "x".join(map(str, g["shape"]))
            row[f"{key}_sx"], row[f"{key}_sy"], row[f"{key}_sz"] = g["spacing"]
            row[f"{key}_orientation"] = g["orientation"]
            row[f"{key}_dtype"] = g["dtype"]
            row[f"{key}_scl_slope"] = g["scl_slope"]
            row[f"{key}_zooms_match_affine"] = bool(
                np.allclose(g["header_zooms"], g["spacing"], atol=nifti.AFFINE_ATOL_MM)
            )
            if key != ref_key:
                row[f"{key}_same_grid_as_{ref_key}"] = nifti.same_grid(img, ref)

        # Slice (lowest-resolution) axis should be the 3rd array axis for slice-wise models.
        sp = nifti.spacing_from_affine(ref.affine)
        row["slice_axis_is_last"] = bool(sp[2] >= max(sp[0], sp[1]) - nifti.AFFINE_ATOL_MM)
        row["thick_slice"] = bool(sp[2] > THICK_SLICE_MM)

        for key in record.images:
            for stat, val in _intensity_stats(imgs[key].get_fdata(dtype=np.float32)).items():
                row[f"{key}_{stat}"] = val

        if record.label is not None:
            raw = imgs["label"].get_fdata()
            uniq = np.unique(raw)
            row["mask_raw_n_unique"] = int(uniq.size)
            row["mask_raw_min"], row["mask_raw_max"] = float(uniq.min()), float(uniq.max())
            row["mask_raw_exact_binary"] = bool(set(uniq.tolist()) <= {0.0, 1.0})
            binary = binarize(raw)
            row["mask_binary_values"] = ",".join(map(str, np.unique(binary).tolist()))
            row["lesion_voxels"] = int(binary.sum())
            row["voxel_volume_mm3"] = round(float(np.prod(sp)), 6)
            row["lesion_volume_ml"] = round(lesion_volume_ml(binary, sp), 6)
            row["has_lesion"] = bool(row["lesion_voxels"] > 0)
            _, n_components = ndimage.label(binary, structure=np.ones((3, 3, 3)))
            row["lesion_components_26"] = int(n_components)
            if row["has_lesion"]:
                zs = np.nonzero(binary.any(axis=(0, 1)))[0]
                row["lesion_slices"] = int(zs.size)
                row["lesion_z_min"], row["lesion_z_max"] = int(zs.min()), int(zs.max())
        row["error"] = ""
    except Exception as exc:  # keep QA running over the whole dataset
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def summarize(rows: list[dict[str, Any]], image_keys: list[str]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    ref = image_keys[0]

    def counts(key: str) -> dict[str, int]:
        return dict(Counter(str(r.get(key)) for r in ok).most_common())

    spacing_triplets = Counter(
        f"{r[f'{ref}_sx']:.4f}x{r[f'{ref}_sy']:.4f}x{r[f'{ref}_sz']:.4f}" for r in ok
    )
    axes = {}
    for ax in ("sx", "sy", "sz"):
        v = np.array([r[f"{ref}_{ax}"] for r in ok])
        axes[ax] = {
            "min": float(v.min()),
            "median": float(np.median(v)),
            "mean": round(float(v.mean()), 4),
            "max": float(v.max()),
        }

    grid_checks = {}
    for key in rows[0]:
        if key.endswith(f"_same_grid_as_{ref}"):
            vals = [bool(r[key]) for r in ok]
            grid_checks[key] = {"true": sum(vals), "false": len(vals) - sum(vals)}

    summary: dict[str, Any] = {
        "n_cases": len(rows),
        "n_errors": len(rows) - len(ok),
        "errors": {r["case_id"]: r["error"] for r in rows if r.get("error")},
        "grid_checks": grid_checks,
        "zooms_match_affine": {
            k: sum(bool(r.get(f"{k}_zooms_match_affine")) for r in ok)
            for k in [*image_keys, "label"]
        },
        "orientation": {k: counts(f"{k}_orientation") for k in [*image_keys, "label"]},
        "dtype": {k: counts(f"{k}_dtype") for k in [*image_keys, "label"]},
        "shape": counts(f"{ref}_shape"),
        "spacing_triplets": dict(spacing_triplets.most_common()),
        "spacing_per_axis": axes,
        "thick_slice_cases": sum(bool(r["thick_slice"]) for r in ok),
        "thick_slice_threshold_mm": THICK_SLICE_MM,
        "slice_axis_is_last": sum(bool(r["slice_axis_is_last"]) for r in ok),
    }

    if "lesion_volume_ml" in ok[0]:
        vol = np.array([r["lesion_volume_ml"] for r in ok])
        pos = vol[vol > 0]
        summary["mask"] = {
            "raw_not_exact_binary": [r["case_id"] for r in ok if not r["mask_raw_exact_binary"]],
            "binary_values_after_threshold": counts("mask_binary_values"),
            "empty_cases": [r["case_id"] for r in ok if not r["has_lesion"]],
        }
        summary["lesion_volume_ml"] = {
            "all": {
                "min": float(vol.min()),
                "median": round(float(np.median(vol)), 4),
                "mean": round(float(vol.mean()), 4),
                "max": float(vol.max()),
            },
            "positive_quartiles": [round(float(q), 4) for q in np.percentile(pos, [25, 50, 75])],
            "bins": {
                "0 (empty)": int((vol == 0).sum()),
                "(0, 1)": int(((vol > 0) & (vol < 1)).sum()),
                "[1, 10)": int(((vol >= 1) & (vol < 10)).sum()),
                "[10, 50)": int(((vol >= 10) & (vol < 50)).sum()),
                "[50, inf)": int((vol >= 50).sum()),
            },
        }
    return summary


def sha256_manifest(root: Path) -> dict[str, str]:
    """SHA-256 of every file under ``root`` keyed by POSIX relative path."""
    manifest = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        manifest[path.relative_to(root).as_posix()] = h.hexdigest()
    return manifest
