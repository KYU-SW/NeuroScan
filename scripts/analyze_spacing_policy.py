"""Compare resampling policies by a GT mask round trip (original -> policy grid -> original).

For every labelled case the binary GT is resampled to each policy grid (linear + >=0.5) and
back onto the original grid, then compared with the untouched original mask. This measures how
much lesion information a policy destroys before any model is involved.

Usage:
    .venv\\Scripts\\python.exe scripts\\analyze_spacing_policy.py --dataset isles2022 \\
        --root data\\raw\\ISLES2022
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from neuroscan_ai.analysis.components import label_components
from neuroscan_ai.datasets import discover
from neuroscan_ai.io import nifti
from neuroscan_ai.preprocessing.geometry import reorient, resample_to_grid, resample_to_spacing
from neuroscan_ai.preprocessing.mask import binarize, lesion_volume_ml

POLICIES: dict[str, tuple[float | None, float | None, float | None]] = {
    "A_isotropic_1mm": (1.0, 1.0, 1.0),
    "B_xy2mm_keep_z": (2.0, 2.0, None),
    "C_median_2mm_iso": (2.0, 2.0, 2.0),
    # Reference variants: B with a finer in-plane target, and no resampling at all.
    "B_xy1mm_keep_z": (1.0, 1.0, None),
    "native_no_resampling": (None, None, None),
}
THICK_SLICE_MM = 3.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2 * np.logical_and(a, b).sum() / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("results/qa/spacing_policy_comparison.json")
    )
    args = parser.parse_args()

    per_case: dict[str, list[dict]] = {p: [] for p in POLICIES}
    for rec in tqdm(discover(args.dataset, args.root), desc="policies"):
        img = nifti.load(rec.label)
        mask, affine = reorient(binarize(img.get_fdata()), img.affine)
        spacing = nifti.spacing_from_affine(affine)
        orig_vol = lesion_volume_ml(mask, spacing)
        labels, n_comp = label_components(mask)
        for name, target in POLICIES.items():
            proc, proc_aff, resampled = resample_to_spacing(mask, affine, target, order=1)
            proc_bin = (proc >= 0.5).astype(np.uint8) if any(resampled) else proc
            proc_spacing = nifti.spacing_from_affine(proc_aff)
            back = resample_to_grid(proc_bin, proc_aff, affine, mask.shape, order=1)
            back_bin = (back >= 0.5).astype(np.uint8)
            lost = sum(1 for i in range(1, n_comp + 1) if not back_bin[labels == i].any())
            per_case[name].append(
                {
                    "case_id": rec.case_id,
                    "thick": spacing[2] > THICK_SLICE_MM,
                    "has_lesion": orig_vol > 0,
                    "voxels": int(np.prod(proc.shape)),
                    "slices": int(proc.shape[2]),
                    "z_step_mm": proc_spacing[2],
                    "synthetic_slice_frac": 0.0
                    if not resampled[2]
                    else max(0.0, 1 - mask.shape[2] / proc.shape[2]),
                    "proc_vol_rel_err": None
                    if orig_vol == 0
                    else lesion_volume_ml(proc_bin, proc_spacing) / orig_vol - 1,
                    "roundtrip_dice": dice(mask, back_bin),
                    "roundtrip_vol_rel_err": None
                    if orig_vol == 0
                    else lesion_volume_ml(back_bin, spacing) / orig_vol - 1,
                    "components": n_comp,
                    "components_lost": lost,
                    "lesion_vanished": orig_vol > 0 and back_bin.sum() == 0,
                    "small_lt_1ml": 0 < orig_vol < 1,
                }
            )

    def agg(rows: list[dict]) -> dict:
        pos = [r for r in rows if r["has_lesion"]]
        d = np.array([r["roundtrip_dice"] for r in pos])
        ve = np.abs([r["roundtrip_vol_rel_err"] for r in pos])
        pe = np.abs([r["proc_vol_rel_err"] for r in pos])
        return {
            "n_cases": len(rows),
            "total_voxels_M": round(sum(r["voxels"] for r in rows) / 1e6, 2),
            "total_slices": sum(r["slices"] for r in rows),
            "z_step_mm": sorted({round(r["z_step_mm"], 3) for r in rows}),
            "mean_synthetic_slice_frac": round(
                float(np.mean([r["synthetic_slice_frac"] for r in rows])), 3
            ),
            "roundtrip_dice_mean": round(float(d.mean()), 4) if d.size else None,
            "roundtrip_dice_min": round(float(d.min()), 4) if d.size else None,
            "roundtrip_dice_lt_0.95": int((d < 0.95).sum()),
            "processed_volume_abs_rel_err_mean": round(float(pe.mean()), 4) if pe.size else None,
            "roundtrip_volume_abs_rel_err_mean": round(float(ve.mean()), 4) if ve.size else None,
            "components_total": sum(r["components"] for r in pos),
            "components_lost": sum(r["components_lost"] for r in pos),
            "cases_lesion_vanished": [r["case_id"] for r in pos if r["lesion_vanished"]],
        }

    report = {}
    for name, rows in per_case.items():
        report[name] = {
            "target_spacing": POLICIES[name],
            "all": agg(rows),
            "thin_slice": agg([r for r in rows if not r["thick"]]),
            "thick_slice": agg([r for r in rows if r["thick"]]),
            "small_lt_1ml": agg([r for r in rows if r["small_lt_1ml"]]),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
