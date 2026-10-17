"""Phase 2 validation of reconstruction, restoration, evaluation and quantification using GT only.

No model is involved. For every processed case:
1. processed GT -> slices (shuffled) -> reconstruct_volume -> must equal processed GT
2. processed GT -> reconstruct -> restore_to_original -> compare with original binary GT
   (also re-derives the Phase 1 round trip independently and checks both agree)
3. negative cases: empty prediction and synthetic false positives
4. synthetic predictions on real GT (perfect/empty/larger/smaller/shifted/tiny)

Usage:
    .venv\\Scripts\\python.exe scripts\\validate_phase2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from neuroscan_ai.analysis.components import label_components
from neuroscan_ai.data.case_metadata import build_case_metadata
from neuroscan_ai.datasets import discover
from neuroscan_ai.evaluation.report import summarize
from neuroscan_ai.evaluation.segmentation import evaluate_case
from neuroscan_ai.io import nifti
from neuroscan_ai.preprocessing.geometry import reorient, resample_to_grid, resample_to_spacing
from neuroscan_ai.preprocessing.mask import binarize
from neuroscan_ai.reconstruction.restore import restore_to_original
from neuroscan_ai.reconstruction.volume import (
    PredictionKind,
    PredictionVolume,
    VolumeGeometry,
    decompose_volume,
    reconstruct_volume,
)

SUBGROUPS = ("split", "spacing_group", "volume_group", "obliquity_group", "adc_unit_group")
SEED = 0


def lesion_centroid_world(mask: np.ndarray, affine: np.ndarray) -> np.ndarray | None:
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return None
    return (affine @ np.append(idx.mean(axis=0), 1.0))[:3]


def lost_components(original: np.ndarray, restored: np.ndarray) -> int:
    labels, n = label_components(original)
    hit = np.unique(labels[(restored > 0) & (labels > 0)])
    return n - hit.size


def reconstruction_check(meta, proc_mask, proc_img, rng) -> dict[str, Any]:
    geom = VolumeGeometry.from_meta(meta, "processed")
    pairs = decompose_volume(proc_mask)
    shuffled = [pairs[i] for i in rng.permutation(len(pairs))]
    rec = reconstruct_volume(shuffled, geom, PredictionKind.BINARY)
    prob = reconstruct_volume(
        [(z, s.astype(np.float32)) for z, s in shuffled], geom, PredictionKind.PROBABILITY
    )
    r = evaluate_case(rec.data, proc_mask, geom.spacing, meta["case_id"])
    return {
        "case_id": meta["case_id"],
        "n_slices": len(pairs),
        "binary_identical": bool(np.array_equal(rec.data, proc_mask)),
        "probability_identical": bool(np.array_equal(prob.data, proc_mask.astype(np.float32))),
        "shape_preserved": list(rec.data.shape) == meta["processed"]["shape"],
        "affine_matches_meta": bool(np.allclose(rec.geometry.affine, meta["processed"]["affine"])),
        "affine_matches_nifti": bool(np.allclose(rec.geometry.affine, proc_img.affine, atol=1e-5)),
        "spacing_preserved": bool(
            np.allclose(rec.geometry.spacing, meta["processed"]["spacing"], atol=1e-5)
        ),
        "orientation": rec.geometry.orientation,
        **{k: r[k] for k in ("has_lesion", "dice", "iou", "sensitivity", "precision")},
        "volume_error_ml": r["volume_error_ml"],
        "negative_outcome": r["negative_outcome"],
    }


def roundtrip_check(meta, proc_mask, orig_gt, orig_img) -> tuple[dict[str, Any], np.ndarray]:
    proc_geom = VolumeGeometry.from_meta(meta, "processed")
    orig_geom = VolumeGeometry(tuple(orig_img.shape[:3]), orig_img.affine)
    rec = reconstruct_volume(decompose_volume(proc_mask), proc_geom, "binary")
    restored = restore_to_original(rec, orig_geom)
    prob = restore_to_original(
        PredictionVolume(proc_mask.astype(np.float32), proc_geom, PredictionKind.PROBABILITY),
        orig_geom,
    )
    sp = nifti.spacing_from_affine(orig_img.affine)
    r = evaluate_case(restored.data, orig_gt, sp, meta["case_id"])

    # Independent re-derivation of the Phase 1 round trip (analyze_spacing_policy.py method).
    ras_gt, ras_aff = reorient(orig_gt, orig_img.affine, "RAS")
    p1_proc, p1_aff, axes = resample_to_spacing(ras_gt, ras_aff, (1.0, 1.0, None), order=1)
    p1_proc = (p1_proc >= 0.5).astype(np.uint8) if any(axes) else p1_proc
    p1_back = (resample_to_grid(p1_proc, p1_aff, ras_aff, ras_gt.shape, order=1) >= 0.5).astype(
        np.uint8
    )
    p1_dice = evaluate_case(p1_back, ras_gt, sp)["dice"]

    c_gt = lesion_centroid_world(orig_gt, orig_img.affine)
    c_rs = lesion_centroid_world(restored.data, orig_img.affine)
    row = {
        **r,
        "component_count_difference": r["component_count_prediction"] - r["component_count_gt"],
        "components_lost": lost_components(orig_gt, restored.data),
        "restored_shape_ok": list(restored.data.shape) == meta["original"]["shape"],
        "restored_affine_ok": bool(
            np.allclose(restored.geometry.affine, orig_img.affine, atol=1e-5)
        ),
        "restored_orientation": restored.geometry.orientation,
        "restored_values_binary": set(np.unique(restored.data).tolist()) <= {0, 1},
        "probability_path_equals_binary_path": bool(
            np.array_equal(prob.data >= 0.5, restored.data.astype(bool))
        ),
        "probability_in_unit_range": bool(prob.data.min() >= 0 and prob.data.max() <= 1),
        "stored_processed_equals_rederived": bool(np.array_equal(p1_proc, proc_mask)),
        "phase1_method_dice": p1_dice,
        "matches_phase1_method": (r["dice"] is None and p1_dice is None)
        or (r["dice"] is not None and abs(r["dice"] - p1_dice) < 1e-12),
        "centroid_shift_mm": None if c_gt is None else float(np.linalg.norm(c_gt - c_rs)),
        "processed_volume_ml": meta["lesion"]["processed_volume_ml"],
        "processed_vs_original_volume_rel_err": None
        if not meta["lesion"]["has_lesion"]
        else meta["lesion"]["processed_volume_ml"] / meta["lesion"]["original_volume_ml"] - 1,
    }
    return row, restored.data


def negative_checks(meta, orig_gt, orig_img) -> list[dict[str, Any]]:
    proc_geom = VolumeGeometry.from_meta(meta, "processed")
    orig_geom = VolumeGeometry(tuple(orig_img.shape[:3]), orig_img.affine)
    sp = nifti.spacing_from_affine(orig_img.affine)
    rows = []

    def add(scenario: str, pred: np.ndarray, expected_ml: float | None) -> None:
        r = evaluate_case(pred, orig_gt, sp, meta["case_id"])
        rows.append(
            {
                "case_id": meta["case_id"],
                "scenario": scenario,
                **{
                    k: r[k]
                    for k in (
                        "has_lesion",
                        "negative_outcome",
                        "predicted_lesion_voxels",
                        "predicted_lesion_volume_ml",
                        "any_false_positive",
                        "false_positive_volume_ml",
                        "false_positive_component_count",
                        "dice",
                    )
                },
                "expected_false_positive_volume_ml": expected_ml,
            }
        )

    empty = restore_to_original(
        reconstruct_volume(
            decompose_volume(np.zeros(proc_geom.shape, np.uint8)), proc_geom, "binary"
        ),
        orig_geom,
    )
    add("empty_prediction_via_pipeline", empty.data, 0.0)

    fp = np.zeros(orig_gt.shape, np.uint8)
    cx, cy, cz = (s // 2 for s in orig_gt.shape)
    fp[cx : cx + 2, cy : cy + 2, cz] = 1
    add("synthetic_fp_2x2x1_original_space", fp, 4 * float(np.prod(sp)) / 1000)

    proc_fp = np.zeros(proc_geom.shape, np.uint8)
    px, py, pz = (s // 2 for s in proc_geom.shape)
    proc_fp[px : px + 4, py : py + 4, pz] = 1  # 4x4 mm at 1 mm in-plane
    restored = restore_to_original(
        reconstruct_volume(decompose_volume(proc_fp), proc_geom, "binary"), orig_geom
    )
    add("synthetic_fp_4x4mm_processed_space_restored", restored.data, None)
    return rows


def synthetic_metric_tests(case_ids, gts, spacings) -> list[dict[str, Any]]:
    ball = ndimage.generate_binary_structure(3, 1)
    tests = []

    def record(case_id, scenario, pred, gt, sp, expect):
        r = evaluate_case(pred, gt, sp, case_id)
        checks = {name: bool(fn(r)) for name, fn in expect.items()}
        nan_free = all(not (isinstance(v, float) and np.isnan(v)) for v in r.values())
        tests.append(
            {
                "case_id": case_id,
                "scenario": scenario,
                "passed": all(checks.values()) and nan_free,
                "checks": checks,
                "nan_free": nan_free,
                "metrics": {
                    k: r[k]
                    for k in (
                        "dice",
                        "iou",
                        "sensitivity",
                        "precision",
                        "gt_volume_ml",
                        "prediction_volume_ml",
                        "volume_error_ml",
                        "relative_volume_error_percent",
                        "negative_outcome",
                        "false_positive_volume_ml",
                    )
                },
            }
        )

    for cid in case_ids:
        gt, sp = gts[cid], spacings[cid]
        if not gt.any():
            record(
                cid,
                "F_gt_empty_pred_empty",
                np.zeros_like(gt),
                gt,
                sp,
                {
                    "correct_negative": lambda r: r["negative_outcome"] == "correct_negative",
                    "dice_undefined": lambda r: r["dice"] is None,
                },
            )
            fp = np.zeros_like(gt)
            fp[tuple(s // 2 for s in gt.shape)] = 1
            one_voxel_ml = float(np.prod(sp)) / 1000
            record(
                cid,
                "G_gt_empty_pred_positive",
                fp,
                gt,
                sp,
                {
                    "false_positive": lambda r: r["negative_outcome"] == "false_positive_case",
                    "fp_volume": lambda r, v=one_voxel_ml: (
                        abs(r["false_positive_volume_ml"] - v) < 1e-12
                    ),
                },
            )
            continue
        record(
            cid,
            "A_perfect",
            gt,
            gt,
            sp,
            {
                "dice_1": lambda r: r["dice"] == 1.0,
                "iou_1": lambda r: r["iou"] == 1.0,
                "volume_error_0": lambda r: r["volume_error_ml"] == 0.0,
            },
        )
        record(
            cid,
            "B_empty_prediction",
            np.zeros_like(gt),
            gt,
            sp,
            {
                "dice_0": lambda r: r["dice"] == 0.0,
                "sens_0": lambda r: r["sensitivity"] == 0.0,
                "precision_undefined": lambda r: r["precision"] is None,
            },
        )
        big = ndimage.binary_dilation(gt, ball).astype(np.uint8)
        record(
            cid,
            "C_larger_dilated",
            big,
            gt,
            sp,
            {
                "sens_1": lambda r: r["sensitivity"] == 1.0,
                "precision_lt_1": lambda r: r["precision"] < 1.0,
                "positive_volume_error": lambda r: r["volume_error_ml"] > 0,
            },
        )
        small = ndimage.binary_erosion(gt, ball).astype(np.uint8)
        if small.any():
            record(
                cid,
                "D_smaller_eroded",
                small,
                gt,
                sp,
                {
                    "precision_1": lambda r: r["precision"] == 1.0,
                    "sens_lt_1": lambda r: r["sensitivity"] < 1.0,
                    "negative_volume_error": lambda r: r["volume_error_ml"] < 0,
                },
            )
        shifted = np.roll(gt, 2, axis=0)
        record(
            cid,
            "E_partial_overlap_shift2x",
            shifted,
            gt,
            sp,
            {
                "dice_matches_counts": lambda r: (
                    abs(r["dice"] - 2 * r["iou"] / (1 + r["iou"])) < 1e-12
                ),
                "dice_in_0_1": lambda r: 0.0 <= r["dice"] <= 1.0,
            },
        )
        if 0 < gts[cid].sum() * np.prod(sp) / 1000 < 1:
            record(
                cid,
                "H_tiny_lesion_lt_1ml_perfect",
                gt,
                gt,
                sp,
                {
                    "dice_1": lambda r: r["dice"] == 1.0,
                    "volume_lt_1": lambda r: r["gt_volume_ml"] < 1.0,
                },
            )
    return tests


def subgroup_table(rows: pd.DataFrame) -> dict[str, Any]:
    def agg(df: pd.DataFrame) -> dict[str, Any]:
        pos = df[df.has_lesion]
        return {
            "n_cases": int(len(df)),
            "n_positive": int(len(pos)),
            "n_negative": int((~df.has_lesion).sum()),
            "dice_mean": None if pos.empty else float(pos.dice.mean()),
            "dice_min": None if pos.empty else float(pos.dice.min()),
            "dice_lt_0.95": int((pos.dice < 0.95).sum()),
            "iou_mean": None if pos.empty else float(pos.iou.mean()),
            "abs_volume_error_ml_mean": None
            if pos.empty
            else float(pos.absolute_volume_error_ml.mean()),
            "rel_volume_error_percent_mean_abs": None
            if pos.empty
            else float(pos.relative_volume_error_percent.abs().mean()),
            "components_gt": int(pos.component_count_gt.sum()),
            "components_lost": int(pos.components_lost.sum()),
            "component_count_difference_sum": int(pos.component_count_difference.sum()),
            "centroid_shift_mm_max": None if pos.empty else float(pos.centroid_shift_mm.max()),
            "correct_negative": int((df.negative_outcome == "correct_negative").sum()),
        }

    out = {"all": agg(rows)}
    for key in ("spacing_group", "volume_group", "obliquity_group", "adc_unit_group", "split"):
        out[key] = {str(v): agg(g) for v, g in rows.groupby(key)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--dataset", default="isles2022")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/ISLES2022"))
    parser.add_argument("--qa", type=Path, default=Path("results/qa/case_summary.csv"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/split_v1_seed42.json"))
    parser.add_argument(
        "--phase1", type=Path, default=Path("results/qa/spacing_policy_comparison.json")
    )
    parser.add_argument("--out", type=Path, default=Path("results/phase2"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records = discover(args.dataset, args.raw_root)
    case_meta = build_case_metadata(args.qa, records, args.split)
    case_meta.to_csv(args.out / "case_metadata.csv", index=False)
    meta_by_id = case_meta.set_index("case_id")

    rng = np.random.default_rng(SEED)
    recon_rows, rt_rows, neg_rows = [], [], []
    gts, spacings = {}, {}
    for rec in tqdm(records, desc="phase2"):
        case_dir = args.processed / rec.case_id
        meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
        proc_img = nib.load(case_dir / "mask.nii.gz")
        proc_mask = np.asanyarray(proc_img.dataobj)
        orig_img = nib.load(rec.label)
        orig_gt = binarize(orig_img.get_fdata())
        extra = {k: meta_by_id.loc[rec.case_id, k] for k in SUBGROUPS}
        extra["z_spacing"] = float(meta_by_id.loc[rec.case_id, "z_spacing"])
        extra["obliquity_deg"] = float(meta_by_id.loc[rec.case_id, "obliquity_deg"])

        recon_rows.append({**reconstruction_check(meta, proc_mask, proc_img, rng), **extra})
        rt, _ = roundtrip_check(meta, proc_mask, orig_gt, orig_img)
        rt_rows.append({**rt, **extra})
        if not meta["lesion"]["has_lesion"]:
            neg_rows += [{**r, **extra} for r in negative_checks(meta, orig_gt, orig_img)]
        gts[rec.case_id] = orig_gt
        spacings[rec.case_id] = nifti.spacing_from_affine(orig_img.affine)

    recon = pd.DataFrame(recon_rows)
    rt = pd.DataFrame(rt_rows)
    neg = pd.DataFrame(neg_rows)
    recon.to_csv(args.out / "reconstruction_validation.csv", index=False)
    rt.to_csv(args.out / "original_space_roundtrip.csv", index=False)
    neg.to_csv(args.out / "negative_case_validation.csv", index=False)

    # Synthetic predictions: all test-split cases (37, incl. its negative) + all negatives
    # + every <1 mL lesion case.
    test_ids = list(case_meta[case_meta.split == "test"].case_id)
    neg_ids = list(case_meta[~case_meta.has_lesion].case_id)
    tiny_ids = list(case_meta[case_meta.volume_group == "(0,1) mL"].case_id)
    synth_ids = sorted(set(test_ids) | set(neg_ids) | set(tiny_ids))
    synth = synthetic_metric_tests(synth_ids, gts, spacings)
    (args.out / "synthetic_metric_tests.json").write_text(
        json.dumps(
            {"n_tests": len(synth), "n_passed": sum(t["passed"] for t in synth), "tests": synth},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    test_recon = recon[recon.split == "test"]
    phase1 = (
        json.loads(args.phase1.read_text(encoding="utf-8"))["B_xy1mm_keep_z"]
        if args.phase1.exists()
        else None
    )
    sub = subgroup_table(rt)
    summary = {
        "reconstruction": {
            "n_cases": int(len(recon)),
            "all_binary_identical": bool(recon.binary_identical.all()),
            "all_probability_identical": bool(recon.probability_identical.all()),
            "all_shape_affine_spacing_preserved": bool(
                (
                    recon.shape_preserved
                    & recon.affine_matches_meta
                    & recon.affine_matches_nifti
                    & recon.spacing_preserved
                ).all()
            ),
            "orientations": recon.orientation.value_counts().to_dict(),
            "test_split": {
                "n_cases": int(len(test_recon)),
                "n_negative": int((~test_recon.has_lesion).sum()),
                "all_identical": bool(test_recon.binary_identical.all()),
                "positive_dice_all_1": bool((test_recon[test_recon.has_lesion].dice == 1.0).all()),
                "positive_iou_sens_prec_all_1": bool(
                    (test_recon[test_recon.has_lesion][["iou", "sensitivity", "precision"]] == 1.0)
                    .all()
                    .all()
                ),
                "volume_error_all_0": bool((test_recon.volume_error_ml == 0).all()),
                "negative_outcomes": test_recon[~test_recon.has_lesion].negative_outcome.tolist(),
            },
        },
        "original_space_roundtrip": {
            "geometry_ok": bool(
                (rt.restored_shape_ok & rt.restored_affine_ok & rt.restored_values_binary).all()
            ),
            "restored_orientations": rt.restored_orientation.value_counts().to_dict(),
            "probability_path_equals_binary_path": bool(
                rt.probability_path_equals_binary_path.all()
            ),
            "probability_in_unit_range": bool(rt.probability_in_unit_range.all()),
            "stored_processed_equals_rederived": bool(rt.stored_processed_equals_rederived.all()),
            "matches_phase1_method_all_cases": bool(rt.matches_phase1_method.all()),
            "subgroups": sub,
            "volume_consistency": {
                "processed_vs_original_abs_rel_err_mean": float(
                    rt.processed_vs_original_volume_rel_err.abs().mean()
                ),
                "processed_vs_original_abs_rel_err_max": float(
                    rt.processed_vs_original_volume_rel_err.abs().max()
                ),
                "restored_vs_original_abs_rel_err_mean_percent": float(
                    rt[rt.has_lesion].relative_volume_error_percent.abs().mean()
                ),
                "restored_vs_original_abs_rel_err_max_percent": float(
                    rt[rt.has_lesion].relative_volume_error_percent.abs().max()
                ),
            },
        },
        "phase1_reference_B_xy1mm_keep_z": None
        if phase1 is None
        else {
            g: {
                "roundtrip_dice_mean": phase1[g]["roundtrip_dice_mean"],
                "roundtrip_dice_min": phase1[g]["roundtrip_dice_min"],
                "components_lost": phase1[g]["components_lost"],
            }
            for g in ("all", "thin_slice", "thick_slice", "small_lt_1ml")
        },
        "negative_cases": neg.to_dict(orient="records"),
        "synthetic_metric_tests": {
            "n_tests": len(synth),
            "n_passed": sum(t["passed"] for t in synth),
            "by_scenario": pd.Series([t["scenario"] for t in synth]).value_counts().to_dict(),
        },
        "evaluation_report_on_roundtrip": summarize(rt_rows, subgroup_keys=["spacing_group"]),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "reconstruction",
                    "phase1_reference_B_xy1mm_keep_z",
                    "synthetic_metric_tests",
                )
            },
            indent=2,
            default=str,
        )
    )
    print(
        json.dumps(
            {k: v for k, v in summary["original_space_roundtrip"].items() if k != "subgroups"},
            indent=2,
            default=str,
        )
    )
    print(json.dumps(sub, indent=2, default=str))


if __name__ == "__main__":
    main()
