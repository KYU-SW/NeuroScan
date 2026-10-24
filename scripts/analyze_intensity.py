"""Train-split-only DWI/ADC intensity analysis and normalisation candidate comparison.

Only the 176 train patients are read (through SplitGuard); validation and test are never opened.

Usage:
    .venv\\Scripts\\python.exe scripts\\analyze_intensity.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from neuroscan_ai.data.case_metadata import adc_unit_group
from neuroscan_ai.data.split_guard import SplitGuard
from neuroscan_ai.training.normalization import METHODS, NormalizationSpec, brain_mask, normalize

PCTS = (0.5, 1, 5, 50, 95, 99, 99.5)
SUMMARY_KEYS = ("p5", "p50", "p95", "lesion_median")


def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {}
    out = {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
    }
    for p, v in zip(PCTS, np.percentile(values, PCTS), strict=True):
        out[f"{prefix}_p{p:g}"] = float(v)
    return out


def group_spread(df: pd.DataFrame, col: str) -> float | None:
    """(max - min of group means) / mean within-group std: how far apart unit groups sit."""
    g = df.groupby("adc_unit_group")[col]
    means, stds = g.mean(), g.std().fillna(0)
    within = float(stds.mean())
    if within == 0 or len(means) < 2:
        return None
    return float((means.max() - means.min()) / within)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/split_v1_seed42.json"))
    parser.add_argument("--out", type=Path, default=Path("results/phase3/intensity"))
    args = parser.parse_args()

    guard = SplitGuard(args.split)
    train_ids = guard.cases("train")
    rows = []
    norm_rows = []
    for cid in tqdm(train_ids, desc="train intensity"):
        d = args.processed / cid
        dwi = nib.load(d / "dwi.nii.gz").get_fdata(dtype=np.float32)
        adc = nib.load(d / "adc.nii.gz").get_fdata(dtype=np.float32)
        lesion = np.asanyarray(nib.load(d / "mask.nii.gz").dataobj) > 0
        brain = brain_mask(dwi)
        adc_brain = adc[brain]
        row = {
            "case_id": cid,
            "adc_unit_group": adc_unit_group(float(np.median(adc_brain))),
            "brain_frac": float(brain.mean()),
            "dwi_zero_frac_all": float((dwi == 0).mean()),
            "adc_zero_frac_in_brain": float((adc_brain == 0).mean()),
            "adc_negative_frac_in_brain": float((adc_brain < 0).mean()),
            "adc_nonzero_outside_brain_frac": float((adc[~brain] != 0).mean()),
            "lesion_in_brain_frac": float(brain[lesion].mean()) if lesion.any() else None,
            **stats(dwi.ravel(), "dwi_all"),
            **stats(dwi[brain], "dwi_brain"),
            **stats(adc.ravel(), "adc_all"),
            **stats(adc_brain, "adc_brain"),
        }
        rows.append(row)

        for modality, vol in (("dwi", dwi), ("adc", adc)):
            raw_entry = {
                "case_id": cid,
                "adc_unit_group": row["adc_unit_group"],
                "modality": modality,
                "method": "raw",
            }
            variants = {"raw": vol}
            for method in METHODS:
                variants[method] = normalize(vol, brain, NormalizationSpec(method))
            for method, x in variants.items():
                v = x[brain]
                p5, p50, p95 = np.percentile(v, [5, 50, 95])
                norm_rows.append(
                    {
                        **raw_entry,
                        "method": method,
                        "finite": bool(np.isfinite(x).all()),
                        "p5": float(p5),
                        "p50": float(p50),
                        "p95": float(p95),
                        "abs_max": float(np.abs(v).max()),
                        "lesion_median": float(np.median(x[lesion])) if lesion.any() else np.nan,
                    }
                )

    args.out.mkdir(parents=True, exist_ok=True)
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(args.out / "train_intensity_stats.csv", index=False)
    norm_df = pd.DataFrame(norm_rows)
    norm_df.to_csv(args.out / "normalization_candidates.csv", index=False)

    comparison = {}
    for (modality, method), g in norm_df.groupby(["modality", "method"]):
        comparison.setdefault(modality, {})[method] = {
            "all_finite": bool(g.finite.all()),
            "abs_max_p99_over_cases": float(np.percentile(g.abs_max, 99)),
            "abs_max_max": float(g.abs_max.max()),
            "group_means": {
                k: {str(grp): float(v) for grp, v in g.groupby("adc_unit_group")[k].mean().items()}
                for k in SUMMARY_KEYS
            },
            "group_spread_effect_size": {k: group_spread(g, k) for k in SUMMARY_KEYS},
        }
    summary = {
        "n_train_cases": len(train_ids),
        "access": guard.access_report(),
        "adc_unit_groups": stats_df.adc_unit_group.value_counts().to_dict(),
        "background": {
            "dwi_zero_frac_all_mean": float(stats_df.dwi_zero_frac_all.mean()),
            "adc_zero_frac_in_brain_mean": float(stats_df.adc_zero_frac_in_brain.mean()),
            "adc_zero_frac_in_brain_max": float(stats_df.adc_zero_frac_in_brain.max()),
            "adc_negative_frac_in_brain_max": float(stats_df.adc_negative_frac_in_brain.max()),
            "adc_nonzero_outside_brain_frac_max": float(
                stats_df.adc_nonzero_outside_brain_frac.max()
            ),
            "lesion_in_brain_frac_min": float(stats_df.lesion_in_brain_frac.dropna().min()),
        },
        "raw_brain_percentiles_by_group": {
            m: stats_df.groupby("adc_unit_group")[
                [f"{m}_brain_p1", f"{m}_brain_p50", f"{m}_brain_p99"]
            ]
            .median()
            .to_dict(orient="index")
            for m in ("dwi", "adc")
        },
        "comparison": comparison,
    }
    (args.out / "normalization_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "n_train_cases",
                    "access",
                    "adc_unit_groups",
                    "background",
                    "raw_brain_percentiles_by_group",
                )
            },
            indent=2,
        )
    )
    for modality in ("dwi", "adc"):
        for method, c in comparison[modality].items():
            spread = {
                k: None if v is None else round(v, 2)
                for k, v in c["group_spread_effect_size"].items()
            }
            lesion = {g: round(v, 3) for g, v in c["group_means"]["lesion_median"].items()}
            p50 = {g: round(v, 3) for g, v in c["group_means"]["p50"].items()}
            print(
                f"{modality:3s} {method:16s} finite={c['all_finite']} "
                f"absmax_p99={c['abs_max_p99_over_cases']:10.3f} spread={spread}"
            )
            print(f"    lesion_median by group: {lesion}  p50: {p50}")


if __name__ == "__main__":
    main()
