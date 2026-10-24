"""Paired patient-wise comparison of two validation runs (Phase 4: 2D baseline vs 2.5D).

Usage:
    .venv\\Scripts\\python.exe scripts\\compare_runs.py ^
        --baseline results\\phase3\\<2d_run_id> --candidate results\\phase4\\<25d_run_id>

Reads each run's validation_metrics.csv, summary.json and config.yaml (no model or image is
loaded, no test patient is touched). Refuses to compare when either run accessed the test split,
the evaluated patients differ, or the configs differ beyond ``ALLOWED_CONFIG_DIFFERENCES``.
Writes results/phase4/comparison/<candidate>_vs_<baseline>/ and never overwrites an existing one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from neuroscan_ai.evaluation.comparison import (  # noqa: E402
    ALLOWED_CONFIG_DIFFERENCES,
    config_diff,
    paired_summary,
    paired_table,
)

METRICS = (
    "dice",
    "iou",
    "sensitivity",
    "precision",
    "volume_error_ml",
    "absolute_volume_error_ml",
    "component_count_prediction",
)
SUBGROUPS = ("spacing_group", "size_group", "adc_unit_group")
NEGATIVE_METRICS = (
    "false_positive_volume_ml",
    "false_positive_component_count",
    "prediction_voxels",
)


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    rows = pd.read_csv(run_dir / "validation_metrics.csv")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if summary["data_access"]["test_cases_accessed"]:
        raise RuntimeError(f"{run_dir}: run accessed the test split; refusing to compare")
    if summary.get("mode") != "full":
        raise RuntimeError(f"{run_dir}: not a full run (mode={summary.get('mode')})")
    return rows, summary, cfg


def to_records(df: pd.DataFrame) -> list[dict]:
    return df.replace({np.nan: None}).to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("results/phase4/comparison"))
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_df, base_sum, base_cfg = load_run(args.baseline)
    cand_df, cand_sum, cand_cfg = load_run(args.candidate)

    diff = config_diff(base_cfg, cand_cfg)
    unexpected = sorted(set(diff) - ALLOWED_CONFIG_DIFFERENCES)
    if unexpected:
        raise RuntimeError(f"configs differ beyond the allowed keys: {unexpected}")
    if sorted(base_sum["evaluated_cases"]) != sorted(cand_sum["evaluated_cases"]):
        raise RuntimeError("the two runs evaluated different patients")

    out = args.out_root / f"{cand_sum['run_id']}_vs_{base_sum['run_id']}"
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier comparison

    pos_b = to_records(base_df[base_df.has_lesion])
    pos_c = to_records(cand_df[cand_df.has_lesion])
    table = paired_table(pos_b, pos_c, METRICS, keep=("gt_volume_ml", *SUBGROUPS))
    table.to_csv(out / "paired_metrics_positive.csv", index=False)

    overall = {m: paired_summary(table, m, n_boot=args.n_boot, seed=args.seed) for m in METRICS}

    subgroups: dict[str, dict] = {}
    for key in SUBGROUPS:
        subgroups[key] = {}
        for name, g in table.groupby(key):
            d = g["dice_delta"].to_numpy(float)
            subgroups[key][str(name)] = {
                "n": int(len(g)),
                "dice_mean_base": float(g["dice_base"].mean()),
                "dice_mean_cand": float(g["dice_cand"].mean()),
                "dice_mean_delta": float(d.mean()),
                "dice_median_delta": float(np.median(d)),
                "n_improved": int((d > 1e-3).sum()),
                "n_worse": int((d < -1e-3).sum()),
            }

    neg_b = to_records(base_df[~base_df.has_lesion])
    neg_c = to_records(cand_df[~cand_df.has_lesion])
    negatives = paired_table(neg_b, neg_c, NEGATIVE_METRICS, keep=("negative_outcome",))
    negatives["negative_outcome_cand"] = [
        r["negative_outcome"] for r in sorted(neg_c, key=lambda r: r["case_id"])
    ]
    negatives.to_csv(out / "paired_metrics_negative.csv", index=False)

    summary = {
        "baseline_run": base_sum["run_id"],
        "candidate_run": cand_sum["run_id"],
        "config_differences": {k: list(v) for k, v in diff.items()},
        "n_positive": int(len(table)),
        "n_negative": int(len(negatives)),
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed, "method": "percentile, mean"},
        "positive_overall": overall,
        "positive_subgroups_dice_exploratory": subgroups,
        "negative_cases": to_records(negatives),
        "training": {
            name: {
                "best_epoch": s["best_epoch"],
                "epochs_run": s["epochs_run"],
                "best_val_mean_dice": s["best_val_mean_dice"],
                "total_training_time_s": s["total_training_time_s"],
                "peak_vram_gb": s["peak_vram_gb"],
                "parameters": s["parameters"],
            }
            for name, s in (("baseline", base_sum), ("candidate", cand_sum))
        },
        "test_cases_accessed": 0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    thick = table["spacing_group"].eq("thick")
    ax.scatter(table.loc[~thick, "dice_base"], table.loc[~thick, "dice_cand"], label="thin", s=22)
    ax.scatter(table.loc[thick, "dice_base"], table.loc[thick, "dice_cand"], label="thick", s=22)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="2D Dice", ylabel="2.5D Dice", xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.set_title("Per-patient Dice (validation, positive)")
    ax.legend()
    ax = axes[1]
    srt = table.sort_values("dice_delta")
    colors = ["tab:green" if d > 0 else "tab:red" for d in srt["dice_delta"]]
    ax.bar(range(len(srt)), srt["dice_delta"], color=colors)
    ax.axhline(0, color="k", lw=0.8)
    s = overall["dice"]
    ax.set_title(
        f"dDice (2.5D - 2D): mean {s['mean_delta']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
    )
    ax.set(xlabel="patients (sorted)", ylabel="dDice")
    fig.tight_layout()
    fig.savefig(out / "dice_comparison.png", dpi=130)

    for m in ("dice", "precision", "sensitivity", "iou"):
        s = overall[m]
        p = "n/a" if s["wilcoxon_p"] is None else f"{s['wilcoxon_p']:.3g}"
        print(
            f"{m:12s} 2D {s['mean_base']:.4f}  2.5D {s['mean_cand']:.4f}  "
            f"delta {s['mean_delta']:+.4f} CI95 [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  "
            f"Wilcoxon p {p}  +{s['n_improved']}/-{s['n_worse']}/={s['n_tied']}"
        )
    print(f"written: {out}")


if __name__ == "__main__":
    main()
