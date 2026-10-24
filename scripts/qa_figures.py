"""Qualitative QA figures for a finished validation run (original patient space).

Picks best / median / worst Dice, the smallest lesion, a thick-slice case, the case with the
largest false-positive volume and the negative patient; shows DWI, ADC, GT, prediction and an
overlay on the slice with the most GT (or, without GT, the most predicted) voxels.
Only pseudonymous dataset case IDs are shown.

Usage:
    .venv\\Scripts\\python.exe scripts\\qa_figures.py --run results\\phase3\\<run_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from neuroscan_ai.preprocessing.mask import binarize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def pick_cases(df: pd.DataFrame) -> dict[str, str]:
    pos = df[df.has_lesion].sort_values("dice")
    picks = {
        "best_dice": pos.iloc[-1].case_id,
        "median_dice": pos.iloc[len(pos) // 2].case_id,
        "worst_dice": pos.iloc[0].case_id,
        "smallest_lesion": pos.sort_values("gt_volume_ml").iloc[0].case_id,
        "most_false_positive": df.sort_values("fp_voxels").iloc[-1].case_id,
    }
    thick = pos[pos.spacing_group == "thick"] if "spacing_group" in pos else pos.iloc[:0]
    if len(thick):
        picks["thick_slice"] = thick.sort_values("dice").iloc[len(thick) // 2].case_id
    neg = df[~df.has_lesion]
    if len(neg):
        picks["negative_patient"] = neg.iloc[0].case_id
    return picks


def canonical(path: Path) -> np.ndarray:
    return np.asanyarray(nib.as_closest_canonical(nib.load(path)).dataobj).astype(np.float32)


def figure(case_id: str, label: str, row: pd.Series, processed: Path, run: Path, out: Path) -> None:
    meta = json.loads((processed / case_id / "meta.json").read_text(encoding="utf-8"))
    dwi = canonical(REPO_ROOT / meta["source"]["dwi"])
    adc = canonical(REPO_ROOT / meta["source"]["adc"])
    gt = binarize(canonical(REPO_ROOT / meta["source"]["label"]))
    pred = canonical(run / "predictions" / f"{case_id}_pred.nii.gz") > 0
    key = gt if gt.any() else pred
    z = int(np.argmax(key.sum(axis=(0, 1)))) if key.any() else gt.shape[2] // 2

    def show(ax, img, title, cmap="gray"):
        ax.imshow(np.rot90(img), cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    d = dwi[:, :, z]
    show(axes[0], d, "DWI")
    show(axes[1], adc[:, :, z], "ADC")
    show(axes[2], gt[:, :, z], "Ground truth")
    show(axes[3], pred[:, :, z], "Prediction")
    rgb = np.repeat(((d - d.min()) / (np.ptp(d) + 1e-8))[..., None], 3, axis=2)
    g, p = gt[:, :, z] > 0, pred[:, :, z]
    rgb[g & ~p] = (0.0, 0.8, 0.0)  # GT only (FN)
    rgb[p & ~g] = (0.9, 0.1, 0.1)  # prediction only (FP)
    rgb[g & p] = (1.0, 0.9, 0.0)  # overlap (TP)
    show(axes[4], rgb, "Overlay: TP yellow / FN green / FP red")
    dice = "n/a (negative)" if pd.isna(row.dice) else f"{row.dice:.3f}"
    fig.suptitle(
        f"{label}: {case_id} | slice {z} | Dice {dice} | GT {row.gt_volume_ml:.2f} mL | "
        f"pred {row.prediction_volume_ml:.2f} mL | z-spacing {row.z_spacing:.1f} mm",
        fontsize=10,
    )
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{label}_{case_id}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--processed", type=Path, default=Path("data/processed/v1"))
    args = parser.parse_args()
    df = pd.read_csv(args.run / "validation_metrics.csv")
    picks = pick_cases(df)
    for label, cid in picks.items():
        figure(
            cid,
            label,
            df.set_index("case_id").loc[cid],
            args.processed,
            args.run,
            args.run / "figures",
        )
    (args.run / "figures" / "selection.json").write_text(json.dumps(picks, indent=2))
    print(json.dumps(picks, indent=2))


if __name__ == "__main__":
    main()
