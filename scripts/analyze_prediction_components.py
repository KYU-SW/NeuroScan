"""Component safety analysis CLI (stage 1: static analysis of an EXISTING prediction).

  .venv\\Scripts\\python.exe scripts\\analyze_prediction_components.py --alias A
      --prediction PRED.nii.gz --probability PROB.nii.gz --dwi DWI.nii.gz --adc ADC.nii.gz
      --brain-mask BM.nii.gz --brain-mask-policy ID --adc-storage-status S
      [--adc-content-status S] [--adc-unit-status S] [--adc-background-code V]
      [--slice-thickness MM] [--spacing-between-slices MM]
      [--pred-2d P] [--pred-25d P] [--prob-2d P] [--prob-25d P] [--gt GT.nii.gz]
      [--out results/component_analysis]

No model is run and no input is modified. All inputs must share one grid (shape and affine within
1e-3 mm). The output folder must be inside the git-ignored results/component_analysis/ and an
existing record is never overwritten. Console output names the alias only.
Exit codes: 0 ok, 2 refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from neuroscan_ai.component_analysis import ComponentAnalysisError, analyze_case  # noqa: E402
from neuroscan_ai.component_analysis.records import is_git_ignored, write_record  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "results" / "component_analysis"
AFFINE_ATOL_MM = 1e-3


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alias", required=True)
    for name in ("prediction", "probability", "dwi", "adc", "brain-mask"):
        p.add_argument(f"--{name}", required=True, type=Path)
    p.add_argument("--brain-mask-policy", required=True)
    p.add_argument("--adc-storage-status", required=True)
    p.add_argument("--adc-content-status")
    p.add_argument("--adc-unit-status")
    p.add_argument("--adc-background-code", type=float)
    p.add_argument("--slice-thickness", type=float)
    p.add_argument("--spacing-between-slices", type=float)
    for name in ("pred-2d", "pred-25d", "prob-2d", "prob-25d", "gt"):
        p.add_argument(f"--{name}", type=Path)
    p.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    return p


def _check_out(out: Path) -> Path:
    out = out if out.is_absolute() else REPO_ROOT / out
    root = OUTPUT_ROOT.resolve()
    if not (out.resolve() == root or root in out.resolve().parents):
        raise ComponentAnalysisError("output must be inside results/component_analysis/")
    if not is_git_ignored(out / "probe", REPO_ROOT):
        raise ComponentAnalysisError("output folder is not git-ignored")
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        out = _check_out(args.out)
        ref = nib.load(str(args.prediction))
        arrays = {}
        for key in (
            "prediction",
            "probability",
            "dwi",
            "adc",
            "brain_mask",
            "pred_2d",
            "pred_25d",
            "prob_2d",
            "prob_25d",
            "gt",
        ):
            path = getattr(args, key)
            if path is None:
                continue
            img = nib.load(str(path))
            if tuple(img.shape) != tuple(ref.shape) or not np.allclose(
                img.affine, ref.affine, atol=AFFINE_ATOL_MM
            ):
                raise ComponentAnalysisError(f"{key} is not on the prediction grid")
            arrays[key] = (
                np.asanyarray(img.dataobj)
                if key in ("prediction", "brain_mask", "pred_2d", "pred_25d", "gt")
                else img.get_fdata()
            )
        record = analyze_case(
            alias=args.alias,
            affine=ref.affine,
            brain_mask_policy_id=args.brain_mask_policy,
            adc_storage_status=args.adc_storage_status,
            adc_content_status=args.adc_content_status,
            adc_unit_status=args.adc_unit_status,
            adc_background_code=args.adc_background_code,
            slice_thickness_mm=args.slice_thickness,
            spacing_between_slices_mm=args.spacing_between_slices,
            **arrays,
        )
        write_record(out, record)
    except (ComponentAnalysisError, OSError) as e:
        # OSError text can carry a host path: print only its type
        print(f"REFUSED: {type(e).__name__ if isinstance(e, OSError) else e}", file=sys.stderr)
        return 2
    s = record["case"]
    print(
        json.dumps(
            {
                "alias": s["alias"],
                "components": s["component_count"],
                "warnings": s["warnings_summary"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
