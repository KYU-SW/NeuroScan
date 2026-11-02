"""NeuroScan v1 failure-case analysis — metrics-only stage and approved image review.

  .venv\\Scripts\\python.exe scripts\\analyze_failure_cases.py
      [--final-test-dir results/final_test/v1] [--output-dir results/failure_analysis/v1]

  image review of the already selected aliases only (read-only DWI / ADC / GT / stored
  predictions, results in <output-dir>/image_review; refused without the exact phrase):
      ... --mode image-review --approve-image-review "<approval phrase>"

Reads the completed Final Test result tables read-only, selects review cases (groups F, Z, I, W,
N, G; docs/failure_analysis_protocol.md), assigns local aliases fa-01.. and writes summaries.
Console output contains counts only, never a case ID. Exit codes: 0 ok, 2 refused.
No mode runs a model, creates a probability map or writes into the Final Test folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

from neuroscan_ai.failure_analysis import schemas as sc  # noqa: E402
from neuroscan_ai.failure_analysis.run import FailureAnalysisError, run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode", choices=[sc.MODE_METRICS_ONLY, "image-review"], default=sc.MODE_METRICS_ONLY
    )
    p.add_argument("--approve-image-review", default=None, metavar="PHRASE")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--final-test-dir", type=Path, default=REPO_ROOT / "results/final_test/v1")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/failure_analysis/v1")
    args = p.parse_args(argv)
    if args.mode == "image-review":
        return image_review(args)
    try:
        res = run(args.final_test_dir, args.output_dir, REPO_ROOT)
    except FailureAnalysisError as e:
        print(f"REFUSED: {e}")
        return 2
    agg, audit = res["aggregate"], res["audit"]
    print(f"metrics-only: {agg['unique_aliases']} aliases -> {res['out'].as_posix()}")
    print("  groups: " + ", ".join(f"{g}={n}" for g, n in agg["group_counts"].items()))
    for m in sc.MODELS:
        t = agg["taxonomy_counts"][m]
        print(f"  {m}: " + ", ".join(f"{c} flagged={t[c][sc.FLAGGED]}" for c in t))
    print(f"  T8 review candidates: {agg['t8_counts'][sc.REVIEW_CANDIDATE]}")
    print(f"  pending image review: {agg['pending_image_review_counts']}")
    print(
        f"  NIfTI opened: 0 (refused attempts {audit['access']['nifti_open_attempts_refused']}), "
        f"final test unchanged: {audit['final_test_result']['unchanged_after_run']}, "
        f"ID scan passed: {audit['public_output_original_id_scan']['passed']}"
    )
    return 0


def image_review(args) -> int:
    from neuroscan_ai.failure_analysis.image_run import ImageReviewRefused
    from neuroscan_ai.failure_analysis.image_run import run as run_images

    try:
        res = run_images(
            args.output_dir,
            args.final_test_dir,
            REPO_ROOT,
            args.approve_image_review,
            figures=not args.no_figures,
        )
    except ImageReviewRefused as e:
        print(f"REFUSED: {e}")
        return 2
    except Exception as e:  # noqa: BLE001 - never echo a message that could name a case
        print(f"FAILED: {type(e).__name__}")
        return 4
    a = res["audit"]
    print(
        f"image-review: {a['aliases_analyzed']}/{a['aliases_accessed']} aliases analysed "
        f"-> {res['out'].as_posix()}"
    )
    print(f"  files opened: {a['files_opened']}, refused attempts: {a['refused_file_attempts']}")
    print(
        f"  maps: {a['difference_maps_written']}, figures: {a['figures_written']}, "
        f"stopped: {len(a['aliases_stopped'])}"
    )
    print(
        f"  final test unchanged: {a['final_test_unchanged']}, metrics-only unchanged: "
        f"{a['metrics_only_files_unchanged']}, ID/path scan passed: "
        f"{a['public_output_original_id_scan']['passed']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
