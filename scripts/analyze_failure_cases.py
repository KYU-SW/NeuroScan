"""NeuroScan v1 failure-case analysis — metrics-only stage (no image, no model, no NIfTI).

  .venv\\Scripts\\python.exe scripts\\analyze_failure_cases.py
      [--final-test-dir results/final_test/v1] [--output-dir results/failure_analysis/v1]

Reads the completed Final Test result tables read-only, selects review cases (groups F, Z, I, W,
N, G; docs/failure_analysis_protocol.md), assigns local aliases fa-01.. and writes summaries.
Console output contains counts only, never a case ID. Exit codes: 0 ok, 2 refused.
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
    p.add_argument("--mode", choices=[sc.MODE_METRICS_ONLY], default=sc.MODE_METRICS_ONLY)
    p.add_argument("--final-test-dir", type=Path, default=REPO_ROOT / "results/final_test/v1")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/failure_analysis/v1")
    args = p.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
