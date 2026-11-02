"""NeuroScan v1 Final Test evaluation (2D vs 2.5D), pre-registered in configs/final_test_v1.yaml.

Default mode is the validation dry-run: the whole procedure on the validation split, the test
split stays sealed and nothing of it is opened.

  PYTHONPATH=backend .venv\\Scripts\\python.exe scripts\\evaluate_final_test.py
      [--mode validation-dry-run] [--output-dir results/final_test_dry_run/v1] [--no-qa-images]

The final test (once, after the lock tag exists and the user approved it):

  .venv\\Scripts\\python.exe scripts\\evaluate_final_test.py --mode final-test
      --approve-test <LF-normalised SHA-256 of configs/final_test_v1.yaml>
      [--confirm-phrase "<phrase>" --non-interactive]   (otherwise the phrase is asked for)

Re-run after a technical failure only:
      ... --resume-after-failure --failure-report <file> --reason "<text>"

Every refusal happens before any file of the evaluated split is opened. Console output never
contains case IDs; in final-test mode it contains no metric either (see the output folder).
Exit codes: 0 ok, 2 pre-flight / approval refused, 3 run-state refused, 4 evaluation failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

from neuroscan_ai.final_test import manifest as mf  # noqa: E402
from neuroscan_ai.final_test import pipeline as pl  # noqa: E402
from neuroscan_ai.final_test import run_state as rs  # noqa: E402


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=pl.MODES, default="validation-dry-run")
    p.add_argument("--manifest", type=Path, default=mf.MANIFEST_PATH)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--approve-test", default=None)
    p.add_argument("--confirm-phrase", default=None)
    p.add_argument(
        "--non-interactive",
        action="store_true",
        help="do not prompt; requires --confirm-phrase (final-test mode)",
    )
    p.add_argument("--resume-after-failure", action="store_true")
    p.add_argument("--failure-report", type=Path, default=None)
    p.add_argument("--reason", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-qa-images", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    final = args.mode == "final-test"
    phrase, approved_at = args.confirm_phrase, None
    if final:
        if phrase is None and not args.non_interactive and sys.stdin.isatty():
            print(f'Type exactly "{mf.CONFIRM_PHRASE}" to approve the one-time final test:')
            phrase = input("> ").strip()
        if phrase == mf.CONFIRM_PHRASE:
            approved_at = rs.now()
    opts = pl.RunOptions(
        mode=args.mode,
        repo_root=REPO_ROOT,
        manifest=args.manifest,
        output_dir=args.output_dir,
        approve_test=args.approve_test,
        confirm_phrase=phrase,
        resume_after_failure=args.resume_after_failure,
        failure_report=args.failure_report,
        reason=args.reason,
        device=args.device,
        qa_images=not args.no_qa_images,
        command=["scripts/evaluate_final_test.py", *(argv if argv is not None else sys.argv[1:])],
        approved_at=approved_at,
    )
    try:
        result = pl.run(opts)
    except pl.PreflightError as e:
        print(f"REFUSED (nothing of the evaluated split was opened): {e}")
        for c in e.checks:
            if not c.ok:
                print(f"  - {c.name}: {c.detail}")
        return 2
    except rs.RunStateError as e:
        print(f"REFUSED (run state): {e}")
        return 3
    except Exception as e:  # noqa: BLE001
        # final-test: the type only (a message may name a case); details are in FAILED locally
        print(f"FAILED: {type(e).__name__}" + ("" if final else f": {e}"))
        return 4
    out = result["out"]
    print(f"{args.mode}: completed (attempt {result['attempt']}) -> {out.as_posix()}")
    if not final:
        p = result["report"]["stats"]["primary"]
        print(
            f"  positive n={p['n']}  Dice 2D {p['mean_base']:.4f}  2.5D {p['mean_cand']:.4f}  "
            f"delta {p['mean_delta']:+.4f}  95% CI [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
            f"  Wilcoxon p={p['wilcoxon_p']:.3f}"
        )
        m = result["match"]
        print(f"  reference match (Phase 3/4 validation): {'YES' if m['match'] else 'NO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
