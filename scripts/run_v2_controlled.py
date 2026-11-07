"""Controlled v2 run CLI (current stage: synthetic MockRunner only; no model, no image input).

  .venv\\Scripts\\python.exe scripts\\run_v2_controlled.py --plan PLAN.json --mock

PLAN.json: run_id, purpose, stage, dataset_id, split_name, aliases, model_id, checkpoint, config,
split_file (repo-relative paths), policies {preprocessing, inference, threshold, brain_mask,
combination}, expected {config_sha256_lf, split_sha256_lf, checkpoint_sha256,
package_lock_sha256_lf, alias_list_sha256, environment?, determinism?}, approved_splits?.
Outputs go to the git-ignored results/v2_runs/<run_id>/; an existing run_id is refused.
Without --mock the command refuses: no real v2 runner is connected yet.
Exit codes: 0 completed, 1 run failed (post-run manifest written), 2 refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

from neuroscan_ai.reproducibility import PathRoots  # noqa: E402
from neuroscan_ai.reproducibility.environment import (  # noqa: E402
    capture_determinism,
    capture_environment,
    capture_source_control,
)
from neuroscan_ai.run_control import (  # noqa: E402
    COMPLETED,
    MockRunner,
    RunPlanError,
    build_plan,
    run_controlled,
)

OUTPUT_ROOT = REPO_ROOT / "results" / "v2_runs"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--mock", action="store_true", help="use the synthetic MockRunner (required)")
    args = p.parse_args(argv)
    if not args.mock:
        print("REFUSED: no real v2 runner is connected; use --mock", file=sys.stderr)
        return 2
    try:
        raw = json.loads(args.plan.read_text(encoding="utf-8"))
        for k in ("checkpoint", "config", "split_file"):
            raw[k] = REPO_ROOT / raw[k]
        raw["output_root"] = OUTPUT_ROOT
        raw["allowed_case_count"] = len(raw.get("aliases", []))
        plan = build_plan(**raw)
        env = capture_environment(lock_file=REPO_ROOT / "requirements-lock.txt")
        det = capture_determinism(
            None, seed=0, precision=None, amp=None, workers=0, batch_size=None
        )
        res = run_controlled(
            plan,
            MockRunner(),
            repo_root=REPO_ROOT,
            roots=PathRoots(REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "data"),
            source_control=capture_source_control(REPO_ROOT),
            environment=env,
            determinism=det,
        )
    except (RunPlanError, OSError, ValueError, TypeError) as e:
        print(f"REFUSED: {type(e).__name__ if isinstance(e, OSError) else e}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1))
    if res.get("pre_run_status") != "approved":
        return 2
    return 0 if res["status"] == COMPLETED else 1


if __name__ == "__main__":
    sys.exit(main())
