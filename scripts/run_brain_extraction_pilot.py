"""Brain-extraction pilot CLI (framework stage: no external tool is installed or executed).

  .venv\\Scripts\\python.exe scripts\\run_brain_extraction_pilot.py specs
      print the SynthStrip / HD-BET command specifications and the audit fields (no execution)

  .venv\\Scripts\\python.exe scripts\\run_brain_extraction_pilot.py run --tool TOOL --alias A
      --trace TRACE.nii.gz [--gt GT.nii.gz] [--out results/brain_extraction_pilot]
      runs one case; external tools are refused in this stage (no executor is configured)

Console output names aliases only. Exit codes: 0 ok, 2 refused, 3 run finished with status fail.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

from neuroscan_ai.brain_extraction.interfaces import (  # noqa: E402
    STATUS_FAIL,
    TOOL_V1,
    TOOLS,
    BrainExtractionError,
    RunRequest,
)
from neuroscan_ai.brain_extraction.runners import HDBetSpec, SynthStripSpec, make_runner  # noqa: E402

OUTPUT_ROOTS = (
    "results/brain_extraction_pilot",
    "data/interim/soop_brain_masks",
    "data/interim/isles_brain_masks",
)


def git_ignored(path: Path) -> bool:
    try:
        rel = (path.resolve() / "probe.nii.gz").relative_to(REPO_ROOT)
    except ValueError:
        return False
    return (
        subprocess.run(["git", "check-ignore", "-q", rel.as_posix()], cwd=REPO_ROOT).returncode == 0
    )


def specs() -> int:
    ss, hb = SynthStripSpec(), HDBetSpec()
    print(json.dumps({
        "synthstrip": {"spec": asdict(ss), "command": ss.command("<trace>", "<mask>")},
        "hd_bet": {"spec": asdict(hb), "command": hb.command("<trace_std>", "<out>"),
                   "mask_output": hb.mask_output("<out>.nii.gz")},
        "executed": False,
    }, indent=1))  # fmt: skip
    return 0


def run(a: argparse.Namespace) -> int:
    out = Path(a.out)
    if not git_ignored(out):
        print("REFUSED: output folder is not git-ignored")
        return 2
    if a.tool != TOOL_V1:
        print("REFUSED: external tool execution is not enabled in this stage")
        return 2
    try:
        req = RunRequest(a.alias, Path(a.trace), Path(a.gt) if a.gt else None, a.tool,
                         "v1-frozen", {}, out)  # fmt: skip
        res = make_runner(a.tool).run(req)
    except BrainExtractionError as e:
        print(f"REFUSED: {e}")
        return 2
    c = res.qa["classification"]
    fails, reviews = c["fail_reasons"], c["review_reasons"]
    print(f"{res.alias} {res.tool}: {res.status} fail={fails} review={reviews}")
    return 3 if res.status == STATUS_FAIL else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("specs")
    r = sub.add_parser("run")
    r.add_argument("--tool", choices=TOOLS, required=True)
    r.add_argument("--alias", required=True)
    r.add_argument("--trace", required=True)
    r.add_argument("--gt")
    r.add_argument("--out", default=str(REPO_ROOT / OUTPUT_ROOTS[0]))
    a = p.parse_args(argv)
    return specs() if a.cmd == "specs" else run(a)


if __name__ == "__main__":
    sys.exit(main())
