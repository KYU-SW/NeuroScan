"""Synthetic CUDA memory smoke of the v2 2.5D baseline (batch 32 candidate). Separately approved.

  python scripts/smoke_v2_25d_cuda_memory.py --approved

Synthetic input [32, 6, 264, 264] only: no MRI, no cache, no split file. Two passes of 3 training
steps (see ``neuroscan_ai.v2.cuda_smoke_25d``). Refused before anything is created when
``--approved`` is missing, the config is invalid, the working tree is not clean, CUDA is not
available or the output folder exists. The record goes to the git-ignored folder
results/v2_smoke/25d_cuda_memory_v1/. A failure is recorded and never retried; batch 16 is never
tried automatically. Exit codes: 0 passed, 1 failed, 2 refused.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch / CUDA initialise

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

import torch  # noqa: E402

from neuroscan_ai.reproducibility.environment import (  # noqa: E402
    capture_environment,
    capture_source_control,
)
from neuroscan_ai.reproducibility.hashing import canonical_json_sha256  # noqa: E402
from neuroscan_ai.v2.baseline_25d_runner import prepare_config  # noqa: E402
from neuroscan_ai.v2.cuda_smoke_25d import run_smoke  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "v2" / "25d_baseline.yaml"
OUT_DIR = REPO_ROOT / "results" / "v2_smoke" / "25d_cuda_memory_v1"


def _refuse(msg: str) -> int:
    print(f"REFUSED: {msg}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--approved", action="store_true", help="the smoke was approved separately")
    a = ap.parse_args(argv)
    if not a.approved:
        return _refuse("separate approval required (--approved)")
    cfg = prepare_config(CONFIG)
    if OUT_DIR.exists():
        return _refuse("output folder exists; never overwritten")
    sc = capture_source_control(REPO_ROOT)
    if sc["working_tree_clean"] is not True:
        return _refuse("working tree not clean")
    if not torch.cuda.is_available():
        return _refuse("cuda is not available; no CPU fallback")
    context = {
        "git_commit": sc.get("git_commit"),
        "config_canonical_sha256": canonical_json_sha256(cfg),
        "batch_policy": cfg["batch_policy"],
        "environment": capture_environment(
            torch_module=torch, lock_file=REPO_ROOT / "requirements-lock.txt"
        ),
    }
    res = run_smoke(OUT_DIR, device=torch.device("cuda"), context=context)
    print(json.dumps(res))
    return 0 if res["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
