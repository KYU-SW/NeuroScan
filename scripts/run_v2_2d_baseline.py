"""v2 2D baseline training CLI (manifest-gated).

  validate            steps 1-4 only: load configs/v2/2d_baseline.yaml, policy / architecture /
                      hyperparameter checks. Opens no data and builds no model.
  run --run-id ID --alias-map PATH --split PATH
                      one manifest-gated training run (results/v2_runs/ID/), then verify_run.
                      The alias map (alias -> case id) and the alias-only split must already exist
                      under the git-ignored results/v2_runs/_internal/; preparing them for the real
                      ISLES train / val data is a separate, approved step (not part of this CLI).

``run`` refuses before the gate when the working tree is not clean (the gate would reject it and
the run ID would be used up). Test, Final Test and SOOP cases are refused by the RunPlan.
Exit codes: 0 ok, 1 run failed, 2 refused.
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

from neuroscan_ai.reproducibility import CASE_KEY_POLICY, PathRoots  # noqa: E402
from neuroscan_ai.reproducibility.environment import (  # noqa: E402
    capture_determinism,
    capture_environment,
    capture_source_control,
)
from neuroscan_ai.reproducibility.hashing import alias_list_sha256, sha256_text_lf  # noqa: E402
from neuroscan_ai.run_control import COMPLETED, verify_run  # noqa: E402
from neuroscan_ai.v2.baseline_runner import prepare_config, run_baseline  # noqa: E402
from neuroscan_ai.v2.dataset import isles_case_files  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "v2" / "2d_baseline.yaml"
OUT_ROOT = REPO_ROOT / "results" / "v2_runs"
INTERNAL = OUT_ROOT / "_internal"
RAW_ROOT = REPO_ROOT / "data" / "raw" / "ISLES2022"


def validate() -> int:
    cfg = prepare_config(CONFIG)
    print(json.dumps({"config": "ok", "batch_size": cfg["training"]["batch_size"]}))
    return 0


def run(run_id: str, alias_map: Path, split: Path) -> int:
    cfg = prepare_config(CONFIG)  # refused here (before any plan / gate) on a config mismatch
    for p in (alias_map, split):
        if not p.resolve().is_relative_to(INTERNAL.resolve()) or not p.is_file():
            print("REFUSED: alias map and split must exist under results/v2_runs/_internal/")
            return 2
    sc = capture_source_control(REPO_ROOT)
    if sc["working_tree_clean"] is not True:
        print("REFUSED: working tree not clean (run ID kept unused)", file=sys.stderr)
        return 2
    amap = json.loads(alias_map.read_text(encoding="utf-8"))["map"]
    splits = json.loads(split.read_text(encoding="utf-8"))["splits"]
    files = {a: isles_case_files(RAW_ROOT, cid) for a, cid in amap.items()}
    tr = cfg["training"]
    env = capture_environment(torch_module=torch, lock_file=REPO_ROOT / "requirements-lock.txt")
    det = capture_determinism(
        torch,
        seed=None,
        precision=tr["precision"],
        amp=tr["amp"],
        workers=tr["num_workers"],
        batch_size=tr["batch_size"],
    )
    aliases = splits["train"] + splits["val"]
    res = run_baseline(
        CONFIG,
        {
            "run_id": run_id,
            "purpose": "v2 2D baseline training",
            "stage": "train",
            "dataset_id": cfg["dataset_id"],
            "split_name": "train",
            "aliases": splits["train"],
            "allowed_case_count": len(splits["train"]),
            "validation_aliases": splits["val"],
            "input_files": {
                a: {"dwi": f.dwi, "adc": f.adc, "gt": f.label} for a, f in files.items()
            },
            "checkpoint": None,
            "split_file": split,
            "output_root": OUT_ROOT,
            "expected": {
                "config_sha256_lf": sha256_text_lf(CONFIG),
                "split_sha256_lf": sha256_text_lf(split),
                "package_lock_sha256_lf": env["package_lock_sha256_lf"],
                "alias_list_sha256": alias_list_sha256(aliases, CASE_KEY_POLICY),
            },
        },
        files,
        repo_root=REPO_ROOT,
        roots=PathRoots(REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "data"),
        source_control=sc,
        environment=env,
        determinism=det,
        forbidden_ids=tuple(amap.values()),
    )
    if res.get("pre_run_status") != "approved":
        print(json.dumps(res))
        return 2
    ver = verify_run(OUT_ROOT / run_id)
    print(json.dumps({"status": res["status"], "verify_ok": ver["ok"]}))
    return 0 if res["status"] == COMPLETED and ver["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    r = sub.add_parser("run")
    r.add_argument("--run-id", required=True)
    r.add_argument("--alias-map", required=True, type=Path)
    r.add_argument("--split", required=True, type=Path)
    a = ap.parse_args()
    if a.cmd == "validate":
        return validate()
    return run(a.run_id, a.alias_map, a.split)


if __name__ == "__main__":
    raise SystemExit(main())
