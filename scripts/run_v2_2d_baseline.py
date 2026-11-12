"""v2 2D baseline training CLI (manifest-gated).

  validate            steps 1-4 only: load configs/v2/2d_baseline.yaml, policy / architecture /
                      hyperparameter / operational-block checks. Opens no data, builds no model.
  run --run-id ID --alias-map PATH --split PATH --package-lock-sha256 HEX --cache-dir DIR
      [--device cuda]
                      one manifest-gated training run (results/v2_runs/ID/), then verify_run.
                      The alias map (alias -> case id) and the alias-only split must already exist
                      under the git-ignored results/v2_runs/_internal/; the disk cache under
                      results/v2_cache/ must already be built (scripts/prepare_v2_2d_cache.py) AT
                      THE CURRENT COMMIT (checked before the gate: otherwise refused, nothing
                      created, run ID unused) and passes its integrity gate before any training.
  resume  (same arguments as run)
                      EXPLICIT resume of an interrupted run (same run ID, nothing changed since:
                      commit, clean tree, config, split, inputs, cache, package lock, environment,
                      determinism; see docs/v2_2d_baseline_run_protocol.md section 13.7). Never
                      started automatically; any mismatch is refused and nothing is written.

Formal ISLES runs require device ``cuda`` (device_policy_v2_baseline_cuda_required_v1): an override
other than ``cuda`` (``cpu``, ``cuda_if_available``) is refused before any data access, no run
folder is created and the run ID stays unused. ``--package-lock-sha256`` must be the full approved
canonical-LF SHA-256 of requirements-lock.txt; a missing lock file or another hash is refused before
any data access (the gate checks it again against the pre-run manifest).
``run`` refuses before the gate when the working tree is not clean (the gate would reject it and
the run ID would be used up). Retired run IDs (``baseline_resume.RETIRED_RUN_IDS``, e.g. the
externally terminated ``v2-2d-baseline-seed42-a``) are refused by both ``run`` and ``resume``.
Test, Final Test and SOOP cases are refused by the RunPlan.
Exit codes: 0 ok, 1 run failed, 2 refused, 3 interrupted (safe stop; resumable on request).
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
from neuroscan_ai.run_control.lifecycle import (  # noqa: E402
    INTERRUPTED_STATUSES,
    ResumeRefusedError,
)
from neuroscan_ai.v2.baseline_cache import CacheCommitError  # noqa: E402
from neuroscan_ai.v2.baseline_resume import RETIRED_RUN_IDS  # noqa: E402
from neuroscan_ai.v2.baseline_runner import (  # noqa: E402
    prepare_config,
    resume_baseline,
    run_baseline,
)
from neuroscan_ai.v2.baseline_training import (  # noqa: E402
    DEVICES,
    PROFILE_FORMAL,
    ConfigValidationError,
    check_device_policy,
)
from neuroscan_ai.v2.dataset import isles_case_files  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "v2" / "2d_baseline.yaml"
OUT_ROOT = REPO_ROOT / "results" / "v2_runs"
INTERNAL = OUT_ROOT / "_internal"
CACHE_ROOT = REPO_ROOT / "results" / "v2_cache"
RAW_ROOT = REPO_ROOT / "data" / "raw" / "ISLES2022"


def validate() -> int:
    cfg = prepare_config(CONFIG)
    print(json.dumps({"config": "ok", "batch_size": cfg["training"]["batch_size"]}))
    return 0


def _refuse(msg: str) -> int:
    print(f"REFUSED: {msg}", file=sys.stderr)
    return 2


def run(
    run_id: str,
    alias_map: Path,
    split: Path,
    lock_sha256: str,
    cache_dir: Path,
    device: str | None = None,
    resume: bool = False,
) -> int:
    if run_id in RETIRED_RUN_IDS:
        return _refuse("run ID is retired (never reused, never resumed)")
    cfg = prepare_config(CONFIG)  # refused here (before any plan / gate) on a config mismatch
    try:
        check_device_policy(PROFILE_FORMAL, cfg["training"]["device"], device, None)
    except ConfigValidationError as e:
        return _refuse(str(e))
    lock = REPO_ROOT / "requirements-lock.txt"
    if len(lock_sha256) != 64 or not lock.is_file() or sha256_text_lf(lock) != lock_sha256.lower():
        return _refuse("package lock missing or not the approved full SHA-256")
    for p in (alias_map, split):
        if not p.resolve().is_relative_to(INTERNAL.resolve()) or not p.is_file():
            return _refuse("alias map and split must exist under results/v2_runs/_internal/")
    cache = cache_dir.resolve()
    if not cache.is_relative_to(CACHE_ROOT.resolve()) or not cache.is_dir():
        return _refuse("cache must be an existing folder under results/v2_cache/")
    sc = capture_source_control(REPO_ROOT)
    if sc["working_tree_clean"] is not True:
        return _refuse("working tree not clean (run ID kept unused)")
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
    plan_kwargs = {
        "run_id": run_id,
        "purpose": "v2 2D baseline training",
        "stage": "train",
        "dataset_id": cfg["dataset_id"],
        "split_name": "train",
        "aliases": splits["train"],
        "allowed_case_count": len(splits["train"]),
        "validation_aliases": splits["val"],
        "input_files": {a: {"dwi": f.dwi, "adc": f.adc, "gt": f.label} for a, f in files.items()},
        "checkpoint": None,
        "split_file": split,
        "output_root": OUT_ROOT,
        "expected": {
            "config_sha256_lf": sha256_text_lf(CONFIG),
            "split_sha256_lf": sha256_text_lf(split),
            "package_lock_sha256_lf": env["package_lock_sha256_lf"],
            "alias_list_sha256": alias_list_sha256(aliases, CASE_KEY_POLICY),
        },
    }
    common = {
        "repo_root": REPO_ROOT,
        "source_control": sc,
        "environment": env,
        "determinism": det,
        "forbidden_ids": tuple(amap.values()),
        "device": device,
    }
    if resume:
        try:
            res = resume_baseline(CONFIG, plan_kwargs, cache, **common)
        except ResumeRefusedError as e:
            return _refuse(", ".join(e.reasons))
    else:
        try:
            res = run_baseline(
                CONFIG,
                plan_kwargs,
                cache,
                roots=PathRoots(REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "data"),
                **common,
            )
        except CacheCommitError as e:  # before the gate: no run folder, run ID unused
            return _refuse(", ".join(e.reasons))
        if res.get("pre_run_status") != "approved":
            print(json.dumps(res))
            return 2
    if res["status"] in INTERRUPTED_STATUSES:
        print(json.dumps({"status": res["status"], "resume": "allowed on explicit request"}))
        return 3
    ver = verify_run(OUT_ROOT / run_id)
    print(json.dumps({"status": res["status"], "verify_ok": ver["ok"]}))
    return 0 if res["status"] == COMPLETED and ver["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    for name in ("run", "resume"):
        r = sub.add_parser(name)
        r.add_argument("--run-id", required=True)
        r.add_argument("--alias-map", required=True, type=Path)
        r.add_argument("--split", required=True, type=Path)
        r.add_argument("--package-lock-sha256", required=True)
        r.add_argument("--cache-dir", required=True, type=Path)
        r.add_argument("--device", choices=DEVICES, default=None)
    a = ap.parse_args()
    if a.cmd == "validate":
        return validate()
    return run(
        a.run_id,
        a.alias_map,
        a.split,
        a.package_lock_sha256,
        a.cache_dir,
        a.device,
        resume=a.cmd == "resume",
    )


if __name__ == "__main__":
    raise SystemExit(main())
