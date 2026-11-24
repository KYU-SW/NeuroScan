"""v2 2.5D baseline training CLI (manifest-gated).

  validate   config / policy / architecture / hyperparameter checks only; no data, no model.
  run        one formal run (results/v2_runs/ID/), then verify_run. Refused while the config
             ``batch_policy.batch_status`` is ``candidate``; smoke run IDs are refused.
  smoke      one approved smoke run: only v2-25d-smoke-seed42-a / -b, seed 42, 2 epochs, all 176
             train / 37 validation cases. A/B exact-match check; never a performance result.
  resume     explicit resume of an interrupted run (``--smoke`` for a smoke run). Never automatic.

Arguments (run / smoke / resume): --run-id ID --seed N --alias-map PATH --split PATH
--package-lock-sha256 HEX --cache-dir DIR [--device cuda]. The seed is always explicit (42 / 43 /
44). The alias map and split must exist under results/v2_runs/_internal/, the cache under
results/v2_cache/, built from this config at the current commit. Dirty trees, other devices than
cuda, retired run IDs and Test / Final Test / SOOP cases are refused before any data access.
Every command except ``validate`` is a separately approved step.
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
from neuroscan_ai.v2.baseline_25d_runner import (  # noqa: E402
    PROFILE_SMOKE,
    SMOKE_COUNTS,
    SMOKE_EPOCH_LIMIT,
    check_device,
    check_run_profile,
    check_seed,
    prepare_config,
    resume_baseline,
    run_baseline,
)
from neuroscan_ai.v2.baseline_cache import CacheCommitError  # noqa: E402
from neuroscan_ai.v2.baseline_resume import RETIRED_RUN_IDS  # noqa: E402
from neuroscan_ai.v2.baseline_training import (  # noqa: E402
    DEVICES,
    PROFILE_FORMAL,
    ConfigValidationError,
    parse_seed_text,
)
from neuroscan_ai.v2.dataset import isles_case_files  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "v2" / "25d_baseline.yaml"
OUT_ROOT = REPO_ROOT / "results" / "v2_runs"
INTERNAL = OUT_ROOT / "_internal"
CACHE_ROOT = REPO_ROOT / "results" / "v2_cache"
RAW_ROOT = REPO_ROOT / "data" / "raw" / "ISLES2022"


def validate() -> int:
    cfg = prepare_config(CONFIG)
    print(json.dumps({"config": "ok", "batch_policy": cfg["batch_policy"]}))
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
    seed_text: str | None = None,
    smoke: bool = False,
) -> int:
    if run_id in RETIRED_RUN_IDS:
        return _refuse("run ID is retired (never reused, never resumed)")
    cfg = prepare_config(CONFIG)  # refused here (before any plan / gate) on a config mismatch
    try:
        profile = PROFILE_SMOKE if smoke else PROFILE_FORMAL
        limit = SMOKE_EPOCH_LIMIT if smoke else None
        check_device(profile, cfg["training"]["device"], device, limit)
        seed = check_seed(profile, parse_seed_text(seed_text))
        check_run_profile(  # the actual case counts are checked again from the RunPlan
            profile,
            run_id=run_id,
            seed=seed,
            epoch_limit=limit,
            batch_status=cfg["batch_policy"]["batch_status"],
            counts=SMOKE_COUNTS,
        )
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
        "purpose": "v2 2.5D baseline smoke (A/B exact match only)"
        if smoke
        else "v2 2.5D baseline training",
        "stage": "train",
        "dataset_id": cfg["dataset_id"],
        "split_name": "train",
        "aliases": splits["train"],
        "allowed_case_count": len(splits["train"]),
        "validation_aliases": splits["val"],
        "input_files": {a: {"dwi": f.dwi, "adc": f.adc, "gt": f.label} for a, f in files.items()},
        "checkpoint": None,
        "seed": seed,
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
        "seed": seed,
        "epoch_limit": limit,
        "smoke": smoke,
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
    for name in ("run", "smoke", "resume"):
        r = sub.add_parser(name)
        r.add_argument("--run-id", required=True)
        # parsed by baseline_training.parse_seed_text (refusal message + exit code 2, not argparse)
        r.add_argument("--seed", default=None)
        r.add_argument("--alias-map", required=True, type=Path)
        r.add_argument("--split", required=True, type=Path)
        r.add_argument("--package-lock-sha256", required=True)
        r.add_argument("--cache-dir", required=True, type=Path)
        r.add_argument("--device", choices=DEVICES, default=None)
        if name == "resume":
            r.add_argument("--smoke", action="store_true", help="resume an approved smoke run")
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
        seed_text=a.seed,
        smoke=a.cmd == "smoke" or bool(getattr(a, "smoke", False)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
