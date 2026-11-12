"""Build the v2 2D baseline disk cache ``disk_backed_preprocessed_cache_v1`` (one new version).

  python scripts/prepare_v2_2d_cache.py --alias-map PATH --split PATH \\
      --package-lock-sha256 HEX --cache-dir results/v2_cache/2d_baseline_v1

Reads the approved Train / Validation cases (read-only), prepares them one case at a time with
``preprocess_v2_isles_1mm_inplane_v1`` and writes ``results/v2_cache/<version>/`` (git-ignored).
Refused before any data access when: the config differs from code, the package lock is not the
approved full SHA-256, the alias map / split are not under results/v2_runs/_internal/, the cache
folder is not under results/v2_cache/ or already exists (never overwritten; a different policy or
input needs a new version folder), or the working tree is not clean. Test, Final Test and SOOP
cases are never read: only the split's ``train`` and ``val`` members are used.
Running this script reads real MRI and is a separately approved step.
Exit codes: 0 built, 1 build failed (nothing published), 2 refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

from neuroscan_ai.reproducibility.environment import capture_source_control  # noqa: E402
from neuroscan_ai.reproducibility.hashing import sha256_text_lf  # noqa: E402
from neuroscan_ai.v2.baseline_cache import CacheError, build_cache  # noqa: E402
from neuroscan_ai.v2.baseline_runner import prepare_config  # noqa: E402
from neuroscan_ai.v2.dataset import isles_case_files  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "v2" / "2d_baseline.yaml"
INTERNAL = REPO_ROOT / "results" / "v2_runs" / "_internal"
CACHE_ROOT = REPO_ROOT / "results" / "v2_cache"
RAW_ROOT = REPO_ROOT / "data" / "raw" / "ISLES2022"
FORBIDDEN_SPLITS = ("test", "final_test")


def _refuse(msg: str) -> int:
    print(f"REFUSED: {msg}", file=sys.stderr)
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alias-map", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--package-lock-sha256", required=True)
    ap.add_argument("--cache-dir", required=True, type=Path)
    a = ap.parse_args()
    cfg = prepare_config(CONFIG)
    lock = REPO_ROOT / "requirements-lock.txt"
    lock_hash = a.package_lock_sha256.lower()
    if len(lock_hash) != 64 or not lock.is_file() or sha256_text_lf(lock) != lock_hash:
        return _refuse("package lock missing or not the approved full SHA-256")
    for p in (a.alias_map, a.split):
        if not p.resolve().is_relative_to(INTERNAL.resolve()) or not p.is_file():
            return _refuse("alias map and split must exist under results/v2_runs/_internal/")
    out = a.cache_dir.resolve()
    if not out.is_relative_to(CACHE_ROOT.resolve()) or out == CACHE_ROOT.resolve():
        return _refuse("cache folder must be a version folder under results/v2_cache/")
    if out.exists():
        return _refuse("cache version exists; never overwritten (use a new version)")
    sc = capture_source_control(REPO_ROOT)
    if sc["working_tree_clean"] is not True:
        return _refuse("working tree not clean")
    amap = json.loads(a.alias_map.read_text(encoding="utf-8"))["map"]
    splits = json.loads(a.split.read_text(encoding="utf-8"))["splits"]
    if set(splits) & set(FORBIDDEN_SPLITS) or not {"train", "val"} <= set(splits):
        return _refuse("split must hold train and val only")
    files = {
        s: {al: isles_case_files(RAW_ROOT, amap[al]) for al in splits[s]} for s in ("train", "val")
    }
    try:
        summary = build_cache(
            out,
            config=cfg,
            split_file=a.split,
            files=files,
            git_commit=sc.get("git_commit"),
            package_lock_sha256_lf=lock_hash,
            forbidden_ids=tuple(amap.values()),
        )
    except (CacheError, ValueError) as e:
        print(f"FAILED: {type(e).__name__} (nothing published)", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {k: summary[k] for k in ("cache_version", "cases", "counts", "cache_index_sha256")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
