"""v2 2D smoke test CLI (NOT a performance experiment).

  prepare             select 4 train + 2 val ISLES cases (SHA-256(salt + case_id), smallest first)
                      from the v1 split (read-only) and write the alias map and an alias-only smoke
                      split under the git-ignored results/v2_runs/_internal/
  run --run-id ID     one manifest-gated run (results/v2_runs/ID/), then verify_run
  compare A B         compare two runs (manifests and output hashes)

The test split is never a candidate; SOOP cases are never read. ``run`` refuses before the gate when
the working tree is not clean (the gate would reject it and the run ID would be used up).
Exit codes: 0 ok / exact, 1 run failed or not exact, 2 refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch / CUDA initialise

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

import torch  # noqa: E402
import yaml  # noqa: E402

from neuroscan_ai.reproducibility import CASE_KEY_POLICY, PathRoots, compare_manifests  # noqa: E402
from neuroscan_ai.reproducibility.environment import (  # noqa: E402
    capture_determinism,
    capture_environment,
    capture_source_control,
)
from neuroscan_ai.reproducibility.hashing import alias_list_sha256, sha256_text_lf  # noqa: E402
from neuroscan_ai.run_control import COMPLETED, build_plan, run_controlled, verify_run  # noqa: E402
from neuroscan_ai.v2 import MODEL_ID, V2TwoDRunner  # noqa: E402
from neuroscan_ai.v2.dataset import isles_case_files  # noqa: E402
from neuroscan_ai.v2.training import set_determinism  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "v2" / "2d_smoke.yaml"
OUT_ROOT = REPO_ROOT / "results" / "v2_runs"
INTERNAL = OUT_ROOT / "_internal"
ALIAS_MAP = INTERNAL / "v2_2d_smoke_aliases.json"
SMOKE_SPLIT = INTERNAL / "v2_2d_smoke_split.json"
RAW_ROOT = REPO_ROOT / "data" / "raw" / "ISLES2022"
SOOP = re.compile(r"soop", re.I)


def _cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _write_or_match(path: Path, obj: dict) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit("REFUSED: existing smoke selection differs; not overwriting")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare() -> int:
    cfg = _cfg()
    d = cfg["data"]
    splits = json.loads((REPO_ROOT / d["split_source"]).read_text(encoding="utf-8"))["splits"]

    def pick(name: str, n: int) -> list[str]:
        cands = [c for c in splits[name] if not SOOP.search(c)]
        return sorted(
            cands, key=lambda c: hashlib.sha256((d["selection_salt"] + c).encode()).hexdigest()
        )[:n]

    train, val = pick("train", d["n_train"]), pick("val", d["n_val"])
    amap = {f"{d['train_alias_prefix']}{i:02d}": c for i, c in enumerate(train, 1)}
    amap.update({f"{d['val_alias_prefix']}{i:02d}": c for i, c in enumerate(val, 1)})
    _write_or_match(ALIAS_MAP, {"selection": "sha256(salt+case_id) ascending", "map": amap})
    _write_or_match(
        SMOKE_SPLIT,
        {
            "splits": {
                "train": sorted(a for a in amap if a.startswith(d["train_alias_prefix"])),
                "val": sorted(a for a in amap if a.startswith(d["val_alias_prefix"])),
            }
        },
    )
    print(
        json.dumps(
            {
                "train_cases": len(train),
                "val_cases": len(val),
                "test_candidates": 0,
                "written": [p.relative_to(REPO_ROOT).as_posix() for p in (ALIAS_MAP, SMOKE_SPLIT)],
            }
        )
    )
    return 0


def run(run_id: str) -> int:
    cfg = _cfg()
    if not (ALIAS_MAP.is_file() and SMOKE_SPLIT.is_file()):
        print("REFUSED: run 'prepare' first", file=sys.stderr)
        return 2
    sc = capture_source_control(REPO_ROOT)
    if sc["working_tree_clean"] is not True:
        print(
            "REFUSED: working tree not clean (the gate would reject this run; run ID kept unused)",
            file=sys.stderr,
        )
        return 2
    amap = json.loads(ALIAS_MAP.read_text(encoding="utf-8"))["map"]
    split = json.loads(SMOKE_SPLIT.read_text(encoding="utf-8"))["splits"]
    case_files = {a: isles_case_files(RAW_ROOT, cid) for a, cid in amap.items()}
    set_determinism(int(cfg["training"]["seed"]))
    env = capture_environment(torch_module=torch, lock_file=REPO_ROOT / "requirements-lock.txt")
    det = capture_determinism(
        torch,
        seed=None,
        precision="fp32",
        amp=False,
        workers=int(cfg["training"]["dataloader_workers"]),
        batch_size=int(cfg["training"]["batch_size"]),
    )
    aliases = split["train"] + split["val"]
    plan = build_plan(
        run_id=run_id,
        purpose="v2 2D smoke test (not a performance experiment)",
        stage="train",
        dataset_id=cfg["dataset_id"],
        split_name="train",
        aliases=split["train"],
        allowed_case_count=len(split["train"]),
        validation_aliases=split["val"],
        model_id=MODEL_ID,
        checkpoint=None,
        config=CONFIG,
        split_file=SMOKE_SPLIT,
        policies=cfg["policies"],
        output_root=OUT_ROOT,
        expected={
            "config_sha256_lf": sha256_text_lf(CONFIG),
            "split_sha256_lf": sha256_text_lf(SMOKE_SPLIT),
            "package_lock_sha256_lf": env["package_lock_sha256_lf"],
            "alias_list_sha256": alias_list_sha256(aliases, CASE_KEY_POLICY),
            "determinism": {
                "precision": "fp32",
                "automatic_mixed_precision": False,
                "tf32_matmul": False,
                "tf32_convolution": False,
                "torch_deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "dataloader_workers": 0,
            },
        },
    )
    res = run_controlled(
        plan,
        V2TwoDRunner(cfg, case_files),
        repo_root=REPO_ROOT,
        roots=PathRoots(REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "data"),
        source_control=sc,
        environment=env,
        determinism=det,
    )
    if res.get("pre_run_status") != "approved":
        print(json.dumps(res, indent=1))
        return 2
    res["verify_run"] = verify_run(plan.run_dir)
    print(json.dumps(res, indent=1))
    return 0 if res["status"] == COMPLETED and res["verify_run"]["ok"] else 1


def compare(a: str, b: str) -> int:
    def load(rid: str, name: str) -> dict:
        return json.loads((OUT_ROOT / rid / name).read_text(encoding="utf-8"))

    pre = compare_manifests(load(a, "pre_run_manifest.json"), load(b, "pre_run_manifest.json"))
    pa, pb = load(a, "post_run_manifest.json"), load(b, "post_run_manifest.json")
    post = compare_manifests(pa, pb)
    outputs = {
        kind: {
            k: pa["outputs"][kind].get(k) == pb["outputs"][kind].get(k)
            for k in sorted(set(pa["outputs"][kind]) | set(pb["outputs"][kind]))
        }
        for kind in pa["outputs"]
    }
    res = {
        "pre_run": pre,
        "post_run": post,
        "output_hash_equal": outputs,
        "verify": {a: verify_run(OUT_ROOT / a), b: verify_run(OUT_ROOT / b)},
    }
    print(json.dumps(res, indent=1))
    return (
        0
        if pre["level"] == "exact_match" and post["level"] in ("exact_match", "unavailable")
        else 1
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("--run-id", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = p.parse_args(argv)
    if args.cmd == "prepare":
        return prepare()
    if args.cmd == "run":
        return run(args.run_id)
    return compare(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
