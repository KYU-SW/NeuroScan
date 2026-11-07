"""Run reproducibility manifest CLI (no model is run; image files are only hashed).

  capture   record the current source control, environment, determinism and the hashes of the
            given config / split / checkpoint (and optional input files) as a pre-run manifest
  validate  rebuild the same manifest now and compare it with a saved one
  compare   compare two saved manifests

  .venv\\Scripts\\python.exe scripts\\capture_run_manifest.py capture --run-id R --stage S
      --purpose P --dataset-id D --split-name val --split data/splits/x.json --config configs/x.yaml
      [--checkpoint models/.../best.pt] [--policy preprocessing=ID ...] [--aliases-file A.json]
      [--seed N --precision fp32 --batch-size N --workers N] [--import-torch] [--lock-tag TAG]
  ... validate --manifest results/reproducibility/R/pre_run.json <same capture arguments>
  ... compare A.json B.json

Manifests are written only under the git-ignored results/reproducibility/ and never overwritten.
Exit codes: 0 ok (compare / validate: exact_match), 1 not exact, 2 refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

from neuroscan_ai.component_analysis.records import is_git_ignored  # noqa: E402
from neuroscan_ai.reproducibility import (  # noqa: E402
    EXACT,
    ManifestError,
    PathRoots,
    RunSpec,
    build_pre_run,
    compare_manifests,
    write_manifest,
)
from neuroscan_ai.reproducibility.environment import (  # noqa: E402
    capture_determinism,
    capture_environment,
    capture_source_control,
)

OUTPUT_ROOT = REPO_ROOT / "results" / "reproducibility"
ROOTS = PathRoots(REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "data")


def _capture_args(p: argparse.ArgumentParser) -> None:
    for name in ("run-id", "stage", "purpose", "dataset-id", "split-name"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--referenced-config", action="append", type=Path, default=[])
    p.add_argument("--checkpoint", type=Path)
    p.add_argument(
        "--policy", action="append", default=[], help="key=value, e.g. threshold=fixed_0.5"
    )
    p.add_argument("--aliases-file", type=Path, help="JSON list of case aliases")
    p.add_argument("--seed", type=int)
    p.add_argument("--precision")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument(
        "--import-torch", action="store_true", help="record torch / CUDA / determinism state"
    )
    p.add_argument("--lock-tag")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    _capture_args(c)
    c.add_argument("--out", type=Path)
    v = sub.add_parser("validate")
    _capture_args(v)
    v.add_argument("--manifest", required=True, type=Path)
    cm = sub.add_parser("compare")
    cm.add_argument("a", type=Path)
    cm.add_argument("b", type=Path)
    return p


def _build(args: argparse.Namespace) -> tuple[dict, list[str]]:
    aliases = json.loads(args.aliases_file.read_text(encoding="utf-8")) if args.aliases_file else []
    policies = dict(kv.split("=", 1) for kv in args.policy)
    torch_mod = None
    if args.import_torch:
        import torch as torch_mod  # noqa: PLC0415
    spec = RunSpec(
        run_id=args.run_id,
        stage=args.stage,
        purpose=args.purpose,
        dataset_id=args.dataset_id,
        split_name=args.split_name,
        aliases=aliases,
        config=args.config,
        split=args.split,
        checkpoint=args.checkpoint,
        referenced_configs=args.referenced_config,
        policies=policies,
        access={"allowed_splits": [args.split_name]},
    )
    manifest = build_pre_run(
        spec,
        roots=ROOTS,
        source_control=capture_source_control(REPO_ROOT, args.lock_tag),
        environment=capture_environment(
            torch_module=torch_mod, lock_file=REPO_ROOT / "requirements-lock.txt"
        ),
        determinism=capture_determinism(
            torch_mod,
            seed=args.seed,
            precision=args.precision,
            amp=args.amp,
            workers=args.workers,
            batch_size=args.batch_size,
        ),
    )
    return manifest, aliases


def _out_path(args: argparse.Namespace) -> Path:
    out = args.out or OUTPUT_ROOT / args.run_id / "pre_run.json"
    out = out if out.is_absolute() else REPO_ROOT / out
    root = OUTPUT_ROOT.resolve()
    if root not in out.resolve().parents:
        raise ManifestError("manifests must be written inside results/reproducibility/")
    if not is_git_ignored(out, REPO_ROOT):
        raise ManifestError("output path is not git-ignored")
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.cmd == "compare":
            res = compare_manifests(
                *(json.loads(p.read_text(encoding="utf-8")) for p in (args.a, args.b))
            )
        elif args.cmd == "validate":
            saved = json.loads(args.manifest.read_text(encoding="utf-8"))
            res = compare_manifests(saved, _build(args)[0])
        else:
            out = _out_path(args)
            manifest, aliases = _build(args)
            write_manifest(out, manifest, aliases=aliases)
            print(json.dumps({"written": out.relative_to(REPO_ROOT).as_posix()}))
            return 0
    except (ManifestError, OSError, ValueError) as e:
        print(f"REFUSED: {type(e).__name__ if isinstance(e, OSError) else e}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1))
    return 0 if res["level"] == EXACT else 1


if __name__ == "__main__":
    sys.exit(main())
