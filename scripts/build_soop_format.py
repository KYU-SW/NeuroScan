"""SOOP format adapter run (K3): (X, Y, Z, 1) -> (X, Y, Z) copies with preservation checks.

  .venv\\Scripts\\python.exe scripts\\build_soop_format.py
      [--root data/raw/soop_pilot] [--aliases data/raw/soop_pilot/_internal/aliases.json]
      [--out data/interim/soop_format_v1]

No resampling, skull stripping, normalisation, ADC conversion or model. Console output names
aliases only. Exit codes: 0 ok, 2 refused, 3 one or more cases failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

from neuroscan_ai.datasets.soop import (  # noqa: E402
    DATASET_LEVEL_OPTIONAL,
    OPTIONAL,
    REQUIRED,
    SoopCase,
    SoopFormatError,
    build_case,
    subject_paths,
)


def git_ignored(path: Path) -> bool:
    probe = path / "probe.nii.gz"
    r = subprocess.run(
        ["git", "check-ignore", "-q", str(probe.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return r.returncode == 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=REPO_ROOT / "data/raw/soop_pilot")
    p.add_argument("--aliases", type=Path, default=None)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data/interim/soop_format_v1")
    p.add_argument(
        "--not-pilot",
        action="store_true",
        help="cases are not the permanently excluded pilot (default: pilot alias table)",
    )
    a = p.parse_args(argv)
    root, out = a.root.resolve(), a.out.resolve()
    aliases_path = (a.aliases or root / "_internal/aliases.json").resolve()
    if not git_ignored(out):
        print("REFUSED: output folder is not git-ignored")
        return 2
    table = json.loads(aliases_path.read_text(encoding="utf-8"))["aliases"]
    dataset_files = {k: root / k for k in DATASET_LEVEL_OPTIONAL if (root / k).is_file()}
    failed = 0
    for alias in sorted(table):
        sid = table[alias]["subject_id"]
        rel = subject_paths(sid)
        files = {k: root / rel[k] for k in (*REQUIRED, *OPTIONAL) if (root / rel[k]).is_file()}
        missing = [k for k in REQUIRED if k not in files]
        if missing:
            print(f"{alias}: excluded (missing {', '.join(missing)})")
            failed += 1
            continue
        try:
            res = build_case(SoopCase(sid, files, dataset_files), alias, out, pilot=not a.not_pilot)
        except SoopFormatError as e:
            print(f"{alias}: REFUSED ({e})")
            failed += 1
            continue
        m, au = res["metadata"], res["audit"]
        checks = au["preservation_checks"]
        ok = all(v for c in checks.values() for k, v in c.items() if k.endswith("_equal"))
        print(
            f"{alias}: ops={au['operations']} preserved={ok} grid_ok="
            f"{all(g['same_grid'] for g in au['grid_checks'].values())} "
            f"adc_group={m['adc_storage_group']} background={m['background_code']} "
            f"masks={['acute', *m['optional_masks_present']]}"
        )
    return 3 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
