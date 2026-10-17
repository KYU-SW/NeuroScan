"""Generate a processed dataset version from a YAML config.

Usage:
    .venv\\Scripts\\python.exe scripts\\preprocess.py --config configs\\preprocess_v1.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from neuroscan_ai.datasets import discover
from neuroscan_ai.preprocessing.pipeline import PreprocessConfig, preprocess_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing out_root")
    args = parser.parse_args()

    cfg_dict = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_root = Path(cfg_dict["out_root"])
    raw_root = Path(cfg_dict["raw_root"]).resolve()
    if out_root.resolve() == raw_root or raw_root in out_root.resolve().parents:
        raise SystemExit("out_root must not be inside raw_root")
    if out_root.exists() and any(out_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out_root} is not empty; pass --overwrite to regenerate")
        shutil.rmtree(out_root)

    cfg = PreprocessConfig(
        version=cfg_dict["version"],
        target_spacing=tuple(cfg_dict["target_spacing"]),
        orientation=cfg_dict.get("orientation", "RAS"),
        image_interp_order=cfg_dict.get("image_interp_order", 1),
    )
    records = discover(cfg_dict["dataset"], cfg_dict["raw_root"])
    rows = []
    for rec in tqdm(records, desc=f"preprocess {cfg.version}"):
        meta = preprocess_case(rec, cfg, out_root)
        rows.append(
            {
                "case_id": meta["case_id"],
                "orig_shape": "x".join(map(str, meta["original"]["shape"])),
                "orig_spacing": "x".join(f"{s:.4f}" for s in meta["original"]["spacing"]),
                "proc_shape": "x".join(map(str, meta["processed"]["shape"])),
                "proc_spacing": "x".join(f"{s:.4f}" for s in meta["processed"]["spacing"]),
                "resampled": meta["transforms"][2]["applied"],
                **meta["lesion"],
            }
        )
    shutil.copyfile(args.config, out_root / "preprocess_config.yaml")
    pd.DataFrame(rows).to_csv(out_root / "cases.csv", index=False)
    print(f"{len(rows)} cases written to {out_root}")


if __name__ == "__main__":
    main()
