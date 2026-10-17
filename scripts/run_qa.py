"""Run dataset QA and write results/qa/{case_summary.csv, dataset_summary.json, raw_manifest.json}.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_qa.py --dataset isles2022 --root data\\raw\\ISLES2022
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from neuroscan_ai.datasets import discover
from neuroscan_ai.qa.dataset_qa import qa_case, sha256_manifest, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/qa"))
    args = parser.parse_args()

    records = discover(args.dataset, args.root)
    if not records:
        raise SystemExit(f"no cases found under {args.root}")
    rows = [qa_case(r) for r in tqdm(records, desc="QA")]
    summary = summarize(rows, image_keys=list(records[0].images))
    summary = {"dataset": args.dataset, "root": args.root.as_posix(), **summary}

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "case_summary.csv", index=False)
    (args.out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = sha256_manifest(args.root)
    (args.out / "raw_manifest.json").write_text(
        json.dumps({"root": args.root.as_posix(), "files": manifest}, indent=2), encoding="utf-8"
    )
    print(f"{len(rows)} cases, {summary['n_errors']} errors, {len(manifest)} raw files hashed")
    print(f"wrote {args.out}/case_summary.csv, dataset_summary.json, raw_manifest.json")


if __name__ == "__main__":
    main()
