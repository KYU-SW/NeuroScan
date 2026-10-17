"""Create the patient-wise split from the QA case summary (original-space lesion volumes).

Usage:
    .venv\\Scripts\\python.exe scripts\\make_split.py --qa results\\qa\\case_summary.csv \\
        --out data\\splits\\split_v1_seed42.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from neuroscan_ai.data.split import SplitCase, make_patient_split


def load_cases(qa_csv: Path) -> list[SplitCase]:
    df = pd.read_csv(qa_csv)
    if df["error"].fillna("").astype(str).str.len().gt(0).any():
        raise SystemExit("QA summary contains errored cases; fix QA first")
    return [
        SplitCase(
            case_id=r.case_id,
            has_lesion=bool(r.has_lesion),
            lesion_volume_ml=float(r.lesion_volume_ml),
            thick_slice=bool(r.thick_slice),
        )
        for r in df.itertuples()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=Path, default=Path("results/qa/case_summary.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    args = parser.parse_args()

    split = make_patient_split(load_cases(args.qa), ratios=args.ratios, seed=args.seed)
    split = {
        "version": args.out.stem,
        "source_qa": args.qa.as_posix(),
        "test_policy": "Test patients are held out: never used for training, tuning, "
        "threshold selection or model selection.",
        **split,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=2), encoding="utf-8")
    for name, s in split["stats"].items():
        print(
            f"{name:5s} n={s['n']:3d} neg={s['negatives']} thick={s['thick_slice']} "
            f"<1mL={s['lt_1ml']} vol_median={s['lesion_volume_ml']['median']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
