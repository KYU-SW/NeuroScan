"""Phase 5A CLI: independent inference for one DWI + ADC pair (research prototype).

Usage:
    .venv\\Scripts\\python.exe scripts\\infer_case.py --dwi <dwi.nii.gz> --adc <adc.nii.gz> ^
        --output outputs [--case-id ID] [--device auto|cuda|cpu] [--strict] [--json]

Writes outputs/<case_id>/<run_id>/ (never overwrites). stdout: a short summary, or with --json a
single JSON object. stderr: progress, warnings and errors (no local paths). Exit codes:
0 success (also when no lesion is predicted), 1 internal, 2 arguments, 3 input, 4 model/config,
5 preprocessing/brain mask, 6 device/inference, 7 output write, 8 output exists, 9 sealed test
case, 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from neuroscan_ai.inference import run_independent_inference
from neuroscan_ai.inference.schemas import DISCLAIMER


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroScan-ai independent inference (research only)")
    p.add_argument("--dwi", type=Path, required=True, help="DWI NIfTI (.nii / .nii.gz)")
    p.add_argument("--adc", type=Path, required=True, help="ADC NIfTI on the same grid")
    p.add_argument("--output", type=Path, required=True, help="output root, e.g. outputs")
    p.add_argument("--checkpoint", type=Path, default=None, help="default: configs/inference_v1")
    p.add_argument("--config", type=Path, default=None, help="training config of the checkpoint")
    p.add_argument("--inference-config", type=Path, default=None)
    p.add_argument("--case-id", default=None, help="de-identified ID; default case-<random>")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--strict", action="store_true", help="treat input warnings as errors")
    p.add_argument("--no-export-inputs", action="store_true", help="do not write inputs/ copies")
    p.add_argument("--record-input-names", action="store_true", help="store input basenames")
    p.add_argument("--json", action="store_true", help="print one JSON object to stdout")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logger = logging.getLogger("neuroscan_ai.inference")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[neuroscan] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(
        logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    )

    def progress(stage: str, frac: float) -> None:
        if not args.quiet:
            print(f"[neuroscan] {frac:4.0%} {stage}", file=sys.stderr, flush=True)

    try:
        result = run_independent_inference(
            args.dwi,
            args.adc,
            args.output,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            inference_config_path=args.inference_config,
            case_id=args.case_id,
            device=args.device,
            strict=args.strict,
            export_inputs=not args.no_export_inputs,
            record_input_names=args.record_input_names,
            progress=progress,
        )
    except KeyboardInterrupt:
        print("[neuroscan] interrupted", file=sys.stderr)
        return 130

    for w in result.warnings:
        print(f"[neuroscan] WARNING {w['code']}: {w['message']}", file=sys.stderr)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False))
        return result.exit_code
    if result.status != "succeeded":
        e = result.error or {}
        print(f"[neuroscan] ERROR {e.get('code')}: {e.get('message')}", file=sys.stderr)
        return result.exit_code
    q = result.quantification
    print(f"case_id: {result.case_id}")
    print(f"run_id:  {result.run_id}")
    print(f"output:  {Path(args.output) / result.case_id / result.run_id}")
    if q["lesion_detected"]:
        print(
            f"model-predicted lesion region: {q['total_lesion_volume_ml']:.2f} mL, "
            f"{q['component_count']} component(s) (research, automatic segmentation)"
        )
    else:
        print("The model did not mark any lesion region (this does not mean no lesion is present).")
    print(f"warnings: {len(result.warnings)}")
    print(DISCLAIMER["en"])
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
