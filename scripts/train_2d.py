"""Training entry point for the 2D (Phase 3) and 2.5D (Phase 4) baselines; tiny overfit with --tiny.

The input width comes from ``data.context_slices`` in the config (0 = 2D, 1 = 2.5D); the script
itself is identical for both, which keeps the comparison fair.

Usage:
    .venv\\Scripts\\python.exe scripts\\train_2d.py --config configs\\2d_tiny_overfit.yaml --tiny
    .venv\\Scripts\\python.exe scripts\\train_2d.py --config configs\\2d_baseline.yaml
    .venv\\Scripts\\python.exe scripts\\train_2d.py --config configs\\25d_tiny_overfit.yaml --tiny
    .venv\\Scripts\\python.exe scripts\\train_2d.py --config configs\\25d_baseline.yaml

Only train and validation patients are loaded (SplitGuard); the test split stays sealed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA initialises

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from neuroscan_ai.data.split_guard import SplitGuard  # noqa: E402
from neuroscan_ai.evaluation.report import summarize  # noqa: E402
from neuroscan_ai.io import nifti  # noqa: E402
from neuroscan_ai.training.config import config_hash, in_channels, load_config  # noqa: E402
from neuroscan_ai.training.dataset import load_split_cases, norm_specs  # noqa: E402
from neuroscan_ai.training.seed import seed_everything  # noqa: E402
from neuroscan_ai.training.trainer import (  # noqa: E402
    Trainer,
    load_model_from_checkpoint,
    validate,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SIZE3 = ((1.0, "<1 mL"), (10.0, "1-10 mL"))


def size3(v: float) -> str:
    if v <= 0:
        return "negative"
    for upper, label in SIZE3:
        if v < upper:
            return label
    return ">=10 mL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true", help="tiny overfit sanity test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = f"{cfg['run_name']}_{time.strftime('%Y%m%d-%H%M%S')}_{config_hash(cfg)}"
    run_dir = Path(cfg["output"]["results_dir"]) / run_id
    model_dir = Path(cfg["checkpoint"]["dir"]) / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    shutil.copyfile(args.config, run_dir / "config_source.yaml")
    repro = seed_everything(cfg["seed"], cfg["training"]["deterministic"])

    guard = SplitGuard(cfg["data"]["split_file"])
    data = cfg["data"]
    specs = norm_specs(data["normalization"])
    root = Path(data["processed_root"])
    policy = data["brain_mask_policy"]
    if args.tiny:
        tiny = cfg["tiny_overfit"]
        train_cases = load_split_cases(
            guard, tiny["split"], root, data["modalities"], specs, tiny["cases"], policy
        )
        val_cases = train_cases  # deliberate: memorisation check
    else:
        train_cases = load_split_cases(
            guard, "train", root, data["modalities"], specs, None, policy
        )
        val_cases = load_split_cases(guard, "val", root, data["modalities"], specs, None, policy)
    print(
        f"run {run_id}: train {len(train_cases)} / eval {len(val_cases)} patients, "
        f"in_channels {in_channels(cfg)}",
        flush=True,
    )

    trainer = Trainer(cfg, guard, train_cases, val_cases, run_dir, model_dir, run_id, REPO_ROOT)
    write_json(run_dir / "train_slice_stats.json", trainer.train_ds.stats())
    fit = trainer.fit(cfg["training"]["max_epochs"], cfg["training"]["early_stopping_patience"])

    # Final evaluation with the best checkpoint, reloaded from disk (also tests save/load).
    model, info = load_model_from_checkpoint(model_dir / "best.pt", cfg, trainer.device)
    final = validate(
        model,
        val_cases,
        trainer.val_gts,
        cfg,
        trainer.device,
        trainer.loss_fn,
        keep_predictions=True,
    )
    # the reloaded best checkpoint must reproduce the Dice logged for that epoch during training
    reload_consistent = abs(final.mean_dice - fit["best_val_mean_dice"]) < 1e-6

    rows = final.rows
    md_path = Path(data["case_metadata"])
    if md_path.exists():
        md = pd.read_csv(md_path).set_index("case_id")
        for r in rows:
            for k in ("spacing_group", "adc_unit_group", "volume_group", "obliquity_group"):
                r[k] = md.loc[r["case_id"], k]
    for r in rows:
        r["size_group"] = size3(r["gt_volume_ml"])
    pd.DataFrame(rows).to_csv(run_dir / "validation_metrics.csv", index=False)

    pred_dir = run_dir / "predictions"
    for case in val_cases:
        _, geom = trainer.val_gts[case.case_id]
        nifti.save(
            final.predictions[case.case_id].astype(np.uint8),
            geom.affine,
            pred_dir / f"{case.case_id}_pred.nii.gz",
        )

    summary = {
        "run_id": run_id,
        "mode": "tiny_overfit" if args.tiny else "full",
        "config": str(args.config),
        "config_hash": config_hash(cfg),
        "reproducibility": repro,
        "best_checkpoint": str(model_dir / "best.pt"),
        "best_checkpoint_info": info,
        "checkpoint_reload_consistent": reload_consistent,
        **fit,
        "evaluation": summarize(
            rows, subgroup_keys=["size_group", "spacing_group", "adc_unit_group"]
        ),
        "data_access": guard.access_report(),
        "evaluated_cases": [r["case_id"] for r in rows],
    }
    if args.tiny:
        crit = cfg["tiny_overfit"]["success_criteria"]
        log = pd.read_csv(run_dir / "training_log.csv")
        pos = [r for r in rows if r["has_lesion"]]
        checks = {
            "mean_patient_dice": (
                final.mean_dice,
                final.mean_dice >= crit["min_mean_patient_dice"],
            ),
            "loss_ratio": (
                float(log.train_loss.iloc[-1] / log.train_loss.iloc[0]),
                log.train_loss.iloc[-1] / log.train_loss.iloc[0]
                <= crit["max_final_to_first_loss_ratio"],
            ),
            "max_probability": (
                min(r["max_probability"] for r in rows),
                min(r["max_probability"] for r in rows) >= crit["min_max_probability"],
            ),
            "not_all_one": (
                max(r["prediction_volume_ml"] / r["gt_volume_ml"] for r in pos),
                max(r["prediction_volume_ml"] / r["gt_volume_ml"] for r in pos)
                <= crit["max_predicted_to_gt_volume_ratio"],
            ),
            "checkpoint_reload": (reload_consistent, reload_consistent),
        }
        summary["tiny_overfit"] = {
            "checks": {k: {"value": v, "passed": bool(ok)} for k, (v, ok) in checks.items()},
            "passed": all(ok for _, ok in checks.values()),
            "per_case_dice": {r["case_id"]: r["dice"] for r in rows},
            "first_epoch": log.iloc[0].to_dict(),
            "last_epoch": log.iloc[-1].to_dict(),
        }
    if summary["data_access"]["test_cases_accessed"]:
        raise RuntimeError("test split accessed")
    write_json(run_dir / "summary.json", summary)
    print(
        f"done: best epoch {fit['best_epoch']} dice {fit['best_val_mean_dice']:.4f}; "
        f"final mean dice {final.mean_dice:.4f}; results {run_dir}"
    )
    if args.tiny:
        print(
            "tiny overfit:",
            "PASSED" if summary["tiny_overfit"]["passed"] else "FAILED",
            {k: v for k, v in summary["tiny_overfit"]["checks"].items()},
        )
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
