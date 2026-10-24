"""Training loop with patient-level validation through the Phase 2 engines.

Fail-fast guards (raise ``TrainingAborted``; the epoch log written so far is kept):
non-finite loss, non-finite gradient, gradient norm above ``guards.max_grad_norm``, validation
predictions empty for every positive patient (all-zero) or covering more than
``guards.all_one_fraction`` of the volume (all-one) for ``guards.collapse_patience_epochs``
consecutive epochs, and any test-split patient (SplitGuard).

Best checkpoint = highest validation positive-patient mean Dice. A new epoch replaces the best
only if its Dice is higher by more than ``tie_tolerance``; within the tolerance the lower
validation loss wins, and on a further tie the earlier epoch is kept.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from neuroscan_ai.data.split_guard import SplitGuard
from neuroscan_ai.evaluation.report import summarize_negative, summarize_positive
from neuroscan_ai.training.augmentation import build_train_transform
from neuroscan_ai.training.brain_gating import check_gt_inside_brain
from neuroscan_ai.training.config import config_hash, in_channels
from neuroscan_ai.training.dataset import CaseVolumes, SliceDataset
from neuroscan_ai.training.inference import load_original_gt, predict_case, restore_and_evaluate
from neuroscan_ai.training.model import build_loss, build_model, count_parameters
from neuroscan_ai.training.precision import autocast, check_precision
from neuroscan_ai.training.sampler import PositiveRatioSampler


class TrainingAborted(RuntimeError):
    pass


@dataclass
class ValidationResult:
    rows: list[dict[str, Any]]
    loss: float
    positive: dict[str, Any]
    negative: dict[str, Any]
    predictions: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def mean_dice(self) -> float:
        return self.positive["dice"].get("mean", 0.0)


def validate(
    model, cases: list[CaseVolumes], gts: dict, cfg: dict, device, loss_fn, keep_predictions=False
) -> ValidationResult:
    rows, losses, preds = [], [], {}
    for case in cases:
        prob, binary_volume, loss = predict_case(
            model,
            case,
            cfg["data"]["context_slices"],
            device,
            cfg["inference"]["size_multiple"],
            cfg["inference"]["batch_size"],
            cfg["training"]["precision"],
            loss_fn,
            threshold=cfg["inference"]["threshold"],
            brain_mask_gating=cfg["inference"]["brain_mask_gating"],
        )
        gt, geom = gts[case.case_id]
        row, binary = restore_and_evaluate(binary_volume, gt, geom, case.case_id)
        if case.brain is not None:
            row["gt_voxels_outside_brain_processed"] = int(
                np.count_nonzero((case.label > 0) & ~case.brain)
            )
        row["z_spacing"] = case.z_spacing
        row["max_probability"] = float(prob.data.max())
        row["prediction_fraction"] = float(binary.mean())
        rows.append(row)
        losses.append(loss)
        if keep_predictions:
            preds[case.case_id] = binary
    defined = [v for v in losses if v is not None]
    mean_loss = float(np.mean(defined)) if defined else float("nan")
    return ValidationResult(
        rows, mean_loss, summarize_positive(rows), summarize_negative(rows), preds
    )


def save_checkpoint(path: Path, model, optimizer, scheduler, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "info": info,
        },
        path,
    )


def load_model_from_checkpoint(path: Path, cfg: dict, device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt["info"]


class Trainer:
    def __init__(
        self,
        cfg: dict,
        guard: SplitGuard,
        train_cases: list[CaseVolumes],
        val_cases: list[CaseVolumes],
        run_dir: Path,
        model_dir: Path,
        run_id: str,
        repo_root: Path,
    ) -> None:
        self.cfg, self.guard, self.run_id = cfg, guard, run_id
        self.run_dir, self.model_dir = run_dir, model_dir
        guard.assert_no_test([c.case_id for c in train_cases], "training")
        guard.assert_no_test([c.case_id for c in val_cases], "validation")
        if cfg["inference"]["brain_mask_gating"]:
            for c in train_cases:  # gating is only valid if no train lesion lies outside the brain
                check_gt_inside_brain(c.label, c.brain, c.case_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tr = cfg["training"]
        seed = cfg["seed"]

        self.train_ds = SliceDataset(
            train_cases,
            cfg["data"]["context_slices"],
            cfg["data"]["train_spatial_size"],
            build_train_transform(cfg["augmentation"], seed),
            exclude_brain_free_slices=cfg["data"]["exclude_brain_free_slices"],
        )
        n_samples = cfg["sampler"]["samples_per_epoch"] or len(self.train_ds)
        self.sampler = PositiveRatioSampler(
            self.train_ds.slice_positive, cfg["sampler"]["positive_fraction"], n_samples, seed
        )
        self.loader = DataLoader(
            self.train_ds,
            batch_size=tr["batch_size"],
            sampler=self.sampler,
            num_workers=tr["num_workers"],
            drop_last=True,
        )
        self.val_cases = val_cases
        self.val_gts = {c.case_id: load_original_gt(c.meta, repo_root) for c in val_cases}

        self.model = build_model(cfg).to(self.device)
        self.loss_fn = build_loss(cfg)
        opt = cfg["optimizer"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=opt["lr"], weight_decay=opt["weight_decay"]
        )
        self.precision = check_precision(tr["precision"])  # fp32: no autocast, no GradScaler
        self.guards = cfg["guards"]
        sch = cfg["scheduler"]
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=tr["max_epochs"], eta_min=sch["eta_min"]
            )
            if sch["name"] == "cosine"
            else None
        )
        self.info_base = {
            "run_id": run_id,
            "config_hash": config_hash(cfg),
            "preprocessing_version": cfg["data"]["preprocessing_version"],
            "split_version": guard.version,
            "normalization": cfg["data"]["normalization"],
            "in_channels": in_channels(cfg),
            "precision": self.precision,
        }

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.sampler.set_epoch(epoch)
        losses, grad_norms, n_pos, n_seen = [], [], 0, 0
        max_norm = float(self.guards["max_grad_norm"])
        for batch in self.loader:
            x = batch["image"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device, self.precision):
                logits = self.model(x)
            loss = self.loss_fn(logits.float(), y)
            if not torch.isfinite(loss):
                raise TrainingAborted(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            # measurement only (no clipping): total L2 norm of all gradients
            norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf")))
            if not np.isfinite(norm):
                raise TrainingAborted(f"non-finite gradient at epoch {epoch}")
            if norm > max_norm:
                self._dump_batch(batch, epoch, norm)
                raise TrainingAborted(f"gradient norm {norm:.3g} > {max_norm:.3g} at epoch {epoch}")
            self.optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(norm)
            n_pos += int(batch["slice_has_lesion"].sum())
            n_seen += len(y)
        if self.scheduler:
            self.scheduler.step()
        return {
            "train_loss": float(np.mean(losses)),
            "sampled_positive_fraction": n_pos / n_seen,
            "grad_norm_mean": float(np.mean(grad_norms)),
            "grad_norm_max": float(np.max(grad_norms)),
        }

    def _dump_batch(self, batch: dict, epoch: int, norm: float) -> None:
        """Record which cases/slices formed the batch that tripped the gradient guard."""
        keys = ("case_id", "slice_index", "slice_has_lesion", "brain_voxels", "z_spacing")
        items = [
            {
                k: (v[i].item() if hasattr(v[i], "item") else v[i])
                for k, v in ((k, batch[k]) for k in keys)
            }
            for i in range(len(batch["case_id"]))
        ]
        write_json(
            self.run_dir / "aborted_batch.json",
            {"epoch": epoch, "grad_norm": norm, "samples": items},
        )

    def check_collapse(self, val: ValidationResult, streak: dict[str, int], epoch: int) -> None:
        pos = [r for r in val.rows if r["has_lesion"]]
        all_zero = bool(pos) and all(r["prediction_voxels"] == 0 for r in pos)
        all_one = any(r["prediction_fraction"] > self.guards["all_one_fraction"] for r in val.rows)
        streak["zero"] = streak["zero"] + 1 if all_zero else 0
        streak["one"] = streak["one"] + 1 if all_one else 0
        limit = self.guards["collapse_patience_epochs"]
        if streak["zero"] >= limit:
            raise TrainingAborted(f"all-zero predictions for {limit} epochs (epoch {epoch})")
        if streak["one"] >= limit:
            raise TrainingAborted(f"all-one predictions for {limit} epochs (epoch {epoch})")

    def fit(self, max_epochs: int, patience: int | None) -> dict[str, Any]:
        tol = self.cfg["checkpoint"]["tie_tolerance"]
        best = {"epoch": None, "dice": -1.0, "loss": float("inf")}
        log, since_best = [], 0
        streak = {"zero": 0, "one": 0}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for epoch in range(1, max_epochs + 1):
            te = time.time()
            lr = self.optimizer.param_groups[0]["lr"]
            tr = self.train_epoch(epoch)
            val = validate(
                self.model, self.val_cases, self.val_gts, self.cfg, self.device, self.loss_fn
            )
            self.check_collapse(val, streak, epoch)
            pos = val.positive
            entry = {
                "epoch": epoch,
                "lr": lr,
                **tr,
                "val_loss": val.loss,
                "val_mean_dice": pos["dice"].get("mean"),
                "val_median_dice": pos["dice"].get("median"),
                "val_mean_iou": pos["iou"].get("mean"),
                "val_mean_sensitivity": pos["sensitivity"].get("mean"),
                "val_mean_precision": pos["precision"].get("mean"),
                "val_precision_undefined": pos["precision"]["n_undefined"],
                "val_mean_abs_volume_error_ml": pos["absolute_volume_error_ml"].get("mean"),
                "val_negative_correct": val.negative["correct_negative"],
                "val_negative_fp_volume_ml": sum(
                    c["false_positive_volume_ml"] or 0 for c in val.negative["cases"]
                ),
                "epoch_time_s": time.time() - te,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3
                if self.device.type == "cuda"
                else 0.0,
            }
            dice = entry["val_mean_dice"] or 0.0
            improved = dice > best["dice"] + tol or (
                abs(dice - best["dice"]) <= tol and val.loss < best["loss"]
            )
            info = {**self.info_base, "epoch": epoch, "val_mean_dice": dice, "val_loss": val.loss}
            save_checkpoint(
                self.model_dir / "latest.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                info,
            )
            if improved:
                best = {"epoch": epoch, "dice": dice, "loss": val.loss}
                save_checkpoint(
                    self.model_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    info,
                )
                since_best = 0
            else:
                since_best += 1
            entry["is_best"] = improved
            log.append(entry)
            pd.DataFrame(log).to_csv(self.run_dir / "training_log.csv", index=False)
            print(
                f"epoch {epoch:3d} loss {tr['train_loss']:.4f} val_loss {val.loss:.4f} "
                f"dice {dice:.4f} (best {best['dice']:.4f} @ {best['epoch']}) "
                f"pos {tr['sampled_positive_fraction']:.2f} gnorm {tr['grad_norm_max']:.2e} "
                f"{entry['epoch_time_s']:.0f}s",
                flush=True,
            )
            if patience is not None and since_best >= patience:
                break
        return {
            "epochs_run": len(log),
            "best_epoch": best["epoch"],
            "best_val_mean_dice": best["dice"],
            "best_val_loss": best["loss"],
            "total_training_time_s": time.time() - t0,
            "peak_vram_gb": max(e["peak_vram_gb"] for e in log),
            "parameters": count_parameters(self.model),
            "train_slices": self.train_ds.stats(),
        }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
