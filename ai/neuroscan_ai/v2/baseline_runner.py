"""v2 2D baseline training runner (run-control profile ``baseline_v1``).

Order (``run_baseline``), nothing opens data or builds a model before step 8:
1 load ``configs/v2/2d_baseline.yaml`` 2 ``validate_policy_config`` 3 architecture / parameter count
(analytic, no model built) 4 every hyperparameter 5 RunPlan 6 pre-run manifest 7 gate approval
8 ``BaselineRunner.run``: approval re-verified, then datasets and the model are created.

Training: AdamW, cosine schedule stepped per epoch, masked Dice + masked BCE (1:1), gradient
clipping (L2, 1.0), deterministic brain-slice sampling (positive fraction 0.5), checkpoint
selection on validation at threshold 0.5 (ungated, ``baseline_selection.select_checkpoint``),
early stopping after 20 epochs without a new selected epoch. Final validation outputs are written
on the original grid from the selected (best) model. Threshold search is not run here.

A failure (any exception, incl. CUDA OOM or a deterministic-algorithm error) writes
``training/failure_record.json`` (stage, error type, redacted message, output hashes so far) and
re-raises; the lifecycle then writes the failed post-run manifest. Nothing is retried or adjusted.

Device policy ``device_policy_v2_baseline_cuda_required_v1``: the dataset profile is derived from
the plan's inputs (``dataset_profile``), not declared by the caller. A formal ISLES profile needs
device ``cuda`` (no cpu / cuda_if_available override, no CPU fallback); a refused device stops
before the gate (no run folder, run ID unused). Synthetic profiles may use any allowed device.

``epoch_limit`` exists for SYNTHETIC tests only: it is refused unless the plan's dataset is
``synthetic`` and it is recorded in the training history.
"""

from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from neuroscan_ai.reproducibility import PathRoots, case_keys
from neuroscan_ai.reproducibility.environment import capture_determinism
from neuroscan_ai.reproducibility.hashing import canonical_json_sha256, sha256_file
from neuroscan_ai.run_control import build_plan, run_controlled
from neuroscan_ai.run_control.gate import GateResult, verify_approval
from neuroscan_ai.run_control.outputs import run_output_path
from neuroscan_ai.run_control.plan import RunPlan
from neuroscan_ai.v2 import baseline_dataset as bd
from neuroscan_ai.v2 import baseline_inference as bi
from neuroscan_ai.v2 import baseline_training as bt
from neuroscan_ai.v2.baseline_preprocess import (
    BRAIN_MASK_POLICY,
    PREPROCESS_POLICY,
    validate_policy_config,
)
from neuroscan_ai.v2.baseline_selection import select_checkpoint
from neuroscan_ai.v2.dataset import CaseFiles
from neuroscan_ai.v2.outputs import write_aggregate, write_case_outputs, write_json_new
from neuroscan_ai.v2.unet_2d import CHANNELS, MODEL_ID, build_model, describe

RUNNER_KIND = "v2_2d_baseline_runner"
CHECKPOINT_SCHEMA = "neuroscan-v2-baseline-checkpoint-v1"
SYNTHETIC_DATASET = bt.PROFILE_SYNTHETIC
TIMING_NAME = "training/timing_record.json"  # wall-clock audit; not a hashed output
_ABS_PATH = re.compile(r"([A-Za-z]:[\\/])|(^|[\s'\"(])/[^\s]")


def prepare_config(config_path: Path) -> dict[str, Any]:
    """Steps 1-4 (before any plan, gate, data or model)."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    validate_policy_config(cfg)
    bt.validate_architecture(cfg)
    bt.validate_hyperparameters(cfg)
    return cfg


def dataset_profile(plan: RunPlan, cfg: Mapping[str, Any], repo_root: Path) -> str:
    """``formal_isles`` when any formal marker is present (formal dataset id, an input file under
    the repository's raw-data folder, the formal split file or the internal run-split folder);
    ``synthetic`` only for dataset id ``synthetic`` with declared inputs and no formal marker."""
    root = Path(repo_root).resolve()
    raw = root / "data" / "raw"
    split = Path(plan.split_file).resolve()
    formal = (
        plan.dataset_id == cfg.get("dataset_id")
        or split == (root / cfg["data"]["split_source"]).resolve()
        or split.is_relative_to(root / "results" / "v2_runs" / "_internal")
        or any(
            Path(p).resolve().is_relative_to(raw)
            for roles in plan.input_files.values()
            for p in roles.values()
        )
    )
    if formal:
        return bt.PROFILE_FORMAL
    if plan.dataset_id == SYNTHETIC_DATASET and plan.input_files:
        return bt.PROFILE_SYNTHETIC
    raise bt.ConfigValidationError("dataset profile cannot be established; refused")


def redact(msg: str, aliases: Iterable[str] = (), forbidden_ids: Iterable[str] = ()) -> str:
    text = str(msg)
    if _ABS_PATH.search(text) or any(a and a in text for a in (*aliases, *forbidden_ids)):
        return "[redacted]"
    return text[:200]


def _hash_outputs(run_dir: Path) -> dict[str, str]:
    return {
        p.relative_to(run_dir).as_posix(): sha256_file(p)
        for p in sorted(run_dir.rglob("*"))
        if p.is_file()
    }


@dataclass
class BaselineRunner:
    config: Mapping[str, Any]
    case_files: Mapping[str, CaseFiles]  # alias -> files (never written to any output)
    device: str | None = None  # execution device override (default: config)
    epoch_limit: int | None = None  # synthetic tests only
    dataset_profile: str = bt.PROFILE_FORMAL  # set by run_baseline; formal unless proven synthetic
    forbidden_ids: tuple[str, ...] = ()
    kind: str = RUNNER_KIND
    training_outputs: str = "baseline_v1"
    _stage: str = field(default="not_started", init=False)

    @property
    def seed(self) -> int:
        return int(self.config["training"]["seed"])

    @property
    def threshold(self) -> float:
        return bt.THRESHOLD

    def apply_determinism(self) -> dict[str, Any]:
        bt.apply_determinism(self.seed)
        tr = self.config["training"]
        return capture_determinism(
            torch,
            seed=self.seed,
            precision=tr["precision"],
            amp=tr["amp"],
            workers=int(tr["num_workers"]),
            batch_size=int(tr["batch_size"]),
        )

    # ------------------------------------------------------------ run-control entry point
    def run(self, gate: GateResult, plan: RunPlan) -> None:
        verify_approval(gate)  # before any data or model is touched
        try:
            self._run(gate, plan)
        except BaseException as e:
            self._write_failure(gate, plan, e)
            raise

    def _write_failure(self, gate: GateResult, plan: RunPlan, e: BaseException) -> None:
        run_dir = gate.run_dir
        path = run_output_path(run_dir, "failure_record")
        if path.exists():
            return
        written = len(list((run_dir / "patient_metrics").glob("*.json")))
        record = {
            "failed_stage": getattr(e, "stage", None) or self._stage,
            "runner_stage": self._stage,
            "error_type": type(e).__name__,
            "error_message": redact(str(e), plan.all_aliases, self.forbidden_ids),
            "validation_cases_succeeded": written,
            "validation_cases_failed": len(plan.validation_aliases) - written,
            "accessed_splits": sorted(plan.split_counts),
            "outputs_before_failure_sha256": _hash_outputs(run_dir),
            "retried": False,
            "hyperparameters_adjusted": False,
        }
        write_json_new(path, record)

    # ------------------------------------------------------------ training
    def _run(self, gate: GateResult, plan: RunPlan) -> None:
        cfg = self.config
        tr = cfg["training"]
        if plan.dataset_id != SYNTHETIC_DATASET:  # defence in depth (run_baseline checks first)
            self.dataset_profile = bt.PROFILE_FORMAL
        effective = bt.check_device_policy(
            self.dataset_profile, tr["device"], self.device, self.epoch_limit
        )
        run_dir = gate.run_dir
        pre = gate.pre_run or {}
        device = bt.resolve_device(effective)
        bt.apply_determinism(self.seed)

        self._stage = "data_loading"
        keys = case_keys(plan.all_aliases)
        train = bd.load_cases({a: self.case_files[a] for a in plan.aliases}, keys)
        val = bd.load_cases({a: self.case_files[a] for a in plan.validation_aliases}, keys)
        if not any(c.original_gt.any() for c in val):
            raise bi.MissingValidationResultError("all validation cases are lesion-free; refused")
        write_json_new(
            run_output_path(run_dir, "preprocessing_record"),
            {
                "preprocess_policy": PREPROCESS_POLICY,
                "brain_mask_policy": BRAIN_MASK_POLICY,
                "train_cases": [c.record for c in train],
                "validation_cases": [c.record for c in val],
                "inputs_modified": False,
            },
        )

        self._stage = "model_build"
        model = build_model(self.seed).to(device)
        info = describe(model, self.seed)
        opt = bt.build_optimizer(model)
        sched = bt.build_scheduler(opt)
        max_epochs = int(tr["max_epochs"])
        n_epochs = min(max_epochs, self.epoch_limit) if self.epoch_limit else max_epochs
        patience = int(tr["early_stopping_patience"])
        batch = int(tr["batch_size"])
        frac = float(tr["sampling"]["positive_fraction"])

        self._stage = "training_loop"
        history, sampling, records, timing = [], [], [], []
        best_state = best_sel = None
        stopped_early = False
        for epoch in range(1, n_epochs + 1):
            t0 = time.perf_counter()
            lr = float(opt.param_groups[0]["lr"])
            order, srec = bd.epoch_sampling(train, self.seed, epoch, frac)
            sampling.append(srec)
            tm = bt.train_one_epoch(model, bd.batches(train, order, batch), opt, device)
            sched.step()
            vloss = bi.validation_loss(model, val, device, batch)
            ev = bi.evaluate(model, val, device, batch)
            u = ev["ungated_summary"]
            records.append(
                {
                    "epoch": epoch,
                    "mean_dice_positive": u["mean_dice_positive"],
                    "val_loss": vloss,
                    "false_positive_voxel_volume_ml": u["false_positive_voxel_volume_ml"],
                    "threshold": bt.THRESHOLD,
                    "output": "ungated",
                }
            )
            sel = select_checkpoint(records)
            if sel["selected_epoch"] == epoch:
                best_state = _snapshot(model, opt, sched)
                best_sel = sel
            history.append(
                {
                    "epoch": epoch,
                    **{k: tm[k] for k in ("train_loss_total", "train_loss_dice", "train_loss_bce")},
                    "validation_loss_total": vloss,
                    "validation_mean_positive_case_dice": u["mean_dice_positive"],
                    "validation_total_false_positive_voxel_volume_ml": u[
                        "false_positive_voxel_volume_ml"
                    ],
                    "validation_lesion_wise_recall": u["lesion_recall_exploratory"],
                    "validation_missed_gt_component_count": u["missed_gt_count"],
                    "learning_rate": lr,
                    "gradient_norm": tm["gradient_norm"],
                    "sampled_positive_ratio": srec["sampled_positive_ratio"],
                    "collapse": ev["collapse"],
                    "selected_epoch_so_far": sel["selected_epoch"],
                }
            )
            timing.append({"epoch": epoch, "duration_seconds": time.perf_counter() - t0})
            if bt.early_stop(epoch, sel["selected_epoch"], patience):
                stopped_early = True
                break
        last_epoch = history[-1]["epoch"]
        last_state = _snapshot(model, opt, sched)

        self._stage = "checkpoints"
        common = self._checkpoint_common(plan, pre, info, device)
        self._save_ckpt(run_dir, "last_checkpoint", last_state, last_epoch, best_sel, common)
        self._save_ckpt(
            run_dir, "best_checkpoint", best_state, best_sel["selected_epoch"], best_sel, common
        )
        write_json_new(
            run_output_path(run_dir, "training_history"),
            {
                "model": info,
                "device_type": device.type,
                "hyperparameters": bt.FIXED_TRAINING,
                "max_epochs": max_epochs,
                "epochs_run": last_epoch,
                "synthetic_epoch_limit": self.epoch_limit,
                "dataset_profile": self.dataset_profile,
                "device_policy": bt.DEVICE_POLICY,
                "collapse_policy": bt.FIXED_COLLAPSE_POLICY,
                "early_stopping_patience": patience,
                "stopped_early": stopped_early,
                "selection": best_sel,
                "epochs": history,
                "duration_recorded_in": TIMING_NAME + " (wall clock; excluded from exact match)",
            },
        )
        write_json_new(run_dir / TIMING_NAME, {"epochs": timing})
        write_json_new(
            run_output_path(run_dir, "sampler_record"),
            {"policy": bd.SAMPLING_POLICY, "seed": self.seed, "epochs": sampling},
        )
        write_json_new(
            run_output_path(run_dir, "threshold_record"),
            {
                "threshold": bt.THRESHOLD,
                "source": bt.THRESHOLD_SOURCE,
                "optimized": False,
                "threshold_search_run": False,
            },
        )

        self._stage = "final_validation"
        model.load_state_dict(best_state["model"])
        ev = bi.evaluate(model, val, device, batch)  # selected model, re-evaluated
        self._write_validation(run_dir, val, ev, best_sel)

    # ------------------------------------------------------------ helpers
    def _checkpoint_common(self, plan, pre, info, device) -> dict[str, Any]:
        split = json.loads(Path(plan.split_file).read_text(encoding="utf-8"))
        env = pre.get("environment", {})
        return {
            "schema": CHECKPOINT_SCHEMA,
            "model_id": MODEL_ID,
            "architecture": info["architecture"],
            "channels": list(CHANNELS),
            "parameter_count": info["trainable_parameters"],
            "seed": self.seed,
            "config_canonical_sha256": canonical_json_sha256(dict(self.config)),
            "split_canonical_sha256": canonical_json_sha256(split),
            "preprocessing_policy": PREPROCESS_POLICY,
            "brain_mask_policy": BRAIN_MASK_POLICY,
            "threshold_policy": bt.THRESHOLD_SOURCE,
            "package_lock_sha256_lf": env.get("package_lock_sha256_lf"),
            "git_commit": pre.get("source_control", {}).get("git_commit"),
            "environment": {
                "torch": torch.__version__,
                "python": env.get("python"),
                "device_type": device.type,
            },
            "sampler": {
                "policy": bd.SAMPLING_POLICY,
                "seed": self.seed,
                "positive_fraction": bt.FIXED_TRAINING["sampling"]["positive_fraction"],
                "regenerate": "baseline_dataset.epoch_sampling(cases, seed, epoch)",
            },
        }

    def _save_ckpt(self, run_dir, label, state, epoch, sel, common) -> None:
        path = run_output_path(run_dir, label)
        if path.exists():
            raise FileExistsError("checkpoint exists; refusing to overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **common,
                "kind": label,
                "epoch": int(epoch),
                "best_metric": {
                    "mean_dice_positive_ungated_t0_5": sel["best_mean_dice_positive"],
                    "selected_epoch": sel["selected_epoch"],
                },
                "model_state": state["model"],
                "optimizer_state": state["optimizer"],
                "scheduler_state": state["scheduler"],
            },
            path,
        )

    def _write_validation(self, run_dir, val, ev, sel) -> None:
        per_case = []
        for c, mu, mg in zip(val, ev["per_case_ungated"], ev["per_case_gated"], strict=True):
            res = ev["outputs"][c.case_key]
            metric = {
                "case_key": c.case_key,
                "official_output": "ungated",
                "threshold": bt.THRESHOLD,
                "threshold_source": bt.THRESHOLD_SOURCE,
                "output_record": res.record,  # boundary + brain-mask round-trip QA
                "gt_decode_audit": c.prepared.record["gt_decode"],
                "target_grid_audit": c.prepared.record["target_size"],
                "ungated": mu,
                "gated_comparison_only": mg,
            }
            write_case_outputs(
                run_dir,
                c.case_key,
                probability=res.probability,
                ungated=res.ungated,
                brain=res.brain_mask,
                gated=res.gated,
                affine=res.grid.affine,
                metric=metric,
            )
            per_case.append(metric)
        summary = {
            "official_output": "ungated",
            "threshold": bt.THRESHOLD,
            "threshold_source": bt.THRESHOLD_SOURCE,
            "selected_epoch": sel["selected_epoch"],
            "ungated": ev["ungated_summary"],
            "gated_comparison_only": ev["gated_summary"],
            "ungated_vs_gated": ev["ungated_vs_gated"],
            "collapse": ev["collapse"],
            "note": "validation cases also select the checkpoint: optimistically biased",
        }
        out = {**summary, "cases": per_case}
        write_json_new(run_output_path(run_dir, "validation_metrics"), out)
        write_aggregate(run_dir, {**summary, "validation_cases": len(per_case)})


def _snapshot(model, opt, sched) -> dict[str, Any]:
    return {
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": copy.deepcopy(opt.state_dict()),
        "scheduler": copy.deepcopy(sched.state_dict()),
    }


def run_baseline(
    config_path: Path,
    plan_kwargs: Mapping[str, Any],
    case_files: Mapping[str, CaseFiles],
    *,
    repo_root: Path,
    roots: PathRoots,
    source_control: Mapping[str, Any],
    environment: Mapping[str, Any],
    determinism: Mapping[str, Any],
    forbidden_ids: Iterable[str] = (),
    device: str | None = None,
    epoch_limit: int | None = None,
) -> dict[str, Any]:
    """Config checks (1-4) -> RunPlan (5) -> gate + manifests (6-7) -> runner (8)."""
    cfg = prepare_config(config_path)
    plan = build_plan(
        **{
            **plan_kwargs,
            "model_id": MODEL_ID,
            "config": Path(config_path),
            "policies": dict(cfg["policies"]),
        }
    )
    profile = dataset_profile(plan, cfg, repo_root)
    # refused device / override / epoch_limit: before the gate (no folder, run ID unused)
    bt.check_device_policy(profile, cfg["training"]["device"], device, epoch_limit)
    runner = BaselineRunner(
        cfg,
        case_files,
        device=device,
        epoch_limit=epoch_limit,
        forbidden_ids=tuple(forbidden_ids),
        dataset_profile=profile,
    )
    return run_controlled(
        plan,
        runner,
        repo_root=repo_root,
        roots=roots,
        source_control=source_control,
        environment=environment,
        determinism=determinism,
        forbidden_ids=forbidden_ids,
    )
