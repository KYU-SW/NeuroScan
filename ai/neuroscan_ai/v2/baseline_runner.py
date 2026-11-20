"""v2 2D baseline training runner (run-control profile ``baseline_v1``).

Order (``run_baseline``), nothing opens data or builds a model before step 8:
1 load ``configs/v2/2d_baseline.yaml`` 2 ``validate_policy_config`` 3 architecture / parameter count
(analytic, no model built) 4 every hyperparameter and the operational blocks (cache, progress,
resume, resource monitor, host memory guard) 5 RunPlan 6 pre-run manifest 7 gate approval
8 ``BaselineRunner.run``: approval re-verified, the disk cache passes its integrity gate
(``baseline_cache.verify_cache``) against the pre-run input hashes, then the memory-mapped
datasets and the model are created.

Data: ``disk_backed_preprocessed_cache_v1`` (built beforehand by
``scripts/prepare_v2_2d_cache.py``). Only small per-case metadata stays in RAM; at most 2 cases'
memory maps are open; a batch copies
only its slices. Per-epoch validation keeps one case's original-grid arrays at a time.

Training: AdamW, cosine schedule stepped per epoch, masked Dice + masked BCE (1:1), gradient
clipping (L2, 1.0), deterministic brain-slice sampling (positive fraction 0.5), checkpoint
selection on validation at threshold 0.5 (ungated, ``baseline_selection.select_checkpoint``),
early stopping after 20 epochs without a new selected epoch. Final validation outputs are written
on the original grid from the selected (best) model. Threshold search is not run here.

After EVERY completed epoch (operational records, never part of a result hash):
``training/resume_checkpoint.pt`` (atomic, ``baseline_resume``) then ``training/progress.json``
(atomic, ``baseline_progress``) with RAM / VRAM readings (``resource_monitor``). Then, if more
epochs remain, the host memory guard (``host_memory_guard_v1``: available RAM < 4 GiB) and any
SIGINT / SIGTERM request are checked; either stops the run safely: an interruption record and a
segment post-run manifest are written (status ``interrupted_resource_guard`` / ``interrupted``,
``resume_allowed: true``). Nothing resumes automatically and nothing (batch size, cache limit,
learning rate, ...) is adjusted. ``resume_baseline`` continues such a run only when explicitly
called and when every identity check passes (``run_control.resume_controlled`` +
``resume_preflight``); the resumed result equals an uninterrupted run (exact match, synthetic test).

A failure (any exception, incl. CUDA OOM or a deterministic-algorithm error) writes
``training/failure_record.json`` (stage, error type, redacted message, output hashes so far) and
re-raises; the lifecycle then writes the failed post-run manifest (a failed run is never
resumed). Nothing is retried or adjusted. A forced stop (second signal, SIGKILL, process reaper)
writes nothing; only the last completed epoch's resume checkpoint is then trusted.

Device policy ``device_policy_v2_baseline_cuda_required_v1``: the dataset profile is derived from
the plan's inputs (``dataset_profile``), not declared by the caller. A formal ISLES profile needs
device ``cuda`` (no cpu / cuda_if_available override, no CPU fallback); a refused device stops
before the gate (no run folder, run ID unused). Synthetic profiles may use any allowed device.

``epoch_limit`` exists for SYNTHETIC tests only: it is refused unless the plan's dataset is
``synthetic`` and it is recorded in the training history.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from neuroscan_ai.reproducibility import PathRoots
from neuroscan_ai.reproducibility.environment import capture_determinism
from neuroscan_ai.reproducibility.hashing import canonical_json_sha256, sha256_file
from neuroscan_ai.run_control import build_plan, run_controlled
from neuroscan_ai.run_control.gate import GateResult, verify_approval
from neuroscan_ai.run_control.lifecycle import (
    INTERRUPTED,
    INTERRUPTED_RESOURCE_GUARD,
    POST_RUN_NAME,
    SEGMENTS_DIR,
    RunInterrupted,
    resume_controlled,
)
from neuroscan_ai.run_control.outputs import (
    AGGREGATE_KIND,
    IMAGE_KINDS,
    METRIC_KIND,
    run_output_path,
)
from neuroscan_ai.run_control.plan import RunPlan
from neuroscan_ai.v2 import baseline_cache as bc
from neuroscan_ai.v2 import baseline_dataset as bd
from neuroscan_ai.v2 import baseline_inference as bi
from neuroscan_ai.v2 import baseline_progress as bp
from neuroscan_ai.v2 import baseline_resume as brs
from neuroscan_ai.v2 import baseline_training as bt
from neuroscan_ai.v2 import resource_monitor as rm
from neuroscan_ai.v2.baseline_preprocess import (
    BRAIN_MASK_POLICY,
    PREPROCESS_POLICY,
    validate_policy_config,
)
from neuroscan_ai.v2.baseline_selection import select_checkpoint
from neuroscan_ai.v2.outputs import write_aggregate, write_case_outputs, write_json_new
from neuroscan_ai.v2.safety_metrics import POSITIVE_CASE
from neuroscan_ai.v2.unet_2d import CHANNELS, EXPECTED_PARAMETERS, MODEL_ID, build_model, describe

RUNNER_KIND = "v2_2d_baseline_runner"
CHECKPOINT_SCHEMA = "neuroscan-v2-baseline-checkpoint-v1"
INTERRUPTION_SCHEMA = "neuroscan-v2-baseline-interruption-v1"
SYNTHETIC_DATASET = bt.PROFILE_SYNTHETIC
TIMING_NAME = "training/timing_record.json"  # wall-clock audit; not a hashed output
INTERRUPTION_NAME = "interruption_record.json"  # inside segments/segment_NN/
OPERATIONAL_BLOCKS: dict[str, dict[str, Any]] = {
    "cache": bc.CACHE_POLICY,
    "progress": bp.PROGRESS_POLICY,
    "resume": brs.RESUME_POLICY,
    "resource_monitor": rm.RESOURCE_MONITOR,
    "host_memory_guard": rm.HOST_MEMORY_GUARD,
}
_ABS_PATH = re.compile(r"([A-Za-z]:[\\/])|(^|[\s'\"(])/[^\s]")


def validate_operational_config(cfg: Mapping[str, Any]) -> None:
    """Cache / progress / resume / resource / guard blocks must equal the code (no tuning)."""
    errs = [k for k, v in OPERATIONAL_BLOCKS.items() if cfg.get(k) != v]
    if errs:
        raise bt.ConfigValidationError("operational config differs from code: " + ", ".join(errs))


def prepare_config(config_path: Path) -> dict[str, Any]:
    """Steps 1-4 (before any plan, gate, data or model)."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    validate_policy_config(cfg)
    bt.validate_architecture(cfg)
    bt.validate_hyperparameters(cfg)
    validate_operational_config(cfg)
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


def _segment_count(run_dir: Path) -> int:
    d = Path(run_dir) / SEGMENTS_DIR
    return sum(1 for p in d.iterdir() if p.is_dir()) if d.is_dir() else 0


@dataclass
class BaselineRunner:
    config: Mapping[str, Any]
    cache_dir: Path  # verified disk cache (never written by the runner)
    device: str | None = None  # execution device override (default: config)
    epoch_limit: int | None = None  # synthetic tests only
    dataset_profile: str = bt.PROFILE_FORMAL  # set by run_baseline; formal unless proven synthetic
    forbidden_ids: tuple[str, ...] = ()
    run_seed: int | None = None  # explicit run seed (formal: required, from FORMAL_SEEDS)
    kind: str = RUNNER_KIND
    training_outputs: str = "baseline_v1"
    memory_probe: Callable[[], int | None] = rm.system_available_memory_bytes
    handle_signals: bool = True
    _stage: str = field(default="not_started", init=False)
    _preflight_cache: Any = field(default=None, init=False, repr=False)

    @property
    def seed(self) -> int:
        """The one seed of this run (``SEED_POLICY``): the explicit run seed; only a synthetic run
        without one falls back to the config reference seed."""
        if self.run_seed is not None:
            return int(self.run_seed)
        return int(self.config["training"]["seed"])

    def seed_record(self) -> dict[str, Any]:
        """Seed fields written next to every seeded record (RunPlan / runner / checkpoint seeds
        must agree; a run seed other than the config reference is the intended explicit seed)."""
        return {
            "seed_policy": bt.SEED_POLICY,
            "run_seed": self.seed,
            "config_reference_seed": bt.FIXED_TRAINING["seed"],
        }

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

    # ------------------------------------------------------------ run-control entry points
    def run(self, gate: GateResult, plan: RunPlan) -> None:
        verify_approval(gate)  # before any data or model is touched
        self._guarded(gate, plan, resume=False)

    def resume(self, gate: GateResult, plan: RunPlan) -> None:
        """Explicit resume (``resume_baseline`` -> ``resume_controlled`` -> preflight first)."""
        verify_approval(gate)
        self._guarded(gate, plan, resume=True)

    def _guarded(self, gate: GateResult, plan: RunPlan, resume: bool) -> None:
        try:
            with brs.InterruptController(self.handle_signals) as ctl:
                self._run(gate, plan, ctl, resume)
        except RunInterrupted:
            raise
        except Exception as e:  # a forced KeyboardInterrupt / SystemExit writes nothing
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

    # ------------------------------------------------------------ cache / identity
    def _verify_cache(self, plan: RunPlan, pre: Mapping[str, Any]) -> bc.VerifiedCache:
        return bc.verify_cache(
            self.cache_dir,
            config=self.config,
            split_file=plan.split_file,
            train_aliases=plan.aliases,
            val_aliases=plan.validation_aliases,
            input_sha256=pre.get("inputs", {}).get("file_sha256", {}),
            package_lock_sha256_lf=pre.get("environment", {}).get("package_lock_sha256_lf"),
            git_commit=pre.get("source_control", {}).get("git_commit"),
        )

    def _n_epochs(self) -> tuple[int, int]:
        max_epochs = int(self.config["training"]["max_epochs"])
        n = min(max_epochs, self.epoch_limit) if self.epoch_limit else max_epochs
        return max_epochs, n

    def _identity(self, plan: RunPlan, pre: Mapping[str, Any], index_sha: str) -> dict[str, Any]:
        split = json.loads(Path(plan.split_file).read_text(encoding="utf-8"))
        env = dict(pre.get("environment", {}))
        inputs = pre.get("inputs", {})
        return {
            "schema": brs.RESUME_SCHEMA,
            "run_id": plan.run_id,
            "seed": self.seed,
            "epoch_limit": self.epoch_limit,
            "max_epochs": self._n_epochs()[0],
            "config_canonical_sha256": canonical_json_sha256(dict(self.config)),
            "split_canonical_sha256": canonical_json_sha256(split),
            "alias_list_sha256": inputs.get("alias_list_sha256"),
            "input_aggregate_sha256": canonical_json_sha256(inputs.get("file_sha256", {})),
            "cache_index_sha256": index_sha,
            "package_lock_sha256_lf": env.get("package_lock_sha256_lf"),
            "git_commit": pre.get("source_control", {}).get("git_commit"),
            "environment": env,
            "determinism": dict(pre.get("determinism", {})),
            "model_id": MODEL_ID,
            "parameter_count": EXPECTED_PARAMETERS,
            "preprocessing_policy": PREPROCESS_POLICY,
            "brain_mask_policy": BRAIN_MASK_POLICY,
            "threshold_policy": bt.THRESHOLD_SOURCE,
            "sampling_policy": bd.SAMPLING_POLICY,
        }

    def resume_preflight(self, gate: GateResult, plan: RunPlan) -> list[str]:
        """Every runner-side resume condition, checked BEFORE the lifecycle starts the segment
        (a refusal here writes nothing). Returns reason codes (empty = resume allowed)."""
        run_dir = gate.run_dir
        pre = gate.pre_run or {}
        if plan.run_id in brs.RETIRED_RUN_IDS:
            return ["run_id_retired"]
        reasons = []
        if (run_dir / POST_RUN_NAME).exists():
            reasons.append("run_already_finalised")
        if run_output_path(run_dir, "failure_record").exists():
            reasons.append("run_failed_not_resumable")
        if not (run_dir / brs.RESUME_NAME).is_file():
            reasons.append("resume_checkpoint_missing")
        if not (run_dir / bp.PROGRESS_NAME).is_file():
            reasons.append("progress_missing")
        finals = [
            label
            for label in (
                "best_checkpoint",
                "last_checkpoint",
                "training_history",
                "validation_metrics",
                "sampler_record",
                "threshold_record",
            )
            if run_output_path(run_dir, label).exists()
        ]
        if (
            finals
            or (run_dir / AGGREGATE_KIND).exists()
            or any((run_dir / k).exists() for k in (*IMAGE_KINDS, METRIC_KIND))
        ):
            reasons.append("final_outputs_present")
        if not run_output_path(run_dir, "preprocessing_record").is_file():
            reasons.append("preprocessing_record_missing")
        if reasons:
            return reasons
        try:
            payload = brs.load_resume_checkpoint(run_dir / brs.RESUME_NAME)
            progress = bp.read_progress(run_dir / bp.PROGRESS_NAME)
        except brs.ResumeRefused as e:
            return list(e.reasons)
        except (OSError, ValueError):
            return ["progress_unreadable"]
        try:
            verified = self._verify_cache(plan, pre)
        except bc.CacheError as e:
            return ["cache_integrity_failed", *e.reasons[:20]]
        identity = self._identity(plan, pre, verified.index_sha256)
        reasons = brs.check_resume_identity(payload, identity)
        done = payload.get("completed_epoch")
        if progress.get("run_id") != plan.run_id or progress.get("completed_epoch") != done:
            reasons.append("progress_checkpoint_epoch_mismatch")
        if progress.get("status") == "training_completed":
            reasons.append("training_already_completed")
        reasons += self._state_reasons(payload, verified, plan)
        self._preflight_cache = verified if not reasons else None
        return reasons

    def _state_reasons(self, payload, verified, plan) -> list[str]:
        """Sampler / history / selection consistency of a resume payload."""
        reasons = []
        done = payload.get("completed_epoch")
        if not isinstance(done, int) or done < 1 or payload.get("next_epoch") != done + 1:
            return ["resume_epoch_invalid"]
        n = len(payload.get("history", []))
        if n != done or len(payload.get("selection_records", [])) != done:
            reasons.append("history_length_mismatch")
        sampling = payload.get("sampling", [])
        if payload.get("run_seed") != self.seed:
            reasons.append("checkpoint_run_seed_mismatch")
        if any(
            not isinstance(rec.get("sampling_seed"), list)
            or rec["sampling_seed"][:1] != [self.seed]
            for rec in sampling
        ):
            reasons.append("sampler_seed_mismatch")
        if len(sampling) != done:
            reasons.append("sampler_state_length_mismatch")
        else:
            train, _ = bc.open_cases(verified, bc.CacheStore(verified.root))
            frac = float(self.config["training"]["sampling"]["positive_fraction"])
            for rec in sampling:
                _, regen = bd.epoch_sampling(train, self.seed, rec.get("epoch"), frac)
                if regen != rec:
                    reasons.append("sampler_state_invalid")
                    break
            if sampling and payload.get("last_sampling_list_sha256") != sampling[-1].get(
                "slice_list_sha256"
            ):
                reasons.append("sampling_list_hash_mismatch")
        records = payload.get("selection_records", [])
        if records and select_checkpoint(records) != payload.get("best_selection"):
            reasons.append("selection_state_mismatch")
        return reasons

    # ------------------------------------------------------------ training
    def _run(self, gate: GateResult, plan: RunPlan, ctl=None, resume: bool = False) -> None:
        cfg = self.config
        tr = cfg["training"]
        if plan.dataset_id != SYNTHETIC_DATASET:  # defence in depth (run_baseline checks first)
            self.dataset_profile = bt.PROFILE_FORMAL
        effective = bt.check_device_policy(
            self.dataset_profile, tr["device"], self.device, self.epoch_limit
        )
        bt.check_run_seed(self.dataset_profile, self.run_seed)
        if plan.seed is not None and plan.seed != self.seed:
            raise bt.ConfigValidationError("run plan seed differs from the runner seed")
        run_dir = gate.run_dir
        pre = gate.pre_run or {}
        device = bt.resolve_device(effective)
        bt.apply_determinism(self.seed)
        t0_run = time.perf_counter()
        needles = (*plan.all_aliases, *self.forbidden_ids)

        self._stage = "cache_integrity"
        verified = self._preflight_cache if resume and self._preflight_cache else None
        verified = verified or self._verify_cache(plan, pre)
        store = bc.CacheStore(
            verified.root, cfg["cache"]["mmap_mode"], int(cfg["cache"]["max_active_cases"])
        )
        self._stage = "data_loading"
        train, val = bc.open_cases(verified, store)
        if not any(c.ground_truth_status == POSITIVE_CASE for c in val):
            raise bi.MissingValidationResultError("all validation cases are lesion-free; refused")
        prep = {
            "preprocess_policy": PREPROCESS_POLICY,
            "brain_mask_policy": BRAIN_MASK_POLICY,
            "train_cases": [c.record for c in train],
            "validation_cases": [c.record for c in val],
            "inputs_modified": False,
            "cache": {
                "policy_id": bc.CACHE_POLICY_ID,
                "cache_index_sha256": verified.index_sha256,
                "transpose_policy_id": bc.TRANSPOSE_POLICY,
                "max_active_cases": store.max_active_cases,
            },
        }
        prep_path = run_output_path(run_dir, "preprocessing_record")
        if resume:
            stored = json.loads(prep_path.read_text(encoding="utf-8"))
            if canonical_json_sha256(stored) != canonical_json_sha256(prep):
                raise brs.ResumeRefused(["preprocessing_record_mismatch"])
        else:
            write_json_new(prep_path, prep)

        self._stage = "model_build"
        model = build_model(self.seed).to(device)
        info = describe(model, self.seed)
        if info["trainable_parameters"] != EXPECTED_PARAMETERS:
            raise bt.ConfigValidationError("parameter count differs from approval")
        opt = bt.build_optimizer(model)
        sched = bt.build_scheduler(opt)
        max_epochs, n_epochs = self._n_epochs()
        patience = int(tr["early_stopping_patience"])
        batch = int(tr["batch_size"])
        frac = float(tr["sampling"]["positive_fraction"])
        identity = self._identity(plan, pre, verified.index_sha256)
        st: dict[str, Any] = {
            "history": [],
            "sampling": [],
            "records": [],
            "timing": [],
            "best_state": None,
            "best_sel": None,
            "elapsed_before": 0.0,
        }
        start_epoch, finished, stopped_early = 1, False, False
        if resume:
            self._stage = "resume_load"
            payload = brs.load_resume_checkpoint(run_dir / brs.RESUME_NAME)
            reasons = brs.check_resume_identity(payload, identity)
            reasons += self._state_reasons(payload, verified, plan)
            if reasons:
                raise brs.ResumeRefused(reasons)
            model.load_state_dict(payload["model_state"])
            opt.load_state_dict(payload["optimizer_state"])
            sched.load_state_dict(payload["scheduler_state"])
            st.update(
                history=payload["history"],
                sampling=payload["sampling"],
                records=payload["selection_records"],
                timing=payload["timing"],
                best_state=payload["best_state"],
                best_sel=payload["best_selection"],
                elapsed_before=float(payload["elapsed_seconds"]),
            )
            start_epoch = payload["next_epoch"]
            finished, stopped_early = payload["training_finished"], payload["stopped_early"]
            brs.restore_rng(payload["rng_state"])
        rm.reset_cuda_peak(device)

        self._stage = "training_loop"
        epoch = start_epoch - 1
        for epoch in range(start_epoch, n_epochs + 1):
            if finished:
                break
            t0 = time.perf_counter()
            lr = float(opt.param_groups[0]["lr"])
            order, srec = bd.epoch_sampling(train, self.seed, epoch, frac)
            st["sampling"].append(srec)
            tm = bt.train_one_epoch(model, bd.batches(train, order, batch), opt, device)
            sched.step()
            vloss = bi.validation_loss(model, val, device, batch)
            ev = bi.evaluate(model, val, device, batch, keep_outputs=False)
            u = ev["ungated_summary"]
            st["records"].append(
                {
                    "epoch": epoch,
                    "mean_dice_positive": u["mean_dice_positive"],
                    "val_loss": vloss,
                    "false_positive_voxel_volume_ml": u["false_positive_voxel_volume_ml"],
                    "threshold": bt.THRESHOLD,
                    "output": "ungated",
                }
            )
            sel = select_checkpoint(st["records"])
            if sel["selected_epoch"] == epoch:
                st["best_state"] = _snapshot(model, opt, sched)
                st["best_sel"] = sel
            st["history"].append(
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
            del ev
            st["timing"].append({"epoch": epoch, "duration_seconds": time.perf_counter() - t0})
            stopped_early = bt.early_stop(epoch, sel["selected_epoch"], patience)
            finished = stopped_early or epoch == n_epochs
            elapsed = st["elapsed_before"] + time.perf_counter() - t0_run
            ckpt_sha = self._save_resume(
                run_dir,
                identity,
                model,
                opt,
                sched,
                st,
                epoch,
                srec,
                finished,
                stopped_early,
                elapsed,
            )
            status, guard = "running", None
            if not finished:
                guard = rm.check_host_memory(self.memory_probe())
                if guard["triggered"]:
                    status = INTERRUPTED_RESOURCE_GUARD
                elif ctl is not None and ctl.requested:
                    status = INTERRUPTED
            self._write_progress(run_dir, plan, status, st, device, elapsed, needles)
            if status != "running":
                self._interrupt(run_dir, plan, status, epoch, ctl, guard, ckpt_sha, needles)
            if finished:
                break
        if not st["history"]:
            raise bt.TrainingStop("no completed epoch")
        last_epoch = st["history"][-1]["epoch"]
        last_state = _snapshot(model, opt, sched)
        best_state, best_sel = st["best_state"], st["best_sel"]

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
                "seed": self.seed,
                **self.seed_record(),
                "hyperparameters": {**bt.FIXED_TRAINING, "seed": self.seed},
                "max_epochs": max_epochs,
                "epochs_run": last_epoch,
                "synthetic_epoch_limit": self.epoch_limit,
                "dataset_profile": self.dataset_profile,
                "device_policy": bt.DEVICE_POLICY,
                "collapse_policy": bt.FIXED_COLLAPSE_POLICY,
                "early_stopping_patience": patience,
                "stopped_early": stopped_early,
                "selection": best_sel,
                "epochs": st["history"],
                "duration_recorded_in": TIMING_NAME + " (wall clock; excluded from exact match)",
            },
        )
        write_json_new(run_dir / TIMING_NAME, {"epochs": st["timing"]})
        write_json_new(
            run_output_path(run_dir, "sampler_record"),
            {
                "policy": bd.SAMPLING_POLICY,
                "seed": self.seed,
                **self.seed_record(),
                "epochs": st["sampling"],
            },
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
        del ev
        elapsed = st["elapsed_before"] + time.perf_counter() - t0_run
        self._write_progress(run_dir, plan, "training_completed", st, device, elapsed, needles)
        store.close()

    # ------------------------------------------------------------ operational records
    def _save_resume(
        self,
        run_dir,
        identity,
        model,
        opt,
        sched,
        st,
        epoch,
        srec,
        finished,
        stopped_early,
        elapsed,
    ) -> str:
        best = st["best_sel"]["selected_epoch"]
        snap = _snapshot(model, opt, sched)
        payload = {
            **identity,
            **self.seed_record(),
            "completed_epoch": epoch,
            "next_epoch": epoch + 1,
            "training_finished": finished,
            "stopped_early": stopped_early,
            "model_state": snap["model"],
            "optimizer_state": snap["optimizer"],
            "scheduler_state": snap["scheduler"],
            "best_state": st["best_state"],
            "best_selection": st["best_sel"],
            "best_epoch": best,
            "early_stopping_counter": epoch - best,
            "selection_records": st["records"],
            "history": st["history"],
            "sampling": st["sampling"],
            "last_sampling_list_sha256": srec["slice_list_sha256"],
            "timing": st["timing"],
            "elapsed_seconds": float(elapsed),
            "rng_state": brs.capture_rng(),
            "created_at_utc": bp.utc_now(),
        }
        return brs.save_resume_checkpoint(run_dir / brs.RESUME_NAME, payload)

    def _write_progress(self, run_dir, plan, status, st, device, elapsed, needles) -> None:
        h = st["history"][-1]
        best = st["best_sel"]["selected_epoch"]
        host = rm.host_memory()
        cuda = rm.cuda_memory(device)
        record = {
            "schema": bp.PROGRESS_SCHEMA,
            "run_id": plan.run_id,
            "status": status,
            "completed_epoch": h["epoch"],
            "max_epochs": self._n_epochs()[1],
            "best_epoch": best,
            "early_stopping_counter": h["epoch"] - best,
            "latest_train_loss_total": h["train_loss_total"],
            "latest_train_loss_dice": h["train_loss_dice"],
            "latest_train_loss_bce": h["train_loss_bce"],
            "latest_validation_loss": h["validation_loss_total"],
            "latest_validation_mean_positive_case_dice": h["validation_mean_positive_case_dice"],
            "latest_false_positive_voxel_volume_ml": h[
                "validation_total_false_positive_voxel_volume_ml"
            ],
            "latest_lesion_wise_recall": h["validation_lesion_wise_recall"],
            "learning_rate": h["learning_rate"],
            "gradient_norm": h["gradient_norm"],
            "sampled_positive_ratio": h["sampled_positive_ratio"],
            "epoch_duration_seconds": st["timing"][-1]["duration_seconds"],
            "elapsed_seconds": float(elapsed),
            **host,
            **cuda,
            "updated_at_utc": bp.utc_now(),
        }
        bp.write_progress(run_dir / bp.PROGRESS_NAME, record, needles)

    def _interrupt(self, run_dir, plan, status, epoch, ctl, guard, ckpt_sha, needles) -> None:
        seg = run_dir / SEGMENTS_DIR / f"segment_{_segment_count(run_dir) + 1:02d}"
        seg.mkdir(parents=True)
        reason = (
            "host_memory_guard"
            if status == INTERRUPTED_RESOURCE_GUARD
            else (ctl.reason if ctl is not None else None)
        )
        record = {
            "schema": INTERRUPTION_SCHEMA,
            "run_id": plan.run_id,
            "status": status,
            "reason": reason,
            "completed_epoch": epoch,
            "next_epoch": epoch + 1,
            "resume_allowed": True,
            "automatic_resume": False,
            "retried": False,
            "batch_size_changed": False,
            "cache_limit_changed": False,
            "host_memory_guard": guard,
            "resume_checkpoint_sha256": ckpt_sha,
            "progress_sha256": sha256_file(run_dir / bp.PROGRESS_NAME),
            "uncatchable_stops": "SIGKILL / process reaper: nothing written; last completed "
            "epoch's resume checkpoint only",
            "created_at_utc": bp.utc_now(),
        }
        text = json.dumps(record, indent=2)
        bp.check_public_text(text, needles)
        write_json_new(seg / INTERRUPTION_NAME, record)
        raise RunInterrupted(
            status,
            seg,
            {
                "segment": seg.name,
                "reason": reason,
                "completed_epoch": epoch,
                "resume_allowed": True,
                "interruption_record_sha256": hashlib.sha256(
                    (seg / INTERRUPTION_NAME).read_bytes()
                ).hexdigest(),
                "resume_checkpoint_sha256": ckpt_sha,
            },
        )

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
            **self.seed_record(),
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
            "seed": self.seed,
            **self.seed_record(),
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


def _cpu_copy(obj: Any) -> Any:
    """Deep copy with every tensor on the CPU, so a snapshot (and any checkpoint written from it)
    is the same whether it stayed in memory or went through a resume checkpoint."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu_copy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_copy(v) for v in obj)
    return copy.deepcopy(obj)


def _snapshot(model, opt, sched) -> dict[str, Any]:
    return {
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": _cpu_copy(opt.state_dict()),
        "scheduler": copy.deepcopy(sched.state_dict()),
    }


def _build(
    config_path,
    plan_kwargs,
    cache_dir,
    repo_root,
    forbidden_ids,
    device,
    epoch_limit,
    memory_probe,
    seed=None,
):
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
    # run seed (SEED_POLICY): explicit and approved for formal runs; the caller's seed and the
    # RunPlan seed must agree. Refused here: before the gate (no folder, run ID unused).
    bt.check_run_seed(profile, seed)
    if plan.seed != seed:
        raise bt.ConfigValidationError("seed differs between the command line and the RunPlan")
    runner = BaselineRunner(
        cfg,
        Path(cache_dir),
        device=device,
        epoch_limit=epoch_limit,
        forbidden_ids=tuple(forbidden_ids),
        run_seed=seed,
        dataset_profile=profile,
        **({"memory_probe": memory_probe} if memory_probe is not None else {}),
    )
    return plan, runner


def run_baseline(
    config_path: Path,
    plan_kwargs: Mapping[str, Any],
    cache_dir: Path,
    *,
    repo_root: Path,
    roots: PathRoots,
    source_control: Mapping[str, Any],
    environment: Mapping[str, Any],
    determinism: Mapping[str, Any],
    forbidden_ids: Iterable[str] = (),
    device: str | None = None,
    epoch_limit: int | None = None,
    memory_probe: Callable[[], int | None] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Config checks (1-4) -> RunPlan (5) -> gate + manifests (6-7) -> runner (8)."""
    plan, runner = _build(
        config_path,
        plan_kwargs,
        cache_dir,
        repo_root,
        forbidden_ids,
        device,
        epoch_limit,
        memory_probe,
        seed,
    )
    if plan.run_id in brs.RETIRED_RUN_IDS:
        raise bt.ConfigValidationError("run ID is retired and may not be reused")
    # cache commit binding BEFORE the gate: a refusal creates nothing and keeps the run ID unused
    reasons = bc.check_cache_commit(cache_dir, source_control)
    if reasons:
        raise bc.CacheCommitError(reasons)
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


def resume_baseline(
    config_path: Path,
    plan_kwargs: Mapping[str, Any],
    cache_dir: Path,
    *,
    repo_root: Path,
    source_control: Mapping[str, Any],
    environment: Mapping[str, Any],
    determinism: Mapping[str, Any],
    forbidden_ids: Iterable[str] = (),
    device: str | None = None,
    epoch_limit: int | None = None,
    memory_probe: Callable[[], int | None] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """EXPLICIT resume of an interrupted run (same run ID, same plan). Never called
    automatically; raises ``run_control.lifecycle.ResumeRefusedError`` on any mismatch."""
    plan, runner = _build(
        config_path,
        plan_kwargs,
        cache_dir,
        repo_root,
        forbidden_ids,
        device,
        epoch_limit,
        memory_probe,
        seed,
    )
    if plan.run_id in brs.RETIRED_RUN_IDS:
        from neuroscan_ai.run_control.lifecycle import ResumeRefusedError  # noqa: PLC0415

        raise ResumeRefusedError(["run_id_retired"])
    return resume_controlled(
        plan,
        runner,
        repo_root=repo_root,
        source_control=source_control,
        environment=environment,
        determinism=determinism,
        forbidden_ids=forbidden_ids,
    )
