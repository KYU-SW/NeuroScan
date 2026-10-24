"""Phase 5A service: one DWI + ADC pair -> mask NIfTI + quantification / metadata / manifest JSON.

The same function serves the CLI (scripts/infer_case.py) and, later, a web API. It never prints,
never imports web or database code, and returns a JSON-serialisable ``InferenceResult`` instead of
raising for expected failures. Order:

  sealed test guard (before any input is opened) -> input validation -> device + verified model
  -> in-memory preprocessing -> 2.5D prediction -> original-space restoration -> quantification
  -> de-identified input copies + mask + JSON (staging, verified) -> atomic commit
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import subprocess
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import scipy
import torch
import yaml

import neuroscan_ai
from neuroscan_ai.inference import schemas
from neuroscan_ai.inference.errors import EXIT_CODES, Code, InferenceError, Issue
from neuroscan_ai.inference.model_loader import LoadedModel, load_inference_model, resolve_device
from neuroscan_ai.inference.outputs import RunDirectory, new_run_id, sha256_path
from neuroscan_ai.inference.predict import predict
from neuroscan_ai.inference.preprocess import load_preprocess_config, prepare_case
from neuroscan_ai.inference.quantify import quantify
from neuroscan_ai.inference.sealed_guard import check_not_sealed
from neuroscan_ai.inference.validation import raise_first_error, validate_inputs
from neuroscan_ai.io import nifti
from neuroscan_ai.training.brain_gating import BrainMaskError
from neuroscan_ai.training.seed import seed_everything

LOGGER = logging.getLogger("neuroscan_ai.inference")
REPO_ROOT = Path(neuroscan_ai.__file__).resolve().parents[2]
DEFAULT_INFERENCE_CONFIG = REPO_ROOT / "configs" / "inference_v1.yaml"
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
INTENDED_USE = {
    "purpose": "research and medical image review support",
    "not_for": ["establishing a diagnosis", "treatment decisions", "prognosis"],
    "clinician_review_required": True,
    "scope": (
        "skull-stripped DWI (b1000) + ADC similar to ISLES 2022; other sites or scanners may "
        "show domain shift"
    ),
    "scores": "uncalibrated model output scores; not probabilities of disease",
    "disclaimer": schemas.DISCLAIMER,
}
SHAREABLE = [
    "prediction_mask.nii.gz",
    "quantification.json",
    "inference_metadata.json",
    "manifest.json",
]
NOT_SHAREABLE = ["inputs/", "logs/"]

Progress = Callable[[str, float], None]


@dataclass
class InferenceResult:
    status: str  # "succeeded" | "failed"
    exit_code: int
    case_id: str | None
    run_id: str | None
    run_dir: str | None
    outputs: dict[str, str] = field(default_factory=dict)
    quantification: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    warnings: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "case_id": self.case_id,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "outputs": self.outputs,
            "quantification": self.quantification,
            "warnings": self.warnings,
            "error": self.error,
        }


class _Redact(logging.Filter):
    """Replace known input paths and any absolute path in log records."""

    def __init__(self, tokens: dict[str, str]) -> None:
        super().__init__()
        self.tokens = {k: v for k, v in tokens.items() if k}

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for raw, token in self.tokens.items():
            msg = msg.replace(raw, token)
        msg = re.sub(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\s'\"]*", "<path>", msg)
        msg = re.sub(r"(?<![\w.])/(?:[\w.-]+/)+[\w.-]*", "<path>", msg)
        record.msg, record.args = msg, ()
        return True


def _utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _display(p: Path) -> str:
    """Repository-relative path, or 'external:<name>' — never an absolute local path."""
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"external:{p.name}"


def _git() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return {"commit": head, "tracked_changes": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tracked_changes": None}


def _safe_traceback(exc: BaseException) -> str:
    frames = traceback.extract_tb(exc.__traceback__)
    lines = [f"  {Path(f.filename).name}:{f.lineno} in {f.name}" for f in frames]
    return f"{type(exc).__name__}\n" + "\n".join(lines)


def _views(orientation: str, shape) -> dict[str, Any]:
    names = {
        "S": "axial",
        "I": "axial",
        "A": "coronal",
        "P": "coronal",
        "L": "sagittal",
        "R": "sagittal",
    }
    return {
        names[c]: {"array_axis": i, "count": int(shape[i]), "axis_increases_toward": c}
        for i, c in enumerate(orientation)
    }


def _volume_entry(vid: str, role: str, uri: str, data: np.ndarray, affine: np.ndarray, sha: str):
    return {
        "id": vid,
        "role": role,
        "uri": uri,
        "sha256": sha,
        "format": "nifti1",
        "compression": "gzip",
        "mime_type": "application/gzip",
        "shape": [int(s) for s in data.shape],
        "spacing_mm": [round(s, 6) for s in nifti.spacing_from_affine(affine)],
        "affine": np.round(affine, 6).tolist(),
        "orientation": nifti.orientation(affine),
        "dtype": str(data.dtype),
    }


def run_independent_inference(
    dwi_path: str | Path,
    adc_path: str | Path,
    output_root: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    inference_config_path: str | Path | None = None,
    case_id: str | None = None,
    device: str = "auto",
    strict: bool = False,
    loaded_model: LoadedModel | None = None,
    export_inputs: bool = True,
    record_input_names: bool = False,
    progress: Progress | None = None,
) -> InferenceResult:
    started_at, t_start = _utc(), time.perf_counter()
    notify = progress or (lambda stage, frac: None)

    # --- arguments and configuration (no output folder yet) -----------------------------------
    try:
        if case_id is None:
            case_id = f"case-{uuid.uuid4().hex[:12]}"
        elif not CASE_ID_PATTERN.fullmatch(case_id) or ".nii" in case_id.lower():
            raise InferenceError(
                Code.INVALID_ARGUMENT, "case_id may only contain letters, digits, '_' and '-'."
            )
        inf_path = (
            Path(inference_config_path) if inference_config_path else DEFAULT_INFERENCE_CONFIG
        )
        try:
            inf_cfg = yaml.safe_load(inf_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise InferenceError(
                Code.CONFIG_MISMATCH,
                "Inference configuration could not be read.",
                type(exc).__name__,
            ) from None
        run_id = new_run_id()
        run = RunDirectory(output_root, case_id, run_id)
    except InferenceError as exc:
        return InferenceResult(
            "failed", exc.exit_code, case_id, None, None, error=exc.issue.to_dict()
        )

    handler = logging.FileHandler(run.log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handler.addFilter(
        _Redact(
            {
                str(dwi_path): "<input:dwi>",
                str(adc_path): "<input:adc>",
                str(output_root): "<output>",
            }
        )
    )
    LOGGER.addHandler(handler)
    if LOGGER.level == logging.NOTSET or LOGGER.level > logging.INFO:
        LOGGER.setLevel(logging.INFO)

    warnings: list[Issue] = []
    timings: dict[str, float] = {}
    meta: dict[str, Any] = {
        "schema_version": schemas.METADATA_SCHEMA,
        "status": "running",
        "case_id": case_id,
        "run_id": run_id,
        "inference_started_at": started_at,
        "inference_completed_at": None,
        "runtime_seconds": None,
        "timings_seconds": timings,
        "model": {},
        "inputs": {},
        "input_qc": {},
        "geometry": {},
        "preprocessing": {},
        "inference": {},
        "device": {},
        "software": {
            "neuroscan_ai": neuroscan_ai.__version__,
            "git": _git(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "nibabel": nib.__version__,
        },
        "test_split_guard": {},
        "validation": {"errors": [], "warnings": [], "proceeded": False, "strict": strict},
        "outputs": {},
        "ground_truth_used": False,
        "intended_use": INTENDED_USE,
    }

    def stage(name: str, frac: float, t0: float) -> float:
        now = time.perf_counter()
        timings[name] = round(now - t0, 3)
        LOGGER.info("stage %s done (%.2fs)", name, now - t0)
        notify(name, frac)
        return now

    report = None
    try:
        LOGGER.info("run %s case %s started", run_id, case_id)
        t = time.perf_counter()
        meta["test_split_guard"] = check_not_sealed(
            [dwi_path, adc_path], _repo_path(inf_cfg["split_file"])
        )
        t = stage("sealed_guard", 0.05, t)

        report = validate_inputs(dwi_path, adc_path, inf_cfg)
        warnings.extend(report.warnings)
        meta["validation"]["errors"] = [i.to_dict() for i in report.errors]
        meta["input_qc"] = report.stats
        raise_first_error(report)
        if strict and report.warnings:
            raise InferenceError(
                Code.STRICT_WARNING,
                "Input warnings are treated as errors in strict mode.",
                ", ".join(w.code.value for w in report.warnings),
            )
        dwi, adc = report.dwi, report.adc
        for item in (dwi, adc):
            entry = {"sha256": item.sha256, "format": item.format, "size_bytes": item.size_bytes}
            if record_input_names:
                entry["file_name"] = item.basename
            meta["inputs"][item.role] = entry
        t = stage("validation", 0.15, t)

        if loaded_model is None:
            dev, dev_issues = resolve_device(device)
            warnings.extend(dev_issues)
            m = inf_cfg["model"]
            loaded_model = load_inference_model(
                _repo_path(checkpoint_path or m["checkpoint"]),
                _repo_path(config_path or m["training_config"]),
                dev,
                m,
            )
        lm = loaded_model
        meta["model"] = {
            "model_name": inf_cfg["model"]["model_name"],
            "model_type": inf_cfg["model"]["model_type"],
            "training_run_id": lm.info.get("run_id"),
            "checkpoint_epoch": lm.info.get("epoch"),
            "validation_positive_mean_dice": lm.info.get("val_mean_dice"),
            "validation_dice_note": (
                "internal validation, used for checkpoint selection (optimistic)"
            ),
            "checkpoint_path": _display(lm.checkpoint_path),
            "checkpoint_sha256": lm.checkpoint_sha256,
            "config_path": _display(lm.config_path),
            "config_sha256": lm.config_sha256,
            "config_hash": lm.config_hash,
            "inference_config": _display(inf_path),
        }
        meta["device"] = {
            "type": lm.device.type,
            "name": torch.cuda.get_device_name(lm.device) if lm.device.type == "cuda" else "cpu",
            "requested": device,
        }
        t = stage("load_model", 0.25, t)

        pre_cfg = load_preprocess_config(_repo_path(inf_cfg["preprocessing_config"]))
        try:
            prepared = prepare_case(
                case_id, dwi.data, adc.data, dwi.affine, pre_cfg, lm.training_cfg
            )
        except BrainMaskError as exc:
            raise InferenceError(
                Code.BRAIN_MASK,
                "Brain region could not be identified after preprocessing.",
                str(exc),
            ) from None
        except Exception as exc:
            raise InferenceError(
                Code.PREPROCESS, "Image preprocessing failed.", type(exc).__name__
            ) from None
        meta["geometry"] = {
            "original": {
                "shape": list(prepared.original.shape),
                "spacing_mm": [round(s, 6) for s in prepared.original.spacing],
                "orientation": prepared.original.orientation,
                "affine": np.round(prepared.original.affine, 6).tolist(),
            },
            "processed": {
                "shape": list(prepared.processed.shape),
                "spacing_mm": [round(s, 6) for s in prepared.processed.spacing],
                "orientation": prepared.processed.orientation,
            },
        }
        meta["preprocessing"] = {"version": pre_cfg.version, "transforms": prepared.transforms}
        t = stage("preprocess", 0.4, t)

        cfg = lm.training_cfg
        determinism = seed_everything(cfg["seed"], cfg["training"]["deterministic"])
        try:
            prediction = predict(lm, prepared, allow_cpu_fallback=(device == "auto"))
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(
                Code.INFERENCE, "Model inference failed.", type(exc).__name__
            ) from None
        warnings.extend(prediction.warnings)
        threshold = float(cfg["inference"]["threshold"])
        meta["inference"] = {
            "context_slices": int(cfg["data"]["context_slices"]),
            "input_channels": int(lm.info["in_channels"]),
            "channel_order": ["dwi z-1", "dwi z", "dwi z+1", "adc z-1", "adc z", "adc z+1"],
            "threshold": threshold,
            "brain_mask_gating": bool(cfg["inference"]["brain_mask_gating"]),
            "precision": cfg["training"]["precision"],
            "batch_size": prediction.batch_size,
            "attempts": prediction.attempts,
            "restoration": "processed -> original DWI grid, linear interpolation then >= 0.5",
            "small_component_removal": False,
            "determinism": determinism,
        }
        meta["device"]["used"] = prediction.device
        t = stage("predict_restore", 0.7, t)

        quant, q_issues = quantify(prediction, prepared, inf_cfg, threshold)
        warnings.extend(q_issues)
        quant = {
            "schema_version": schemas.QUANTIFICATION_SCHEMA,
            "case_id": case_id,
            "run_id": run_id,
            **quant,
            "disclaimer": schemas.DISCLAIMER,
        }
        t = stage("quantify", 0.8, t)

        mask = prediction.binary_original.data
        affine = prepared.original.affine
        run.write_nifti("prediction_mask.nii.gz", mask, affine, binary=True)
        volumes = []
        if export_inputs:
            for item in (dwi, adc):
                run.write_nifti(f"inputs/{item.role}.nii.gz", item.data, item.affine, binary=False)
                volumes.append(
                    _volume_entry(
                        item.role,
                        "image",
                        f"inputs/{item.role}.nii.gz",
                        item.data,
                        item.affine,
                        sha256_path(run.staging / "inputs" / f"{item.role}.nii.gz"),
                    )
                )
        mask_entry = _volume_entry(
            "prediction_mask",
            "label",
            "prediction_mask.nii.gz",
            mask,
            affine,
            sha256_path(run.staging / "prediction_mask.nii.gz"),
        )
        mask_entry["labels"] = {"0": "background", "1": "model-predicted lesion region"}
        mask_entry["same_grid_as"] = "dwi"
        mask_entry["kind"] = "model_prediction"
        volumes.append(mask_entry)
        manifest = {
            "schema_version": schemas.MANIFEST_SCHEMA,
            "case_id": case_id,
            "run_id": run_id,
            "coordinate_system": {
                "world": "RAS+ (NIfTI sform/qform), millimetres",
                "note": "use each volume's affine; array axes are stored as in the input image",
            },
            "volumes": volumes,
            "inputs_exported": export_inputs,
            "views": _views(prepared.original.orientation, prepared.original.shape),
            "render_3d": {
                "mask_available": True,
                "empty_mask": not quant["lesion_detected"],
                "suggested": "surface extraction (e.g. marching cubes) on prediction_mask",
            },
            "results": {
                "quantification": "quantification.json",
                "metadata": "inference_metadata.json",
            },
            "lesion_summary": {
                "lesion_detected": quant["lesion_detected"],
                "total_lesion_volume_ml": quant["total_lesion_volume_ml"],
                "component_count": quant["component_count"],
                "hemisphere_status": quant["hemisphere"]["status"],
                "label": "model-predicted lesion region (research)",
            },
            "sharing": {
                "shareable_by_default": SHAREABLE,
                "not_shareable_by_default": NOT_SHAREABLE,
            },
            "disclaimer": schemas.DISCLAIMER,
        }
        names = [] if record_input_names else [dwi.basename, adc.basename]
        for obj, schema in (
            (quant, schemas.QUANTIFICATION_SCHEMA),
            (manifest, schemas.MANIFEST_SCHEMA),
        ):
            bad = schemas.problems(obj, schema, names)
            if bad:
                raise InferenceError(
                    Code.OUTPUT_VERIFY, "Result failed contract checks.", "; ".join(bad)
                )
        run.write_json("quantification.json", quant)
        run.write_json("manifest.json", manifest)
        t = stage("write_outputs", 0.95, t)

        meta["status"] = "succeeded"
        meta["validation"]["warnings"] = [w.to_dict() for w in warnings]
        meta["validation"]["proceeded"] = True
        meta["outputs"] = {
            "files": {rel: sha256_path(run.staging / rel) for rel in run.written},
            "metadata": "inference_metadata.json",
            "log": "logs/inference.log",
            "probability_map_saved": False,
        }
        meta["inference_completed_at"] = _utc()
        meta["runtime_seconds"] = round(time.perf_counter() - t_start, 3)
        bad = schemas.problems(meta, schemas.METADATA_SCHEMA, names)
        if bad:
            raise InferenceError(
                Code.OUTPUT_VERIFY, "Metadata failed contract checks.", "; ".join(bad)
            )
        run.write_json("inference_metadata.json", meta)
        LOGGER.info(
            "run %s succeeded: lesion_detected=%s volume_ml=%.3f warnings=%d",
            run_id,
            quant["lesion_detected"],
            quant["total_lesion_volume_ml"],
            len(warnings),
        )
        _close(handler)
        final = run.commit()
        notify("done", 1.0)
        return InferenceResult(
            "succeeded",
            0,
            case_id,
            run_id,
            str(final),
            outputs={rel: rel for rel in [*run.written]},
            quantification=quant,
            metadata=meta,
            warnings=[w.to_dict() for w in warnings],
        )
    except InferenceError as exc:
        return _failed(run, handler, meta, warnings, exc.issue, exc.exit_code, t_start)
    except KeyboardInterrupt:
        issue = Issue(Code.INTERRUPTED, "Inference was interrupted.")
        _failed(run, handler, meta, warnings, issue, EXIT_CODES[Code.INTERRUPTED], t_start)
        raise
    except Exception as exc:
        LOGGER.error("internal error: %s", _safe_traceback(exc))
        issue = Issue(Code.INTERNAL, "An internal error occurred.", type(exc).__name__)
        return _failed(run, handler, meta, warnings, issue, EXIT_CODES[Code.INTERNAL], t_start)


def _close(handler: logging.Handler) -> None:
    LOGGER.removeHandler(handler)
    handler.close()


def _failed(run, handler, meta, warnings, issue: Issue, exit_code: int, t_start) -> InferenceResult:
    LOGGER.error("run %s failed: %s %s", run.run_id, issue.code.value, issue.message)
    meta["status"] = "failed"
    meta["error"] = issue.to_dict()
    meta["validation"]["warnings"] = [w.to_dict() for w in warnings]
    meta["validation"]["proceeded"] = False
    meta["outputs"] = {"files": {}, "note": "failed run: no images or results are kept (D5)"}
    meta["inference_completed_at"] = _utc()
    meta["runtime_seconds"] = round(time.perf_counter() - t_start, 3)
    _close(handler)
    try:
        for p in list(run.staging.iterdir()):
            if p.name == "inference_metadata.json":
                p.unlink()
        run.write_json("inference_metadata.json", meta)
    except (InferenceError, OSError):
        pass
    failed_dir = run.fail()
    return InferenceResult(
        "failed",
        exit_code,
        meta["case_id"],
        meta["run_id"],
        str(failed_dir) if failed_dir else None,
        metadata=meta,
        warnings=[w.to_dict() for w in warnings],
        error=issue.to_dict(),
    )
