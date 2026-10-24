"""Run directory with staging, verified writes and atomic commit — never overwrites anything.

  outputs/<case_id>/.<run_id>.partial/   while running
  outputs/<case_id>/<run_id>/            after success (rename of the staging directory)
  outputs/<case_id>/.<run_id>.failed/    after failure: inference_metadata.json + logs/ only (D5)

A run id is a UTC timestamp plus random hex; the staging, final and failed directories must all
be absent before the run starts, and the final directory is re-checked right before the rename.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neuroscan_ai.inference.errors import Code, InferenceError
from neuroscan_ai.inference.validation import HEADER_TEXT_FIELDS
from neuroscan_ai.io import nifti

KEEP_ON_FAILURE = ("inference_metadata.json", "logs")


def new_run_id(now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(dt.UTC)
    return f"{now:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunDirectory:
    def __init__(self, output_root: str | Path, case_id: str, run_id: str) -> None:
        self.case_dir = Path(output_root) / case_id
        self.run_id = run_id
        self.staging = self.case_dir / f".{run_id}.partial"
        self.final = self.case_dir / run_id
        self.failed = self.case_dir / f".{run_id}.failed"
        if any(p.exists() for p in (self.staging, self.final, self.failed)):
            raise InferenceError(Code.OUTPUT_EXISTS, "An output folder with this run id exists.")
        try:
            self.case_dir.mkdir(parents=True, exist_ok=True)
            self.staging.mkdir(exist_ok=False)
            (self.staging / "logs").mkdir()
        except FileExistsError:
            raise InferenceError(
                Code.OUTPUT_EXISTS, "An output folder with this run id exists."
            ) from None
        except OSError as exc:
            raise InferenceError(
                Code.OUTPUT_WRITE, "Output folder could not be created.", type(exc).__name__
            ) from None
        self.written: list[str] = []

    @property
    def log_path(self) -> Path:
        return self.staging / "logs" / "inference.log"

    def _target(self, rel: str) -> Path:
        p = self.staging / rel
        if p.exists():
            raise InferenceError(Code.OUTPUT_EXISTS, "Output file already exists.", rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, rel: str, obj: dict[str, Any]) -> None:
        p = self._target(rel)
        try:
            text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)
            p.write_text(text + "\n", encoding="utf-8")
        except (OSError, ValueError, TypeError) as exc:
            raise InferenceError(
                Code.OUTPUT_WRITE, "Result file could not be saved.", f"{rel}: {type(exc).__name__}"
            ) from None
        self.written.append(rel)

    def write_nifti(self, rel: str, data: np.ndarray, affine: np.ndarray, binary: bool) -> None:
        """New NIfTI from voxel data + affine only (fresh header), then reload and verify."""
        p = self._target(rel)
        try:
            nifti.save(data, affine, p)
        except OSError as exc:
            raise InferenceError(
                Code.OUTPUT_WRITE,
                "Result image could not be saved.",
                f"{rel}: {type(exc).__name__}",
            ) from None
        verify_nifti(p, data, affine, binary)
        self.written.append(rel)

    def commit(self) -> Path:
        if self.final.exists():
            raise InferenceError(Code.OUTPUT_EXISTS, "Final output folder already exists.")
        try:
            os.rename(self.staging, self.final)
        except OSError as exc:
            raise InferenceError(
                Code.OUTPUT_WRITE, "Results could not be finalised.", type(exc).__name__
            ) from None
        return self.final

    def fail(self) -> Path | None:
        """D5: keep only inference_metadata.json and logs/, then rename to .failed."""
        if not self.staging.exists():
            return None
        for child in list(self.staging.iterdir()):
            if child.name in KEEP_ON_FAILURE:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        if self.failed.exists():
            return self.staging
        os.rename(self.staging, self.failed)
        return self.failed


def verify_nifti(path: Path, data: np.ndarray, affine: np.ndarray, binary: bool) -> None:
    img = nib.load(str(path))
    back = np.asanyarray(img.dataobj)
    hdr = img.header
    text = [n for n in HEADER_TEXT_FIELDS if n in hdr and bytes(hdr[n].tobytes()).strip(b"\x00 ")]
    ok = (
        back.shape == data.shape
        and np.allclose(img.affine, affine, atol=1e-5)
        and back.dtype == data.dtype
        and not text
        and not len(hdr.extensions)
        and int(hdr["sform_code"]) == 1
        and int(hdr["qform_code"]) == 1
    )
    if ok and binary:
        ok = (
            back.dtype == np.uint8
            and set(np.unique(back).tolist()) <= {0, 1}
            and int(back.sum()) == int(data.sum())
        )
    elif ok:
        ok = np.array_equal(back, data)
    if not ok:
        raise InferenceError(Code.OUTPUT_VERIFY, "Saved image failed verification.", path.name)
