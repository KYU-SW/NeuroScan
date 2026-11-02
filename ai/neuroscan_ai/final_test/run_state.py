"""One-shot run state of a final-test (or dry-run) output directory.

Layout of ``<out>`` (e.g. results/final_test/v1):

  LOCK              created atomically (O_CREAT | O_EXCL) right before the first case is loaded;
                    kept forever as the record that the evaluation started
  .attempt-<n>/     staging area of attempt n; nothing is written to <out> directly
  COMPLETED         success: staged files moved into <out>, then this marker
  FAILED            failure: staging moved to failed_attempts/attempt-<n>/, reason recorded here
  failed_attempts/attempt-<n>/
                    partial output, the LOCK / FAILED of that attempt, the failure report

Refused by default: an existing LOCK, COMPLETED, FAILED, result file or prediction directory.
A failed run may be resumed only with --resume-after-failure, a written failure report and a
reason; a completed run is never repeated.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOCK, COMPLETED, FAILED = "LOCK", "COMPLETED", "FAILED"
RESULT_FILES = (
    "patient_metrics.csv",
    "patient_metrics_2d.csv",
    "patient_metrics_25d.csv",
    "paired_metrics.csv",
    "aggregate_metrics.json",
    "statistical_comparison.json",
)
PREDICTION_DIRS = ("predictions_2d", "predictions_25d")


class RunStateError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _attempts(out: Path) -> int:
    failed = out / "failed_attempts"
    return len([p for p in failed.glob("attempt-*") if p.is_dir()]) if failed.exists() else 0


def check_output_dir(out: Path, resume: bool) -> int:
    """Return the attempt number to use; raise when the directory state forbids a run."""
    if (out / COMPLETED).exists():
        raise RunStateError(f"{out.as_posix()}: a completed evaluation exists; never re-run")
    if (out / FAILED).exists():
        if not resume:
            raise RunStateError(
                f"{out.as_posix()}: a failed attempt exists; re-running needs "
                "--resume-after-failure, --failure-report and --reason"
            )
        return _attempts(out) + 1
    if resume:
        raise RunStateError(f"{out.as_posix()}: no failed attempt to resume")
    present = [n for n in (LOCK, *RESULT_FILES, *PREDICTION_DIRS) if (out / n).exists()]
    if present:
        raise RunStateError(f"{out.as_posix()}: existing run artefacts {present}; refusing")
    return _attempts(out) + 1


def archive_failed_attempt(out: Path, failure_report: Path, reason: str) -> Path:
    """Before a resume: move LOCK + FAILED of the failed attempt into failed_attempts/."""
    if not failure_report.is_file() or failure_report.stat().st_size == 0:
        raise RunStateError("--failure-report must be an existing, non-empty file")
    if not reason.strip():
        raise RunStateError("--reason must not be empty")
    failed = json.loads((out / FAILED).read_text(encoding="utf-8"))
    dest = out / "failed_attempts" / f"attempt-{failed['attempt']}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in (LOCK, FAILED):
        if (out / name).exists():
            (out / name).replace(dest / name)
    shutil.copyfile(failure_report, dest / "failure_report_for_resume.txt")
    _write_json(dest / "resume.json", {"resumed_at": now(), "reason": reason})
    return dest


def acquire_lock(out: Path, info: dict[str, Any]) -> Path:
    """Atomic: exactly one process can create LOCK; an existing LOCK always wins."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / LOCK
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RunStateError(f"{path.as_posix()} exists; another run started") from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"locked_at": now(), "pid": os.getpid(), **info}, f, indent=2, default=str)
    return path


def staging_dir(out: Path, attempt: int) -> Path:
    d = out / f".attempt-{attempt}"
    if d.exists():
        raise RunStateError(f"{d.as_posix()} already exists")
    d.mkdir(parents=True)
    return d


def mark_completed(out: Path, staging: Path, info: dict[str, Any]) -> None:
    for p in sorted(staging.iterdir()):
        target = out / p.name
        if target.exists():
            raise RunStateError(f"{target.as_posix()} exists; refusing to overwrite")
        p.replace(target)
    staging.rmdir()
    _write_json(out / COMPLETED, {"completed_at": now(), **info})


def mark_failed(out: Path, staging: Path | None, attempt: int, reason: str) -> Path:
    dest = out / "failed_attempts" / f"attempt-{attempt}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if staging is not None and staging.exists():
        staging.replace(dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
    _write_json(
        out / FAILED,
        {
            "failed_at": now(),
            "attempt": attempt,
            "reason": reason,
            "partial_output": dest.relative_to(out).as_posix(),
        },
    )
    return dest
