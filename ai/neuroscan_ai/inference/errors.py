"""Error and warning model of the independent inference engine.

Every problem is an ``Issue`` with a stable machine code, a short user-facing message (no paths,
no header text) and an optional developer ``detail`` that is also kept free of local paths.
Errors stop the run before any output image is written; warnings are recorded and the run goes on
(``--strict`` promotes warnings to errors).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Code(StrEnum):
    # errors
    INVALID_ARGUMENT = "E_INVALID_ARGUMENT"
    INPUT_NOT_FOUND = "E_INPUT_NOT_FOUND"
    INPUT_FORMAT = "E_INPUT_FORMAT"
    INPUT_CORRUPT = "E_INPUT_CORRUPT"
    INPUT_NOT_3D = "E_INPUT_NOT_3D"
    GRID_MISMATCH = "E_GRID_MISMATCH"
    NONFINITE = "E_NONFINITE"
    EMPTY_IMAGE = "E_EMPTY_IMAGE"
    SHAPE_TOO_SMALL = "E_SHAPE_TOO_SMALL"
    SPACING_INVALID = "E_SPACING_INVALID"
    BRAIN_MASK = "E_BRAIN_MASK"
    SEALED_TEST_CASE = "E_SEALED_TEST_CASE"
    CHECKPOINT = "E_CHECKPOINT"
    CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
    PREPROCESS = "E_PREPROCESS"
    DEVICE = "E_DEVICE"
    INFERENCE = "E_INFERENCE"
    INFERENCE_OOM = "E_INFERENCE_OOM"
    OUTPUT_EXISTS = "E_OUTPUT_EXISTS"
    OUTPUT_WRITE = "E_OUTPUT_WRITE"
    OUTPUT_VERIFY = "E_OUTPUT_VERIFY"
    STRICT_WARNING = "E_STRICT_WARNING"
    INTERRUPTED = "E_INTERRUPTED"
    INTERNAL = "E_INTERNAL"
    # warnings
    W_ORIENTATION_UNUSUAL = "W_ORIENTATION_UNUSUAL"
    W_OBLIQUE = "W_OBLIQUE"
    W_SPACING_OUT_OF_TRAIN_RANGE = "W_SPACING_OUT_OF_TRAIN_RANGE"
    W_FEW_SLICES = "W_FEW_SLICES"
    W_BRAIN_FRACTION_UNUSUAL = "W_BRAIN_FRACTION_UNUSUAL"
    W_POSSIBLE_SWAP = "W_POSSIBLE_SWAP"
    W_ADC_UNIT_UNUSUAL = "W_ADC_UNIT_UNUSUAL"
    W_ADC_NEGATIVE = "W_ADC_NEGATIVE"
    W_HEADER_TEXT_PRESENT = "W_HEADER_TEXT_PRESENT"
    W_CPU_FALLBACK = "W_CPU_FALLBACK"
    W_OOM_RETRY = "W_OOM_RETRY"
    W_LARGE_LESION = "W_LARGE_LESION"
    W_MANY_COMPONENTS = "W_MANY_COMPONENTS"

    @property
    def is_warning(self) -> bool:
        return self.value.startswith("W_")


# Process exit codes of scripts/infer_case.py (0 = success, including "no lesion predicted").
EXIT_CODES: dict[Code, int] = {
    Code.INVALID_ARGUMENT: 2,
    Code.INPUT_NOT_FOUND: 3,
    Code.INPUT_FORMAT: 3,
    Code.INPUT_CORRUPT: 3,
    Code.INPUT_NOT_3D: 3,
    Code.GRID_MISMATCH: 3,
    Code.NONFINITE: 3,
    Code.EMPTY_IMAGE: 3,
    Code.SHAPE_TOO_SMALL: 3,
    Code.SPACING_INVALID: 3,
    Code.STRICT_WARNING: 3,
    Code.CHECKPOINT: 4,
    Code.CONFIG_MISMATCH: 4,
    Code.BRAIN_MASK: 5,
    Code.PREPROCESS: 5,
    Code.DEVICE: 6,
    Code.INFERENCE: 6,
    Code.INFERENCE_OOM: 6,
    Code.OUTPUT_WRITE: 7,
    Code.OUTPUT_VERIFY: 7,
    Code.OUTPUT_EXISTS: 8,
    Code.SEALED_TEST_CASE: 9,
    Code.INTERRUPTED: 130,
    Code.INTERNAL: 1,
}


@dataclass(frozen=True)
class Issue:
    code: Code
    message: str
    detail: str | None = None

    @property
    def severity(self) -> str:
        return "warning" if self.code.is_warning else "error"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["code"] = self.code.value
        d["severity"] = self.severity
        return d


class InferenceError(Exception):
    """Stops the run. ``issue.message`` is safe to show to a user."""

    def __init__(self, code: Code, message: str, detail: str | None = None) -> None:
        super().__init__(f"{code.value}: {message}")
        self.issue = Issue(code, message, detail)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.issue.code, 1)
