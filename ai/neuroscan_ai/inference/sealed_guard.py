"""Refuse inputs that belong to the sealed ISLES'22 test split — before any input file is opened.

The check works on path *strings* first: if any path component or file name contains an ISLES case
ID (``sub-strokecaseNNNN``) that ``SplitGuard`` lists as a test patient, the run is refused without
touching the file. Only paths that pass this string check are then resolved through symlinks
(metadata only, no content read) and checked again.

Limitation: a test image copied under a name without its case ID cannot be recognised; using test
images for inference remains forbidden by the research protocol (docs/phase5a_*).
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path

from neuroscan_ai.data.split_guard import SealedSplitAccessError, SplitGuard
from neuroscan_ai.inference.errors import Code, InferenceError

ISLES_CASE_ID = re.compile(r"sub-strokecase\d{4}")


def case_ids_in(path_text: str) -> set[str]:
    return set(ISLES_CASE_ID.findall(path_text))


def check_not_sealed(paths: Sequence[str | os.PathLike], split_file: str | Path) -> dict:
    """Raise ``E_SEALED_TEST_CASE`` for a sealed test case; return a path-free audit record."""
    guard = SplitGuard(split_file)
    found: set[str] = set()
    for p in paths:
        ids = case_ids_in(os.path.abspath(os.fspath(p)))  # pure string operation
        found |= ids
        _refuse_if_sealed(guard, ids)
    for p in paths:  # symlink targets, only for paths that passed the string check
        ids = case_ids_in(os.path.realpath(os.fspath(p)))
        found |= ids
        _refuse_if_sealed(guard, ids)
    return {
        "checked": True,
        "split_version": guard.version,
        "isles_case_ids_in_input_paths": len(found),
        "sealed_test_case_detected": False,
    }


def _refuse_if_sealed(guard: SplitGuard, ids: set[str]) -> None:
    try:
        guard.assert_no_test(ids, "independent inference input")
    except SealedSplitAccessError:
        raise InferenceError(
            Code.SEALED_TEST_CASE,
            "Sealed evaluation (test split) data cannot be used.",
            "input path refers to an ISLES'22 test-split case",
        ) from None
