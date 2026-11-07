"""Common runner interface: request / result types, tool names, status values, alias rules."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOOL_V1 = "v1_background"
TOOL_SYNTHSTRIP = "synthstrip"
TOOL_HDBET = "hd_bet"
TOOLS = (TOOL_V1, TOOL_SYNTHSTRIP, TOOL_HDBET)

STATUS_PASS = "pass"
STATUS_REVIEW = "review_required"
STATUS_FAIL = "fail"
STATUSES = (STATUS_PASS, STATUS_REVIEW, STATUS_FAIL)

# Public outputs carry aliases only. An alias is lower-case words / digits joined by '-', and must
# not look like an original subject ID (BIDS 'sub-' prefix, ISLES 'strokecase').
_ALIAS_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
_ID_LIKE = re.compile(r"(^|[^a-z])sub-|strokecase", re.I)


class BrainExtractionError(ValueError):
    """Refused request (message never contains an original subject ID or a path)."""


def validate_alias(alias: str) -> str:
    if not isinstance(alias, str) or not _ALIAS_RE.match(alias) or _ID_LIKE.search(alias):
        raise BrainExtractionError("alias must be a public alias, not an original ID")
    return alias


@dataclass(frozen=True)
class RunRequest:
    """One brain-extraction run of one tool on one case (Policy A: TRACE input only)."""

    alias: str
    trace_path: Path  # 3D TRACE / DWI NIfTI (SOOP: format-adapter output)
    gt_path: Path | None  # acute lesion mask on the TRACE grid (QA only, never shown to tools)
    tool: str
    tool_version: str
    config: dict[str, Any] = field(default_factory=dict)
    out_dir: Path = Path("results/brain_extraction_pilot")

    def __post_init__(self) -> None:
        validate_alias(self.alias)
        if self.tool not in TOOLS:
            raise BrainExtractionError(f"unknown tool '{self.tool}', expected one of {TOOLS}")


@dataclass
class RunResult:
    alias: str
    tool: str
    status: str
    mask_path: Path | None
    qa: dict[str, Any]
    audit: dict[str, Any]
    log: list[str]
    figure_info: dict[str, Any] = field(default_factory=dict)
