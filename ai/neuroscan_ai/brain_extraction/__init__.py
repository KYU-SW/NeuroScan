"""Brain-extraction pilot framework (runner interface, mask QA, figures, audit, selection).

Policy A: a brain mask is produced from the TRACE / DWI image only and applied unchanged to the
ADC on the same grid; no mask is produced from ADC, T1w or FLAIR.

This package never installs, downloads or executes an external tool by itself. External tools
(SynthStrip, HD-BET) are described by command *specifications*; a real run needs an explicitly
injected executor, which does not exist in this repository stage. Masks are never resampled,
dilated, edited or repaired: a mask that does not fit the TRACE grid or loses a ground-truth
voxel is reported as a failure. The v1 frozen code is only imported read-only (the v1 baseline
mask policy ``skull_stripped_background``).
"""

from neuroscan_ai.brain_extraction.interfaces import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_REVIEW,
    TOOLS,
    BrainExtractionError,
    RunRequest,
    RunResult,
)

__all__ = [
    "STATUS_FAIL",
    "STATUS_PASS",
    "STATUS_REVIEW",
    "TOOLS",
    "BrainExtractionError",
    "RunRequest",
    "RunResult",
]
