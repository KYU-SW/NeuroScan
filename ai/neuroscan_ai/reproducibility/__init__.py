"""NeuroScan v2 run reproducibility manifests.

Records, before and after a run, everything needed to tell whether a later run reproduces it:
source control state, config / split / checkpoint hashes (raw and LF-canonical), software
versions, GPU driver, determinism settings, input and output hashes and split access. Built from
the lessons of a v1 reproduction attempt in which neither the run-time commit nor the environment
had been recorded and probability maps had not been kept.

This package never runs a model and never reads image data itself: callers pass paths, and only
SHA-256 hashes of those files are recorded. Public manifests contain token paths (``<WORKDIR>``,
``<CHECKPOINT>``, ``<DATASET>``), hashes, versions, policy IDs and position-based case keys only;
case aliases appear only as the hash of the sorted alias list.
Schema: ``neuroscan-run-manifest-v1``.
"""

from neuroscan_ai.reproducibility.manifest import (
    CASE_KEY_POLICY,
    SCHEMA,
    ManifestError,
    PathRoots,
    RunSpec,
    build_post_run,
    build_pre_run,
    case_keys,
    check_schema,
    is_linked,
    write_case_key_map,
    write_manifest,
)
from neuroscan_ai.reproducibility.validation import (
    COMPATIBLE,
    EXACT,
    MISMATCH,
    UNAVAILABLE,
    compare_manifests,
)

__all__ = [
    "CASE_KEY_POLICY",
    "COMPATIBLE",
    "EXACT",
    "MISMATCH",
    "UNAVAILABLE",
    "ManifestError",
    "PathRoots",
    "RunSpec",
    "SCHEMA",
    "build_post_run",
    "build_pre_run",
    "case_keys",
    "check_schema",
    "compare_manifests",
    "is_linked",
    "write_case_key_map",
    "write_manifest",
]
