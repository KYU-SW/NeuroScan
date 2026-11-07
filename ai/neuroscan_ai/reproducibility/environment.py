"""Source-control, software-environment and determinism capture.

Every probe is injectable (``run`` for subprocesses, ``torch_module`` for PyTorch, ``versions`` for
package metadata), so tests use synthetic environments. PyTorch is only imported when a caller asks
for it; nothing here builds or runs a model. Missing information is recorded as ``None``, never
guessed.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

Runner = Callable[..., subprocess.CompletedProcess]
PACKAGES = ("torch", "monai", "numpy", "nibabel", "scipy")


def _git(run: Runner, repo: Path, *args: str) -> str | None:
    try:
        p = run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def _clean_remote(url: str | None) -> str | None:
    """Remote URL without credentials or scheme, e.g. ``github.com/org/repo``."""
    if not url:
        return None
    url = re.sub(r"^[a-z+]+://", "", url.strip())
    url = re.sub(r"^[^@/]+@", "", url)  # user[:token]@host
    url = (
        url.replace(":", "/", 1) if re.match(r"^[\w.-]+:[^/]", url) else url
    )  # scp-like git@host:org
    return re.sub(r"\.git$", "", url)


def capture_source_control(
    repo: Path, lock_tag: str | None = None, run: Runner = subprocess.run
) -> dict:
    status = _git(run, repo, "status", "--porcelain", "--untracked-files=all")
    lines = status.splitlines() if status is not None else None
    tag_target = _git(run, repo, "rev-parse", f"{lock_tag}^{{commit}}") if lock_tag else None
    return {
        "repository": _clean_remote(_git(run, repo, "remote", "get-url", "origin")),
        "branch": _git(run, repo, "branch", "--show-current"),
        "git_commit": _git(run, repo, "rev-parse", "HEAD"),
        "working_tree_clean": (len(lines) == 0) if lines is not None else None,
        "tracked_diff_present": any(not ln.startswith("??") for ln in lines)
        if lines is not None
        else None,
        "untracked_files_present": any(ln.startswith("??") for ln in lines)
        if lines is not None
        else None,
        "lock_tag": lock_tag,
        "lock_tag_target": tag_target,
    }


def _version(name: str, versions: Callable[[str], str]) -> str | None:
    try:
        return versions(name)
    except Exception:  # noqa: BLE001 - not installed
        return None


def gpu_driver(run: Runner = subprocess.run) -> dict[str, str | None]:
    """GPU name and driver from nvidia-smi; ``None`` when unavailable."""
    try:
        p = run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return {"gpu_name": None, "gpu_driver_version": None}
    if p.returncode != 0 or not p.stdout.strip():
        return {"gpu_name": None, "gpu_driver_version": None}
    name, _, drv = p.stdout.strip().splitlines()[0].partition(",")
    return {"gpu_name": name.strip() or None, "gpu_driver_version": drv.strip() or None}


def capture_environment(
    *,
    torch_module: Any = None,
    import_torch: bool = False,
    run: Runner = subprocess.run,
    versions: Callable[[str], str] = metadata.version,
    lock_file: Path | None = None,
) -> dict[str, Any]:
    from neuroscan_ai.reproducibility.hashing import sha256_text_lf

    if torch_module is None and import_torch:
        import torch as torch_module  # noqa: PLC0415 - only on explicit request
    cuda = None
    cudnn = None
    if torch_module is not None:
        if torch_module.cuda.is_available():
            cuda = torch_module.version.cuda
        cudnn = (
            torch_module.backends.cudnn.version()
            if torch_module.backends.cudnn.is_available()
            else None
        )
    drv = (
        gpu_driver(run)
        if (torch_module is not None and torch_module.cuda.is_available())
        else {"gpu_name": None, "gpu_driver_version": None}
    )
    if drv["gpu_name"] is None and torch_module is not None and torch_module.cuda.is_available():
        drv["gpu_name"] = torch_module.cuda.get_device_name(0)
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "packages": {p: _version(p, versions) for p in PACKAGES},
        "torch_cuda_available": bool(torch_module.cuda.is_available())
        if torch_module is not None
        else None,
        "cuda_runtime": cuda,
        "cudnn": cudnn,
        **drv,
        "package_lock_sha256_lf": sha256_text_lf(lock_file)
        if lock_file and Path(lock_file).is_file()
        else None,
    }


def capture_determinism(
    torch_module: Any,
    *,
    seed: int | None,
    precision: str | None,
    amp: bool | None,
    workers: int | None,
    batch_size: int | None,
) -> dict[str, Any]:
    t = torch_module
    return {
        "seed": seed,
        "torch_deterministic_algorithms": bool(t.are_deterministic_algorithms_enabled())
        if t
        else None,
        "torch_deterministic_warn_only": bool(t.is_deterministic_algorithms_warn_only_enabled())
        if t
        else None,
        "cudnn_deterministic": bool(t.backends.cudnn.deterministic) if t else None,
        "cudnn_benchmark": bool(t.backends.cudnn.benchmark) if t else None,
        "tf32_matmul": bool(t.backends.cuda.matmul.allow_tf32) if t else None,
        "tf32_convolution": bool(t.backends.cudnn.allow_tf32) if t else None,
        "precision": precision,
        "automatic_mixed_precision": amp,
        "dataloader_workers": workers,
        "batch_size": batch_size,
    }
