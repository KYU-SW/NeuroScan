"""Host and CUDA resource readings for long v2 runs (standard library + torch; no psutil).

Readings are operational records only (``training/progress.json``, interruption records); they
are never part of a result hash. A reading that the platform cannot provide is ``None``.

- Process RSS / peak RSS: Windows ``K32GetProcessMemoryInfo`` (working set / peak working set),
  Linux ``/proc/self/status`` (VmRSS / VmHWM). Other platforms: ``None``.
- System available RAM: Windows ``GlobalMemoryStatusEx`` (ullAvailPhys), Linux
  ``/proc/meminfo`` (MemAvailable). Other platforms: ``None``.
- CUDA: ``torch.cuda.memory_allocated / memory_reserved / max_memory_allocated /
  max_memory_reserved`` of the run device; the peak counters are reset once when a run (or a
  resumed segment) starts.

Host memory guard ``host_memory_guard_v1``: checked at every epoch boundary; when the system's
available RAM is below 4 GiB the run asks for a safe stop (resume checkpoint, progress and
interruption record are written, status ``interrupted_resource_guard``). 4 GiB is a host
operating-safety limit, not a model or performance threshold. Nothing is adjusted automatically:
no resume, no batch-size change, no cache-size change. An unavailable reading never triggers it.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any

import torch

GIB = 1024**3
RESOURCE_MONITOR: dict[str, Any] = {
    "record_process_rss": True,
    "record_system_available_memory": True,
    "record_cuda_memory": True,
}
HOST_MEMORY_GUARD: dict[str, Any] = {
    "policy_id": "host_memory_guard_v1",
    "check_frequency": "epoch",
    "minimum_available_gib": 4.0,
    "automatic_resume": False,
}


# ---------------------------------------------------------------- host readings
def _windows_process_memory() -> tuple[int, int] | None:
    from ctypes import wintypes

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    fn = k32.K32GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    c = _Counters()
    c.cb = ctypes.sizeof(_Counters)
    if not fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return None
    return int(c.WorkingSetSize), int(c.PeakWorkingSetSize)


def _windows_available() -> int | None:
    class _Status(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    s = _Status()
    s.dwLength = ctypes.sizeof(_Status)
    if not ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(s)):
        return None
    return int(s.ullAvailPhys)


def _proc_kib(path: str, key: str) -> int | None:
    try:
        for line in Path(path).read_text(encoding="ascii").splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def process_rss_bytes() -> int | None:
    if sys.platform == "win32":
        m = _windows_process_memory()
        return m[0] if m else None
    return _proc_kib("/proc/self/status", "VmRSS")


def process_peak_rss_bytes() -> int | None:
    if sys.platform == "win32":
        m = _windows_process_memory()
        return m[1] if m else None
    return _proc_kib("/proc/self/status", "VmHWM")


def system_available_memory_bytes() -> int | None:
    if sys.platform == "win32":
        return _windows_available()
    return _proc_kib("/proc/meminfo", "MemAvailable")


def host_memory() -> dict[str, int | None]:
    return {
        "process_rss_bytes": process_rss_bytes(),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "system_available_memory_bytes": system_available_memory_bytes(),
    }


# ---------------------------------------------------------------- CUDA readings
CUDA_FIELDS = (
    "cuda_memory_allocated_bytes",
    "cuda_memory_reserved_bytes",
    "cuda_max_memory_allocated_bytes",
    "cuda_max_memory_reserved_bytes",
)


def reset_cuda_peak(device: torch.device) -> bool:
    """Reset the CUDA peak counters of ``device`` (run / segment start). False on CPU."""
    if device.type != "cuda":
        return False
    torch.cuda.reset_peak_memory_stats(device)
    return True


def cuda_memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return dict.fromkeys(CUDA_FIELDS)
    return {
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


# ---------------------------------------------------------------- host memory guard
def check_host_memory(available_bytes: int | None) -> dict[str, Any]:
    """Guard decision at an epoch boundary (``triggered`` asks for a safe stop)."""
    minimum = int(HOST_MEMORY_GUARD["minimum_available_gib"] * GIB)
    return {
        "policy_id": HOST_MEMORY_GUARD["policy_id"],
        "minimum_available_bytes": minimum,
        "system_available_memory_bytes": available_bytes,
        "reading_available": available_bytes is not None,
        "triggered": available_bytes is not None and available_bytes < minimum,
        "automatic_resume": False,
        "note": "host operating-safety limit; not a model or performance threshold",
    }
