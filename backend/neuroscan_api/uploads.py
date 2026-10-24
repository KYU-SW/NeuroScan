"""Upload handling: extension and sealed-test filename checks (before any byte is read), streamed
1 MiB copies with SHA-256 and size limits, gzip-bomb limit and a minimal NIfTI header check.

Full NIfTI validation (grid, NaN, skull-strip, ...) stays in the Phase 5A engine. The client file
name is used only for these checks and is never stored or logged.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import struct
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from neuroscan_api.errors import ApiError
from neuroscan_api.settings import Limits
from neuroscan_api.storage import Storage, insufficient_storage, is_no_space

CHUNK = 1 << 20  # 1 MiB
ISLES_CASE_ID = re.compile(r"sub-strokecase\d{4}")
# single-file NIfTI only (".hdr/.img" pairs, magic ni1/ni2, are not accepted)
NIFTI1_MAGIC = (b"n+1\x00",)
NIFTI2_MAGIC = (b"n+2\x00\r\n\x1a\n",)
ALLOWED_BITPIX = (8, 16, 32, 64)


@dataclass
class TempUpload:
    role: str
    path: Path
    suffix: str  # ".nii.gz" | ".nii"
    size_bytes: int
    sha256: str


def normalized_suffix(filename: str | None) -> str:
    name = (filename or "").lower().strip()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".nii"):
        return ".nii"
    raise ApiError(400, "UNSUPPORTED_EXTENSION", "Only .nii and .nii.gz files are accepted.")


def check_sealed_filename(filename: str | None, guard) -> None:
    """Refuse an ISLES test-split case by its file name, before the content is read."""
    from neuroscan_ai.data.split_guard import SealedSplitAccessError  # torch-free module

    ids = set(ISLES_CASE_ID.findall(filename or ""))
    try:
        guard.assert_no_test(ids, "API upload")
    except SealedSplitAccessError:
        raise ApiError(
            403, "SEALED_TEST_CASE", "Sealed evaluation (test split) data cannot be used."
        ) from None


async def save_stream(
    upload: UploadFile, role: str, storage: Storage, limits: Limits
) -> TempUpload:
    suffix = normalized_suffix(upload.filename)
    target = storage.temp / f"{uuid.uuid4().hex}.part"
    h, size = hashlib.sha256(), 0
    try:
        with open(target, "xb") as out:
            while chunk := await upload.read(CHUNK):
                size += len(chunk)
                if size > limits.max_file_bytes:
                    raise ApiError(413, "FILE_TOO_LARGE", f"{role.upper()} file is too large.")
                h.update(chunk)
                out.write(chunk)
    except OSError as exc:
        target.unlink(missing_ok=True)
        if is_no_space(exc):
            raise insufficient_storage() from None
        raise
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    if size == 0:
        target.unlink(missing_ok=True)
        raise ApiError(400, "MISSING_FILE", f"{role.upper()} file is empty.")
    return TempUpload(role, target, suffix, size, h.hexdigest())


def _invalid(role: str, why: str) -> ApiError:
    return ApiError(
        422,
        "INVALID_NIFTI",
        f"{role.upper()} is not a supported NIfTI file.",
        details={"check": why},
    )


def _uncompressed_size_and_head(path: Path, suffix: str, limits: Limits, role: str):
    if suffix == ".nii":
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(560)
        return size, head
    head, total = b"", 0
    try:
        with gzip.open(path, "rb") as f:
            while chunk := f.read(CHUNK):
                if len(head) < 560:
                    head += chunk[: 560 - len(head)]
                total += len(chunk)
                if total > limits.max_uncompressed_bytes:
                    raise ApiError(
                        422,
                        "ARCHIVE_TOO_LARGE",
                        f"{role.upper()} expands beyond the allowed size.",
                    )
    except (OSError, EOFError, zlib.error):
        raise _invalid(role, "gzip") from None
    return total, head


def inspect_nifti(tmp: TempUpload, limits: Limits) -> dict:
    """Minimal structural check of a single-file NIfTI-1/2 (no image data is interpreted)."""
    role = tmp.role
    total, head = _uncompressed_size_and_head(tmp.path, tmp.suffix, limits, role)
    if total > limits.max_uncompressed_bytes:
        raise ApiError(422, "ARCHIVE_TOO_LARGE", f"{role.upper()} is larger than allowed.")
    if len(head) < 348:
        raise _invalid(role, "header length")
    for endian in "<>":
        sizeof_hdr = struct.unpack(f"{endian}i", head[:4])[0]
        if sizeof_hdr in (348, 540):
            break
    else:
        raise _invalid(role, "sizeof_hdr")
    if sizeof_hdr == 348:
        if head[344:348] not in NIFTI1_MAGIC:
            raise _invalid(role, "magic")
        dims = struct.unpack(f"{endian}8h", head[40:56])
        bitpix = struct.unpack(f"{endian}h", head[72:74])[0]
        vox_offset = struct.unpack(f"{endian}f", head[108:112])[0]
    else:
        if len(head) < 540 or head[4:12] not in NIFTI2_MAGIC:
            raise _invalid(role, "magic")
        bitpix = struct.unpack(f"{endian}h", head[14:16])[0]
        dims = struct.unpack(f"{endian}8q", head[16:80])
        vox_offset = struct.unpack(f"{endian}q", head[168:176])[0]
    ndim = dims[0]
    if ndim not in (3, 4) or (ndim == 4 and dims[4] > 1):
        raise _invalid(role, "dimensions")
    shape = dims[1:4]
    if any(n < 1 or n > limits.max_axis for n in shape):
        raise _invalid(role, "axis size")
    voxels = shape[0] * shape[1] * shape[2]
    if voxels > limits.max_voxels:
        raise ApiError(422, "ARCHIVE_TOO_LARGE", f"{role.upper()} has too many voxels.")
    if bitpix not in ALLOWED_BITPIX:
        raise _invalid(role, "bitpix")
    if vox_offset < sizeof_hdr or total < vox_offset + voxels * bitpix // 8:
        raise _invalid(role, "data length")
    return {"shape": list(shape), "bitpix": bitpix}
