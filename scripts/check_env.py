"""Phase 0 environment check: Python, PyTorch/CUDA, MONAI, NiBabel, SimpleITK.

Usage:
    .venv\\Scripts\\python.exe scripts\\check_env.py

Exits with a non-zero code if any check fails. Writes only to a temporary directory.
"""

from __future__ import annotations

import platform
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str) -> Callable[[Callable[[], str]], Callable[[], str]]:
    def decorator(fn: Callable[[], str]) -> Callable[[], str]:
        try:
            detail = fn()
            RESULTS.append((name, True, detail))
        except Exception as exc:  # report every failure, keep going
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn

    return decorator


@check("Python")
def _python() -> str:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"expected Python 3.11, got {platform.python_version()}")
    return f"{platform.python_version()} ({sys.executable})"


@check("PyTorch / CUDA")
def _torch() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__}: CUDA not available")
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        raise RuntimeError(f"GPU arch {arch} not in torch build {torch.cuda.get_arch_list()}")
    props = torch.cuda.get_device_properties(0)
    cudnn = torch.backends.cudnn.version()
    return (
        f"torch {torch.__version__}, CUDA runtime {torch.version.cuda}, "
        f"cuDNN {cudnn // 10000}.{cudnn // 100 % 100}.{cudnn % 100}, GPU {props.name} ({arch}), "
        f"VRAM {props.total_memory / 1024**3:.1f} GB"
    )


@check("CUDA tensor ops")
def _cuda_ops() -> str:
    import torch

    torch.manual_seed(0)
    a = torch.randn(2048, 2048)
    b = torch.randn(2048, 2048)
    expected = a @ b
    got = (a.cuda() @ b.cuda()).cpu()
    max_err = (got - expected).abs().max().item()
    if max_err > 1e-1:  # TF32 may be used on GPU; allow small numeric drift
        raise RuntimeError(f"GPU/CPU matmul mismatch, max abs err {max_err:.3e}")

    conv = torch.nn.Conv2d(6, 16, 3, padding=1).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        y = conv(torch.randn(4, 6, 128, 128, device="cuda"))
    torch.cuda.synchronize()
    return f"matmul 2048x2048 max abs err {max_err:.2e}; AMP conv2d {tuple(y.shape)} {y.dtype}"


@check("MONAI")
def _monai() -> str:
    import monai
    import torch
    from monai.networks.nets import UNet

    # Shape-only smoke test for the two planned input layouts (2ch 2D, 6ch 2.5D). Not training.
    shapes = []
    for in_ch in (2, 6):
        net = UNet(
            spatial_dims=2,
            in_channels=in_ch,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
        ).cuda()
        with torch.no_grad():
            out = net(torch.randn(2, in_ch, 128, 128, device="cuda"))
        shapes.append(f"{in_ch}ch->{tuple(out.shape)}")
    return f"monai {monai.__version__}; UNet(2D) forward on GPU: {', '.join(shapes)}"


@check("NiBabel")
def _nibabel() -> str:
    import nibabel as nib
    import numpy as np

    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    affine[:3, 3] = [-10.0, 20.0, 5.0]
    data = np.random.default_rng(0).random((8, 9, 10)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roundtrip.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), path)
        img = nib.load(path)
        if not np.allclose(img.get_fdata(), data) or not np.allclose(img.affine, affine):
            raise RuntimeError("NIfTI write/read round trip mismatch")
        zooms = tuple(float(z) for z in img.header.get_zooms())
    return f"nibabel {nib.__version__}; NIfTI round trip OK, zooms {zooms}"


@check("SimpleITK")
def _simpleitk() -> str:
    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk

    # Write with NiBabel (RAS affine), read with SimpleITK (LPS physical space)
    # to confirm both libraries agree on spacing and origin.
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    affine[:3, 3] = [-10.0, 20.0, 5.0]
    data = np.arange(8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cross.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), path)
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)  # (z, y, x)
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
    if not np.allclose(arr.transpose(2, 1, 0), data):
        raise RuntimeError("voxel order mismatch between NiBabel and SimpleITK")
    if not np.allclose(spacing, (1.0, 1.0, 2.0)):
        raise RuntimeError(f"spacing mismatch: {spacing}")
    if not np.allclose(origin, (10.0, -20.0, 5.0)):  # RAS -> LPS flips x and y
        raise RuntimeError(f"origin mismatch (expected LPS 10,-20,5): {origin}")
    return f"SimpleITK {sitk.Version_VersionString()}; spacing {spacing}, LPS origin {origin}"


def main() -> int:
    width = max(len(name) for name, _, _ in RESULTS)
    print("\nNeuroScan-ai Phase 0 environment check")
    print("=" * 60)
    for name, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    failed = sum(not ok for _, ok, _ in RESULTS)
    print("=" * 60)
    print("All checks passed." if not failed else f"{failed} check(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
