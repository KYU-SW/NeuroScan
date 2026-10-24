"""Synthetic inputs and a tiny checkpoint for Phase 5A inference tests (no ISLES data needed)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

from neuroscan_ai.training.config import config_hash, load_config
from neuroscan_ai.training.model import build_model
from tests.conftest import REPO_ROOT, las_affine

BASE_25D = REPO_ROOT / "configs" / "25d_baseline.yaml"
INFERENCE_CFG = REPO_ROOT / "configs" / "inference_v1.yaml"


def synthetic_volumes(shape=(48, 48, 10), seed=0, adc_scale=1.0, lesion=True):
    """Skull-stripped-like DWI/ADC: ellipsoid brain (~21% of the volume) on a zero background."""
    rng = np.random.default_rng(seed)
    idx = np.indices(shape).astype(float)
    centre = [(s - 1) / 2 for s in shape]
    radii = (15.0, 17.0, 4.6)
    brain = sum(((idx[i] - centre[i]) / radii[i]) ** 2 for i in range(3)) <= 1.0
    dwi = np.zeros(shape, np.float32)
    adc = np.zeros(shape, np.float32)
    dwi[brain] = rng.normal(200, 20, int(brain.sum()))
    adc[brain] = rng.normal(0.0009, 0.0001, int(brain.sum()))
    lesion_mask = np.zeros(shape, np.uint8)
    if lesion:
        lesion_mask[14:20, 16:22, 3:6] = 1
        lesion_mask &= brain
        dwi[lesion_mask > 0] = 450
        adc[lesion_mask > 0] = 0.0005
    return dwi, (adc * adc_scale).astype(np.float32), lesion_mask, brain


def write_nifti(path: Path, data, affine, header_text: dict[str, str] | None = None) -> Path:
    img = nib.Nifti1Image(np.asarray(data), affine)
    for k, v in (header_text or {}).items():
        img.header[k] = v.encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


def write_case(
    root: Path,
    spacing=(1.8, 1.8, 4.8),
    shape=(48, 48, 10),
    adc_scale=1.0,
    swap=False,
    header_text=None,
    affine=None,
    lesion=True,
    names=("dwi.nii.gz", "adc.nii.gz"),
):
    dwi, adc, label, _ = synthetic_volumes(shape, adc_scale=adc_scale, lesion=lesion)
    aff = las_affine(spacing) if affine is None else affine
    a, b = (adc, dwi) if swap else (dwi, adc)
    dwi_p = write_nifti(root / names[0], a, aff, header_text)
    adc_p = write_nifti(root / names[1], b, aff, header_text)
    return dwi_p, adc_p, label, aff


def tiny_model_setup(tmp: Path, bias: float | None = None, seed: int = 0, **info_overrides):
    """Small 6-channel UNet checkpoint + matching training config + inference config in ``tmp``."""
    cfg = load_config(BASE_25D)
    cfg["model"]["channels"] = [8, 16, 32]
    cfg["model"]["strides"] = [2, 2]
    cfg_path = tmp / "train_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    torch.manual_seed(seed)
    model = build_model(cfg)
    if bias is not None:
        convs = [
            m for m in model.modules() if isinstance(m, torch.nn.Conv2d | torch.nn.ConvTranspose2d)
        ]
        with torch.no_grad():
            convs[-1].bias.fill_(bias)
    info = {
        "run_id": "tiny_test",
        "config_hash": config_hash(cfg),
        "preprocessing_version": "v1",
        "in_channels": 6,
        "epoch": 0,
        "val_mean_dice": None,
        **info_overrides,
    }
    ckpt = tmp / "tiny.pt"
    torch.save({"model": model.state_dict(), "info": info}, ckpt)
    inf = yaml.safe_load(INFERENCE_CFG.read_text(encoding="utf-8"))
    inf["model"].update(
        checkpoint=str(ckpt),
        training_config=str(cfg_path),
        expected_config_hash=config_hash(cfg),
        model_name="tiny_test",
    )
    inf_path = tmp / "inference.yaml"
    inf_path.write_text(yaml.safe_dump(inf, sort_keys=False), encoding="utf-8")
    return inf_path, ckpt, cfg_path, cfg
