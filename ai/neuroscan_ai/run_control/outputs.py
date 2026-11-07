"""Required run outputs and their consistency checks. Checks only read; nothing is repaired.

Per case (case key ``case_NNNN``): probability map, ungated binary, brain mask, gated binary
(NIfTI) and a metric JSON; per run: an aggregate JSON. Relations:
ungated == (probability >= threshold); gated == ungated AND brain mask; gated <= ungated.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

IMAGE_KINDS = ("probability_maps", "ungated_predictions", "brain_masks", "gated_predictions")
METRIC_KIND = "patient_metrics"
AGGREGATE_KIND = "aggregate_results"
AGGREGATE_LABEL = "aggregate"
# train-stage run-level outputs (recorded under aggregate_results with these labels)
TRAINING_FILES = {
    "checkpoint": "training/checkpoint.pt",
    "training_history": "training/training_history.json",
    "validation_metrics": "training/validation_metrics.json",
    "threshold_record": "training/threshold_record.json",
}


def case_path(run_dir: Path, kind: str, case_key: str) -> Path:
    ext = ".json" if kind == METRIC_KIND else ".nii.gz"
    return Path(run_dir) / kind / f"{case_key}{ext}"


def aggregate_path(run_dir: Path) -> Path:
    return Path(run_dir) / AGGREGATE_KIND / f"{AGGREGATE_LABEL}.json"


def run_output_path(run_dir: Path, label: str) -> Path:
    """Path of a run-level output label (``aggregate`` or a training output)."""
    if label == AGGREGATE_LABEL:
        return aggregate_path(run_dir)
    if label not in TRAINING_FILES:
        raise ValueError(f"unknown run-level output '{label}'")
    return Path(run_dir) / TRAINING_FILES[label]


def save_image(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError("output exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, str(path))


def _binary(a: np.ndarray) -> bool:
    return bool(np.isin(a, (0, 1)).all())


def check_case(run_dir: Path, case_key: str, threshold: float) -> list[str]:
    """Error codes for one case's image outputs (empty list = consistent)."""
    paths = {k: case_path(run_dir, k, case_key) for k in IMAGE_KINDS}
    if not all(p.is_file() for p in paths.values()):
        return ["image_output_missing"]
    imgs = {k: nib.load(str(p)) for k, p in paths.items()}
    ref = imgs["probability_maps"]
    errors = []
    for k, im in imgs.items():
        if tuple(im.shape) != tuple(ref.shape):
            errors.append(f"shape_mismatch:{k}")
        elif not np.allclose(im.affine, ref.affine, atol=1e-6):
            errors.append(f"affine_mismatch:{k}")
    if errors:
        return errors
    prob = np.asanyarray(ref.dataobj).astype(np.float64)
    ung, mask, gat = (np.asanyarray(imgs[k].dataobj) for k in IMAGE_KINDS[1:])
    if not np.isfinite(prob).all():
        return ["probability_nonfinite"]
    if prob.min() < 0 or prob.max() > 1:
        errors.append("probability_out_of_range")
    for name, a in (("ungated", ung), ("brain_mask", mask), ("gated", gat)):
        if not _binary(a):
            errors.append(f"non_binary:{name}")
    if errors:
        return errors
    ung, mask, gat = (a.astype(bool) for a in (ung, mask, gat))
    if not np.array_equal(ung, prob >= threshold):
        errors.append("ungated_not_threshold_of_probability")
    if not np.array_equal(gat, ung & mask):
        errors.append("gated_not_ungated_and_brain_mask")
    if gat.sum() > ung.sum():
        errors.append("gated_exceeds_ungated")
    return errors
