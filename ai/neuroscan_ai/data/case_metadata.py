"""Per-case subgroup metadata for stratified evaluation (dataset-agnostic).

Built from the Phase 1 QA table (generic QA columns), an optional split file and optional
BIDS JSON sidecars. Nothing here changes images; ADC values are only *classified* by magnitude,
never normalised (that belongs to the model input pipeline).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from neuroscan_ai.datasets.base import CaseRecord

VOLUME_GROUPS = ((0.0, "0 (empty)"), (1.0, "(0,1) mL"), (10.0, "[1,10) mL"), (50.0, "[10,50) mL"))
OBLIQUITY_GROUPS_DEG = ((1.0, "<1 deg"), (10.0, "[1,10) deg"), (20.0, "[10,20) deg"))


def volume_group(volume_ml: float) -> str:
    if volume_ml <= 0:
        return VOLUME_GROUPS[0][1]
    for upper, label in VOLUME_GROUPS[1:]:
        if volume_ml < upper:
            return label
    return "[50,inf) mL"


def adc_unit_group(adc_median: float | None) -> str:
    """Classify the stored ADC unit from the median brain ADC (typical ~0.8e-3 mm^2/s)."""
    if adc_median is None or not np.isfinite(adc_median):
        return "unknown"
    if adc_median < 0.01:
        return "mm2/s"
    if adc_median < 10:
        return "1e-3 mm2/s"
    return "1e-6 mm2/s"


def obliquity_deg(affine: np.ndarray) -> float:
    return float(np.degrees(nib.affines.obliquity(affine)).max())


def obliquity_group(deg: float) -> str:
    for upper, label in OBLIQUITY_GROUPS_DEG:
        if deg < upper:
            return label
    return ">=20 deg"


def _sidecar(image_path: Path) -> dict[str, Any]:
    side = image_path.with_name(image_path.name.replace(".nii.gz", ".json"))
    if not side.exists():
        return {}
    return json.loads(side.read_text(encoding="utf-8"))


def build_case_metadata(
    qa_csv: Path,
    records: Sequence[CaseRecord],
    split_json: Path | None = None,
    reference_image: str = "dwi",
) -> pd.DataFrame:
    qa = pd.read_csv(qa_csv).set_index("case_id")
    split_of: dict[str, str] = {}
    if split_json is not None and split_json.exists():
        for name, ids in json.loads(split_json.read_text(encoding="utf-8"))["splits"].items():
            split_of.update(dict.fromkeys(ids, name))

    rows = []
    for rec in records:
        q = qa.loc[rec.case_id]
        ref_path = rec.images[reference_image]
        deg = obliquity_deg(nib.load(str(ref_path)).affine)
        side = _sidecar(ref_path)
        vol = float(q["lesion_volume_ml"])
        rows.append(
            {
                "case_id": rec.case_id,
                "dataset": rec.dataset,
                "split": split_of.get(rec.case_id, "none"),
                "has_lesion": bool(q["has_lesion"]),
                "lesion_volume_ml": vol,
                "volume_group": volume_group(vol),
                "z_spacing": float(q[f"{reference_image}_sz"]),
                "spacing_group": "thick" if bool(q["thick_slice"]) else "thin",
                "adc_unit_group": adc_unit_group(q.get("adc_p50")),
                "obliquity_deg": round(deg, 3),
                "obliquity_group": obliquity_group(deg),
                # Centre is not recorded in ISLES'22; scanner fields come from sidecars if present.
                "center": "unavailable",
                "scanner_manufacturer": side.get("Manufacturer"),
                "scanner_model": side.get("ManufacturerModelName"),
                "field_strength_t": side.get("MagneticFieldStrength"),
            }
        )
    return pd.DataFrame(rows)
