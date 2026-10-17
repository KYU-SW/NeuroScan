from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CaseRecord:
    """One patient case as found on disk.

    images: model input modalities, e.g. {"dwi": ..., "adc": ...}
    label:  ground-truth lesion mask, or None (e.g. unlabeled external data)
    extra:  modalities kept for QA statistics only, e.g. {"flair": ...}
    """

    dataset: str
    case_id: str
    images: dict[str, Path]
    label: Path | None = None
    extra: dict[str, Path] = field(default_factory=dict)

    def all_paths(self) -> dict[str, Path]:
        paths = dict(self.images)
        if self.label is not None:
            paths["label"] = self.label
        paths.update(self.extra)
        return paths
