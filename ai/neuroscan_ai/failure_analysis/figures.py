"""Local-only QA figures of the image review (PNG). Titles and file names carry the alias only.

Display rules: brain-voxel 1-99 percentile window per modality, Research Mode meaning colours
(TP #009E73, FP #E69F00, FN #56B4E9, GT outline #CC79A7), fixed outline width, a 10 mm scale bar,
pixel aspect from the voxel spacing, text legend. Images are shown in array-index orientation
(i, j, k of the original grid), not a radiological convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from neuroscan_ai.failure_analysis.image_review import (
    MD_2D_ONLY,
    MD_25D_ONLY,
    MD_BOTH,
    SLICE_AXIS,
    neighbours,
)
from neuroscan_ai.research.compare import FN, FP, TP

COLORS = {
    "tp": "#009E73",
    "fp": "#E69F00",
    "fn": "#56B4E9",
    "gt": "#CC79A7",
    "both": "#F0E442",
    "2d_only": "#0072B2",
    "25d_only": "#D55E00",
}
LINE = 1.0


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def window(volume: np.ndarray, brain: np.ndarray) -> tuple[float, float]:
    v = volume[np.asarray(brain) > 0]
    if v.size == 0:
        return float(volume.min()), float(volume.max())
    lo, hi = np.percentile(v, [1, 99])
    return float(lo), float(hi if hi > lo else lo + 1e-6)


def _overlay(labels: np.ndarray, mapping: dict[int, str]) -> np.ndarray:
    from matplotlib.colors import to_rgba

    rgba = np.zeros((*labels.shape, 4))
    for value, color in mapping.items():
        rgba[labels == value] = to_rgba(color, 0.85)
    return rgba


def _scale_bar(ax, mm_per_px: float) -> None:
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    ax.add_artist(
        AnchoredSizeBar(
            ax.transData,
            10.0 / mm_per_px,
            "10 mm",
            "lower right",
            pad=0.3,
            color="white",
            frameon=False,
            size_vertical=0.5,
            fontproperties={"size": 6},
        )
    )


def _show(ax, img2d: np.ndarray, win, sx: float, sy: float) -> None:
    """img2d indexed [u, v] (array order); drawn with u horizontal, v vertical."""
    ax.imshow(
        img2d.T,
        cmap="gray",
        vmin=win[0],
        vmax=win[1],
        origin="lower",
        aspect=sy / sx,
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def panel_a(path: Path, alias: str, z: int, dwi, adc, diff_2d, diff_25d, wins, spacing) -> None:
    """4 x 3: rows DWI / ADC / 2D TP-FP-FN / 2.5D TP-FP-FN, columns z-1, z, z+1."""
    plt = _plt()
    sx, sy = spacing[0], spacing[1]
    zs = neighbours(z, dwi.shape[SLICE_AXIS])
    fig, axes = plt.subplots(4, 3, figsize=(7.5, 10))
    mapping = {TP: COLORS["tp"], FP: COLORS["fp"], FN: COLORS["fn"]}
    rows = (
        ("DWI", dwi, None),
        ("ADC", adc, None),
        ("2D TP/FP/FN", dwi, diff_2d),
        ("2.5D TP/FP/FN", dwi, diff_25d),
    )
    for r, (name, base, diff) in enumerate(rows):
        for c, k in enumerate(zs):
            ax = axes[r, c]
            if k is None:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "out of range",
                    ha="center",
                    va="center",
                    fontsize=7,
                    transform=ax.transAxes,
                )
                continue
            _show(ax, base[:, :, k], wins["adc" if name == "ADC" else "dwi"], sx, sy)
            if diff is not None:
                ax.imshow(
                    _overlay(diff[:, :, k], mapping).transpose(1, 0, 2),
                    origin="lower",
                    aspect=sy / sx,
                    interpolation="nearest",
                )
            if c == 1:
                _scale_bar(ax, sx)
            ax.set_title(f"{name}  k={k}", fontsize=7)
    fig.suptitle(
        f"{alias}  panel A  centre slice k={z}\n"
        "TP green (#009E73) / FP orange (#E69F00) / FN light blue (#56B4E9)",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def panel_b(path: Path, alias: str, z: int, dwi, gt, mdiff, win, spacing) -> None:
    """DWI + GT outline + model difference (both / 2D only / 2.5D only) on slice z."""
    plt = _plt()
    sx, sy = spacing[0], spacing[1]
    fig, ax = plt.subplots(figsize=(5, 5.4))
    _show(ax, dwi[:, :, z], win, sx, sy)
    ax.imshow(
        _overlay(
            mdiff[:, :, z],
            {
                MD_BOTH: COLORS["both"],
                MD_2D_ONLY: COLORS["2d_only"],
                MD_25D_ONLY: COLORS["25d_only"],
            },
        ).transpose(1, 0, 2),
        origin="lower",
        aspect=sy / sx,
        interpolation="nearest",
    )
    if gt[:, :, z].any():
        ax.contour(
            gt[:, :, z].T.astype(float),
            levels=[0.5],
            colors=COLORS["gt"],
            linewidths=LINE,
            origin="lower",
        )
    _scale_bar(ax, sx)
    ax.set_title(
        f"{alias}  panel B  k={z}\nGT outline pink / both yellow / 2D only blue / "
        "2.5D only vermillion",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def panel_c(
    path: Path, alias: str, voxel: Sequence[int], dwi, gt, diff_2d, diff_25d, win, spacing
) -> None:
    """Three planes through ``voxel``: rows 2D / 2.5D TP-FP-FN, columns i-j, i-k, j-k planes."""
    plt = _plt()
    i, j, k = (int(v) for v in voxel)
    sx, sy, sz = spacing[:3]
    planes = (
        ("plane k (i-j)", lambda a: a[:, :, k], sx, sy),
        ("plane j (i-k)", lambda a: a[:, j, :], sx, sz),
        ("plane i (j-k)", lambda a: a[i, :, :], sy, sz),
    )
    mapping = {TP: COLORS["tp"], FP: COLORS["fp"], FN: COLORS["fn"]}
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.4))
    for r, (name, diff) in enumerate((("2D", diff_2d), ("2.5D", diff_25d))):
        for c, (pname, cut, du, dv) in enumerate(planes):
            ax = axes[r, c]
            _show(ax, cut(dwi), win, du, dv)
            ax.imshow(
                _overlay(cut(diff), mapping).transpose(1, 0, 2),
                origin="lower",
                aspect=dv / du,
                interpolation="nearest",
            )
            g = cut(gt)
            if g.any():
                ax.contour(
                    g.T.astype(float),
                    levels=[0.5],
                    colors=COLORS["gt"],
                    linewidths=LINE,
                    origin="lower",
                )
            _scale_bar(ax, du)
            ax.set_title(f"{name} {pname}", fontsize=7)
    fig.suptitle(
        f"{alias}  panel C  through voxel ({i}, {j}, {k})\n"
        "TP green / FP orange / FN light blue / GT outline pink",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
