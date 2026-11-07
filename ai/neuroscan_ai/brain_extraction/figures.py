"""Static brain-mask QA figure (Panels A-D) and an automatic layout check.

Panel A  axial middle slice + mask outline (primary tool)
Panel B  slice with the largest GT area + GT fill + mask outline
Panel C  axial / coronal / sagittal mask outlines through the GT centroid (or the volume centre)
Panel D  one axis per tool on the Panel B slice (all outlines in a shared legend)

Titles contain the alias and categories only. The same layout is used for SOOP and ISLES.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from neuroscan_ai.brain_extraction.interfaces import (  # noqa: E402
    TOOL_HDBET,
    TOOL_SYNTHSTRIP,
    TOOL_V1,
    BrainExtractionError,
    validate_alias,
)

STYLE = {
    TOOL_V1: {"color": "#8c8c8c", "linestyle": "--", "label": "v1 background"},
    TOOL_SYNTHSTRIP: {"color": "#0072B2", "linestyle": "-", "label": "SynthStrip"},
    TOOL_HDBET: {"color": "#E69F00", "linestyle": "-", "label": "HD-BET"},
}
GT_COLOR = (0.8, 0.0, 0.8)  # magenta
GT_ALPHA = 0.45


def _plane(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    return np.take(vol, idx, axis=axis).T  # rows = second remaining axis, shown with origin lower


def _show(ax, img2d: np.ndarray, title: str) -> None:
    lo, hi = np.percentile(img2d, [1, 99]) if img2d.size else (0, 1)
    ax.imshow(img2d, cmap="gray", origin="lower", vmin=lo, vmax=hi if hi > lo else lo + 1)
    ax.set_title("\n".join(textwrap.wrap(title, 34)), fontsize=9)
    ax.axis("off")


def _outline(ax, mask2d: np.ndarray, tool: str) -> None:
    if mask2d.any() and not mask2d.all():
        st = STYLE[tool]
        ax.contour(mask2d.astype(float), levels=[0.5], colors=[st["color"]],
                   linestyles=[st["linestyle"]], linewidths=1.2, origin="lower")  # fmt: skip


def _gt(ax, gt2d: np.ndarray | None) -> None:
    if gt2d is not None and gt2d.any():
        rgba = np.zeros((*gt2d.shape, 4))
        rgba[gt2d] = (*GT_COLOR, GT_ALPHA)
        ax.imshow(rgba, origin="lower")


def check_layout(fig) -> list[str]:
    """Text inside the figure; axes titles must not overlap another axes; axes must not overlap."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    problems = []
    texts = list(fig.texts) + ([fig._suptitle] if fig._suptitle else [])
    for ax in fig.axes:
        texts.append(ax.title)
        if ax.get_legend() is not None:
            texts += ax.get_legend().get_texts()
    for t in texts:
        if not t.get_text() or not t.get_visible():
            continue
        e = t.get_window_extent(r)
        if e.x0 < fb.x0 - 1 or e.y0 < fb.y0 - 1 or e.x1 > fb.x1 + 1 or e.y1 > fb.y1 + 1:
            problems.append(f"text outside figure: {t.get_text()[:30]!r}")
    boxes = [ax.get_window_extent(r) for ax in fig.axes]
    for i, ax in enumerate(fig.axes):
        if ax.title.get_text():
            e = ax.title.get_window_extent(r)
            for j, b in enumerate(boxes):
                if j != i and e.overlaps(b):
                    problems.append(f"title of axes {i} overlaps axes {j}")
    for t in fig.texts:
        e = t.get_window_extent(r)
        if any(e.overlaps(b) for b in boxes):
            problems.append("figure text overlaps an axes")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            ov = (min(a.x1, b.x1) - max(a.x0, b.x0)) * (min(a.y1, b.y1) - max(a.y0, b.y0))
            if a.overlaps(b) and ov > 4:
                problems.append(f"axes {i} and {j} overlap")
    return problems


def case_figure(
    trace: np.ndarray,
    gt: np.ndarray | None,
    masks: dict[str, np.ndarray],
    alias: str,
    categories: list[str] | None = None,
    primary: str | None = None,
):
    """Build (not save) the Panel A-D figure; returns (fig, info)."""
    validate_alias(alias)
    if trace.ndim != 3 or any(m.shape != trace.shape for m in masks.values()):
        raise BrainExtractionError("TRACE and masks must be 3D on the same grid")
    if not masks or any(t not in STYLE for t in masks):
        raise BrainExtractionError("masks must be keyed by known tool names")
    primary = primary or next(iter(masks))
    zmid = trace.shape[2] // 2
    if gt is not None and gt.any():
        zgt = int(np.argmax(gt.sum(axis=(0, 1))))
        cen = np.argwhere(gt).mean(axis=0).round().astype(int)
    else:
        zgt, cen = zmid, np.array(trace.shape) // 2
    fig = plt.figure(figsize=(16, 10.5))
    head = f"{alias}" + (f"   categories: {', '.join(categories)}" if categories else "")
    fig.text(0.01, 0.985, head, va="top", ha="left", fontsize=12, family="monospace")
    n_tools = len(masks)
    gs = fig.add_gridspec(2, max(5, n_tools), left=0.02, right=0.98, top=0.88, bottom=0.08,
                          wspace=0.12, hspace=0.32)  # fmt: skip
    ax = fig.add_subplot(gs[0, 0])
    _show(ax, _plane(trace, 2, zmid), f"A  axial mid slice k={zmid} ({STYLE[primary]['label']})")
    _outline(ax, _plane(masks[primary], 2, zmid), primary)
    ax = fig.add_subplot(gs[0, 1])
    _show(ax, _plane(trace, 2, zgt), f"B  largest GT slice k={zgt}")
    _gt(ax, None if gt is None else _plane(gt, 2, zgt))
    _outline(ax, _plane(masks[primary], 2, zgt), primary)
    for col, (axis, name) in zip((2, 3, 4), ((2, "axial"), (1, "coronal"), (0, "sagittal")),
                                 strict=True):  # fmt: skip
        ax = fig.add_subplot(gs[0, col])
        _show(ax, _plane(trace, axis, int(cen[axis])), f"C  {name} {int(cen[axis])}")
        _outline(ax, _plane(masks[primary], axis, int(cen[axis])), primary)
    for col, (tool, m) in enumerate(masks.items()):
        ax = fig.add_subplot(gs[1, col])
        _show(ax, _plane(trace, 2, zgt), f"D  {STYLE[tool]['label']} (k={zgt})")
        _gt(ax, None if gt is None else _plane(gt, 2, zgt))
        _outline(ax, _plane(m, 2, zgt), tool)
    handles = [Line2D([0], [0], color=STYLE[t]["color"], linestyle=STYLE[t]["linestyle"],
                      label=STYLE[t]["label"]) for t in masks]  # fmt: skip
    handles.append(Patch(facecolor=(*GT_COLOR, GT_ALPHA), label="acute GT"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.005))  # fmt: skip
    info = {"alias": alias, "axial_mid": zmid, "gt_max_axial": zgt, "centre": cen.tolist(),
            "tools": list(masks), "primary": primary}  # fmt: skip
    return fig, info


def save_case_figure(fig, out_path: Path) -> list[str]:
    """Run the layout check, then save (refuses to overwrite). Returns layout problems."""
    if out_path.exists():
        raise BrainExtractionError(f"{out_path.name} exists; refusing to overwrite")
    problems = check_layout(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return problems


__all__: list[Any] = ["STYLE", "case_figure", "check_layout", "save_case_figure"]
