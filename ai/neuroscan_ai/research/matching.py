"""Prediction vs ground-truth component matching by voxel overlap (research comparison only).

Components: 26-connectivity on both masks (``analysis.components``), identified by size rank
(1 = largest) — the same numbering as the viewer's "영역 n" (prediction) and "GT n".
Matching (MVP): every prediction/GT component pair that shares at least one voxel is an edge;
connected groups of that bipartite overlap graph are reported, with no IoU threshold and no
centroid-distance assignment. Nothing is dropped for being small.

Group types: one_to_one (1 P : 1 G), split (n P : 1 G), merge (1 P : n G), complex (n : m),
prediction_only (P without any GT overlap), ground_truth_only (G without any P overlap).
Centroid distance (mm, true affine) is auxiliary information only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from neuroscan_ai.analysis.components import component_analysis, label_components
from neuroscan_ai.preprocessing.mask import voxel_volume_mm3


def _group_type(n_pred: int, n_gt: int) -> str:
    if n_gt == 0:
        return "prediction_only"
    if n_pred == 0:
        return "ground_truth_only"
    if n_pred == 1 and n_gt == 1:
        return "one_to_one"
    if n_gt == 1:
        return "split"
    if n_pred == 1:
        return "merge"
    return "complex"


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _rank_map(mask: np.ndarray, spacing, affine, slice_axis):
    """Labelmap of size ranks (int32: rank per voxel) + component list (largest first)."""
    _, comps = component_analysis(mask, spacing, affine, slice_axis)
    labels, n = label_components(mask)  # same deterministic labelling as component_analysis
    lut = np.zeros(n + 1, dtype=np.int32)
    for c in comps:
        lut[c["label"]] = c["size_rank"]
    return lut[labels], comps


def match_components(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    spacing: Sequence[float],
    affine: np.ndarray,
    slice_axis: int,
) -> dict[str, Any]:
    pred_ranks, pred_comps = _rank_map(prediction, spacing, affine, slice_axis)
    gt_ranks, gt_comps = _rank_map(ground_truth, spacing, affine, slice_axis)
    ml = voxel_volume_mm3(spacing) / 1000.0
    pred_size = {c["size_rank"]: c["voxel_count"] for c in pred_comps}
    gt_size = {c["size_rank"]: c["voxel_count"] for c in gt_comps}

    both = (pred_ranks > 0) & (gt_ranks > 0)
    pairs: dict[tuple[int, int], int] = {}
    if both.any():
        keys, counts = np.unique(
            np.stack([pred_ranks[both], gt_ranks[both]], axis=1), axis=0, return_counts=True
        )
        pairs = {(int(p), int(g)): int(n) for (p, g), n in zip(keys, counts, strict=True)}

    uf = _UnionFind()
    for r in pred_size:
        uf.find(("p", r))
    for r in gt_size:
        uf.find(("g", r))
    for p, g in pairs:
        uf.union(("p", p), ("g", g))
    members: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for node in list(uf.parent):
        members.setdefault(uf.find(node), []).append(node)

    def centroid(ranks_arr: np.ndarray, ranks: list[int]) -> np.ndarray | None:
        if not ranks:
            return None
        idx = np.argwhere(np.isin(ranks_arr, ranks))
        return (affine @ np.array([*idx.mean(axis=0), 1.0]))[:3]

    groups = []
    for nodes in members.values():
        p_ids = sorted(r for kind, r in nodes if kind == "p")
        g_ids = sorted(r for kind, r in nodes if kind == "g")
        overlap = sum(n for (p, g), n in pairs.items() if p in p_ids and g in g_ids)
        p_vox = sum(pred_size[r] for r in p_ids)
        g_vox = sum(gt_size[r] for r in g_ids)
        union = p_vox + g_vox - overlap
        cp, cg = centroid(pred_ranks, p_ids), centroid(gt_ranks, g_ids)
        groups.append(
            {
                "group_type": _group_type(len(p_ids), len(g_ids)),
                "prediction_component_ids": p_ids,
                "ground_truth_component_ids": g_ids,
                "overlap_voxel_count": int(overlap),
                "union_voxel_count": int(union),
                "iou": overlap / union if union else None,
                "prediction_volume_ml": p_vox * ml,
                "gt_volume_ml": g_vox * ml,
                "centroid_distance_mm": None
                if cp is None or cg is None
                else float(np.linalg.norm(cp - cg)),
            }
        )
    # stable order: groups with prediction components by their best rank, then GT-only groups
    groups.sort(
        key=lambda g: (
            0 if g["prediction_component_ids"] else 1,
            (g["prediction_component_ids"] or g["ground_truth_component_ids"])[0],
        )
    )
    for i, g in enumerate(groups, start=1):
        g["group_id"] = i
    group_of = {
        (kind, r): g["group_id"]
        for g in groups
        for kind, ids in (
            ("p", g["prediction_component_ids"]),
            ("g", g["ground_truth_component_ids"]),
        )
        for r in ids
    }
    type_of = {g["group_id"]: g["group_type"] for g in groups}

    pred_info = []
    for c in pred_comps:
        r = c["size_rank"]
        tp = sum(n for (p, _), n in pairs.items() if p == r)
        gid = group_of[("p", r)]
        pred_info.append(
            {
                "size_rank": r,
                "gt_overlap": tp > 0,
                "group_id": gid,
                "group_type": type_of[gid],
                "tp_voxels": int(tp),
                "fp_voxels": int(c["voxel_count"] - tp),
                "tp_ratio": tp / c["voxel_count"],
                "ground_truth_component_ids": sorted(g for (p, g) in pairs if p == r),
            }
        )
    gt_info = []
    for c in gt_comps:
        r = c["size_rank"]
        detected = sum(n for (_, g), n in pairs.items() if g == r)
        gid = group_of[("g", r)]
        gt_info.append(
            {
                "size_rank": r,
                "voxel_count": c["voxel_count"],
                "volume_ml": c["voxel_count"] * ml,
                "representative_voxel": c["representative_voxel"],
                "representative_world_mm": c["representative_world_mm"],
                "slice_range": c["slice_range"],
                "detected_voxels": int(detected),
                "missed_voxels": int(c["voxel_count"] - detected),
                "group_id": gid,
                "group_type": type_of[gid],
                "prediction_component_ids": sorted(p for (p, g) in pairs if g == r),
            }
        )
    return {
        "method": "overlap_graph_26_connectivity_min_1_voxel",
        "groups": groups,
        "prediction_components": pred_info,
        "ground_truth_components": gt_info,
    }


def check_matching(matching: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    """Totals of the matching must equal evaluate_case's voxel counts (empty = OK)."""
    out = []
    pred = matching["prediction_components"]
    gt = matching["ground_truth_components"]
    if sum(p["tp_voxels"] for p in pred) != metrics["tp_voxels"]:
        out.append("matching TP total")
    if sum(p["fp_voxels"] for p in pred) != metrics["fp_voxels"]:
        out.append("matching FP total")
    if sum(g["missed_voxels"] for g in gt) != metrics["fn_voxels"]:
        out.append("matching FN total")
    if sum(g["overlap_voxel_count"] for g in matching["groups"]) != metrics["tp_voxels"]:
        out.append("matching group overlap total")
    if len(pred) != metrics["component_count_prediction"]:
        out.append("prediction component count")
    if len(gt) != metrics["component_count_gt"]:
        out.append("ground-truth component count")
    return out
