/**
 * Phase 7-A component review helpers (pure; no React, no engine).
 *
 * Positions are file-grid voxels of the base volume. Navigation always goes through a voxel:
 * the backend's representative voxel (inside the component) or, for results made before
 * Phase 7-A, the voxel computed from the API's world mm with the volume's true affine — never
 * through NiiVue's axis-aligned approximation (wrong by up to 20 voxels on oblique images).
 */
import type { ResultComponent, ViewerVolume } from "../../api/schemas";
import { applyAffine, invertAffine, type Vec3 } from "../viewer/location";
import type { ComponentFilter, ComponentSort } from "../viewer/viewerStore";

/** "작은 분리 영역": a region other than the largest with less than this share of the largest
 * region's volume. A display/sorting convention only — no medical meaning; frozen before Test. */
export const SMALL_REGION_RATIO = 0.1;

export function isSmallRegion(c: ResultComponent, largestMl: number): boolean {
  if (c.size_rank === 1 || largestMl <= 0) return false;
  return c.volume_ml < SMALL_REGION_RATIO * largestMl;
}

function clampVoxel(v: readonly number[], shape: readonly number[]): Vec3 {
  return [0, 1, 2].map((i) =>
    Math.min((shape[i] ?? 1) - 1, Math.max(0, Math.round(v[i] ?? 0))),
  ) as Vec3;
}

export function worldToFileVoxel(mm: readonly number[], volume: ViewerVolume): Vec3 {
  const inv = invertAffine(volume.affine);
  return clampVoxel(applyAffine(inv, [mm[0] ?? 0, mm[1] ?? 0, mm[2] ?? 0]), volume.shape);
}

/** Voxel to move to when the component is selected (always inside it when the API provides
 * the representative voxel). */
export function targetVoxel(c: ResultComponent, volume: ViewerVolume): Vec3 {
  if (c.representative_voxel) return clampVoxel(c.representative_voxel, volume.shape);
  return worldToFileVoxel(c.centroid_world_mm, volume);
}

/** Lesion centre (API centre of mass, file-grid voxel) — replaces the Phase 6 world-mm jump. */
export function lesionCenterVoxel(com: readonly number[] | null, volume: ViewerVolume | undefined) {
  return com && volume ? clampVoxel(com, volume.shape) : null;
}

/** 1-based slice range as shown by the axial slider (display grid runs inferior -> superior). */
export function displaySliceRange(
  c: ResultComponent,
  volume: ViewerVolume,
): { first: number; last: number } | null {
  const axis =
    c.slice_range?.axis ?? [...volume.orientation].findIndex((ch) => ch === "S" || ch === "I");
  if (axis < 0 || axis > 2) return null;
  const first = c.slice_range?.first ?? c.bbox_min_voxel[axis] ?? 0;
  const last = c.slice_range?.last ?? c.bbox_max_voxel[axis] ?? 0;
  const n = volume.shape[axis] ?? 1;
  const flip = volume.orientation[axis] === "I";
  const a = flip ? n - 1 - first : first;
  const b = flip ? n - 1 - last : last;
  return { first: Math.min(a, b) + 1, last: Math.max(a, b) + 1 };
}

export function visibleComponents(
  components: readonly ResultComponent[],
  sort: ComponentSort,
  filter: ComponentFilter,
  selected: number | null,
  hidden: ReadonlySet<number>,
): ResultComponent[] {
  const list = components.filter((c) => {
    if (filter === "selected") return c.display_label !== null && c.display_label === selected;
    if (filter === "visible") return c.display_label === null || !hidden.has(c.display_label);
    return true;
  });
  const rank = (c: ResultComponent) => c.size_rank ?? Number.MAX_SAFE_INTEGER;
  // ties in volume keep the backend's size rank order (largest first) for a stable list
  return list.sort((a, b) =>
    sort === "largest"
      ? b.volume_ml - a.volume_ml || rank(a) - rank(b)
      : a.volume_ml - b.volume_ml || rank(b) - rank(a),
  );
}
