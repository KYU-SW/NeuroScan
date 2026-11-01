/**
 * Phase 7-0: crosshair location model and one-way synchronisation.
 *
 * Data flow (the engine is the single source of truth for the crosshair):
 *   UI command (slider, keys, component) --> engine --> location event --> sink --> store
 * There is no store -> engine subscription, so an engine event can never echo back into the
 * engine. The sink coalesces events per animation frame and drops updates whose integer voxel
 * did not change, so mouse drags and repeated programmatic moves cannot flood the store.
 *
 * Coordinates:
 *   vox   - integer voxel index in the *file* grid of the base volume (the API's voxel space)
 *   mm    - RAS+ scanner world millimetres through the true (possibly oblique) affine
 *   frac  - engine texture fraction 0..1 (display grid, RAS order)
 *   slice - integer slice index per anatomical axis [sagittal, coronal, axial] of the display
 *           grid; this is what the slice sliders show
 * Pure module: no NiiVue import. The payload of NiiVue's locationChange event is typed unknown,
 * so it is checked at run time and copied (NiiVue passes its live crosshair array).
 */
import type { SliceAxis, ViewerEngine } from "./ViewerEngine";

export type Vec3 = [number, number, number];
/** Row-major 4x4 affine (NIfTI sform/qform convention: world = A * [i, j, k, 1]). */
export type Affine = readonly (readonly number[])[];

export interface CrosshairLocation {
  vox: Vec3;
  mm: Vec3;
  frac: Vec3;
  slice: Vec3;
}

export interface RawLocation {
  frac: Vec3;
  mm: Vec3;
}

function finiteTriple(value: unknown): Vec3 | null {
  if (!Array.isArray(value) && !(ArrayBuffer.isView(value) && !(value instanceof DataView))) {
    return null;
  }
  const arr = value as ArrayLike<unknown>;
  if (arr.length < 3) return null;
  const out: number[] = [];
  for (let i = 0; i < 3; i++) {
    const v = arr[i];
    if (typeof v !== "number" || !Number.isFinite(v)) return null;
    out.push(v);
  }
  return [out[0] ?? 0, out[1] ?? 0, out[2] ?? 0];
}

/** Run-time check of a NiiVue locationChange payload; returns copies of frac and mm. */
export function parseNiivueLocation(raw: unknown): RawLocation | null {
  if (typeof raw !== "object" || raw === null) return null;
  const frac = finiteTriple(Reflect.get(raw, "frac"));
  const mm = finiteTriple(Reflect.get(raw, "mm"));
  if (!frac || !mm) return null;
  if (frac.some((f) => f < -1e-6 || f > 1 + 1e-6)) return null;
  return { frac, mm };
}

function row(a: Affine, r: number): readonly number[] {
  const out = a[r];
  if (!out || out.length < 4) throw new Error("invalid affine");
  return out;
}

export function applyAffine(a: Affine, p: Vec3): Vec3 {
  const [x, y, z] = p;
  const f = (r: number) => {
    const m = row(a, r);
    return (m[0] ?? 0) * x + (m[1] ?? 0) * y + (m[2] ?? 0) * z + (m[3] ?? 0);
  };
  return [f(0), f(1), f(2)];
}

/** Inverse of the 3x4 part of an affine (rotation/zoom/shear + translation). */
export function invertAffine(a: Affine): number[][] {
  const m = [0, 1, 2].map((r) => row(a, r));
  const g = (r: number, c: number) => m[r]?.[c] ?? 0;
  const det =
    g(0, 0) * (g(1, 1) * g(2, 2) - g(1, 2) * g(2, 1)) -
    g(0, 1) * (g(1, 0) * g(2, 2) - g(1, 2) * g(2, 0)) +
    g(0, 2) * (g(1, 0) * g(2, 1) - g(1, 1) * g(2, 0));
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) throw new Error("singular affine");
  const inv = [
    [
      (g(1, 1) * g(2, 2) - g(1, 2) * g(2, 1)) / det,
      (g(0, 2) * g(2, 1) - g(0, 1) * g(2, 2)) / det,
      (g(0, 1) * g(1, 2) - g(0, 2) * g(1, 1)) / det,
    ],
    [
      (g(1, 2) * g(2, 0) - g(1, 0) * g(2, 2)) / det,
      (g(0, 0) * g(2, 2) - g(0, 2) * g(2, 0)) / det,
      (g(0, 2) * g(1, 0) - g(0, 0) * g(1, 2)) / det,
    ],
    [
      (g(1, 0) * g(2, 1) - g(1, 1) * g(2, 0)) / det,
      (g(0, 1) * g(2, 0) - g(0, 0) * g(2, 1)) / det,
      (g(0, 0) * g(1, 1) - g(0, 1) * g(1, 0)) / det,
    ],
  ];
  return inv.map((r, i) => {
    const t = -((r[0] ?? 0) * g(0, 3) + (r[1] ?? 0) * g(1, 3) + (r[2] ?? 0) * g(2, 3));
    return [r[0] ?? 0, r[1] ?? 0, r[2] ?? 0, i < 3 ? t : 0];
  });
}

export function worldToVoxel(inverse: Affine, mm: Vec3): Vec3 {
  return applyAffine(inverse, mm);
}

/** Integer voxel index inside dims (throws otherwise). */
export function checkVoxel(vox: readonly number[], dims: readonly number[]): Vec3 {
  if (vox.length !== 3) throw new Error("voxel must have 3 indices");
  vox.forEach((v, i) => {
    const d = dims[i] ?? 0;
    if (!Number.isInteger(v) || v < 0 || v >= d) throw new Error("voxel out of range");
  });
  return [vox[0] ?? 0, vox[1] ?? 0, vox[2] ?? 0];
}

export function sliceOf(frac: Vec3, dimsRAS: readonly number[]): Vec3 {
  const s = (a: SliceAxis) => {
    const d = dimsRAS[a] ?? 1;
    return Math.min(d - 1, Math.max(0, Math.floor((frac[a] ?? 0) * d)));
  };
  return [s(0), s(1), s(2)];
}

/** Engine-independent normalisation shared by NiivueEngine and FakeViewerEngine. */
export function toCrosshairLocation(
  raw: RawLocation,
  inverseAffine: Affine,
  fileDims: readonly number[],
  dimsRAS: readonly number[],
): CrosshairLocation {
  const v = worldToVoxel(inverseAffine, raw.mm);
  const vox: Vec3 = [0, 1, 2].map((i) =>
    Math.min((fileDims[i] ?? 1) - 1, Math.max(0, Math.round(v[i] ?? 0))),
  ) as Vec3;
  return { vox, mm: [...raw.mm], frac: [...raw.frac], slice: sliceOf(raw.frac, dimsRAS) };
}

export function sameVoxel(a: CrosshairLocation | null, b: CrosshairLocation | null): boolean {
  return !!a && !!b && a.vox[0] === b.vox[0] && a.vox[1] === b.vox[1] && a.vox[2] === b.vox[2];
}

export type Schedule = (task: () => void) => void;

const nextFrame: Schedule = (task) => {
  requestAnimationFrame(() => task());
};

export interface LocationSink {
  push(location: CrosshairLocation): void;
  /** Applies the pending location now (tests / frame callback). */
  flush(): void;
  close(): void;
}

/** Coalesces locations per frame and skips updates that stay on the same integer voxel. */
export function createLocationSink(
  apply: (location: CrosshairLocation) => void,
  schedule: Schedule = nextFrame,
): LocationSink {
  let pending: CrosshairLocation | null = null;
  let last: CrosshairLocation | null = null;
  let scheduled = false;
  let closed = false;
  const flush = () => {
    scheduled = false;
    const next = pending;
    pending = null;
    if (closed || !next || sameVoxel(last, next)) return;
    last = next;
    apply(next);
  };
  return {
    push(location) {
      if (closed) return;
      pending = location;
      if (!scheduled) {
        scheduled = true;
        schedule(flush);
      }
    },
    flush,
    close() {
      closed = true;
      pending = null;
    },
  };
}

/** Engine events -> store. Returns a disconnect function (unsubscribe + close the sink). */
export function connectCrosshair(
  engine: Pick<ViewerEngine, "onLocationChange">,
  apply: (location: CrosshairLocation) => void,
  schedule?: Schedule,
): () => void {
  const sink = createLocationSink(apply, schedule);
  const unsubscribe = engine.onLocationChange((loc) => sink.push(loc));
  return () => {
    unsubscribe();
    sink.close();
  };
}
