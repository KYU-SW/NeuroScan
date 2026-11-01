/** Phase 7-0: component LUT, location guard, one-way crosshair sync and voxel <-> mm maths.
 * Pure modules + FakeViewerEngine (jsdom has no WebGL; the NiiVue side is the /__spike page). */
import { describe, expect, it } from "vitest";
import { create } from "zustand";

import { FAKE_DIMS, FakeViewerEngine } from "../src/features/viewer/FakeViewerEngine";
import {
  buildComponentLut,
  checkDisplay,
  type ComponentDisplay,
  componentColor,
  DIM_ALPHA,
  DIM_GRAY,
  DIM_GRAY_MIX,
  isDisplayLabel,
  labelAlpha,
  labelColor,
  MAX_COMPONENT_LABEL,
  OTHER_COLOR,
  OTHER_LABEL,
  SHOW_ALL,
} from "../src/features/viewer/labelLut";
import {
  type Affine,
  applyAffine,
  connectCrosshair,
  type CrosshairLocation,
  createLocationSink,
  invertAffine,
  parseNiivueLocation,
  toCrosshairLocation,
  type Vec3,
} from "../src/features/viewer/location";

const rgba = (lut: Uint8ClampedArray, label: number) => [...lut.slice(label * 4, label * 4 + 4)];
const sel = (selected: number | null, hidden: number[] = []): ComponentDisplay => ({
  selected,
  hidden: new Set(hidden),
});

describe("component label LUT", () => {
  it("1. label 0 is fully transparent in every display state", () => {
    for (const d of [
      SHOW_ALL,
      sel(3),
      sel(null, [1, 2]),
      { ...sel(4), dimStyle: "gray" as const },
    ]) {
      expect(rgba(buildComponentLut(d), 0)).toEqual([0, 0, 0, 0]);
      expect(labelAlpha(0, d)).toBe(0);
    }
  });

  it("2. labels 1..254 have valid, pairwise distinct colours away from grey", () => {
    const colours = Array.from({ length: MAX_COMPONENT_LABEL }, (_, n) => componentColor(n + 1));
    expect(new Set(colours.map((c) => c.join())).size).toBe(254);
    let minDistance = Infinity;
    for (let a = 0; a < colours.length; a++) {
      const c = colours[a] ?? [0, 0, 0];
      expect(c.every((v) => Number.isInteger(v) && v >= 0 && v <= 255)).toBe(true);
      expect(Math.max(...c) - Math.min(...c)).toBeGreaterThanOrEqual(70); // not grey-like
      for (let b = a + 1; b < colours.length; b++) {
        const d = colours[b] ?? [0, 0, 0];
        minDistance = Math.min(
          minDistance,
          Math.max(...[0, 1, 2].map((i) => Math.abs((c[i] ?? 0) - (d[i] ?? 0)))),
        );
      }
    }
    expect(minDistance).toBeGreaterThanOrEqual(35);
    const lut = buildComponentLut(SHOW_ALL);
    expect(lut).toHaveLength(1024);
    for (let l = 1; l <= 254; l++) expect(rgba(lut, l)).toEqual([...componentColor(l), 255]);
  });

  it("3. label 255 is the reserved neutral 'other' entry", () => {
    expect(OTHER_LABEL).toBe(255);
    expect(componentColor(255)).toEqual(OTHER_COLOR);
    expect(rgba(buildComponentLut(SHOW_ALL), 255)).toEqual([...OTHER_COLOR, 255]);
    const colours = Array.from({ length: 254 }, (_, n) => componentColor(n + 1).join());
    expect(colours).not.toContain(OTHER_COLOR.join());
    expect(isDisplayLabel(255)).toBe(true);
    expect(rgba(buildComponentLut(sel(null, [255])), 255)[3]).toBe(0);
  });

  it("4./5. selected label keeps its colour; others are grey-dimmed by default (2D = 3D)", () => {
    const g = buildComponentLut(sel(7)); // Phase 7-A default policy: dimStyle "gray"
    expect(rgba(g, 7)).toEqual([...componentColor(7), 255]);
    const spread = (c: readonly number[]) => Math.max(...c) - Math.min(...c);
    for (const l of [1, 6, 8, 254, 255]) {
      const dimmed = labelColor(l, sel(7));
      expect(rgba(g, l)).toEqual([...dimmed, 255]); // opaque: no partial alpha
      expect(dimmed).toEqual(
        componentColor(l).map((v, i) =>
          Math.round(v * (1 - DIM_GRAY_MIX) + (DIM_GRAY[i] ?? 0) * DIM_GRAY_MIX),
        ),
      );
      if (l !== 255) expect(spread(dimmed)).toBeLessThan(spread(componentColor(l)) * 0.4);
    }
    expect(buildComponentLut({ ...sel(7), dimStyle: "gray" })).toEqual(g);
    // the partial-alpha style stays available only for the 7-0 spike comparison
    const alpha = buildComponentLut({ ...sel(7), dimStyle: "alpha" });
    expect(rgba(alpha, 8)).toEqual([...componentColor(8), Math.round(255 * DIM_ALPHA)]);
  });

  it("single colour mode: every label uses the palette colour, dimming still applies", () => {
    const orange = [230, 159, 0] as const;
    const lut = buildComponentLut({ ...SHOW_ALL, coloring: "single" }, orange);
    for (const l of [1, 2, 200, 255]) expect(rgba(lut, l)).toEqual([...orange, 255]);
    expect(rgba(lut, 0)).toEqual([0, 0, 0, 0]);
    const sel3 = buildComponentLut({ ...sel(3), coloring: "single" }, orange);
    expect(rgba(sel3, 3)).toEqual([...orange, 255]);
    expect(rgba(sel3, 4)).toEqual([
      ...labelColor(4, { ...sel(3), coloring: "single" }, orange),
      255,
    ]);
    expect(() => checkDisplay({ ...SHOW_ALL, coloring: "rainbow" as "single" })).toThrow();
  });

  it("6. hidden labels get alpha 0, also when selected", () => {
    const lut = buildComponentLut(sel(5, [5, 9]));
    expect(rgba(lut, 5)[3]).toBe(0);
    expect(rgba(lut, 9)[3]).toBe(0);
    expect(rgba(lut, 10)[3]).toBe(255); // dimmed (grey), still visible
    expect(rgba(buildComponentLut({ ...sel(5, [9]), dimStyle: "gray" }), 9)[3]).toBe(0);
  });

  it("7. restoring 'show all' gives exactly the initial LUT", () => {
    const initial = buildComponentLut(SHOW_ALL);
    buildComponentLut(sel(3, [1, 2]));
    expect(buildComponentLut(sel(null))).toEqual(initial);
  });

  it("8. invalid labels and display states are rejected", () => {
    for (const bad of [0, -1, 1.5, 256, Number.NaN, Infinity]) {
      expect(isDisplayLabel(bad)).toBe(false);
      expect(() => componentColor(bad)).toThrow();
      expect(() => buildComponentLut(sel(bad))).toThrow();
      expect(() => buildComponentLut(sel(null, [bad]))).toThrow();
    }
    expect(isDisplayLabel("3")).toBe(false);
    expect(() => checkDisplay(Object.assign(sel(1), { dimStyle: "blur" as "gray" }))).toThrow();
    const engine = new FakeViewerEngine();
    expect(() => engine.setComponentDisplay(sel(300))).toThrow();
    engine.setComponentDisplay(sel(2));
    expect(engine.componentDisplay?.selected).toBe(2);
    engine.setComponentDisplay(null); // back to the Phase 6 single-colour mask
    expect(engine.componentDisplay).toBeNull();
  });
});

describe("location payload type guard", () => {
  it("9. accepts NiiVue-shaped payloads, copies them, rejects everything else", () => {
    const frac = new Float32Array([0.25, 0.5, 0.75]);
    const raw = { frac, mm: new Float32Array([1, 2, 3, 1]), vox: [1, 2, 3], string: "x" };
    const parsed = parseNiivueLocation(raw);
    expect(parsed).toEqual({ frac: [0.25, 0.5, 0.75], mm: [1, 2, 3] });
    frac[0] = 0.9; // NiiVue passes its live crosshair array
    expect(parsed?.frac[0]).toBe(0.25);
    for (const bad of [
      null,
      undefined,
      42,
      "frac",
      {},
      { frac: [0.1, 0.2, 0.3] },
      { mm: [1, 2, 3] },
      { frac: [0.1, 0.2], mm: [1, 2, 3] },
      { frac: [0.1, Number.NaN, 0.3], mm: [1, 2, 3] },
      { frac: [0.1, 0.2, 1.5], mm: [1, 2, 3] },
      { frac: ["0.1", 0.2, 0.3], mm: [1, 2, 3] },
      { frac: [0.1, 0.2, 0.3], mm: new DataView(new ArrayBuffer(12)) },
    ]) {
      expect(parseNiivueLocation(bad)).toBeNull();
    }
  });
});

describe("one-way crosshair synchronisation", () => {
  const loc = (vox: Vec3): CrosshairLocation => ({
    vox,
    mm: vox,
    frac: [0.5, 0.5, 0.5],
    slice: vox,
  });

  it("10. the sink drops updates that stay on the same integer voxel", () => {
    const applied: Vec3[] = [];
    const sink = createLocationSink(
      (l) => applied.push(l.vox),
      (task) => task(),
    );
    sink.push(loc([1, 2, 3]));
    sink.push(loc([1, 2, 3]));
    sink.push(loc([1, 2, 4]));
    sink.push(loc([1, 2, 4]));
    sink.push(loc([1, 2, 3]));
    expect(applied).toEqual([
      [1, 2, 3],
      [1, 2, 4],
      [1, 2, 3],
    ]);
  });

  it("11. a slider command goes to the engine and comes back as a location event", () => {
    const engine = new FakeViewerEngine();
    const seen: CrosshairLocation[] = [];
    engine.onLocationChange((l) => seen.push(l));
    engine.setSliceVoxel(2, 7);
    expect(engine.called("setSliceVoxel")).toEqual([[2, 7]]);
    expect(seen.at(-1)?.slice).toEqual([32, 32, 7]);
    expect(seen.at(-1)?.vox).toEqual([32, 32, 7]);
    expect(() => engine.setSliceVoxel(2, FAKE_DIMS[2])).toThrow();
    expect(() => engine.setSliceVoxel(0, 1.5)).toThrow();
    engine.jumpToVoxel([10, 11, 12]);
    expect(seen.at(-1)?.vox).toEqual([10, 11, 12]);
    expect(() => engine.jumpToVoxel([FAKE_DIMS[0], 0, 0])).toThrow();
  });

  it("12. engine events reach the store; store changes never call the engine", () => {
    const store = create<{
      crosshair: CrosshairLocation | null;
      set: (l: CrosshairLocation) => void;
    }>()((set) => ({ crosshair: null, set: (crosshair) => set({ crosshair }) }));
    const engine = new FakeViewerEngine();
    const tasks: (() => void)[] = [];
    const disconnect = connectCrosshair(engine, store.getState().set, (t) => tasks.push(t));
    expect(engine.emitRawLocation({ frac: [0.3, 0.4, 0.5], mm: [19, 25, 9] })).toBe(true);
    expect(store.getState().crosshair).toBeNull(); // applied on the next frame
    tasks.splice(0).forEach((t) => t());
    expect(store.getState().crosshair?.vox).toEqual([19, 25, 9]);
    expect(store.getState().crosshair?.slice).toEqual([33, 44, 36]); // floor(frac * 112|112|73)
    const calls = engine.calls.length;
    store.getState().set(loc([1, 1, 1]));
    expect(engine.calls.length).toBe(calls);
    expect(engine.emitRawLocation({ frac: [2, 0, 0], mm: [0, 0, 0] })).toBe(false);
    disconnect();
    engine.jumpToVoxel([5, 5, 5]);
    tasks.splice(0).forEach((t) => t());
    expect(store.getState().crosshair?.vox).toEqual([1, 1, 1]);
  });

  it("13. 100 repeated events cause only the needed store updates (frame batching)", () => {
    const engine = new FakeViewerEngine();
    const tasks: (() => void)[] = [];
    let updates = 0;
    const frame = () => tasks.splice(0).forEach((t) => t());
    connectCrosshair(
      engine,
      () => updates++,
      (t) => tasks.push(t),
    );
    for (let n = 0; n < 100; n++) engine.jumpToVoxel([20, 20, 5]);
    frame();
    expect(updates).toBe(1);
    expect(tasks).toHaveLength(0);
    for (let n = 0; n < 100; n++) engine.jumpToVoxel([20, 20, 5]);
    frame();
    expect(updates).toBe(1);
    // a drag: 100 events spread over 10 frames, ending on 10 different voxels
    for (let f = 0; f < 10; f++) {
      for (let n = 0; n < 10; n++) engine.jumpToVoxel([21 + f, 20, n % 2]);
      frame();
    }
    expect(updates).toBe(11);
  });
});

describe("voxel <-> world mm", () => {
  const LAS: Affine = [
    [-2, 0, 0, 63],
    [0, 2, 0, -63],
    [0, 0, 3, -27],
    [0, 0, 0, 1],
  ];
  /** Synthetic oblique grid: 31 deg about x, 15 deg about z, anisotropic, LAS-like flip. */
  function oblique(): Affine {
    const ax = (31 * Math.PI) / 180;
    const az = (15 * Math.PI) / 180;
    const R = [
      [Math.cos(az), -Math.sin(az) * Math.cos(ax), Math.sin(az) * Math.sin(ax)],
      [Math.sin(az), Math.cos(az) * Math.cos(ax), -Math.cos(az) * Math.sin(ax)],
      [0, Math.sin(ax), Math.cos(ax)],
    ];
    const zoom = [-1.8, 1.8, 2.5];
    const t = [101.2, -88.4, -40.7];
    return [
      ...[0, 1, 2].map((r) => [
        ...[0, 1, 2].map((c) => (R[r]?.[c] ?? 0) * (zoom[c] ?? 1)),
        t[r] ?? 0,
      ]),
      [0, 0, 0, 1],
    ];
  }

  it("14. voxel -> mm -> voxel round trip is exact (LAS and oblique)", () => {
    for (const a of [LAS, oblique()]) {
      const inv = invertAffine(a);
      let worst = 0;
      for (let n = 0; n < 500; n++) {
        const v: Vec3 = [(n * 37) % 112, (n * 53) % 112, (n * 11) % 73];
        const back = applyAffine(inv, applyAffine(a, v));
        worst = Math.max(worst, ...[0, 1, 2].map((i) => Math.abs((back[i] ?? 0) - (v[i] ?? 0))));
      }
      expect(worst).toBeLessThan(1e-9);
    }
    expect(applyAffine(LAS, [40, 38, 9])).toEqual([-17, 13, 0]);
    expect(() =>
      invertAffine([
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 1, 0],
      ]),
    ).toThrow();
  });

  it("15. oblique: a location in world mm maps back to the exact file voxel and slider", () => {
    const a = oblique();
    const inv = invertAffine(a);
    const dims = [112, 112, 73];
    const dimsRAS = [112, 73, 112]; // display grid is a permutation of the file grid
    for (const v of [
      [69, 48, 37],
      [0, 0, 0],
      [111, 111, 72],
      [55, 12, 30],
    ] as Vec3[]) {
      const mm = applyAffine(a, v);
      const jitter: Vec3 = [mm[0] + 0.2, mm[1] - 0.2, mm[2] + 0.1]; // inside the same voxel
      const l = toCrosshairLocation({ frac: [0.5, 0.25, 0.999], mm: jitter }, inv, dims, dimsRAS);
      expect(l.vox).toEqual(v);
      expect(l.slice).toEqual([56, 18, 111]);
    }
  });
});
