/**
 * Phase 7-0 spike (dev-only, /__spike?suite=p7&autorun=1): component labelmap colours and
 * per-label alpha in 2D/3D, NiiVue locationChange, programmatic navigation, event-loop
 * suppression and oblique-affine voxel <-> world round trips. Synthetic volumes only; an
 * optional `oblique` query parameter supplies a real grid geometry (dims, affine and a few
 * voxel indices — numbers only, no image data) for the oblique checks.
 */
import {
  type ComponentDisplay,
  componentColor,
  buildComponentLut,
  labelColor,
  SHOW_ALL,
} from "../features/viewer/labelLut";
import {
  applyAffine,
  type CrosshairLocation,
  connectCrosshair,
  type Vec3,
} from "../features/viewer/location";
import { hexToRgb, MASK_PALETTE } from "../features/viewer/ViewerEngine";
import { NiivueEngine } from "../features/viewer/NiivueEngine";
import type { CheckResult } from "./runSpike";
import { AFFINE, DIMS, gzip, niftiFromArray, voxToWorld } from "./synthetic";

type Rgb = readonly [number, number, number];
type Affine = number[][];

export interface ObliqueInput {
  name: string;
  dims: [number, number, number];
  affine: Affine;
  points: { label: number; vox: Vec3; mm: Vec3 }[];
}

/** Built-in oblique geometry: LAS, 30 deg about x then 20 deg about z, 1.5 x 1.5 x 3 mm. */
export function builtinOblique(): ObliqueInput {
  const rx = (30 * Math.PI) / 180;
  const rz = (20 * Math.PI) / 180;
  const R = [
    [Math.cos(rz), -Math.sin(rz) * Math.cos(rx), Math.sin(rz) * Math.sin(rx)],
    [Math.sin(rz), Math.cos(rz) * Math.cos(rx), -Math.cos(rz) * Math.sin(rx)],
    [0, Math.sin(rx), Math.cos(rx)],
  ];
  const zoom = [-1.5, 1.5, 3];
  const affine = [0, 1, 2].map((r) => [
    ...[0, 1, 2].map((c) => (R[r]?.[c] ?? 0) * (zoom[c] ?? 1)),
    [40, -60, -30][r] ?? 0,
  ]);
  affine.push([0, 0, 0, 1]);
  const pts: Vec3[] = [
    [30, 40, 12],
    [50, 20, 5],
    [10, 60, 25],
    [63, 63, 29],
    [0, 0, 0],
  ];
  return {
    name: "builtin",
    dims: [64, 72, 30],
    affine,
    points: pts.map((vox, n) => ({ label: n + 1, vox, mm: applyAffine(affine, vox) })),
  };
}

function snapshot(engine: NiivueEngine, canvas: HTMLCanvasElement): Uint8ClampedArray {
  const nv = engine.niivue;
  if (!nv) throw new Error("no viewer");
  nv.drawScene();
  const c = document.createElement("canvas");
  c.width = canvas.width;
  c.height = canvas.height;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("no 2d context");
  ctx.drawImage(canvas, 0, 0);
  return ctx.getImageData(0, 0, c.width, c.height).data;
}

function px(a: Uint8ClampedArray, p: number): Rgb {
  return [a[p] ?? 0, a[p + 1] ?? 0, a[p + 2] ?? 0];
}

function close(c: Rgb, d: Rgb, tol = 3): boolean {
  return (
    Math.abs(c[0] - d[0]) <= tol && Math.abs(c[1] - d[1]) <= tol && Math.abs(c[2] - d[2]) <= tol
  );
}

function changedPixels(a: Uint8ClampedArray, b: Uint8ClampedArray): number[] {
  const out: number[] = [];
  for (let p = 0; p < a.length; p += 4) {
    const d =
      Math.abs((a[p] ?? 0) - (b[p] ?? 0)) +
      Math.abs((a[p + 1] ?? 0) - (b[p + 1] ?? 0)) +
      Math.abs((a[p + 2] ?? 0) - (b[p + 2] ?? 0));
    if (d > 6) out.push(p);
  }
  return out;
}

/** label -> pixel offsets whose colour equals that label's colour (±3). */
function byLabel(img: Uint8ClampedArray, pixels: number[], labels: number[], width = 512) {
  const sets = new Map<number, number[]>(labels.map((l) => [l, []]));
  let unmatched = 0;
  const examples: string[] = [];
  for (const p of pixels) {
    const c = px(img, p);
    const hit = labels.find((l) => close(c, componentColor(l)));
    if (hit === undefined) {
      unmatched++;
      if (examples.length < 6)
        examples.push(`(${(p / 4) % width},${Math.floor(p / 4 / width)}:${c.join(".")})`);
    } else sets.get(hit)?.push(p);
  }
  return { sets, unmatched, examples };
}

function blendShare(
  base: Uint8ClampedArray,
  img: Uint8ClampedArray,
  pixels: number[],
  c: Rgb,
  o: number,
) {
  let ok = 0;
  for (const p of pixels) {
    const b = px(base, p);
    const v = px(img, p);
    if (
      [0, 1, 2].every(
        (ch) => Math.abs((v[ch] ?? 0) - ((b[ch] ?? 0) * (1 - o) + (c[ch] ?? 0) * o)) <= 4,
      )
    )
      ok++;
  }
  return pixels.length ? ok / pixels.length : 0;
}

function meanRgb(img: Uint8ClampedArray, pixels: number[]): Rgb {
  const s = [0, 0, 0];
  for (const p of pixels) for (let ch = 0; ch < 3; ch++) s[ch] = (s[ch] ?? 0) + (img[p + ch] ?? 0);
  const n = Math.max(1, pixels.length);
  return [(s[0] ?? 0) / n, (s[1] ?? 0) / n, (s[2] ?? 0) / n];
}

function chroma(c: Rgb): [number, number, number] {
  const t = c[0] + c[1] + c[2] || 1;
  return [c[0] / t, c[1] / t, c[2] / t];
}

function nearestLabel(c: Rgb, labels: number[]): number {
  const q = chroma(c);
  let best = -1;
  let bestD = Infinity;
  for (const l of labels) {
    const r = chroma(componentColor(l));
    const d = Math.hypot(q[0] - r[0], q[1] - r[1], q[2] - r[2]);
    if (d < bestD) {
      bestD = d;
      best = l;
    }
  }
  return best;
}

const idx = (i: number, j: number, k: number, d: readonly number[]) =>
  i + (d[0] ?? 0) * (j + (d[1] ?? 0) * k);

/** Constant "brain" base volume (float32) on any grid. */
function baseVolume(dims: readonly number[]): Float32Array {
  const n = (dims[0] ?? 0) * (dims[1] ?? 0) * (dims[2] ?? 0);
  return new Float32Array(n).fill(100);
}

/** Component blocks on slices k 8..11 with disjoint i ranges (no overlap in the front view). */
const BLOCKS: { label: number; i: [number, number]; j: [number, number] }[] = [
  { label: 1, i: [8, 15], j: [10, 17] },
  { label: 2, i: [22, 28], j: [38, 45] },
  { label: 3, i: [44, 52], j: [20, 27] },
  { label: 255, i: [34, 39], j: [26, 31] },
];
const BLOCK_LABELS = BLOCKS.map((b) => b.label);

async function componentVolumes(grid: boolean) {
  const n = DIMS[0] * DIMS[1] * DIMS[2];
  const labels = new Uint8Array(n);
  if (grid) {
    // axial slice k = 10: 16 x 16 cells of 4 x 4 voxels -> labels 0..255 (cell 0 = background)
    for (let j = 0; j < DIMS[1]; j++)
      for (let i = 0; i < DIMS[0]; i++)
        labels[idx(i, j, 10, DIMS)] = Math.floor(j / 4) * 16 + Math.floor(i / 4);
  } else {
    for (const b of BLOCKS)
      for (let k = 8; k <= 11; k++)
        for (let j = b.j[0]; j <= b.j[1]; j++)
          for (let i = b.i[0]; i <= b.i[1]; i++) labels[idx(i, j, k, DIMS)] = b.label;
  }
  const base = await gzip(niftiFromArray(baseVolume(DIMS), "float32", DIMS, AFFINE));
  return {
    dwi: base,
    adc: base.slice(0),
    mask: await gzip(niftiFromArray(labels, "uint8", DIMS, AFFINE)),
  };
}

async function obliqueVolumes(o: ObliqueInput) {
  const labels = new Uint8Array(o.dims[0] * o.dims[1] * o.dims[2]);
  for (const p of o.points) labels[idx(p.vox[0], p.vox[1], p.vox[2], o.dims)] = p.label;
  const base = await gzip(niftiFromArray(baseVolume(o.dims), "float32", o.dims, o.affine));
  return {
    dwi: base,
    adc: base.slice(0),
    mask: await gzip(niftiFromArray(labels, "uint8", o.dims, o.affine)),
  };
}

function sameVox(a: readonly number[], b: readonly number[]) {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

function maxAbs(a: readonly number[], b: readonly number[]) {
  return Math.max(...[0, 1, 2].map((i) => Math.abs((a[i] ?? 0) - (b[i] ?? 0))));
}

/** Label stored in the drawing bitmap (display grid) under the crosshair. */
function bitmapLabelAtCrosshair(engine: NiivueEngine): number {
  const nv = engine.niivue;
  const d = nv?.volumes[0]?.dimsRAS;
  const bmp = nv?.drawBitmap;
  if (!nv || !d || !bmp) return -1;
  const s = engine.getCrosshair().slice;
  return bmp[idx(s[0], s[1], s[2], [d[1] ?? 0, d[2] ?? 0, d[3] ?? 0])] ?? -1;
}

/** Parses the optional `oblique` query value (base64url JSON with dims, affine and points). */
export function parseObliqueParam(value: string | null): ObliqueInput | null {
  if (!value) return null;
  try {
    const json: unknown = JSON.parse(atob(value.replace(/-/g, "+").replace(/_/g, "/")));
    if (typeof json !== "object" || json === null) return null;
    const dims = Reflect.get(json, "dims");
    const affine = Reflect.get(json, "affine");
    const points = Reflect.get(json, "points");
    const nums = (a: unknown, n: number): a is number[] =>
      Array.isArray(a) &&
      a.length === n &&
      a.every((v) => typeof v === "number" && Number.isFinite(v));
    if (!nums(dims, 3) || !Array.isArray(affine) || affine.length < 3 || !Array.isArray(points)) {
      return null;
    }
    if (!affine.slice(0, 3).every((r) => nums(r, 4))) return null;
    const pts = points.flatMap((p: unknown) => {
      if (typeof p !== "object" || p === null) return [];
      const label = Reflect.get(p, "label");
      const vox = Reflect.get(p, "vox");
      const mm = Reflect.get(p, "mm");
      return typeof label === "number" && nums(vox, 3) && nums(mm, 3)
        ? [
            {
              label,
              vox: [vox[0] ?? 0, vox[1] ?? 0, vox[2] ?? 0] as Vec3,
              mm: [mm[0] ?? 0, mm[1] ?? 0, mm[2] ?? 0] as Vec3,
            },
          ]
        : [];
    });
    const rows = affine.slice(0, 3) as number[][];
    return {
      name: "param",
      dims: [dims[0] ?? 1, dims[1] ?? 1, dims[2] ?? 1],
      affine: [...rows, [0, 0, 0, 1]],
      points: pts,
    };
  } catch {
    return null;
  }
}

/** Manual check: the component blocks stay on screen, interactive, with a live location feed. */
export async function startManualSpike(
  canvas: HTMLCanvasElement,
  onLocation: (location: CrosshairLocation) => void,
): Promise<NiivueEngine> {
  const engine = new NiivueEngine();
  await engine.attach(canvas);
  await engine.load(await componentVolumes(false));
  engine.setMaskOpacity(1);
  engine.setComponentDisplay(SHOW_ALL);
  engine.setView("multiplanar");
  connectCrosshair(engine, onLocation);
  engine.jumpToVoxel([12, 13, 9]);
  return engine;
}

export async function runPhase7Spike(
  newCanvas: () => HTMLCanvasElement,
  extraOblique: ObliqueInput | null,
): Promise<CheckResult[]> {
  const out: CheckResult[] = [];
  const add = (id: string, label: string, pass: boolean, detail: string) =>
    out.push({ id: `p7:${id}`, label, pass, detail });
  const run = async (fn: () => Promise<void>, id: string) => {
    try {
      await fn();
    } catch (e) {
      add(`${id}_error`, `${id}: 예외`, false, e instanceof Error ? e.message : "unknown error");
    }
  };

  // --- 1. labels 0..255 on one axial slice (2D) -------------------------------------------
  await run(async () => {
    const canvas = newCanvas();
    const engine = new NiivueEngine();
    try {
      await engine.attach(canvas);
      await engine.load(await componentVolumes(true));
      engine.setView("axial");
      engine.setMaskOpacity(1);
      engine.setComponentDisplay(SHOW_ALL);
      engine.setSliceVoxel(2, 10);
      const lut = engine.drawLutBytes();
      const expected = buildComponentLut(SHOW_ALL);
      add(
        "lut_exact",
        "draw LUT = labelLut 결과와 바이트 단위 일치 (label 0 alpha 0)",
        lut.length === 1024 && expected.every((v, i) => lut[i] === v) && lut[3] === 0,
        `bytes=${lut.length} alpha0=${lut[3]}`,
      );
      engine.setMaskVisible(false);
      const off = snapshot(engine, canvas);
      engine.setMaskVisible(true);
      const on = snapshot(engine, canvas);
      const changed = changedPixels(off, on);
      // NiiVue draws orientation letters / crosshair on top of the slice: compare only pixels whose
      // unmasked colour is the plain base grey (the modal colour), and report the overlay pixels.
      const counts = new Map<string, number>();
      for (const p of changed)
        counts.set(px(off, p).join(), (counts.get(px(off, p).join()) ?? 0) + 1);
      const mode = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "";
      const plain = changed.filter((p) => px(off, p).join() === mode);
      const all = Array.from({ length: 255 }, (_, n) => n + 1);
      const { sets, unmatched, examples } = byLabel(on, plain, all);
      const missing = all.filter((l) => (sets.get(l)?.length ?? 0) === 0);
      add(
        "grid_255",
        "2D: label 1~254 + 255(기타) 모두 고유 색으로 정확히 표시",
        missing.length === 0,
        `labels shown=${255 - missing.length}/255 missing=[${missing.slice(0, 8).join(",")}]`,
      );
      add(
        "grid_nearest",
        "2D nearest: 경계 혼합 픽셀 없음 (overlay 제외 모든 변경 픽셀이 label 색 중 하나)",
        unmatched === 0 && plain.length > 0,
        `compared=${plain.length} unmatched=${unmatched} overlay(text/crosshair) excluded=${changed.length - plain.length} ${examples.join(" ")}`,
      );
      engine.setComponentDisplay(null);
      const p6 = engine.drawLutBytes();
      const orange = hexToRgb(MASK_PALETTE.orange);
      add(
        "phase6_lut",
        "setComponentDisplay(null) → Phase 6 단색 LUT 복귀 (0 투명, 1 = #E69F00)",
        p6[3] === 0 &&
          p6[4] === orange[0] &&
          p6[5] === orange[1] &&
          p6[6] === orange[2] &&
          p6[7] === 255,
        `alpha0=${p6[3]} label1=${p6[4]},${p6[5]},${p6[6]},${p6[7]}`,
      );
    } finally {
      engine.dispose();
    }
  }, "grid");

  // --- 2. selection / dim / hide / restore in 2D and 3D ------------------------------------
  await run(async () => {
    const canvas = newCanvas();
    const engine = new NiivueEngine();
    try {
      await engine.attach(canvas);
      await engine.load(await componentVolumes(false));
      engine.setMaskOpacity(1);
      engine.setView("axial");
      engine.setSliceVoxel(2, 9);
      const shot = (display: ComponentDisplay | null, visible = true) => {
        if (display) engine.setComponentDisplay(display);
        engine.setMaskVisible(visible);
        return snapshot(engine, canvas);
      };
      const off = shot(SHOW_ALL, false);
      const all = shot(SHOW_ALL);
      const changed = changedPixels(off, all);
      const { sets, unmatched } = byLabel(all, changed, BLOCK_LABELS);
      const counts = BLOCK_LABELS.map((l) => sets.get(l)?.length ?? 0);
      add(
        "2d_colors",
        "2D: label 1,2,3,255 각각 고유 색 + label 0 투명 (변경 픽셀 = label 픽셀)",
        counts.every((c) => c > 0) && unmatched === 0,
        `pixels per label=${counts.join("/")} other=${unmatched}`,
      );
      const set = (l: number) => sets.get(l) ?? [];
      const sel = shot({ selected: 2, hidden: new Set(), dimStyle: "alpha" });
      const selExact = set(2).filter((p) => close(px(sel, p), componentColor(2))).length;
      const dimShares = [1, 3, 255].map((l) =>
        blendShare(off, sel, set(l), componentColor(l), 64 / 255),
      );
      add(
        "2d_selected",
        "2D 선택: 선택 label alpha 1.0 (정확한 색)",
        selExact === set(2).length && selExact > 0,
        `exact=${selExact}/${set(2).length}`,
      );
      add(
        "2d_dimmed",
        "2D 비선택: alpha 0.25 (= base·0.75 + color·0.25, ±4)",
        dimShares.every((s) => s >= 0.99),
        `blend match label1/3/255=${dimShares.map((s) => s.toFixed(3)).join("/")}`,
      );
      const grayDisplay: ComponentDisplay = { selected: 2, hidden: new Set(), dimStyle: "gray" };
      const gray = shot(grayDisplay);
      const grayExact = [1, 2, 3, 255].map(
        (l) =>
          set(l).filter((p) => close(px(gray, p), labelColor(l, grayDisplay))).length /
          Math.max(1, set(l).length),
      );
      add(
        "2d_dimmed_gray",
        "2D 대안(gray): 비선택 label = 회색 혼합 색(불투명), 선택 label = 원래 색",
        grayExact.every((r) => r === 1),
        `exact share label1/2/3/255=${grayExact.map((r) => r.toFixed(3)).join("/")}`,
      );
      const hid = shot({ selected: null, hidden: new Set([3]) });
      const hiddenLeft = changedPixels(off, hid).filter((p) => set(3).includes(p)).length;
      const othersKept = [1, 2, 255].every((l) =>
        set(l).every((p) => close(px(hid, p), componentColor(l))),
      );
      add(
        "2d_hidden",
        "2D 숨김: label 3 alpha 0 (원래 영상 그대로), 나머지 유지",
        hiddenLeft === 0 && othersKept,
        `label3 pixels still coloured=${hiddenLeft} others kept=${othersKept}`,
      );
      const back = shot(SHOW_ALL);
      add(
        "2d_restore",
        "2D 전체 표시 복원 = 초기 화면과 동일",
        changedPixels(all, back).length === 0,
        `diff=${changedPixels(all, back).length}`,
      );
      engine.jumpToVoxel([48, 23, 9]); // inside label 3
      for (const view of ["coronal", "sagittal"] as const) {
        engine.setView(view);
        const v = changedPixels(shot(SHOW_ALL, false), shot(SHOW_ALL));
        const r = byLabel(snapshot(engine, canvas), v, BLOCK_LABELS);
        add(
          `2d_${view}`,
          `${view}: label 색 정확 (label 3 단면)`,
          (r.sets.get(3)?.length ?? 0) > 0 && r.unmatched === 0,
          `label3 pixels=${r.sets.get(3)?.length ?? 0} other=${r.unmatched}`,
        );
      }

      // 3D (background off, front camera): isolate each label to find its screen region
      engine.setView("render");
      engine.setRender3DBackground(false);
      engine.setCameraPreset("front");
      const off3 = shot(SHOW_ALL, false);
      const all3 = shot(SHOW_ALL);
      const region = new Map<number, number[]>();
      const nearest: string[] = [];
      for (const l of BLOCK_LABELS) {
        const only = shot({ selected: null, hidden: new Set(BLOCK_LABELS.filter((x) => x !== l)) });
        const r = changedPixels(off3, only);
        region.set(l, r);
        nearest.push(`${l}->${nearestLabel(meanRgb(only, r), BLOCK_LABELS)}`);
      }
      const regionOk = BLOCK_LABELS.every((l) => (region.get(l)?.length ?? 0) > 0);
      add(
        "3d_colors",
        "3D: label별 색 구분 (각 영역 평균 색도가 자기 label 색에 가장 가까움)",
        regionOk && nearest.every((s) => s.split("->")[0] === s.split("->")[1]),
        `regions=${BLOCK_LABELS.map((l) => region.get(l)?.length ?? 0).join("/")} nearest ${nearest.join(" ")}`,
      );
      const lum = (img: Uint8ClampedArray, l: number) => {
        const c = meanRgb(img, region.get(l) ?? []);
        return c[0] + c[1] + c[2];
      };
      const sel3 = shot({ selected: 2, hidden: new Set(), dimStyle: "alpha" });
      const ratioSel = lum(sel3, 2) / Math.max(1, lum(all3, 2));
      const ratioDim = [1, 3, 255].map((l) => lum(sel3, l) / Math.max(1, lum(all3, l)));
      add(
        "3d_selected",
        "3D 선택: 선택 label 밝기 유지 (비율 0.9~1.1)",
        ratioSel > 0.9 && ratioSel < 1.1,
        `ratio=${ratioSel.toFixed(3)}`,
      );
      add(
        "3d_dimmed",
        "3D 비선택: label별 alpha 반영 (밝기 비율 0.05~0.6, 0이 아님)",
        ratioDim.every((r) => r > 0.05 && r < 0.6),
        `ratio label1/3/255=${ratioDim.map((r) => r.toFixed(3)).join("/")}`,
      );
      const sat = (img: Uint8ClampedArray, l: number) => {
        const c = meanRgb(img, region.get(l) ?? []);
        return Math.max(...c) - Math.min(...c);
      };
      const gray3 = shot({ selected: 2, hidden: new Set(), dimStyle: "gray" });
      const graySel = lum(gray3, 2) / Math.max(1, lum(all3, 2));
      const graySat = [1, 3, 255].map((l) => sat(gray3, l) / Math.max(1, sat(all3, l)));
      const grayLum = [1, 3, 255].map((l) => lum(gray3, l) / Math.max(1, lum(all3, l)));
      add(
        "3d_dimmed_gray",
        "3D 대안(gray): 비선택 label 채도 감소(비율 < 0.4), 선택 label 밝기 유지",
        graySel > 0.9 && graySel < 1.1 && graySat.every((r) => r < 0.4),
        `selected=${graySel.toFixed(3)} saturation label1/3/255=${graySat.map((r) => r.toFixed(3)).join("/")} brightness=${grayLum.map((r) => r.toFixed(3)).join("/")}`,
      );
      const hid3 = shot({ selected: null, hidden: new Set([3]) });
      const r3 = region.get(3) ?? [];
      const left3 = changedPixels(off3, hid3).filter((p) => r3.includes(p)).length;
      add(
        "3d_hidden",
        "3D 숨김: label 3 영역이 배경으로 돌아감 (잔여 < 5%)",
        r3.length > 0 && left3 / r3.length < 0.05,
        `remaining=${left3}/${r3.length}`,
      );
      const back3 = shot(SHOW_ALL);
      add(
        "3d_restore",
        "3D 전체 표시 복원 = 초기 화면과 동일",
        changedPixels(all3, back3).length === 0,
        `diff=${changedPixels(all3, back3).length}`,
      );

      // --- location events and programmatic navigation (synthetic LAS grid) ----------------
      engine.setView("axial");
      const events: CrosshairLocation[] = [];
      const unsubscribe = engine.onLocationChange((l) => events.push(l));
      const target: Vec3 = [40, 38, 9];
      engine.jumpToVoxel(target);
      const e1 = events.at(-1);
      add(
        "loc_jumpToVoxel",
        "jumpToVoxel → location event: vox 일치, mm = affine·vox, frac 0~1, slider(LAS: 63−i)",
        !!e1 &&
          sameVox(e1.vox, target) &&
          maxAbs(e1.mm, voxToWorld(...target)) < 1e-3 &&
          e1.frac.every((f) => f > 0 && f < 1) &&
          sameVox(e1.slice, [63 - 40, 38, 9]),
        e1 ? `vox=${e1.vox} mm=${e1.mm.map((v) => v.toFixed(3))} slice=${e1.slice}` : "no event",
      );
      engine.setSliceVoxel(2, 5);
      const e2 = events.at(-1);
      add(
        "loc_setSliceVoxel",
        "setSliceVoxel(axial, 5) → event: slice[2]=5, 나머지 축 유지",
        !!e2 && e2.slice[2] === 5 && e2.slice[0] === 23 && e2.slice[1] === 38 && e2.vox[2] === 5,
        e2 ? `slice=${e2.slice} vox=${e2.vox}` : "no event",
      );
      const before = { ...engine.locationStats };
      const n0 = events.length;
      engine.stepSlice(2, 1);
      add(
        "loc_niivueEvent",
        "NiiVue 공개 locationChange(moveCrosshairInVox) 수신 + type guard 통과",
        engine.locationStats.niivueEvents > before.niivueEvents &&
          engine.locationStats.rejected === before.rejected &&
          events.length > n0 &&
          events.at(-1)?.slice[2] === 6,
        `niivue events +${engine.locationStats.niivueEvents - before.niivueEvents} rejected=${engine.locationStats.rejected} slice=${events.at(-1)?.slice}`,
      );
      const rect = canvas.getBoundingClientRect();
      const click = (fx: number, fy: number) => {
        const init = {
          bubbles: true,
          cancelable: true,
          button: 0,
          buttons: 1,
          clientX: rect.left + rect.width * fx,
          clientY: rect.top + rect.height * fy,
        };
        canvas.dispatchEvent(new MouseEvent("mousedown", init));
        canvas.dispatchEvent(new MouseEvent("mouseup", { ...init, buttons: 0 }));
      };
      const beforeClick = engine.locationStats.niivueEvents;
      const startVox = engine.getCrosshair().vox;
      click(0.3, 0.3);
      const clicked = events.at(-1);
      add(
        "loc_mouse",
        "Canvas 클릭(mousedown/up) → NiiVue locationChange → 정규화 위치 (축 슬라이스 유지)",
        engine.locationStats.niivueEvents > beforeClick &&
          !!clicked &&
          !sameVox(clicked.vox, startVox) &&
          clicked.slice[2] === 6,
        `events +${engine.locationStats.niivueEvents - beforeClick} vox ${startVox} -> ${clicked?.vox} slice=${clicked?.slice}`,
      );

      // event loop: real engine events through the sink with a manual frame scheduler
      const tasks: (() => void)[] = [];
      let applied = 0;
      const disconnect = connectCrosshair(
        engine,
        () => applied++,
        (t) => tasks.push(t),
      );
      const frame = () => tasks.splice(0).forEach((t) => t());
      for (let n = 0; n < 100; n++) engine.jumpToVoxel([20, 20, 9]);
      frame();
      for (let n = 0; n < 100; n++) engine.jumpToVoxel([20, 20, 9]);
      frame();
      const afterSame = applied;
      const path: Vec3[] = [
        [21, 20, 9],
        [21, 20, 9],
        [22, 20, 9],
        [22, 20, 9],
        [21, 20, 9],
      ];
      for (const v of path) {
        engine.jumpToVoxel(v);
        engine.jumpToVoxel(v);
        frame();
      }
      disconnect();
      add(
        "loop",
        "event loop 방지: 같은 voxel 200회 → store 갱신 1회, 이후 5프레임 중 voxel 변화 3회 → 3회 추가",
        afterSame === 1 && applied === 4,
        `same-voxel updates=${afterSame} total=${applied}`,
      );
      let roundTrip = 0;
      for (let n = 0; n < 200; n++) {
        const v: Vec3 = [(n * 37) % DIMS[0], (n * 53) % DIMS[1], (n * 7) % DIMS[2]];
        engine.jumpToVoxel(v);
        const c = engine.getCrosshair();
        if (sameVox(c.vox, v) && maxAbs(c.mm, voxToWorld(...v)) < 1e-3) roundTrip++;
      }
      add(
        "roundtrip_las",
        "LAS 격자 voxel → mm → voxel 왕복 200개",
        roundTrip === 200,
        `${roundTrip}/200`,
      );
      unsubscribe();
    } finally {
      engine.dispose();
    }
  }, "components");

  // --- 3. oblique affine ---------------------------------------------------------------------
  for (const o of [builtinOblique(), ...(extraOblique ? [extraOblique] : [])]) {
    await run(async () => {
      const canvas = newCanvas();
      const engine = new NiivueEngine();
      try {
        await engine.attach(canvas);
        await engine.load(await obliqueVolumes(o));
        engine.setView("multiplanar");
        engine.setComponentDisplay(SHOW_ALL);
        let ok = 0;
        let worst = 0;
        let labelsOk = 0;
        let sliderOk = 0;
        const detail: string[] = [];
        let phase6Err = 0;
        for (const p of o.points) {
          engine.jumpToVoxel(p.vox);
          const c = engine.getCrosshair();
          const err = maxAbs(c.mm, p.mm);
          worst = Math.max(worst, err);
          if (sameVox(c.vox, p.vox) && err < 0.01) ok++;
          if (bitmapLabelAtCrosshair(engine) === p.label) labelsOk++;
          const s = c.slice;
          let stable = true;
          for (const axis of [0, 1, 2] as const) {
            engine.setSliceVoxel(axis, s[axis]);
            if (!sameVox(engine.getCrosshair().vox, p.vox)) stable = false;
          }
          if (stable) sliderOk++;
          detail.push(`L${p.label} vox ${p.vox}→slider ${s}`);
          engine.jumpToWorldMm(p.mm); // Phase 6 path (orthogonal approximation)
          phase6Err = Math.max(phase6Err, maxAbs(engine.getCrosshair().vox, p.vox));
        }
        const n = o.points.length;
        add(
          `obl_${o.name}_jump`,
          `oblique(${o.name}): jumpToVoxel → event vox 정확, mm 오차 < 0.01 mm`,
          ok === n,
          `${ok}/${n} worst mm err=${worst.toExponential(2)}`,
        );
        add(
          `obl_${o.name}_label`,
          `oblique(${o.name}): 이동한 crosshair 위치의 labelmap 값 = 해당 component`,
          labelsOk === n,
          `${labelsOk}/${n}`,
        );
        add(
          `obl_${o.name}_slider`,
          `oblique(${o.name}): slider 값(axial/coronal/sagittal) 재설정 시 voxel 불변`,
          sliderOk === n,
          `${sliderOk}/${n}; ${detail.join("; ")}`,
        );
        let rt = 0;
        const [dx, dy, dz] = o.dims;
        for (let k = 0; k < 300; k++) {
          const v: Vec3 = [(k * 37) % dx, (k * 53) % dy, (k * 11) % dz];
          engine.jumpToVoxel(v);
          const c = engine.getCrosshair();
          if (sameVox(c.vox, v) && maxAbs(c.mm, applyAffine(o.affine, v)) < 0.01) rt++;
        }
        add(
          `obl_${o.name}_roundtrip`,
          `oblique(${o.name}): voxel → mm → voxel 왕복 300개 (오차 < 0.01 mm)`,
          rt === 300,
          `${rt}/300`,
        );
        add(
          `obl_${o.name}_phase6Info`,
          `정보: Phase 6 jumpToWorldMm(직교 근사)의 oblique voxel 오차 (판정 아님)`,
          true,
          `max voxel error=${phase6Err}`,
        );
      } finally {
        engine.dispose();
      }
    }, `obl_${o.name}`);
  }
  return out;
}
