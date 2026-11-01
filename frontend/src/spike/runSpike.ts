/**
 * NiiVue drawing-layer spike: pixel-level checks in a real WebGL2 browser (dev-only page).
 * Snapshot = drawScene() then copy the WebGL canvas into a 2D canvas in the same task.
 * Re-run after any NiiVue version change: /__spike?autorun=1
 */
import {
  CAMERA_PRESETS,
  DEFAULT_MASK_COLOR,
  DEFAULT_MASK_OPACITY,
  hexToRgb,
  isMaskColorName,
  MASK_COLOR_NAMES,
  MASK_PALETTE,
  RENDER_BACKGROUND_OPACITY,
} from "../features/viewer/ViewerEngine";
import { type MaskColorStrategy, NiivueEngine } from "../features/viewer/NiivueEngine";
import { DIMS, LESION, LESION_CENTER_MM, syntheticVolumes, voxToWorld } from "./synthetic";

export interface CheckResult {
  id: string;
  label: string;
  pass: boolean;
  detail: string;
}

type Rgb = [number, number, number];

interface Diff {
  changed: number;
  exact: number;
  cx: number;
  cy: number;
  set: Set<number>;
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

function diff(a: Uint8ClampedArray, b: Uint8ClampedArray, width: number, color: Rgb | null): Diff {
  let changed = 0;
  let exact = 0;
  let sx = 0;
  let sy = 0;
  const set = new Set<number>();
  for (let p = 0; p < a.length; p += 4) {
    const d =
      Math.abs((a[p] ?? 0) - (b[p] ?? 0)) +
      Math.abs((a[p + 1] ?? 0) - (b[p + 1] ?? 0)) +
      Math.abs((a[p + 2] ?? 0) - (b[p + 2] ?? 0));
    if (d <= 6) continue;
    changed++;
    const px = p / 4;
    set.add(px);
    sx += px % width;
    sy += Math.floor(px / width);
    if (
      color &&
      Math.abs((b[p] ?? 0) - color[0]) <= 3 &&
      Math.abs((b[p + 1] ?? 0) - color[1]) <= 3 &&
      Math.abs((b[p + 2] ?? 0) - color[2]) <= 3
    )
      exact++;
  }
  return { changed, exact, cx: changed ? sx / changed : -1, cy: changed ? sy / changed : -1, set };
}

/** Share of changed pixels whose colour equals base*(1-opacity) + colour*opacity (±4). */
function blendMatch(
  a: Uint8ClampedArray,
  b: Uint8ClampedArray,
  set: Set<number>,
  c: Rgb,
  o: number,
) {
  let ok = 0;
  for (const px of set) {
    const p = px * 4;
    let good = true;
    for (let ch = 0; ch < 3; ch++) {
      const expected = (a[p + ch] ?? 0) * (1 - o) + (c[ch] ?? 0) * o;
      if (Math.abs((b[p + ch] ?? 0) - expected) > 4) good = false;
    }
    if (good) ok++;
  }
  return set.size ? ok / set.size : 0;
}

function grayHistogram(a: Uint8ClampedArray): Map<number, number> {
  const hist = new Map<number, number>();
  for (let p = 0; p < a.length; p += 4) {
    const g = a[p] ?? 0;
    if (g > 10 && g < 250 && g === a[p + 1] && g === a[p + 2]) hist.set(g, (hist.get(g) ?? 0) + 1);
  }
  return hist;
}

/** The voxel checkerboard shows as two dominant gray levels under nearest filtering. Under
 * linear filtering those exact levels (almost) disappear because neighbouring voxels blend. */
function checkerLevelShare(nearest: Uint8ClampedArray, linear: Uint8ClampedArray) {
  const hn = grayHistogram(nearest);
  const levels = [...hn.entries()]
    .sort((x, y) => y[1] - x[1])
    .slice(0, 2)
    .map(([g]) => g);
  const share = (h: Map<number, number>) => {
    const total = [...h.values()].reduce((s, v) => s + v, 0);
    return total ? levels.reduce((s, g) => s + (h.get(g) ?? 0), 0) / total : 0;
  };
  return { levels, nearest: share(hn), linear: share(grayHistogram(linear)) };
}

function jaccard(x: Set<number>, y: Set<number>): number {
  let inter = 0;
  for (const v of x) if (y.has(v)) inter++;
  const union = x.size + y.size - inter;
  return union ? inter / union : 1;
}

export async function runSpike(
  canvas: HTMLCanvasElement,
  strategy: MaskColorStrategy,
): Promise<CheckResult[]> {
  const out: CheckResult[] = [];
  const add = (id: string, label: string, pass: boolean, detail: string) =>
    out.push({ id: `${strategy}:${id}`, label, pass, detail });
  const color: Rgb =
    strategy === "customLut" ? hexToRgb(MASK_PALETTE[DEFAULT_MASK_COLOR]) : [255, 0, 0];
  const engine = new NiivueEngine(strategy);
  const W = canvas.width;
  const area = canvas.width * canvas.height;
  let disposedNv: NiivueEngine["niivue"] = null;
  try {
    await engine.attach(canvas);
    await engine.load(await syntheticVolumes());
    add(
      "load",
      ".nii.gz ArrayBuffer 3개 로드 (DWI·ADC·mask)",
      true,
      "NVImage.loadFromUrl({url: ArrayBuffer})",
    );

    engine.jumpToWorldMm(LESION_CENTER_MM);
    engine.setView("axial");
    engine.setMaskOpacity(1);
    const masked = (show: boolean) => {
      engine.setMaskVisible(show);
      return snapshot(engine, canvas);
    };
    const showHide = (rgb: Rgb | null = color): Diff => {
      const a = masked(false);
      const b = masked(true);
      return diff(a, b, W, rgb);
    };

    const ax = showHide();
    add(
      "transparent0",
      "mask 0 투명 (변경 픽셀이 병변 영역에 국한)",
      ax.changed > 0 && ax.changed < area * 0.05,
      `changed=${ax.changed} (${((100 * ax.changed) / area).toFixed(2)}% of canvas)`,
    );
    add(
      "color1",
      `mask 1 = 지정 색 rgb(${color.join(",")}) (opacity 1)`,
      ax.changed > 0 && ax.exact / ax.changed >= 0.99,
      `exact=${ax.exact}/${ax.changed}`,
    );
    add(
      "nearestMask",
      "mask nearest: 경계 혼합 픽셀 없음 (DWI linear 활성 상태)",
      ax.changed > 0 && ax.exact / ax.changed >= 0.99,
      `blended boundary pixels=${ax.changed - ax.exact}`,
    );

    const nv = engine.niivue;
    if (!nv) throw new Error("no viewer");
    engine.setMaskVisible(false);
    nv.setInterpolation(true); // measurement only: compare against nearest filtering
    const nearestImg = snapshot(engine, canvas);
    nv.setInterpolation(false);
    const linearImg = snapshot(engine, canvas);
    const cl = checkerLevelShare(nearestImg, linearImg);
    add(
      "linearBase",
      "DWI linear: voxel checkerboard 값이 보간으로 섞임 (nearest 지배 회색 2개 비율 linear ≪ nearest)",
      cl.nearest >= 0.8 && cl.linear <= 0.2,
      `checker levels ${cl.levels.join("/")}: nearest=${cl.nearest.toFixed(3)} linear=${cl.linear.toFixed(3)}`,
    );
    const recheck = showHide();
    add(
      "nearestAfterLinear",
      "linear 복귀 후에도 mask 경계 유지",
      recheck.changed > 0 && recheck.exact / recheck.changed >= 0.99,
      `exact=${recheck.exact}/${recheck.changed}`,
    );

    for (const view of ["coronal", "sagittal"] as const) {
      engine.setView(view);
      const d = showHide();
      add(
        `view_${view}`,
        `${view}: mask 표시·경계 유지`,
        d.changed > 0 && d.exact / d.changed >= 0.99,
        `changed=${d.changed} exact=${d.exact}`,
      );
    }
    engine.setView("multiplanar");
    const mpr = showHide();
    add("view_mpr", "MPR(3면): mask 표시", mpr.changed > 0, `changed=${mpr.changed}`);

    engine.setView("axial");
    engine.setMaskOpacity(0.5);
    const half = showHide();
    add(
      "opacity",
      "opacity 0.5: 같은 영역, 혼합 색",
      half.changed > 0 &&
        Math.abs(half.changed - ax.changed) / ax.changed < 0.1 &&
        half.exact / half.changed < 0.05,
      `changed=${half.changed} (opacity1 ${ax.changed}) exact=${half.exact}`,
    );
    engine.setMaskOpacity(1);

    engine.setRadiological(true);
    const radio = showHide();
    engine.setRadiological(false);
    const neuro = showHide();
    engine.setRadiological(true);
    add(
      "orientation",
      "LAS 환자 좌측 병변: radiological=화면 오른쪽, neurological=왼쪽",
      radio.cx > W / 2 && neuro.cx < W / 2,
      `centroid x radiological=${radio.cx.toFixed(0)} neurological=${neuro.cx.toFixed(0)} (W/2=${W / 2})`,
    );

    const dwiSet = radio.set;
    engine.setBase("adc");
    const adcD = showHide();
    const j = jaccard(dwiSet, adcD.set);
    add(
      "baseSwitch",
      "DWI→ADC 전환 후 mask 좌표 동일",
      j >= 0.98 && adcD.exact / adcD.changed >= 0.99,
      `jaccard=${j.toFixed(4)} exact=${adcD.exact}/${adcD.changed}`,
    );
    engine.setBase("dwi");
    engine.jumpToWorldMm(LESION_CENTER_MM);

    const w0 = snapshot(engine, canvas);
    const range = engine.getWindowRange();
    engine.setWindow({ min: range.min, max: range.min + (range.max - range.min) * 0.3 });
    const w1 = snapshot(engine, canvas);
    engine.resetWindow();
    const w2 = snapshot(engine, canvas);
    const dW = diff(w0, w1, W, null).changed;
    const dR = diff(w0, w2, W, null).changed;
    add("window", "window/level 변경·초기화", dW > 0 && dR === 0, `changed=${dW} afterReset=${dR}`);

    engine.jumpToWorldMm(voxToWorld(DIMS[0] / 2, DIMS[1] / 2, 2));
    const far = showHide();
    engine.jumpToWorldMm(LESION_CENTER_MM);
    const near = showHide();
    add(
      "jump",
      "world mm 이동: 병변 slice에서만 mask 표시",
      far.changed === 0 && near.changed > 0,
      `far=${far.changed} lesion=${near.changed} (lesion k=${LESION.k.join("-")})`,
    );

    // --- 3D brain background and camera presets ---------------------------------------------
    const nvr = engine.niivue;
    if (!nvr) throw new Error("no viewer");
    const camAtLoad = engine.getCameraPreset();
    add(
      "cam_default",
      "3D 기본 카메라 = 정면 (azimuth 180, elevation 0)",
      camAtLoad === "front" && nvr.scene.renderAzimuth === 180 && nvr.scene.renderElevation === 0,
      `preset=${camAtLoad} az=${nvr.scene.renderAzimuth} el=${nvr.scene.renderElevation}`,
    );
    engine.setView("render");
    const bgOpacity3d = nvr.volumes[0]?.opacity;
    engine.setView("axial");
    const bgOpacity2d = nvr.volumes[0]?.opacity;
    engine.setView("render");
    add(
      "bg_opacity",
      `3D 뇌 배경 기본 표시 opacity ${RENDER_BACKGROUND_OPACITY}, 2D는 1.0, mask 기본 ${DEFAULT_MASK_OPACITY}`,
      bgOpacity3d === RENDER_BACKGROUND_OPACITY &&
        bgOpacity2d === 1 &&
        DEFAULT_MASK_OPACITY === 0.5,
      `3D=${bgOpacity3d} 2D=${bgOpacity2d}`,
    );
    engine.setRender3DBackground(false);
    // lesion block: patient left (+24 mm) and anterior (+16 mm) of the volume centre
    const H = canvas.height;
    const at = (preset: keyof typeof CAMERA_PRESETS) => {
      engine.setCameraPreset(preset);
      return showHide(null);
    };
    const front = at("front");
    const top = at("top");
    const left = at("left");
    const right = at("right");
    const fmt = (d: Diff) => `(${d.cx.toFixed(0)},${d.cy.toFixed(0)})`;
    add(
      "cam_front",
      "정면: 환자 좌측 병변이 화면 오른쪽 (화면 왼쪽 = 환자 오른쪽)",
      engine.getCameraPreset() !== null && front.changed > 0 && front.cx > W / 2 + 10,
      `centroid ${fmt(front)} W/2=${W / 2}`,
    );
    add(
      "cam_top",
      "위: 환자 좌측 = 화면 오른쪽, 앞쪽 = 화면 아래",
      top.changed > 0 && top.cx > W / 2 + 10 && top.cy > H / 2 + 5,
      `centroid ${fmt(top)} H/2=${H / 2}`,
    );
    add(
      "cam_left",
      "왼쪽: 앞쪽 병변이 화면 왼쪽",
      left.changed > 0 && left.cx < W / 2 - 5,
      `centroid ${fmt(left)}`,
    );
    add(
      "cam_right",
      "오른쪽: 앞쪽 병변이 화면 오른쪽",
      right.changed > 0 && right.cx > W / 2 + 5,
      `centroid ${fmt(right)}`,
    );
    engine.resetView();
    const back = showHide(null);
    add(
      "cam_reset",
      "기본 시점 복귀 = 정면과 같은 화면 (mask Jaccard 1)",
      engine.getCameraPreset() === "front" && jaccard(back.set, front.set) === 1,
      `preset=${engine.getCameraPreset()} jaccard=${jaccard(back.set, front.set).toFixed(4)}`,
    );
    engine.setCameraPreset("top");
    const topDwi = showHide(null);
    engine.setBase("adc");
    const topAdc = showHide(null);
    const camKept = engine.getCameraPreset();
    engine.setBase("dwi");
    add(
      "cam_baseSwitch",
      "DWI→ADC 전환 후 카메라 유지 + 3D mask Jaccard 1",
      camKept === "top" && jaccard(topDwi.set, topAdc.set) === 1,
      `preset=${camKept} jaccard=${jaccard(topDwi.set, topAdc.set).toFixed(4)}`,
    );
    let camEvents = 0;
    const unsubscribe = engine.onCameraChange(() => camEvents++);
    nvr.setRenderAzimuthElevation(123, 17); // free rotation (what a mouse drag does)
    const free = engine.getCameraPreset();
    unsubscribe();
    nvr.setRenderAzimuthElevation(10, 10);
    add(
      "cam_free",
      "자유 회전 시 preset 해제 + 변경 알림",
      free === null && camEvents === 1,
      `preset=${free} events=${camEvents}`,
    );
    engine.setCameraPreset("front");

    const r0 = showHide();
    add("render_mask", "3D: mask 단독 렌더링", r0.changed > 0, `changed=${r0.changed}`);
    const bgOff = masked(false);
    engine.setRender3DBackground(true);
    const bgOn = masked(false);
    const r1 = showHide();
    const bgDiff = diff(bgOff, bgOn, W, null).changed;
    add(
      "render_bg",
      "3D: DWI 반투명 배경 토글 (배경 표시됨 + mask 유지)",
      bgDiff > 0 && r1.changed > 0,
      `background pixels=${bgDiff} mask changed=${r1.changed}`,
    );
    engine.setRender3DBackground(false);
    engine.setView("axial");

    if (strategy === "customLut") {
      // --- custom palette checks (palette colours set inside the NiivueEngine adapter) ----
      engine.setMaskOpacity(1);
      const def = showHide(hexToRgb(MASK_PALETTE.orange));
      add(
        "c_defaultColor",
        "기본 색 = RGB(230,159,0) 정확히",
        def.changed > 0 && def.exact === def.changed,
        `exact=${def.exact}/${def.changed}`,
      );
      const [a0, a1] = engine.maskLutAlpha();
      add("c_alpha0", "LUT: background(0) alpha = 0", a0 === 0, `alpha0=${a0}`);
      for (const o of [0.5, 0.3]) {
        engine.setMaskOpacity(o);
        const a = masked(false);
        const b = masked(true);
        const share = blendMatch(a, b, def.set, hexToRgb(MASK_PALETTE.orange), o);
        add(
          `c_alpha1_${o}`,
          `mask(1) 표시 alpha = opacity ${o} (LUT alpha1=255 × drawOpacity)`,
          a1 === 255 && share >= 0.99,
          `lutAlpha1=${a1} blend match=${share.toFixed(4)}`,
        );
      }
      engine.setMaskOpacity(1);
      for (const name of MASK_COLOR_NAMES) {
        engine.setMaskColor(name);
        const d = showHide(hexToRgb(MASK_PALETTE[name]));
        const jj = jaccard(def.set, d.set);
        add(
          `c_color_${name}`,
          `팔레트 ${name} ${MASK_PALETTE[name]}: 정확한 색 + 위치 불변`,
          d.changed > 0 && d.exact === d.changed && jj === 1,
          `exact=${d.exact}/${d.changed} jaccard=${jj.toFixed(4)}`,
        );
      }
      const rejected = ["#FF0000", "red", "", "orange;", "__proto__", "constructor"].every((v) => {
        if (isMaskColorName(v)) return false;
        try {
          Reflect.apply(engine.setMaskColor, engine, [v]);
          return false;
        } catch {
          return true;
        }
      });
      add(
        "c_reject",
        "팔레트 밖 색상 값 거부 (런타임 guard)",
        rejected,
        "#FF0000, red, '', __proto__ …",
      );

      engine.setMaskColor("skyBlue");
      const sky = hexToRgb(MASK_PALETTE.skyBlue);
      const dwiSky = showHide(sky);
      engine.setBase("adc");
      const adcSky = showHide(sky);
      const js = jaccard(dwiSky.set, adcSky.set);
      add(
        "c_baseSwitchAfterColor",
        "색 변경 후 DWI↔ADC 전환: Jaccard 1.0, 색 유지",
        js === 1 && adcSky.exact === adcSky.changed && adcSky.changed > 0,
        `jaccard=${js.toFixed(4)} exact=${adcSky.exact}/${adcSky.changed}`,
      );
      engine.setBase("dwi");
      engine.jumpToWorldMm(LESION_CENTER_MM);
      engine.setView("render");
      const r3 = showHide(null);
      add("c_render3d", "색 변경 후 3D mask 유지", r3.changed > 0, `changed=${r3.changed}`);
      engine.setView("axial");
      engine.setMaskColor("orange");
    }
    disposedNv = engine.niivue;
  } catch (e) {
    add("error", "예외", false, e instanceof Error ? e.message : "unknown error");
  } finally {
    engine.dispose();
    const gl = canvas.getContext("webgl2");
    add(
      "dispose",
      "dispose: drawing·volume 해제, WebGL context lost",
      engine.niivue === null && (gl?.isContextLost() ?? true),
      `contextLost=${gl?.isContextLost() ?? "n/a"}`,
    );
    if (strategy === "customLut") {
      add(
        "c_disposeDrawing",
        "dispose 후 drawing bitmap·volume 비움 + engine 해제",
        disposedNv !== null && disposedNv.drawBitmap === null && disposedNv.volumes.length === 0,
        `drawBitmap=${disposedNv?.drawBitmap === null ? "null" : "present"} volumes=${disposedNv?.volumes.length ?? "n/a"}`,
      );
    }
  }
  return out;
}
