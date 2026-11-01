/**
 * NiiVue implementation of ViewerEngine — the only module that imports @niivue/niivue.
 * Public, typed API only: no @internal members, no private fields, no casts.
 *
 * Interpolation (F7): DWI/ADC are ordinary volumes under NiiVue's global linear filtering
 * (setInterpolation(false)). The prediction mask is shown through the drawing layer
 * (loadDrawing / setDrawOpacity / setDrawingEnabled / closeDrawing), whose R8 texture is always
 * sampled with NEAREST filtering, so its 0/1 voxel edges never blur. Editing stays disabled.
 *
 * Mask colour (F12): the approved palette colour is written to the public `drawLut` field via the
 * public `cmapper.makeDrawLut(ColorMap)` (label 0 alpha 0 = transparent, label 1 = colour).
 * 3D: the base volume is a translucent brain background (opacity RENDER_BACKGROUND_OPACITY, on by
 * default) and the camera starts at the front preset; presets use setRenderAzimuthElevation only.
 *
 * Re-run the drawing-layer spike (/__spike) whenever the NiiVue version changes.
 */
import { cmapper, Niivue, NVImage, SLICE_TYPE } from "@niivue/niivue";

import {
  type BaseKind,
  CAMERA_PRESET_NAMES,
  CAMERA_PRESETS,
  type CameraPreset,
  DEFAULT_CAMERA_PRESET,
  DEFAULT_MASK_COLOR,
  DEFAULT_MASK_OPACITY,
  hexToRgb,
  type IntensityWindow,
  isMaskColorName,
  MASK_PALETTE,
  type MaskColorName,
  RENDER_BACKGROUND_OPACITY,
  type SliceAxis,
  type ViewerEngine,
  type ViewKind,
  type VolumeBuffers,
} from "./ViewerEngine";

/** "customLut" (production): palette colour via drawLut. "builtin": setDrawColormap("$itksnap")
 * (label 1 = red) — kept only so the spike can compare against the approved-list baseline. */
export type MaskColorStrategy = "builtin" | "customLut";

const SLICE: Record<ViewKind, SLICE_TYPE> = {
  axial: SLICE_TYPE.AXIAL,
  coronal: SLICE_TYPE.CORONAL,
  sagittal: SLICE_TYPE.SAGITTAL,
  multiplanar: SLICE_TYPE.MULTIPLANAR,
  render: SLICE_TYPE.RENDER,
};

const NAMES: Record<BaseKind | "mask", string> = {
  dwi: "dwi.nii.gz",
  adc: "adc.nii.gz",
  mask: "prediction_mask.nii.gz",
};

type DrawColorMap = Parameters<typeof cmapper.makeDrawLut>[0];

/** Display window NiiVue computed on load (robust range); 0..1 if it reports none. */
function windowOf(img: NVImage): IntensityWindow {
  const min = img.cal_min ?? 0;
  const max = img.cal_max ?? 1;
  return { min, max: max > min ? max : min + 1 };
}

export class NiivueEngine implements ViewerEngine {
  private nv: Niivue | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private images: Partial<Record<BaseKind, NVImage>> = {};
  private defaultWindows: Partial<Record<BaseKind, IntensityWindow>> = {};
  private mask: NVImage | null = null;
  private base: BaseKind | null = null;
  private view: ViewKind = "axial";
  private maskVisible = true;
  private maskOpacity = DEFAULT_MASK_OPACITY;
  private maskColor: MaskColorName = DEFAULT_MASK_COLOR;
  private render3DBackground = true;
  private cameraListeners = new Set<() => void>();

  constructor(private readonly colorStrategy: MaskColorStrategy = "customLut") {}

  /** Diagnostics for the dev-only spike page; React components never use it. */
  get niivue(): Niivue | null {
    return this.nv;
  }

  /** Diagnostics: alpha of draw-LUT entries 0 (background) and 1 (lesion). */
  maskLutAlpha(): [number, number] {
    const lut = this.requireNv().drawLut.lut;
    return [lut[3] ?? -1, lut[7] ?? -1];
  }

  async attach(canvas: HTMLCanvasElement): Promise<void> {
    this.canvas = canvas;
    this.nv = new Niivue({
      isRadiologicalConvention: true,
      isOrientationTextVisible: true,
      showAllOrientationMarkers: true,
      isNearestInterpolation: false,
      drawingEnabled: false,
      isColorbar: false,
      show3Dcrosshair: false,
      isOrientCube: true, // 3D: L/R/A/P/S/I orientation cube
      backColor: [0, 0, 0, 1],
      logLevel: "warn",
      loadingText: "",
    });
    await this.nv.attachToCanvas(canvas);
    this.nv.setInterpolation(false); // DWI/ADC linear (the drawing layer is NEAREST regardless)
    this.nv.onAzimuthElevationChange = () => {
      for (const listener of this.cameraListeners) listener();
    };
    this.setCameraPreset(DEFAULT_CAMERA_PRESET);
  }

  async load(volumes: VolumeBuffers): Promise<void> {
    const nv = this.requireNv();
    const [dwi, adc, mask] = await Promise.all([
      NVImage.loadFromUrl({ url: volumes.dwi, name: NAMES.dwi, colormap: "gray" }),
      NVImage.loadFromUrl({ url: volumes.adc, name: NAMES.adc, colormap: "gray" }),
      NVImage.loadFromUrl({ url: volumes.mask, name: NAMES.mask, colormap: "gray" }),
    ]);
    if (this.nv !== nv) return; // disposed while loading
    this.images = { dwi, adc };
    this.defaultWindows = { dwi: windowOf(dwi), adc: windowOf(adc) };
    this.mask = mask;
    this.setBase("dwi");
  }

  setBase(kind: BaseKind): void {
    const nv = this.requireNv();
    const next = this.images[kind];
    if (!next) throw new Error("volume not loaded");
    if (this.base === kind && nv.volumes[0] === next) return; // already displayed
    for (const v of [...nv.volumes]) nv.removeVolume(v);
    nv.addVolume(next);
    this.base = kind;
    nv.setInterpolation(false);
    if (this.mask) {
      nv.setDrawingEnabled(false);
      if (!nv.loadDrawing(this.mask)) throw new Error("mask grid does not match the base volume");
      this.applyMaskStyle();
    }
    this.applyView();
  }

  setView(view: ViewKind): void {
    this.view = view;
    this.applyView();
  }

  setRadiological(isRadiological: boolean): void {
    this.requireNv().setRadiologicalConvention(isRadiological);
  }

  setMaskVisible(visible: boolean): void {
    this.maskVisible = visible;
    this.applyMaskStyle();
  }

  setMaskOpacity(opacity: number): void {
    if (!Number.isFinite(opacity)) throw new Error("invalid opacity");
    this.maskOpacity = Math.min(1, Math.max(0, opacity));
    this.applyMaskStyle();
  }

  setMaskColor(color: MaskColorName): void {
    if (!isMaskColorName(color)) throw new Error("colour not in the approved palette");
    this.maskColor = color;
    this.applyMaskStyle();
  }

  setWindow(window: IntensityWindow): void {
    const img = this.currentImage();
    img.cal_min = window.min;
    img.cal_max = window.max;
    this.requireNv().updateGLVolume();
  }

  getWindow(): IntensityWindow {
    return windowOf(this.currentImage());
  }

  resetWindow(): IntensityWindow {
    const w = this.base ? this.defaultWindows[this.base] : undefined;
    if (!w) throw new Error("volume not loaded");
    this.setWindow(w);
    return w;
  }

  getWindowRange(): IntensityWindow {
    const img = this.currentImage();
    const w = windowOf(img);
    return { min: img.global_min ?? w.min, max: img.global_max ?? w.max };
  }

  setRender3DBackground(visible: boolean): void {
    this.render3DBackground = visible;
    this.applyView();
  }

  jumpToWorldMm(mm: [number, number, number]): void {
    const nv = this.requireNv();
    nv.scene.crosshairPos = this.currentImage().convertMM2Frac(mm);
    nv.drawScene();
  }

  stepSlice(axis: SliceAxis, delta: number): void {
    const step = Math.trunc(delta);
    if (step === 0) return;
    this.requireNv().moveCrosshairInVox(
      axis === 0 ? step : 0,
      axis === 1 ? step : 0,
      axis === 2 ? step : 0,
    );
  }

  getCrosshairFraction(): [number, number, number] {
    const p = this.requireNv().scene.crosshairPos;
    return [p[0] ?? 0.5, p[1] ?? 0.5, p[2] ?? 0.5];
  }

  setSliceFraction(axis: SliceAxis, fraction: number): void {
    const nv = this.requireNv();
    const pos = this.getCrosshairFraction();
    pos[axis] = Math.min(1, Math.max(0, fraction));
    nv.scene.crosshairPos = this.currentImage().convertMM2Frac(
      this.currentImage().convertFrac2MM(pos),
    );
    nv.drawScene();
  }

  zoomBy(factor: number): void {
    const nv = this.requireNv();
    const [x, y, z, zoom] = nv.scene.pan2Dxyzmm;
    const next = Math.min(8, Math.max(0.5, (zoom ?? 1) * factor));
    nv.setPan2Dxyzmm([x ?? 0, y ?? 0, z ?? 0, next]);
  }

  setCameraPreset(preset: CameraPreset): void {
    const { azimuth, elevation } = CAMERA_PRESETS[preset];
    this.requireNv().setRenderAzimuthElevation(azimuth, elevation);
  }

  getCameraPreset(): CameraPreset | null {
    const { renderAzimuth, renderElevation } = this.requireNv().scene;
    const azimuth = ((renderAzimuth % 360) + 360) % 360;
    return (
      CAMERA_PRESET_NAMES.find(
        (p) =>
          Math.abs(CAMERA_PRESETS[p].azimuth - azimuth) < 0.5 &&
          Math.abs(CAMERA_PRESETS[p].elevation - renderElevation) < 0.5,
      ) ?? null
    );
  }

  onCameraChange(listener: () => void): () => void {
    this.cameraListeners.add(listener);
    return () => this.cameraListeners.delete(listener);
  }

  resetView(): void {
    const nv = this.requireNv();
    this.setCameraPreset(DEFAULT_CAMERA_PRESET);
    nv.setPan2Dxyzmm([0, 0, 0, 1]);
  }

  dispose(): void {
    const nv = this.nv;
    this.nv = null;
    this.cameraListeners.clear();
    if (nv) {
      nv.onAzimuthElevationChange = () => {};
      nv.closeDrawing();
      for (const v of [...nv.volumes]) nv.removeVolume(v);
      nv.cleanup();
    }
    const gl = this.canvas?.getContext("webgl2");
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
    this.images = {};
    this.mask = null;
    this.canvas = null;
    this.base = null;
  }

  private applyView(): void {
    const nv = this.requireNv();
    if (!this.base) return;
    // 3D: the DWI/ADC brain background is a translucent layer (on by default, toggle B).
    const backgroundOpacity =
      this.view !== "render" ? 1 : this.render3DBackground ? RENDER_BACKGROUND_OPACITY : 0;
    nv.setOpacity(0, backgroundOpacity);
    nv.setSliceType(SLICE[this.view]);
  }

  private applyMaskStyle(): void {
    const nv = this.requireNv();
    if (this.colorStrategy === "customLut") {
      const [r, g, b] = hexToRgb(MASK_PALETTE[this.maskColor]);
      const cmap: DrawColorMap = { R: [0, r], G: [0, g], B: [0, b], A: [0, 255], I: [0, 1] };
      nv.drawLut = cmapper.makeDrawLut(cmap);
    } else {
      nv.setDrawColormap("$itksnap");
    }
    nv.setDrawOpacity(this.maskVisible ? this.maskOpacity : 0);
    nv.updateGLVolume();
  }

  private currentImage(): NVImage {
    const img = this.base ? this.images[this.base] : undefined;
    if (!img) throw new Error("volume not loaded");
    return img;
  }

  private requireNv(): Niivue {
    if (!this.nv) throw new Error("viewer not attached");
    return this.nv;
  }
}
