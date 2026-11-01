/** Test double: records calls instead of rendering (jsdom has no WebGL). */
import {
  type BaseKind,
  CAMERA_PRESETS,
  type CameraPreset,
  DEFAULT_CAMERA_PRESET,
  type IntensityWindow,
  isMaskColorName,
  type MaskColorName,
  type SliceAxis,
  type ViewerEngine,
  type ViewKind,
  type VolumeBuffers,
} from "./ViewerEngine";

export class FakeViewerEngine implements ViewerEngine {
  calls: Array<[string, ...unknown[]]> = [];
  attached = false;
  loaded: VolumeBuffers | null = null;
  disposed = false;
  fraction: [number, number, number] = [0.5, 0.5, 0.5];
  window: IntensityWindow = { min: 0, max: 100 };
  camera: CameraPreset | null = DEFAULT_CAMERA_PRESET;
  private cameraListeners = new Set<() => void>();

  private log(name: string, ...args: unknown[]) {
    this.calls.push([name, ...args]);
  }

  called(name: string): unknown[][] {
    return this.calls.filter((c) => c[0] === name).map((c) => c.slice(1));
  }

  async attach(canvas: HTMLCanvasElement) {
    this.log("attach", canvas.tagName);
    this.attached = true;
  }
  async load(volumes: VolumeBuffers) {
    this.log("load");
    this.loaded = volumes;
  }
  setBase(kind: BaseKind) {
    this.log("setBase", kind);
  }
  setView(view: ViewKind) {
    this.log("setView", view);
  }
  setRadiological(isRadiological: boolean) {
    this.log("setRadiological", isRadiological);
  }
  setMaskVisible(visible: boolean) {
    this.log("setMaskVisible", visible);
  }
  setMaskOpacity(opacity: number) {
    this.log("setMaskOpacity", opacity);
  }
  setMaskColor(color: MaskColorName) {
    if (!isMaskColorName(color)) throw new Error("colour not in the approved palette");
    this.log("setMaskColor", color);
  }
  setWindow(window: IntensityWindow) {
    this.log("setWindow", window);
    this.window = window;
  }
  getWindow(): IntensityWindow {
    return this.window;
  }
  resetWindow(): IntensityWindow {
    this.log("resetWindow");
    this.window = { min: 0, max: 100 };
    return this.window;
  }
  getWindowRange(): IntensityWindow {
    return { min: 0, max: 400 };
  }
  setRender3DBackground(visible: boolean) {
    this.log("setRender3DBackground", visible);
  }
  setCameraPreset(preset: CameraPreset) {
    if (!Object.prototype.hasOwnProperty.call(CAMERA_PRESETS, preset)) throw new Error("preset");
    this.log("setCameraPreset", preset);
    this.camera = preset;
    for (const listener of this.cameraListeners) listener();
  }
  getCameraPreset(): CameraPreset | null {
    return this.camera;
  }
  onCameraChange(listener: () => void): () => void {
    this.cameraListeners.add(listener);
    return () => this.cameraListeners.delete(listener);
  }
  /** Simulates a free mouse rotation (no preset matches afterwards). */
  rotateFreely() {
    this.camera = null;
    for (const listener of this.cameraListeners) listener();
  }
  jumpToWorldMm(mm: [number, number, number]) {
    this.log("jumpToWorldMm", mm);
  }
  stepSlice(axis: SliceAxis, delta: number) {
    this.log("stepSlice", axis, delta);
  }
  getCrosshairFraction(): [number, number, number] {
    return this.fraction;
  }
  setSliceFraction(axis: SliceAxis, fraction: number) {
    this.log("setSliceFraction", axis, fraction);
    this.fraction[axis] = fraction;
  }
  zoomBy(factor: number) {
    this.log("zoomBy", factor);
  }
  resetView() {
    this.log("resetView");
    this.setCameraPreset(DEFAULT_CAMERA_PRESET);
  }
  dispose() {
    this.log("dispose");
    this.disposed = true;
  }
}
