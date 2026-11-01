/** Test double: records calls instead of rendering (jsdom has no WebGL). */
import { checkDisplay, type ComponentDisplay } from "./labelLut";
import {
  checkVoxel,
  type CrosshairLocation,
  parseNiivueLocation,
  toCrosshairLocation,
  type Vec3,
} from "./location";
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

/** Matches the test fixtures' image shape (112 x 112 x 73). */
export const FAKE_DIMS = [112, 112, 73] as const;
const IDENTITY = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

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

  // --- Phase 7-0/7-A: identity affine on a FAKE_DIMS grid (file grid = display grid) ---------
  componentDisplay: ComponentDisplay | null = null;
  private locationListeners = new Set<(location: CrosshairLocation) => void>();
  private crosshair: CrosshairLocation = this.at([32, 32, 10]);

  private at(vox: Vec3): CrosshairLocation {
    const frac = vox.map((v, i) => (v + 0.5) / (FAKE_DIMS[i] ?? 1)) as Vec3;
    return toCrosshairLocation({ frac, mm: [...vox] }, IDENTITY, FAKE_DIMS, FAKE_DIMS);
  }
  private move(location: CrosshairLocation) {
    this.crosshair = location;
    for (const listener of this.locationListeners) listener(location);
  }
  setComponentDisplay(display: ComponentDisplay | null) {
    this.componentDisplay = display ? checkDisplay(display) : null;
    this.log("setComponentDisplay", display);
  }
  onLocationChange(listener: (location: CrosshairLocation) => void): () => void {
    this.locationListeners.add(listener);
    return () => this.locationListeners.delete(listener);
  }
  /** Number of subscribed location listeners (unmount cleanup checks). */
  get locationListenerCount(): number {
    return this.locationListeners.size;
  }
  getCrosshair(): CrosshairLocation {
    return this.crosshair;
  }
  jumpToVoxel(vox: [number, number, number]) {
    this.log("jumpToVoxel", vox);
    this.move(this.at(checkVoxel(vox, FAKE_DIMS)));
  }
  setSliceVoxel(axis: SliceAxis, index: number) {
    if (!Number.isInteger(index) || index < 0 || index >= (FAKE_DIMS[axis] ?? 0)) {
      throw new Error("slice index out of range");
    }
    this.log("setSliceVoxel", axis, index);
    const vox: Vec3 = [...this.crosshair.vox];
    vox[axis] = index;
    this.move(this.at(vox));
  }
  /** Simulates NiiVue's locationChange payload (mouse click / drag); goes through the guard. */
  emitRawLocation(raw: unknown): boolean {
    const parsed = parseNiivueLocation(raw);
    if (!parsed) return false;
    this.move(toCrosshairLocation(parsed, IDENTITY, FAKE_DIMS, FAKE_DIMS));
    return true;
  }
  dispose() {
    this.log("dispose");
    this.disposed = true;
  }
}
