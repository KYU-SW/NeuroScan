/**
 * Viewer engine boundary. React components and the Zustand store depend only on this interface;
 * the NiiVue implementation (NiivueEngine) and the test double (FakeViewerEngine) sit behind it.
 *
 * Coordinates are RAS+ world millimetres, the same space as the API's *_world_mm fields.
 */

export type BaseKind = "dwi" | "adc";
export type ViewKind = "axial" | "coronal" | "sagittal" | "multiplanar" | "render";
/** Anatomical slice axis in RAS order: 0 = sagittal (L-R), 1 = coronal (P-A), 2 = axial (I-S). */
export type SliceAxis = 0 | 1 | 2;

export interface VolumeBuffers {
  /** De-identified DWI copy from the API (gzip NIfTI bytes). */
  dwi: ArrayBuffer;
  /** De-identified ADC copy from the API (gzip NIfTI bytes). */
  adc: ArrayBuffer;
  /** Binary prediction mask (uint8 0/1) on the DWI grid. */
  mask: ArrayBuffer;
}

export interface IntensityWindow {
  min: number;
  max: number;
}

/** Approved mask palette (F12, colour-blind safe Okabe-Ito colours). Nothing else is accepted. */
export const MASK_PALETTE = {
  orange: "#E69F00",
  skyBlue: "#56B4E9",
  bluishGreen: "#009E73",
  reddishPurple: "#CC79A7",
} as const;
export type MaskColorName = keyof typeof MASK_PALETTE;
export const MASK_COLOR_NAMES = Object.keys(MASK_PALETTE) as MaskColorName[];
export const DEFAULT_MASK_COLOR: MaskColorName = "orange";
export const DEFAULT_MASK_OPACITY = 0.5;

/** Opacity of the DWI/ADC brain background in the 3D view (2D slices always use 1). */
export const RENDER_BACKGROUND_OPACITY = 0.4;

/** 3D camera presets (degrees, NiiVue azimuth/elevation; verified by the spike):
 * front = viewed from the face (screen left = patient right), top = viewed from the vertex
 * (screen left = patient right, anterior at the bottom), left/right = viewed from that side. */
export const CAMERA_PRESETS = {
  front: { azimuth: 180, elevation: 0 },
  top: { azimuth: 180, elevation: 90 },
  left: { azimuth: 90, elevation: 0 },
  right: { azimuth: 270, elevation: 0 },
} as const;
export type CameraPreset = keyof typeof CAMERA_PRESETS;
export const CAMERA_PRESET_NAMES = Object.keys(CAMERA_PRESETS) as CameraPreset[];
export const DEFAULT_CAMERA_PRESET: CameraPreset = "front";

export function isMaskColorName(value: string): value is MaskColorName {
  return Object.prototype.hasOwnProperty.call(MASK_PALETTE, value);
}

export interface ViewerEngine {
  attach(canvas: HTMLCanvasElement): Promise<void>;
  load(volumes: VolumeBuffers): Promise<void>;
  setBase(kind: BaseKind): void;
  setView(view: ViewKind): void;
  setRadiological(isRadiological: boolean): void;
  setMaskVisible(visible: boolean): void;
  setMaskOpacity(opacity: number): void;
  /** Only palette names are accepted; anything else throws. */
  setMaskColor(color: MaskColorName): void;
  setWindow(window: IntensityWindow): void;
  /** Current display window of the active base volume. */
  getWindow(): IntensityWindow;
  resetWindow(): IntensityWindow;
  getWindowRange(): IntensityWindow;
  setRender3DBackground(visible: boolean): void;
  setCameraPreset(preset: CameraPreset): void;
  /** Preset matching the current 3D camera, or null after free rotation. */
  getCameraPreset(): CameraPreset | null;
  /** Called whenever the 3D camera moves (preset or mouse rotation); returns an unsubscribe. */
  onCameraChange(listener: () => void): () => void;
  jumpToWorldMm(mm: [number, number, number]): void;
  /** Move the crosshair by whole voxels along one anatomical axis. */
  stepSlice(axis: SliceAxis, delta: number): void;
  /** Crosshair position as fractions (0..1) of the RAS volume extent. */
  getCrosshairFraction(): [number, number, number];
  setSliceFraction(axis: SliceAxis, fraction: number): void;
  zoomBy(factor: number): void;
  /** 2D zoom/pan back to fit and the 3D camera back to the default (front) preset. */
  resetView(): void;
  dispose(): void;
}

export function hexToRgb(hex: string): [number, number, number] {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m || m[1] === undefined || m[2] === undefined || m[3] === undefined) {
    throw new Error("invalid colour");
  }
  return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)];
}
