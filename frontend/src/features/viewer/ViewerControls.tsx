import { useEffect } from "react";

import { ko } from "../../i18n/ko";
import { useViewerStore } from "./viewerStore";
import {
  CAMERA_PRESET_NAMES,
  DEFAULT_CAMERA_PRESET,
  MASK_COLOR_NAMES,
  MASK_PALETTE,
  type SliceAxis,
  type ViewerEngine,
  type ViewKind,
} from "./ViewerEngine";

const VIEWS: ViewKind[] = ["axial", "coronal", "sagittal", "multiplanar", "render"];
/** One slider per anatomical axis (display grid index = what the engine reports as slice). */
const SLIDERS: { view: "axial" | "coronal" | "sagittal"; axis: SliceAxis }[] = [
  { view: "axial", axis: 2 },
  { view: "coronal", axis: 1 },
  { view: "sagittal", axis: 0 },
];

interface Props {
  engine: ViewerEngine | null;
  sliceCounts: Record<"axial" | "coronal" | "sagittal", number>;
  /** Lesion centre as a file-grid voxel (navigation never uses NiiVue's world-mm approximation). */
  lesionVoxel: [number, number, number] | null;
  labelmapAvailable: boolean;
  emptyMask: boolean;
}

export function ViewerControls({
  engine,
  sliceCounts,
  lesionVoxel,
  labelmapAvailable,
  emptyMask,
}: Props) {
  // s.tick changes whenever the engine moved outside React, so these reads stay current
  const s = useViewerStore();
  const bumpTick = s.bumpTick;
  // mouse rotation moves the 3D camera outside React: re-read the active preset
  useEffect(() => engine?.onCameraChange(bumpTick), [engine, bumpTick]);
  const camera = engine ? engine.getCameraPreset() : DEFAULT_CAMERA_PRESET;
  const win = engine ? engine.getWindow() : null;
  const range = engine ? engine.getWindowRange() : { min: 0, max: 1 };
  const setWin = (next: { min: number; max: number }) => {
    engine?.setWindow(next);
    s.bumpTick();
  };

  const disabled = engine === null;
  return (
    <div className="controls">
      <fieldset>
        <legend>단면</legend>
        <div className="segmented" role="radiogroup" aria-label="단면">
          {VIEWS.map((v) => (
            <button
              key={v}
              type="button"
              role="radio"
              aria-checked={s.view === v}
              className={s.view === v ? "active" : ""}
              onClick={() => s.setView(v)}
            >
              {ko.viewer.views[v]}
            </button>
          ))}
        </div>
        {s.view === "render" && (
          <label className="row" title={ko.viewer.background3dHint}>
            <input
              type="checkbox"
              checked={s.render3DBackground}
              aria-keyshortcuts="B"
              aria-description={ko.viewer.background3dHint}
              onChange={(e) => s.setRender3DBackground(e.target.checked)}
            />
            {ko.viewer.background3d}
          </label>
        )}
        {s.view === "render" && emptyMask && (
          <p className="note" role="status">
            {ko.viewer.emptyMask3d}
          </p>
        )}
      </fieldset>

      {s.view === "render" && (
        <fieldset>
          <legend>{ko.viewer.camera}</legend>
          <div className="segmented" role="radiogroup" aria-label={ko.viewer.camera}>
            {CAMERA_PRESET_NAMES.map((p) => (
              <button
                key={p}
                type="button"
                role="radio"
                aria-checked={camera === p}
                aria-label={ko.viewer.cameraHints[p]}
                title={ko.viewer.cameraHints[p]}
                className={camera === p ? "active" : ""}
                disabled={disabled}
                onClick={() => {
                  engine?.setCameraPreset(p);
                  bumpTick();
                }}
              >
                {ko.viewer.cameraPresets[p]}
              </button>
            ))}
          </div>
          <button
            type="button"
            title={ko.viewer.cameraDefaultHint}
            aria-label={ko.viewer.cameraDefaultHint}
            disabled={disabled}
            onClick={() => {
              engine?.setCameraPreset(DEFAULT_CAMERA_PRESET);
              bumpTick();
            }}
          >
            {ko.viewer.cameraDefault}
          </button>
          <p className="note" aria-live="polite">
            {camera ? ko.viewer.cameraHints[camera] : ko.viewer.cameraFree}
          </p>
          <p className="note">{ko.viewer.orientCube}</p>
        </fieldset>
      )}

      <fieldset>
        <legend>{ko.viewer.base}</legend>
        <div className="segmented" role="radiogroup" aria-label={ko.viewer.base}>
          {(["dwi", "adc"] as const).map((b) => (
            <button
              key={b}
              type="button"
              role="radio"
              aria-checked={s.base === b}
              className={s.base === b ? "active" : ""}
              onClick={() => s.setBase(b)}
            >
              {b.toUpperCase()}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend>{ko.viewer.slice}</legend>
        {/* UI -> engine command only; the values come back through the engine's location event
            (store.crosshair), so canvas clicks, keys and region jumps update all three at once */}
        {SLIDERS.map(({ view, axis }) => {
          const count = Math.max(1, sliceCounts[view]);
          const value = Math.min(count - 1, s.crosshair?.slice[axis] ?? 0);
          return (
            <label className="row" key={view}>
              <span className="slider-name">{ko.viewer.views[view]}</span>
              <input
                type="range"
                aria-label={`${ko.viewer.slice} (${ko.viewer.views[view]})`}
                min={0}
                max={count - 1}
                value={value}
                disabled={disabled || !s.crosshair}
                onChange={(e) => engine?.setSliceVoxel(axis, Number(e.target.value))}
              />
              <span>
                {value + 1} / {count}
              </span>
            </label>
          );
        })}
      </fieldset>

      <fieldset>
        <legend>{ko.viewer.window}</legend>
        {win && (
          <>
            <label className="row">
              {ko.viewer.windowMin}
              <input
                type="range"
                aria-label={`${ko.viewer.window} ${ko.viewer.windowMin}`}
                min={range.min}
                max={range.max}
                step={(range.max - range.min) / 200 || 1}
                value={win.min}
                disabled={disabled}
                onChange={(e) => {
                  const next = { min: Math.min(Number(e.target.value), win.max), max: win.max };
                  setWin(next);
                }}
              />
            </label>
            <label className="row">
              {ko.viewer.windowMax}
              <input
                type="range"
                aria-label={`${ko.viewer.window} ${ko.viewer.windowMax}`}
                min={range.min}
                max={range.max}
                step={(range.max - range.min) / 200 || 1}
                value={win.max}
                disabled={disabled}
                onChange={(e) => {
                  const next = { min: win.min, max: Math.max(Number(e.target.value), win.min) };
                  setWin(next);
                }}
              />
            </label>
          </>
        )}
        <button
          type="button"
          disabled={disabled}
          onClick={() => {
            engine?.resetWindow();
            s.bumpTick();
          }}
        >
          {ko.viewer.resetWindow}
        </button>
      </fieldset>

      <fieldset>
        <legend>{ko.viewer.mask}</legend>
        <label className="row">
          <input
            type="checkbox"
            checked={s.maskVisible}
            onChange={(e) => s.setMaskVisible(e.target.checked)}
          />
          {ko.viewer.maskVisible}
        </label>
        <label className="row">
          {ko.viewer.opacity}
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            aria-label={ko.viewer.opacity}
            value={s.maskOpacity}
            onChange={(e) => s.setMaskOpacity(Number(e.target.value))}
          />
          <span>{s.maskOpacity.toFixed(2)}</span>
        </label>
        {labelmapAvailable && (
          <div className="segmented" role="radiogroup" aria-label={ko.components.coloring}>
            {(["component", "single"] as const).map((c) => (
              <button
                key={c}
                type="button"
                role="radio"
                aria-checked={s.componentColoring === c}
                className={s.componentColoring === c ? "active" : ""}
                onClick={() => s.setComponentColoring(c)}
              >
                {c === "component" ? ko.components.coloringComponent : ko.components.coloringSingle}
              </button>
            ))}
          </div>
        )}
        {labelmapAvailable && <p className="note">{ko.components.paletteNote}</p>}
        <div className="swatches" role="radiogroup" aria-label={ko.viewer.color}>
          {MASK_COLOR_NAMES.map((name) => (
            <button
              key={name}
              type="button"
              role="radio"
              aria-checked={s.maskColor === name}
              aria-label={ko.viewer.colors[name]}
              title={ko.viewer.colors[name]}
              className={s.maskColor === name ? "swatch active" : "swatch"}
              style={{ backgroundColor: MASK_PALETTE[name] }}
              onClick={() => s.setMaskColor(name)}
            />
          ))}
        </div>
        <button
          type="button"
          disabled={disabled || !lesionVoxel}
          onClick={() => lesionVoxel && engine?.jumpToVoxel(lesionVoxel)}
        >
          {ko.viewer.jumpLesion}
        </button>
      </fieldset>

      <fieldset>
        <legend>{ko.viewer.orientation}</legend>
        <p className="note" aria-live="polite">
          {s.radiological ? ko.viewer.radiological : ko.viewer.neurological}
        </p>
        <button type="button" onClick={() => s.setRadiological(!s.radiological)}>
          {s.radiological ? "Neurological로 전환" : "Radiological로 전환"}
        </button>
      </fieldset>

      <fieldset>
        <legend>보기</legend>
        <div className="row">
          <button type="button" disabled={disabled} onClick={() => engine?.zoomBy(1.25)}>
            {ko.viewer.zoomIn}
          </button>
          <button type="button" disabled={disabled} onClick={() => engine?.zoomBy(0.8)}>
            {ko.viewer.zoomOut}
          </button>
          <button type="button" disabled={disabled} onClick={() => engine?.resetView()}>
            {ko.viewer.resetView}
          </button>
        </div>
      </fieldset>
    </div>
  );
}
