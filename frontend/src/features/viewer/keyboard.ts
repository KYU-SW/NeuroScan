import type { SliceAxis, ViewerEngine, ViewKind } from "./ViewerEngine";
import type { ViewerState } from "./viewerStore";

const VIEW_KEYS: Record<string, ViewKind> = {
  "1": "axial",
  "2": "coronal",
  "3": "sagittal",
  "4": "multiplanar",
  "5": "render",
};

export function sliceAxisOf(view: ViewKind): SliceAxis {
  return view === "sagittal" ? 0 : view === "coronal" ? 1 : 2;
}

/** Returns true when the key was handled (caller prevents the default scroll). */
export function handleViewerKey(
  key: string,
  state: ViewerState,
  engine: ViewerEngine | null,
  lesionCenter: [number, number, number] | null,
): boolean {
  const view = VIEW_KEYS[key];
  if (view) {
    state.setView(view);
    return true;
  }
  switch (key) {
    case "m":
    case "M":
      state.toggleMask();
      return true;
    case "[":
      state.setMaskOpacity(Math.round((state.maskOpacity - 0.1) * 10) / 10);
      return true;
    case "]":
      state.setMaskOpacity(Math.round((state.maskOpacity + 0.1) * 10) / 10);
      return true;
    case "d":
    case "D":
      state.setBase("dwi");
      return true;
    case "a":
    case "A":
      state.setBase("adc");
      return true;
    case "b":
    case "B":
      state.toggleRender3DBackground();
      return true;
  }
  if (!engine) return false;
  const handled = engineKey(key, state, engine, lesionCenter);
  if (handled) state.bumpTick();
  return handled;
}

function engineKey(
  key: string,
  state: ViewerState,
  engine: ViewerEngine,
  lesionCenter: [number, number, number] | null,
): boolean {
  switch (key) {
    case "ArrowUp":
    case "PageUp":
      engine.stepSlice(sliceAxisOf(state.view), 1);
      return true;
    case "ArrowDown":
    case "PageDown":
      engine.stepSlice(sliceAxisOf(state.view), -1);
      return true;
    case "+":
    case "=":
      engine.zoomBy(1.25);
      return true;
    case "-":
      engine.zoomBy(0.8);
      return true;
    case "0":
      engine.resetView();
      return true;
    case "l":
    case "L":
      if (lesionCenter) engine.jumpToWorldMm(lesionCenter);
      return lesionCenter !== null;
  }
  return false;
}
