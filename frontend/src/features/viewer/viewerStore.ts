/** Viewer manipulation state (F3). Server data lives in TanStack Query, not here.
 *
 * Phase 7-A component review state (selection, hidden labels, list order) lives here only — in
 * browser memory, reset on page load; it never changes the prediction files or any result.
 * `crosshair` mirrors the engine (the engine is the source of truth): it is written only by the
 * engine -> store connection (connectCrosshair) and is never pushed back into the engine. */
import { create } from "zustand";

import type { Coloring } from "./labelLut";
import type { CrosshairLocation } from "./location";

import {
  type BaseKind,
  DEFAULT_MASK_COLOR,
  DEFAULT_MASK_OPACITY,
  isMaskColorName,
  type MaskColorName,
  type ViewKind,
} from "./ViewerEngine";

export interface ViewerState {
  view: ViewKind;
  base: BaseKind;
  maskVisible: boolean;
  maskOpacity: number;
  maskColor: MaskColorName;
  radiological: boolean;
  render3DBackground: boolean;
  componentColoring: Coloring;
  selectedComponent: number | null;
  hiddenComponents: ReadonlySet<number>;
  componentSort: ComponentSort;
  componentFilter: ComponentFilter;
  crosshair: CrosshairLocation | null;
  /** Incremented whenever the engine state changed outside React (re-read slice/window). */
  tick: number;
  bumpTick: () => void;
  setView: (view: ViewKind) => void;
  setBase: (base: BaseKind) => void;
  setMaskVisible: (visible: boolean) => void;
  toggleMask: () => void;
  setMaskOpacity: (opacity: number) => void;
  setMaskColor: (color: string) => void;
  setRadiological: (radiological: boolean) => void;
  setRender3DBackground: (visible: boolean) => void;
  toggleRender3DBackground: () => void;
  setComponentColoring: (coloring: Coloring) => void;
  /** Selects a display label (re-selecting keeps it selected; a hidden label is shown again). */
  selectComponent: (label: number) => void;
  clearSelection: () => void;
  /** Hides from the display only; hiding the selected label also clears the selection. */
  hideComponent: (label: number) => void;
  showComponent: (label: number) => void;
  /** Shows only `label` (hides every other label in `all`) — display only. */
  showOnly: (label: number, all: readonly number[]) => void;
  showAllComponents: () => void;
  setComponentSort: (sort: ComponentSort) => void;
  setComponentFilter: (filter: ComponentFilter) => void;
  setCrosshair: (location: CrosshairLocation) => void;
  reset: () => void;
}

export type ComponentSort = "largest" | "smallest";
export type ComponentFilter = "all" | "selected" | "visible";

const initial = {
  view: "axial" as ViewKind,
  base: "dwi" as BaseKind,
  maskVisible: true,
  maskOpacity: DEFAULT_MASK_OPACITY,
  maskColor: DEFAULT_MASK_COLOR,
  radiological: true,
  render3DBackground: true, // 3D brain background on by default (translucent DWI/ADC)
  componentColoring: "component" as Coloring,
  selectedComponent: null as number | null,
  hiddenComponents: new Set<number>() as ReadonlySet<number>,
  componentSort: "largest" as ComponentSort,
  componentFilter: "all" as ComponentFilter,
  crosshair: null as CrosshairLocation | null,
  tick: 0,
};

const without = (set: ReadonlySet<number>, label: number) => {
  const next = new Set(set);
  next.delete(label);
  return next;
};

export const useViewerStore = create<ViewerState>()((set) => ({
  ...initial,
  bumpTick: () => set((s) => ({ tick: s.tick + 1 })),
  setView: (view) => set({ view }),
  setBase: (base) => set({ base }),
  setMaskVisible: (maskVisible) => set({ maskVisible }),
  toggleMask: () => set((s) => ({ maskVisible: !s.maskVisible })),
  setMaskOpacity: (opacity) =>
    set({
      maskOpacity: Number.isFinite(opacity)
        ? Math.min(1, Math.max(0, opacity))
        : DEFAULT_MASK_OPACITY,
    }),
  // only palette names reach the engine
  setMaskColor: (color) => {
    if (isMaskColorName(color)) set({ maskColor: color });
  },
  setRadiological: (radiological) => set({ radiological }),
  setRender3DBackground: (render3DBackground) => set({ render3DBackground }),
  toggleRender3DBackground: () => set((s) => ({ render3DBackground: !s.render3DBackground })),
  setComponentColoring: (componentColoring) => set({ componentColoring }),
  selectComponent: (label) =>
    set((s) => ({
      selectedComponent: label,
      hiddenComponents: without(s.hiddenComponents, label),
    })),
  clearSelection: () => set({ selectedComponent: null }),
  hideComponent: (label) =>
    set((s) => ({
      hiddenComponents: new Set([...s.hiddenComponents, label]),
      selectedComponent: s.selectedComponent === label ? null : s.selectedComponent,
    })),
  showComponent: (label) => set((s) => ({ hiddenComponents: without(s.hiddenComponents, label) })),
  showOnly: (label, all) =>
    set({ selectedComponent: label, hiddenComponents: new Set(all.filter((l) => l !== label)) }),
  showAllComponents: () => set({ hiddenComponents: new Set<number>() }),
  setComponentSort: (componentSort) => set({ componentSort }),
  setComponentFilter: (componentFilter) => set({ componentFilter }),
  setCrosshair: (crosshair) => set({ crosshair }),
  reset: () => set({ ...initial, hiddenComponents: new Set<number>() }),
}));
