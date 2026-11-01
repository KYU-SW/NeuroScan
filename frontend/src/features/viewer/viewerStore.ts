/** Viewer manipulation state (F3). Server data lives in TanStack Query, not here. */
import { create } from "zustand";

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
  reset: () => void;
}

const initial = {
  view: "axial" as ViewKind,
  base: "dwi" as BaseKind,
  maskVisible: true,
  maskOpacity: DEFAULT_MASK_OPACITY,
  maskColor: DEFAULT_MASK_COLOR,
  radiological: true,
  render3DBackground: true, // 3D brain background on by default (translucent DWI/ADC)
  tick: 0,
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
  reset: () => set(initial),
}));
