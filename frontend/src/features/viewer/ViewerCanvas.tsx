/**
 * Hosts the viewer engine. The canvas is created inside the effect and removed on cleanup, so a
 * React StrictMode remount never reuses a canvas whose WebGL context was already released.
 * Store changes are pushed to the engine; the engine is disposed on unmount.
 * Crosshair (Phase 7-A): engine location events flow into the store through connectCrosshair
 * (frame-batched, same-voxel updates dropped); the store never moves the engine's crosshair.
 */
import { useEffect, useRef, useState } from "react";

import { ko } from "../../i18n/ko";
import { useEngineEnvironment } from "./engineContext";
import { connectCrosshair } from "./location";
import { useViewerStore } from "./viewerStore";
import type { ViewerEngine, VolumeBuffers } from "./ViewerEngine";

interface Props {
  buffers: VolumeBuffers;
  description: string;
  onEngine: (engine: ViewerEngine | null) => void;
  onError: (message: string) => void;
}

export function ViewerCanvas({ buffers, description, onEngine, onError }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const { createEngine } = useEngineEnvironment();
  const [engine, setEngine] = useState<ViewerEngine | null>(null);
  const state = useViewerStore();

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let alive = true;
    const canvas = document.createElement("canvas");
    canvas.className = "viewer-canvas";
    canvas.setAttribute("aria-hidden", "true");
    el.appendChild(canvas);
    const eng = createEngine();
    eng
      .attach(canvas)
      .then(() => eng.load(buffers))
      .then(() => {
        if (!alive) return;
        setEngine(eng);
        onEngine(eng);
      })
      .catch(() => {
        if (alive) onError(ko.viewer.loadFailed);
      });
    return () => {
      alive = false;
      onEngine(null);
      setEngine(null);
      eng.dispose();
      canvas.remove();
    };
  }, [buffers, createEngine, onEngine, onError]);

  // One effect per setting so a change only touches what changed; every push bumps the store
  // tick so the controls re-read slice/window from the engine.
  const bump = () => useViewerStore.getState().bumpTick();
  useEffect(() => {
    if (!engine) return;
    engine.setBase(state.base);
    bump();
  }, [engine, state.base]);
  // Phase 7-B: research overlays exist only when the difference labelmap was loaded (research
  // pages); general results never have it, so their engine calls stay exactly as before
  const hasDifference = buffers.difference !== undefined;
  const baseOnly = hasDifference && state.researchMode === "base";
  useEffect(() => {
    if (!engine) return;
    engine.setMaskColor(state.maskColor);
    engine.setMaskOpacity(state.maskOpacity);
    engine.setMaskVisible(state.maskVisible && !baseOnly);
  }, [engine, state.maskColor, state.maskOpacity, state.maskVisible, baseOnly]);
  useEffect(() => {
    if (!engine || !hasDifference) return;
    engine.setResearchOverlay(
      state.researchMode === "difference"
        ? { kind: "difference", visible: state.differenceClasses }
        : state.researchMode === "gt"
          ? { kind: "ground_truth" }
          : null,
    );
  }, [engine, hasDifference, state.researchMode, state.differenceClasses]);
  const hasLabelmap = buffers.components !== undefined;
  useEffect(() => {
    if (!engine) return;
    // component colours / selection / hiding need the labelmap; without it (results made
    // before Phase 7-A) the mask keeps the Phase 6 single-colour display
    engine.setComponentDisplay(
      hasLabelmap
        ? {
            selected: state.selectedComponent,
            hidden: state.hiddenComponents,
            coloring: state.componentColoring,
          }
        : null,
    );
  }, [
    engine,
    hasLabelmap,
    state.selectedComponent,
    state.hiddenComponents,
    state.componentColoring,
  ]);
  useEffect(() => {
    if (!engine) return;
    const setCrosshair = useViewerStore.getState().setCrosshair;
    setCrosshair(engine.getCrosshair());
    return connectCrosshair(engine, setCrosshair); // unsubscribes on unmount / engine change
  }, [engine]);
  useEffect(() => {
    if (!engine) return;
    engine.setRadiological(state.radiological);
  }, [engine, state.radiological]);
  useEffect(() => {
    if (!engine) return;
    engine.setRender3DBackground(state.render3DBackground);
    engine.setView(state.view);
    bump();
  }, [engine, state.render3DBackground, state.view]);

  return (
    <div className="viewer-host" ref={host} role="img" aria-label={description}>
      {!engine && <p className="viewer-status">{ko.viewer.loading}</p>}
    </div>
  );
}
