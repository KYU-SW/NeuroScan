/**
 * Hosts the viewer engine. The canvas is created inside the effect and removed on cleanup, so a
 * React StrictMode remount never reuses a canvas whose WebGL context was already released.
 * Store changes are pushed to the engine; the engine is disposed on unmount.
 */
import { useEffect, useRef, useState } from "react";

import { ko } from "../../i18n/ko";
import { useEngineEnvironment } from "./engineContext";
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
  useEffect(() => {
    if (!engine) return;
    engine.setMaskColor(state.maskColor);
    engine.setMaskOpacity(state.maskOpacity);
    engine.setMaskVisible(state.maskVisible);
  }, [engine, state.maskColor, state.maskOpacity, state.maskVisible]);
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
