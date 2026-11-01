/** Dependency injection for the viewer: production uses NiiVue, tests inject a fake. */
import { createContext, type ReactNode, useContext } from "react";

import { NiivueEngine } from "./NiivueEngine";
import type { ViewerEngine } from "./ViewerEngine";

export interface EngineEnvironment {
  createEngine: () => ViewerEngine;
  isWebGL2Supported: () => boolean;
}

export function browserSupportsWebGL2(): boolean {
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2");
    if (!gl) return false;
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return true;
  } catch {
    return false;
  }
}

const defaultEnvironment: EngineEnvironment = {
  createEngine: () => new NiivueEngine(),
  isWebGL2Supported: browserSupportsWebGL2,
};

const EngineContext = createContext<EngineEnvironment>(defaultEnvironment);

export function EngineProvider(props: { value: EngineEnvironment; children: ReactNode }) {
  return <EngineContext.Provider value={props.value}>{props.children}</EngineContext.Provider>;
}

export function useEngineEnvironment(): EngineEnvironment {
  return useContext(EngineContext);
}
