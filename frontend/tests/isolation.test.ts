/** NiiVue stays behind the adapter: only NiivueEngine.ts imports @niivue/niivue or touches the
 * custom draw LUT (drawLut / cmapper); no source file logs or uses forbidden constructs.
 * Sources are read with Vite's import.meta.glob (raw), so no Node type definitions are needed. */
import { describe, expect, it } from "vitest";

const raw = import.meta.glob<string>("../src/**/*.{ts,tsx}", {
  query: "?raw",
  import: "default",
  eager: true,
});
const sources = Object.entries(raw).map(([key, text]) => ({
  path: key.replace(/^\.\.\/src\//, ""),
  text,
}));
const ADAPTER = "features/viewer/NiivueEngine.ts";

describe("viewer engine isolation", () => {
  it("reads the source tree", () => {
    expect(sources.length).toBeGreaterThan(20);
    expect(sources.some((s) => s.path === ADAPTER)).toBe(true);
  });

  it("only the adapter imports NiiVue or uses drawLut / cmapper", () => {
    const offenders = sources
      .filter((s) => s.path !== ADAPTER)
      .filter((s) => /@niivue\/niivue|\bdrawLut\b|\bcmapper\b/.test(s.text))
      .map((s) => s.path);
    expect(offenders).toEqual([]);
    expect(sources.find((s) => s.path === ADAPTER)?.text).toMatch(/drawLut/);
  });

  it("no private NiiVue API, type escapes or console logging anywhere in src", () => {
    for (const s of sources) {
      expect(s.text, s.path).not.toMatch(
        /updateInterpolation|\bas any\b|@ts-ignore|@ts-expect-error|console\.(log|info|debug)/,
      );
      expect(s.text, s.path).not.toMatch(/\.prototype\.\w+\s*=/);
    }
  });

  it("React components and the store reach the viewer only through ViewerEngine", () => {
    const ui = sources.filter((s) => s.path.endsWith(".tsx") || s.path.endsWith("viewerStore.ts"));
    for (const s of ui) {
      // engineContext constructs the default engine; the dev-only spike diagnoses the adapter itself
      if (s.path === "features/viewer/engineContext.tsx" || s.path.startsWith("spike/")) continue;
      expect(s.text, s.path).not.toMatch(/NiivueEngine/);
    }
  });
});
