// @vitest-environment node
/** Research layout CSS contract (Phase 7-B layout fix). Read straight from disk: under Vitest a
 * CSS import is an empty module, even with ?raw. No Node type package is installed, so the
 * built-in fs module is reached through process.getBuiltinModule via Reflect. */
import { describe, expect, it } from "vitest";

function readText(path: string): string {
  const proc: unknown = Reflect.get(globalThis, "process");
  if (typeof proc !== "object" || proc === null) throw new Error("not running in Node");
  const getBuiltin: unknown = Reflect.get(proc, "getBuiltinModule");
  if (typeof getBuiltin !== "function") throw new Error("process.getBuiltinModule unavailable");
  const fsModule: unknown = Reflect.apply(getBuiltin, proc, ["node:fs"]);
  if (typeof fsModule !== "object" || fsModule === null) throw new Error("node:fs unavailable");
  const readFileSync: unknown = Reflect.get(fsModule, "readFileSync");
  if (typeof readFileSync !== "function") throw new Error("readFileSync unavailable");
  const text: unknown = Reflect.apply(readFileSync, fsModule, [path, "utf8"]);
  if (typeof text !== "string") throw new Error("could not read file");
  return text;
}

const css = readText("src/styles/app.css");

/** Declarations of the first rule whose selector list is exactly `selector`. */
function rule(selector: string): string {
  const norm = (s: string) => s.replace(/\s+/g, " ").trim();
  const re = /([^{}]+)\{([^{}]*)\}/g;
  for (let m = re.exec(css); m; m = re.exec(css)) {
    const sel = norm((m[1] ?? "").replace(/\/\*[\s\S]*?\*\//g, ""));
    if (sel === norm(selector)) return m[2] ?? "";
  }
  throw new Error(`no CSS rule for ${selector}`);
}
const has = (block: string, decl: string) =>
  block.replace(/\s+/g, " ").includes(decl.replace(/\s+/g, " "));

it("reads the stylesheet", () => {
  expect(css.length).toBeGreaterThan(1000);
});

describe("layout CSS contract", () => {
  it("viewer-slot is a definite, clipped box and the host is absolutely positioned in it", () => {
    const slot = rule(".viewer-slot");
    expect(has(slot, "position: relative;")).toBe(true);
    expect(has(slot, "min-height: 320px;")).toBe(true);
    expect(has(slot, "overflow: hidden;")).toBe(true);
    expect(has(slot, "flex: 1 1 auto;")).toBe(true);
    const host = rule(".viewer-slot > .viewer-host");
    for (const d of ["position: absolute;", "inset: 0;", "height: auto;", "min-height: 0;"])
      expect(has(host, d), d).toBe(true);
    expect(has(rule(".viewer-slot .viewer-canvas"), "min-height: 0;")).toBe(true);
  });

  it("research centre and side panels may shrink and scroll inside the viewport", () => {
    const center = rule(".research-layout > .center.research-center");
    for (const d of [
      "min-height: 0;",
      "overflow: hidden;",
      "display: flex;",
      "flex-direction: column;",
    ])
      expect(has(center, d), d).toBe(true);
    const sides = rule(".research-layout > .left, .research-layout > .right");
    for (const d of ["min-height: 0;", "overflow-y: auto;", "position: relative;"])
      expect(has(sides, d), d).toBe(true);
    const grid = rule(".research-layout");
    expect(has(grid, "grid-template-rows: minmax(0, 1fr);")).toBe(true);
    expect(has(grid, "minmax(360px, 28vw)")).toBe(true);
    expect(has(rule(".app:has(.research-page)"), "height: 100dvh;")).toBe(true);
    expect(has(rule(".research-center > .selection-summary"), "overflow-y: auto;")).toBe(true);
    // the banner is in normal flow (never sticky over the panels)
    expect(rule(".research-banner")).not.toMatch(/position:\s*sticky/);
  });

  it("leaves the general viewer rules unchanged", () => {
    const hostCanvas = rule(".viewer-host, .viewer-canvas");
    for (const d of ["width: 100%;", "height: 100%;", "min-height: 600px;"])
      expect(has(hostCanvas, d), d).toBe(true);
    expect(has(rule(".result-layout"), "grid-template-columns: 280px minmax(0, 1fr) 340px;")).toBe(
      true,
    );
    expect(has(rule(".left, .right"), "max-height: calc(100vh - 120px);")).toBe(true);
    expect(has(rule(".center"), "min-height: 600px;")).toBe(true);
  });
});
