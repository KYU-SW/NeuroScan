// @vitest-environment node
/** Phase 7-0 #16: the dev-only spike (Phase 6 and 7-0) never reaches the production bundle.
 * Builds the app in memory with the project's Vite config and scans every emitted file. */
import { build, type Rollup } from "vite";
import { afterEach, describe, expect, it, vi } from "vitest";

const SPIKE_MARKERS = [
  "__spike",
  "spike-host",
  "NiiVue spike",
  "p7:",
  "Phase 7-0 수동 확인",
  "drawing-layer spike",
  "oblique(",
];

async function bundleText(nodeEnv: string): Promise<{ files: number; text: string }> {
  vi.stubEnv("NODE_ENV", nodeEnv);
  const result = await build({
    configFile: "vite.config.ts",
    mode: nodeEnv,
    logLevel: "silent",
    build: { write: false, emptyOutDir: false },
  });
  const files = outputs(result);
  const text = files
    .map((f) =>
      f.type === "chunk"
        ? f.code
        : typeof f.source === "string"
          ? f.source
          : new TextDecoder().decode(f.source),
    )
    .join(String.fromCharCode(10));
  return { files: files.length, text };
}

function outputs(result: Awaited<ReturnType<typeof build>>): Rollup.OutputBundle[number][] {
  const list = Array.isArray(result) ? result : [result];
  return list.flatMap((r) => ("output" in r ? r.output : []));
}

describe("production bundle", () => {
  afterEach(() => vi.unstubAllEnvs());

  it("contains no spike code or spike route", async () => {
    // Vitest runs with NODE_ENV=test, which Vite keeps; build exactly like `npm run build`.
    const { files, text } = await bundleText("production");
    expect(files).toBeGreaterThan(1);
    // positive controls: the app itself is there and it is the production React build
    expect(text.includes("DWI · ADC 영상 업로드")).toBe(true);
    expect(text.includes("is deprecated in plain JavaScript React classes")).toBe(false);
    expect(SPIKE_MARKERS.filter((marker) => text.includes(marker))).toEqual([]);
  }, 120_000);

  it("negative control: a development build does contain the spike (the scan can see it)", async () => {
    const { text } = await bundleText("development");
    expect(["spike-host", "Phase 7-0 수동 확인", "p7:"].every((m) => text.includes(m))).toBe(true);
  }, 120_000);
});
