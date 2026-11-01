/** Research Viewer layout contract (Phase 7-B layout fix).
 *
 * The canvas host lives in .viewer-slot, a sibling *after* the mode band, legend and selection
 * summary, and the slot's CSS makes the host absolutely positioned — so the canvas can never size
 * the layout (the regression was a ResizeObserver feedback loop that grew the canvas forever).
 * jsdom has no layout engine: real sizes are checked with tools/research-layout-check.mjs;
 * the CSS contract is in researchLayoutCss.test.ts. */
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { sha256Hex } from "../src/api/client";
import type { ResearchCase, RunResult } from "../src/api/schemas";
import type { FakeViewerEngine } from "../src/features/viewer/FakeViewerEngine";
import { useViewerStore } from "../src/features/viewer/viewerStore";
import { result, RUN } from "./fixtures";
import { HEALTH_OK, json, mockFetch, renderApp } from "./render";
import { researchCase } from "./researchFixtures";

const bytes = (n: number) => new Uint8Array(64).fill(n).buffer;

async function serve(detail: ResearchCase | null, general: RunResult | null = null) {
  const r = detail ? structuredClone(detail) : null;
  const g = general ? structuredClone(general) : null;
  if (r) for (const [i, v] of r.volumes.entries()) v.sha256 = await sha256Hex(bytes(i + 1));
  if (g) for (const [i, v] of g.viewer.volumes.entries()) v.sha256 = await sha256Hex(bytes(i + 1));
  const urls = [...(r?.volumes ?? []), ...(g?.viewer.volumes ?? [])];
  const byUrl = new Map(urls.map((v, i) => [v.url, bytes((i % (r?.volumes.length ?? 99)) + 1)]));
  return mockFetch({
    "GET /api/v1/health": () => json(HEALTH_OK),
    "GET /api/v1/research/cases/": () => json(r),
    "GET /api/v1/runs/": () => json(g),
    "GET /api/v1/files/": (url) => new Response(byUrl.get(url) ?? "", { status: 200 }),
  });
}

async function openResearch() {
  await serve(researchCase());
  const user = userEvent.setup();
  const view = renderApp("/research/cases/val-001");
  await waitFor(() => expect(view.engines[0]?.loaded).toBeTruthy());
  await waitFor(() => expect(document.querySelector("canvas.viewer-canvas")).not.toBeNull());
  return { ...view, user, engine: view.engines[0] as FakeViewerEngine };
}

beforeEach(() => useViewerStore.getState().reset());
afterEach(() => vi.unstubAllGlobals());

describe("research viewer DOM structure", () => {
  it("puts the canvas in viewer-slot, after the band, legend and summary (siblings)", async () => {
    const { user } = await openResearch();
    const center = document.querySelector("section.center.research-center");
    const slot = document.querySelector(".viewer-slot");
    expect(center).not.toBeNull();
    expect(slot?.parentElement).toBe(center);
    expect(slot?.querySelector(".viewer-host canvas.viewer-canvas")).not.toBeNull();
    expect(center?.lastElementChild).toBe(slot);
    await user.click(screen.getByRole("button", { name: "영역 2 선택 및 위치로 이동" }));
    for (const sel of [".mode-band", ".legend", ".selection-summary"]) {
      const el = document.querySelector(sel);
      expect(el?.parentElement, sel).toBe(center);
      expect(slot?.contains(el ?? null), sel).toBe(false);
    }
    expect(document.querySelector(".research-page .research-layout")).not.toBeNull();
    expect(document.querySelector(".result-layout")).toBeNull();
  });

  it("keeps one mounted canvas through selection, summary and mode changes", async () => {
    const { user, engines } = await openResearch();
    const canvas = document.querySelector("canvas.viewer-canvas");
    await user.click(screen.getByRole("button", { name: "영역 2 선택 및 위치로 이동" }));
    for (const m of ["Ground Truth", "AI 예측", "기본 영상만", "차이 지도 (TP/FP/FN)"])
      await user.click(screen.getByRole("radio", { name: m }));
    await user.click(
      within(screen.getByRole("region", { name: "선택 요약" })).getByRole("button", {
        name: "선택 해제",
      }),
    );
    expect(engines).toHaveLength(1);
    expect(engines[0]?.disposed).toBe(false);
    expect(document.querySelector("canvas.viewer-canvas")).toBe(canvas);
  });

  it("does not apply the research layout to the general result viewer", async () => {
    await serve(null, result());
    const view = renderApp(`/runs/${RUN}/viewer`);
    await waitFor(() => expect(view.engines[0]?.loaded).toBeTruthy());
    await waitFor(() => expect(document.querySelector("canvas.viewer-canvas")).not.toBeNull());
    expect(document.querySelector(".result-layout")).not.toBeNull();
    for (const sel of [".viewer-slot", ".research-layout", ".research-page", ".research-center"])
      expect(document.querySelector(sel), sel).toBeNull();
    // general viewer: the canvas host is still a direct child of the centre section
    expect(document.querySelector("section.center > .viewer-host")).not.toBeNull();
  });
});
