/** Phase 7-A Connected Component Review: list, selection (voxel jump), grey dimming, hiding,
 * three synchronised sliders, unchanged results and the pre-7-A fallback. FakeViewerEngine. */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { sha256Hex } from "../src/api/client";
import type { ResultComponent, RunResult, ViewerVolume } from "../src/api/schemas";
import {
  displaySliceRange,
  isSmallRegion,
  targetVoxel,
  visibleComponents,
  worldToFileVoxel,
} from "../src/features/components/reviewModel";
import type { FakeViewerEngine } from "../src/features/viewer/FakeViewerEngine";
import { buildComponentLut, componentColor, labelColor } from "../src/features/viewer/labelLut";
import { useViewerStore } from "../src/features/viewer/viewerStore";
import { FORBIDDEN_UI_PHRASES } from "../src/i18n/ko";
import { legacyResult, result, RUN } from "./fixtures";
import { HEALTH_OK, json, mockFetch, renderApp } from "./render";

const bytes = (n: number) => new Uint8Array(64).fill(n).buffer;

async function serve(base: RunResult) {
  const r = structuredClone(base);
  for (const [i, v] of r.viewer.volumes.entries()) v.sha256 = await sha256Hex(bytes(i + 1));
  const byUrl = new Map(r.viewer.volumes.map((v, i) => [v.url, bytes(i + 1)]));
  return mockFetch({
    "GET /api/v1/health": () => json(HEALTH_OK),
    "GET /api/v1/runs/": () => json(r),
    "GET /api/v1/files/": (url) => {
      const b = byUrl.get(url);
      return b ? new Response(b, { status: 200 }) : new Response("", { status: 404 });
    },
  });
}

async function open(base: RunResult = result()) {
  const served = await serve(base);
  const user = userEvent.setup();
  const view = renderApp(`/runs/${RUN}/viewer`);
  await waitFor(() => expect(view.engines[0]?.loaded).toBeTruthy());
  await waitFor(() => expect(screen.queryByText("영상을 불러오는 중")).toBeNull());
  const engine = view.engines[0] as FakeViewerEngine;
  return { ...view, engine, user, calls: served.calls };
}

const rows = () =>
  within(screen.getByRole("list", { name: "연결 영역 검토" })).getAllByRole("listitem");
const lastDisplay = (e: FakeViewerEngine) => e.called("setComponentDisplay").at(-1)?.[0];
const slider = (name: string) => screen.getByRole("slider", { name: `Slice (${name})` });

beforeEach(() => useViewerStore.getState().reset());
afterEach(() => vi.unstubAllGlobals());

describe("component list", () => {
  it("shows number, volume, slice range, side and the neutral small-region tag", async () => {
    await open();
    const items = rows();
    expect(items).toHaveLength(9);
    expect(items[0]).toHaveTextContent("영역 1 · 48.500 mL · slice 29–31 · 환자 우측");
    expect(items[0]).not.toHaveTextContent("작은 분리 영역"); // the largest region
    expect(items[8]).toHaveTextContent("영역 9 · 0.008 mL · slice 37–39 · 추정 불가");
    expect(items[7]).toHaveTextContent("정중선 부근");
    expect(screen.getAllByText("작은 분리 영역")).toHaveLength(8);
    expect(screen.getByText(/검토 편의를 위해 흐리게 표시됩니다\. AI 원본 예측 결과는 변경되지/));
    const text = document.body.textContent ?? "";
    for (const bad of FORBIDDEN_UI_PHRASES) expect(text).not.toContain(bad);
    expect(text).not.toMatch(/\.nii|C:\\|sub-strokecase|prediction_components/);
  });

  it("sorts largest first by default and smallest first on request", async () => {
    const { user } = await open();
    expect(rows()[0]).toHaveTextContent("영역 1 ·");
    await user.selectOptions(screen.getByRole("combobox", { name: "정렬" }), "smallest");
    expect(rows()[0]).toHaveTextContent("영역 9 ·");
    expect(rows()[8]).toHaveTextContent("영역 1 ·");
  });
});

describe("selection, grey dimming and hiding", () => {
  it("selects a region: jump to its representative voxel, own colour, the others grey", async () => {
    const { user, engine } = await open();
    await user.click(screen.getByRole("button", { name: "영역 3 선택 및 위치로 이동" }));
    expect(engine.called("jumpToVoxel").at(-1)).toEqual([[62, 50, 30]]);
    expect(useViewerStore.getState().selectedComponent).toBe(3);
    const display = lastDisplay(engine) as Parameters<typeof buildComponentLut>[0];
    expect(display).toMatchObject({ selected: 3, coloring: "component" });
    const lut = buildComponentLut(display);
    expect([...lut.slice(12, 16)]).toEqual([...componentColor(3), 255]);
    expect([...lut.slice(4, 8)]).toEqual([...labelColor(1, display), 255]); // grey-mixed, opaque
    expect(labelColor(1, display)).not.toEqual(componentColor(1));
    const row = rows()[2];
    expect(row).toHaveAttribute("aria-current", "true");
    expect(
      within(row as HTMLElement).getByRole("button", { name: /선택 및 위치로 이동/ }),
    ).toHaveAttribute("aria-pressed", "true");
    // re-selecting keeps the selection; "선택 해제" restores every colour
    await user.click(screen.getByRole("button", { name: "영역 3 선택 및 위치로 이동" }));
    expect(useViewerStore.getState().selectedComponent).toBe(3);
    await user.click(screen.getByRole("button", { name: "선택 해제" }));
    expect(lastDisplay(engine)).toMatchObject({ selected: null });
  });

  it("hides from the display only, restores everything, and keeps the results", async () => {
    const { user, engine, calls } = await open();
    const maskBefore = new Uint8Array(engine.loaded?.mask ?? new ArrayBuffer(0)).slice();
    await user.click(screen.getByRole("button", { name: "영역 2 표시에서 숨기기" }));
    expect([...(lastDisplay(engine) as { hidden: Set<number> }).hidden]).toEqual([2]);
    expect(rows()[1]).toHaveTextContent("표시에서 숨김");
    await user.selectOptions(screen.getByRole("combobox", { name: "목록 보기" }), "visible");
    expect(rows()).toHaveLength(8);
    await user.selectOptions(screen.getByRole("combobox", { name: "목록 보기" }), "all");
    // hiding the selected region clears the selection
    await user.click(screen.getByRole("button", { name: "영역 4 선택 및 위치로 이동" }));
    await user.click(screen.getByRole("button", { name: "영역 4 표시에서 숨기기" }));
    expect(useViewerStore.getState().selectedComponent).toBeNull();
    await user.click(screen.getByRole("button", { name: "전체 다시 표시" }));
    expect(lastDisplay(engine)).toMatchObject({ selected: null });
    expect((lastDisplay(engine) as { hidden: Set<number> }).hidden.size).toBe(0);
    // nothing about the result changed: same mask bytes, same volumes, only GET requests
    expect(new Uint8Array(engine.loaded?.mask ?? new ArrayBuffer(0))).toEqual(maskBefore);
    expect(screen.getByText(/48\.98 mL/)).toBeInTheDocument();
    expect(calls.every((c) => c.startsWith("GET "))).toBe(true);
  });

  it("'show only the selected region' hides the others; the list filter shows one row", async () => {
    const { user, engine } = await open();
    await user.click(screen.getByRole("button", { name: "영역 5 선택 및 위치로 이동" }));
    await user.click(screen.getByRole("button", { name: "선택 영역만 표시" }));
    const d = lastDisplay(engine) as { selected: number; hidden: Set<number> };
    expect(d.selected).toBe(5);
    expect([...d.hidden].sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 6, 7, 8, 9]);
    await user.selectOptions(screen.getByRole("combobox", { name: "목록 보기" }), "selected");
    expect(rows()).toHaveLength(1);
    expect(rows()[0]).toHaveTextContent("영역 5 ·");
  });

  it("single-colour mode keeps the Phase 6 palette display", async () => {
    const { user, engine } = await open();
    await user.click(screen.getByRole("radio", { name: "단색" }));
    expect(lastDisplay(engine)).toMatchObject({ coloring: "single" });
  });
});

describe("crosshair and the three sliders", () => {
  it("a region jump updates the axial, coronal and sagittal sliders at once", async () => {
    const { user } = await open();
    await user.click(screen.getByRole("button", { name: "영역 1 선택 및 위치로 이동" }));
    await waitFor(() => expect(slider("축상면")).toHaveValue("30"));
    expect(slider("관상면")).toHaveValue("50");
    expect(slider("시상면")).toHaveValue("60");
  });

  it("a slider command goes through the engine and every slider follows", async () => {
    const { engine } = await open();
    fireEvent.change(slider("축상면"), { target: { value: "12" } });
    expect(engine.called("setSliceVoxel").at(-1)).toEqual([2, 12]);
    await waitFor(() => expect(slider("축상면")).toHaveValue("12"));
    expect(slider("관상면")).toHaveValue("55"); // unchanged (lesion centre)
  });

  it("a canvas location event (mouse) updates the sliders; repeats are deduplicated", async () => {
    const { engine } = await open();
    const updates = vi.fn();
    const unsubscribe = useViewerStore.subscribe((s, prev) => {
      if (s.crosshair !== prev.crosshair) updates();
    });
    act(() => {
      for (let n = 0; n < 50; n++)
        engine.emitRawLocation({ frac: [0.2, 0.3, 0.4], mm: [22, 33, 29] });
    });
    await waitFor(() => expect(slider("시상면")).toHaveValue("22"));
    expect(slider("관상면")).toHaveValue("33");
    expect(slider("축상면")).toHaveValue("29");
    expect(updates).toHaveBeenCalledTimes(1);
    unsubscribe();
  });

  it("removes the location listener on unmount", async () => {
    const { engine, unmount } = await open();
    expect(engine.locationListenerCount).toBe(1);
    unmount();
    expect(engine.locationListenerCount).toBe(0);
    expect(engine.disposed).toBe(true);
  });
});

describe("results made before Phase 7-A (no labelmap)", () => {
  it("keeps the Phase 6 display and still jumps through the true affine", async () => {
    const { user, engine, calls } = await open(legacyResult());
    expect(calls.filter((c) => c.includes("/files/"))).toHaveLength(3);
    expect(engine.loaded?.components).toBeUndefined();
    expect(lastDisplay(engine)).toBeNull();
    expect(screen.getByText(/영역 구분 파일이 없어/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /표시에서 숨기기/ })).toBeNull();
    await user.click(screen.getByRole("button", { name: "영역 1 선택 및 위치로 이동" }));
    // centroid (-11, -10, 10) mm through the inverse affine -> voxel (60.5 -> 61, 50, 30)
    expect(engine.called("jumpToVoxel").at(-1)).toEqual([[61, 50, 30]]);
  });
});

describe("review model (pure)", () => {
  const vol = (orientation: string, affine: number[][]): ViewerVolume =>
    ({ ...result().viewer.volumes[0], orientation, affine }) as ViewerVolume;
  const comp = (over: Partial<ResultComponent> = {}): ResultComponent => ({
    ...(result().components[1] as ResultComponent),
    ...over,
  });
  const LAS = result().viewer.volumes[0]?.affine ?? [];

  it("slice ranges follow the axial slider, also for an inferior-pointing slice axis", () => {
    const c = comp({ slice_range: { axis: 2, first: 10, last: 14 } });
    expect(displaySliceRange(c, vol("LAS", LAS))).toEqual({ first: 11, last: 15 });
    expect(displaySliceRange(c, vol("LAI", LAS))).toEqual({ first: 73 - 14, last: 73 - 10 });
    // pre-7-A result: bbox along the S/I axis
    expect(displaySliceRange(comp({ slice_range: null }), vol("LAS", LAS))).toEqual({
      first: 30,
      last: 32,
    });
  });

  it("small-region tag is relative to the largest region and never on the largest", () => {
    expect(isSmallRegion(comp({ size_rank: 1, volume_ml: 0.1 }), 48.5)).toBe(false);
    expect(isSmallRegion(comp({ size_rank: 2, volume_ml: 4.84 }), 48.5)).toBe(true);
    expect(isSmallRegion(comp({ size_rank: 2, volume_ml: 4.86 }), 48.5)).toBe(false);
  });

  it("target voxel: representative voxel, else the true-affine inverse (oblique)", () => {
    expect(targetVoxel(comp(), vol("LAS", LAS))).toEqual([61, 50, 30]);
    const a = (31 * Math.PI) / 180;
    const obl = [
      [-1.8, 0, 0, 101],
      [0, 1.8 * Math.cos(a), -2.5 * Math.sin(a), -88],
      [0, 1.8 * Math.sin(a), 2.5 * Math.cos(a), -40],
      [0, 0, 0, 1],
    ];
    const v = [55, 12, 30];
    const mm = [0, 1, 2].map(
      (r) =>
        (obl[r]?.[0] ?? 0) * 55 +
        (obl[r]?.[1] ?? 0) * 12 +
        (obl[r]?.[2] ?? 0) * 30 +
        (obl[r]?.[3] ?? 0),
    );
    expect(worldToFileVoxel(mm, vol("LAS", obl))).toEqual(v);
    expect(
      targetVoxel(comp({ representative_voxel: null, centroid_world_mm: mm }), vol("LAS", obl)),
    ).toEqual(v);
  });

  it("filters and sorts without touching the input", () => {
    const list = result().components;
    const copy = structuredClone(list);
    expect(
      visibleComponents(list, "smallest", "all", null, new Set()).map((c) => c.size_rank),
    ).toEqual([9, 8, 7, 6, 5, 4, 3, 2, 1]);
    expect(visibleComponents(list, "largest", "visible", null, new Set([1, 2])).length).toBe(7);
    expect(
      visibleComponents(list, "largest", "selected", 4, new Set()).map((c) => c.size_rank),
    ).toEqual([4]);
    expect(list).toEqual(copy);
  });
});
