import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { sha256Hex } from "../src/api/client";
import type { RunResult } from "../src/api/schemas";
import type { FakeViewerEngine } from "../src/features/viewer/FakeViewerEngine";
import { handleViewerKey } from "../src/features/viewer/keyboard";
import {
  CAMERA_PRESETS,
  DEFAULT_CAMERA_PRESET,
  DEFAULT_MASK_OPACITY,
  RENDER_BACKGROUND_OPACITY,
} from "../src/features/viewer/ViewerEngine";
import { useViewerStore } from "../src/features/viewer/viewerStore";
import { FORBIDDEN_UI_PHRASES, ko } from "../src/i18n/ko";
import { manyComponentsResult, noLesionResult, result, RUN } from "./fixtures";
import { HEALTH_OK, json, mockFetch, renderApp } from "./render";

const bytes = (n: number) => new Uint8Array(64).fill(n).buffer;

/** Serve the result plus three volume files whose SHA-256 matches the result. */
async function serve(base: RunResult, opts: { tamper?: boolean } = {}) {
  const r = structuredClone(base);
  for (const [i, v] of r.viewer.volumes.entries()) v.sha256 = await sha256Hex(bytes(i + 1));
  const byUrl = new Map(
    r.viewer.volumes.map((v, i) => [v.url, bytes(opts.tamper && i === 0 ? 99 : i + 1)]),
  );
  return mockFetch({
    "GET /api/v1/health": () => json(HEALTH_OK),
    "GET /api/v1/runs/": () => json(r),
    "GET /api/v1/files/": (url) => {
      const b = byUrl.get(url);
      return b ? new Response(b, { status: 200 }) : new Response("", { status: 404 });
    },
  });
}

function textOfPage(): string {
  return document.body.textContent ?? "";
}

async function loadedEngine(engines: FakeViewerEngine[]): Promise<FakeViewerEngine> {
  await waitFor(() => {
    expect(engines.length).toBeGreaterThan(0);
    expect(engines[0]?.loaded).toBeTruthy();
  });
  const e = engines[0];
  if (!e) throw new Error("no engine");
  await waitFor(() => expect(screen.queryByText("영상을 불러오는 중")).toBeNull());
  return e;
}

beforeEach(() => useViewerStore.getState().reset());
afterEach(() => vi.unstubAllGlobals());

describe("Result viewer", () => {
  it("loads only the API volume URLs (DWI, ADC, mask, component labelmap; verified)", async () => {
    const { calls } = await serve(result());
    const { engines } = renderApp(`/runs/${RUN}/viewer`);
    const engine = await loadedEngine(engines);
    const fileCalls = calls.filter((c) => c.includes("/files/"));
    expect(fileCalls).toHaveLength(4);
    expect(engine.loaded?.components).toBeInstanceOf(ArrayBuffer);
    expect(calls.every((c) => / \/api\/v1\//.test(c))).toBe(true);
    expect(engine.called("setBase").at(-1)).toEqual(["dwi"]);
    expect(engine.called("setMaskColor").at(-1)).toEqual(["orange"]);
    expect(engine.called("setMaskOpacity").at(-1)).toEqual([0.5]);
    expect(engine.called("setRadiological").at(-1)).toEqual([true]);
  });

  it("switches DWI/ADC, views, mask visibility, opacity and palette colour", async () => {
    await serve(result());
    const user = userEvent.setup();
    const { engines } = renderApp(`/runs/${RUN}/viewer`);
    const engine = await loadedEngine(engines);
    const baseGroup = screen.getByRole("radiogroup", { name: "기준 영상" });
    await user.click(within(baseGroup).getByRole("radio", { name: "ADC" }));
    expect(engine.called("setBase").at(-1)).toEqual(["adc"]);
    for (const [label, view] of [
      ["관상면", "coronal"],
      ["시상면", "sagittal"],
      ["3면", "multiplanar"],
      ["3D", "render"],
      ["축상면", "axial"],
    ] as const) {
      await user.click(screen.getByRole("radio", { name: label }));
      expect(engine.called("setView").at(-1)).toEqual([view]);
    }
    await user.click(screen.getByRole("checkbox", { name: "표시" }));
    expect(engine.called("setMaskVisible").at(-1)).toEqual([false]);
    fireEvent.change(screen.getByRole("slider", { name: "불투명도" }), {
      target: { value: "0.8" },
    });
    expect(engine.called("setMaskOpacity").at(-1)).toEqual([0.8]);
    const palette = screen.getByRole("radiogroup", { name: "색상" });
    expect(within(palette).getAllByRole("radio")).toHaveLength(4);
    await user.click(within(palette).getByRole("radio", { name: "하늘색" }));
    expect(engine.called("setMaskColor").at(-1)).toEqual(["skyBlue"]);
  });

  it("ignores colours outside the approved palette", () => {
    const before = useViewerStore.getState().maskColor;
    for (const bad of ["#FF0000", "red", "__proto__", "constructor"])
      useViewerStore.getState().setMaskColor(bad);
    expect(useViewerStore.getState().maskColor).toBe(before);
  });

  it("jumps to the lesion centre and components, and supports keyboard control", async () => {
    await serve(result());
    const user = userEvent.setup();
    const { engines } = renderApp(`/runs/${RUN}/viewer`);
    const engine = await loadedEngine(engines);
    // opened on the lesion centre automatically (API centre-of-mass voxel), then the button
    // jumps there again; navigation is voxel-based (never NiiVue's world-mm approximation)
    expect(engine.called("jumpToVoxel")[0]).toEqual([[65, 55, 30]]);
    await user.click(screen.getByRole("button", { name: "병변 중심으로 이동" }));
    expect(engine.called("jumpToVoxel").at(-1)).toEqual([[65, 55, 30]]);
    await user.click(screen.getByRole("button", { name: "영역 1 선택 및 위치로 이동" }));
    expect(engine.called("jumpToVoxel").at(-1)).toEqual([[60, 50, 30]]); // representative voxel
    const viewer = screen.getByRole("region", { name: /영상 Viewer/ });
    viewer.focus();
    await user.keyboard("{ArrowUp}{ArrowDown}2m]a+0l");
    expect(engine.called("stepSlice")).toEqual([
      [2, 1],
      [2, -1],
    ]);
    const s = useViewerStore.getState();
    expect(s.view).toBe("coronal");
    expect(s.maskVisible).toBe(false);
    expect(s.maskOpacity).toBeCloseTo(0.6);
    expect(s.base).toBe("adc");
    expect(engine.called("zoomBy")).toEqual([[1.25]]);
    expect(engine.called("resetView")).toHaveLength(1);
    expect(engine.called("jumpToVoxel")).toHaveLength(4);
    expect(engine.called("jumpToWorldMm")).toHaveLength(0);
  });

  it("shows radiological orientation with a neurological switch", async () => {
    await serve(result());
    const user = userEvent.setup();
    const { engines } = renderApp(`/runs/${RUN}/viewer`);
    const engine = await loadedEngine(engines);
    expect(screen.getByText("Radiological (영상 오른쪽 = 환자 왼쪽)")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Neurological로 전환" }));
    expect(engine.called("setRadiological").at(-1)).toEqual([false]);
    expect(screen.getByText("Neurological (영상 오른쪽 = 환자 오른쪽)")).toBeInTheDocument();
  });

  it("disposes the engine on unmount", async () => {
    await serve(result());
    const { engines } = renderApp(`/runs/${RUN}/viewer`);
    const engine = await loadedEngine(engines);
    const { cleanup } = await import("@testing-library/react");
    cleanup();
    expect(engine.disposed).toBe(true);
  });

  it("refuses volumes whose SHA-256 does not match", async () => {
    await serve(result(), { tamper: true });
    const { engines } = renderApp(`/runs/${RUN}/viewer`);
    expect(
      await screen.findByText("영상 무결성 검증(SHA-256)에 실패했습니다."),
    ).toBeInTheDocument();
    expect(engines[0]?.loaded ?? null).toBeNull();
  });

  it("shows a WebGL2-unsupported message and still renders the results", async () => {
    const { calls } = await serve(result());
    renderApp(`/runs/${RUN}/viewer`, { webgl: false });
    expect(await screen.findByText(/WebGL2를 지원하지 않아/)).toBeInTheDocument();
    expect(screen.getByText("48.98 mL", { exact: false })).toBeInTheDocument();
    expect(calls.some((c) => c.includes("/files/"))).toBe(false);
  });

  it("offers a download only for shareable files (the mask)", async () => {
    await serve(result());
    renderApp(`/runs/${RUN}/viewer`);
    const link = await screen.findByRole("link", { name: "prediction mask 다운로드" });
    expect(link).toHaveAttribute("href", "/api/v1/files/00000003-0000-4000-8000-000000000003");
    expect(
      screen.getAllByRole("link").filter((a) => a.getAttribute("href")?.includes("/files/")),
    ).toHaveLength(1);
  });
});

describe("Result panel wording", () => {
  it("shows the API values, estimated hemisphere, warnings and a collapsed score panel", async () => {
    await serve(result());
    renderApp(`/runs/${RUN}/viewer`);
    expect(await screen.findByText("모델이 병변 영역을 표시함")).toBeInTheDocument();
    expect(screen.getByText(/48\.98 mL/)).toBeInTheDocument();
    expect(screen.getByText("추정")).toBeInTheDocument();
    expect(screen.getByText(/Image axes are oblique/)).toBeInTheDocument();
    const scores = screen.getByText("모델 출력 점수 (보정되지 않은 기술 지표)").closest("details");
    expect(scores).not.toHaveAttribute("open");
    expect(scores?.textContent).toContain("calibrated = false");
    expect(scores?.textContent).not.toMatch(/\d+(\.\d+)?\s*%/);
    expect(
      screen.getAllByText(
        /연구 및 의료영상 검토 보조용 프로토타입 결과이며, 최종 판단은 의료진이 해야 합니다\./,
      ).length,
    ).toBeGreaterThan(0);
  });

  it("uses the required no-lesion wording and the 3D empty state", async () => {
    await serve(noLesionResult());
    const user = userEvent.setup();
    const { engines } = renderApp(`/runs/${RUN}/viewer`);
    await loadedEngine(engines);
    expect(
      screen.getAllByText("모델이 병변 영역을 표시하지 않았습니다. 병변이 없다는 의미가 아닙니다.")
        .length,
    ).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "병변 중심으로 이동" })).toBeDisabled();
    expect(engines[0]?.called("jumpToWorldMm")).toEqual([]); // nothing to centre on
    expect(engines[0]?.called("jumpToVoxel")).toEqual([]);
    await user.click(screen.getByRole("radio", { name: "3D" }));
    expect(screen.getAllByText(/3D로 표시할 내용이 없습니다/).length).toBeGreaterThan(0);
  });

  it("handles many components and multiple warnings", async () => {
    await serve(manyComponentsResult());
    const user = userEvent.setup();
    const { engines } = renderApp(`/runs/${RUN}/viewer`);
    await loadedEngine(engines);
    expect(await screen.findByText("상위 100개만 표시됩니다.")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /선택 및 위치로 이동/ }).length).toBe(10);
    await user.click(screen.getByRole("button", { name: "더 보기" }));
    expect(screen.getAllByText(/mL ·/).length).toBe(100);
    expect(screen.getByText(/split into many separate parts/)).toBeInTheDocument();
  });

  it.each([
    ["detected", result()],
    ["no lesion", noLesionResult()],
    ["many components", manyComponentsResult()],
  ])("renders no forbidden wording, paths or file names (%s)", async (_name, r) => {
    await serve(r);
    renderApp(`/runs/${RUN}/viewer`);
    await screen.findByText("정량 결과 (연구용 자동 분할)");
    const text = textOfPage();
    for (const phrase of FORBIDDEN_UI_PHRASES) {
      const re = /^[A-Z]+$/.test(phrase) ? new RegExp(`\\b${phrase}\\b`) : new RegExp(phrase);
      expect(text).not.toMatch(re);
    }
    expect(text).not.toMatch(/[A-Za-z]:\\|\/Users\/|\/home\/|file:\/\//);
    expect(text).not.toMatch(/\.nii(\.gz)?\b(?!\s*(파일|·))/);
    expect(text).not.toMatch(/Traceback/);
    expect(text).not.toMatch(/신뢰도\s*\d|확률\s*\d+%/);
  });
});

describe("3D brain background and camera presets", () => {
  async function open3d() {
    await serve(result());
    const user = userEvent.setup();
    const { engines } = renderApp(`/runs/${RUN}/viewer`);
    const engine = await loadedEngine(engines);
    await user.click(screen.getByRole("radio", { name: "3D" }));
    return { user, engine };
  }
  const presetRadio = (label: string) =>
    within(screen.getByRole("radiogroup", { name: "3D 시점" })).getByRole("radio", {
      name: new RegExp(`^${label} 시점`),
    });

  it("shows the brain background by default at opacity 0.40 (mask stays 0.50)", async () => {
    expect(useViewerStore.getState().render3DBackground).toBe(true);
    expect(RENDER_BACKGROUND_OPACITY).toBe(0.4);
    expect(DEFAULT_MASK_OPACITY).toBe(0.5);
    const { engine } = await open3d();
    expect(engine.called("setView").at(-1)).toEqual(["render"]);
    expect(engine.called("setRender3DBackground").at(-1)).toEqual([true]);
    expect(engine.called("setMaskOpacity").at(-1)).toEqual([0.5]);
    const box = screen.getByRole("checkbox", { name: "3D 뇌 배경 표시" });
    expect(box).toBeChecked();
    expect(box).toHaveAttribute("aria-keyshortcuts", "B");
    expect(textOfPage()).not.toContain("DWI 배경 반투명");
  });

  it("keeps the B key and the checkbox in sync", async () => {
    const { user, engine } = await open3d();
    const box = screen.getByRole("checkbox", { name: "3D 뇌 배경 표시" });
    screen.getByRole("region", { name: /영상 Viewer/ }).focus();
    await user.keyboard("b");
    expect(box).not.toBeChecked();
    expect(engine.called("setRender3DBackground").at(-1)).toEqual([false]);
    await user.keyboard("B");
    expect(box).toBeChecked();
    expect(engine.called("setRender3DBackground").at(-1)).toEqual([true]);
    await user.click(box);
    expect(useViewerStore.getState().render3DBackground).toBe(false);
    screen.getByRole("region", { name: /영상 Viewer/ }).focus();
    await user.keyboard("b");
    expect(box).toBeChecked();
    expect(ko.viewer.shortcuts).toContain("B 3D 뇌 배경");
  });

  it("B toggles the background even without an engine", () => {
    const handled = handleViewerKey("b", useViewerStore.getState(), null, null);
    expect(handled).toBe(true);
    expect(useViewerStore.getState().render3DBackground).toBe(false);
  });

  it("starts at the front preset and offers labelled, selectable presets", async () => {
    const { user, engine } = await open3d();
    expect(DEFAULT_CAMERA_PRESET).toBe("front");
    expect(CAMERA_PRESETS.front).toEqual({ azimuth: 180, elevation: 0 });
    const group = screen.getByRole("radiogroup", { name: "3D 시점" });
    const radios = within(group).getAllByRole("radio");
    expect(radios.map((r) => r.textContent)).toEqual(["정면", "위", "왼쪽", "오른쪽"]);
    for (const r of radios) expect(r.getAttribute("title")).toBe(r.getAttribute("aria-label"));
    expect(presetRadio("정면")).toHaveAttribute("aria-checked", "true");
    expect(presetRadio("정면")).toHaveClass("active");
    for (const [label, preset] of [
      ["위", "top"],
      ["왼쪽", "left"],
      ["오른쪽", "right"],
    ] as const) {
      await user.click(presetRadio(label));
      expect(engine.called("setCameraPreset").at(-1)).toEqual([preset]);
      expect(presetRadio(label)).toHaveAttribute("aria-checked", "true");
      expect(presetRadio("정면")).toHaveAttribute("aria-checked", "false");
    }
    expect(
      screen.getByText(/오른쪽 시점: 환자 오른쪽에서 봄/, { selector: "p" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "기본 시점(정면)으로 돌아가기" }));
    expect(engine.called("setCameraPreset").at(-1)).toEqual(["front"]);
    expect(presetRadio("정면")).toHaveAttribute("aria-checked", "true");
    // keyboard: focus a preset and activate it with Enter
    presetRadio("위").focus();
    await user.keyboard("{Enter}");
    expect(engine.camera).toBe("top");
  });

  it("clears the selection after free rotation and keeps the camera across DWI/ADC", async () => {
    const { user, engine } = await open3d();
    await user.click(presetRadio("위"));
    const presetCalls = engine.called("setCameraPreset").length;
    const baseGroup = screen.getByRole("radiogroup", { name: "기준 영상" });
    await user.click(within(baseGroup).getByRole("radio", { name: "ADC" }));
    await user.click(within(baseGroup).getByRole("radio", { name: "DWI" }));
    expect(engine.called("setBase").slice(-2)).toEqual([["adc"], ["dwi"]]);
    expect(engine.called("setCameraPreset")).toHaveLength(presetCalls);
    expect(presetRadio("위")).toHaveAttribute("aria-checked", "true");
    act(() => engine.rotateFreely());
    const group = screen.getByRole("radiogroup", { name: "3D 시점" });
    expect(
      within(group)
        .getAllByRole("radio")
        .every((r) => r.getAttribute("aria-checked") === "false"),
    ).toBe(true);
    expect(screen.getByText(/자유 회전 중/)).toBeInTheDocument();
    screen.getByRole("region", { name: /영상 Viewer/ }).focus();
    await user.keyboard("0");
    expect(engine.camera).toBe("front");
    expect(presetRadio("정면")).toHaveAttribute("aria-checked", "true");
  });

  it("hides camera presets outside 3D", async () => {
    await serve(result());
    const { engines } = renderApp(`/runs/${RUN}/viewer`);
    await loadedEngine(engines);
    expect(screen.queryByRole("radiogroup", { name: "3D 시점" })).toBeNull();
    expect(screen.queryByRole("checkbox", { name: "3D 뇌 배경 표시" })).toBeNull();
  });
});
