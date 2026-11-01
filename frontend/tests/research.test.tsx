/** Phase 7-B Research Validation Mode (FakeViewerEngine; synthetic contract fixtures). */
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { sha256Hex } from "../src/api/client";
import type { ResearchCase, RunResult } from "../src/api/schemas";
import { buildDifferenceLut, GT_ONLY_COLOR } from "../src/features/viewer/differenceLut";
import type { FakeViewerEngine } from "../src/features/viewer/FakeViewerEngine";
import { hexToRgb } from "../src/features/viewer/ViewerEngine";
import { useViewerStore } from "../src/features/viewer/viewerStore";
import { FORBIDDEN_UI_PHRASES } from "../src/i18n/ko";
import { result, RUN } from "./fixtures";
import { HEALTH_OK, json, mockFetch, renderApp } from "./render";
import { negativeResearchCase, researchCase, researchList } from "./researchFixtures";

const bytes = (n: number) => new Uint8Array(64).fill(n).buffer;

async function serveResearch(detail: ResearchCase | null, list = researchList()) {
  const r = detail ? structuredClone(detail) : null;
  if (r) for (const [i, v] of r.volumes.entries()) v.sha256 = await sha256Hex(bytes(i + 1));
  const byUrl = new Map((r?.volumes ?? []).map((v, i) => [v.url, bytes(i + 1)]));
  const notFound = () =>
    json(
      {
        error: {
          code: "NOT_FOUND",
          message: "Not Found",
          request_id: "r",
          retryable: false,
          details: {},
        },
      },
      404,
    );
  return mockFetch({
    "GET /api/v1/health": () => json(HEALTH_OK),
    "GET /api/v1/research/cases/": () => (r ? json(r) : notFound()),
    "GET /api/v1/research/cases": () => (detail === null ? notFound() : json(list)),
    "GET /api/v1/files/": (url) => {
      const b = byUrl.get(url);
      return b ? new Response(b, { status: 200 }) : new Response("", { status: 404 });
    },
  });
}

async function openCase(detail: ResearchCase = researchCase()) {
  const served = await serveResearch(detail);
  const user = userEvent.setup();
  const view = renderApp(`/research/cases/${detail.alias}`);
  await waitFor(() => expect(view.engines[0]?.loaded).toBeTruthy());
  await waitFor(() => expect(screen.queryByText(/영상을 불러오는 중/)).toBeNull());
  return { ...view, engine: view.engines[0] as FakeViewerEngine, user, calls: served.calls };
}

const lastOverlay = (e: FakeViewerEngine) => e.called("setResearchOverlay").at(-1)?.[0];
const text = () => document.body.textContent ?? "";

beforeEach(() => useViewerStore.getState().reset());
afterEach(() => vi.unstubAllGlobals());

describe("research list", () => {
  it("lists aliases under the always-visible research banner", async () => {
    await serveResearch(researchCase());
    renderApp("/research");
    expect(await screen.findByRole("link", { name: "val-001 열기" })).toHaveAttribute(
      "href",
      "/research/cases/val-001",
    );
    expect(screen.getByRole("note", { name: "연구·검증 모드" })).toHaveTextContent(
      "Validation 데이터에서 AI 결과와 Ground Truth를 비교합니다. 임상 진단 결과가 아닙니다.",
    );
  });

  it("says the mode is unavailable when the API answers 404 (research_enabled=false)", async () => {
    await serveResearch(null);
    renderApp("/research");
    expect(await screen.findByRole("alert")).toHaveTextContent("연구·검증 모드가 비활성화");
  });

  it("treats malformed aliases as not found without calling the API", async () => {
    const { calls } = await serveResearch(researchCase());
    renderApp("/research/cases/sub-strokecase0001");
    expect(await screen.findByText("페이지를 찾을 수 없습니다.")).toBeInTheDocument();
    expect(calls.some((c) => c.includes("/research/"))).toBe(false);
  });
});

describe("research case viewer", () => {
  it("loads the five verified volumes and shows alias, dataset, model and threshold", async () => {
    const { engine, calls } = await openCase();
    expect(calls.filter((c) => c.includes("/files/"))).toHaveLength(5);
    expect(engine.loaded?.difference).toBeInstanceOf(ArrayBuffer);
    const info = screen.getByText("케이스 (익명)").closest("dl") as HTMLElement;
    expect(info).toHaveTextContent("val-001");
    expect(info).toHaveTextContent("isles2022");
    expect(info).toHaveTextContent("0.5");
    expect(screen.getByRole("note", { name: "연구·검증 모드" })).toBeInTheDocument();
    expect(text()).not.toMatch(/sub-strokecase|\.nii|C:\\|storage\//);
    for (const bad of FORBIDDEN_UI_PHRASES) expect(text()).not.toContain(bad);
  });

  it("switches difference / GT / AI / base-only overlays with a mode-specific legend", async () => {
    const { user, engine } = await openCase();
    // 1. default = difference map; 2. button order puts it first
    expect(lastOverlay(engine)).toEqual({
      kind: "difference",
      visible: { tp: true, fp: true, fn: true },
    });
    const radios = within(screen.getByRole("radiogroup", { name: "표시" })).getAllByRole("radio");
    expect(radios.map((r) => r.textContent)).toEqual([
      "차이 지도 (TP/FP/FN)",
      "Ground Truth",
      "AI 예측",
      "기본 영상만",
    ]);
    const band = () => screen.getByText("현재 표시:").closest("p") as HTMLElement;
    const legendItems = () =>
      within(screen.getByLabelText("범례"))
        .getAllByRole("listitem")
        .map((li) => li.textContent);
    // 3./4. difference: band + TP/FP/FN only
    expect(band()).toHaveTextContent("현재 표시: 차이 지도 (TP / FP / FN)");
    expect(legendItems()).toEqual([
      "일치 영역 (TP)",
      "AI만 표시한 영역 (FP)",
      "GT에 있으나 AI가 놓친 영역 (FN)",
    ]);
    // 5. GT only
    await user.click(screen.getByRole("radio", { name: "Ground Truth" }));
    expect(lastOverlay(engine)).toEqual({ kind: "ground_truth" });
    expect(band()).toHaveTextContent("현재 표시: Ground Truth");
    expect(legendItems()).toEqual(["Ground Truth 영역 (TP + FN)"]);
    // 6. AI prediction + grey meaning only
    await user.click(screen.getByRole("radio", { name: "AI 예측" }));
    expect(lastOverlay(engine)).toBeNull();
    expect(band()).toHaveTextContent("현재 표시: AI 예측 전체 (TP + FP)");
    expect(legendItems()).toEqual([
      "AI 예측 영역 (TP + FP)",
      "회색: 선택하지 않은 영역을 검토 편의를 위해 흐리게 표시합니다. AI 원본 결과나 신뢰도를 변경하지 않습니다.",
    ]);
    // 7. base only
    await user.click(screen.getByRole("radio", { name: "기본 영상만" }));
    expect(band()).toHaveTextContent("현재 표시: 기본 영상만");
    expect(legendItems()).toEqual(["Overlay 없음"]);
    expect(engine.called("setMaskVisible").at(-1)).toEqual([false]);
    await user.click(screen.getByRole("radio", { name: "차이 지도 (TP/FP/FN)" }));
    expect(engine.called("setMaskVisible").at(-1)).toEqual([true]);
  });

  it("turns TP / FP / FN on and off individually", async () => {
    const { user, engine } = await openCase();
    await user.click(screen.getByRole("checkbox", { name: "AI만 표시한 영역 (FP)" }));
    expect(lastOverlay(engine)).toEqual({
      kind: "difference",
      visible: { tp: true, fp: false, fn: true },
    });
    await user.click(screen.getByRole("checkbox", { name: "일치 영역 (TP)" }));
    expect(lastOverlay(engine)).toMatchObject({ visible: { tp: false, fp: false, fn: true } });
  });

  it("shows the backend's patient metrics (positive ground truth)", async () => {
    await openCase();
    const m = screen.getByRole("region", { name: "환자 단위 평가 지표 (연구용)" });
    expect(m).toHaveTextContent("Dice0.9468");
    expect(m).toHaveTextContent("IoU0.8989");
    expect(m).toHaveTextContent("Sensitivity0.9692");
    expect(m).toHaveTextContent("Precision0.9254");
    expect(m).toHaveTextContent("GT 병변 부피46.760 mL");
    expect(m).toHaveTextContent("AI 병변 부피48.976 mL");
    expect(m).toHaveTextContent("상대 부피 오차4.7 %");
    expect(m).toHaveTextContent("5665 / 457 / 180");
    expect(m).toHaveTextContent("불일치 0개");
  });

  it("empty ground truth: overlap metrics undefined (never 0 or 1), AI-only voxels listed", async () => {
    await openCase(negativeResearchCase());
    const m = screen.getByRole("region", { name: "환자 단위 평가 지표 (연구용)" });
    expect(within(m).getAllByText("정의되지 않음: Ground Truth 병변 없음")).toHaveLength(4);
    expect(m).not.toHaveTextContent(/Dice\s*[01]\.0000/);
    expect(m).toHaveTextContent("AI False Positive voxel 수60");
    expect(m).toHaveTextContent("AI False Positive 연결 영역 수3");
    expect(m).toHaveTextContent("Correct Negative아니오");
  });

  it("adds GT comparison to the component review and keeps selection / jumps", async () => {
    const { user, engine } = await openCase();
    const list = within(screen.getByRole("list", { name: "연결 영역 검토" }));
    const rows = list.getAllByRole("listitem");
    expect(rows[0]).toHaveTextContent("GT와 겹침 · 일대일 겹침 · TP voxel 5665");
    expect(rows[0]).toHaveTextContent("연결된 GT GT 1");
    expect(rows[1]).toHaveTextContent("AI만 표시된 영역 · AI만 표시된 영역");
    await user.click(screen.getByRole("radio", { name: "AI 예측" }));
    await user.click(screen.getByRole("button", { name: "영역 2 선택 및 위치로 이동" }));
    expect(engine.called("jumpToVoxel").at(-1)).toEqual([[81, 51, 57]]);
    expect(engine.called("setComponentDisplay").at(-1)?.[0]).toMatchObject({ selected: 2 });
    await waitFor(() =>
      expect(screen.getByRole("slider", { name: "Slice (축상면)" })).toHaveValue("57"),
    );
  });

  it("lists matching groups and jumps to a ground-truth-only region", async () => {
    const { user, engine } = await openCase();
    const table = screen.getByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(5);
    expect(table).toHaveTextContent("일대일 겹침");
    expect(table).toHaveTextContent("GT에만 표시된 영역");
    await user.click(screen.getByRole("button", { name: "GT 2 위치로 이동" }));
    expect(engine.called("jumpToVoxel").at(-1)).toEqual([[50, 55, 25]]);
  });

  it("disposes the engine and its listeners on unmount", async () => {
    const { engine, unmount } = await openCase();
    expect(engine.locationListenerCount).toBe(1);
    unmount();
    expect(engine.locationListenerCount).toBe(0);
    expect(engine.disposed).toBe(true);
  });
});

describe("research viewer UX (P0)", () => {
  const summary = () => screen.getByRole("region", { name: "선택 요약" });

  it("AI prediction is one orange by default; region colours are an opt-in review toggle", async () => {
    const { user, engine } = await openCase();
    // 8. research default: single colour (#E69F00 via the orange palette entry)
    expect(useViewerStore.getState().componentColoring).toBe("single");
    expect(useViewerStore.getState().maskColor).toBe("orange");
    expect(engine.called("setComponentDisplay").at(-1)?.[0]).toMatchObject({ coloring: "single" });
    await user.click(screen.getByRole("radio", { name: "AI 예측" }));
    // 9./10. toggle off by default, with its note; on -> legend explains it has no meaning
    const toggle = screen.getByRole("checkbox", { name: "영역별 색 구분 (검토용)" });
    expect(toggle).not.toBeChecked();
    expect(
      screen.getByText(
        "색상은 영역 순서를 구분하기 위한 표시이며 의료적 의미나 정확도를 나타내지 않습니다.",
      ),
    ).toBeInTheDocument();
    await user.click(toggle);
    expect(engine.called("setComponentDisplay").at(-1)?.[0]).toMatchObject({
      coloring: "component",
    });
    expect(screen.getByLabelText("범례")).toHaveTextContent(
      "영역별 색은 순서 구분용이며 의료적 의미가 없습니다.",
    );
    // the general palette / colour-mode switch is not offered on research pages
    expect(screen.queryByRole("radio", { name: "단색" })).toBeNull();
  });

  it("summarises the selected AI region with its GT comparison (11.-16.)", async () => {
    const { user } = await openCase();
    expect(screen.queryByRole("region", { name: "선택 요약" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "영역 2 선택 및 위치로 이동" }));
    const p = summary();
    expect(p).toHaveTextContent("선택: AI 영역 2");
    expect(p).toHaveTextContent("부피0.400 mL");
    expect(p).toHaveTextContent("Slice57–59");
    expect(p).toHaveTextContent("위치환자 좌측 추정");
    expect(p).toHaveTextContent("GT 비교GT와 겹치지 않음 (AI만 표시된 영역)");
    expect(p).toHaveTextContent("MatchingAI만 표시된 영역");
    expect(p).toHaveTextContent("TP / FP0 / 50");
    expect(p).toHaveTextContent("TP 비율0.00");
    expect(p).toHaveTextContent("작은 분리 영역");
    // 16. default difference mode: highlighting does not apply, said so
    expect(p).toHaveTextContent("현재 차이 지도 표시에서는 AI 영역 강조가 적용되지 않습니다.");
    expect(p).toHaveTextContent("3D에서는 작은 영역이 배경 볼륨과 시점에 따라");
    expect(p).toHaveTextContent("연구용 안내이며 진단 기준이나 의료적 판단 절차가 아닙니다.");
    await user.click(screen.getByRole("button", { name: "영역 1 선택 및 위치로 이동" }));
    expect(summary()).toHaveTextContent("GT 비교일부 겹침");
    expect(summary()).toHaveTextContent("Matching일대일 겹침");
    expect(summary()).toHaveTextContent("연결된 GTGT 1");
    expect(summary()).toHaveTextContent("TP / FP5665 / 397");
    expect(summary()).toHaveTextContent("TP 비율0.93");
    expect(summary()).not.toHaveTextContent("작은 분리 영역");
    // 25./26. after selecting: still no case IDs, paths or forbidden wording
    for (const bad of FORBIDDEN_UI_PHRASES) expect(text()).not.toContain(bad);
    expect(text()).not.toMatch(/sub-strokecase|\.nii|C:\\|storage\//);
  });

  it("shows an unavailable side as '추정 불가' only (no '추정 불가 추정')", async () => {
    const base = researchCase();
    const comps = base.prediction_components.map((c) =>
      c.size_rank === 3 ? { ...c, hemisphere_estimate: "not_available" } : c,
    );
    const { user } = await openCase(researchCase({ prediction_components: comps }));
    await user.click(screen.getByRole("button", { name: "영역 3 선택 및 위치로 이동" }));
    expect(summary()).toHaveTextContent("위치추정 불가");
    expect(summary()).not.toHaveTextContent("추정 불가 추정");
  });

  it("summarises a ground-truth-only region chosen from the matching table", async () => {
    const { user, engine } = await openCase();
    await user.click(screen.getByRole("button", { name: "GT 2 위치로 이동" }));
    expect(engine.called("jumpToVoxel").at(-1)).toEqual([[50, 55, 25]]);
    const p = summary();
    expect(p).toHaveTextContent("선택: GT 영역 2");
    expect(p).toHaveTextContent("GT 비교AI가 표시하지 않은 GT 영역");
    expect(p).toHaveTextContent("연결된 AI 영역연결된 AI 영역 없음");
    expect(within(p).getByRole("button", { name: "AI 예측에서 강조 보기" })).toBeDisabled();
  });

  it("review buttons: AI highlight, DWI/ADC keep the position, 3-plane check, clear (17.-22.)", async () => {
    const { user, engine } = await openCase();
    await user.click(screen.getByRole("button", { name: "영역 3 선택 및 위치로 이동" }));
    await waitFor(() => expect(useViewerStore.getState().crosshair?.vox).toEqual([55, 44, 2]));
    const p = () => within(summary());
    // 17. difference -> AI highlight keeps the selection
    await user.click(p().getByRole("button", { name: "AI 예측에서 강조 보기" }));
    expect(useViewerStore.getState().researchMode).toBe("ai");
    expect(lastOverlay(engine)).toBeNull();
    expect(engine.called("setComponentDisplay").at(-1)?.[0]).toMatchObject({
      selected: 3,
      coloring: "single",
    });
    // 21. DWI / ADC switch keeps the crosshair
    await user.click(p().getByRole("button", { name: "ADC 보기" }));
    expect(engine.called("setBase").at(-1)).toEqual(["adc"]);
    await user.click(p().getByRole("button", { name: "DWI 보기" }));
    expect(engine.called("setBase").at(-1)).toEqual(["dwi"]);
    expect(useViewerStore.getState().crosshair?.vox).toEqual([55, 44, 2]);
    // selecting never changes the view by itself
    expect(engine.called("setView").at(-1)).toEqual(["axial"]);
    // 18.-20. 3-plane check: multiplanar + representative voxel, sliders follow
    await user.click(p().getByRole("button", { name: "3면에서 확인" }));
    expect(engine.called("setView").at(-1)).toEqual(["multiplanar"]);
    expect(engine.called("jumpToVoxel").at(-1)).toEqual([[55, 44, 2]]);
    await waitFor(() =>
      expect(screen.getByRole("slider", { name: "Slice (축상면)" })).toHaveValue("2"),
    );
    expect(screen.getByRole("slider", { name: "Slice (관상면)" })).toHaveValue("44");
    // difference button goes back, selection kept
    await user.click(p().getByRole("button", { name: "차이 지도 보기" }));
    expect(useViewerStore.getState().researchMode).toBe("difference");
    expect(useViewerStore.getState().selectedComponent).toBe(3);
    // 22. clear selection
    await user.click(p().getByRole("button", { name: "선택 해제" }));
    expect(screen.queryByRole("region", { name: "선택 요약" })).toBeNull();
    expect(useViewerStore.getState().selectedComponent).toBeNull();
  });

  it("disables hide / highlight controls outside AI mode and says why (23.)", async () => {
    const { user } = await openCase();
    const hide = screen.getByRole("button", { name: "영역 1 표시에서 숨기기" });
    expect(hide).toBeDisabled();
    expect(hide).toHaveAttribute(
      "title",
      "숨기기와 선택 강조는 AI 예측 표시에서 사용할 수 있습니다.",
    );
    expect(screen.getByRole("button", { name: "전체 다시 표시" })).toBeDisabled();
    // difference mode swatches show the comparison state, not region colours
    expect(screen.getAllByTitle("일부 GT와 겹침 (TP + FP)")).toHaveLength(1);
    expect(screen.getAllByTitle("AI만 (FP)")).toHaveLength(2);
    await user.click(screen.getByRole("radio", { name: "Ground Truth" }));
    expect(screen.getByRole("button", { name: "영역 1 표시에서 숨기기" })).toHaveAttribute(
      "title",
      "현재 Ground Truth 표시에서는 AI 영역 강조가 적용되지 않습니다.",
    );
    await user.click(screen.getByRole("radio", { name: "AI 예측" }));
    expect(screen.getByRole("button", { name: "영역 1 표시에서 숨기기" })).toBeEnabled();
  });
});

describe("general inference mode stays separate", () => {
  it("shows no ground truth, TP/FP/FN, patient metrics or matching", async () => {
    const r: RunResult = structuredClone(result());
    for (const [i, v] of r.viewer.volumes.entries()) v.sha256 = await sha256Hex(bytes(i + 1));
    const byUrl = new Map(r.viewer.volumes.map((v, i) => [v.url, bytes(i + 1)]));
    mockFetch({
      "GET /api/v1/health": () => json(HEALTH_OK),
      "GET /api/v1/runs/": () => json(r),
      "GET /api/v1/files/": (url) => new Response(byUrl.get(url) ?? "", { status: 200 }),
    });
    const { engines } = renderApp(`/runs/${RUN}/viewer`);
    await waitFor(() => expect(engines[0]?.loaded).toBeTruthy());
    await waitFor(() => expect(screen.queryByText(/영상을 불러오는 중/)).toBeNull());
    const page = text();
    for (const s of [
      "Ground Truth",
      "(TP)",
      "(FP)",
      "(FN)",
      "IoU",
      "Sensitivity",
      "Precision",
      "GT 병변",
      "연구·검증",
      "AI–GT",
      "GT와 겹침",
      "환자 단위",
      "현재 표시:",
      "선택 요약",
      "영역별 색 구분 (검토용)",
      "3면에서 확인",
    ])
      expect(page, s).not.toContain(s);
    expect(engines[0]?.called("setResearchOverlay")).toEqual([]);
    expect(engines[0]?.loaded?.difference).toBeUndefined();
  });
});

describe("difference LUT (pure)", () => {
  it("colours 1/2/3 per class, GT mode = TP + FN in the GT colour, others transparent", () => {
    const all = buildDifferenceLut({
      kind: "difference",
      visible: { tp: true, fp: true, fn: true },
    });
    expect([...all.slice(4, 8)]).toEqual([0, 158, 115, 255]);
    expect([...all.slice(8, 12)]).toEqual([230, 159, 0, 255]);
    expect([...all.slice(12, 16)]).toEqual([86, 180, 233, 255]);
    expect([...all.slice(0, 4)]).toEqual([0, 0, 0, 0]);
    expect([...all.slice(16)].every((v) => v === 0)).toBe(true);
    const noFp = buildDifferenceLut({
      kind: "difference",
      visible: { tp: true, fp: false, fn: true },
    });
    expect(noFp[11]).toBe(0);
    const gt = buildDifferenceLut({ kind: "ground_truth" });
    const pink = [...hexToRgb(GT_ONLY_COLOR), 255];
    expect([...gt.slice(4, 8)]).toEqual(pink);
    expect([...gt.slice(12, 16)]).toEqual(pink);
    expect(gt[11]).toBe(0);
    expect(() => buildDifferenceLut({ kind: "x" } as never)).toThrow();
  });
});
