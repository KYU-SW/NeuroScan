import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CASE, caseInfo, RUN, RUN2, runStatus } from "./fixtures";
import { apiError, HEALTH_OK, json, mockFetch, renderApp } from "./render";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Case page", () => {
  it("starts a run and goes to its progress page", async () => {
    mockFetch({
      "GET /api/v1/cases/": () => json(caseInfo()),
      "POST /api/v1/cases/": () =>
        json(
          {
            run_id: RUN,
            case_id: CASE,
            status: "queued",
            queue_position: 1,
            worker_online: true,
            links: {},
          },
          202,
        ),
      "GET /api/v1/runs/": () => json(runStatus()),
    });
    const user = userEvent.setup();
    renderApp(`/cases/${CASE}`);
    await user.click(await screen.findByRole("button", { name: "분석 실행" }));
    await waitFor(() => expect(screen.getByTestId("location")).toHaveTextContent(`/runs/${RUN}`));
  });

  it("follows the active run on 409 RUN_ALREADY_ACTIVE (duplicate submit)", async () => {
    mockFetch({
      "GET /api/v1/cases/": () => json(caseInfo()),
      "POST /api/v1/cases/": () => apiError(409, "RUN_ALREADY_ACTIVE", { run_id: RUN2 }),
      "GET /api/v1/runs/": () => json(runStatus({ run_id: RUN2 })),
    });
    const user = userEvent.setup();
    renderApp(`/cases/${CASE}`);
    await user.click(await screen.findByRole("button", { name: "분석 실행" }));
    await waitFor(() => expect(screen.getByTestId("location")).toHaveTextContent(`/runs/${RUN2}`));
  });

  it("offers view-existing or rerun on 409 RESULT_EXISTS", async () => {
    const { calls } = mockFetch({
      "GET /api/v1/cases/": () => json(caseInfo()),
      "POST /api/v1/cases/": (_u, init) =>
        String(init?.body).includes("true")
          ? json(
              {
                run_id: RUN2,
                case_id: CASE,
                status: "queued",
                queue_position: 1,
                worker_online: true,
                links: {},
              },
              202,
            )
          : apiError(409, "RESULT_EXISTS", { run_id: RUN }),
      "GET /api/v1/runs/": () => json(runStatus({ run_id: RUN2 })),
    });
    const user = userEvent.setup();
    renderApp(`/cases/${CASE}`);
    await user.click(await screen.findByRole("button", { name: "분석 실행" }));
    expect(await screen.findByText("같은 모델로 완료된 결과가 이미 있습니다.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "다시 실행" }));
    await waitFor(() => expect(screen.getByTestId("location")).toHaveTextContent(`/runs/${RUN2}`));
    expect(calls.filter((c) => c.startsWith("POST"))).toHaveLength(2);
  });

  it("deletes the case only after the in-page confirmation", async () => {
    const { calls } = mockFetch({
      "GET /api/v1/cases/": () => json(caseInfo()),
      "DELETE /api/v1/cases/": () =>
        json({ case_id: CASE, deleted: true, runs_deleted: 0, files_deleted: 2 }),
      "GET /api/v1/health": () => json(HEALTH_OK),
    });
    const user = userEvent.setup();
    renderApp(`/cases/${CASE}`);
    await user.click(await screen.findByRole("button", { name: "Case 삭제" }));
    expect(screen.getByRole("alertdialog")).toBeInTheDocument();
    expect(calls.some((c) => c.startsWith("DELETE"))).toBe(false);
    await user.click(screen.getByRole("button", { name: "삭제" }));
    await waitFor(() => expect(screen.getByTestId("location")).toHaveTextContent(/^\/$/));
    expect(calls).toContain(`DELETE /api/v1/cases/${CASE}?confirm=true`);
  });

  it("treats malformed ids as not found without calling the API", async () => {
    const { calls } = mockFetch({});
    renderApp("/cases/..%2F..%2Fetc");
    expect(await screen.findByText("페이지를 찾을 수 없습니다.")).toBeInTheDocument();
    expect(calls.filter((c) => c.includes("/cases/"))).toEqual([]);
  });
});

describe("Run progress", () => {
  it("polls queued → running → succeeded, shows stages, then opens the viewer and stops", async () => {
    const seq = [
      runStatus(),
      runStatus({
        status: "running",
        current_stage: "predicting",
        progress: 0.4,
        queue_position: null,
      }),
      runStatus({ status: "succeeded", current_stage: "done", progress: 1, queue_position: null }),
    ];
    let i = 0;
    const { calls } = mockFetch({
      "GET /api/v1/runs/": (url) =>
        url.endsWith("/result")
          ? apiError(409, "RUN_NOT_FINISHED")
          : json(seq[Math.min(i++, seq.length - 1)]),
    });
    renderApp(`/runs/${RUN}`);
    expect(await screen.findByText(/대기열 1번째/)).toBeInTheDocument();
    expect(await screen.findByText(/분석 중 · 40%/, {}, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByText("예측 · 원본 공간 복원")).toHaveAttribute("aria-current", "step");
    await waitFor(
      () => expect(screen.getByTestId("location")).toHaveTextContent(`/runs/${RUN}/viewer`),
      {
        timeout: 4000,
      },
    );
    const polls = calls.filter((c) => c === `GET /api/v1/runs/${RUN}`).length;
    await new Promise((r) => setTimeout(r, 1500));
    expect(calls.filter((c) => c === `GET /api/v1/runs/${RUN}`).length).toBe(polls); // polling stopped
  }, 10_000);

  it("shows engine failures distinctly and stops polling", async () => {
    const { calls } = mockFetch({
      "GET /api/v1/runs/": () =>
        json(
          runStatus({
            status: "failed",
            error: {
              code: "E_BRAIN_MASK",
              message: "Only skull-stripped images are supported.",
              retryable: false,
            },
          }),
        ),
    });
    renderApp(`/runs/${RUN}`);
    expect(await screen.findByText(/skull-strip된 영상만 지원합니다/)).toBeInTheDocument();
    expect(screen.getByText(/입력 영상 조건을 확인하세요/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Case로 돌아가 다시 실행" })).toBeInTheDocument();
    const n = calls.length;
    await new Promise((r) => setTimeout(r, 1300));
    expect(calls.filter((c) => c.startsWith("GET /api/v1/runs/")).length).toBe(
      calls.slice(0, n).filter((c) => c.startsWith("GET /api/v1/runs/")).length,
    );
  });

  it("cancels a queued run and shows cancelled", async () => {
    mockFetch({
      "GET /api/v1/runs/": () => json(runStatus()),
      "POST /api/v1/runs/": () => json(runStatus({ status: "cancelled", queue_position: null })),
    });
    const user = userEvent.setup();
    renderApp(`/runs/${RUN}`);
    await user.click(await screen.findByRole("button", { name: "대기 중인 작업 취소" }));
    expect(await screen.findByTestId("run-status")).toHaveTextContent("취소됨");
  });

  it("shows the API-unavailable banner when health fails", async () => {
    mockFetch({
      "GET /api/v1/health": () => new Response("", { status: 502 }),
      "GET /api/v1/runs/": () => json(runStatus()),
    });
    renderApp(`/runs/${RUN}`);
    expect(await screen.findByText(/API 서버에 연결할 수 없습니다/)).toBeInTheDocument();
  });

  it("shows the worker-offline banner", async () => {
    mockFetch({
      "GET /api/v1/health": () => json({ ...HEALTH_OK, status: "degraded", worker: "offline" }),
      "GET /api/v1/runs/": () => json(runStatus()),
    });
    renderApp(`/runs/${RUN}`);
    expect(await screen.findByText(/추론 worker가 실행 중이 아닙니다/)).toBeInTheDocument();
  });
});
