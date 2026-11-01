import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { api } from "../api/endpoints";
import { errorMessage } from "../api/errors";
import { type RunStatus, UUID_ID } from "../api/schemas";
import { Layout } from "../components/Layout";
import { ko } from "../i18n/ko";
import { NotFoundPage } from "./NotFoundPage";

const STAGES = [
  "validating",
  "preprocessing",
  "predicting",
  "quantifying",
  "writing",
  "registering",
];
export const RUN_POLL_MS = 1000;

export function isTerminal(status: RunStatus["status"] | undefined): boolean {
  return status === "succeeded" || status === "failed" || status === "cancelled";
}

export function RunProgressPage() {
  const { runId = "" } = useParams();
  if (!UUID_ID.test(runId)) return <NotFoundPage />;
  return <RunView runId={runId} />;
}

function RunView({ runId }: { runId: string }) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const run = useQuery({
    queryKey: ["run", runId],
    queryFn: ({ signal }) => api.getRun(runId, signal),
    refetchInterval: (q) => (isTerminal(q.state.data?.status) ? false : RUN_POLL_MS),
  });
  const cancel = useMutation({
    mutationFn: () => api.cancelRun(runId),
    onSuccess: (data) => qc.setQueryData(["run", runId], data),
  });
  const status = run.data?.status;

  useEffect(() => {
    if (status === "succeeded") navigate(`/runs/${runId}/viewer`, { replace: true });
  }, [status, runId, navigate]);

  const r = run.data;
  const stageIdx = r?.current_stage ? STAGES.indexOf(r.current_stage) : -1;
  return (
    <Layout caseId={r?.case_id}>
      <section className="panel narrow">
        <h1>{ko.run.title}</h1>
        {run.error && (
          <p className="error" role="alert">
            {errorMessage(run.error)}
          </p>
        )}
        {r && (
          <>
            <p className="status-line" aria-live="polite" data-testid="run-status">
              {ko.run[r.status]}
              {r.status === "queued" && r.queue_position !== null
                ? ` · ${ko.run.queuePosition(r.queue_position)}`
                : ""}
              {r.status === "running" ? ` · ${Math.round(r.progress * 100)}%` : ""}
            </p>
            <div
              className="progress"
              role="progressbar"
              aria-label={ko.run.title}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(r.progress * 100)}
            >
              <div style={{ width: `${Math.round(r.progress * 100)}%` }} />
            </div>
            <ol className="stages">
              {STAGES.map((s, i) => (
                <li
                  key={s}
                  className={
                    i < stageIdx || r.status === "succeeded"
                      ? "done"
                      : i === stageIdx
                        ? "current"
                        : ""
                  }
                  aria-current={i === stageIdx ? "step" : undefined}
                >
                  {ko.stages[s]}
                </li>
              ))}
            </ol>
            {r.status === "queued" && (
              <button type="button" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
                {ko.run.cancelQueued}
              </button>
            )}
            {cancel.error && (
              <p className="error" role="alert">
                {errorMessage(cancel.error)}
              </p>
            )}
            {r.status === "failed" && r.error && (
              <div className="card error-card" role="alert">
                <p>
                  <span aria-hidden="true">⛔</span>{" "}
                  {ko.errors.byCode[r.error.code] ?? r.error.message} <code>{r.error.code}</code>
                </p>
                <p className="note">
                  {r.error.retryable ? ko.run.retryLater : ko.run.inputProblem}
                </p>
              </div>
            )}
            {(r.status === "failed" || r.status === "cancelled") && (
              <Link to={`/cases/${r.case_id}`}>{ko.run.retry}</Link>
            )}
          </>
        )}
      </section>
    </Layout>
  );
}
