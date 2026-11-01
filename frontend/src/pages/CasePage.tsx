import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError } from "../api/client";
import { api } from "../api/endpoints";
import { errorMessage } from "../api/errors";
import { CASE_ID } from "../api/schemas";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Layout } from "../components/Layout";
import { ko } from "../i18n/ko";
import { NotFoundPage } from "./NotFoundPage";

const STATUS_LABELS: Record<string, string> = {
  queued: ko.run.queued,
  running: ko.run.running,
  succeeded: ko.run.succeeded,
  failed: ko.run.failed,
  cancelled: ko.run.cancelled,
};

function statusLabel(status: string): string {
  return STATUS_LABELS[status] ?? status;
}

export function CasePage() {
  const { caseId = "" } = useParams();
  if (!CASE_ID.test(caseId)) return <NotFoundPage />;
  return <CaseView caseId={caseId} />;
}

function CaseView({ caseId }: { caseId: string }) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [existingRun, setExistingRun] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const info = useQuery({
    queryKey: ["case", caseId],
    queryFn: ({ signal }) => api.getCase(caseId, signal),
  });

  const start = useMutation({
    mutationFn: (allowRerun: boolean) => api.startRun(caseId, allowRerun),
    onSuccess: (run) => navigate(`/runs/${run.run_id}`),
    onError: (e) => {
      if (e instanceof ApiError && e.code === "RUN_ALREADY_ACTIVE") {
        const id = e.detail("run_id");
        if (id) navigate(`/runs/${id}`);
      } else if (e instanceof ApiError && e.code === "RESULT_EXISTS") {
        setExistingRun(e.detail("run_id"));
      }
    },
  });
  const remove = useMutation({
    mutationFn: () => api.deleteCase(caseId),
    onSuccess: () => {
      qc.removeQueries({ queryKey: ["case", caseId] });
      navigate("/", { replace: true });
    },
  });

  const startError =
    start.error &&
    !(
      start.error instanceof ApiError &&
      ["RUN_ALREADY_ACTIVE", "RESULT_EXISTS"].includes(start.error.code)
    )
      ? errorMessage(start.error)
      : null;

  return (
    <Layout caseId={caseId}>
      <section className="panel narrow">
        <h1>
          {ko.case.title} <code>{caseId}</code>
        </h1>
        {info.isPending && <p>불러오는 중</p>}
        {info.error && (
          <p className="error" role="alert">
            {errorMessage(info.error)}
          </p>
        )}
        {info.data && (
          <>
            <h2>{ko.case.inputs}</h2>
            <ul>
              {info.data.inputs.map((i) => (
                <li key={i.role}>
                  {i.role.toUpperCase()} · SHA-256 {i.sha256.slice(0, 12)}… ·{" "}
                  {(i.size_bytes / 1024).toFixed(0)} KiB
                </li>
              ))}
            </ul>
            <h2>{ko.case.runs}</h2>
            {info.data.runs.length === 0 ? (
              <p>—</p>
            ) : (
              <ul>
                {info.data.runs.map((r) => (
                  <li key={r.run_id}>
                    <Link
                      to={
                        r.status === "succeeded" ? `/runs/${r.run_id}/viewer` : `/runs/${r.run_id}`
                      }
                    >
                      {r.created_at} · {statusLabel(r.status)}
                    </Link>
                  </li>
                ))}
              </ul>
            )}
            <div className="row">
              <button
                type="button"
                className="primary"
                disabled={start.isPending}
                onClick={() => start.mutate(false)}
              >
                {start.isPending ? ko.case.starting : ko.case.start}
              </button>
              <button
                type="button"
                className="danger"
                disabled={remove.isPending}
                onClick={() => setConfirmDelete(true)}
              >
                {ko.case.delete}
              </button>
            </div>
            {existingRun && (
              <div className="card" role="status">
                <p>{ko.case.resultExists}</p>
                <div className="row">
                  <button type="button" onClick={() => navigate(`/runs/${existingRun}/viewer`)}>
                    {ko.case.viewExisting}
                  </button>
                  <button type="button" onClick={() => start.mutate(true)}>
                    {ko.case.rerun}
                  </button>
                </div>
              </div>
            )}
            {startError && (
              <p className="error" role="alert">
                {startError}
              </p>
            )}
            {remove.error && (
              <p className="error" role="alert">
                {errorMessage(remove.error)}
              </p>
            )}
          </>
        )}
        <ConfirmDialog
          open={confirmDelete}
          title={ko.case.deleteConfirmTitle}
          body={ko.case.deleteConfirmBody}
          confirmLabel={ko.case.deleteConfirm}
          cancelLabel={ko.case.deleteCancel}
          onCancel={() => setConfirmDelete(false)}
          onConfirm={() => {
            setConfirmDelete(false);
            remove.mutate();
          }}
        />
      </section>
    </Layout>
  );
}
