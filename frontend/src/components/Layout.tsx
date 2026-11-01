import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import { ApiError, ApiUnavailableError } from "../api/client";
import { api } from "../api/endpoints";
import { ko } from "../i18n/ko";

export function HealthBanner() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => api.health(signal),
    refetchInterval: 10_000,
    retry: false,
  });
  let text: string | null = null;
  let level: "error" | "warn" = "warn";
  if (health.error) {
    text = ko.health.apiDown;
    level = "error";
  } else if (health.data && health.data.worker !== "online") {
    text = {
      offline: ko.health.workerOffline,
      stale: ko.health.workerStale,
      degraded: ko.health.workerDegraded,
      loading: ko.health.workerLoading,
    }[health.data.worker];
  }
  if (!text) return null;
  return (
    <div className={`banner banner-${level}`} role={level === "error" ? "alert" : "status"}>
      <span aria-hidden="true">{level === "error" ? "⛔" : "⚠"}</span> {text}
    </div>
  );
}

export function ModelInfo() {
  const model = useQuery({
    queryKey: ["model"],
    queryFn: ({ signal }) => api.model(signal),
    retry: false,
    staleTime: 60_000,
  });
  if (model.error) {
    const msg =
      model.error instanceof ApiError || model.error instanceof ApiUnavailableError
        ? "모델 정보 없음 (worker 대기)"
        : "모델 정보 없음";
    return <span className="model-info">{msg}</span>;
  }
  const m = model.data;
  if (!m) return null;
  return (
    <details className="model-info">
      <summary>
        {m.model_name ?? "모델"} · {m.checkpoint_sha256?.slice(0, 8) ?? "—"}
      </summary>
      <div className="popover" role="note">
        <p>
          {ko.result.modelHelp}: {m.model_type} · 입력 {m.input_channels}채널 · context{" "}
          {m.context_slices} · threshold {m.threshold}
        </p>
        {m.validation_positive_mean_dice !== null && (
          <p>
            Validation positive mean Dice {m.validation_positive_mean_dice.toFixed(3)} —{" "}
            {ko.result.validationNote}
          </p>
        )}
      </div>
    </details>
  );
}

export function Layout(props: { caseId?: string; children: ReactNode }) {
  return (
    <div className="app">
      <header className="app-header">
        <Link to="/" className="brand">
          {ko.appName}
        </Link>
        {props.caseId && <span className="case-id">Case {props.caseId}</span>}
        <ModelInfo />
        <span className="badge badge-research">{ko.researchBadge}</span>
        <span className="localhost">{ko.localhostOnly}</span>
      </header>
      <HealthBanner />
      <main className="app-main">{props.children}</main>
      <footer className="app-footer">
        <p>{ko.disclaimer}</p>
        <p>
          {ko.noLesionGeneral} {ko.notMedicalDevice}
        </p>
      </footer>
    </div>
  );
}
