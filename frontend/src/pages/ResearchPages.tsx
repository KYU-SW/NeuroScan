/**
 * Phase 7-B Research Validation Mode pages (/research, /research/cases/:alias).
 *
 * Separate from the general upload / result screens: own API contract (researchCaseSchema),
 * alias identifiers only, a banner that is always visible, and the TP/FP/FN overlays from the
 * backend's difference labelmap. When research mode is off the API answers 404 and these pages
 * only say that the mode is not available.
 */
import { useQuery } from "@tanstack/react-query";
import { type KeyboardEvent, useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { ApiError } from "../api/client";
import { api } from "../api/endpoints";
import { errorMessage } from "../api/errors";
import { RESEARCH_ALIAS, type ResearchCase } from "../api/schemas";
import { Layout } from "../components/Layout";
import { ComponentReview } from "../features/components/ComponentReview";
import { SelectionSummary } from "../features/components/SelectionSummary";
import {
  MatchingTable,
  ResearchBanner,
  researchComponentExtra,
  ResearchLegend,
  ResearchMetrics,
  ResearchModeBand,
  ResearchModeControls,
  researchSwatch,
} from "../features/research/ResearchPanels";
import { useEngineEnvironment } from "../features/viewer/engineContext";
import { handleViewerKey } from "../features/viewer/keyboard";
import { IntegrityError, useVolumes } from "../features/viewer/useVolumes";
import { ViewerCanvas } from "../features/viewer/ViewerCanvas";
import { ViewerControls } from "../features/viewer/ViewerControls";
import type { ViewerEngine } from "../features/viewer/ViewerEngine";
import { useViewerStore } from "../features/viewer/viewerStore";
import { ko } from "../i18n/ko";
import { NotFoundPage } from "./NotFoundPage";

function unavailable(error: unknown): string {
  return error instanceof ApiError && error.status === 404
    ? ko.research.unavailable
    : errorMessage(error);
}

export function ResearchListPage() {
  const list = useQuery({
    queryKey: ["research-cases"],
    queryFn: ({ signal }) => api.listResearchCases(signal),
    retry: false,
  });
  return (
    <Layout>
      <ResearchBanner />
      <section className="panel narrow">
        <h1>{ko.research.listTitle}</h1>
        {list.isPending && <p>{ko.research.loading}</p>}
        {list.error && (
          <p className="error" role="alert">
            {unavailable(list.error)}
          </p>
        )}
        {list.data && list.data.cases.length === 0 && <p>{ko.research.empty}</p>}
        {list.data && list.data.cases.length > 0 && (
          <ul className="research-list">
            {list.data.cases.map((c) => (
              <li key={c.alias}>
                <Link to={`/research/cases/${c.alias}`} aria-label={ko.research.open(c.alias)}>
                  {c.alias}
                </Link>{" "}
                · {c.dataset_id} · Validation
              </li>
            ))}
          </ul>
        )}
      </section>
    </Layout>
  );
}

export function ResearchCasePage() {
  const { alias = "" } = useParams();
  if (!RESEARCH_ALIAS.test(alias)) return <NotFoundPage />;
  return <ResearchCaseView alias={alias} />;
}

function ResearchCaseView({ alias }: { alias: string }) {
  const detail = useQuery({
    queryKey: ["research-case", alias],
    queryFn: ({ signal }) => api.getResearchCase(alias, signal),
    staleTime: Infinity,
    retry: false,
  });
  const reset = useViewerStore((s) => s.reset);
  const enterResearch = useViewerStore((s) => s.enterResearch);
  // fresh state per case, then the research defaults (difference map, orange AI prediction)
  useEffect(() => {
    reset();
    enterResearch();
  }, [alias, reset, enterResearch]);
  return (
    <Layout>
      {/* research-page owns the remaining viewport height (see app.css "research layout") */}
      <div className="research-page">
        <ResearchBanner />
        {detail.isPending && <p className="panel">{ko.research.loading}</p>}
        {detail.error && (
          <section className="panel narrow">
            <p className="error" role="alert">
              {unavailable(detail.error)}
            </p>
            <Link to="/research">{ko.research.listTitle}</Link>
          </section>
        )}
        {detail.data && <ResearchLayout data={detail.data} />}
      </div>
    </Layout>
  );
}

function ResearchLayout({ data }: { data: ResearchCase }) {
  const env = useEngineEnvironment();
  const webgl = useMemo(() => env.isWebGL2Supported(), [env]);
  const volumes = useVolumes(webgl ? data.volumes : undefined);
  const [engine, setEngine] = useState<ViewerEngine | null>(null);
  const [viewerError, setViewerError] = useState<string | null>(null);
  const state = useViewerStore();
  const base = data.volumes.find((v) => v.id === "dwi");
  const count = (name: string) => {
    const n = data.views[name]?.["count"];
    return typeof n === "number" ? n : 1;
  };
  const sliceCounts = {
    axial: count("axial"),
    coronal: count("coronal"),
    sagittal: count("sagittal"),
  };
  // open on the largest ground-truth region (else the largest AI region): voxel-based
  const start =
    data.ground_truth_components[0]?.representative_voxel ??
    data.prediction_components[0]?.representative_voxel ??
    null;
  const [sx, sy, sz] = start ?? [-1, -1, -1];
  const startVoxel: [number, number, number] | null =
    start && sx !== undefined && sy !== undefined && sz !== undefined ? [sx, sy, sz] : null;
  const onEngine = useCallback(
    (e: ViewerEngine | null) => {
      if (e && sx !== undefined && sx >= 0 && sy !== undefined && sz !== undefined)
        e.jumpToVoxel([sx, sy, sz]);
      setEngine(e);
    },
    [sx, sy, sz],
  );
  const onError = useCallback((m: string) => setViewerError(m), []);
  const extra = researchComponentExtra(data);

  function onKey(e: KeyboardEvent<HTMLDivElement>) {
    if (handleViewerKey(e.key, state, engine, startVoxel)) e.preventDefault();
  }

  let center;
  if (!webgl)
    center = (
      <p className="viewer-message" role="status">
        {ko.viewer.webglUnsupported}
      </p>
    );
  else if (volumes.error)
    center = (
      <p className="viewer-message error" role="alert">
        {volumes.error instanceof IntegrityError
          ? ko.viewer.integrityFailed
          : errorMessage(volumes.error)}
      </p>
    );
  else if (viewerError)
    center = (
      <p className="viewer-message error" role="alert">
        {viewerError}
      </p>
    );
  else if (!volumes.buffers)
    center = (
      <p className="viewer-message" role="status">
        {ko.viewer.loading} ({volumes.done}/{volumes.total})
      </p>
    );
  else
    center = (
      <ViewerCanvas
        buffers={volumes.buffers}
        description={`${ko.research.bannerTitle} ${data.alias} · ${ko.research.modes[state.researchMode]}`}
        onEngine={onEngine}
        onError={onError}
      />
    );

  return (
    <>
      <dl className="research-info kv">
        <dt>{ko.research.alias}</dt>
        <dd>{data.alias}</dd>
        <dt>{ko.research.dataset}</dt>
        <dd>
          {data.dataset_id} · {ko.research.split} Validation
        </dd>
        <dt>{ko.research.model}</dt>
        <dd>
          {data.model.model_name ?? ko.research.none} · {data.model.config_hash ?? ""}
        </dd>
        <dt>{ko.research.threshold}</dt>
        <dd>{data.model.threshold}</dd>
        <dt>{ko.research.preprocessing}</dt>
        <dd>{data.model.preprocessing_version ?? ko.research.none}</dd>
      </dl>
      {/* research-layout: research-only grid (the general result page keeps result-layout) */}
      <div className="research-layout">
        <aside className="left" aria-label="Viewer 조작">
          <ResearchModeControls legend={data.legend} />
          <ViewerControls
            engine={engine}
            sliceCounts={sliceCounts}
            lesionVoxel={startVoxel}
            labelmapAvailable={state.researchMode === "ai"}
            emptyMask={data.metrics.prediction_voxels === 0}
            hideMaskStyle
          />
        </aside>
        <section
          className="center research-center"
          tabIndex={0}
          onKeyDown={onKey}
          aria-label="영상 Viewer (키보드 조작 가능)"
        >
          <ResearchModeBand />
          <ResearchLegend legend={data.legend} />
          <SelectionSummary data={data} base={base} engine={engine} />
          {/* viewer-slot: a definite box for the canvas. The canvas host is absolutely positioned
              inside it, so the canvas never sizes the layout (no ResizeObserver feedback loop). */}
          <div className="viewer-slot">{center}</div>
        </section>
        <aside className="right">
          <ResearchMetrics data={data} />
          {data.prediction_components.length > 0 && (
            <ComponentReview
              components={data.prediction_components}
              truncated={false}
              largestMl={data.prediction_components[0]?.volume_ml ?? 0}
              base={base}
              engine={engine}
              labelmapAvailable
              extra={extra}
              overlayControls={{
                enabled: state.researchMode === "ai",
                reason:
                  state.researchMode === "difference"
                    ? ko.research.controlsOnlyInAi
                    : ko.research.noHighlight[state.researchMode],
                action:
                  state.researchMode === "difference" ? (
                    <button type="button" onClick={() => state.setResearchMode("ai")}>
                      {ko.research.showInAi}
                    </button>
                  ) : undefined,
              }}
              swatchFor={
                state.researchMode === "ai" ? undefined : researchSwatch(data, state.researchMode)
              }
            />
          )}
          <MatchingTable data={data} engine={engine} />
        </aside>
      </div>
    </>
  );
}
