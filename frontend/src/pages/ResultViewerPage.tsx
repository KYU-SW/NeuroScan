import { useQuery } from "@tanstack/react-query";
import { type KeyboardEvent, useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { isApiFileUrl } from "../api/client";
import { api } from "../api/endpoints";
import { errorMessage } from "../api/errors";
import { type RunResult, UUID_ID } from "../api/schemas";
import { Layout } from "../components/Layout";
import { ResultPanel } from "../features/results/ResultPanel";
import { useEngineEnvironment } from "../features/viewer/engineContext";
import { handleViewerKey } from "../features/viewer/keyboard";
import { IntegrityError, useVolumes } from "../features/viewer/useVolumes";
import { ViewerCanvas } from "../features/viewer/ViewerCanvas";
import { ViewerControls } from "../features/viewer/ViewerControls";
import type { ViewerEngine } from "../features/viewer/ViewerEngine";
import { useViewerStore } from "../features/viewer/viewerStore";
import { ko } from "../i18n/ko";
import { NotFoundPage } from "./NotFoundPage";

export function ResultViewerPage() {
  const { runId = "" } = useParams();
  if (!UUID_ID.test(runId)) return <NotFoundPage />;
  return <ResultView runId={runId} />;
}

function ResultView({ runId }: { runId: string }) {
  const result = useQuery({
    queryKey: ["result", runId],
    queryFn: ({ signal }) => api.getResult(runId, signal),
    staleTime: Infinity,
    retry: false,
  });
  const reset = useViewerStore((s) => s.reset);
  useEffect(() => reset(), [runId, reset]);

  return (
    <Layout caseId={result.data?.case_id}>
      {result.isPending && <p className="panel">결과를 불러오는 중</p>}
      {result.error && (
        <section className="panel narrow">
          <p className="error" role="alert">
            {errorMessage(result.error)}
          </p>
          <Link to={`/runs/${runId}`}>{ko.run.title}</Link>
        </section>
      )}
      {result.data && <ResultLayout result={result.data} />}
    </Layout>
  );
}

function ResultLayout({ result }: { result: RunResult }) {
  const env = useEngineEnvironment();
  const webgl = useMemo(() => env.isWebGL2Supported(), [env]);
  const volumes = useVolumes(webgl ? result.viewer.volumes : undefined);
  const [engine, setEngine] = useState<ViewerEngine | null>(null);
  const [viewerError, setViewerError] = useState<string | null>(null);
  const state = useViewerStore();
  const views = result.viewer.views;
  const sliceCounts = {
    axial: views.axial?.count ?? 1,
    coronal: views.coronal?.count ?? 1,
    sagittal: views.sagittal?.count ?? 1,
  };
  const c = result.location.center_of_mass_world_mm;
  const lesionCenter: [number, number, number] | null = c
    ? [c[0] ?? 0, c[1] ?? 0, c[2] ?? 0]
    : null;
  const mask = result.viewer.volumes.find((v) => v.id === "prediction_mask");
  const [cx, cy, cz] = lesionCenter ?? [Number.NaN, Number.NaN, Number.NaN];
  // when the model marked a region, open the viewer on its centre slice (API centre of mass)
  const onEngine = useCallback(
    (e: ViewerEngine | null) => {
      if (e && Number.isFinite(cx)) e.jumpToWorldMm([cx, cy, cz]);
      setEngine(e);
    },
    [cx, cy, cz],
  );
  const onError = useCallback((m: string) => setViewerError(m), []);
  const description = ko.viewer.canvasLabel(
    ko.viewer.views[state.view],
    state.base.toUpperCase(),
    result.lesion.total_volume_ml.toFixed(2),
    result.lesion.component_count,
  );

  function onKey(e: KeyboardEvent<HTMLDivElement>) {
    if (handleViewerKey(e.key, state, engine, lesionCenter)) e.preventDefault();
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
        description={description}
        onEngine={onEngine}
        onError={onError}
      />
    );

  return (
    <div className="result-layout">
      <aside className="left" aria-label="Viewer 조작">
        <ViewerControls
          engine={engine}
          sliceCounts={sliceCounts}
          lesionCenter={lesionCenter}
          emptyMask={result.viewer.render_3d.empty_mask}
        />
        {mask?.shareable && isApiFileUrl(mask.url) && (
          <a className="download" href={mask.url} download="prediction_mask.nii.gz">
            {ko.viewer.download}
          </a>
        )}
        <p className="note">{ko.viewer.shortcuts}</p>
      </aside>
      <section
        className="center"
        tabIndex={0}
        onKeyDown={onKey}
        aria-label="영상 Viewer (키보드 조작 가능)"
      >
        {center}
        {state.view === "render" && result.viewer.render_3d.empty_mask && (
          <p className="viewer-overlay" role="status">
            {ko.viewer.emptyMask3d}
          </p>
        )}
      </section>
      <aside className="right">
        <ResultPanel result={result} onJump={engine ? (mm) => engine.jumpToWorldMm(mm) : null} />
      </aside>
    </div>
  );
}
