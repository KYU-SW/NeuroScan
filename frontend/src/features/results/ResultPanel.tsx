/**
 * Quantification panel. Uses only RunResultResponse fields (validated by Zod). Wording follows
 * the medical-safety rules: "model-marked lesion region", hemisphere marked as estimated, scores
 * uncalibrated and never shown as percentages, no vascular territory names.
 */
import type { RunResult } from "../../api/schemas";
import { ko } from "../../i18n/ko";
import { ComponentReview } from "../components/ComponentReview";
import type { ViewerEngine } from "../viewer/ViewerEngine";

const fmt = (v: number, d = 2) => v.toFixed(d);
const fmtVec = (v: number[] | null) => (v ? v.map((x) => x.toFixed(1)).join(", ") : "—");
interface Props {
  result: RunResult;
  engine: ViewerEngine | null;
  labelmapAvailable: boolean;
}

export function ResultPanel({ result, engine, labelmapAvailable }: Props) {
  const { lesion, location, research_score_summary: scores } = result;
  const h = location.hemisphere;

  return (
    <section className="result-panel" aria-labelledby="result-title">
      <h2 id="result-title">{ko.result.title}</h2>
      <p className="disclaimer-inline">{ko.disclaimer}</p>

      <div className="card" aria-live="polite">
        {lesion.detected ? (
          <p className="lesion-yes">
            <span aria-hidden="true">●</span> {ko.result.detected}
          </p>
        ) : (
          <p className="lesion-no" role="status">
            {ko.noLesion}
          </p>
        )}
        <dl className="kv">
          <dt>{ko.result.totalVolume}</dt>
          <dd>
            {fmt(lesion.total_volume_ml)} mL <small>({ko.result.gridNote})</small>
          </dd>
          <dt>{ko.result.componentCount}</dt>
          <dd>{lesion.component_count}</dd>
          <dt>{ko.result.largest}</dt>
          <dd>{fmt(lesion.largest_component_volume_ml)} mL</dd>
          <dt>{ko.result.slices}</dt>
          <dd>{lesion.affected_slice_count}</dd>
        </dl>
      </div>

      <div className="card">
        <h3>
          {ko.result.hemisphere}{" "}
          <span className="badge">
            {h.status === "estimated" ? ko.result.estimated : ko.result.notAvailable}
          </span>
        </h3>
        {h.status === "estimated" ? (
          <dl className="kv">
            <dt>{ko.result.left}</dt>
            <dd>{h.left_volume_ml !== null ? `${fmt(h.left_volume_ml)} mL` : "—"}</dd>
            <dt>{ko.result.right}</dt>
            <dd>{h.right_volume_ml !== null ? `${fmt(h.right_volume_ml)} mL` : "—"}</dd>
            <dt>{ko.result.midline}</dt>
            <dd>
              {h.midline_band_volume_ml !== null ? `${fmt(h.midline_band_volume_ml)} mL` : "—"}
            </dd>
            <dt>{ko.result.bilateral}</dt>
            <dd>{h.bilateral ? "예" : "아니오"}</dd>
          </dl>
        ) : (
          <p>{h.reason ?? ko.result.notAvailable}</p>
        )}
        <p className="note">{h.caveat}</p>
      </div>

      <div className="card">
        <h3>{ko.result.warnings}</h3>
        {result.warnings.length === 0 ? (
          <p>{ko.result.noWarnings}</p>
        ) : (
          <ul className="warnings">
            {result.warnings.map((w, i) => (
              <li key={`${w.code}-${i}`}>
                <span aria-hidden="true">⚠</span> <span className="sr-only">경고: </span>
                {w.message} <code>{w.code}</code>
              </li>
            ))}
          </ul>
        )}
      </div>

      <details className="card">
        <summary>{ko.result.location}</summary>
        <dl className="kv">
          <dt>{ko.result.center}</dt>
          <dd>{fmtVec(location.center_of_mass_world_mm)}</dd>
          <dt>{ko.result.bbox}</dt>
          <dd>
            {location.bounding_box_world_mm
              ? `${fmtVec(location.bounding_box_world_mm.min ?? null)} ~ ${fmtVec(location.bounding_box_world_mm.max ?? null)}`
              : "—"}
          </dd>
          <dt>{ko.result.vascular}</dt>
          <dd>{result.vascular_territory_note}</dd>
        </dl>
      </details>

      {result.components.length > 0 && (
        <ComponentReview result={result} engine={engine} labelmapAvailable={labelmapAvailable} />
      )}

      <details className="card scores">
        <summary>{ko.result.scores}</summary>
        <p className="note">{ko.result.scoresNote}</p>
        <dl className="kv">
          {scores.predicted_region && (
            <>
              <dt>{ko.result.meanScore}</dt>
              <dd>{fmt(scores.predicted_region.mean_score, 3)}</dd>
              <dt>{ko.result.medianScore}</dt>
              <dd>{fmt(scores.predicted_region.median_score, 3)}</dd>
              <dt>{ko.result.maxScore}</dt>
              <dd>{fmt(scores.predicted_region.max_score, 3)}</dd>
              <dt>{ko.result.lowScore}</dt>
              <dd>{fmt(scores.predicted_region.low_score_fraction, 3)}</dd>
            </>
          )}
          <dt>{ko.result.maxInBrain}</dt>
          <dd>{scores.max_score_in_brain !== null ? fmt(scores.max_score_in_brain, 3) : "—"}</dd>
          <dt>{ko.result.uncertain}</dt>
          <dd>
            {scores.uncertain_fraction_of_brain !== null
              ? fmt(scores.uncertain_fraction_of_brain, 4)
              : "—"}
          </dd>
        </dl>
        <p className="note">calibrated = {String(scores.calibrated)}</p>
      </details>
    </section>
  );
}
