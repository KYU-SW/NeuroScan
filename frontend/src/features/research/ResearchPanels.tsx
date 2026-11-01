/**
 * Phase 7-B Research Validation Mode panels. Everything shown comes from the validated research
 * API response: metrics are those of the backend's evaluate_case (never recomputed here), the
 * TP/FP/FN map is the backend's difference labelmap. Wording is a research comparison only.
 */
import type { ResearchCase, ResultComponent } from "../../api/schemas";
import { ko } from "../../i18n/ko";
import { DIFFERENCE_CLASSES, DIFFERENCE_PALETTE, GT_ONLY_COLOR } from "../viewer/differenceLut";
import { MASK_PALETTE, type ViewerEngine } from "../viewer/ViewerEngine";
import { type ResearchMode, useViewerStore } from "../viewer/viewerStore";

/** Difference map first: it is the core of the research comparison (and the default). */
const MODES: ResearchMode[] = ["difference", "gt", "ai", "base"];
const fmt = (v: number, d = 2) => v.toFixed(d);
const AI_COLOR = MASK_PALETTE.orange; // #E69F00 — AI prediction = TP + FP

export function ResearchBanner() {
  return (
    <div className="research-banner" role="note" aria-label={ko.research.bannerTitle}>
      <strong>{ko.research.bannerTitle}</strong> {ko.research.bannerText}
    </div>
  );
}

export function ResearchModeControls({ legend }: { legend: ResearchCase["legend"] }) {
  const s = useViewerStore();
  return (
    <fieldset className="research-modes">
      <legend>{ko.research.modeTitle}</legend>
      <div className="segmented" role="radiogroup" aria-label={ko.research.modeTitle}>
        {MODES.map((m) => (
          <button
            key={m}
            type="button"
            role="radio"
            aria-checked={s.researchMode === m}
            className={s.researchMode === m ? "active" : ""}
            onClick={() => s.setResearchMode(m)}
          >
            {ko.research.modes[m]}
          </button>
        ))}
      </div>
      {s.researchMode === "difference" && (
        <div role="group" aria-label={ko.research.classToggle}>
          {DIFFERENCE_CLASSES.map((cls) => (
            <label className="row" key={cls}>
              <input
                type="checkbox"
                checked={s.differenceClasses[cls]}
                onChange={() => s.toggleDifferenceClass(cls)}
              />
              {legend.find((l) => l.key === cls)?.ko ?? cls}
            </label>
          ))}
        </div>
      )}
      {s.researchMode === "ai" && (
        <>
          <label className="row">
            <input
              type="checkbox"
              checked={s.componentColoring === "component"}
              onChange={(e) => s.setComponentColoring(e.target.checked ? "component" : "single")}
            />
            {ko.research.componentColors}
          </label>
          <p className="note">{ko.research.componentColorsNote}</p>
        </>
      )}
    </fieldset>
  );
}

/** Text band right above the viewer: what the overlay currently shows. */
export function ResearchModeBand() {
  const mode = useViewerStore((s) => s.researchMode);
  return (
    <p className="mode-band" role="status" aria-live="polite">
      {ko.research.bandPrefix} <strong>{ko.research.band[mode]}</strong>
    </p>
  );
}

function Swatch({ color }: { color: string }) {
  return <span className="legend-swatch" aria-hidden="true" style={{ backgroundColor: color }} />;
}

/** Legend for the current mode only; every colour comes with its meaning in text. */
export function ResearchLegend({ legend }: { legend: ResearchCase["legend"] }) {
  const s = useViewerStore();
  let items: { color: string | null; text: string }[];
  if (s.researchMode === "difference") {
    items = legend.map((l) => ({ color: DIFFERENCE_PALETTE[l.key], text: l.ko }));
  } else if (s.researchMode === "gt") {
    items = [{ color: GT_ONLY_COLOR, text: ko.research.gtLegend }];
  } else if (s.researchMode === "ai") {
    items =
      s.componentColoring === "component"
        ? [{ color: null, text: ko.research.componentColorsLegend }]
        : [{ color: AI_COLOR, text: ko.research.aiLegend }];
    items.push({ color: "#8a8a8a", text: ko.research.grayLegend });
  } else {
    items = [{ color: null, text: ko.research.baseLegend }];
  }
  return (
    <div className="legend" aria-label={ko.research.legendTitle}>
      <h4>{ko.research.legendTitle}</h4>
      <ul>
        {items.map((i) => (
          <li key={i.text}>
            {i.color && <Swatch color={i.color} />}
            {i.text}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Component swatch matching the current overlay (research): status colours in the difference
 * map, the AI colour / region colour in AI mode, nothing where AI regions are not drawn. */
export function researchSwatch(data: ResearchCase, mode: ResearchMode) {
  const byRank = new Map(data.matching.prediction.map((p) => [p.size_rank, p]));
  return function swatchFor(c: ResultComponent) {
    if (mode !== "difference")
      return <span className="component-swatch empty" aria-hidden="true" />;
    const m = c.size_rank === null ? undefined : byRank.get(c.size_rank);
    const tp = (m?.tp_voxels ?? 0) > 0;
    const fp = (m?.fp_voxels ?? 0) > 0;
    const text =
      tp && fp ? ko.research.statusMixed : tp ? ko.research.statusTp : ko.research.statusFp;
    return (
      <span className="component-status" title={text}>
        <span className="component-swatch split" aria-hidden="true">
          {tp && <span style={{ backgroundColor: DIFFERENCE_PALETTE.tp }} />}
          {fp && <span style={{ backgroundColor: DIFFERENCE_PALETTE.fp }} />}
        </span>
        <span className="sr-only">{text}</span>
      </span>
    );
  };
}

export function ResearchMetrics({ data }: { data: ResearchCase }) {
  const m = data.metrics;
  const pct = (v: number | null) => (v === null ? ko.research.undefinedNoGt : fmt(v, 4));
  return (
    <section className="card research-metrics" aria-labelledby="research-metrics-title">
      <h3 id="research-metrics-title">{ko.research.metricsTitle}</h3>
      <dl className="kv">
        <dt>{ko.research.dice}</dt>
        <dd>{pct(m.dice)}</dd>
        <dt>{ko.research.iou}</dt>
        <dd>{pct(m.iou)}</dd>
        <dt>{ko.research.sensitivity}</dt>
        <dd>{pct(m.sensitivity)}</dd>
        <dt>{ko.research.precision}</dt>
        <dd>{pct(m.precision)}</dd>
        <dt>{ko.research.gtVolume}</dt>
        <dd>{fmt(m.gt_volume_ml, 3)} mL</dd>
        <dt>{ko.research.aiVolume}</dt>
        <dd>{fmt(m.prediction_volume_ml, 3)} mL</dd>
        {m.has_lesion && (
          <>
            <dt>{ko.research.volumeError}</dt>
            <dd>{fmt(m.volume_error_ml, 3)} mL</dd>
            <dt>{ko.research.absVolumeError}</dt>
            <dd>{fmt(m.absolute_volume_error_ml, 3)} mL</dd>
            <dt>{ko.research.relVolumeError}</dt>
            <dd>
              {m.relative_volume_error_percent === null
                ? ko.research.none
                : `${fmt(m.relative_volume_error_percent, 1)} %`}
            </dd>
          </>
        )}
        <dt>{ko.research.gtComponents}</dt>
        <dd>{m.component_count_gt}</dd>
        <dt>{ko.research.aiComponents}</dt>
        <dd>{m.component_count_prediction}</dd>
        <dt>{ko.research.voxels}</dt>
        <dd>
          {m.tp_voxels} / {m.fp_voxels} / {m.fn_voxels}
        </dd>
        {!m.has_lesion && (
          <>
            <dt>{ko.research.fpVoxels}</dt>
            <dd>{m.predicted_lesion_voxels ?? ko.research.none}</dd>
            <dt>{ko.research.fpVolume}</dt>
            <dd>
              {m.false_positive_volume_ml === null
                ? ko.research.none
                : `${fmt(m.false_positive_volume_ml, 3)} mL`}
            </dd>
            <dt>{ko.research.fpComponents}</dt>
            <dd>{m.false_positive_component_count ?? ko.research.none}</dd>
            <dt>{ko.research.correctNegative}</dt>
            <dd>{m.negative_outcome === "correct_negative" ? ko.research.yes : ko.research.no}</dd>
          </>
        )}
      </dl>
      <p className="note">{ko.research.validationNote}</p>
      {data.reproducibility.mismatched_voxels !== null && (
        <p className="note">{ko.research.reproduced(data.reproducibility.mismatched_voxels)}</p>
      )}
    </section>
  );
}

/** Extra line per AI region for the shared Component Review (research only). */
export function researchComponentExtra(data: ResearchCase) {
  const byRank = new Map(data.matching.prediction.map((p) => [p.size_rank, p]));
  return function Extra(c: ResultComponent) {
    const info = c.size_rank === null ? undefined : byRank.get(c.size_rank);
    if (!info) return null;
    return (
      <span className="research-extra">
        {info.gt_overlap ? ko.research.overlap : ko.research.aiOnly} ·{" "}
        {ko.research.groupTypes[info.group_type]} · {ko.research.tpVoxels} {info.tp_voxels} ·{" "}
        {ko.research.fpVoxelsShort} {info.fp_voxels} · {ko.research.tpRatio} {fmt(info.tp_ratio, 2)}
        {info.ground_truth_component_ids.length > 0 &&
          ` · ${ko.research.linkedGt} ${info.ground_truth_component_ids
            .map(ko.research.gtName)
            .join(", ")}`}
      </span>
    );
  };
}

export function MatchingTable({
  data,
  engine,
}: {
  data: ResearchCase;
  engine: ViewerEngine | null;
}) {
  const gtByRank = new Map(data.ground_truth_components.map((g) => [g.size_rank, g]));
  const selectGt = useViewerStore((st) => st.selectGtComponent);
  const list = (ids: number[], name: (n: number) => string) =>
    ids.length ? ids.map(name).join(", ") : ko.research.none;
  return (
    <details className="card research-matching" open>
      <summary>{ko.research.matchingTitle}</summary>
      <p className="note">{ko.research.matchingNote}</p>
      <table>
        <thead>
          <tr>
            <th scope="col">{ko.research.colType}</th>
            <th scope="col">{ko.research.colAi}</th>
            <th scope="col">{ko.research.colGt}</th>
            <th scope="col">{ko.research.colOverlap}</th>
            <th scope="col">{ko.research.colIou}</th>
            <th scope="col">{ko.research.colAiVolume}</th>
            <th scope="col">{ko.research.colGtVolume}</th>
            <th scope="col">{ko.research.colDistance}</th>
          </tr>
        </thead>
        <tbody>
          {data.matching.groups.map((g) => {
            const firstGt = g.ground_truth_component_ids[0];
            const gt = firstGt === undefined ? undefined : gtByRank.get(firstGt);
            return (
              <tr key={g.group_id} data-group-type={g.group_type}>
                <td>{ko.research.groupTypes[g.group_type]}</td>
                <td>{list(g.prediction_component_ids, ko.research.aiName)}</td>
                <td>
                  {list(g.ground_truth_component_ids, ko.research.gtName)}
                  {gt && engine && (
                    <button
                      type="button"
                      aria-label={ko.research.gtJump(gt.size_rank)}
                      onClick={() => {
                        const [x = 0, y = 0, z = 0] = gt.representative_voxel;
                        selectGt(gt.size_rank);
                        engine.jumpToVoxel([x, y, z]);
                      }}
                    >
                      ⌖
                    </button>
                  )}
                </td>
                <td>{g.overlap_voxel_count}</td>
                <td>{g.iou === null ? ko.research.none : fmt(g.iou, 3)}</td>
                <td>{fmt(g.prediction_volume_ml, 3)} mL</td>
                <td>{fmt(g.gt_volume_ml, 3)} mL</td>
                <td>
                  {g.centroid_distance_mm === null
                    ? ko.research.none
                    : `${fmt(g.centroid_distance_mm, 1)} mm`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </details>
  );
}
