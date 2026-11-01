/**
 * Phase 7-A Connected Component Review. Selecting a region moves the viewer to the region's
 * representative voxel (engine.jumpToVoxel; the sliders follow through the engine's location
 * events) and highlights it; the others are dimmed (grey mix, 2D and 3D alike). Hiding removes a
 * region from the display only. Selection/hiding live in browser memory: the prediction files,
 * the volumes and every number in the result stay unchanged.
 */
import { useState } from "react";

import type { ResultComponent, RunResult } from "../../api/schemas";
import { ko } from "../../i18n/ko";
import { componentColor, OTHER_LABEL } from "../viewer/labelLut";
import { MASK_PALETTE, type ViewerEngine } from "../viewer/ViewerEngine";
import { useViewerStore } from "../viewer/viewerStore";
import { displaySliceRange, isSmallRegion, targetVoxel, visibleComponents } from "./reviewModel";

const SIDE: Record<string, string> = {
  left: ko.result.left,
  right: ko.result.right,
  midline: ko.result.midline,
  not_available: ko.result.notAvailable,
};
const PAGE = 10;

interface Props {
  result: RunResult;
  engine: ViewerEngine | null;
  /** The result has the component labelmap (colours, highlighting and hiding need it). */
  labelmapAvailable: boolean;
}

export function ComponentReview({ result, engine, labelmapAvailable }: Props) {
  const s = useViewerStore();
  const [showAll, setShowAll] = useState(false);
  const base = result.viewer.volumes.find((v) => v.id === "dwi");
  const largest = result.lesion.largest_component_volume_ml;
  const labels = result.components.flatMap((c) =>
    c.display_label === null ? [] : [c.display_label],
  );
  const list = visibleComponents(
    result.components,
    s.componentSort,
    s.componentFilter,
    s.selectedComponent,
    s.hiddenComponents,
  );
  const shown = showAll ? list : list.slice(0, PAGE);
  const swatch = (c: ResultComponent) =>
    s.componentColoring === "single" || c.display_label === null
      ? MASK_PALETTE[s.maskColor]
      : `rgb(${componentColor(c.display_label).join(",")})`;

  function select(c: ResultComponent) {
    if (labelmapAvailable && c.display_label !== null) s.selectComponent(c.display_label);
    if (engine && base) engine.jumpToVoxel(targetVoxel(c, base));
  }

  return (
    <details className="card components-review" open>
      <summary>
        {ko.components.title} ({result.components.length})
      </summary>
      {labelmapAvailable ? (
        <>
          <p className="note">{ko.components.dimNote}</p>
          <p className="note">
            {ko.components.dimMeaning} {ko.components.hideNote}
          </p>
        </>
      ) : (
        <p className="note">{ko.components.noLabelmap}</p>
      )}
      {result.components_truncated && <p className="note">{ko.result.truncated}</p>}

      <div className="row wrap">
        <label>
          {ko.components.sort}{" "}
          <select
            value={s.componentSort}
            onChange={(e) =>
              s.setComponentSort(e.target.value === "smallest" ? "smallest" : "largest")
            }
          >
            <option value="largest">{ko.components.sortLargest}</option>
            <option value="smallest">{ko.components.sortSmallest}</option>
          </select>
        </label>
        <label>
          {ko.components.filter}{" "}
          <select
            value={s.componentFilter}
            onChange={(e) =>
              s.setComponentFilter(
                e.target.value === "selected"
                  ? "selected"
                  : e.target.value === "visible"
                    ? "visible"
                    : "all",
              )
            }
          >
            <option value="all">{ko.components.filterAll}</option>
            <option value="selected">{ko.components.filterSelected}</option>
            <option value="visible">{ko.components.filterVisible}</option>
          </select>
        </label>
      </div>
      {labelmapAvailable && (
        <div className="row wrap">
          <button type="button" disabled={s.selectedComponent === null} onClick={s.clearSelection}>
            {ko.components.clearSelection}
          </button>
          <button
            type="button"
            disabled={s.selectedComponent === null}
            onClick={() => s.selectedComponent !== null && s.showOnly(s.selectedComponent, labels)}
          >
            {ko.components.showOnly}
          </button>
          <button
            type="button"
            disabled={s.hiddenComponents.size === 0}
            onClick={s.showAllComponents}
          >
            {ko.components.showAll}
          </button>
        </div>
      )}

      {shown.length === 0 && s.componentFilter === "selected" ? (
        <p className="note">{ko.components.noneSelected}</p>
      ) : (
        <ol className="components" aria-label={ko.components.title}>
          {shown.map((c) => {
            // pre-7-A results have no size_rank: the API list is already largest first
            const rank = c.size_rank ?? result.components.indexOf(c) + 1;
            const dl = c.display_label;
            const selected = labelmapAvailable && dl !== null && dl === s.selectedComponent;
            const hidden = labelmapAvailable && dl !== null && s.hiddenComponents.has(dl);
            const range = base ? displaySliceRange(c, base) : null;
            return (
              <li
                key={c.label}
                className={`component${selected ? " selected" : ""}${hidden ? " hidden" : ""}`}
                aria-current={selected ? "true" : undefined}
              >
                <span
                  className="component-swatch"
                  style={{ backgroundColor: swatch(c) }}
                  aria-hidden="true"
                />
                <span className="component-text">
                  <strong>{ko.components.region(rank)}</strong> · {c.volume_ml.toFixed(3)} mL
                  {range && ` · ${ko.components.slices(range.first, range.last)}`} ·{" "}
                  {SIDE[c.hemisphere_estimate] ?? c.hemisphere_estimate}
                  {dl === OTHER_LABEL && ` · ${ko.components.other}`}
                  {isSmallRegion(c, largest) && (
                    <span className="badge neutral" title={ko.components.smallHint}>
                      {ko.components.small}
                    </span>
                  )}
                  {hidden && <span className="badge neutral">{ko.components.hiddenTag}</span>}
                </span>
                <span className="component-actions">
                  <button
                    type="button"
                    aria-pressed={labelmapAvailable ? selected : undefined}
                    aria-label={ko.components.selectLabel(rank)}
                    disabled={!engine || !base}
                    onClick={() => select(c)}
                  >
                    {selected ? ko.components.selected : ko.components.select}
                  </button>
                  {labelmapAvailable && dl !== null && (
                    <button
                      type="button"
                      aria-label={
                        hidden ? ko.components.showLabel(rank) : ko.components.hideLabel(rank)
                      }
                      onClick={() => (hidden ? s.showComponent(dl) : s.hideComponent(dl))}
                    >
                      {hidden ? ko.components.show : ko.components.hide}
                    </button>
                  )}
                </span>
              </li>
            );
          })}
        </ol>
      )}
      {!showAll && list.length > PAGE && (
        <button type="button" onClick={() => setShowAll(true)}>
          {ko.result.more}
        </button>
      )}
    </details>
  );
}
