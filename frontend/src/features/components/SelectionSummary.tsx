/**
 * Research Viewer selection summary (Phase 7-B UX, research pages only).
 *
 * Shows, above the viewer, what is selected (an AI region or a ground-truth region), its
 * research comparison from the validated API response, and buttons for the review flow. The
 * buttons only change the display (base image, overlay mode, view) or move the crosshair to the
 * region's representative voxel; nothing is recomputed and the view is never changed
 * automatically by selecting.
 */
import type { ResearchCase } from "../../api/schemas";
import { ko } from "../../i18n/ko";
import type { ViewerEngine } from "../viewer/ViewerEngine";
import { useViewerStore } from "../viewer/viewerStore";
import { displaySliceRange, type GridVolume, isSmallRegion, targetVoxel } from "./reviewModel";

const SIDE: Record<string, string> = {
  left: ko.result.left,
  right: ko.result.right,
  midline: ko.result.midline,
  not_available: ko.result.notAvailable,
};
/** Regions this small are hard to make out in 3D whatever the size ratio (3 x 3 x 3 voxels). */
const TINY_VOXELS = 27;

interface Props {
  data: ResearchCase;
  base: GridVolume | undefined;
  engine: ViewerEngine | null;
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </>
  );
}

export function SelectionSummary({ data, base, engine }: Props) {
  const s = useViewerStore();
  const ai =
    s.selectedComponent === null
      ? undefined
      : data.prediction_components.find((c) => c.display_label === s.selectedComponent);
  const gt =
    s.selectedGtComponent === null
      ? undefined
      : data.ground_truth_components.find((g) => g.size_rank === s.selectedGtComponent);
  if (!ai && !gt) return null;

  const fmt = (v: number, d = 3) => v.toFixed(d);
  const range = (r: { first: number; last: number } | null) =>
    r ? (r.first === r.last ? `${r.first}` : `${r.first}–${r.last}`) : ko.research.none;
  let title = "";
  let rows: { k: string; v: string }[] = [];
  let small = false;
  let voxel: [number, number, number] | null = null;

  if (ai) {
    const m = data.matching.prediction.find((p) => p.size_rank === ai.size_rank);
    const largest = data.prediction_components[0]?.volume_ml ?? 0;
    small = isSmallRegion(ai, largest) || ai.voxel_count < TINY_VOXELS;
    voxel = base ? targetVoxel(ai, base) : null;
    title = ko.research.selectedAi(ai.size_rank ?? 0);
    const overlap = !m
      ? ko.research.none
      : m.tp_ratio >= 1
        ? ko.research.overlapAll
        : m.tp_ratio > 0
          ? ko.research.overlapPartial
          : ko.research.overlapNone;
    rows = [
      { k: ko.research.volume, v: `${fmt(ai.volume_ml)} mL` },
      { k: ko.research.slices, v: range(base ? displaySliceRange(ai, base) : null) },
      {
        k: ko.research.location,
        // "환자 좌측 추정" — but a side that could not be estimated is shown as such, alone
        v:
          ai.hemisphere_estimate === "not_available"
            ? ko.result.notAvailable
            : ko.research.locationEstimated(SIDE[ai.hemisphere_estimate] ?? ai.hemisphere_estimate),
      },
      { k: ko.research.gtComparison, v: overlap },
      {
        k: ko.research.matchingType,
        v: m ? (ko.research.groupTypes[m.group_type] ?? m.group_type) : ko.research.none,
      },
      {
        k: ko.research.linkedGtLabel,
        v:
          m && m.ground_truth_component_ids.length
            ? m.ground_truth_component_ids.map(ko.research.gtName).join(", ")
            : ko.research.none,
      },
      { k: ko.research.tpFp, v: m ? `${m.tp_voxels} / ${m.fp_voxels}` : ko.research.none },
      { k: ko.research.tpRatio, v: m ? fmt(m.tp_ratio, 2) : ko.research.none },
    ];
  } else if (gt) {
    const m = data.matching.ground_truth.find((g) => g.size_rank === gt.size_rank);
    small = gt.voxel_count < TINY_VOXELS;
    const [x = 0, y = 0, z = 0] = gt.representative_voxel;
    voxel = [x, y, z];
    title = ko.research.selectedGt(gt.size_rank);
    const n = base?.shape[gt.slice_range.axis] ?? 1;
    const flip = base?.orientation[gt.slice_range.axis] === "I";
    const a = flip ? n - 1 - gt.slice_range.first : gt.slice_range.first;
    const b = flip ? n - 1 - gt.slice_range.last : gt.slice_range.last;
    rows = [
      { k: ko.research.volume, v: `${fmt(gt.volume_ml)} mL` },
      { k: ko.research.slices, v: range({ first: Math.min(a, b) + 1, last: Math.max(a, b) + 1 }) },
      {
        k: ko.research.gtComparison,
        v:
          !m || m.prediction_component_ids.length === 0
            ? ko.research.gtOnlyRegion
            : ko.research.gtDetected(m.prediction_component_ids.map(ko.research.aiName).join(", ")),
      },
      {
        k: ko.research.matchingType,
        v: m ? (ko.research.groupTypes[m.group_type] ?? m.group_type) : ko.research.none,
      },
      {
        k: ko.research.linkedAiLabel,
        v:
          m && m.prediction_component_ids.length
            ? m.prediction_component_ids.map(ko.research.aiName).join(", ")
            : ko.research.noLinkedAi,
      },
      {
        k: ko.research.detectedMissed,
        v: m ? `${m.detected_voxels} / ${m.missed_voxels}` : ko.research.none,
      },
    ];
  }

  const noHighlight =
    ai && s.researchMode !== "ai" ? ko.research.noHighlight[s.researchMode] : null;
  const show3dNote = s.view === "render" || small;
  const checkMpr = () => {
    s.setView("multiplanar");
    if (engine && voxel) engine.jumpToVoxel(voxel); // keep the representative voxel
  };

  return (
    <section className="selection-summary" aria-label={ko.research.summaryTitle}>
      <h3>{title}</h3>
      <dl className="kv">
        {rows.map((r) => (
          <Row key={r.k} k={r.k} v={r.v} />
        ))}
      </dl>
      {small && ai && <span className="badge neutral">{ko.components.small}</span>}
      {noHighlight && (
        <p className="note" role="status">
          {noHighlight}
        </p>
      )}
      {show3dNote && (
        <p className="note" role="status">
          {ko.research.smallIn3d}
        </p>
      )}
      <div className="row wrap" role="group" aria-label={ko.research.summaryTitle}>
        <button type="button" aria-pressed={s.base === "dwi"} onClick={() => s.setBase("dwi")}>
          {ko.research.showDwi}
        </button>
        <button type="button" aria-pressed={s.base === "adc"} onClick={() => s.setBase("adc")}>
          {ko.research.showAdc}
        </button>
        <button
          type="button"
          aria-pressed={s.researchMode === "difference"}
          onClick={() => s.setResearchMode("difference")}
        >
          {ko.research.showDifference}
        </button>
        <button
          type="button"
          aria-pressed={s.researchMode === "ai"}
          disabled={!ai}
          title={ai ? undefined : ko.research.gtNoAiHighlight}
          onClick={() => s.setResearchMode("ai")}
        >
          {ko.research.showInAi}
        </button>
        <button type="button" disabled={!engine || !voxel} onClick={checkMpr}>
          {ko.research.checkMpr}
        </button>
        <button type="button" onClick={s.clearSelection}>
          {ko.research.clearSelection}
        </button>
      </div>
      <p className="note review-hint">{ko.research.reviewHint}</p>
    </section>
  );
}
