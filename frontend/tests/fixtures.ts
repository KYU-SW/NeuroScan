/** Synthetic API responses shaped like the backend contract (no patient data). */
import type { CaseInfo, RunResult, RunStatus } from "../src/api/schemas";

export const CASE = "case-0123456789ab";
export const RUN = "11111111-2222-4333-8444-555555555555";
export const RUN2 = "66666666-7777-4888-9999-aaaaaaaaaaaa";
const FILE = (n: number) => `0000000${n}-0000-4000-8000-00000000000${n}`;
const SHA = (c: string) => c.repeat(64);

export function caseInfo(runs: CaseInfo["runs"] = []): CaseInfo {
  return {
    case_id: CASE,
    created_at: "2026-10-25T00:00:00Z",
    status: "ready",
    inputs: [
      {
        role: "dwi",
        sha256: SHA("a"),
        size_bytes: 395539,
        raw_retained: true,
        downloadable: false,
      },
      {
        role: "adc",
        sha256: SHA("b"),
        size_bytes: 446482,
        raw_retained: true,
        downloadable: false,
      },
    ],
    runs,
    links: { self: `/api/v1/cases/${CASE}` },
  };
}

export function runStatus(over: Partial<RunStatus> = {}): RunStatus {
  return {
    run_id: RUN,
    case_id: CASE,
    status: "queued",
    current_stage: null,
    progress: 0,
    queue_position: 1,
    created_at: "2026-10-25T00:00:00Z",
    started_at: null,
    completed_at: null,
    warning_count: 0,
    warnings: [],
    error: null,
    model: {
      model_name: "25d_baseline",
      checkpoint_sha256: "ff61bc7f".padEnd(64, "0"),
      config_hash: "34d18df104da",
    },
    links: {},
    ...over,
  };
}

const volume = (id: "dwi" | "adc" | "prediction_mask", n: number, shareable: boolean) => ({
  id,
  role: id === "prediction_mask" ? "label" : "image",
  file_id: FILE(n),
  url: `/api/v1/files/${FILE(n)}`,
  mime_type: "application/gzip",
  sha256: SHA(String(n)),
  shape: [112, 112, 73],
  spacing_mm: [2, 2, 2],
  affine: [
    [-2, 0, 0, 110],
    [0, 2, 0, -110],
    [0, 0, 2, -50],
    [0, 0, 0, 1],
  ],
  orientation: "LAS",
  dtype: id === "prediction_mask" ? "uint8" : "float32",
  shareable,
  labels:
    id === "prediction_mask" ? { "0": "background", "1": "model-predicted lesion region" } : null,
  same_grid_as: id === "prediction_mask" ? "dwi" : null,
});

export function result(over: Partial<RunResult> = {}): RunResult {
  return {
    run_id: RUN,
    case_id: CASE,
    engine_run_id: "20261025T000000Z-0b5fe12c",
    completed_at: "2026-10-25T00:00:05Z",
    model: {
      model_name: "25d_baseline",
      checkpoint_sha256: "ff61bc7f".padEnd(64, "0"),
      config_hash: "34d18df104da",
    },
    lesion: {
      detected: true,
      total_volume_ml: 48.976,
      voxel_count: 6122,
      component_count: 9,
      largest_component_volume_ml: 45.2,
      affected_slice_count: 21,
      prediction_threshold: 0.5,
      space: "original_dwi_grid",
      coordinate_system: "RAS+ scanner world coordinates (mm), voxel-centre convention",
      display_text: null,
    },
    location: {
      bounding_box_voxel: { min: [50, 40, 20], max: [80, 70, 40] },
      bounding_box_world_mm: { min: [-50, -30, -10], max: [10, 30, 30] },
      center_of_mass_voxel: [65, 55, 30],
      center_of_mass_world_mm: [-20, 0, 10],
      hemisphere: {
        status: "estimated",
        estimated: true,
        method: "brain_mask_midline_world_x",
        left_volume_ml: 39.44,
        right_volume_ml: 0.536,
        midline_band_volume_ml: 9.0,
        bilateral: true,
        reason: null,
        caveat:
          "Estimated from the brain-mask midline in scanner (RAS+) coordinates; not registered to an atlas.",
      },
    },
    components: Array.from({ length: 9 }, (_, i) => ({
      label: i + 1,
      voxel_count: 100 - i,
      volume_ml: 5 - i * 0.1,
      centroid_voxel: [60 + i, 50, 30],
      centroid_world_mm: [-10 - i, 0, 10],
      bbox_min_voxel: [60, 50, 30],
      bbox_max_voxel: [61, 51, 31],
      hemisphere_estimate: i % 2 ? "left" : "right",
    })),
    components_truncated: false,
    research_score_summary: {
      kind: "uncalibrated_model_output_score",
      calibrated: false,
      note: "Uncalibrated model output scores; not probabilities of disease.",
      max_score_in_brain: 0.998,
      uncertain_fraction_of_brain: 0.0012,
      predicted_region: {
        mean_score: 0.91,
        median_score: 0.95,
        max_score: 0.998,
        low_score_fraction: 0.05,
      },
    },
    vascular_territory: null,
    vascular_territory_note: "Not estimated: requires atlas registration (not performed).",
    flags: [],
    warnings: [{ code: "W_OBLIQUE", message: "Image axes are oblique." }],
    viewer: {
      coordinate_system: {
        world: "RAS+ (NIfTI sform/qform), millimetres",
        note: "use each volume's affine",
      },
      volumes: [
        volume("dwi", 1, false),
        volume("adc", 2, false),
        volume("prediction_mask", 3, true),
      ],
      views: {
        sagittal: { array_axis: 0, count: 112, axis_increases_toward: "L" },
        coronal: { array_axis: 1, count: 112, axis_increases_toward: "A" },
        axial: { array_axis: 2, count: 73, axis_increases_toward: "S" },
      },
      render_3d: { mask_available: true, empty_mask: false, suggested: "surface extraction" },
    },
    disclaimer: { en: "Research prototype output.", ko: "연구용 결과입니다." },
    ...over,
  };
}

export function noLesionResult(): RunResult {
  const r = result();
  return {
    ...r,
    lesion: {
      ...r.lesion,
      detected: false,
      total_volume_ml: 0,
      voxel_count: 0,
      component_count: 0,
      largest_component_volume_ml: 0,
      affected_slice_count: 0,
      display_text:
        "The model did not mark any lesion region. This does not mean that no lesion is present.",
    },
    location: {
      ...r.location,
      bounding_box_voxel: null,
      bounding_box_world_mm: null,
      center_of_mass_voxel: null,
      center_of_mass_world_mm: null,
    },
    components: [],
    research_score_summary: { ...r.research_score_summary, predicted_region: null },
    viewer: { ...r.viewer, render_3d: { ...r.viewer.render_3d, empty_mask: true } },
  };
}

export function manyComponentsResult(): RunResult {
  const r = result();
  const base = r.components[0];
  if (!base) throw new Error("fixture needs a component");
  return {
    ...r,
    components: Array.from({ length: 100 }, (_, i) => ({ ...base, label: i + 1 })),
    components_truncated: true,
    lesion: { ...r.lesion, component_count: 120 },
    warnings: [
      { code: "W_MANY_COMPONENTS", message: "Predicted region is split into many separate parts." },
      { code: "W_LARGE_LESION", message: "Predicted region is unusually large." },
    ],
  };
}
