/**
 * Zod schemas mirroring backend/neuroscan_api/schemas.py (Phase 5B). Every object is strict:
 * an unexpected key (e.g. a path or a forbidden field) fails validation instead of reaching the UI.
 */
import { z } from "zod";

const ISO = z.string();
const SHA256 = z.string().regex(/^[0-9a-f]{64}$/);
export const CASE_ID = /^case-[0-9a-f]{12}$/;
export const UUID_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
export const FILE_URL =
  /^\/api\/v1\/files\/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

const caseId = z.string().regex(CASE_ID);
const uuid = z.string().regex(UUID_ID);
const links = z.record(z.string(), z.string());

export const disclaimerSchema = z.strictObject({ en: z.string(), ko: z.string() });

export const errorResponseSchema = z.strictObject({
  error: z.strictObject({
    code: z.string(),
    message: z.string(),
    request_id: z.string().nullable(),
    retryable: z.boolean(),
    details: z.record(z.string(), z.unknown()),
  }),
});

export const healthSchema = z.strictObject({
  status: z.enum(["ok", "degraded"]),
  database: z.literal("ok"),
  worker: z.enum(["online", "offline", "stale", "degraded", "loading"]),
  queue_depth: z.number().int(),
  api_version: z.string(),
  disclaimer: disclaimerSchema,
});

export const modelInfoSchema = z.strictObject({
  model_name: z.string().nullable(),
  model_type: z.string().nullable(),
  training_run_id: z.string().nullable(),
  checkpoint_sha256: z.string().nullable(),
  config_hash: z.string().nullable(),
  input_channels: z.number().nullable(),
  context_slices: z.number().nullable(),
  threshold: z.number().nullable(),
  device: z.string().nullable(),
  validation_positive_mean_dice: z.number().nullable(),
  validation_note: z.string(),
  loaded_at: z.string().nullable(),
  worker_state: z.string(),
});

const runStatusEnum = z.enum(["queued", "running", "succeeded", "failed", "cancelled"]);

export const caseSchema = z.strictObject({
  case_id: caseId,
  created_at: ISO,
  status: z.string(),
  inputs: z.array(
    z.strictObject({
      role: z.enum(["dwi", "adc"]),
      sha256: SHA256,
      size_bytes: z.number().int(),
      raw_retained: z.boolean(),
      downloadable: z.boolean(),
    }),
  ),
  runs: z.array(
    z.strictObject({
      run_id: uuid,
      status: z.string(),
      created_at: ISO,
      completed_at: ISO.nullable(),
    }),
  ),
  links,
});

export const caseDeleteSchema = z.strictObject({
  case_id: caseId,
  deleted: z.boolean(),
  runs_deleted: z.number().int(),
  files_deleted: z.number().int(),
});

const runModelSchema = z.strictObject({
  model_name: z.string().nullable(),
  checkpoint_sha256: z.string().nullable(),
  config_hash: z.string().nullable(),
});
const warningSchema = z.strictObject({ code: z.string(), message: z.string() });

export const runCreateSchema = z.strictObject({
  run_id: uuid,
  case_id: caseId,
  status: z.literal("queued"),
  queue_position: z.number().int(),
  worker_online: z.boolean(),
  links,
});

export const runStatusSchema = z.strictObject({
  run_id: uuid,
  case_id: caseId,
  status: runStatusEnum,
  current_stage: z.string().nullable(),
  progress: z.number().min(0).max(1),
  queue_position: z.number().int().nullable(),
  created_at: ISO,
  started_at: ISO.nullable(),
  completed_at: ISO.nullable(),
  warning_count: z.number().int(),
  warnings: z.array(warningSchema),
  error: z
    .strictObject({ code: z.string(), message: z.string(), retryable: z.boolean() })
    .nullable(),
  model: runModelSchema,
  links,
});

const vec3 = z.array(z.number()).length(3);
const voxel3 = z.array(z.number().int().nonnegative()).length(3);

/** Phase 7-A fields are additive and null for results made before the component labelmap. */
export const componentSchema = z.strictObject({
  label: z.number().int(),
  voxel_count: z.number().int(),
  volume_ml: z.number(),
  centroid_voxel: vec3,
  centroid_world_mm: vec3,
  bbox_min_voxel: z.array(z.number().int()).length(3),
  bbox_max_voxel: z.array(z.number().int()).length(3),
  hemisphere_estimate: z.string(),
  display_label: z.number().int().min(1).max(255).nullable(),
  size_rank: z.number().int().min(1).nullable(),
  representative_voxel: voxel3.nullable(),
  representative_world_mm: vec3.nullable(),
  slice_range: z
    .strictObject({
      axis: z.number().int().min(0).max(2),
      first: z.number().int().nonnegative(),
      last: z.number().int().nonnegative(),
    })
    .nullable(),
});

export const viewerVolumeSchema = z.strictObject({
  id: z.enum(["dwi", "adc", "prediction_mask", "prediction_components"]),
  role: z.string(),
  file_id: uuid,
  url: z.string().regex(FILE_URL),
  mime_type: z.string(),
  sha256: SHA256,
  shape: z.array(z.number().int()).length(3),
  spacing_mm: vec3,
  affine: z.array(z.array(z.number()).length(4)).length(4),
  orientation: z.string().regex(/^[LRAPSI]{3}$/),
  dtype: z.string(),
  shareable: z.boolean(),
  labels: z.record(z.string(), z.string()).nullable().optional(),
  same_grid_as: z.string().nullable().optional(),
});

export const runResultSchema = z.strictObject({
  run_id: uuid,
  case_id: caseId,
  engine_run_id: z.string(),
  completed_at: ISO.nullable(),
  model: runModelSchema,
  lesion: z.strictObject({
    detected: z.boolean(),
    total_volume_ml: z.number(),
    voxel_count: z.number().int(),
    component_count: z.number().int(),
    largest_component_volume_ml: z.number(),
    affected_slice_count: z.number().int(),
    prediction_threshold: z.number(),
    space: z.string(),
    coordinate_system: z.string(),
    display_text: z.string().nullable(),
  }),
  location: z.strictObject({
    bounding_box_voxel: z.record(z.string(), z.array(z.number().int())).nullable(),
    bounding_box_world_mm: z.record(z.string(), z.array(z.number())).nullable(),
    center_of_mass_voxel: vec3.nullable(),
    center_of_mass_world_mm: vec3.nullable(),
    hemisphere: z.strictObject({
      status: z.string(),
      estimated: z.boolean(),
      method: z.string().nullable(),
      left_volume_ml: z.number().nullable(),
      right_volume_ml: z.number().nullable(),
      midline_band_volume_ml: z.number().nullable(),
      bilateral: z.boolean().nullable(),
      reason: z.string().nullable(),
      caveat: z.string(),
    }),
  }),
  components: z.array(componentSchema),
  components_truncated: z.boolean(),
  research_score_summary: z.strictObject({
    kind: z.string(),
    calibrated: z.boolean(),
    note: z.string(),
    max_score_in_brain: z.number().nullable(),
    uncertain_fraction_of_brain: z.number().nullable(),
    predicted_region: z
      .strictObject({
        mean_score: z.number(),
        median_score: z.number(),
        max_score: z.number(),
        low_score_fraction: z.number(),
      })
      .nullable(),
  }),
  vascular_territory: z.null(),
  vascular_territory_note: z.string(),
  flags: z.array(z.string()),
  warnings: z.array(warningSchema),
  viewer: z.strictObject({
    coordinate_system: z.record(z.string(), z.string()),
    volumes: z.array(viewerVolumeSchema),
    views: z.record(
      z.string(),
      z.strictObject({
        array_axis: z.number().int(),
        count: z.number().int(),
        axis_increases_toward: z.string(),
      }),
    ),
    render_3d: z.strictObject({
      mask_available: z.boolean(),
      empty_mask: z.boolean(),
      suggested: z.string(),
    }),
  }),
  disclaimer: disclaimerSchema,
});

export type Health = z.infer<typeof healthSchema>;
export type ModelInfo = z.infer<typeof modelInfoSchema>;
export type CaseInfo = z.infer<typeof caseSchema>;
export type RunCreate = z.infer<typeof runCreateSchema>;
export type RunStatus = z.infer<typeof runStatusSchema>;
export type RunResult = z.infer<typeof runResultSchema>;
export type ViewerVolume = z.infer<typeof viewerVolumeSchema>;
export type ResultComponent = z.infer<typeof componentSchema>;
export type ErrorResponse = z.infer<typeof errorResponseSchema>;

// --- Phase 7-B research validation (own contract; never mixed with RunResult) ---------------

export const RESEARCH_ALIAS = /^val-\d{3}$/;
const nullableNumber = z.number().nullable();
const groupType = z.enum([
  "one_to_one",
  "split",
  "merge",
  "complex",
  "prediction_only",
  "ground_truth_only",
]);
const sliceRange = z.strictObject({
  axis: z.number().int().min(0).max(2),
  first: z.number().int().nonnegative(),
  last: z.number().int().nonnegative(),
});

export const researchCaseListSchema = z.strictObject({
  cases: z.array(
    z.strictObject({
      alias: z.string().regex(RESEARCH_ALIAS),
      dataset_id: z.string(),
      split: z.literal("val"),
      has_ground_truth: z.boolean(),
      created_at: z.string(),
    }),
  ),
  notice: z.string(),
});

export const researchVolumeSchema = z.strictObject({
  id: z.enum(["dwi", "adc", "prediction_mask", "prediction_components", "difference"]),
  role: z.string(),
  file_id: uuid,
  url: z.string().regex(FILE_URL),
  mime_type: z.string(),
  sha256: SHA256,
  shape: z.array(z.number().int()).length(3),
  spacing_mm: vec3,
  affine: z.array(z.array(z.number()).length(4)).length(4),
  orientation: z.string().regex(/^[LRAPSI]{3}$/),
  dtype: z.string(),
});

export const researchCaseSchema = z.strictObject({
  alias: z.string().regex(RESEARCH_ALIAS),
  dataset_id: z.string(),
  split: z.literal("val"),
  created_at: z.string(),
  model: z.strictObject({
    model_name: z.string().nullable(),
    checkpoint_sha256: z.string().nullable(),
    config_hash: z.string().nullable(),
    threshold: z.number(),
    preprocessing_version: z.string().nullable(),
  }),
  metrics: z.strictObject({
    has_lesion: z.boolean(),
    gt_voxels: z.number().int(),
    prediction_voxels: z.number().int(),
    tp_voxels: z.number().int(),
    fp_voxels: z.number().int(),
    fn_voxels: z.number().int(),
    dice: nullableNumber,
    iou: nullableNumber,
    sensitivity: nullableNumber,
    precision: nullableNumber,
    gt_volume_ml: z.number(),
    prediction_volume_ml: z.number(),
    volume_error_ml: z.number(),
    absolute_volume_error_ml: z.number(),
    relative_volume_error_percent: nullableNumber,
    component_count_gt: z.number().int(),
    component_count_prediction: z.number().int(),
    negative_outcome: z.enum(["correct_negative", "false_positive_case"]).nullable(),
    predicted_lesion_voxels: z.number().int().nullable(),
    predicted_lesion_volume_ml: nullableNumber,
    any_false_positive: z.boolean().nullable(),
    false_positive_volume_ml: nullableNumber,
    false_positive_component_count: z.number().int().nullable(),
  }),
  prediction_components: z.array(componentSchema),
  ground_truth_components: z.array(
    z.strictObject({
      size_rank: z.number().int().min(1),
      voxel_count: z.number().int(),
      volume_ml: z.number(),
      centroid_world_mm: vec3,
      representative_voxel: voxel3,
      representative_world_mm: vec3,
      slice_range: sliceRange,
    }),
  ),
  matching: z.strictObject({
    method: z.string(),
    groups: z.array(
      z.strictObject({
        group_id: z.number().int(),
        group_type: groupType,
        prediction_component_ids: z.array(z.number().int()),
        ground_truth_component_ids: z.array(z.number().int()),
        overlap_voxel_count: z.number().int(),
        union_voxel_count: z.number().int(),
        iou: nullableNumber,
        prediction_volume_ml: z.number(),
        gt_volume_ml: z.number(),
        centroid_distance_mm: nullableNumber,
      }),
    ),
    prediction: z.array(
      z.strictObject({
        size_rank: z.number().int(),
        gt_overlap: z.boolean(),
        group_id: z.number().int(),
        group_type: groupType,
        tp_voxels: z.number().int(),
        fp_voxels: z.number().int(),
        tp_ratio: z.number(),
        ground_truth_component_ids: z.array(z.number().int()),
      }),
    ),
    ground_truth: z.array(
      z.strictObject({
        size_rank: z.number().int(),
        detected_voxels: z.number().int(),
        missed_voxels: z.number().int(),
        group_id: z.number().int(),
        group_type: groupType,
        prediction_component_ids: z.array(z.number().int()),
      }),
    ),
  }),
  volumes: z.array(researchVolumeSchema),
  views: z.record(z.string(), z.record(z.string(), z.unknown())),
  legend: z.array(
    z.strictObject({
      label: z.number().int(),
      key: z.enum(["tp", "fp", "fn"]),
      color: z.string().regex(/^#[0-9A-Fa-f]{6}$/),
      ko: z.string(),
      en: z.string(),
    }),
  ),
  gt_only_color: z.string().regex(/^#[0-9A-Fa-f]{6}$/),
  reproducibility: z.strictObject({
    reference: z.string().nullable(),
    identical: z.boolean().nullable(),
    mismatched_voxels: z.number().int().nullable(),
  }),
  notice: z.string(),
  disclaimer: z.strictObject({ en: z.string(), ko: z.string() }),
});

export type ResearchCaseList = z.infer<typeof researchCaseListSchema>;
export type ResearchCase = z.infer<typeof researchCaseSchema>;
export type ResearchVolume = z.infer<typeof researchVolumeSchema>;
export type GroupType = z.infer<typeof groupType>;
