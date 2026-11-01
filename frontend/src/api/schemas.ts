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

export const viewerVolumeSchema = z.strictObject({
  id: z.enum(["dwi", "adc", "prediction_mask"]),
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
  components: z.array(
    z.strictObject({
      label: z.number().int(),
      voxel_count: z.number().int(),
      volume_ml: z.number(),
      centroid_voxel: vec3,
      centroid_world_mm: vec3,
      bbox_min_voxel: z.array(z.number().int()).length(3),
      bbox_max_voxel: z.array(z.number().int()).length(3),
      hemisphere_estimate: z.string(),
    }),
  ),
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
export type ErrorResponse = z.infer<typeof errorResponseSchema>;
