import { apiFetch } from "./client";
import {
  CASE_ID,
  caseDeleteSchema,
  caseSchema,
  healthSchema,
  modelInfoSchema,
  runCreateSchema,
  runResultSchema,
  runStatusSchema,
  UUID_ID,
} from "./schemas";

function caseId(id: string): string {
  if (!CASE_ID.test(id)) throw new Error("invalid case id");
  return id;
}

function runId(id: string): string {
  if (!UUID_ID.test(id)) throw new Error("invalid run id");
  return id;
}

export const api = {
  health: (signal?: AbortSignal) =>
    apiFetch("/api/v1/health", healthSchema, { signal: signal ?? null }),
  model: (signal?: AbortSignal) =>
    apiFetch("/api/v1/models/current", modelInfoSchema, { signal: signal ?? null }),
  getCase: (id: string, signal?: AbortSignal) =>
    apiFetch(`/api/v1/cases/${caseId(id)}`, caseSchema, { signal: signal ?? null }),
  deleteCase: (id: string) =>
    apiFetch(`/api/v1/cases/${caseId(id)}?confirm=true`, caseDeleteSchema, { method: "DELETE" }),
  startRun: (id: string, allowRerun = false) =>
    apiFetch(`/api/v1/cases/${caseId(id)}/runs`, runCreateSchema, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ allow_rerun: allowRerun }),
    }),
  getRun: (id: string, signal?: AbortSignal) =>
    apiFetch(`/api/v1/runs/${runId(id)}`, runStatusSchema, { signal: signal ?? null }),
  cancelRun: (id: string) =>
    apiFetch(`/api/v1/runs/${runId(id)}/cancel`, runStatusSchema, { method: "POST" }),
  getResult: (id: string, signal?: AbortSignal) =>
    apiFetch(`/api/v1/runs/${runId(id)}/result`, runResultSchema, { signal: signal ?? null }),
};
