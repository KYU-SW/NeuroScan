import { ko } from "../i18n/ko";
import { ApiError, ApiUnavailableError, ContractError } from "./client";

/** User-facing message: code-mapped text, else the server's own short message. Never stack
 * traces, paths or raw details. */
export function errorMessage(e: unknown): string {
  if (e instanceof ApiUnavailableError) return ko.health.apiDown;
  if (e instanceof ContractError) return ko.errors.contract;
  if (e instanceof ApiError) return ko.errors.byCode[e.code] ?? e.message ?? ko.errors.generic;
  return ko.errors.generic;
}

export function errorCode(e: unknown): string | null {
  return e instanceof ApiError ? e.code : null;
}
