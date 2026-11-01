/**
 * Same-origin API client. Only relative "/api/v1/..." paths are allowed (served by the Vite
 * proxy); absolute, protocol-relative, file:// and other-origin URLs are rejected before any
 * request. Responses are validated with strict Zod schemas. Server error details are reduced to
 * code / message / request id; nothing is logged.
 */
import type { z } from "zod";

import { errorResponseSchema, FILE_URL } from "./schemas";

export const API_PREFIX = "/api/v1/";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly requestId: string | null,
    readonly retryable: boolean,
    readonly details: Record<string, unknown>,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** A string detail (e.g. details.run_id), or null when absent / not a string. */
  detail(key: string): string | null {
    const v = this.details[key];
    return typeof v === "string" ? v : null;
  }
}

/** The API could not be reached (server down, proxy error, network failure). */
export class ApiUnavailableError extends Error {
  constructor() {
    super("api unavailable");
    this.name = "ApiUnavailableError";
  }
}

/** Response violated the expected contract (unknown field, wrong type). */
export class ContractError extends Error {
  constructor() {
    super("unexpected response");
    this.name = "ContractError";
  }
}

export function isSafeApiPath(path: string): boolean {
  return (
    path.startsWith(API_PREFIX) &&
    !path.includes("//") &&
    !path.includes("\\") &&
    !path.includes("..") &&
    !/^[a-z][a-z0-9+.-]*:/i.test(path)
  );
}

export function isApiFileUrl(url: string): boolean {
  return FILE_URL.test(url);
}

export function toApiError(status: number, body: unknown): ApiError {
  const parsed = errorResponseSchema.safeParse(body);
  if (parsed.success) {
    const e = parsed.data.error;
    return new ApiError(status, e.code, e.message, e.request_id, e.retryable, e.details);
  }
  return new ApiError(status, "HTTP_ERROR", "요청을 처리하지 못했습니다.", null, status >= 500, {});
}

export async function apiFetch<S extends z.ZodType>(
  path: string,
  schema: S,
  init: RequestInit = {},
): Promise<z.infer<S>> {
  if (!isSafeApiPath(path)) throw new Error("blocked non-API path");
  let res: Response;
  try {
    res = await fetch(path, { ...init, credentials: "omit", cache: "no-store" });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiUnavailableError();
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    if (res.status === 502 || res.status === 504) throw new ApiUnavailableError(); // proxy: API down
    throw toApiError(res.status, body);
  }
  const parsed = schema.safeParse(body);
  if (!parsed.success) throw new ContractError();
  return parsed.data;
}

/** Download an API file (relative URL from a validated result) as bytes. */
export async function fetchApiFile(url: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  if (!isApiFileUrl(url)) throw new Error("blocked non-API file URL");
  let res: Response;
  try {
    res = await fetch(url, { credentials: "omit", cache: "no-store", signal: signal ?? null });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiUnavailableError();
  }
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      body = null;
    }
    throw toApiError(res.status, body);
  }
  return res.arrayBuffer();
}

export async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
