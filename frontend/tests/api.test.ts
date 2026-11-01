import { describe, expect, it } from "vitest";

import {
  ApiError,
  apiFetch,
  ApiUnavailableError,
  ContractError,
  fetchApiFile,
  isApiFileUrl,
  isSafeApiPath,
} from "../src/api/client";
import { api } from "../src/api/endpoints";
import { healthSchema, runResultSchema } from "../src/api/schemas";
import { result } from "./fixtures";
import { apiError, HEALTH_OK, json, mockFetch } from "./render";

describe("API client", () => {
  it("accepts only relative /api/v1 paths", () => {
    expect(isSafeApiPath("/api/v1/health")).toBe(true);
    for (const bad of [
      "http://evil.example/api/v1/health",
      "//evil.example/api/v1/x",
      "file:///C:/x.nii.gz",
      "/api/v1/../../etc",
      "/api/v1/files\\x",
      "/other",
      "javascript:alert(1)",
    ])
      expect(isSafeApiPath(bad)).toBe(false);
  });

  it("refuses non-API file URLs before any request", async () => {
    const { fn } = mockFetch({});
    expect(isApiFileUrl("/api/v1/files/00000001-0000-4000-8000-000000000001")).toBe(true);
    for (const bad of [
      "https://cdn.example/x.nii.gz",
      "file:///x.nii.gz",
      "/api/v1/files/../x",
      "/api/v1/files/1",
    ])
      await expect(fetchApiFile(bad)).rejects.toThrow("blocked");
    await expect(apiFetch("https://evil.example/api/v1/health", healthSchema)).rejects.toThrow(
      "blocked",
    );
    expect(fn).not.toHaveBeenCalled();
  });

  it("sends credentials: omit and validates responses", async () => {
    const { fn } = mockFetch({ "GET /api/v1/health": () => json(HEALTH_OK) });
    await expect(api.health()).resolves.toMatchObject({ worker: "online" });
    const init = fn.mock.calls[0]?.[1];
    expect(init?.credentials).toBe("omit");
  });

  it("rejects responses with unexpected keys (strict contract)", async () => {
    mockFetch({ "GET /api/v1/health": () => json({ ...HEALTH_OK, local_path: "C:\\Users\\x" }) });
    await expect(api.health()).rejects.toBeInstanceOf(ContractError);
    const leaky = { ...result(), checkpoint_path: "C:/models/best.pt" };
    expect(runResultSchema.safeParse(leaky).success).toBe(false);
    expect(runResultSchema.safeParse(result()).success).toBe(true);
  });

  it("maps HTTP errors to ApiError and outages to ApiUnavailableError", async () => {
    mockFetch({ "GET /api/v1/runs/": () => apiError(404, "RUN_NOT_FOUND") });
    const err = await api.getRun("11111111-2222-4333-8444-555555555555").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err instanceof ApiError && err.code).toBe("RUN_NOT_FOUND");

    mockFetch({ "GET /api/v1/health": () => new Response("Bad gateway", { status: 502 }) });
    await expect(api.health()).rejects.toBeInstanceOf(ApiUnavailableError);
    mockFetch({
      "GET /api/v1/health": () => {
        throw new TypeError("Failed to fetch");
      },
    });
    await expect(api.health()).rejects.toBeInstanceOf(ApiUnavailableError);
  });

  it("validates ids before building request paths", () => {
    expect(() => api.getCase("../../etc")).toThrow();
    expect(() => api.getRun("not-a-uuid")).toThrow();
  });
});
