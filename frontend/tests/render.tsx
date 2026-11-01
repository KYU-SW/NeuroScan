import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { vi } from "vitest";

import { AppRoutes } from "../src/app/App";
import { EngineProvider } from "../src/features/viewer/engineContext";
import { FakeViewerEngine } from "../src/features/viewer/FakeViewerEngine";

export function LocationProbe() {
  const loc = useLocation();
  return <output data-testid="location">{loc.pathname}</output>;
}

export interface Harness {
  engines: FakeViewerEngine[];
}

/** Render the app routes at `path` with a fresh QueryClient, fake viewer engine and WebGL flag. */
export function renderApp(path: string, opts: { webgl?: boolean } = {}): Harness {
  const engines: FakeViewerEngine[] = [];
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false, refetchOnWindowFocus: false },
      mutations: { retry: false },
    },
  });
  const env = {
    createEngine: () => {
      const e = new FakeViewerEngine();
      engines.push(e);
      return e;
    },
    isWebGL2Supported: () => opts.webgl ?? true,
  };
  const ui: ReactNode = (
    <QueryClientProvider client={client}>
      <EngineProvider value={env}>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path="*" element={<AppRoutes />} />
          </Routes>
          <LocationProbe />
        </MemoryRouter>
      </EngineProvider>
    </QueryClientProvider>
  );
  render(ui);
  return { engines };
}

type Handler = (url: string, init: RequestInit | undefined) => Response | Promise<Response>;

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export function apiError(
  status: number,
  code: string,
  details: Record<string, unknown> = {},
): Response {
  return json(
    { error: { code, message: `${code} message`, request_id: "r1", retryable: false, details } },
    status,
  );
}

/** Route fetch calls by "METHOD path" prefix; records every requested URL. */
export function mockFetch(routes: Record<string, Handler>) {
  const calls: string[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const method = (init?.method ?? "GET").toUpperCase();
    calls.push(`${method} ${url}`);
    const key = Object.keys(routes)
      .sort((a, b) => b.length - a.length)
      .find((k) => `${method} ${url}`.startsWith(k));
    if (!key)
      return json({
        status: "degraded",
        database: "ok",
        worker: "online",
        queue_depth: 0,
        api_version: "t",
        disclaimer: { en: "", ko: "" },
      });
    const handler = routes[key];
    if (!handler) throw new Error("no handler");
    return handler(url, init);
  });
  vi.stubGlobal("fetch", fn);
  return { calls, fn };
}

export const HEALTH_OK = {
  status: "ok",
  database: "ok",
  worker: "online",
  queue_depth: 0,
  api_version: "0.1.0",
  disclaimer: { en: "x", ko: "x" },
};
