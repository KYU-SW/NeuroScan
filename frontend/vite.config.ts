/// <reference types="vitest/config" />
// Phase 6 viewer: Vite dev/preview bound to 127.0.0.1; /api is proxied to the local FastAPI
// (same origin for the browser, so the backend CORS policy stays unchanged).
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const API_TARGET = "http://127.0.0.1:8000";
const proxy = { "/api": { target: API_TARGET, changeOrigin: true } };

// Production builds get a strict Content-Security-Policy. The dev server is excluded because
// the React refresh preamble is an inline script.
const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "connect-src 'self'",
  "img-src 'self' data: blob:",
  "worker-src 'self' blob:",
  "style-src 'self' 'unsafe-inline'",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'self'",
].join("; ");

function cspForBuild(): Plugin {
  return {
    name: "neuroscan-csp",
    apply: "build",
    transformIndexHtml(html) {
      return html.replace(
        "<!-- CSP -->",
        `<meta http-equiv="Content-Security-Policy" content="${CSP}" />`,
      );
    },
  };
}

export default defineConfig({
  plugins: [react(), cspForBuild()],
  server: { host: "127.0.0.1", port: 5173, strictPort: true, proxy },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true, proxy },
  build: { sourcemap: false },
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
    restoreMocks: true,
  },
});
