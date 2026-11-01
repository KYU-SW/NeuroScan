/**
 * Downloads the three viewer volumes named by a validated run result (de-identified DWI/ADC copies
 * and the prediction mask) and verifies each SHA-256 against the result. Only API file URLs are
 * accepted; raw uploads have no URL in the API contract and are never requested.
 */
import { useQueries } from "@tanstack/react-query";
import { useMemo } from "react";

import { fetchApiFile, isApiFileUrl, sha256Hex } from "../../api/client";
import type { ViewerVolume } from "../../api/schemas";
import type { VolumeBuffers } from "./ViewerEngine";

export class IntegrityError extends Error {
  constructor() {
    super("volume integrity check failed");
    this.name = "IntegrityError";
  }
}

export async function downloadVerified(
  v: ViewerVolume,
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  if (!isApiFileUrl(v.url)) throw new Error("blocked non-API file URL");
  const buf = await fetchApiFile(v.url, signal);
  if ((await sha256Hex(buf)) !== v.sha256) throw new IntegrityError();
  return buf;
}

const IDS = ["dwi", "adc", "prediction_mask"] as const;

export function useVolumes(volumes: ViewerVolume[] | undefined) {
  const picked = IDS.map((id) => volumes?.find((v) => v.id === id));
  const results = useQueries({
    queries: picked.map((v, i) => ({
      queryKey: ["volume", v?.file_id ?? `missing-${i}`, v?.sha256 ?? ""],
      queryFn: ({ signal }: { signal: AbortSignal }) => {
        if (!v) throw new Error("volume missing from result");
        return downloadVerified(v, signal);
      },
      enabled: v !== undefined,
      staleTime: Infinity,
      gcTime: 5 * 60_000,
      retry: false,
    })),
  });
  const [dwi, adc, mask] = results.map((r) => r.data);
  // stable identity: the viewer engine is rebuilt only when the bytes themselves change
  const buffers = useMemo<VolumeBuffers | null>(
    () => (dwi && adc && mask ? { dwi, adc, mask } : null),
    [dwi, adc, mask],
  );
  const error = results.find((r) => r.error)?.error ?? null;
  const done = results.filter((r) => r.data).length;
  return { buffers, error, done, total: IDS.length };
}
