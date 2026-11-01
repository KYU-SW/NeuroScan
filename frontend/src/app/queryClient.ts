import { QueryClient } from "@tanstack/react-query";

import { ApiError, ContractError } from "../api/client";

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        // client errors and contract violations are not retried; outages back off
        retry: (count, e) =>
          !(e instanceof ContractError) && !(e instanceof ApiError && e.status < 500) && count < 3,
        retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 10_000),
      },
      mutations: { retry: false },
    },
  });
}
