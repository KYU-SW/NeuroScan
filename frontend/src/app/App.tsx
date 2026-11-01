import { QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";

import { CasePage } from "../pages/CasePage";
import { NotFoundPage } from "../pages/NotFoundPage";
import { ResultViewerPage } from "../pages/ResultViewerPage";
import { RunProgressPage } from "../pages/RunProgressPage";
import { UploadPage } from "../pages/UploadPage";
import { SpikePage } from "../spike/SpikePage";
import { createQueryClient } from "./queryClient";

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/cases/:caseId" element={<CasePage />} />
      <Route path="/runs/:runId" element={<RunProgressPage />} />
      <Route path="/runs/:runId/viewer" element={<ResultViewerPage />} />
      {import.meta.env.DEV && <Route path="/__spike" element={<SpikePage />} />}
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}

export function App() {
  const [client] = useState(createQueryClient);
  return (
    <QueryClientProvider client={client}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
