import { useEffect, useRef, useState } from "react";

import type { MaskColorStrategy } from "../features/viewer/NiivueEngine";
import { type CheckResult, runSpike } from "./runSpike";

/** Development-only page (/__spike): runs the NiiVue drawing-layer checks on synthetic data. */
export function SpikePage() {
  const [results, setResults] = useState<CheckResult[]>([]);
  const [status, setStatus] = useState<"idle" | "running" | "done">("idle");

  async function run() {
    setStatus("running");
    const all: CheckResult[] = [];
    for (const strategy of ["builtin", "customLut"] as MaskColorStrategy[]) {
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 512;
      canvas.style.width = "512px";
      canvas.style.height = "512px";
      const host = document.getElementById("spike-host");
      host?.replaceChildren(canvas);
      all.push(...(await runSpike(canvas, strategy)));
    }
    setResults(all);
    setStatus("done");
  }

  // ?autorun=1 starts the checks on load (headless Chrome smoke without a mouse)
  const started = useRef(false);
  useEffect(() => {
    if (started.current || !new URLSearchParams(window.location.search).has("autorun")) return;
    started.current = true;
    void run();
  });

  const passed = results.filter((r) => r.pass).length;
  return (
    <main style={{ padding: 16, fontFamily: "system-ui", color: "#111" }}>
      <h1>NiiVue drawing-layer spike (합성 데이터)</h1>
      <button type="button" onClick={run} disabled={status === "running"}>
        spike 실행
      </button>
      <p data-spike-status={status}>
        상태: {status} {status === "done" ? `— ${passed}/${results.length} 통과` : ""}
      </p>
      <div id="spike-host" style={{ width: 512, height: 512, background: "#000" }} />
      <table>
        <tbody>
          {results.map((r) => (
            <tr key={r.id}>
              <td>{r.pass ? "PASS" : "FAIL"}</td>
              <td>{r.id}</td>
              <td>{r.label}</td>
              <td>{r.detail}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}
