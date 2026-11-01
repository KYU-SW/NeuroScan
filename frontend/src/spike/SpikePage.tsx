import { useEffect, useRef, useState } from "react";

import type { CrosshairLocation } from "../features/viewer/location";
import type { MaskColorStrategy, NiivueEngine } from "../features/viewer/NiivueEngine";
import type { ViewKind } from "../features/viewer/ViewerEngine";
import { type CheckResult, runSpike } from "./runSpike";
import { parseObliqueParam, runPhase7Spike, startManualSpike } from "./runSpike7";

function freshCanvas(): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  canvas.style.width = "512px";
  canvas.style.height = "512px";
  document.getElementById("spike-host")?.replaceChildren(canvas);
  return canvas;
}

/** Development-only page (/__spike): NiiVue checks on synthetic data.
 * ?suite=p6 (Phase 6 drawing layer) | p7 (Phase 7-0) | default both; ?autorun=1 runs on load;
 * ?oblique=<base64url JSON> adds an oblique geometry (numbers only) to the Phase 7 checks. */
export function SpikePage() {
  const [results, setResults] = useState<CheckResult[]>([]);
  const [status, setStatus] = useState<"idle" | "running" | "done">("idle");
  const [location, setLocation] = useState<CrosshairLocation | null>(null);
  const [manual, setManual] = useState<NiivueEngine | null>(null);
  const params = new URLSearchParams(window.location.search);
  const suite = params.get("suite");

  async function run() {
    setManual(null); // effect cleanup disposes the previous manual engine
    setStatus("running");
    const all: CheckResult[] = [];
    if (suite !== "p7") {
      for (const strategy of ["builtin", "customLut"] as MaskColorStrategy[]) {
        all.push(...(await runSpike(freshCanvas(), strategy)));
      }
    }
    if (suite !== "p6") {
      all.push(...(await runPhase7Spike(freshCanvas, parseObliqueParam(params.get("oblique")))));
    }
    setResults(all);
    setStatus("done");
  }

  async function startManual() {
    setManual(null);
    setManual(await startManualSpike(freshCanvas(), setLocation));
  }

  const act = (fn: (e: NiivueEngine) => void) => () => {
    if (manual) fn(manual);
  };
  const views: ViewKind[] = ["axial", "coronal", "sagittal", "multiplanar", "render"];

  // ?autorun=1 starts the checks on load (headless Chrome smoke without a mouse)
  const started = useRef(false);
  useEffect(() => {
    if (started.current || !params.has("autorun")) return;
    started.current = true;
    void run();
  });
  useEffect(() => () => manual?.dispose(), [manual]);

  const passed = results.filter((r) => r.pass).length;
  return (
    <main style={{ padding: 16, fontFamily: "system-ui", color: "#111" }}>
      <h1>NiiVue spike (합성 데이터) — Phase 6 drawing layer · Phase 7-0</h1>
      <button type="button" onClick={run} disabled={status === "running"}>
        spike 실행
      </button>{" "}
      <button type="button" onClick={startManual} disabled={status === "running"}>
        Phase 7-0 수동 확인 (component 4개)
      </button>
      <p data-spike-status={status}>
        상태: {status} {status === "done" ? `— ${passed}/${results.length} 통과` : ""}
      </p>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", margin: "6px 0" }}>
        {views.map((v) => (
          <button key={v} type="button" onClick={act((e) => e.setView(v))}>
            {v}
          </button>
        ))}
        <button
          type="button"
          onClick={act((e) => e.setComponentDisplay({ selected: null, hidden: new Set() }))}
        >
          전체 표시
        </button>
        {[1, 2, 3, 255].map((l) => (
          <button
            key={`s${l}`}
            type="button"
            onClick={act((e) => e.setComponentDisplay({ selected: l, hidden: new Set() }))}
          >
            {l} 선택
          </button>
        ))}
        <button
          type="button"
          onClick={act((e) => e.setComponentDisplay({ selected: null, hidden: new Set([3]) }))}
        >
          3 숨김
        </button>
        <button type="button" onClick={act((e) => e.jumpToVoxel([48, 23, 9]))}>
          label 3로 이동
        </button>
        <button type="button" onClick={act((e) => e.setRender3DBackground(false))}>
          3D 배경 끔
        </button>
      </div>
      <p data-testid="spike-location">
        {location
          ? `vox ${location.vox.join(",")} · mm ${location.mm.map((v) => v.toFixed(1)).join(",")} · slider(sag,cor,ax) ${location.slice.join(",")}`
          : "위치: (수동 확인 시작 후 canvas 클릭)"}
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
