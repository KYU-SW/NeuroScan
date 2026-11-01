# NeuroScan Viewer (Phase 6, research prototype)

> ⚠️ Research prototype for medical image review support — **not a medical device** and no
> diagnosis. Model scores are uncalibrated and are not probabilities of disease. No
> authentication: run on `127.0.0.1` only.

React + TypeScript + Vite + NiiVue viewer for the Phase 5B FastAPI backend: upload a DWI + ADC
pair, run the analysis, follow progress, and review the model-marked lesion region in 2D/3D with
the quantification panel.

## Run (development)

```powershell
# backend (repository root): API + exactly one GPU worker, see backend/README.md
$env:PYTHONPATH = "backend"
.venv\Scripts\python.exe -m neuroscan_api
.venv\Scripts\python.exe -m neuroscan_api.worker

# frontend
cd frontend
npm ci --ignore-scripts
npm run dev        # http://127.0.0.1:5173  (/api is proxied to http://127.0.0.1:8000)
```

Production-like check: `npm run build && npm run preview` (http://127.0.0.1:4173, same proxy,
strict Content-Security-Policy). Quality gates: `npm run typecheck`, `npm run lint`,
`npm run test`, `npm run format`.

## Design rules

- The browser calls only relative `/api/v1/...` paths (Vite proxy, same origin; the backend CORS
  policy is unchanged). File URLs are accepted only when they come from a validated result and
  match `/api/v1/files/<uuid4>`; downloads are SHA-256 verified. Raw uploads are never requested.
- All API responses are validated with strict Zod schemas (unknown keys are rejected).
- NiiVue lives only in `src/features/viewer/NiivueEngine.ts` behind the `ViewerEngine` interface.
  DWI/ADC use linear interpolation; the prediction mask is the NiiVue drawing layer (always
  nearest-neighbour). The mask colour comes from a fixed palette (#E69F00 default, #56B4E9,
  #009E73, #CC79A7) via the public `drawLut` field and `cmapper.makeDrawLut`.
- 3D view: the base volume (DWI or ADC) is shown as a translucent brain background (opacity 0.40,
  on by default, shortcut **B**); the camera starts at the front preset (front / top / left /
  right / default buttons, public `setRenderAzimuthElevation` only) with NiiVue's orientation cube.
- **After any NiiVue version change, re-run the drawing-layer spike**: `npm run dev`, then open
  `http://127.0.0.1:5173/__spike?autorun=1` (all checks must pass).
- File names, paths, header text, tracebacks and form data are never shown or logged.

## Manual smoke checklist (real Chrome/Edge)

1. Upload a validation DWI/ADC pair: only role, format and size are shown (no file name).
2. Run → queued → running stages → result page opens on the lesion-centre slice.
3. Panel volume equals the API value; hemisphere shows the "추정" badge; warnings listed.
4. DWI ↔ ADC, axial/coronal/sagittal/MPR/3D, mask toggle/opacity/palette, radiological ↔
   neurological, keyboard shortcuts.
5. 3D: brain background visible on entry, front view shows both hemispheres, top/left/right
   presets, "기본 시점" returns to front, B toggles the background (checkbox follows).
6. Browser devtools network tab: only `/api/v1/...` requests.
7. Delete the case from the case page (in-page confirmation).
