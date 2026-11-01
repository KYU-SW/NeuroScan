# Phase 6 — React 2D·3D Viewer (NiiVue MVP)

작성 2026-10-25 · 기준 커밋 `c605e35` (Phase 5B) · 상태: MVP 구현 완료 (미커밋) · **Test Split 미사용**

> ⚠️ 연구 및 의료영상 검토 보조용 프로토타입이며 의료기기가 아닙니다. 인증 없는 개발 버전으로
> localhost에서만 사용합니다.

## 1. 확정 결정 (2026-10-25 승인)

| ID | 결정 |
|---|---|
| F1 | Viewer는 NiiVue. vtk.js(고급 3D)·Cornerstone3D(DICOM/PACS)는 후속 |
| F2 | Vite dev/preview proxy `/api` → `http://127.0.0.1:8000`, Vite는 127.0.0.1 바인딩. 프론트는 상대 경로 `/api/v1/...`만 호출. FastAPI CORS 무수정 |
| F3 | Viewer 조작 상태 Zustand, 서버 상태 TanStack Query, 업로드 File·진행률은 component local state |
| F4 | ~~3D는 기본 mask만~~ → **변경(2026-10-25, A안)**: 3D 뇌 배경(기준 영상 반투명, opacity 0.40) 기본 표시, 기본 카메라 정면, 시점 preset. surface mesh·component별 색상은 후속 (§7) |
| F5 | Playwright 미설치. Vitest + RTL + FakeViewerEngine + 실제 Chrome/Edge 수동 WebGL smoke |
| F6 | 기본 radiological. L/R/A/P/S/I 항상 표시, 현재 방식 화면 명시, neurological 전환 버튼. 반구 정량은 API 값만 사용 |
| F7 | DWI·ADC linear, mask nearest. private/@internal API·any cast 금지 |
| F8 | 모델 출력 점수 패널 기본 접힘, calibrated=false·진단 확률 아님 명시, % 변환 금지 |
| F9 | Validation Dice는 주 화면 비표시, 모델 도움말에서만 (내부 validation·checkpoint 선택 사용으로 낙관적·외부 검증 아님) |
| F10 | 다운로드는 shareable=true 파일만. 비식별 DWI·ADC는 Viewer 로딩 전용, 원본 업로드는 요청하지 않음 |
| F11 | 업로드 전 원본 파일명 비표시 (역할·형식·크기·선택 여부만). 브라우저 기본 file input은 숨기고 버튼으로 연다 (§8) |
| F12 | mask 기본 색 #E69F00, opacity 0.5, 허용 색·opacity 변경 가능 |
| F13 | 한국어 기본, 문자열 `src/i18n/ko.ts` 분리, i18n 패키지 없음 |
| F14 | npm 패키지: 승인 22개 정확한 버전 (`frontend/package.json`) |

추가 결정 (2026-10-25):
- TypeScript 6.0.3 (7.0.2는 typescript-eslint 8.x peer `<6.1.0` 위반).
- `@types/react`·`@types/react-dom` 19.3.0, `@testing-library/dom` 10.4.2 직접 devDependency.
- zod 4.6.7 (4.6.5는 GHSA-mrhx-c7xc-m3w5, high).
- mask 색상(F12): 공개 typed API `Niivue.drawLut` + `cmapper.makeDrawLut(ColorMap)` 사용 승인 (adapter 내부 격리,
  팔레트 4색만 허용).
- mask는 NiiVue **drawing layer** 공개 API(`loadDrawing`, `closeDrawing`, `setDrawOpacity`,
  `setDrawColormap`, `setDrawingEnabled`)로 렌더링: drawing texture는 R8 + `gl.NEAREST` 고정
  (`core/gl.ts r8Tex`)이라 전역 `setInterpolation(false)`(DWI·ADC linear)와 독립. 공개
  `setInterpolation`은 background·overlay에 같은 필터를 적용하므로 overlay volume 방식은 F7 불가.

## 2. 의존성 (설치 2026-10-25, `npm install --ignore-scripts`)

dependencies: @niivue/niivue 0.69.0, @tanstack/react-query 5.104.1, react 19.3.0, react-dom 19.3.0,
react-router-dom 7.18.4, zod 4.6.7, zustand 5.0.15.
devDependencies: @testing-library/dom 10.4.2, @testing-library/jest-dom 7.0.1, @testing-library/react 16.3.3,
@testing-library/user-event 14.6.7, @types/react 19.3.0, @types/react-dom 19.3.0, @vitejs/plugin-react 6.1.2,
eslint 10.12.0, eslint-plugin-react-hooks 7.1.1, jsdom 30.1.2, prettier 3.9.10, typescript 6.0.3,
typescript-eslint 8.71.1, vite 8.3.4, vitest 5.0.3.
`npm audit`: 설치 시점(2026-10-25) 0 vulnerabilities → 재검증(2026-11-01) moderate 10 (§5.1).
`npm ls`: invalid/extraneous/missing 없음 (UNMET OPTIONAL = 선택 peer·타 OS binary).

## 3. NiiVue drawing-layer spike (headless Chrome, SwiftShader WebGL2, 합성 데이터)

`/__spike?autorun=1` (개발 전용 페이지) — **64/64 통과** (공통 26 × {builtin, customLut} + custom palette 12; §7 A안에서 3D 배경·카메라 9항목 × 2 추가, 이전 46).

| 검증 | 결과 |
|---|---|
| `.nii.gz` ArrayBuffer 로드 (DWI·ADC·mask) | ✅ |
| mask 0 완전 투명 (변경 픽셀은 병변 영역 1.2%뿐), LUT alpha0 = 0 | ✅ |
| mask 1 = 지정 색 정확히 (기본 RGB 230,159,0: 3136/3136) | ✅ |
| nearest: 경계 혼합 픽셀 0 (DWI linear 활성 상태에서) | ✅ |
| DWI/ADC linear: nearest의 checkerboard 회색 2개 비율 0.999 → linear 0.000 | ✅ |
| axial·coronal·sagittal·MPR 동일 표시 | ✅ |
| opacity 0.5/0.3: 표시 색 = base·(1−o) + color·o (일치율 1.0000) | ✅ |
| radiological(화면 오른쪽 = 환자 좌측) / neurological 전환 | ✅ |
| DWI↔ADC 전환 후 mask 위치 Jaccard 1.0000 (색 변경 후 포함) | ✅ |
| 팔레트 4색 각각 정확, 위치 불변; 팔레트 밖 값(#FF0000, red, `__proto__` …) 거부 | ✅ |
| world mm 이동(`convertMM2Frac`), window/level 변경·초기화 | ✅ |
| 3D: mask 단독 렌더링, DWI 반투명 배경 토글, 색 변경 후 유지 | ✅ |
| dispose: drawing bitmap·volume 해제, WebGL context lost | ✅ |

사용한 NiiVue API는 모두 공개·typed: `attachToCanvas`, `NVImage.loadFromUrl({url: ArrayBuffer})`,
`addVolume/removeVolume`, `setInterpolation(false)`, `loadDrawing/closeDrawing/setDrawOpacity/
setDrawingEnabled/setDrawColormap`, `drawLut` + `cmapper.makeDrawLut(ColorMap)` (승인),
`setSliceType`, `setRadiologicalConvention`, `setOpacity`, `scene.crosshairPos` +
`NVImage.convertMM2Frac/convertFrac2MM`, `moveCrosshairInVox`, `setPan2Dxyzmm`,
`setRenderAzimuthElevation`, `updateGLVolume`, `drawScene`, `cleanup`. private/@internal API,
any cast, `@ts-ignore`, prototype patch 없음. **NiiVue 버전을 바꾸면 spike를 반드시 다시 실행한다.**

## 4. 구현

```
frontend/src/
├── api/        client.ts(상대 /api/v1만, credentials omit, ApiError·ApiUnavailable·Contract),
│               schemas.ts(Zod strict, 백엔드 schemas.py 1:1), endpoints.ts, upload.ts(XHR 진행률), errors.ts
├── app/        App.tsx(라우팅·QueryClient), queryClient.ts
├── components/ Layout(헤더·health/worker 배너·모델 도움말·고지문 footer), ConfirmDialog
├── features/viewer/  ViewerEngine(인터페이스·팔레트), NiivueEngine(유일한 NiiVue 사용처),
│               FakeViewerEngine, engineContext(주입), viewerStore(Zustand), useVolumes(다운로드·SHA·abort),
│               ViewerCanvas(엔진 수명), ViewerControls, keyboard
├── features/results/ ResultPanel(허용 필드만)
├── pages/      Upload, Case, RunProgress, ResultViewer, NotFound
├── i18n/ko.ts  문자열·금지 표현 목록
└── spike/      개발 전용 drawing-layer 검증 (production build에서 라우트 제외)
```

- 페이지: `/` 업로드 → `/cases/:id` (분석 실행·기록·삭제) → `/runs/:id` (1초 polling, 종료 시 중지)
  → `/runs/:id/viewer`. 새로고침은 URL의 ID로 복구. 409 `RUN_ALREADY_ACTIVE`는 진행 중 run으로 이동,
  `RESULT_EXISTS`는 "기존 결과 보기 / 다시 실행" 선택.
- 결과 화면: 병변이 있으면 API center of mass slice로 열림. 왼쪽 조작(단면·DWI/ADC·slice·W/L·mask 표시/
  불투명도/팔레트·병변 중심 이동·좌우 표시·확대/축소/초기화·mask 다운로드(shareable만)), 가운데 viewer
  (키보드 조작), 오른쪽 정량 패널(총 부피·연결 영역·최대 영역·slice 수·반구 "추정"·경고·위치·연결 영역 목록·
  기본 접힌 점수 패널 calibrated=false·혈관 영역 비추정).
- WebGL2 미지원 시 viewer 대신 안내 + 정량 패널 유지. 미검출: 필수 문구, 3D 빈 화면 안내.
- production build에 CSP (`default-src 'self'`, `connect-src 'self'` …).

## 5. 검증 결과

- 프론트엔드: `typecheck`, `lint`(console.log·any·ts-ignore 금지 규칙 포함), `format`, `build` 통과,
  **Vitest 49 passed** (5개 파일: API client·schema, 업로드, case·run, viewer·결과 패널·3D 시점, 격리).
  격리 테스트: `@niivue/niivue`·`drawLut`·`cmapper`는 `NiivueEngine.ts`에만 존재, UI 컴포넌트·store는
  `ViewerEngine`만 사용.
- `npm audit`: 2026-10-25 기준 0 vulnerabilities. 현재 결과는 §5.1 참고(moderate 10).
- **Validation 0086 통합 smoke** (API + GPU worker + `vite preview` production build/CSP, headless Chrome):
  case 생성·run 성공(48.976 mL, 9개), 결과 화면 로드(canvas, 오류 없음, 병변 중심 slice 14/73에서 주황
  nearest mask가 환자 좌측에 표시), **패널 부피 48.98 mL = API 48.976**, 브라우저 file URL로 받은 mask가
  **Phase 4 예측과 불일치 0 voxel**, Dice 0.94677 = Phase 4, 브라우저 세션의 API 요청은 `/api/v1/` 경로뿐
  (result·files×3·health·models), DOM에 파일명·경로·ISLES ID·금지 표현 0건. Test 접근 0건.
  (업로드는 브라우저 자동화 확장이 연결되지 않아 같은 proxy endpoint로 수행. 파일 선택 UI 업로드는
  `frontend/README.md` 수동 체크리스트로 확인.)
- Python 회귀: pytest **271 passed, 1 skipped**, ruff check·format 통과. Phase 1~5B 코드·설정 diff 없음,
  `results/`·`models/`·`outputs/` 237개 파일 해시 동일.

### 5.1 커밋 전 재검증 (2026-11-01, 코드·패키지 변경 없음)

- 프론트엔드: `typecheck`·`lint`·`format`·`build` 통과, **Vitest 49 passed**. `npm ci --ignore-scripts`를 깨끗한
  디렉터리에서 실행해 lock 파일로 재현 확인(250 packages, lock 변경 없음, package.json 직접 의존성 22개 일치).
- build 경고: 500 kB 초과 chunk (index 1.31 MB, zstd 0.75 MB, blosc 0.60 MB) — §6 code-splitting 후속.
- **`npm audit`: moderate 10, high 0, critical 0.** 10건 모두 동일 advisory(minimatch < 11.0.1,
  GHSA-g3hv-75wg-m249, DoS)에서 파생되며 경로는 `eslint`·`typescript-eslint` 계열 devDependency(lint 도구)뿐이다.
  production bundle에는 포함되지 않는다. `npm audit fix --force`가 제시하는 수정은 eslint 4.0 등으로의
  major 하향이라 현재 호환성을 유지하는 안전한 자동 수정은 없다. 패키지 버전은 바꾸지 않았고, 상위 도구
  (eslint·typescript-eslint)의 호환 업데이트가 나오면 재검토한다.
- Python: **Non-Test test suite 267 passed, 1 skipped** (symlink 권한), ruff check·format 통과.
  Test Split 데이터를 실제로 읽는 기존 integration test 4개(`tests/test_phase2_data.py`, GT 재구성 round-trip)는
  **Test Split 봉인 정책으로 제외**했다. 기존 기록상 전체 suite는 271 passed, 1 skipped.
  **Test Split 모델 성능 평가는 아직 수행하지 않았다.**
- Backend 연동(API + `vite preview`, worker·GPU 미사용, 임시 storage): proxy 경유 `/health` 200(worker offline →
  degraded), `/models/current` 503 `MODEL_NOT_READY`, 없는 run·case·file 404, 외부 Origin에 CORS 헤더 없음,
  production HTML CSP meta 존재, production bundle에 spike 코드 없음. 결과·파일 다운로드 경로는 저장된 run이
  없어 이번에 재실행하지 않았다(Validation 0086 기록은 Phase 4 `validation_metrics.csv`와 일치: Dice 0.94677,
  48.976 mL, 9개).

## 6. 한계와 후속

Phase 6 Viewer MVP 범위: DWI/ADC base volume, AI 예측 mask overlay, Axial/Coronal/Sagittal/MPR/3D,
병변 부피와 연결 영역 정량, 모델 도움말의 Validation 평균 성능, 연구용 진단 보조·의료기기 아님 고지.

**구현되지 않음** (완료가 아님):

- Ground Truth mask overlay와 환자별 Dice 등 평가 지표 표시 — 업로드 입력의 독립 추론 화면에는 GT가 없다.
  공개 데이터 기반 연구/검증 모드의 향후 기능으로 분리한다.
- surface mesh(brain·lesion), component별 3D 색상, screenshot, DICOM, Playwright E2E, 영어 i18n.
- 인증·HTTPS (localhost 전용 개발 버전).

기타 한계:

- 실제 GPU WebGL 대신 SwiftShader로 자동 검증 — 사용자 Chrome/Edge에서 수동 smoke 권장.
- slice slider는 렌더링·키보드·조작 시 갱신되며, canvas 마우스 클릭으로 crosshair를 옮긴 직후에는 다음
  조작 때 갱신된다(NiiVue location 이벤트 연동은 후속).
- production bundle이 크다(NiiVue의 zarr codec chunk 포함; NIfTI 경로에서는 로드되지 않음). code-splitting 후속.
- component별 색상·surface mesh·screenshot·Playwright E2E·i18n 영어는 후속.
- 운영 전 필수: 인증·HTTPS·저장 암호화·접근 감사(백엔드와 동일).

## 7. 3D 표시 개선 (A안, 2026-10-25 승인)

실제 GPU 확인에서 3D mask만으로는 해부학적 위치를 알기 어려웠다. 진단(Validation 0086, scratchpad 비교):
NiiVue 자동 범위(cal 0.24–385 ≈ 뇌 p0–p93)는 정상이었고, 원인은 ① 렌더링 결과 전체에 곱해지는 배경 opacity
(0.25 → 밝기 상한 25%) ② gray colormap의 alpha ramp(어두운 CSF·가장자리 투명) ③ 측면 사선 기본 카메라였다.
percentile 범위 조정은 효과가 없었고(채택 안 함), brain envelope는 후속 후보로 남긴다.

| 항목 | 변경 |
|---|---|
| 3D 뇌 배경 | 기본 표시 (`render3DBackground: true`), opacity **0.40** (`RENDER_BACKGROUND_OPACITY`). DWI·ADC 동일, 2D는 1.0, mask 0.5 그대로 |
| 카메라 | 기본 **정면** (azimuth 180, elevation 0). preset: 정면(180,0)·위(180,90)·왼쪽(90,0)·오른쪽(270,0)·기본 시점(=정면). 공개 `setRenderAzimuthElevation`만 사용 |
| 방향 | 정면·위: 화면 왼쪽 = 환자 오른쪽 (radiological과 같은 좌우), 위: 화면 아래 = 앞쪽, 왼쪽/오른쪽 시점: 앞쪽이 화면 왼쪽/오른쪽. NiiVue 공개 옵션 `isOrientCube`로 3D 왼쪽 아래에 방향 큐브(L/R/A/P/S/I) 표시 |
| UI | "3D 뇌 배경 표시" 체크박스 (`aria-keyshortcuts="B"`), 단축키 **B**, "3D 시점" radiogroup(접근 가능한 이름 = tooltip, 선택 표시), 자유 회전 시 선택 해제 |

검증: spike 64/64 (기본 정면, 3D 0.40 / 2D 1.0, 합성 병변(환자 좌측·앞쪽)의 preset별 화면 위치, 기본 시점 복귀
Jaccard 1, DWI→ADC 후 카메라 유지 + 3D mask Jaccard 1, 자유 회전 알림), Vitest 49, 0086 production build 캡처
(진입 즉시 배경+mask, 정면 좌우 반구·정중선, 위 시점 정중선·병변, B 키·체크박스 동기화, 부피 48.98 mL·9개,
mask SHA-256 동일, `/api/v1`만 요청). mask·보간·affine·정량·백엔드 변경 없음.

## 8. 수동 smoke에서 발견한 업로드 문제와 수정 (2026-11-01)

**증상**: 사용자 Chrome에서 0086 DWI·ADC를 선택하자 형식 `.nii`, 크기 0.00 MiB로 표시되고, 업로드하면
"DWI와 ADC 파일이 모두 필요합니다."가 나왔다. 브라우저 기본 file input 옆에 원본 파일명도 보였다.

**원인**: 프론트엔드 state 손실이 아니었다. 업로드 요청은 서버에 도달했고, 서버가 0 byte 파일을
`400 MISSING_FILE`(`uploads.py`, "file is empty")로 거부했다. 원본은 `.nii.gz`(1.75·1.57 MiB)인데 표시 형식이
`.nii`였으므로, Windows 11 파일 대화상자가 `.nii.gz`를 압축 폴더로 열어 보여 준 내부 `.nii` 항목(브라우저에는
빈 파일)이 선택된 것으로 판단했다. `.nii.gz` 파일 자체를 선택하자 업로드·GPU 추론·2D/3D 표시·정량이 모두
정상이었다. 클라이언트가 빈 파일을 거르지 않았고, 서버 오류 문구가 실제 원인(빈 파일)과 맞지 않았다.
또 F11은 앱 텍스트에서는 지켜졌지만 브라우저 기본 file input이 파일명을 그려 실제 화면에서는 지켜지지 않았다
(jsdom 테스트로는 확인할 수 없는 부분).

**수정** (프론트엔드만, 백엔드·API 계약·Viewer·추론·패키지 변경 없음):

| 항목 | 내용 |
|---|---|
| 빈 파일 | 0 byte `.nii`·`.nii.gz`를 선택 시점에 거부 ("빈 파일입니다 … 압축 내부 항목이 아니라 파일 자체를 선택") |
| 최소 크기 | `.nii`가 352 byte(NIfTI-1 header 348 + extension flag 4, `MIN_NII_BYTES`) 미만이면 거부 |
| 오류 문구 | `MISSING_FILE` → "DWI 또는 ADC 파일이 없거나 비어 있습니다. .nii.gz 파일은 압축 내부 항목이 아니라 파일 자체를 선택하세요." |
| 파일 선택 UI | 기본 input은 `sr-only`·`aria-hidden`·`tabIndex=-1`로 숨기고 "DWI 파일 선택"/"ADC 파일 선택" 버튼으로 연다. 선택 직후 input value를 비워 DOM에도 파일명이 남지 않는다. 화면에는 역할·형식·크기만 표시 |
| 버튼 | 잘못된 파일을 고르면 해당 역할 선택이 해제되고 업로드 버튼 비활성화, 요청 미전송 |

테스트 (`tests/upload.test.tsx`, +9): 0 byte `.nii.gz`·`.nii`, 100 byte `.nii` 거부(버튼 비활성·XHR 미호출),
유효 선택을 빈 파일로 바꾸면 비활성, 실제 크기(1,839,273·1,642,955 B) `.nii.gz` 허용과 1.75·1.57 MiB 표시,
352 byte `.nii` 허용, 기본 input 숨김과 버튼으로 열기, 파일명·로컬 경로 비표시(input value 비움),
서버 `MISSING_FILE` 문구. 결과: typecheck·lint·format·build 통과, **Vitest 58 passed** (이전 49).
