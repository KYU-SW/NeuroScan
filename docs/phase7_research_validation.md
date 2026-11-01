# Phase 7-B — Research Validation Mode

작성 2026-11-01 · 기준 커밋 `4d20a4e` (Phase 7-A) · 상태: 구현 완료 (미커밋) · **Test Split 미사용**

> ⚠️ 연구·검증 모드는 Validation 데이터에서 AI 결과와 Ground Truth(GT)를 비교하는 연구용 기능입니다.
> 임상 진단 결과가 아니며, 비교 결과는 병변 확정·오류 판정이 아닙니다.

## 1. Inference Mode와 Research Mode 분리

| 항목 | Inference Mode (기존) | Research Validation Mode (신규) |
|---|---|---|
| 입력 | 사용자 업로드 DWI/ADC | CLI로 등록한 **Validation** case만 (GT 업로드 UI 없음) |
| 화면 | `/`, `/cases/:id`, `/runs/:id/viewer` | `/research`, `/research/cases/:alias` (상단 배너 항상 표시) |
| API | `/api/v1/cases`, `/runs`, `/files` | `/api/v1/research/cases[/{alias}]` + `/files/{id}` (연구 파일) |
| DB | `cases`, `inference_runs`, `file_resources` | `research_cases`, `research_files` |
| 저장소 | `storage/uploads`, `storage/runs` | `storage/research/<alias>` |
| GT·TP/FP/FN·환자별 지표·영역 비교 | **없음** | 있음 |

- `research_enabled: false`(기본, `configs/api_v1.yaml`)이면 research router를 등록하지 않는다 → 모든 research
  경로는 존재하지 않는 경로와 같은 404. `/files/{id}`도 연구 파일은 404. 켜는 방법: 설정 파일 또는
  `NEUROSCAN_RESEARCH_ENABLED=1`(localhost 전용 원칙 유지).
- 일반 결과 API·화면에는 연구 데이터가 연결되지 않는다(별도 table, 별도 schema). 일반 결과 화면에 GT,
  TP/FP/FN, IoU·Sensitivity·Precision, GT 부피, 영역 비교 문구가 없음을 테스트로 확인한다. 엔진의 연구 overlay는
  차이 labelmap이 로드된 연구 화면에서만 호출된다.
- Research API는 일반 업로드 case를 연구 case로 다룰 수 없다(별도 table만 조회, alias 형식 `val-NNN`).

## 2. Research Data Flow (CLI 전용)

```
scripts/build_research_case.py --case <validation case key> --alias val-001
 1. alias 형식·중복 검사
 2. SplitGuard: case key 문자열만으로 판정 — test → 즉시 거부(SEALED_TEST_CASE),
    validation이 아니면 거부(NOT_VALIDATION). 이 단계 전에는 어떤 파일도 열지 않는다.
 3. research/.staging/<uuid>에서 동결된 Phase 5A 엔진으로 재추론 (case id = alias)
 4. 저장된 Phase 4 Validation 예측과 voxel 단위 비교 (shape·affine·foreground·불일치 0) — 다르면 중단
 5. GT 검증: 유한값, 3-D, 값 {0,1}만, 유효 affine
 6. 예측/GT 같은 격자 검사 (shape 동일, affine 오차 1e-4 이내)
 7. evaluate_case (Phase 2 정의 그대로)
 8. 예측·GT 연결 영역 (26-connectivity, Phase 7-A component_analysis)
 9. TP/FP/FN 차이 labelmap
10. 영역 matching (overlap graph)
11. 교차 검증 (아래 §4·§7) + 파일 재로드 검증 + 공개 JSON의 case key·절대 경로 검사
12. 원자적 등록: staging → research/<alias> rename + DB 행(하나의 트랜잭션) + 로컬 alias map
    — 어느 단계든 실패하면 staging 삭제, 아무것도 등록하지 않음
```

Bundle (`research/<alias>/`): `dwi.nii.gz`·`adc.nii.gz`(Phase 5A 비식별 사본), `prediction_mask.nii.gz`,
`prediction_components.nii.gz`, `difference.nii.gz`, `metrics.json`, `components.json`, `matching.json`,
`research_manifest.json`(파일 SHA-256, 모델 정보, 재현성 결과). 엔진 로그·메타데이터는 bundle에 넣지 않는다.

## 3. Alias 정책

- 화면·URL·API 응답·bundle·DB에는 익명 alias(`val-001` …)만 쓴다. 원본 case key와 원본 파일명은 어디에도
  내보내지 않으며 API 응답은 case key 패턴 검사를 통과해야 한다.
- alias ↔ case 대응은 `storage/research/_internal/aliases.json`(로컬 파일, 서빙 대상 아님, Git 제외 경로)에만 있다.
- 이 문서·README에는 어떤 alias가 어떤 ISLES case인지 적지 않는다.

## 4. Difference Labelmap 계약과 TP/FP/FN 정의

- 생성 위치: backend/AI `neuroscan_ai.research.compare` (프론트엔드는 계산하지 않음).
- `uint8`, DWI와 같은 shape·affine·orientation. 0 = background, **1 = TP**(AI와 GT 모두),
  **2 = FP**(AI만), **3 = FN**(GT만).
- 등록 전 검증: label 1/2/3 voxel 수 = `evaluate_case`의 tp/fp/fn, TP + FP = 예측 foreground,
  TP + FN = GT foreground. 하나라도 다르면 등록하지 않는다.
- 표시 색(승인 팔레트): TP `#009E73`, FP `#E69F00`, FN `#56B4E9`, GT 단독 `#CC79A7`. 텍스트 범례
  ("일치 영역 (TP)", "AI만 표시한 영역 (FP)", "GT에 있으나 AI가 놓친 영역 (FN)", "Ground Truth 영역 (TP + FN)")를
  모든 표시 모드에서 항상 보여 준다.

## 5. Research Viewer

- 표시 모드(버튼 순서, 기본 = 차이 지도): **차이 지도**(TP/FP/FN, 클래스별 ON/OFF) / **Ground Truth**(차이
  labelmap의 1·3을 GT 색, 2는 숨김) / **AI 예측**(예측 labelmap, 주황 단색 기본 + 선택 강조·숨김) /
  **기본 영상만**(overlay 숨김). UX 정책은 §14.
- 엔진: `setResearchOverlay()`가 drawing layer를 예측 labelmap ↔ 차이 labelmap으로 바꾸고 LUT를 적용한다
  (nearest, DWI↔ADC 전환 후에도 유지 — spike 확인).
- Component Review(Phase 7-A 재사용): 영역별 "GT와 겹침 / AI만 표시된 영역", 묶음 유형, TP·FP voxel, TP 비율,
  연결된 GT. 강조·흐림은 AI 예측 모드에서 적용, 위치 이동은 모든 모드에서 voxel 기준.
- 영역 비교 표: 묶음 유형, AI 영역, GT 영역(GT 위치 이동 버튼), 겹침 voxel, IoU, 부피, 중심 거리(참고).
- 문구: 연구용 비교 표현만 사용("GT와 겹침", "AI만 표시된 영역", "GT에만 표시된 영역", "분할됨", "병합됨").
  "실제 병변 확정", "위양성 확정", "오진", "정상 판정", "치료 판단" 등은 금지 문구 DOM 검사에 포함.

## 6. Metric 정책과 Negative GT

- 지표는 `evaluate_case`(Phase 2) 결과를 그대로 저장·전달한다. 프론트엔드는 다시 계산하지 않는다.
- Positive GT: Dice, IoU, Sensitivity, Precision, GT/AI 부피, 부호 있는·절대·상대 부피 오차, GT/AI 연결 영역 수,
  TP/FP/FN voxel.
- Negative GT: Dice·IoU·Sensitivity·Precision은 **"정의되지 않음: Ground Truth 병변 없음"**(0이나 1로 바꾸지
  않음). AI False Positive voxel 수·부피·연결 영역 수, Correct Negative 여부를 표시한다.
- Validation은 checkpoint 선택에 쓰였으므로 결과가 낙관적일 수 있고 외부 검증이 아니라는 안내를 함께 표시한다.

## 7. Component Matching

26-connectivity로 예측·GT 영역을 각각 구하고 1 voxel 이상 겹치는 쌍을 edge로 하는 overlap graph의 연결 묶음을
보고한다. IoU 기준·중심 거리로 강제 matching하지 않으며 작은 영역도 제외하지 않는다.

| 유형 | 의미 |
|---|---|
| `one_to_one` | AI 1 : GT 1 |
| `split` | AI 여러 개 : GT 1 ("분할됨") |
| `merge` | AI 1 : GT 여러 개 ("병합됨") |
| `complex` | AI 여러 개 : GT 여러 개 |
| `prediction_only` | GT와 겹치지 않는 AI 영역 |
| `ground_truth_only` | AI와 겹치지 않는 GT 영역 |

묶음별: 영역 ID(크기 순위), overlap / union voxel, IoU, AI·GT 부피, 중심 거리(mm, 실제 affine, 참고). 등록 전
검증: 영역별 TP 합 = TP, FP 합 = FP, GT 미검출 합 = FN, 묶음 overlap 합 = TP, 영역 수 = evaluate_case.

## 8. Storage·DB 격리와 Schema v2

- `SCHEMA_VERSION = 2`: `research_cases`(id, alias, dataset_id, split, source_model, checkpoint_sha256,
  config_hash, threshold, preprocessing_version, bundle_rel, metrics_path, component_matching_path,
  has_ground_truth, created_at)와 `research_files`(id, research_case_id, role, storage_area="research",
  rel_path, mime_type, size_bytes, sha256). 지표·영역·matching은 JSON 파일에 두고 DB에는 메타데이터·경로만.
- `file_resources.case_id`가 일반 `cases`를 참조하는 NOT NULL FK라서, 기존 table을 바꾸지 않기 위해 연구 파일은
  별도 `research_files` table(storage_area `research`)에 둔다. 기존 table의 필드·의미는 변경 없음.
- 업그레이드 정책(Alembic 없음): v1 DB를 열면 먼저 `neuroscan.db.v1-<UTC>.bak` 전체 백업(SQLite online
  backup), 이어서 새 table만 생성(`create_all`은 기존 table·행을 바꾸지 않음)하고 version을 2로 기록. 알 수 없는
  버전(예: 99)은 파일을 건드리지 않고 거부. 테스트로 데이터 보존·백업·멱등성 확인.
- **운영 중인 API/worker는 재시작 시 v2로 업그레이드된다(백업 생성).** 이전 코드(v1)를 v2 DB에 다시 띄우면
  version 불일치로 시작을 거부한다.

## 9. Validation Prediction 재현성

등록한 Validation 4건 모두 Phase 5A 재추론 결과가 저장된 Phase 4 예측과 **shape·affine 동일, 불일치 0 voxel**.
모델 checkpoint, 2.5D 구조, 전처리, 정규화, threshold 0.5, brain mask gating, 영역 제거 정책(제거 안 함),
후처리, 지표 정의는 바꾸지 않았다. 불일치가 있었다면 `PREDICTION_MISMATCH`로 등록을 중단하도록 구현·테스트했다.

## 10. 검증 결과

| alias | 유형 | 재현성 | Dice (API·화면) | TP / FP / FN | AI·GT 영역 | 위치 이동 (AI / GT) |
|---|---|---|---|---|---|---|
| val-001 | positive | 0 voxel 불일치 | 0.94677 / 0.9468 (Phase 4 기록과 일치) | 5665 / 457 / 180 | 9 · 2 | 9/9 · 2/2 |
| val-002 | positive, oblique 31.2° | 0 | 0.86216 / 0.8622 | 172 / 35 / 20 | 5 · 1 | 5/5 · 1/1 |
| val-003 | **negative GT** | 0 | 정의되지 않음 | 0 / 27 / 0 | 4 · 0 | 4/4 · — |
| val-004 | 작은 병변 (GT 0.568 mL) | 0 | 0.75 / 0.7500 | 54 / 19 / 17 | 8 · 1 | 8/8 · 1/1 |

- 4건 모두 production build + headless Chrome(CDP)에서 배너 항상 표시, 네 표시 모드 전환과 범례, 지표, 영역 비교,
  금지 문구·case key·경로 0건, `/api/v1` 요청만 확인. val-001에서는 작은 분리 영역 하나가 작은 GT 영역과 겹치는
  `one_to_one`으로 나타나 작은 영역을 자동 삭제하지 않는 정책의 의미를 보여 준다.
- 거부 확인: 중복 alias, test case key(`SEALED_TEST_CASE`, 파일 열기 전), train case key(`NOT_VALIDATION`) —
  staging 잔여물 0.
- 자동 검사: Python **Non-Test suite 289 passed, 1 skipped**(신규 `tests/test_research.py` 14개), Vitest
  **109 passed**(신규 `research.test.tsx` 19개, §14 UX 포함), spike 회귀 Phase 6 64/64 · Phase 7 40/41(실패 1 = 문서화된 3D 부분 alpha 제한,
  신규 7-B 렌더링 7항목 모두 통과), typecheck·lint·format·build 통과, production bundle에 spike 없음.

## 11. Test Split 봉인

Test Split의 영상·GT·예측을 열지 않았고 Test 예측·지표를 만들지 않았다. 연구 등록은 SplitGuard로 Validation만
허용하며 test case key는 파일을 열기 전에 거부한다(자동 테스트는 합성 split 파일 사용). Python 회귀에서 Test GT를
읽는 `tests/test_phase2_data.py`는 계속 제외했다.

## 12. External Validation 확장

`neuroscan_ai.research.sources`의 `ResearchSourceAdapter` protocol(dataset_id, split, alias, base_image,
optional secondary_image, ground_truth, reference prediction, 내부 case key)만 dataset별로 구현하면 된다. ISLES
파일 규칙은 `Isles2022ResearchAdapter` 안에만 있다. 차이 labelmap, 지표, matching, bundle 검증, research API는
dataset과 무관하다. GT가 없는 외부 데이터는 연구 모드로 등록할 수 없고 Inference Mode만 쓴다. 외부 데이터
다운로드는 하지 않았다.

## 13. 한계

- 연구 case는 CLI로만 등록·추가된다(삭제 CLI 없음; 필요 시 별도 승인).
- 영역 matching은 겹침 기반 MVP — IoU 기준, Hungarian 1:1, 거리 허용 matching은 하지 않는다.
- 3D 부분 alpha 제한은 Phase 7-A와 같다(강조는 회색 흐림, 차이 지도는 불투명 색).
- GT 영역 labelmap은 만들지 않는다(GT 표시는 차이 labelmap으로 충분). GT 영역 목록에는 위치 이동만 제공.
- 연구 지표는 Validation 4건의 검토용이며 모델 성능 주장에 쓰지 않는다. 최종 성능은 별도 Test 평가 Phase에서만.

## 14. Research Viewer UX 개선 (P0, 커밋 전 반영)

**배경**: 수동 smoke에서 기능은 정상이었지만 (1) AI 예측의 영역별 색과 TP/FP/FN 의미 색이 같은 계열이라 혼동되고,
(2) 차이 지도·GT 표시에서는 선택 강조가 적용되지 않는데 목록은 "선택됨"으로 보이고, (3) 회색 흐림의 이유와 현재
표시 모드가 Viewer 주변에 보이지 않으며, (4) 선택 후 무엇을 확인하면 되는지 안내가 없었다. 연구 화면의 표시와 안내만
고쳤다(API·엔진·지표·matching·차이 labelmap·모델·일반 Inference Viewer 변경 없음).

**모드와 색 의미** (현재 모드의 범례만 Viewer 위에 표시, 색마다 텍스트 의미 동반)

| 모드 (버튼 순서) | 현재 표시 띠 | 범례 |
|---|---|---|
| 차이 지도 (기본) | 현재 표시: 차이 지도 (TP / FP / FN) | 일치 영역 (TP) `#009E73` · AI만 표시한 영역 (FP) `#E69F00` · GT에 있으나 AI가 놓친 영역 (FN) `#56B4E9` |
| Ground Truth | 현재 표시: Ground Truth | Ground Truth 영역 (TP + FN) `#CC79A7` |
| AI 예측 | 현재 표시: AI 예측 전체 (TP + FP) | AI 예측 영역 (TP + FP) `#E69F00` · 회색 = 선택하지 않은 영역의 검토용 흐림(원본 결과·신뢰도 변경 없음) |
| 기본 영상만 | 현재 표시: 기본 영상만 | Overlay 없음 |

**AI 예측 단색 기본**: 연구 화면에 들어오면 AI 예측 전체를 `#E69F00` 단색으로 표시한다(선택 영역은 주황 유지,
나머지는 승인된 회색 흐림, 숨긴 영역은 표시 안 함). 영역별 다색은 "영역별 색 구분 (검토용)" 토글(기본 OFF)이며,
"색상은 영역 순서를 구분하기 위한 표시이며 의료적 의미나 정확도를 나타내지 않습니다."를 함께 표시한다. 연구 화면에서는
일반 Viewer의 팔레트·색 모드 전환을 숨긴다. **일반 Inference Viewer의 기본 색 정책(영역별 색 기본)은 바꾸지 않았다.**

**선택 요약 패널** (연구 화면 전용, Viewer 위): AI 영역 — 번호, 부피, slice 범위, 위치 추정, 작은 분리 영역 여부,
GT 비교(전체 겹침 / 일부 겹침 / GT와 겹치지 않음), matching 유형, 연결된 GT, TP/FP voxel, TP 비율. GT 영역(비교 표에서
선택) — 번호, 부피, slice 범위, "AI가 표시하지 않은 GT 영역" 또는 겹친 AI 영역, 겹친/놓친 voxel. 버튼: DWI 보기 · ADC 보기
(위치·overlay 유지) · 차이 지도 보기 · AI 예측에서 강조 보기(선택 유지; GT 선택 시 비활성과 이유) · 3면에서 확인 ·
선택 해제. 아래에 "검토 예: DWI 신호 확인 → ADC 확인 → 차이 지도에서 Ground Truth와 비교. 연구용 안내이며 진단 기준이나
의료적 판단 절차가 아닙니다."

**모드별 제어 활성화**: 숨기기·선택 영역만 표시·전체 다시 표시는 AI 예측에서만 활성화된다. 다른 모드에서는 버튼을
비활성화하고 이유를 표시한다(차이 지도: "숨기기와 선택 강조는 AI 예측 표시에서 사용할 수 있습니다." + [AI 예측에서
강조 보기]; Ground Truth: "현재 Ground Truth 표시에서는 AI 영역 강조가 적용되지 않습니다."; 기본 영상만: "기본 영상만
표시 중"). 영역 선택·위치 이동은 모든 모드에서 동작한다. 차이 지도에서 목록 견본은 영역 색이 아니라 비교 상태(TP 초록,
FP 주황, 혼합 시 두 색 + 텍스트)를 보여 준다.

**3면에서 확인과 작은 영역**: 영역을 선택해도 사용자가 고른 2D/3D 보기는 바꾸지 않는다. 현재 보기가 3D이거나 선택 영역이
작은 분리 영역(또는 27 voxel 미만)이면 "3D에서는 작은 영역이 배경 볼륨과 시점에 따라 잘 보이지 않을 수 있습니다. 2D 단면에서
원본 DWI와 ADC를 함께 확인하세요." 안내를 표시한다. [3면에서 확인]은 3면(MPR) 보기로 바꾸고 대표 voxel로 다시 이동해
crosshair와 세 slider를 맞춘다.

**검증**: Vitest 신규 UX 테스트(모드 순서·기본값·현재 모드 띠·모드별 범례 4종·단색 기본과 토글·요약 내용·GT 영역 요약·
비활성 제어와 이유·AI 강조 전환·DWI/ADC 위치 유지·3면 확인·선택 해제·위치 "추정 불가" 표기·일반 화면에 UX 요소 0건), production
build + headless Chrome에서 val-002(oblique)의 1 voxel 영역 선택 → 요약·안내 표시, [3면에서 확인] 후 slider = 대표 voxel
(54, 11, 30), ADC 전환 후 위치 유지, 예외 0건.

## 15. Research Viewer 레이아웃 회귀와 수정 (커밋 전)

**회귀**: §14 UX P0 반영 후 연구 화면에서 canvas 높이가 계속 커졌다(1920×1080, 조작 없이 0.5초 간격 canvas
픽셀 높이 1600 → 2080 → … → 7520; 선택 후 Viewer 영역 8300 px, 1366×768에서는 2만 px 이상). 증상: 영상 확대·잘림,
과대 crosshair, 문서 높이 증가, sticky 배너가 왼쪽 패널 상단을 덮음, 오른쪽 패널 340 px에서 과도한 줄바꿈.

**원인**: Viewer 영역(`.center`) 안에서 canvas 위에 현재 모드 띠·범례·선택 요약을 두었는데, canvas host
(`.viewer-host`)는 `height: 100%; min-height: 600px`였다. host = 부모의 100% + 위쪽 요소 → 내용이 부모보다
커짐 → grid 행 증가 → NiiVue ResizeObserver가 canvas 픽셀 높이를 키움 → canvas 고유 높이가 다시 행을 키우는
되먹임 루프. Phase 6·7-A의 일반 화면은 Viewer 영역에 host 하나만 있어 발생하지 않았다.

**수정 (연구 화면 DOM·CSS만, 일반 Viewer·엔진·API 변경 없음)**

- DOM: `.research-page`(연구 화면 전체) › 배너 · 한 줄 정보 · `.research-layout`(연구 전용 grid) ›
  `section.center.research-center` › 현재 모드 띠 · 범례 · 선택 요약 · **`.viewer-slot`** › `ViewerCanvas`.
- `.viewer-slot`: `position: relative; flex: 1 1 auto; min-height: 320px; overflow: hidden`. 그 안의
  `.viewer-host`는 `position: absolute; inset: 0; height: auto; min-height: 0` → canvas가 레이아웃 크기에 기여하지
  않고 ResizeObserver는 확정된 slot 크기만 따른다.
- 연구 중앙 영역: `display: flex; flex-direction: column; min-height: 0; overflow: hidden`, 띠·범례 `flex: none`,
  선택 요약 `max-height: 38%; overflow-y: auto`(2~4열 요약).
- viewport 소유: `.app:has(.research-page)`만 `height: 100dvh`, 그 아래 `.app-main`·`.research-page`·grid가
  `min-height: 0` flex 체인으로 남은 높이를 나눈다(고정 120 px 계산 미사용). grid
  `260px minmax(480px, 1fr) minmax(360px, 28vw)`, `grid-template-rows: minmax(0, 1fr)`.
- 좌우 패널: `min-height: 0; overflow-y: auto; position: relative`(패널 안 `.sr-only` 같은 절대 위치 요소가 문서를
  늘리지 않도록). 연구 배너는 sticky 제거(정상 흐름, 화면이 viewport에 맞으므로 항상 보임). 비교 표 가로 스크롤,
  영역 행 버튼 줄바꿈, 범례·정보 한 줄화.
- 1280 px 미만: 오른쪽 패널을 아래로, 페이지 스크롤 허용, Viewer 영역 높이 `clamp(460px, 72vh, 900px)`.

**검증**

- jsdom 구조 테스트(`tests/researchLayout.test.tsx`): canvas가 `.viewer-slot` 안, 띠·범례·요약은 slot 밖 형제,
  선택·모드 전환·선택 해제 동안 canvas 한 번만 mount, 일반 Viewer에는 연구 레이아웃 class 0건.
  CSS 계약 테스트(`tests/researchLayoutCss.test.ts`): slot·host·연구 중앙·좌우 패널 규칙, 일반 Viewer 규칙 불변.
- 브라우저 측정(`frontend/tools/research-layout-check.mjs`, 개발용, production build + headless Chrome): 상태
  A 로딩 · B 선택(요약) · C 3면 · D 3D · E 창 크기 변경 후 · F 10초 후 · G 선택 해제, 상태마다 0.5초 간격 10회.

| 크기 (CSS px @ 배율) | canvas 높이 A / B~F / G | 요약 | 오른쪽 | 문서 높이 | 결과 |
|---|---|---|---|---|---|
| 1920×1080 @1 | 790 / 504 / 790 | 282 | 538 | 1080 (= viewport) | 통과 |
| 1536×864 @1 | 574 / 331 / 574 | 239 | 430 | 864 | 통과 |
| 1536×864 @1.25 (1920×1080, 125%) | 574 / 331 / 574 | 239 | 430 | 864 | 통과 |
| 1366×768 @1 | 478 / 320 / 478 | 154 | 382 | 768 | 통과 |
| 1280×720 @1 | 430 / 320 / 430 | 106 | 360 | 720 | 통과 |
| 1100×800 @1 (반응형) | 509 / 320 / 509 | 185 | 1061 (아래로) | 2374 (스크롤 허용) | 통과 |

모든 상태에서 canvas CSS·픽셀 높이와 문서 높이의 10회 측정 변동 0, host ⊂ slot ⊂ 중앙 영역, slot ≥ 320 px,
배너와 왼쪽 패널 겹침 0, 가로 overflow 0, 페이지 예외 0. 선택 요약이 나타나면 canvas가 한 번 줄고(예: −286 px)
해제하면 같은 만큼 한 번 늘어난 뒤 안정. 1600×900 화면 캡처로 영상이 Viewer 안에 맞고 crosshair가 정상 크기임을 확인.

일반 Inference Viewer(Phase 7-A 결과, 1920/1536/1366/1280): 모든 상태에서 안정(변동 0), 연구 요소 없음. 창을
줄였다 되돌리는 E 단계 후 1920·1536에서 canvas가 한 번 다른 크기로 정착(960→1009, 744→825)하고 이후 안정 —
일반 화면의 기존(Phase 6) 레이아웃 특성이며 이번 변경과 무관(일반 화면 DOM·CSS 미변경). 같은 slot 구조 적용은
Phase 7-C 후보.

**남은 반응형 한계**: 1280 px 미만에서는 viewport 고정을 풀고 페이지 스크롤을 사용한다. 높이가 매우 낮은 창
(약 650 px 미만)에서는 slot 최소 320 px를 위해 선택 요약이 짧아지고 내부 스크롤된다. `:has()`가 필요하다(Chrome·Edge
105+).
