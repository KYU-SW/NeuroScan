# Phase 5A — 독립 단일 환자 추론 파이프라인

작성 2026-10-24 · 기준 커밋 `fb85ba1` (Phase 4) · 기본 모델 `models/phase4/25d_baseline_20261024-192617_34d18df104da/best.pt`
(sha256 `ff61bc7f…dbe7`, config hash `34d18df104da`, Val positive mean Dice 0.7428) ·
**Test Split 미사용 (접근 기록 0명)**

> ⚠️ 연구 및 의료영상 검토 보조용 프로토타입입니다. 진단을 확정하지 않으며, 최종 판단은 의료진이
> 해야 합니다. 출력 점수는 보정(calibration)되지 않은 모델 점수이며 질병 확률이 아닙니다.

## 1. 확정 결정 (2026-10-24 승인)

| ID | 결정 |
|---|---|
| D1 | `predict_case`·`CaseVolumes` 계약 유지. 메모리 안에서만 0으로 채운 dummy GT 사용 — 파일 저장·GT 지표 계산·GT 관련 출력 필드 없음. Phase 3·4 코드 무수정 |
| D2 | 기존 전처리 함수(`reorient`, `resample_to_spacing`)를 메모리에서 동일 순서로 조합. `preprocess_case`와 동등성 테스트 필수 — 불일치 또는 Phase 1 수정 필요 시 중단 |
| D3 | 좌우 반구는 `estimated` 기능. RAS+ world 좌표와 검증된 brain midline 사용, 비스듬하거나 검증 실패 시 `not_available`. 배열 인덱스 절반으로 판정 금지. MCA/ACA/PCA 출력 금지 |
| D4 | 비식별 DWI·ADC 사본 기본 저장: voxel data + affine으로 새 NIfTI 생성(원본 header 미전달), 원본 파일명 미사용, 공유 기본 대상에서 `inputs/` 제외 |
| D5 | 실패 시 `.failed/`에 `inference_metadata.json`과 비식별 로그만 보존. 영상 사본·부분 mask·불완전 JSON 보존 금지 |
| D6 | probability map NIfTI 저장 안 함 (옵션도 없음). 후속 연구 기능으로만 문서화 (§9) |
| D7 | DWI/ADC 순서 뒤바뀜 의심 = warning, `--strict`에서 error. 자동 교정 금지 |
| D8 | 결과는 `outputs/` (Git 무시). `results/` = 연구 평가, `outputs/` = 독립 추론 |
| D9 | device 기본 auto: CUDA 없으면 CPU + `W_CPU_FALLBACK`. `--device cuda` 명시 시 CUDA 없으면 오류 |
| D10 | 입력 파일명 기본 미기록, SHA-256만. `--record-input-names` 시에만 basename |
| D11 | 기술적 경고(임상 기준 아님): `W_LARGE_LESION` (> 300 mL 또는 brain의 30% 초과), `W_MANY_COMPONENTS` (> 50). 결과 변경 없음 |
| D12 | Validation smoke 허용 오차: CPU끼리 mask 완전 일치, CUDA는 Phase 4 예측 대비 불일치 voxel ≤ 0.1%, Dice 재현 ±0.001 |

## 2. 아키텍처

```
scripts/infer_case.py ── (향후 FastAPI도 같은 함수 호출)
        ▼
neuroscan_ai.inference.run_independent_inference()          service.py
  1 sealed_guard.check_not_sealed   파일을 열기 전, 경로 문자열의 ISLES case ID → SplitGuard
  2 validation.validate_inputs      error → 중단 / warning → 기록 후 진행 (--strict: error)
  3 model_loader                    weights_only=True, sha256, config hash·context·채널·전처리 버전 검증
  4 preprocess.prepare_case         reorient → XY 1 mm → brain mask → clip_zscore (메모리, Phase 1/3 함수)
  5 predict.predict                 predict_case(context=1, loss_fn=None) → restore_to_original
  6 quantify.quantify               lesion_summary·components + bbox/world/반구(estimated)/점수 요약
  7 outputs.RunDirectory            staging → 저장 후 재로드 검증 → 계약 검사 → os.rename 확정
        ▼
InferenceResult (JSON 직렬화 가능, 예외 대신 status/exit_code 반환)
```

엔진은 web·DB·인증 패키지를 import하지 않으며(테스트로 강제), 출력은 하지 않고 logger
`neuroscan_ai.inference`와 `progress(stage, fraction)` callback만 사용한다. 설정은
`configs/inference_v1.yaml`(checkpoint, 학습 config, 기대 해시, 검증·경고 기준) 한 곳에 모았다.

Phase 1~4 코드 수정 없음 — 재사용: `io.nifti`, `preprocessing.geometry.reorient/resample_to_spacing`,
`training.normalization`, `training.brain_gating`, `training.dataset.CaseVolumes/norm_specs`,
`training.model.build_model`, `training.inference.predict_case`, `reconstruction.restore_to_original`,
`analysis.lesion_summary/components`, `data.split_guard.SplitGuard`, `training.seed.seed_everything`.

## 3. 입력 계약과 검증

skull-strip된 DWI(b1000) + ADC, 같은 격자, `.nii`/`.nii.gz`, 3-D (4-D는 t=1만 허용).

| 판정 | code | 조건 |
|---|---|---|
| error | `E_INPUT_NOT_FOUND` / `E_INPUT_FORMAT` / `E_INPUT_CORRUPT` / `E_INPUT_NOT_3D` | 파일 없음, 확장자·NIfTI header 크기, 읽기 실패(잘림·손상), 4-D t>1 |
| error | `E_GRID_MISMATCH` | shape / affine(atol 1e-3 mm) / orientation 불일치 — 5A는 ADC를 resampling하지 않음 |
| error | `E_NONFINITE` / `E_EMPTY_IMAGE` | NaN·Inf, 상수 영상, brain 내부 대비 없음 |
| error | `E_SHAPE_TOO_SMALL` / `E_SPACING_INVALID` | in-plane < 32 또는 slice < 3; spacing < 0.5, in-plane > 5, slice > 10 mm, header zooms ≠ affine |
| error | `E_BRAIN_MASK` | brain 비율 5–60% 밖 (skull-strip 안 된 영상) |
| error | `E_SEALED_TEST_CASE` | 경로에 Test split case ID — **파일 open 전에** 거부 |
| warning | `W_SPACING_OUT_OF_TRAIN_RANGE`, `W_FEW_SLICES`, `W_ORIENTATION_UNUSUAL`, `W_OBLIQUE` (>1°), `W_BRAIN_FRACTION_UNUSUAL` | 학습 분포(XY 0.875–2.0, Z 2.0–5.0 mm, slice ≥ 16, LAS, brain 14.7–26.6%) 밖 |
| warning | `W_POSSIBLE_SWAP` | 강도 범위가 DWI↔ADC 뒤바뀐 배정에만 맞음 (입력은 변경하지 않음) |
| warning | `W_ADC_UNIT_UNUSUAL`, `W_ADC_NEGATIVE` | ADC brain 중앙값이 학습 3개 단위군(×0.5–×2) 밖; 음수 > 5% |
| warning | `W_HEADER_TEXT_PRESENT` | `descrip`/`aux_file`/`db_name`/`intent_name`/extension 존재 — **필드 이름만** 기록 |

## 4. 출력 계약

```
outputs/<case_id>/<run_id>/            run_id = <UTC yyyymmddThhmmssZ>-<8 hex>
├── prediction_mask.nii.gz             uint8 {0,1}, 원본 DWI shape·affine, qform/sform code 1, 새 header
├── quantification.json                neuroscan.quantification/1
├── inference_metadata.json            neuroscan.inference_metadata/1
├── manifest.json                      neuroscan.manifest/1 (Viewer·API용, 상대 URI)
├── inputs/{dwi,adc}.nii.gz            voxel data + affine으로 새로 만든 비식별 사본 (공유 기본 제외)
└── logs/inference.log                 경로 비식별 로그
```

- 덮어쓰기 금지: staging·final·failed 폴더가 모두 없을 때만 시작, 확정 직전 재확인, `--overwrite` 없음.
- 모든 NIfTI는 저장 후 재로드하여 shape·affine·dtype·값·header(글자 필드·extension 없음) 검증.
- 모든 JSON은 저장 전 계약 검사: 필수 키, 절대 경로 없음, 입력 파일명 없음, GT·Dice·"질병 확률" 키와
  금지 표현(뇌졸중 확률, 진단 확정, 혈관 영역명) 없음, 고지문 포함. 위반 시 `E_OUTPUT_VERIFY`.
- 실패: `.<run_id>.failed/`에 `inference_metadata.json`(status failed, error)과 `logs/`만 남김.
- `case_id`: 기본 `case-<uuid12>`, 지정 시 `^[A-Za-z0-9_-]{1,64}$` — 입력 파일명으로 만들지 않음.

**quantification.json** — 원본 DWI 격자에서 계산: `lesion_detected`, `voxel_count`,
`total_lesion_volume_ml`, `component_count`, `largest_component_volume_ml`, `affected_slice_count`(S/I 축),
`bounding_box_voxel/world_mm`, `center_of_mass_voxel/world_mm`(RAS+ scanner mm, atlas 아님),
component별 부피·bbox·중심·`hemisphere_estimate`, `hemisphere`(아래), `research_score_summary`(아래),
`flags`, `vascular_territory: null`(atlas 정합 없이 추정하지 않음), `disclaimer`(영/한).

**좌우 반구 (D3, estimated)** — brain mask의 world-x 무게중심을 midline으로 추정하고, brain bbox 중심과
5 mm 이내·좌우 폭 비대칭 10% 이하일 때만 채택. 병변 voxel의 world x(RAS+: +x = 환자 우측)를
midline ± 5 mm band 기준으로 left/right/midline 분류. 기울기 > 10° 또는 검증 실패 시 `not_available`
(사유 기록). 배열 인덱스는 사용하지 않으며 머리 회전은 보정하지 않는다.

**inference_metadata.json** — case/run ID, UTC 시각, 단계별 시간, 모델(이름·유형·training run·epoch·
validation Dice 주석·checkpoint/config 상대 경로·SHA-256·config hash), 소프트웨어(Git commit·tracked
변경 여부·패키지 버전), 입력(SHA-256·형식·크기; 파일명은 opt-in), input QC 통계, original/processed
geometry, 전처리 변환 목록, 추론 설정(context 1, 6채널 순서, threshold 0.5, gating, fp32, batch, 시도
기록, 작은 component 제거 없음, 결정성 설정), device, Test guard 결과, warnings/errors, 출력 SHA-256,
`ground_truth_used: false`, `intended_use`. 개인정보 필드 없음.

**manifest.json** — volume별 상대 `uri`·SHA-256·`mime_type`(application/gzip)·shape·spacing·affine·
orientation·dtype, mask label 의미와 `same_grid_as: dwi`, axial/coronal/sagittal 배열 축(affine 축 코드),
3D 렌더링 가능 여부(`empty_mask`), 결과 JSON 경로, 병변 요약, 공유 기본 대상/제외 목록, 고지문.
로컬 절대 경로를 노출하지 않으며 API가 `uri`를 파일 ID로 바꿔 제공한다.

## 5. 신뢰도 표현

`research_score_summary`는 **보정되지 않은 모델 출력 점수**(gating 후 sigmoid, processed 격자)의 기술
지표다: 예측 영역 평균·중앙·최대 점수, 0.5–0.7(low-score)·0.7–0.9·≥0.9 비율, brain 내 0.3–0.7 구간
voxel 수와 비율, brain 내 최대 점수. `calibrated: false`와 "질병 확률이 아님" 주석을 항상 포함한다.
환자 단위 "병변 확률", "뇌졸중 확률", % 표시, 확실/불확실 판정은 calibration과 독립 검증 없이는
제공하지 않는다. 미검출은 "모델이 병변 영역을 표시하지 않음 — 병변이 없다는 의미가 아님"으로 표현한다.

## 6. Exit code

0 성공(미검출 포함) · 1 내부 오류 · 2 인자(`case_id` 등) · 3 입력 검증·strict 경고 · 4 모델/config ·
5 전처리/brain mask · 6 device/추론(OOM) · 7 출력 저장·검증 · 8 출력 폴더 존재 · 9 봉인 Test case ·
130 중단. stdout은 요약(또는 `--json` 한 줄), stderr는 진행·경고·오류(경로 없음).
CUDA OOM은 batch 64 → 16 → 4로 재시도 후 `--device auto`일 때만 CPU 1회 재시도.

## 7. 검증 결과

**테스트**: 신규 70개 (입력 검증·Test guard 20, 파이프라인 23, 출력 계약 23, Validation smoke 4) +
기존 139개 = **209 passed**; ruff check / format 통과.

- 전처리 동등성(D2): 합성 3종 spacing(2.0 iso / 1.8×4.8 / 1.1458×4.8)과 실제 Validation 2명
  (0086 2 mm iso, 0199 XY 1.146 mm resampling) 모두 모델 입력·brain mask·processed affine **bit 일치**.
- 실제 checkpoint: `weights_only=True` 로드, config hash `34d18df104da` 일치, 파라미터 2,639,305.
- Validation smoke (D12): Phase 4 공식 예측 대비

| case | 장치 | 불일치 voxel | Dice (5A) | Dice (Phase 4) | 병변 부피 | component | 반구 |
|---|---|---|---|---|---|---|---|
| 0086 | CPU ×2 | 0 / 915,712 (CPU끼리 완전 일치) | 0.94677 | 0.94677 | 48.98 mL | 9 | estimated (좌 39.44 / 우 0.54 mL) |
| 0086 | CUDA | 0 / 915,712 | 0.94677 | 0.94677 | | | |
| 0199 | CPU ×2 | 0 / 1,105,920 (CPU끼리 완전 일치) | 0.86767 | 0.86767 | 10.77 mL | 9 | not_available (기울기 13.2° > 10°) |
| 0199 | CUDA | 0 / 1,105,920 | 0.86767 | 0.86767 | | | |

  런타임(모델 로드 후): CUDA 0.5–1.1 s, CPU 0.7–2.3 s.
- 개인정보: 실제 0199 입력의 header 글자와 합성 `PATIENT^NAME`·`Hong_Gildong_*` 파일명이 출력(압축 해제한
  NIfTI header 포함)·로그·JSON 어디에도 없음. 절대 경로 없음.
- 기존 결과 무결성: `results/`·`models/` 230개 파일 SHA-256 실행 전후 동일.

## 8. 한계

- 입력 범위: skull-strip된 ISLES형 DWI+ADC만. 비 skull-strip 영상은 `E_BRAIN_MASK`로 막지만, brain 비율이
  우연히 5–60% 안에 드는 비표준 입력은 통과할 수 있다. 다른 기관·장비는 domain shift 가능.
- Test guard는 경로 문자열의 case ID에 의존한다. ID가 없는 이름으로 복사된 Test 영상은 식별할 수 없다
  (연구 규칙으로 금지).
- 반구 추정은 scanner 좌표 기반이며 머리 회전 보정이 없다. ISLES 다수 case가 1–10° 기울어져 있고,
  10° 초과(예: 0199)는 `not_available`.
- 점수 요약은 보정되지 않았다. 모델 성능은 Validation(36명, checkpoint 선택에 사용)에서만 측정되었고
  Test는 미평가.
- CUDA OOM 후 CPU 재시도 시 재사용 중인 `LoadedModel`은 CPU에 남는다(API 계층에서 재로딩 필요).
- ADC를 DWI 격자로 resampling하지 않는다(격자 불일치는 오류).

## 9. 후속 연구 기능 (구현하지 않음)

- probability map NIfTI (D6): calibration(reliability diagram, ECE) 후 연구용 옵션으로 검토.
- atlas 정합 기반 혈관 영역(MCA/ACA/PCA)·정밀 반구 판정.
- ADC→DWI 격자 resampling, 비 skull-strip 입력용 brain extraction 정책.
