# NeuroScan v1 Final Test 프로토콜 (사전 등록)

> 연구 및 교육 목적의 프로토타입이며, 임상 사용이 승인된 의료기기가 아닙니다. 이 문서는 최종 내부 Test 평가를
> **실행하기 전에** 평가 방법을 고정하기 위한 사전 등록 문서이며 Test 결과를 포함하지 않습니다.

- 기계가 읽는 동결 값: [`configs/final_test_v1.yaml`](../configs/final_test_v1.yaml) (manifest)
- 실행 스크립트: `scripts/evaluate_final_test.py` (로직: `ai/neuroscan_ai/final_test/`)
- 보호 장치 테스트: `tests/test_final_test.py` (합성 데이터만 사용)

## 1. 확정 결정 (2026-11-02 사용자 승인)

| 항목 | 결정 |
|---|---|
| 비교 모델 | 저장된 단일 seed(42) 공식 run: 2D `2d_baseline_20261017-133753_38dfcea89a50` (epoch 42), 2.5D `25d_baseline_20261024-192617_34d18df104da` (epoch 49) |
| Seed 43·44 | 이번 최종 Test 범위에서 제외. 다중 seed 실험은 Test 이후 별도 후속 연구 |
| 강도 증강 R1 재학습 | 공식 비교에서 제외. 현재 모델·학습 정책을 그대로 동결하고 한계·후속 ablation으로 기록 |
| 평가 경로 | 학습 Validation과 같은 경로: processed/v1 → `predict_case` → 원본 공간 복원 → 원본 GT 비교. Phase 5A 서비스 경로는 공식 평가에 쓰지 않음 |
| Primary endpoint | positive Test 환자의 환자 단위 평균 Dice, 같은 환자에서 2.5D − 2D 짝지은 비교 |
| Secondary endpoint | IoU, Sensitivity, Precision, 부호 있는 부피 오차, 절대 부피 오차, 상대 부피 오차, 예측·GT component 수 — 탐색적(효과 크기·분포 중심, p-value로 결론 내지 않음) |
| Negative 환자 | positive 평균에서 제외하고 별도 보고(correct negative 여부, FP voxel·부피·component 수). 정의되지 않는 지표를 0 또는 1로 바꾸지 않음 |
| Subgroup | lesion volume group, slice spacing group, ADC unit group, obliquity group, center — n < 10이면 기술통계만 |
| 실패 case | 환자 단위 Dice < 0.50 (어느 모델이든). QA·실패 분석 전용이며 재조정 조건이 아님 |
| QA 이미지 | 2.5D 개선 상위 3명, 악화 상위 3명, Dice < 0.50 전부, FP가 생긴 Negative |
| 작은 분리 영역 10% 기준 | Viewer 표시 규칙일 뿐, 지표·제거·제외에 쓰지 않음 |

## 2. 동결 값

| 대상 | 값 |
|---|---|
| 2D checkpoint | `models/phase3/2d_baseline_20261017-133753_38dfcea89a50/best.pt` · SHA-256 `e0d2512780468bcee51806bb75f259790877470042c7c8069b1bd12386939b59` · config_hash `38dfcea89a50` |
| 2.5D checkpoint | `models/phase4/25d_baseline_20261024-192617_34d18df104da/best.pt` · SHA-256 `ff61bc7f86655b7addee3298eefacbcfe40a1a872e361148ce2609f5f24adbe7` · config_hash `34d18df104da` |
| `configs/2d_baseline.yaml` | `381ede5dcc0059c2009af35983540aa0bc2926d6cc2e92e71177fc6ae330f5f2` |
| `configs/25d_baseline.yaml` | `773dc21fefaed49e01a7d43daf425ee8d93657b98175af1f7e8de7713d48c469` |
| `configs/preprocess_v1.yaml` | `c4d18b85dede08a57bebd0e646c538bcad7ec5cd5ad113448fc705da4dae5a74` |
| `configs/inference_v1.yaml` | `4b8f26b7521f27380fff411b3982c83fdff901a34c15cd90f83c4435f314ed80` |
| `data/splits/split_v1_seed42.json` | `c086c6508d1608d8521b97150f82a299492d8d32d0691bd46695766749dcc65e` (176 / 37 / 37, seed 42) |
| 입력 | DWI + ADC. 2D `[DWI z, ADC z]`, 2.5D `[DWI z−1, z, z+1, ADC z−1, z, z+1]` (modality 우선, 가장자리 slice 복제) |
| 전처리 | processed/v1 (RAS, XY 1.0 mm, Z 원본 유지) |
| 정규화 | DWI·ADC 모두 case별 뇌 voxel 기준 clip_zscore (0.5 / 99.5 percentile) |
| Brain mask | `skull_stripped_background`, 추론 시 gating, 뇌 없는 slice는 학습 sampling에서만 제외 |
| Threshold / 후처리 | 0.5 / 없음 (작은 component 제거 없음) |
| 정밀도 / device | fp32 / CUDA 사용 가능 시 CUDA, 아니면 CPU |
| 통계 | 짝지은 차이의 percentile bootstrap 95% CI (10,000회, seed 42), Wilcoxon signed-rank 양측 |

**해시 규칙**: yaml·json은 CRLF→LF 정규화 후 SHA-256(= Git이 저장하는 바이트). 작업 폴더의 줄바꿈과 무관하다.
checkpoint(Git 미포함)는 바이트 그대로 SHA-256. 승인 해시(`--approve-test`)도 manifest의 LF 정규화 SHA-256이다.

## 3. Test Ground Truth의 과거 제한적 사용

> Test Ground Truth는 데이터 분할 층화와 재구성·원본 공간 복원 엔진 검증에 제한적으로 사용되었다.
> Test Prediction 생성, 모델 선택, Threshold 조정, 후처리 선택 또는 성능 기반 정책 결정에는 사용되지 않았다.
> 최종 모델 성능 평가는 아직 수행되지 않았다.

구체적으로: Phase 1 QA·분할 층화(전체 환자의 병변 부피), Phase 2 `scripts/validate_phase2.py`(모델이 아닌 합성 예측으로
재구성·복원·지표 엔진 검증), `tests/test_phase2_data.py`(엔진 통합 테스트, 이후 봉인 정책으로 pytest에서 제외).
"Test Split에 한 번도 접근하지 않았다"는 표현은 쓰지 않는다.

## 4. 실행 절차와 보호 장치

모든 거부는 평가 대상 split의 파일을 열기 **전에** 일어난다(exit 2 = 사전 점검·승인, 3 = 실행 상태).

1. **사전 점검**: checkpoint 2개·config 4개·split SHA-256, checkpoint에 기록된 config_hash·epoch·채널 수,
   동결 정책(manifest = 코드 = 두 학습 config: threshold 0.5, gating, 후처리 없음, fp32, seed, 정규화, 2D↔2.5D는
   `context_slices`만 다름), Git(작업 트리 clean, `frozen_commit`, lock tag).
2. **승인 (final-test만)**: `--mode final-test`, `--approve-test <manifest SHA-256>`, 확인 문구
   `I APPROVE THE NEUROSCAN V1 FINAL TEST`(대화형 입력, 또는 자동 환경에서는 `--non-interactive --confirm-phrase`).
   기본값은 항상 거부. 출력 경로는 manifest의 `results/final_test/v1`로 고정.
3. **Git 동결 방식**: commit은 자기 hash를 담을 수 없으므로 ① 평가 코드·manifest를 커밋(= frozen commit) →
   ② manifest의 `git.frozen_commit`만 채운 커밋 → ③ 그 커밋에 `neuroscan-v1.0.0-test-locked` tag. 평가기는 HEAD가
   frozen commit의 후손이고 그 사이 변경이 manifest 파일뿐이며, tag가 HEAD를 가리킬 때만 실행한다.
4. **1회 실행 보호**: `LOCK`·`COMPLETED`·`FAILED`·결과 파일·prediction 폴더가 있으면 중단. 평가 대상 case를
   읽기 직전에 `LOCK`을 원자적으로(O_CREAT|O_EXCL) 생성. 결과는 `.attempt-<n>/`에 쓰고 성공 시 옮긴 뒤
   `COMPLETED`. 실패 시 `FAILED` + 부분 출력은 `failed_attempts/attempt-<n>/`. 완료된 실행은 다시 하지 않는다.
5. **재실행(기술적 실패만)**: `--resume-after-failure --failure-report <파일> --reason "<사유>"`. 이전 시도의
   LOCK·FAILED·보고서는 `failed_attempts/`에 보존되고 감사 기록에 남는다. 수치를 본 뒤 정책을 바꿔 재실행하는 것은
   금지이며, 정책을 바꾸면 해시·승인 값이 달라져 실행 자체가 거부된다.
6. **SplitGuard(allow_test=True)** 는 `ai/neuroscan_ai/final_test/` 안에서만 쓰인다(테스트로 강제).
   dry-run은 sealed guard로 Validation만 읽는다.

## 5. 출력 (Git 제외: `results/**`)

| 파일 | 내용 | 공개 범위 |
|---|---|---|
| `aggregate_metrics.json`, `statistical_comparison.json`, `subgroup_metrics.json` | 집계 결과 | 보고서에 반영 가능 |
| `patient_metrics_2d.csv`, `patient_metrics_25d.csv`, `paired_metrics.csv`, `negative_case_summary.json`, `qa_manifest.json`, `qa_images/` | 환자 단위 결과 | 로컬 전용 |
| `predictions_2d/`, `predictions_25d/` | 원본 공간 예측 NIfTI | 로컬 전용 |
| `execution_audit.json` | 승인·시작·종료 시각, commit, tag, manifest·checkpoint·config·split 해시, 사전 점검 결과, 접근 case 수, 성공·실패 수, 재시도·중단 사유, 실행 명령, 모델별 시간·VRAM (case ID 없음) | 보고서에 요약 가능 |
| `environment.json` | Python·PyTorch·CUDA·cuDNN·MONAI 등 버전, GPU·VRAM, seed 설정 | 보고서에 요약 가능 |
| `audit_internal/case_access.json` | 접근한 case ID와 guard 접근 기록 | 로컬 전용 |

콘솔에는 case ID를 출력하지 않으며, final-test 모드에서는 지표도 출력하지 않는다.
Test 예측은 Research Mode에 등록할 수 없다(연구 case 등록은 Test case key를 파일을 열기 전에 거부).

## 6. Validation dry-run (2026-11-02)

`PYTHONPATH=backend .venv\Scripts\python.exe scripts\evaluate_final_test.py` (기본 모드) — Validation 37명
(positive 36 + negative 1), Test 접근 0건, 출력 `results/final_test_dry_run/v1/`.

| 대조 | 결과 |
|---|---|
| 2D 환자별 지표 vs Phase 3 `validation_metrics.csv` (37명) | Dice·IoU·Sensitivity·Precision 최대 차이 ≈1e−16, voxel 수·component 수 차이 0 |
| 2.5D 환자별 지표 vs Phase 4 `validation_metrics.csv` (37명) | 동일 (최대 ≈1e−16, 부피 오차 ≈1e−14 mL) |
| 짝지은 비교 vs Phase 4 comparison `summary.json` | 평균 0.7309 / 0.7428, Δ +0.0119, 95% CI [−0.0017, +0.0284], Wilcoxon p = 0.286 — 차이 ≤ 1e−17 |
| 실행 시간 / VRAM | case 로딩 6.6 s, 2D 6.9 s (0.58 GB), 2.5D 8.0 s (0.64 GB), RTX 5060 Ti |

재실행은 `COMPLETED`로 거부되고(exit 3), final-test 모드는 승인·Git 동결 없이 거부됨(exit 2)을 확인했다.

## 7. 알려진 제한

- `center` 열은 현재 메타데이터에서 모두 `unavailable`이라 center subgroup은 사실상 단일 그룹이다.
- Test negative 환자는 1명이므로 FP 경향에 대한 비율 지표는 내지 않는다.
- CUDA 일부 연산 비결정성: Validation 재현은 ≤1e−16 차이로 일치했지만 bitwise 재현을 보장하지는 않는다.
- Git 동결 검사는 로컬 저장소 기준이며 원격 tag push 여부는 별도 확인 절차로 둔다.
- 보호 장치는 연구 규칙을 코드로 강제하는 것이지 악의적 우회(파일 직접 열기 등)를 막는 보안 장치가 아니다.

## 8. 남은 단계 (각 단계 사용자 승인 필요)

1. 이 변경 검토 후 커밋(= frozen commit 후보).
2. manifest `git.frozen_commit` 기입 커밋 → `neuroscan-v1.0.0-test-locked` tag.
3. 최종 dry-run(Git 동결 점검 포함 전부 통과 확인).
4. Final Test 1회 실행 → 집계 결과 보고.
