# NeuroScan v1 실패 사례 정성 분석 프로토콜 — 1단계 Metrics-only

> 연구 및 교육 목적의 프로토타입이며, 임상 사용이 승인된 의료기기가 아닙니다. 이 분석은 고정 내부 평가 결과의
> 검토를 돕는 **기술적 관찰**이며 진단, 오진 판정, Ground Truth의 옳고 그름 판단이 아니고, v1 모델·정책을 바꾸는
> 근거로 쓰지 않습니다.

- 코드: `ai/neuroscan_ai/failure_analysis/`, 실행 `scripts/analyze_failure_cases.py`
- 테스트: `tests/test_failure_analysis.py` (합성 표만 사용)
- 입력: 완료된 `results/final_test/v1/`의 결과 표(읽기 전용) — 동결된 `final_test` 패키지·manifest·결과는 수정하지 않는다.

## 1. 범위 (1단계)

| 허용 | 금지 |
|---|---|
| `patient_metrics_2d.csv`, `patient_metrics_25d.csv`, `paired_metrics.csv`, 집계·subgroup·negative JSON, `qa_manifest.json`, 표에 저장된 메타데이터 | Test DWI·ADC·GT·Prediction NIfTI 열기, QA 이미지 생성, 모델 추론, probability map, Final Test 재실행, v1 결과·정책 변경 |

실행 중에는 `nibabel.load`가 차단되며(시도 시 거부·집계), Final Test 폴더의 모든 파일 해시를 실행 전후로 비교한다.

## 2. 사례 선정 (사전 고정, 결정적)

| 그룹 | 규칙 | 정렬·동점 |
|---|---|---|
| F | positive, 2D 또는 2.5D Dice < 0.50 | 두 모델 중 낮은 Dice 오름차순 → 내부 case 순서 |
| Z | positive, 2.5D Dice = 0 (F와 중복 가능, 별도 사유 기록) | 내부 case 순서 |
| I | Dice 차이(2.5D − 2D) 상위 3 | 차이 내림차순 → 내부 case 순서 |
| W | Dice 차이 하위 3 | 차이 오름차순 → 내부 case 순서 |
| N | 병변 없는 환자 | 내부 case 순서 |
| G | 두 모델 모두 Dice ≥ 0.85인 사례 중 thin 1, thick 1 | 그 slice 그룹 후보의 (2D+2.5D)/2 중앙값에 가장 가까운 사례 → \|차이\| 작은 쪽 → 내부 case 순서 |

Alias는 `fa-01`부터 F → Z → I → W → N → G 순서로 처음 나타날 때 부여한다. 여러 그룹에 속한 사례는 alias 하나에
모든 그룹과 사유를 기록한다.

## 3. Metrics-only 분류 (원인 판정 아님)

| 코드 | 규칙 | 1단계 |
|---|---|---|
| T1 과소 분할 | Sensitivity < 0.5 이고 FN voxel > FP voxel | 모델별 판정 |
| T2 과대 분할 | Precision < 0.5 이고 FP voxel > FN voxel (Precision 정의 불가 시 판정 안 함) | 모델별 판정 |
| T3 위치 불일치 후보 | 예측 voxel > 0 이고 TP = 0 | 모델별 판정 |
| T4 작은 GT component 놓침 (< 0.1 mL) | component 지도 필요 | `pending_image_review` |
| T5 떨어진 FP component (최근접 GT > 10 mm) | component 지도 필요 | `pending_image_review` |
| T6 경계 불일치 | FP·FN 공간 분포 필요 | `pending_image_review` |
| T7 분할·병합 차이 | 예측 − GT component 수만 관찰값으로 기록, matching 유형은 영상 필요 | `pending_image_review` |
| T8 모델 간 문맥 차이 후보 | thick slice 이고 2.5D가 2D보다 0.001 넘게 낮음 | `review_candidate` |
| T9 영상 신호·GT 불일치 후보 | DWI·ADC 필요 | `pending_image_review` |

병변 없는 사례에는 T1–T3·T8을 적용하지 않고(`not_applicable`), T4·T6은 보류 목록에서 뺀다. 위치 불일치의 겹침 기준
(GT overlap < 0.10)은 영상 단계에서 쓴다. 이 임계값들은 지표 계산, 후처리, component 제거, 평가, 의료적 판단에
쓰지 않는다.

## 4. 사례 요약 구조 (`cases/<alias>/summary.json`)

`observations`(두 모델의 지표, Dice 차이, component 수 차이, 메타데이터 그룹, negative 결과) · `taxonomy_flags` ·
`pending_image_review` · `hypotheses`(1단계에서는 항상 빈 배열 — 자동으로 원인을 채우지 않음) · `unknowns` ·
`suggested_experiments`(빈 배열). 금지 표현: 오진, 진단, GT 오류, 잘못된 라벨.

## 5. 출력과 공개 범위

```
results/failure_analysis/v1/            (Git 제외: results/**)
├── _internal/aliases.json              alias -> 원본 ID (로컬 전용)
├── _internal/case_access.json          표에서 읽은 ID와 선정 ID (로컬 전용)
├── selection.json · overview.csv · aggregate_summary.json · audit.json
└── cases/fa-NN/summary.json
```

- 원본 ID는 `_internal/`에만 있다. 공개 파일은 쓰기 전에 원본 ID 검사를 통과해야 하며(실패 시 결과를 남기지 않음),
  콘솔에는 개수만 출력한다.
- 결과는 임시 폴더에 쓴 뒤 검사를 통과하면 제자리로 옮긴다. 기존 결과가 있으면 덮어쓰지 않는다.
- Final Test 결과가 불완전하면(`COMPLETED` 없음, `FAILED` 있음, 표 누락) 실행을 거부한다.
- `audit.json`: 실행 시각, Git commit, lock tag와 그 commit, Final Test 완료 상태·불변 여부, alias 수, 그룹별 수,
  영상·NIfTI 접근 0(차단된 시도 수), 모델 추론 0, probability map 0, 원본 ID 검사, Git 제외 여부.
- Git에 포함할 수 있는 것은 코드, 합성 테스트, 이 프로토콜뿐이다.

## 6. 한계

- 사후 선정된 소수 사례이며 일반화할 수 없다. 표 수치만으로는 FP·FN의 위치, component 대응, 영상 신호를 알 수 없다
  (T4–T7·T9 보류).
- T8은 thick slice와 2.5D 저하가 함께 나타났다는 표시일 뿐, 인접 slice 문맥이 원인이라는 뜻이 아니다.
- probability map이 없어 확신도 분석을 하지 않는다.

## 7. 다음 단계 (별도 승인)

영상 기반 2단계: 대상 alias에 한해 Test DWI·ADC·GT·예측을 읽고(승인 인자·감사 기록), 차이 지도, component 표,
인접 slice·3면 그림, 신호 요약을 만들어 T4–T7·T9를 검토한다.
