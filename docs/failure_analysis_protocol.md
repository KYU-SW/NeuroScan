# NeuroScan v1 실패 사례 정성 분석 프로토콜 — 1단계 Metrics-only · 2단계 영상 기반

> 연구 및 교육 목적의 프로토타입이며, 임상 사용이 승인된 의료기기가 아닙니다. 이 분석은 고정 내부 평가 결과의
> 검토를 돕는 **기술적 관찰**이며 의료적 판단이나 Ground Truth의 옳고 그름에 대한 판단이 아니고, v1 모델·정책을
> 바꾸는 근거로 쓰지 않습니다.

- 코드: `ai/neuroscan_ai/failure_analysis/`, 실행 `scripts/analyze_failure_cases.py`
  - 1단계(§1–§6): `schemas.py`, `selection.py`, `taxonomy.py`, `audit.py`, `run.py`
  - 2단계(§7): `image_review.py`, `figures.py`, `image_run.py`, `review.py`
- 테스트: `tests/test_failure_analysis.py` (합성 표), `tests/test_failure_image_review.py` (합성 영상)
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

## 7. 2단계 — 영상 기반 분석 (2026-11-02 승인·1회 실행)

### 7.1 목적

- 1단계에서 선정된 실패·개선·악화·비교 사례(alias)의 객관적 영상 자료 생성
- 2D와 2.5D의 TP·FP·FN, component, 공간적 차이 관찰
- NeuroScan v2에서 검증할 수 있는 가설 후보 도출
- v1 결과나 모델을 수정하지 않는 사후 정성 분석

### 7.2 명시적 승인

```
scripts/analyze_failure_cases.py --mode image-review --approve-image-review "<승인 문구>"
승인 문구: I APPROVE READ-ONLY FAILURE ANALYSIS OF SELECTED TEST CASES
```

문구가 없거나 다르면 어떤 영상도 열기 전에 거부한다(기본값은 항상 거부). 추가 사전 점검: 1단계 결과
(`selection.json`, `audit.json`, `_internal/aliases.json`)가 있고 ID 검사를 통과했을 것, alias 대응표와 selection이
일치할 것, alias ≤ 10, Final Test `COMPLETED` 있음·`FAILED` 없음, 출력 폴더가 아직 없을 것.

### 7.3 허용 입력과 금지

허용(선정된 alias 최대 10건, 읽기 전용): Test DWI, Test ADC, Test Ground Truth, Final Test에서 저장된 2D·2.5D binary
prediction, 1단계 사례 요약의 지표·메타데이터, 원본 경로 확인용 `data/processed/v1/<case>/meta.json`.
실행 중 `nibabel.load`는 정확히 이 파일들(사례당 5개)만 열 수 있도록 허용 목록으로 제한되며, 다른 파일은 거부·집계된다.

금지: 선정되지 않은 Test case 접근, 모델 재추론, probability map 생성, threshold 변경, 후처리 실험, checkpoint·config
변경, Final Test 재실행, v1 지표 수정, Research Viewer 등록, 환자별 자료의 공개 또는 Git 포함.

### 7.4 정성 분석 기준 (원본 격자, 26-연결, 거리 mm)

| 코드 | 기준 | 표시 |
|---|---|---|
| T1 | Sensitivity < 0.5 이고 FN voxel > FP voxel | flagged |
| T2 | Precision < 0.5 이고 FP voxel > FN voxel | flagged |
| T3 위치 불일치 | 예측 foreground가 있고 TP = 0 또는 GT overlap ratio(TP / GT voxel) < 0.10 | flagged |
| T4 작은 GT component 놓침 | 부피 < 0.1 mL 이고 matching 유형이 `ground_truth_only` | flagged |
| T5 떨어진 FP component | `prediction_only` component 이고 최근접 GT까지 > 10 mm | flagged |
| T6 경계 불일치 후보 | TP > 0 이고 FP·FN voxel의 50% 초과가 GT 경계로부터 2 mm 이내 (FP: GT까지 거리, FN: GT 바깥까지 거리) | review_candidate |
| T7 분할·병합 | matching group이 split / merge / complex | flagged |
| T8 모델 간 문맥 차이 | 1단계 규칙(thick slice 이고 2.5D가 2D보다 0.001 넘게 낮음)을 그대로 사용 | review_candidate |
| T9 신호 추가 검토 | ① FP 영역 DWI 중앙값 > brain reference DWI Q3 이고 FP 영역 ADC 중앙값 < brain reference ADC Q1, 또는 ② FN 영역 DWI 중앙값이 brain reference DWI의 [Q1, Q3] 안 | review_candidate |

병변 없는 사례는 T1–T8을 적용하지 않는다(`not_applicable`). T6·T9는 판별력이 낮을 수 있는 검토 후보 표시이며,
실제로 다수 사례에서 표시되었다. 이 기준들은 지표 재정의, 후처리, component 삭제, 의료적 판단에 사용하지 않는다.

### 7.5 생성 자료 (사례별)

- `maps/difference_2d.nii.gz`, `maps/difference_25d.nii.gz`: 0 배경, 1 TP, 2 FP, 3 FN (`research.compare`
  재사용). 저장된 Final Test TP·FP·FN·GT·예측 voxel 수와 일치해야 하며, 다르면 그 사례는 중단(`stopped`)한다.
  격자·affine이 GT와 다른 경우도 중단한다. 저장 NIfTI는 원본 header 없이 affine만 담는다.
- `maps/model_difference.nii.gz`: 0 배경, 1 둘 다, 2 2D만, 3 2.5D만 (GT 관계와 분리)
- `components_gt.csv`, `components_2d.csv`, `components_25d.csv`: component ID(크기 순위), voxel 수, 부피, slice 범위,
  대표 voxel(component 내부), 대표 world mm, bounding box, 최근접 상대 component 거리, brain mask 경계 거리
  (작은 component도 제거하지 않음)
- `matching_2d.json`, `matching_25d.json`: `research.matching` 겹침 그래프(one_to_one, split, merge, complex,
  prediction_only, ground_truth_only), overlap·union voxel, IoU
- `intensity.json`: §7.6
- slice 선정: GT 최대 → FN 최대 → 2D FP 최대 → 2.5D FP 최대 → 모델 차이 최대 → 주요 component 대표 slice 순,
  중복은 합치고 최대 5개, 각 중심 slice의 z−1·z·z+1 (범위 밖은 "out of range"로 표시)
- `figures/`: Panel A (4×3: DWI, ADC, 2D TP/FP/FN, 2.5D TP/FP/FN × z−1, z, z+1), Panel B (DWI + GT 윤곽 + 둘 다 /
  2D만 / 2.5D만), Panel C (대표 voxel을 지나는 3면 × 2D·2.5D). 뇌 voxel 1–99 percentile 밝기, Research Mode 의미
  색(TP #009E73, FP #E69F00, FN #56B4E9, GT #CC79A7), 텍스트 범례, 10 mm 축척, voxel 간격 비율. 배열 index 방향으로
  그리며 영상의학적 방향 표기가 아니다. 제목·파일명에는 alias와 slice 번호만 쓴다.

### 7.6 신호 정규화

학습과 같은 함수(`training.normalization.normalize`, case별 뇌 voxel 기준 clip-zscore, 0.5 / 99.5 percentile)를
원본 격자의 DWI·ADC에 적용한다. 뇌 mask는 학습과 같은 `skull_stripped_background` 정책이다. 영역: TP, FP, FN,
brain reference(뇌 안에서 GT와 두 모델 예측을 모두 뺀 영역). 통계: voxel 수, 중앙값, Q1, Q3. 빈 영역은 0이 아니라
정의되지 않음(null)으로 둔다. 신호 차이는 의료적 확정 근거가 아니다.

### 7.7 관찰과 가설 분리 (`cases/<alias>/summary.json`)

`observations`(ID O-NN, 자동 측정값만 `observed`) · `taxonomy_flags` · `review_candidates` · `hypotheses` ·
`alternative_explanations` · `unknowns` · `suggested_experiments`. 가설은 flag에 연결된 **템플릿**만 자동으로 만들며
(T8, T4, T5, T9), 각각 근거 observation ID, 대안 설명, 추가 검증 방법, 확신 수준(`possible` 또는 `undetermined`)을
가진다. 원인을 확정하지 않는다. 사용하지 않는 표현: 오진, 확정 진단, 실제 병변 확정, Ground Truth 오류 확정.

### 7.8 출력과 감사 기록

```
results/failure_analysis/v1/image_review/      (Git 제외: results/**)
├── audit.json · aggregate_observations.json
├── _internal/case_access.json                alias ↔ 원본 ID, 범주별 파일 열기 수 (로컬 전용)
├── cases/fa-NN/  summary.json · intensity.json · components_*.csv · matching_*.json · maps/ · figures/
└── contact_sheets/                           §7.9
```

결과는 임시 폴더에 쓴 뒤 검사(공개 파일의 원본 ID·파일명·로컬 경로 0건, Final Test·1단계 파일 불변, 거부된 파일
열기 0건)를 통과해야 제자리로 옮겨지며, 기존 결과는 덮어쓰지 않는다. `audit.json`: 승인 시각·종료 시각, Git HEAD,
lock tag와 그 commit, Final Test 결과 스냅샷 해시, 1단계 selection 해시, 승인 문구 확인, 선정·접근 alias 수(최대 10),
DWI·ADC·GT·2D 예측·2.5D 예측 열기 수, 거부된 시도 수, 모델 추론 0, probability map 0, torch 미사용, 지도·그림 수,
공개 출력 ID·경로 검사, Final Test·1단계 불변, Git 제외 여부.

### 7.9 Contact sheet와 사용자 QA

- 이미 생성된 PNG만 묶은 로컬 검토 자료(영상·NIfTI 재접근 없음): alias별 sheet 10장(대표 Panel A·B·C + alias, 그룹,
  Dice, 차이, T1–T9, 두께·부피·ADC 그룹), overview 2장, 검토 체크리스트(md·json). 생성 스크립트는 저장소 밖의 일회성
  도구이며 커밋하지 않는다.
- 사용자 육안 검토 완료, 최종 판정 **PASS WITH NOTE**. 영상·overlay·정렬·색상·범례·개인정보 기준은 통과했고,
  자동 결과와 이미지 사이에 명백한 모순이 없었다.
- 공통 표시 제한: 모든 alias sheet의 Panel C 아래쪽 행 첫 번째 제목이 위 그림과 일부 겹친다. 영상·overlay·공간
  위치·분석 결과에는 영향이 없으며, 이 표시 문제를 고치기 위해 Test 영상을 다시 열거나 이미지를 재생성하지 않았다.
- 체크리스트에는 의료적 판단을 기록하지 않는다.

### 7.10 로컬 전용 정책

alias 대응표, contact sheet, QA PNG, 차이 NIfTI, 사례별 JSON·CSV, 환자별 지표, DWI·ADC·GT·prediction, 내부 감사
파일은 Git에 포함하지 않는다. 모두 `results/**`의 기존 ignore 규칙으로 제외되며(`git check-ignore`로 확인),
Git에는 코드, 합성 테스트, 이 프로토콜만 포함한다.

### 7.11 분석 한계

- 사후 선정된 10건의 정성 분석이며 일반화할 수 없다. 원인과 인과관계를 확정할 수 없다.
- probability map이 없어 threshold 근처의 행동을 분석할 수 없다.
- 신호 패턴의 의료적 의미는 판단하지 않는다.
- 일부 사례에서 thick slice와 ADC 단위(및 기관)가 같은 사례에 겹쳐 효과를 분리할 수 없다.
- T6·T9 규칙은 판별력이 낮을 수 있다.
- 이 평가 세트는 이미 평가·분석에 쓰였으므로 v2의 독립 평가에는 새로운 데이터가 필요하다.
