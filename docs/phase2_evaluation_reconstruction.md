# Phase 2: Reconstruction, Restoration, Evaluation, Quantification

작성일 2026-10-17 · 기준 커밋 `97113bf` · 결과 파일 `results/phase2/` (Git에 포함하지 않음)

## 1. 목적

2D/2.5D 모델이 공통으로 쓸 다음 경로를 **모델과 독립적으로** 구현하고 검증합니다.

```
Slice Prediction → Processed 3D Volume → Original Space → GT 비교 → Patient-level Metrics
```

모델이 아직 없으므로 GT 자기검증과 합성 Prediction으로 엔진 자체가 정확한지를 확인합니다.

| 모듈 | 책임 |
|---|---|
| `reconstruction/volume.py` | `(z, 2D slice)` → 환자 Volume, Slice 검증, 역연산 `decompose_volume` |
| `reconstruction/restore.py` | processed → original 격자 복원 (binary와 probability 정책 분리) |
| `evaluation/segmentation.py` | `evaluate_case(prediction, ground_truth, spacing)`, probability 이진화 |
| `evaluation/negative_cases.py` | 빈 GT Case 평가 |
| `evaluation/report.py` | Case 단위 결과 집계, subgroup, 결과 폴더 범위 |
| `analysis/lesion.py`, `analysis/components.py` | 병변 정량과 component 분석 (분석만 하고 mask는 바꾸지 않음) |
| `data/case_metadata.py` | subgroup metadata (z-spacing, 부피, ADC 단위, 기울기, 장비) |
| `scripts/validate_phase2.py` | 실제 GT 250명 검증 → `results/phase2/` |

이 모듈들은 모델이나 학습 코드를 import하지 않습니다. 재사용한 기존 기능은 `preprocessing.geometry.resample_to_grid`,
`preprocessing.mask.binarize/lesion_volume_ml`입니다. component 계산은 `analysis.components`로 옮겨서
QA와 정책 분석도 같은 함수를 쓰게 했습니다(QA 결과 250명 모두 동일한 것을 확인).

## 2. Reconstruction 정의

`reconstruct_volume(slices, geometry, kind, slice_axis=2)`

- 입력은 `(z_index, 2D array)` 쌍입니다. **위치는 입력 순서가 아니라 index로 정해집니다.**
- 출력은 `PredictionVolume(data, geometry, kind)`입니다. `geometry`(shape와 affine)를 그대로 갖고 있으므로
  shape, affine, spacing, orientation이 보존됩니다.
- 오류로 처리하는 경우:
  - Slice 누락
  - Slice 중복
  - Slice shape가 다름
  - index 범위 밖이거나 정수가 아님
  - binary에 0/1 이외의 값
  - probability에 NaN/inf 또는 [0,1] 밖의 값
- `kind`는 `binary`(uint8)와 `probability`(float32) 두 가지입니다. **Reconstruction은 threshold를 하지 않습니다.**

## 3. Processed → Original 복원

`restore_to_original(volume, original_geometry)`는 `resample_to_grid`로 world 좌표를 거쳐 원래 격자에서
다시 샘플링합니다. LAS↔RAS 뒤집기, XY spacing 변경, oblique 회전이 모두 affine만으로 되돌려집니다.
원본 shape와 affine은 meta.json의 `original` 또는 원본 NIfTI header에서 가져옵니다.

Phase 1 정책 B'에서는 Z를 voxel 그대로 유지했기 때문에, processed Slice k가 원본 Slice k와 정확히
대응합니다. 그래서 복원은 XY 방향에서만 보간이 일어납니다.

## 4. Binary / Probability 보간 정책

| kind | 보간 | 결과 |
|---|---|---|
| binary | {0,1}을 linear 보간한 뒤 **≥ 0.5** | 정확히 {0,1} (label 보존). 국소 부분부피 다수결과 같은 의미이고, 전처리와 같은 규칙이라 Phase 1에서 측정한 경로를 그대로 거꾸로 따라감 |
| probability | linear 보간 | 연속값을 유지하고 [0,1] 범위 안에 있음 (볼록 결합). **threshold하지 않음** |

Nearest neighbour를 쓰지 않는 이유: 2mm↔1mm처럼 원본 voxel 중심이 processed voxel 두 개의 정확히
중간에 놓이는 경우가 많아서, nearest는 반올림 방향에 따라 결과가 한쪽으로 치우칩니다. 검증 결과
probability 경로를 복원한 뒤 ≥ 0.5로 자르면 binary 경로와 250명 모두 voxel 단위로 같았습니다.

Probability를 binary로 바꾸는 것은 evaluation/inference 정책입니다:
`binarize_prediction(prob, threshold=0.5)` (`prob >= threshold`, 기본값 `DEFAULT_PROBABILITY_THRESHOLD`,
모델 단계에서는 config로 정함). 참고로 GT 원본의 이진화(`> 0.5`)는 Phase 1에서 정한 별도 규칙입니다.

## 5. Metric 정의

TP/FP/FN은 voxel 수입니다.

| Metric | 정의 | 정의되지 않는 경우 |
|---|---|---|
| Dice (Primary) | 2TP / (2TP+FP+FN) | GT가 비어 있을 때 |
| IoU | TP / (TP+FP+FN) | GT가 비어 있을 때 |
| Sensitivity | TP / (TP+FN) | GT가 비어 있을 때 |
| Precision | TP / (TP+FP) | **Prediction이 비어 있을 때** (0/0) |
| Volume error (mL) | pred − GT (부호 있음, 양수면 과분할) | 항상 정의됨 |
| Absolute volume error (mL) | \|pred − GT\| | 항상 정의됨 |
| Relative volume error (%) | 100 × (pred − GT) / GT | GT가 비어 있을 때 |

정의되지 않는 값은 `None`으로 두고 **0이나 1로 바꾸지 않습니다.** 집계에서는 빼고 `n_undefined`로 따로 셉니다.

## 6. Empty GT 정책

대상: `sub-strokecase0150`, `0151`, `0170` (삭제하지 않음, Split마다 1명씩)

| GT | Prediction | 결과 | 보고 값 |
|---|---|---|---|
| 비어 있음 | 비어 있음 | `correct_negative` | 예측 voxel 0, 부피 0 |
| 비어 있음 | 있음 | `false_positive_case` | `predicted_lesion_voxels`, `predicted_lesion_volume_ml`, `any_false_positive`, `false_positive_volume_ml`, `false_positive_component_count` |

- 빈 GT에서는 Dice/IoU/Sensitivity/Precision을 계산하지 않습니다(`None`). **빈 GT와 빈 Prediction에
  Dice=1을 주는 관례는 쓰지 않습니다.**
- 집계 방식: positive 환자 통계(Dice 평균 등)는 `has_lesion=True`인 환자만으로 계산합니다. negative 환자는
  `correct_negative`와 `false_positive_cases` 개수, 그리고 FP 부피 목록으로 따로 보고합니다. 두 통계를 하나의
  평균으로 섞지 않습니다.
- Negative가 3명뿐이라 Specificity 같은 비율을 대표 지표로 쓰지 않습니다.

## 7. 부피 계산

```
volume_mm3 = voxel_count × spacing_x × spacing_y × spacing_z      (spacing = affine 열 벡터 크기)
volume_mL  = volume_mm3 / 1000
```

**공식 환자 병변 부피는 원본 공간 spacing으로 계산합니다.** Viewer에 표시할 기본값도 원본 공간 부피입니다.
processed 부피는 참고용입니다.

일관성 검증 결과 (positive 247명):

| 비교 | 평균 \|상대오차\| | 최대 |
|---|---|---|
| processed vs original | 0.29% | 7.13% |
| restored vs original | 0.076% | 5.55% (`0248`, XY 1.146mm) |

## 8. Connected Component 분석

- 26-연결(`analysis.components.CONNECTIVITY_26`)을 프로젝트 전체에서 사용합니다.
- component별로 voxel 수, 부피(mL), bounding box(min/max index 포함), centroid(voxel 좌표와 world mm)를
  계산하고 크기순으로 정렬합니다.
- 환자 단위로는 voxel 수, 부피, 병변이 있는 Slice 수, 첫 Slice와 마지막 Slice를 계산합니다.
- **분석만 하고 후처리는 하지 않습니다.** 작은 component도 지우지 않고 모두 보고합니다.

## 9. 원본 공간 평가 원칙

평가는 **원본 GT(원본 격자, `> 0.5`로 이진화)** 와 비교합니다. processed 공간 예측은 반드시
`restore_to_original`을 거친 다음 평가합니다. 이렇게 하면 resampling 손실도 성능에 정직하게 반영됩니다.

## 10. Phase 1 Spacing 정책과의 관계 (GT 왕복 검증)

경로: 원본 GT → processed/v1 mask → Slice 분해 → 재구성 → 원본 복원 → 원본 GT와 비교 (250명)

| 그룹 | n (pos/neg) | Dice 평균 | Dice 최소 | 사라진 component / 전체 | 부피 \|상대오차\| 평균 |
|---|---|---|---|---|---|
| 전체 | 247 / 3 | 0.99929 | 0.9682 | 68 / 2349 | 0.076% |
| A. Thin Z (2mm) | 193 / 3 | 0.99979 | 0.9758 | 1 / 1504 | 0.009% |
| B. Thick Z (4.4–5mm) | 54 / 0 | 0.99748 | 0.9682 | 67 / 845 | 0.317% |
| C. 병변 < 1 mL | 43 / 0 | **1.0** | 1.0 | 0 / 79 | 0% |
| D. Negative GT | 0 / 3 | – | – | – | 3/3 correct_negative |
| E. Oblique ≥ 20° | 55 / 0 | 0.99885 | 0.9682 | 32 / 595 | 0.129% |
| Test Split | 36 / 1 | 0.99943 | 0.9864 | 5 / 421 | 0.092% |

- **Phase 1 측정값(B': 평균 0.9993, 최소 0.9682, thin 1개, thick 67개 사라짐)과 정확히 일치합니다.**
  250명 모두 Phase 2 경로의 Dice가 Phase 1 방법으로 독립적으로 다시 계산한 값과 같았고(차이 < 1e-12),
  저장된 processed mask도 다시 계산한 결과와 voxel 단위로 같았습니다. 모순은 없습니다.
- 기울기 분포: < 1° 2명, 1–10° 80명, 10–20° 113명, ≥ 20° 55명 (최대 37.7°). 원본 affine 중
  248명이 oblique입니다. 병변 centroid 이동은 최대 0.45 mm로, 최소 voxel 크기보다 작습니다.
- **완전히 같은 경우:** XY 2.0mm 191/191명, XY 1.797mm 29/29명은 Dice 1.0입니다.
- **손실의 원인:** 사라진 68개 component는 모두 XY 1.146mm 그룹(67개)과 0.875mm 그룹(1개)에서 나왔고,
  크기는 1 voxel 58개, 2 voxel 9개, 4 voxel 1개입니다. 비율이 정수가 아닐 때(1.146→1.0mm)
  linear 보간 후 ≥ 0.5 규칙이 1–2 voxel 크기 component를 지키지 못하는 경우입니다. 38개는
  processed mask(학습 GT)에서 이미 없고, 30개는 복원할 때 사라집니다. component 수 차이가 0인 환자는
  235/250명입니다.
- **허용 오차의 근거:** 통합 테스트는 (1) Phase 1 방법과 voxel 단위로 같을 것, (2) 환자별 Dice ≥ Phase 1
  B'의 최소값 0.9682, (3) XY 2mm 환자는 Dice = 1.0을 요구합니다. 임의로 정한 값은 없습니다.

## 11. External Validation 확장 구조

- 엔진은 데이터셋 이름을 받지 않습니다. 배열, `VolumeGeometry`, spacing만 받습니다.
- 외부 데이터셋은 `data/raw/external/<dataset>/` → 어댑터 → 같은 전처리 → 같은 재구성, 복원, 평가 순서로
  처리합니다.
- 결과 폴더는 `evaluation.report.results_dir(root, scope, ...)`로 정합니다. scope는
  `internal_test` 또는 `external_validation`이고, 다른 값은 오류로 처리합니다.
- Subgroup metadata는 QA 표의 범용 컬럼과 BIDS sidecar에서 만듭니다. ISLES'22에는 center 정보가 없어서
  `center = "unavailable"`로 두고, sidecar가 있는 Case(144명, manufacturer가 기록된 것은 129명)만 manufacturer, model, 자기장 세기를 기록합니다.
  ADC 단위 그룹은 크기로 분류만 할 뿐 정규화하지 않습니다(Phase 3에서 처리).

## 12. Phase 2 테스트 결과

- `pytest`: **72 passed** (Phase 0/1의 기존 30개 + Phase 2의 신규 42개)
  - `test_reconstruction.py` (18): Slice 순서, 누락, 중복, shape, index 범위와 타입, binary/probability 값 검증,
    binary와 probability 재구성, affine/spacing/orientation 보존, 원본 복원, probability 복원, oblique 왕복
  - `test_evaluation.py` (12): 시나리오 A–H, threshold 설정, 입력 검증, positive/negative 분리 집계, 결과 폴더 범위
  - `test_analysis.py` (8): 부피, 병변 Slice, 빈 mask, 작은 병변, 26-연결, bounding box, centroid, subgroup 분류
  - `test_phase2_data.py` (4, 실제 데이터): Test Split 37명 재구성, 원본 공간 왕복(Phase 1과 일치),
    oblique ≥ 30° 위치 보존, 실제 Case probability 복원
- `validate_phase2.py`:
  - 재구성: 250명 binary와 probability 모두 voxel 단위로 같음. shape, affine, spacing 보존. 모두 RAS
  - Test Split 37명: positive 36명 Dice/IoU/Sensitivity/Precision 모두 1, 부피 오차 0, negative 1명 correct_negative
  - 합성 metric 테스트 397개 모두 통과 (A 72, B 72, C 72, D 60, E 72, H 43, F 3, G 3)
  - Negative 3명: 빈 예측은 correct_negative. 원본 공간에 2×2×1 FP를 넣으면 0.032 mL(기대값과 같음),
    processed에 4×4 mm FP를 넣고 복원해도 0.032 mL, 1 component

## 13. Limitations (승인된 상태로 유지)

- **XY resampling에 따른 작은 component 손실:** processed/v1(XY 1.0mm, Z 유지)에서는 XY 비율이 정수가 아닌
  그룹(XY 1.146mm 23명, 0.875mm 2명)에서 1–2 voxel 크기 component 68/2349개가 사라질 수 있습니다.
  그중 38개는 processed mask(학습 GT)에 이미 없고, 30개는 원본 공간으로 복원할 때 사라집니다.
- **원본 공간 왕복 Dice 상한:** 완벽한 processed 공간 예측이라도 원본 공간에서는 평균 Dice 0.99929,
  최소 0.9682(thick 그룹 평균 0.99748)가 상한입니다. XY 2.0mm(191명)과 1.797mm(29명)은 상한이 1.0입니다.
  평가는 원본 GT 기준이므로 이 손실도 성능 수치에 그대로 반영됩니다.
- **결정:** processed/v1을 유지하고 processed/v2는 만들지 않습니다(Phase 2 승인 시 결정). 손실을 줄이려면
  전처리 mask 규칙을 바꾼 새 버전이 필요하며, 필요해지면 따로 검토합니다.
