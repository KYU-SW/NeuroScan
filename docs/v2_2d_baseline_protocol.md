# v2 2D Baseline 프로토콜

> 상태: **설계·학습 Hyperparameter 확정, Training Runner는 합성 검증 단계.** 실제 학습, 추론,
> Threshold 탐색은 아직 실행하지 않았고 각각 별도 승인 대상이다. Hyperparameter·Runner·산출물·실패
> 기록은 `docs/v2_2d_baseline_run_protocol.md`에 있다.

- 코드: `ai/neuroscan_ai/v2/` — `unet_2d`(모델), `geometry`(1 mm 격자·역변환),
  `baseline_preprocess`(전처리), `baseline_outputs`(원본 격자 출력), `safety_metrics`(평가·안전 지표),
  `baseline_selection`(Checkpoint 선택·Threshold 탐색), `baseline_slices`(Padding·valid_region_mask·
  Masked Loss·Padding Crop 추론).
- Config: `configs/v2/2d_baseline.yaml`. 테스트: `tests/test_v2_baseline_geometry.py`,
  `tests/test_v2_baseline_safety.py`, `tests/test_v2_baseline_policies.py`(합성 배열·CPU만).
- 실행 통제: `docs/v2_run_control_protocol.md`(Training 출력 Profile `baseline_v1`),
  Manifest: `docs/reproducibility_protocol.md`.
- v1 학습·추론·전처리·분석·Final Test 코드를 import하거나 수정하지 않는다(테스트로 확인).
  Smoke Runner(`neuroscan_v2_2d_smoke_v1`)와 기존 Smoke 결과는 바꾸지 않는다.

## 1. 데이터 범위

ISLES 2022만 쓴다. v1 Split의 train 176건으로 학습하고 val 37건으로 선택·평가한다(Split 파일은
읽기 전용). `test`·`final_test` Split과 SOOP(파일럿 포함)은 학습·선택·Threshold 탐색 어디에도 쓰지
않는다. 공개 출력은 Case Key(`case_0001` …)만 쓰며 원본 ID 대응표는 Git 제외 `_internal/`에만 둔다.

## 2. 모델 `neuroscan_v2_2d_baseline_v1`

| 항목 | 값 |
|---|---|
| 구조 | 4-level 2D U-Net, Channel [36, 72, 144, 288] |
| Block | (Conv3×3 → InstanceNorm2d(affine) → ReLU) × 2 |
| Down / Up | MaxPool2d(2) / ConvTranspose2d(2, stride 2) → Skip concat → Block |
| 입력 / 출력 | 2채널(DWI z-score, ADC z-score) / 1 logit → Sigmoid Probability |
| Trainable Parameter | **2,439,577** (Build 시 확인, 다르면 거부) |
| 입력 크기 | H·W가 8의 배수(3회 Pooling). 아니면 거부 |
| 초기화 | 명시 Seed, Pretrained Weight 없음 |

Slice 입력은 끝쪽 0 Padding으로 8의 배수를 맞추고, 출력은 원래 크기로 Crop한다(왕복 무손실).
세부 규칙은 §4A를 따른다.

## 3. 전처리 `preprocess_v2_isles_1mm_inplane_v1`

1. **원본 검증**: DWI·ADC·GT가 3D이고 Shape·Affine(1e-3 mm)이 같으며, Affine이 유한·비특이이고
   DWI·ADC가 유한값인지 확인한다.
   - GT 해석 정책 `binary_mask_decode_v1_atol_1e_6`: NIfTI scaling이 적용된 decoded 값을 쓴다
     (`get_unscaled()` 미사용). |값 − 0| ≤ 1e-6이면 0, |값 − 1| ≤ 1e-6이면 1로 스냅한다. 그 밖의 값
     (0.6, 0.999, 음수 Label, NaN·Inf 등)이 하나라도 있으면 거부한다. `> 0.5` 같은 단순 Threshold는
     쓰지 않는다. 예: int16 + scl_slope 저장 GT의 0.9999999997671694는 1로 스냅된다.
   - 사례별 기록(`gt_decode`): decoded 고유값, 스냅된 값 종류, 스냅 Voxel 수, 최대 보정량,
     허용오차, 거부 값 수, 정책 ID. GT 원본 배열·파일은 수정하지 않는다.
2. **Brain Mask `skull_stripped_background_isles_v2`는 원본 격자에서 만든다**: DWI가 일정 배경값
   (8개 모서리 최빈값)과 다른 Voxel, 전체의 5–60% 범위 검사.
3. **대상 격자(사례당 하나, 네 영상 공통)**:
   - 배열 축 0·1이 In-plane, 축 2가 Slice 축이다.
   - 각 축의 방향 Cosine을 그대로 둔다(사선 Affine은 사선으로 유지하고 재방향하지 않음).
   - In-plane 간격은 1.0 mm로 바꾼다. 축마다 raw = 크기 × 간격 / 1.0을 구하고,
     `np.isclose(raw, round(raw), rtol=1e-6, atol=1e-6)`이면 round(raw)(Header 부동소수 오차 스냅,
     예: 2.00000006 mm × 112 → 224), 아니면 Coverage 보존을 위해 ceil(raw)를 쓴다. 최소 1.
     무조건 round하지 않는다. 축별 `raw_target_size`, `nearest_integer_size`, `final_target_size`,
     `integer_snap_applied`, 원본·대상 Coverage와 차이(mm)를 `target_size`로 기록한다.
   - Slice 축의 간격·크기·위치는 그대로 둔다(Z 미보간).
   - Voxel 0의 바깥 모서리를 원본과 일치시킨다. 대상 격자는 원본 시야를 모두 덮으며, 먼 쪽 끝으로
     1 Voxel 미만 넓어질 수 있다.
   - 원본이 이미 1 mm In-plane이면 대상 격자가 원본과 같고, 값을 그대로 복사한다(`resampled: false`).
4. **보간**: 대상 Voxel 중심을 World 좌표를 거쳐 원본 격자로 보낸다(`inv(src) @ dst`).
   - DWI·ADC는 Linear, GT·Brain Mask는 Nearest로 보간한다.
   - 원본 밖 표본은 가장 가까운 경계 Voxel 값을 쓴다.
   - Nearest 결과는 정확히 0/1이다.
5. **정규화**: 대상 격자 Brain Mask 안 통계로 DWI·ADC를 사례별 z-score한다. 입력 정규화일 뿐이며
   저장된 ADC 값은 변환하지 않는다.
6. 원본 배열·파일은 수정하지 않는다. 원본·대상 격자의 Shape·Affine·간격·축 방향과 보간 방식을
   전처리 기록에 남긴다(Run 산출물 `training/preprocessing_record.json`).

### Config·코드 정합 검사

`configs/v2/2d_baseline.yaml`의 `preprocessing.gt_decode`, `preprocessing.target_grid`,
`safety_metrics.thick_slice_policies`(ISLES 소그룹과 SOOP 잠정 경고는 별도 블록)는
`baseline_preprocess.validate_policy_config`가 코드 상수와 비교하며, 다르면 `PolicyConfigError`로
거부한다. Baseline Runner는 Gate 전에 이 검사를 호출해야 한다. (v2 모듈 소스에는 SOOP 접근 방지 검사
때문에 SOOP 정책 ID 문자열을 두지 않으며, 그 ID는 합성 테스트가 고정한다. 경고 블록의 수치·provisional
여부·ISLES와의 분리는 코드가 검사한다.)

## 4. 출력 정책 (원본 격자)

**모든 평가와 Component 분석은 원본 공간(원본 격자)에서 한다.**

### Brain Mask 역할 (확정)

| Mask | 용도 |
|---|---|
| 원본 격자에서 직접 만든 Brain Mask | **원본 공간의 유일한 기준**: 최종 Gated Prediction, Brain Mask 경계 분석, 원본 공간 QA |
| 1 mm 모델 격자 Brain Mask | 입력 정규화, 모델 입력 QA |
| Nearest 왕복 복원 Mask (모델 → 원본) | 원본 Mask와의 XOR 비교, 왕복 보간 QA **전용** |

- 금지 1: 왕복 Mask로 원본 Mask를 교체하지 않는다.
- 금지 2: 왕복 Mask로 원본 Mask를 덮어쓰지 않는다.
- 금지 3: 왕복 불일치를 조용히 수정하지 않는다.
- 사례마다 다음 QA 지표를 기록한다:
  - `roundtrip_mask_xor_voxels`
  - `roundtrip_mask_xor_ratio`: XOR Voxel 수 ÷ 원본 Mask Voxel 수
  - `roundtrip_mask_dice`
  - `roundtrip_mask_exact`
- 원본 간격이 1 mm 미만이면 왕복 불일치가 생길 수 있다. 이 불일치는 그대로 기록하며, Gating에는
  언제나 원본 Mask를 쓴다.

### 출력 규칙

- Probability는 모델 격자에서 원본 격자로 **Linear** 복원한다. 값은 [0, 1]을 유지하고, 범위를 벗어난
  입력은 Clip하지 않고 거부한다.
- Threshold는 **원본 격자에서** 적용한다: `ungated = probability ≥ threshold`.
- Gating에는 원본 격자 Brain Mask만 쓴다(위 표 참조).
- `gated = ungated AND brain_mask`. **공식 출력은 Ungated**이고 Gated는 비교용이다. Brain Mask 밖
  예측은 Voxel 수로 기록하며 Ungated 출력에서 지우지 않는다.
- 사례별 출력 기록(원본 격자, Ungated 기준, 커밋된 `component_analysis.boundary` 재사용):
  `outside_brain_mask_voxels`, `outside_brain_mask_ratio`(Brain Mask 밖 Voxel ÷ Ungated Voxel),
  `touches_image_edge`(Image-edge 접촉), `min_distance_to_brain_boundary_mm`,
  `boundary_warnings`(`BOUNDARY_CONTACT`, `EXTENDS_OUTSIDE_BRAINMASK`). 경계 접촉 기준(1 In-plane
  Voxel 또는 3 mm)은 잠정값이며 임상 판정이 아니다. 예측이 비어 있으면 비율과 거리는 `null`이다.
- 복원된 모든 배열은 원본과 Shape·Affine·축 방향(Orientation)·간격이 같아야 하고, Mask는 정확히
  0/1이어야 한다. 위반은 오류이며 값을 고쳐서 통과시키지 않는다.

## 4A. Padding과 valid_region_mask

H·W를 8의 배수로 맞추기 위해 각 In-plane 축의 끝쪽을 0으로 채운다.

- 채우는 값:
  - 정규화 DWI 0.0, 정규화 ADC 0.0 (사례별 정규화 평균값)
  - GT 0, Brain Mask 0
- `valid_region_mask`: 실제 영상 영역은 1, Padding 영역은 0이다.
- **Training Loss는 `valid_region_mask` 안에서만 계산한다.**
  - BCE: 표본마다 유효 Voxel 평균을 낸 뒤 Batch 평균을 낸다.
  - Soft Dice: 표본마다 유효 Voxel에서 Dice를 계산한 뒤 Batch 평균을 낸다.
- 위 규칙에 따라 다음이 보장된다:
  - Padding Voxel은 Background 학습 Voxel이 아니다.
  - Padding 면적은 표본의 Loss 가중치를 바꾸지 않는다.
  - Padding 영역의 Logit은 Loss와 Gradient에 영향을 주지 않는다.
- Soft Dice Smooth 1.0과 두 Loss의 결합 비율은 학습 Hyperparameter로, 아직 승인 전이다.
- 추론 순서:
  1. Slice별로 Padding한 뒤 모델을 실행하고 Sigmoid를 적용한다.
  2. **Padding을 Crop해** 모델 격자 Shape로 복원한다.
  3. 그 뒤 원본 격자로 역변환한다(§4).
- Padding 영역의 추론 결과는 저장하지 않는다.

### 작은 병변 왕복 확인 (합성)

원본 간격이 1 mm 이상(1.0, 1.1, 1.3×1.7, 1.5×1.25, 2.0 mm)이면 실제 Pipeline 경로가 1 Voxel과 2×2
병변을 모두 보존한다. 이 경로는 GT를 Nearest로 모델 격자에 보내고, 그 위의 완벽한 예측을 Linear로
복원한 뒤 0.5로 Threshold한다. 확인 결과:

- Nearest Mask 왕복은 정확하다.
- 복원 Probability의 최솟값은 약 0.55였다(1.1 mm, 1 Voxel). 0.5에 가까우므로 작은 병변은
  Threshold 선택에 민감하다.
- 원본 격자 Probability를 Linear로 두 번 보간하면 1 Voxel 값이 0.5 아래로 내려갈 수 있다
  (1.5×1.25 mm에서 0.469). Pipeline은 이런 이중 보간을 하지 않는다.
- 원본 간격이 1 mm 미만이면 모델 격자로 갈 때 Down-sampling되므로 1 Voxel 미만 크기의 구조가 사라질
  수 있다. 이것은 1 mm 정책의 알려진 한계다.

### 원본 Voxel보다 작은 예측의 복원 한계 (확정 기록)

위 확인은 GT 크기 이상의 예측을 다룬다. 모델 격자에서 원본 Voxel보다 작은 확률 패턴은 다음 한계가 있다.

- 1 mm 모델 격자에서 원본 격자로 Linear 복원할 때, 원본 Voxel보다 작은 확률 패턴은 약화될 수 있다.
- 2 mm 원본 격자에서 모델 격자의 단일 Voxel(1 mm) 패턴은 복원 후 최대 확률이 0.5 미만일 수 있다
  (실제 ISLES 전처리 QA의 합성 패턴: 2 mm 원본 0.25, 1.8 mm 원본 0.36; 0.875 mm 원본은 0.88로 유지).
- Threshold는 원본 격자에서 적용하므로 이런 작은 후보는 Binary 결과에서 사라질 수 있다.
- 이는 모델 성능과 별개의 기하·표본화 한계다. 이를 이유로 임의 보정이나 Morphology를 추가하지 않는다.
- 감시: Lesion-wise Recall과 작은 병변(0.1·0.5·1.0 mL 미만) Recall로 반드시 감시하고, Probability
  Map은 항상 보존한다. Threshold 탐색(§7) 후보별 기록에는 0.1·0.5·1.0 mL 미만 구간의 Missed 소병변 수를 함께 남긴다(이미 구현).

## 5. 평가와 안전 지표 (원본 격자, Ungated 기준)

- Dice는 사례 단위로 계산한다. 공식 요약은 GT 양성 사례의 평균 Dice다.
- Component는 26-연결로 나누며, ID 규칙은 `component_analysis.extraction`과 같다.
- **Overlap ≥ 1 Voxel 매칭은 탐색적 후보다(`overlap_ge_1_voxel_exploratory_not_clinical_tp`).** 임상적
  True Positive로 해석하지 않는다. 쌍마다 다음을 기록한다:
  - Overlap Voxel 수
  - Prediction·GT Overlap 비율
  - Dice, IoU
- 예측과 겹치지 않는 GT Component는 Missed GT로 따로 기록한다(개수와 부피 분포).
- **FP 계열 지표 4종은 서로 다른 개념이며 섞지 않는다.** 어느 것도 임상적 False-positive 병변
  수가 아니다.

  | 지표 | 정의 | 수준 |
  |---|---|---|
  | `false_positive_voxel_volume_ml` | `Prediction AND NOT GT` 부피 | Voxel-level overprediction. Checkpoint Tie-break에 쓴다 |
  | `no_gt_overlap_component_volume_ml` | GT와 1 Voxel도 겹치지 않는 Prediction Component의 부피 합 | Component-level no-GT-overlap 후보 |
  | `no_gt_overlap_component_count` | 위 Component의 수 | Component-level no-GT-overlap 후보 |
  | `oversegmentation_volume_ml` | GT와 겹치는 Prediction Component 중 GT 밖 Voxel의 부피 | Voxel-level, 겹치는 Component 한정 |

  항등식: `false_positive_voxel_volume_ml` = `no_gt_overlap_component_volume_ml` +
  `oversegmentation_volume_ml`.
- 요약 지표:
  - Lesion-wise Recall(탐색적)
  - Missed GT 수와 부피(최소·중앙·최대)
  - 0.1·0.5·1.0 mL 미만 GT의 Recall
  - 사례 단위 Sensitivity
  - 무병변 사례의 FP 부피와 예측이 있는 무병변 사례 수
  - 위 FP 계열 지표 4종(사례 합계)
  - Thick·Thin Slice 소그룹 `isles_v1_qa_thick_subgroup_3mm`: 유효 Z 간격 ≤ 3.0 mm는 thin,
    > 3.0 mm는 thick(ISLES 분석용 소그룹). SOOP 관련 잠정 경고 `soop_provisional_thick_warning_5mm`
    (Component Safety와 같은 규칙, 기록 키 `component_safety_provisional_thick_warning`; 유효 Z > 5.0 mm 또는
    Slice 두께 ≥ 5.0 mm)는 별도 provisional 정책이며 두 기준을 같은 의미로 쓰지 않는다. 둘 다
    임상 Threshold가 아니다. 예: Z 4.8 mm는 ISLES thick 소그룹이지만 SOOP 5 mm 경고 대상은 아니다.
  - Ungated−Gated 차이
- Ungated와 Gated 요약은 섞지 않는다. FP 기준값은 최적화하지 않는다.
- 원본 GT가 빈 사례는 정상 음성 사례(`gt_case_status_v1`)다. 사례별 Dice·Lesion-wise Recall·Missed GT·
  Small-lesion Recall은 `not_applicable`, FP 지표는 항상 계산한다(`docs/v2_2d_baseline_run_protocol.md`).

## 6. Checkpoint 선택 (val, Threshold 0.5, Ungated)

1. 1차 기준은 GT 양성 Validation 사례의 평균 Ungated Dice이며, 높을수록 좋다.
2. 최고값과 1e-4 이내인 Epoch들 중에서는 Validation Loss가 낮은 쪽을 고른다.
3. 그다음은 `false_positive_voxel_volume_ml`(Voxel-level overprediction)이 낮은 쪽이다. 이 값을
   False-positive 병변 수로 표현하지 않는다.
4. 마지막으로 더 이른 Epoch를 고른다.

Threshold가 0.5가 아니거나 Gated 기준인 기록은 거부한다. 선택 결과는
`training/training_history.json`의 `selection`과 Checkpoint의 `best_metric`에 기록한다.

## 7. Threshold 탐색 (Checkpoint 고정 후 1회)

- Checkpoint를 고정한(SHA-256 기록) 뒤 val에서 한 번만 탐색한다. 후보는 0.10–0.90을 0.05 간격으로
  나눈 17개다.
- 1차 기준은 양성 사례 평균 Ungated Dice이고, 동률이면 0.5에 가까운 값을 고른다.
- 후보마다 Lesion Recall, 크기 구간별 Missed 소병변 수, `false_positive_voxel_volume_ml`, 무병변 FP
  부피를 기록하되
  최적화하지 않는다.
- 같은 Validation 사례로 고르고 보고하므로 낙관적 편향이 있다. 이 사실을 기록에 명시한다.
- Test·Final Test는 쓰지 않는다. val 외 Split을 요청하면 거부한다.
- 기록 `training/threshold_search.json`은 한 번만 쓰고, 다시 쓰려 하면 거부한다. 이 탐색은 학습 Run에
  포함되지 않으며(`baseline_v1` 필수 산출물 아님), 별도 승인 후 고정 Checkpoint로 실행한다.

## 8. 실행 통제 연결

Training Run은 Runner의 `training_outputs = "baseline_v1"` Profile로 다음 산출물을 필수로 요구한다.

- `best_checkpoint`·`last_checkpoint`·`training_history`·`validation_metrics`·`threshold_record`·
  `sampler_record`·`preprocessing_record`(모두 `training/` 아래)와 실행 단위 Aggregate
  (`aggregate_results/aggregate.json`) 및 Validation 사례별 출력
- 실패 Run은 `training/failure_record.json`을 남기며, 있으면 Post-run Manifest에 Hash가 기록된다.

하나라도 없으면 `failed_incomplete_outputs`이다. 알 수 없는 Profile은 Gate 전에 거부한다.
Smoke Runner의 기본 Profile `smoke_v1`과 기존 Smoke 결과는 바뀌지 않는다.

## 9. 아직 하지 않은 것 (별도 승인 필요)

- 실제 ISLES Train·Val 데이터 준비(Alias 대응표·Split)와 정식 학습 실행
- 연속 붕괴(예: 여러 Epoch 연속 빈 Prediction) 중단 정책
- Validation Threshold 탐색 실행
- Test·Final Test·SOOP 평가
