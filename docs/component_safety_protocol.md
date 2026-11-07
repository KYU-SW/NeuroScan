# Component Safety Analysis 프로토콜 (NeuroScan v2, 1단계)

> 상태: 1단계 정적 분석만 구현. Verifier, Soft-Masked Pooling, LayerCAM·Attention·Occlusion,
> UI는 없다. 실제 MRI·Prediction에는 아직 실행하지 않았다. 모든 Threshold는 provisional이다.
> 이 분석은 임상 판단을 대체하지 않는다.

## 1. 목적

기존 최종 병변 Prediction을 원본 격자에서 Connected Component로 나누고, Component마다
물리·영상 특성, Brain Mask 경계 관계, 2D·2.5D 일치 정도를 기록해 임상 안전 경고를 만든다.
Prediction은 삭제·수정·잘라내기 없이 그대로 둔다.

- 코드: `ai/neuroscan_ai/component_analysis/`
  (`extraction`·`measurements`·`boundary`·`agreement`·`warnings`·`records`)
- CLI: `scripts/analyze_prediction_components.py`
- 테스트: `tests/test_component_analysis.py` (합성 데이터만)

## 2. Brain Mask와 Prediction의 역할 분리

- Brain Mask는 **정규화와 경계 거리 측정에만** 쓴다.
- Prediction을 Brain Mask로 곱하거나 잘라내지 않는다. Brain Mask 밖 Prediction Voxel도
  Component에 그대로 포함한다.
- 경계 접촉과 Brain Mask 밖 연장은 Warning으로만 기록한다.
- Brain Mask 정책 자체(SynthStrip·HD-BET·합집합 후보 등)는 별도 결정 사항이며, 레코드에는
  `brain_mask_policy_id`로 어떤 Mask를 썼는지만 남긴다.

## 3. Component 추출 규칙

- 최종 Binary Prediction(0/1만 허용)에서 원본 격자의 26-connectivity로 추출한다.
  Resampling 없음, 작은 Component 포함 어떤 Component도 삭제하지 않는다.
- Component ID는 결정적이다: Volume 내림차순 → 최소 k → 최소 j → 최소 i
  (각 축 최솟값을 따로 사용) → 그래도 같으면 (k, j, i) 사전순 첫 Voxel.
- 모든 Component의 합집합은 원본 Prediction과 Voxel 단위로 같아야 하며, 다르면 분석을 거부한다.
- v1 표시용 `analysis/components.py`(scipy label 순서로 동률 처리)와는 별개 모듈이다. v1 코드를
  호출하거나 바꾸지 않는다.

## 4. 물리 단위 측정

- Spacing은 Affine 열 벡터의 길이에서 구한다. Volume(mL) = Voxel 수 × Voxel 부피(mm³) / 1000.
- Identity: `component_id`, `voxel_count`, `volume_ml`, `bbox_ijk`, `slice_min`·`slice_max`,
  `n_slices`, `centroid_ijk`, `centroid_world_mm`(Affine 적용, 경사 Affine 포함),
  `probability_mean`·`probability_max`.
- DWI: mean·median·p05·p95·min·max (유한값만).
- ADC: 같은 통계 + `negative_voxel_count`, `background_code_voxel_count`(Background Code가
  주어졌을 때), `valid_voxel_count`. **ADC 값을 변환하지 않는다**: 단위 확정, Clip, abs,
  Offset, 음수의 0 치환 없음. 음수도 유효값으로 통계에 남긴다.
- ADC 상태 `adc_storage_status`·`adc_content_status`·`adc_unit_status`를 그대로 기록한다.

## 5. Brain Mask 경계 분석

- Voxel의 경계 거리 = 반대 영역(내부면 가장 가까운 외부, 외부면 가장 가까운 내부)
  Voxel 중심까지의 물리 거리(mm, Spacing 반영). 경계 바로 옆 Voxel은 한 Voxel 간격이다.
- 기록: `min_/median_distance_to_brain_boundary_mm`, `touches_boundary_1_inplane_voxel`,
  `touches_boundary_3mm`, `voxels_outside_brain_mask`, `outside_brain_mask_ratio`,
  `touches_image_edge`.
- Brain Mask가 배열 전체를 덮어 경계가 없으면 거리는 `null`, 비어 있으면 분석을 거부한다.
- 두께 6 mm 같은 Thick Slice에서는 3 mm 기준이 Slice 사이를 넘지 못한다는 점에 유의한다.

## 6. 2D·2.5D 일치 분석

두 모델의 Binary 결과가 모두 주어졌을 때만 계산한다.

- 비교 영역 R = 최종 Component ∪ 그 Component와 겹치는 2D·2.5D 각각의 26-연결 Component.
- 기록: 각 모델과 겹친 Voxel 수·비율, R 안에서의 2D·2.5D Dice·IoU,
  `conf_2d_mean/max`·`conf_25d_mean/max`(확률 지도가 있을 때, Component 내부).
- 상태: `both`, `2d_only`, `25d_only`, `neither_or_unavailable`(한쪽이라도 없으면 이 값).

## 7. Warning 코드

| 코드 | 조건 | Threshold |
|---|---|---|
| `BOUNDARY_CONTACT` | 경계 거리 ≤ in-plane 1 Voxel **또는** ≤ 3.0 mm (두 결과 모두 기록) | provisional |
| `EXTENDS_OUTSIDE_BRAINMASK` | Brain Mask 밖 Voxel > 0 | 정의상 고정 |
| `MODEL_DISAGREEMENT` | `2d_only`·`25d_only`(고정), 또는 `both`이면서 Dice < 0.5 | 낮은 overlap은 provisional |
| `SMALL_LESION` | Volume < 1.0 mL | provisional |
| `THICK_SLICE` | 유효 z 간격 > 5.0 mm 또는 SliceThickness ≥ 5.0 mm (SOOP 6 mm 해당) | provisional |
| `ADC_STORAGE_UNKNOWN` | ADC 상태 중 하나라도 없음·unknown·probable·suspicious | 정의상 고정 |

- `THICK_SLICE`·`ADC_STORAGE_UNKNOWN`은 Case 단위이며 모든 Component에도 붙는다.
- 각 Warning에 `provisional_threshold`와 `threshold_source`(`design_candidate_not_validated`)를 남긴다.
- 보류: `HIGH_UNCERTAINTY`(불확실성 모델 없음), `ATTRIBUTION_MISMATCH`(Attribution 없음).

## 8. Threshold 정책

현재 모든 수치 Threshold는 설계 후보이며 검증되지 않았다. 최종 값은 **Validation 데이터에서만**
고정하고, 고정한 값과 출처(Validation 실행 ID)를 레코드에 남긴다. Test 데이터로 Threshold를
고르지 않는다.

## 9. Record

- Case: alias, shape, spacing, SliceThickness·SpacingBetweenSlices·유효 z 간격, Brain Mask
  정책 ID, 2D·2.5D 모델 ID와 Weight Hash(입력으로 받은 값), 결합 정책, Threshold ID, ADC 상태,
  Component 수, Warning 요약, provisional Threshold 표, 수행하지 않은 작업 목록.
- Component: Identity, DWI·ADC 통계, 경계 지표, 2D·2.5D 일치, Warning,
  `review_state = "unreviewed"`, `reviewer_note = null`.
- Lesion GT는 개발·검증 분석에서만 선택 입력으로 받는다. 주어지면 Component에
  `development_only_gt`만 추가되고 다른 값은 그대로다. 임상 출력은 GT 없이 동작한다.
- 공개 레코드에는 alias만 넣는다. 원본 Subject ID처럼 보이는 alias, 호스트 경로, 계정 이름이
  들어가면 기록을 거부한다.

## 10. 출력과 Git

- 로컬 출력: `results/component_analysis/<alias>/component_record.json` (Git 제외).
  기존 레코드는 덮어쓰지 않는다. CLI는 이 폴더 밖이나 Git에 포함되는 위치로는 쓰지 않는다.
- Git 포함 가능: 분석 코드, 합성 테스트, 이 문서.
- Git 제외: 환자별 레코드, Prediction·Probability NIfTI, Component Mask, 통계 결과,
  사용자 Review, 실제 사례 그림.

## 11. v1 동결 영역

이 모듈은 v1 전처리·추론·학습 코드를 import하거나 호출하지 않으며 모델을 실행하지 않는다.
`configs/2d_baseline.yaml`, `configs/25d_baseline.yaml`, v1 전처리·추론·학습, 체크포인트,
Final Test Manifest, Test Lock Tag, 기존 Brain Extraction 결과, SOOP Adapter 출력과 파일럿 QA
결과는 변경하지 않는다.

## 12. 아직 없는 것

Component Verifier, Soft-Masked Pooling, LayerCAM·Slice Attention·Occlusion, 공간 일관성 검사,
UI. Warning은 Prediction을 삭제하거나 판정을 바꾸지 않으며, 설명 지도는 앞으로도 병변 경계·
Brain Mask·단독 판정 근거로 쓰지 않는다.
