# v2 2D Baseline Training Runner 프로토콜

> 상태: **합성 Runner 검증 단계.** Training Loop, Validation, 산출물, Manifest Gate, 실패 동작은 합성
> 데이터로만 검증했다. 실제 ISLES 학습·추론·Threshold 탐색은 실행하지 않았고 각각 별도 승인 대상이다.
> 합성 결과는 성능 결과가 아니다.

- 코드: `ai/neuroscan_ai/v2/` — `baseline_training`(고정 Hyperparameter·검증·Loss·1 Epoch),
  `baseline_dataset`(사례·Slice·Padding·Sampling), `baseline_inference`(Validation·붕괴 검사),
  `baseline_runner`(Gate 연결·Checkpoint·산출물·실패 기록). CLI: `scripts/run_v2_2d_baseline.py`.
- 테스트: `tests/test_v2_baseline_runner.py`(합성 NIfTI, CPU, 임시 git 저장소).
- 모델·전처리·출력·평가 정책: `docs/v2_2d_baseline_protocol.md`. 실행 통제: `docs/v2_run_control_protocol.md`.
- v1 학습·추론·전처리·분석·Final Test 코드는 import하지 않는다(테스트로 확인).

## 1. Hyperparameter (승인 확정)

| 항목 | 값 |
|---|---|
| Model | `neuroscan_v2_2d_baseline_v1`, [36, 72, 144, 288], 2,439,577 Parameter, DWI+ADC → 1 logit, 사전학습 없음 |
| Epoch | 최대 100, Early stopping patience 20(선택 Epoch가 20 Epoch 동안 바뀌지 않으면 중단) |
| Batch·Worker·Seed | 32 · 0 · 42 |
| 정밀도·결정론 | FP32, AMP 없음, TF32 matmul·convolution 없음, deterministic algorithms, cudnn benchmark 없음 |
| Optimizer | AdamW, LR 1e-3, weight decay 1e-4 |
| Scheduler | Cosine annealing(T_max 100 Epoch), 최소 LR 1e-6, Epoch마다 step |
| Loss | Masked Dice 1.0 + Masked BCE 1.0, Dice smooth 1.0 |
| Gradient clipping | L2 norm 1.0 |
| Sampling | Brain Mask가 있는 Slice, Lesion-positive 기대 비율 0.5, Seed·Epoch로 결정 |
| Augmentation | 없음 |
| Threshold | 0.5 고정, source `baseline_checkpoint_fixed_0_5`, 최적화 안 함 |

근거: Baseline은 재현 가능한 단일 기준점이 목적이다. 값은 일반적인 2D 분할 기본값(AdamW 1e-3, Cosine,
Dice+BCE 1:1, Clipping 1.0)으로 정했으며 실제 데이터로 조정하지 않았다. Config의 `training`·
`determinism` 블록은 `baseline_training.validate_hyperparameters`가 코드 상수와 값·타입까지 비교한다.
실행 장치(`device`)만 Hyperparameter가 아니며 `cpu`·`cuda`·`cuda_if_available` 중 하나여야 한다
(그 밖의 값은 Gate 전에 거부).

### Device 정책 `device_policy_v2_baseline_cuda_required_v1`

- Baseline Config의 `training.device`는 `cuda`다(다른 값이면 Config 검증에서 거부).
- Dataset Profile은 호출자가 선언하지 않고 RunPlan 입력으로 정한다(`baseline_runner.dataset_profile`).
  정식 Dataset ID, 저장소 원본 데이터 폴더 아래 입력, 정식 Split 파일, 내부 Run Split 폴더 중 하나라도
  있으면 `formal_isles`다. Dataset ID `synthetic`이고 입력이 선언됐으며 정식 표지가 없을 때만
  `synthetic`이다. 그 밖은 거부한다.
- `formal_isles`: 장치는 `cuda`만 허용한다. `cpu`·`cuda_if_available` Override와 `epoch_limit`은
  Gate 전에 거부된다(Dataset·Model 생성 0건, Run 폴더 없음, Run ID 미사용). CLI `--device`도 같다.
- `synthetic`: `cpu`·`cuda`·`cuda_if_available` 모두 허용한다(`cuda_if_available`은 개발·합성 편의용).
- `cuda`를 요청했는데 GPU를 쓸 수 없으면 CPU로 자동 전환하지 않고 실패한다(`DeviceUnavailableError`,
  실패 기록·실패 Post-run Manifest, Dataset을 열기 전). 다른 GPU를 자동 선택하지 않는다.
- Package Lock: CLI `run`은 승인된 `requirements-lock.txt` 정규화(LF) SHA-256 전체값을
  `--package-lock-sha256`로 받아 데이터 접근 전에 비교한다. 파일이 없거나 값이 다르면 거부한다.
  Config에는 복사하지 않으며 Pre-run Manifest가 기록하고 Gate가 다시 검증한다.

## 2. 자동 조절 금지

Runner는 Batch 축소, LR 변경, AMP·TF32 활성화, Worker 변경, Architecture·Threshold·Loss 비율·
Dice smooth 변경, OOM 뒤 작은 Batch 재시도를 하지 않는다. CUDA OOM이나 결정론 연산 오류가 나면
실패 기록과 실패 Post-run Manifest를 남기고 중단한다(`retried: false`, `hyperparameters_adjusted: false`).
`epoch_limit`은 합성 테스트 전용이며 Dataset이 `synthetic`이 아니면 거부되고 History에 기록된다.

## 3. 실행 순서와 Manifest Gate

1. `configs/v2/2d_baseline.yaml` 로딩
2. `validate_policy_config`(GT decode·대상 격자·Thick 기준)
3. Architecture·Parameter 수 검증(해석적 계산, 모델을 만들지 않음)
4. Hyperparameter 전체 검증
5. RunPlan 검증(Test·Final Test·SOOP 파일럿 거부, val Split만 Validation)
6. Pre-run Manifest 생성
7. Gate 승인(깨끗한 Working Tree, Config·Split·Package lock·Alias 목록 Hash 일치)
8. 승인 재확인 후에만 Dataset과 Model 생성

1–4에서 실패하면 Run 폴더도 만들지 않는다. Gate 거부 시 Runner는 호출되지 않는다.

## 4. Dataset

`baseline_preprocess.load_and_prepare`(읽기 전용): 1 mm in-plane 대상 격자, Z 간격 유지, DWI·ADC Linear,
GT·Brain Mask Nearest, GT decode `binary_mask_decode_v1_atol_1e_6`, 모델 격자 Brain Mask 안 통계로
z-score. Slice는 8의 배수로 끝쪽 0 Padding하고 `valid_region_mask`를 만든다. Batch 안 크기가 다르면
Batch 최대 크기까지 0으로 더 채우고 그 영역도 `valid_region_mask` 0이다.

항목: `image` [2, H, W], `target` [1, H, W], `brain_mask` [1, H, W], `valid_region_mask` [1, H, W],
`case_key`(`case_NNNN`), `slice_index`, Geometry(모델 격자 Shape·간격, Padding 전후 크기).
Alias·원본 ID·파일 경로는 Batch와 출력에 들어가지 않는다.

### GT 사례 상태 `gt_case_status_v1`

Config `gt_case_status`와 코드 `safety_metrics.GT_CASE_STATUS`가 같아야 한다(다르면 Gate 전 거부):

```yaml
gt_case_status:
  policy_id: gt_case_status_v1
  empty_original_gt_is_valid_negative: true
  positive_gt_lost_after_preprocessing_is_blocked: true
  negative_case_dice: not_applicable
  negative_case_fp_metrics_required: true
```

| 상태 | 조건 | 처리 |
|---|---|---|
| `positive_case` | 원본 GT Voxel > 0, Component > 0, 전처리 후 보존 | 정상 |
| `negative_case` | 원본 GT Voxel 0, decode·파일·Shape·Affine 정상 | 정상 음성 사례(ready). 제외·대체 금지 |
| `gt_lost_during_preprocessing` | 원본 양성인데 모델 격자 GT가 비거나 원본 Component가 Nearest 왕복 후 소실 | blocked(`PreprocessError`) |
| `invalid_gt` | 파일 누락, 허용오차 밖 Label, NaN·Inf, decode 실패, Shape·Affine 불일치 | blocked |

빈 GT라는 이유만으로 실패시키지 않는다. Train 음성 사례는 Dataset에 그대로 들어가며 모든 Brain
Slice가 Lesion-negative Pool에 등록된다(`negative_only_case: true`). Positive 50% Sampling은 전체 Train
Pool 기준이다. Validation 음성 사례는 항상 평가한다: Ungated·Gated Prediction Voxel 수, FP 지표 3종
(`false_positive_voxel_volume_ml`, `no_gt_overlap_component_volume_ml`, `no_gt_overlap_component_count`),
Positive Voxel 비율, Brain Mask 밖 Voxel 수, `prediction_empty`. Dice·Lesion-wise Recall·Missed GT
Component·Small-lesion Recall은 `not_applicable`이며 0이나 1로 바꾸지 않는다. Checkpoint 1차 Dice는
양성 사례만 평균하고, 음성 사례의 FP는 Safety Report와 FP Tie-break 합계에 들어간다. 현재 Validation
음성 사례는 1건뿐이므로 그 결과를 일반적인 특이도나 임상 성능으로 표현하지 않는다.

## 5. Sampling

후보는 모델 격자 Brain Mask가 있는 Slice다. Epoch 길이는 후보 Slice 수와 같고, round(0.5 × 길이)는
Lesion-positive Slice, 나머지는 Lesion-negative Slice에서 뽑는다. 각 부류는 Seed된 순열을 차례로 돌며
(부족하면 반복) 마지막에 합쳐 섞는다. 난수는 `default_rng([seed, epoch])`. Slice나 사례를 제거하지
않으며 Negative Slice도 Epoch에 걸쳐 모두 방문된다. 학습 Positive Slice가 없으면 거부한다.

Epoch마다 `sampler_record`에 전체·Positive·Negative Slice 수, 실제 Sampling 수와 Positive 비율, Seed,
Epoch, Slice 목록 SHA-256을 기록한다. 같은 Seed·Epoch는 같은 Hash, 다른 Seed는 다른 Hash다.

## 6. Loss와 valid_region_mask

Masked Dice와 Masked BCE 모두 표본마다 `valid_region_mask` 안에서만 계산하고 Batch 평균한다. Padding
Logit은 Loss와 Gradient에 영향이 없다(Gradient 0). 합계 = 1.0 × Dice + 1.0 × BCE. Validation Loss는 같은
Loss를 Validation 사례의 모든 Brain Slice에 대해(Sampling 없이) 계산한다.

## 7. 학습 Loop와 Checkpoint 선택

Epoch마다 기록: `train_loss_total`·`_dice`·`_bce`, `validation_loss_total`,
`validation_mean_positive_case_dice`, `validation_total_false_positive_voxel_volume_ml`,
`validation_lesion_wise_recall`, `validation_missed_gt_component_count`, `learning_rate`,
`gradient_norm`(Clipping 전 평균·최대), `sampled_positive_ratio`, 붕괴 기록. Epoch 소요 시간은
비결정적이므로 Hash 대상이 아닌 `training/timing_record.json`에 따로 남긴다(정확 재현 비교 제외).

선택(Validation, Threshold 0.5, Ungated): 양성 사례 평균 Dice 최대 → 1e-4 이내면 Validation Loss 낮음 →
`false_positive_voxel_volume_ml` 낮음 → 이른 Epoch. Validation 사례가 모두 무병변이면 실행을 거부한다.

## 8. 붕괴·중단

Config `collapse_policy`와 코드 `baseline_training.FIXED_COLLAPSE_POLICY`가 같아야 한다(다르면 거부).

| 구분 | 코드 |
|---|---|
| 즉시 중단 | `non_finite_loss`, `non_finite_parameter`, `non_finite_gradient`, `probability_out_of_range`, `all_one_validation_prediction`, `missing_required_validation_output`, `cuda_oom`, `deterministic_algorithm_error` |
| 기록만 | `empty_validation_prediction`, `all_validation_cases_empty_in_single_epoch`, `low_validation_dice` |
| 연속 빈 Epoch | `rely_on_early_stopping`(patience 20), 별도 연속 빈 Epoch 기준 없음(`separate_empty_epoch_limit: null`) |

Epoch마다 기록: 모든 Validation Prediction이 빈 경우, 전체 1인 사례 수, 무병변 FP Volume, 전체 Positive
Voxel 비율. 빈 Prediction은 단일 Epoch에서 전체 Validation이 비어도 즉시 중단하지 않으며, 선택 Epoch가
20 Epoch 동안 바뀌지 않으면 Early stopping으로 끝난다. 전체 1 Prediction은 즉시 중단한다.

## 9. 안전지표와 Ungated·Gated

공식 평가는 원본 격자 `ungated_prediction`(Probability 원본 격자 Linear 복원 후 0.5), 비교 출력은
`gated_prediction`(AND 원본 격자 Brain Mask). 사례별 Metric: Boundary Record(`outside_brain_mask_voxels`·
`_ratio`, `touches_image_edge`, `min_distance_to_brain_boundary_mm`, `boundary_warnings`), Brain Mask
왕복 QA, GT decode Audit, Target-grid Audit, Ungated·Gated 지표. 요약: Lesion-wise Recall, Missed GT,
0.1·0.5·1.0 mL 미만 Recall, 사례 Sensitivity, 무병변 FP Volume, FP 지표 4종, Thin·Thick 소그룹,
Ungated−Gated 차이. 어떤 값도 임상 판정이 아니다.

## 10. 필수 산출물 (Profile `baseline_v1`)

`training/`: `best_checkpoint.pt`, `last_checkpoint.pt`, `training_history.json`,
`validation_metrics.json`, `threshold_record.json`(0.5, `baseline_checkpoint_fixed_0_5`,
`optimized: false`), `sampler_record.json`, `preprocessing_record.json`. 실행 단위 Aggregate
(`aggregate_results/aggregate.json`, 요청의 `aggregate_result.json`에 해당), Validation 사례별
`probability_maps`·`ungated_predictions`·`brain_masks`·`gated_predictions`(원본 격자 NIfTI)·
`patient_metrics`(각각 요청 이름 probability_map·ungated_prediction·brain_mask·gated_prediction·
case_metric에 해당, 파일명은 `case_NNNN`), 그리고 `pre_run_manifest.json`·`post_run_manifest.json`·
`post_run_manifest.sha256`.
이번 단계에서는 Threshold 탐색 결과를 만들지 않는다.

Checkpoint 내용: schema, model_id, architecture, channels, `parameter_count`, model·optimizer·scheduler
state, epoch, best metric, seed, Config·Split canonical Hash, 전처리·Brain Mask·Threshold 정책 ID,
Package lock Hash, Git Commit, 환경 요약, Sampler 재생성 정보. Alias·원본 ID·절대경로는 넣지 않는다.

## 11. 실패 기록

예외가 나면 Runner가 `training/failure_record.json`(실패 단계, 오류 유형, 마스킹된 메시지 — 경로·Alias·
ID가 있으면 `[redacted]`, 성공·실패 Validation 사례 수, 접근 Split, 실패 전 산출물 SHA-256)을 쓰고
예외를 다시 던진다. Lifecycle이 실패 Post-run Manifest와 Seal을 쓰며 Failure Record Hash도 포함한다.
실패한 Run ID는 다시 쓸 수 없다(Gate가 기존 폴더를 거부).

## 12. 해석상 한계

- Validation 사례로 Checkpoint를 고르므로 Validation 지표는 낙관적으로 편향된다.
- 1 mm 모델 격자에서 원본 Voxel보다 작은 예측은 원본 격자 복원 후 0.5를 넘지 못해 사라질 수 있다
  (`docs/v2_2d_baseline_protocol.md`). 작은 병변 Recall로 감시하고 Probability Map은 항상 보존한다.
