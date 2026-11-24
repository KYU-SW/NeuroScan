# v2 2.5D Baseline 프로토콜

> 상태: **구현·합성 검증 단계.** 실제 MRI 접근, 실제 Cache 생성, CUDA Smoke, ISLES Smoke, 학습, 추론,
> Threshold 탐색은 실행하지 않았고 각각 별도 승인 대상이다. 합성 결과는 성능 결과가 아니다.

- 코드: `ai/neuroscan_ai/v2/` — `unet_25d`(모델), `baseline_25d_dataset`(3-Slice Stack·Sampling),
  `baseline_25d_inference`(Validation), `baseline_25d_runner`(Gate·학습·Resume), `cuda_smoke_25d`(합성 메모리 Smoke).
- CLI: `scripts/run_v2_25d_baseline.py`(validate·run·smoke·resume), `scripts/prepare_v2_25d_cache.py`,
  `scripts/smoke_v2_25d_cuda_memory.py`. Config: `configs/v2/25d_baseline.yaml`.
- 테스트: `tests/test_v2_25d_*.py`(합성 배열·CPU만).
- 동결된 2D 파일과 공용 모듈은 수정하지 않는다. 2.5D 생산 코드는 `unet_2d`, `baseline_dataset`,
  `baseline_inference`, `baseline_runner`, `run_v2_2d_baseline`을 직접 import하지 않는다(AST 검사).
  공용 모듈은 읽기 전용으로 쓴다. 공용 모듈을 통한 간접 import는 허용한다.

## 1. 목표

1. Thick Slice <0.1 mL Recall 개선 2. 세 Seed 공통 Missed 28개 감소 3. 전체 Small-lesion Recall 개선
4. 병변 주변 Oversegmentation 감소. 2D와 다른 것은 입력 문맥뿐이다. 2D 결과를 보고 사후 튜닝하지 않는다.

## 2. 입력 `input_25d_slice_major_dwi_adc_pairs_v1`

| Channel | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 값 | DWI z-1 | ADC z-1 | DWI z | ADC z | DWI z+1 | ADC z+1 |

- Channel 2·3은 같은 z의 2D 입력과 bit 단위로 같다. 학습과 추론은 같은 함수(`stack_slices`)를 쓴다.
- Target, Sampling 분류, Brain Mask, valid_region_mask는 중앙 Slice z의 것이다.

## 3. 이웃 `neighbor_fixed_adjacent_index_pm1_v1`, 경계 `edge_slice_repeat_v1`

- Source Slice는 `[max(z-1,0), z, min(z+1,Z-1)]`이고 DWI·ADC에 같은 목록을 쓴다. Z=1은 `[0,0,0]`, Z=2는
  `[0,0,1]`, `[0,1,1]`이다. Zero Slice, Wrap-around, 경계 제외, Z 보간, 거리 기반 이웃 선택은 쓰지 않는다.
- 한계: 인덱스 기준이므로 물리적 문맥이 사례마다 다르다. z-1~z+1 범위는 Z 간격 2.0 mm에서 4.0 mm, 4.4 mm에서
  8.8 mm, 4.8 mm에서 9.6 mm, 5.0 mm에서 10.0 mm다. 모든 사례에서 같은 물리적 문맥이라고 표현하지 않는다.
  Thick 사례의 작은 병변은 이웃 Slice의 도움을 받지 못할 수 있다.

## 4. 정규화 `normalization_case_volume_zscore_brainmask_no_slice_renorm_v1`

Cache에 저장된 사례 Volume 단위 z-score(Brain Mask 내부 통계)를 그대로 읽는다. 중앙 Slice별, 이웃 Slice별, Stack
생성 후 정규화는 하지 않는다. 각 Channel은 Cache Volume의 대응 Slice와 bit 단위로 같다.

## 5. 모델 `neuroscan_v2_25d_baseline_v1`

4-level 2D U-Net, Channels 36·72·144·288, 입력 6, 출력 1(중앙 Slice Logit), InstanceNorm(affine), ReLU,
ConvTranspose, H·W 8의 배수, Pretrained weight 없음. Parameter 수가 **2,440,873**이 아니면 생성을 거부한다.

| 후보 | Parameter | 결정 |
|---|---|---|
| A: 2D 구조, 입력 6 | 2,440,873 | 선택 |
| B: 1×1 Stem(6→2) + 2D 구조 | 2,439,591 | 제외(입력 압축 병목) |
| C: [32,64,128,256] | 1,929,281 | 제외(목표 2.0M~3.5M 밖) |
| C2: [28,56,112,224] | 1,477,785 | 제외(목표 범위 밖) |

2D와 Shape가 다른 Parameter는 첫 Conv weight뿐이다. 같은 Seed라도 난수 소비량이 달라 이후 층의 초기값이
다르므로, 2D·2.5D는 층별 초기 Weight를 짝지은 실험이 아니다.

## 6. Sampling

- 양성은 중앙 Slice GT로만 정한다(`center_positive`). 이웃에만 병변이 있는 Slice(`neighbor_only_positive`)는 음성이다.
- 중앙 Slice 순서는 `(seed, epoch)`만으로 정해지고 2D 규칙과 같다(`slice_list_sha256` 동일). `source_list_sha256`은
  Source Slice 목록의 Hash다.
- Sample 기록: `case_key`, `center_slice_index`, `source_slice_indices`, `edge_repeat_applied`, `z_spacing_mm`,
  `thick_slice_subgroup`, `input_channel_policy_id`, `center_positive`, `neighbor_only_positive`,
  `positive_neighbor_count`. 후보 Slice 전체 표는 `sampler_record.json`의 `slice_table`에 있다. 원본 ID·Alias는 없다.

## 7. 정책 기록과 Manifest 연결

공용 Manifest Schema는 바꾸지 않는다. Manifest에는 `model.architecture_id`(model_id)와
`reproducibility.runner_kind`(`v2_25d_baseline`)가 있다. 전체 정책 10개(`POLICY_RECORD`)는 Config, preprocessing record,
Best·Last·Resume Checkpoint, Training history, Validation metrics, Aggregate에 기록하고, Manifest는 Config Hash와
산출물 Hash로 연결한다. 사본 사이에 차이가 있으면 Config 검사, Run 또는 Resume가 거부·실패한다.

- `input_channels`: 6, 추론: `v2_25d_slicewise_stack_sigmoid_original_grid_v1`(모든 z를 Stack으로 예측하고 원본 격자로
  복원), 조합: `single_model_25d`. 나머지 ID는 §2~§5, §8에 있다.

## 8. 학습·Checkpoint·Threshold

- Hyperparameter, Loss, 결정론, Optimizer, Scheduler, Early stopping, Seed(42·43·44 CLI 명시), Device(cuda)는 2D와 같다.
- Checkpoint 선택은 Threshold 0.50, Ungated, 양성 Validation 평균 Dice이고 동률 규칙도 2D와 같다.
- `threshold_v2_25d_initial_fixed_0_50_not_frozen_v1`: 초기 고정값이며 동결이 아니다. 2D와의 공식 비교도 0.50에서 한다.
  2.5D Threshold 탐색은 3-Seed 결과 후 별도 승인 전까지 하지 않는다(코드 없음).

## 9. Batch와 CUDA Smoke

- `batch_policy.batch_size: 32`, `batch_status: approved`. candidate이면 정식 Run은 거부된다.
- approved이면 승인 필드 6개의 존재·형식·상호 일치와 reserved 메모리 기준(0 초과, 14 GiB 미만)을 Runner가 Config 검사에서
  확인한다. RunPlan, Cache, Dataset, Model, Run 폴더보다 먼저이며 위반 시 `BatchApprovalError`로 거부한다. Smoke 파일의
  존재는 요구하지 않고(Commit·Hash는 감사 추적용), Cache·Manifest는 Config Hash로 이 승인 상태에 연결된다.
- 승인 근거: 합성 CUDA 메모리 Smoke(`synthetic_cuda_memory_smoke_v1`, Commit `13e477ddf3bd5d09c3a0f171b7e38a12a8ee44c6`,
  결과 파일 SHA-256 `4847b02986feed74290df16cfa11356b26569fb5e03a04309fc2aefe2a959fb3`).
  - 입력 `[32,6,264,264]`, Seed 42, 2회 × 3 Step. Loss·Output·Parameter exact match, OOM·NaN·Inf·재시도 0.
  - max allocated 7.60 GiB(8,159,441,408 bytes), max reserved 9.56 GiB(10,261,364,736 bytes), 14 GiB 기준까지
    4.44 GiB 여유, 최소 가용 RAM 17.93 GiB(각 Step 직후 측정).
  - Process Peak RSS는 기록되지 않았다(사전 통과 기준 아님). ISLES Smoke의 Runner 자원 기록으로 확인한다.
  - Batch 16은 실행하지 않았다.
  - 실제 Cache mmap·Validation·장시간 메모리는 ISLES 2-Epoch Smoke에서 추가 확인한다. 이 Smoke는 실제 ISLES 학습
    안전성을 최종 보장하는 검사가 아니다.
- Smoke: 합성 `[32,6,264,264]`, FP32, 같은 Seed·입력으로 2회, 각 3 Step 연속. 통과 조건은 OOM·NaN·Inf 없음,
  두 결과 일치, `max_memory_reserved` < 14 GiB, 가용 RAM ≥ 4 GiB다. 값을 읽지 못하면 실패다.
- 기록 위치는 `results/v2_smoke/25d_cuda_memory_v1/`(git 제외, 덮어쓰기 거부). 실패하면 기록 후 끝나며 자동 축소,
  Batch 16 실행, 재시도는 없다. 결과가 Config를 바꾸지 않는다.
- `batch_status`를 바꾸면 Config Hash가 바뀐다. 따라서 Cache는 상태 승인과 Commit **뒤에** 만든다.

## 10. Cache

경로 후보 `results/v2_cache/25d_baseline_v1/`. Volume 배열 7종만 저장하고 3-Slice Stack은 저장하지 않는다.
Commit·Config Hash 결합은 기존 Cache Gate를 그대로 따른다. 2.5D 정책은 Config Hash로 결합된다.

## 11. Run Profile

| Profile | Run ID | Epoch 제한 | 조건 |
|---|---|---|---|
| `formal_isles` | Smoke ID 거부 | 거부 | `batch_status: approved`, Seed 42·43·44 |
| `formal_isles_smoke` | `v2-25d-smoke-seed42-a`, `-b`만 | 정확히 2 | Seed 42, Train 176 / Val 37 전체 |
| `synthetic` | 합성 테스트 | 허용 | 정식 데이터 표식이 없어야 함 |

Smoke는 A/B exact match 확인용이다. 성능으로 해석하지 않고 3-Seed 집계에서 제외한다. 산출물 Profile은 `baseline_v1`이다.

## 12. 성공 기준 (사전 명세, 판정 코드 없음)

- P1: Thick <0.1 mL 탐지 합계 ≥ 24/51(2D 21/51: Seed 42·43·44에서 8·5·8), 2개 이상 Seed에서 대응 2D Seed보다
  낮지 않음. 반올림 비율이 아니라 원시 개수로 판정한다.
- P2: 세 Seed 공통 Missed < 28. P3: 전체 <0.1 mL Recall 평균 ≥ 0.644. P4: Oversegmentation 평균 ≤ 119.74 mL
  (<112.48 mL이면 개선으로 표시).
- 안전장치: Dice 평균 ≥ 0.7607, Lesion Recall 평균 ≥ 0.7437, FP 평균 ≤ 128.70 mL, Dice 표본 SD ≤ 0.0179, 음성 사례
  FP 각 Seed ≤ 0.5 mL, Brain Mask 밖 예측 각 Seed ≤ 20 voxel, GT Component 소실 0, 금지 Split 접근 0.
- SUCCESS: 전부 충족. PARTIAL: Primary 일부 미충족, 안전장치 전부 충족. FAIL: 안전장치 또는 실행 무결성 위반.
- 2D와 같은 Validation을 쓰므로 낙관적 편향이 있고, 임상 성능 주장에 쓸 수 없다.

## 13. 순서와 접근 경계

1. 합성 CPU 테스트 2. 합성 CUDA 메모리 Smoke 3. `batch_status` 승인·Commit 4. Cache 생성 5. ISLES Smoke A/B
6. 공식 Seed 42 7. Seed 43 8. Seed 44 9. 3-Seed 집계 10. 2D 비교 11. Threshold 별도 결정. 각 단계는 별도 승인이다.
ISLES Test, Final Test, SOOP는 접근하지 않는다. 2D 결과와 2D Threshold는 바꾸지 않는다.
