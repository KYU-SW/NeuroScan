# v2 2D Smoke Test 프로토콜

> 성능 실험이 아니다. 실제 PyTorch 학습 Runner가 Manifest Gate를 통과해 필수 산출물을 남기고,
> 같은 Seed의 두 실행이 재현되는지 확인하는 기술 점검이다. 결과 수치는 참고값이며 해석하지 않는다.

- 코드: `ai/neuroscan_ai/v2/` (`dataset`·`model_2d`·`training`·`inference`·`outputs`·`runner`),
  Config: `configs/v2/2d_smoke.yaml`, CLI: `scripts/run_v2_2d_smoke.py`,
  테스트: `tests/test_v2_real_runner.py`(합성 볼륨·CPU만).
- 실행 통제: `docs/v2_run_control_protocol.md`, Manifest: `docs/reproducibility_protocol.md`.

## 1. v1과의 분리

v2 코드는 v1 학습·추론·전처리·분석·Final Test 모듈을 import하거나 수정하지 않는다(테스트로 확인).
v1 Checkpoint를 쓰지 않으며 Pretrained Weight도 쓰지 않는다. v1 Split 파일은 사례 후보 선정에만
읽기 전용으로 쓴다.

## 2. 모델

`neuroscan_v2_2d_smoke_v1`: DWI·ADC 2채널 입력, 1 logit 출력의 작은 2단 U-Net(Conv3×3 블록,
MaxPool, Transposed-conv 업샘플, 1×1 Head), base channel 8, Parameter 6,497개. Sigmoid로 Probability를
만든다. 가중치는 명시 Seed(42)로 무작위 초기화하고 Parameter 수와
초기화 Seed를 Training History와 Checkpoint 메타데이터에 기록한다. CPU·CUDA 모두에서 실행된다.

## 3. 사례 선정

v1 Split의 train에서 4건, val에서 2건을 `SHA-256("neuroscan-v2-smoke-v1:" + case_id)` 오름차순으로
고른다. Test Split은 후보에도 들어가지 않으며 SOOP 사례는 후보에서 제외한다. 공개 출력은 Alias
(`v2-smoke-train-01..04`, `v2-smoke-val-01..02`)만 쓰고, Manifest와 출력 파일은 다시 위치 Case Key
(`case_0001` ...)만 쓴다. 원본 ID 대응표는 Git 제외 경로 `results/v2_runs/_internal/`에만 둔다.

## 4. 전처리와 Brain Mask

- 정책 `preprocess_v2_isles_smoke_v1`: DWI·ADC·GT를 저장된 원본 격자 그대로 읽고, 세 영상의
  Shape·Affine(1e-3 mm) 일치를 확인한다. Resampling·재방향 없음. Slice는 마지막 배열 축 방향.
- ADC 저장값은 변환하지 않는다. 모델 입력만 Brain Mask 안 통계로 사례별 z-score 한다.
- Brain Mask `skull_stripped_background_v2_smoke`: DWI가 일정 배경값(8개 모서리 최빈값)과 다른 Voxel,
  전체의 5–60% 범위 검사. 정규화 통계와 Gated 출력에만 쓴다. **Smoke Test용 기술 정책이며 임상 v2
  최종 Brain Mask 정책이 아니다.**
- Ungated 출력을 반드시 따로 저장하며, Prediction을 조용히 지우지 않는다.

## 5. 학습·결정론 고정값

Epoch 1, Seed 42, Batch 8, Worker 0, FP32, AMP Off, TF32 matmul·convolution Off, deterministic
algorithms On, cuDNN deterministic On·benchmark Off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Optimizer Adam,
Learning Rate 1e-3, Loss BCE-with-logits + soft Dice. 이 결과로 Hyperparameter를 평가하지 않는다.

Runner는 Gate 직전에 결정론 설정을 적용하고, 그 실제 상태를 Pre-run·Post-run Manifest가 함께 기록한다.

## 6. 실행 흐름과 산출물

RunPlan(train 4건 + validation 2건) → Pre-run Manifest → Gate 승인 → 학습 → Validation 추론 →
필수 출력 검증 → Post-run Manifest·Seal → `verify_run`. Gate 승인 전에는 사례를 읽거나 모델을 만들지
않는다.

- 학습 산출물: `training/checkpoint.pt`, `training/training_history.json`,
  `training/validation_metrics.json`, `training/threshold_record.json`
  (Threshold 0.5, source `smoke_test_fixed_not_validated`, 최적화 없음).
- Validation 사례별: Probability Map, Ungated Prediction, Brain Mask, Gated Prediction, 사례 Metric,
  그리고 실행 단위 Aggregate. 모두 원본 격자, `ungated = probability ≥ 0.5`,
  `gated = ungated AND brain_mask`.

## 7. 재현성 판정

같은 코드·Config로 `v2-2d-smoke-seed42-a`와 `-b`를 실행하고 `compare`로 Pre-run Manifest와 Checkpoint·
Probability·Ungated·Brain Mask·Gated·History·Aggregate Hash를 비교한다. Checkpoint나 Probability가
다르면 실패로 기록하고 원인을 조사하며, 결과를 맞추려고 재실행하거나 Seed를 바꾸지 않는다.

## 8. 실행 전제

Gate는 Working Tree가 clean이어야 승인한다. 따라서 Smoke Test 실행 코드는 커밋된 상태여야 한다.
CLI `run`은 Tree가 clean이 아니면 Gate를 호출하기 전에 거부해 run ID를 소모하지 않는다.
