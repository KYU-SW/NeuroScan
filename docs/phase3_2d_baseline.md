# Phase 3 — 2D U-Net Baseline (DWI + ADC)

작성 2026-10-17 · 기준 커밋 `9208000` · 공식 run `results/phase3/2d_baseline_20261017-133753_38dfcea89a50`
(results/, models/ 는 Git 미포함) · **Test Split 37명 미사용 (접근 기록 0명)**

## 1. 목적

의료영상 학습 파이프라인을 검증하고, Phase 4 2.5D와 공정하게 비교할 2D baseline을 확보한다.
입력 `[DWI(z), ADC(z)]` (2채널) → 출력 병변 logit (1채널). 예측은 Phase 2 엔진(재구성·원본 공간
복원·평가)으로 환자 단위 평가한다. 최고 성능 경쟁이 목적이 아니다.

| 모듈 | 책임 |
|---|---|
| `data/split_guard.py` | split 접근 단일 창구, Test 봉인(`SealedSplitAccessError`), 접근 기록 |
| `training/normalization.py` | case별 강도 정규화, brain mask(배경 상수 판별) |
| `training/brain_gating.py` | 데이터셋별 brain mask 정책, gating, GT-in-brain guard |
| `training/dataset.py` | in-memory slice dataset (2D/2.5D 공용, context 0/1) |
| `training/sampler.py` | positive/negative slice 비율 sampler |
| `training/augmentation.py` | train 전용 MONAI augmentation |
| `training/model.py` | MONAI UNet / DiceCELoss factory (logit 출력 계약) |
| `training/precision.py` | fp32 공식, bf16 대안, fp16 금지 |
| `training/inference.py` | 모델 → Phase 2 엔진 어댑터 (공식 추론 순서) |
| `training/trainer.py` | 학습 루프, fail-fast guard, checkpoint, 환자 단위 validation |
| `scripts/analyze_intensity.py`, `train_2d.py`, `qa_figures.py` | 분석·학습·QA 실행 |
| `configs/2d_baseline.yaml`, `2d_tiny_overfit.yaml` | 모든 실험값 (tiny는 baseline을 `extends`) |

## 2. DWI normalization

Train 176명만 분석 (`results/phase3/intensity/`, SplitGuard 접근 기록: train 176, test 0).

**채택: `clip_zscore` — case별 brain voxel의 [p0.5, p99.5]로 clip 후 z-score, 뇌 밖 = 0.**

| 후보 | 정규화 후 \|값\| p99 | ADC 단위군 간 차이 (effect size, p5/p50/p95/병변) |
|---|---|---|
| raw | 2349 | 1.73 / 0.77 / 0.71 / 0.41 |
| A. nonzero z-score | 19.9 | 0.63 / 0.58 / 0.32 / 0.51 |
| **B. clip + z-score** | **4.7** | 0.55 / 0.56 / 0.36 / 0.45 |
| C. robust (median/IQR) | 5.6 | 0.48 / – / 0.42 / 0.33 |

A는 극단값(|z| 19.9)이 남고, C는 병변 대비가 작아진다(병변 median 1.6–1.8 vs B 1.9–2.2).
B는 극단값이 없고 단순하며 설명 가능하다.

**Brain mask**: "DWI ≠ 0"은 쓰지 않는다. `sub-strokecase0046`의 DWI 배경이 NIfTI scaling 때문에
−0.000992로 저장되어 있어 "≠ 0" 규칙이면 전체 볼륨이 뇌가 된다. 공식 규칙은 **볼륨 8개 모서리 값의
최빈값을 배경 상수로 보고, 그 값과 다른 voxel을 뇌**로 한다. 뇌 비율은 train 분석에서 14.7–26.6%였고,
5–60%를 벗어나면 실패한다(guard). Train 병변 voxel은 100% brain mask 안에 있었다.

## 3. ADC normalization

**채택: DWI와 동일한 `clip_zscore` (case별, brain mask는 DWI 기준).**

| 후보 | \|값\| p99 | 단위군 간 차이 (p5/p50/p95/병변) | 비고 |
|---|---|---|---|
| raw | 4656 | 18.3 / 29.2 / 30.2 / 16.9 | 단위 차이 그대로 |
| A. 물리 단위 통일 | – | – | **미채택**: metadata에 단위 정보가 없어 크기 기반 추측이 됨 |
| B. nonzero z-score | 6.8 | 0.17 / 0.10 / 0.63 / 0.19 | |
| **C'. clip + z-score** | **4.7** | 0.17 / 0.10 / 0.60 / 0.20 | 채택 |
| C. robust | 9.8 | 0.40 / – / 0.29 / 0.13 | 꼬리 큼 |
| D. log + clip + z-score | 5.7 | 1.59 / 0.87 / 1.51 / 0.56 | 차이 오히려 증가 → 미채택 |

ADC는 brain mask에 사용하지 않는다(뇌 안에서 0인 voxel이 최대 2.8%).

## 4. ADC scale 문제와 해결

ISLES'22 ADC는 세 가지 저장 단위가 섞여 있다 (train: mm²/s 70명, 10⁻³ mm²/s 50명, 10⁻⁶ mm²/s 56명;
중앙값 0.001 / 0.9 / 920). **case별 z-score는 case별 선형 배율을 정확히 상쇄**하므로
(`normalize(a·x) = normalize(x)`, 단위 테스트), 단위를 추측해 변환하지 않고도 단위 차이가 제거된다.
정규화 후 단위군 간 brain median 차이 effect size 29.2 → 0.10, 병변 ADC 중앙값 −0.72 ~ −0.76 로
세 군이 거의 같다. Validation 성능도 단위군 간 큰 차이가 없었다 (§16).

## 5. Slice sampling 정책

Train 176명: 전체 11,045 slice.

| 구분 | slice 수 | 학습 |
|---|---|---|
| Brain-free (brain voxel 0) | 871 (7.9%, 158명) | **제외 (train sampling만)** |
| Lesion-positive (brain-positive) | 3,364 | 포함 |
| Lesion-negative + brain-positive | 6,810 | 포함 (Negative 환자 0150의 뇌 slice 포함) |
| **사용** | **10,174** (positive 33.1%) | |

- `PositiveRatioSampler`: epoch당 10,174 draw(복원추출), 각 draw가 positive일 확률 0.5. 실제 epoch
  평균 positive 비율 0.501.
- brain-free slice 제외 근거: 입력 전체가 상수 → InstanceNorm 분산 0 → gradient norm ~1.5×10⁷
  (다른 slice 10⁻²–10⁰). 추론에서는 gating으로 항상 0이 되는 영역이라 학습할 정보가 없다.
  **lesion-negative slice 제거가 아니다**: 뇌가 있는 음성 slice는 모두 유지한다.
- Validation/추론은 항상 전체 볼륨(모든 slice)을 처리한다.

## 6. Data augmentation (train 전용, tiny overfit에서는 끔)

MONAI `RandAffined` 하나로 image(모든 채널)와 label에 같은 공간 변환: p=0.5, 회전 ±10°, scale ±10%,
이동 ±10 px, image bilinear / label nearest. 강도: channel-wise scale ±10%·shift ±0.1 (p=0.5),
Gaussian noise σ=0.05 (p=0.2). **좌우 반전 끔** (반구 정보는 향후 위치 분석에 의미). image·label
공간 일치와 DWI/ADC 동일 변환은 테스트로 검증.

## 7. Model architecture

MONAI `UNet(spatial_dims=2, in_channels=2, out_channels=1, channels=(32,64,128,256,512),
strides=(2,2,2,2), num_res_units=0, norm=instance, dropout=0)` — **2,638,153 parameters**.
`num_res_units=0` 필수: >0이면 출력 1채널에 InstanceNorm+PReLU가 붙어 logit이 영상마다 정규화된다
(§11). 출력은 raw logit이며, 출력 conv 뒤에 정규화/활성화가 없음을 회귀 테스트로 보장.

## 8. Loss

MONAI `DiceCELoss(sigmoid=True, batch=True, λ_dice=1, λ_ce=1)`. 1채널에서 Dice 항은 내부에서 sigmoid,
CE 항은 `BCEWithLogitsLoss`(raw logit) — 모델은 sigmoid를 적용하지 않는다(테스트로 확인:
loss = Dice(sigmoid(logit)) + BCEWithLogits(logit)).

공식 추론 순서: **logits → sigmoid(1회) → brain-mask gating → threshold 0.5 → binary slice →
3D 재구성 → 원본 공간 복원 → 평가.**

## 9. Optimizer

AdamW (lr 1e-3, weight decay 1e-4), cosine annealing (T_max = max_epochs, η_min 1e-6),
gradient clipping 없음 (원인 제거 후 gradient norm 최대 6.07로 불필요).

## 10. Training configuration

`configs/2d_baseline.yaml`: seed 42, batch 32, train 크기 256×256 (center pad/crop; 추론은 전체 slice
16배수 pad), max 100 epoch, early stopping patience 20, **precision fp32**, num_workers 0, deterministic.
Best checkpoint: val positive-patient mean Dice; 1e-4 이내 동률이면 낮은 val loss, 그다음 이른 epoch.
Checkpoint는 `best.pt`, `latest.pt` 두 개만 (`models/phase3/<run_id>/`), metadata: run_id, epoch,
val Dice, config hash, preprocessing v1, split_v1_seed42, normalization, in_channels, precision.

Fail-fast guard: loss/gradient non-finite, gradient norm > 1e7, val 예측 all-zero / all-one이 15 epoch
연속, Test 환자 접근. guard 발동 시 해당 batch의 case/slice를 `aborted_batch.json`에 기록.

재현성: Python/NumPy/PyTorch/MONAI seed 42, cudnn deterministic, `use_deterministic_algorithms(warn_only)`,
sampler seed = 42 + epoch, augmentation seed 42, workers 0. 일부 CUDA backward kernel은 결정적
구현이 없어 bitwise 동일성은 보장되지 않는다.

## 11. Tiny Overfit (train 0002 0.94 mL / 0028 7.3 mL / 0077 40.4 mL, augmentation 끔, 40 epoch)

기준(실행 전 고정, 변경 없음): 평균 Dice ≥ 0.80, loss 비 ≤ 0.5, 최대 확률 ≥ 0.5, 예측/GT 부피 ≤ 3,
checkpoint reload 일치.

| # | 설정 | Dice | loss 비 | 부피 비 | 결과 | 원인 / 조치 |
|---|---|---|---|---|---|---|
| 1 | res2, fp16 | 0.565 | 0.79 | 7.6 | 실패 | 출력층 InstanceNorm(1ch) → `num_res_units=0` |
| 2 | res0, fp16 | 0.254 | 0.56 | 30.2 | 실패 | fp16 gradient overflow, GradScaler 붕괴 → **fp32** |
| 3 | res0, fp32 | 0.391 | 0.46 | 39.7 | 실패 | brain-free slice 고정 FP 패턴 (FP의 92–99.6%) → **brain-mask gating** |
| 4 | + gating | 0.969 | 0.46 | 1.06 | 통과 | (전체 학습에서 brain-free gradient spike 발견) |
| 5 | + brain-free 학습 제외 | **0.999** | **0.006** | **1.00** | **통과** | gradient norm 최대 1.52 |

1–4번 run과 진단(`results/phase3/diagnostics/`)은 보존. 1번 run에 `FAILED_SANITY_RUN.md`.

### Troubleshooting 기록
- **FP16 AMP 금지**: fp32 gradient norm이 최대 1.5×10⁵로 fp16 한계(65,504) 초과 → GradScaler가 step을
  건너뛰며 scale이 ~0으로 붕괴 → gradient underflow → Adam 관성·weight decay만으로 loss가 120 epoch간
  단조 상승. fp32와 bf16은 동일 조건에서 정상 수렴(120 epoch loss 0.52). **bf16은 정상 수렴이 확인된
  대안**이나 공식 설정이 아니다. `precision: fp16`은 코드에서 오류.
- **Brain-mask gating**: 아래 §13·§18 참고.
- **전체 학습 1차 중단**: epoch 2에서 gradient guard(1.03×10⁷ > 1e7) 발동. 원인은 brain-free slice
  (batch gradient norm 1.48×10⁷). guard 상한을 올리거나 clipping을 넣지 않고 원인을 제거했다
  (`2d_baseline_20261017-132907_*` 보존).

## 12. Validation 결과 (Val 37명 = positive 36 + negative 1, 원본 공간)

Best epoch 42 / 62 (early stopping), threshold 0.5 고정.

| Metric (positive 36명) | mean ± std | median [IQR] | min–max |
|---|---|---|---|
| **Dice (primary)** | **0.731 ± 0.214** | **0.807** [0.685–0.866] | 0.122–0.945 |
| IoU | 0.612 ± 0.223 | 0.677 [0.521–0.764] | 0.065–0.896 |
| Sensitivity | 0.697 ± 0.233 | 0.780 [0.626–0.845] | 0.065–0.973 |
| Precision | 0.843 ± 0.159 | 0.886 [0.810–0.937] | 0.125–0.983 |
| Volume error (mL, signed) | −3.09 ± 19.0 | −0.32 [−1.65–0.11] | −103.7–31.3 |
| Absolute volume error (mL) | 6.65 ± 18.0 | 0.94 [0.25–3.59] | 0.04–103.7 |
| Relative volume error (%) | −9.6 ± 54.8 | −11.7 [−29.2–3.1] | −93.3–263.9 |

Precision > Sensitivity, 부호 있는 부피 오차 음수 → **전반적으로 과소분할** 경향.
주의: best checkpoint를 이 validation으로 골랐으므로 이 수치는 낙관적이며 독립 성능이 아니다.
독립 평가는 Phase 5 Test(봉인)에서만 한다.

## 13. Negative patient (sub-strokecase0151)

`false_positive_case`: 예측 14 voxel, **FP 0.112 mL, 4 component** (모두 brain 내부; gating은
brain 밖만 제거). 1명이므로 specificity 등 비율 지표로 일반화하지 않는다.

## 14. Lesion size subgroup (positive Dice)

| 군 | n | mean | median |
|---|---|---|---|
| < 1 mL | 5 | 0.765 | 0.732 |
| 1–10 mL | 15 | 0.687 | 0.796 |
| ≥ 10 mL | 16 | 0.762 | 0.833 |

## 15. Z-spacing subgroup

| 군 | n | mean | median |
|---|---|---|---|
| Thin (2 mm) | 28 | 0.718 | 0.798 |
| Thick (4.4–5 mm) | 8 | 0.777 | 0.838 |

## 16. ADC unit subgroup

| 저장 단위 | n | mean | median |
|---|---|---|---|
| mm²/s | 18 | 0.729 | 0.808 |
| 10⁻³ mm²/s | 8 | 0.751 | 0.758 |
| 10⁻⁶ mm²/s | 10 | 0.718 | 0.825 |

단위군 간 큰 차이 없음 → 단위 shortcut 징후 없음. 단, n이 작아 subgroup 결과는 탐색적이며 임상적
결론이 아니다.

## 17. GPU / 학습 시간

Epoch 평균 60.7 s, 총 3,767 s (62 epoch, 약 63분), peak VRAM 1.17 GB (fp32, batch 32, 256²),
gradient norm 최대 6.07 (epoch 평균의 평균 0.30). RTX 5060 Ti 16 GB.

## 18. Limitations

- Validation으로 checkpoint를 선택 → validation 수치는 낙관적. Test는 Phase 5까지 봉인.
- **Brain-mask gating** (공식 추론 정책): 해부학적 ROI 제한이며 lesion-size 기반 후처리가 아니다.
  ISLES'22가 skull-stripped라는 특성에 기반한다. 외부 데이터가 skull-stripped가 아니면 이 정책을
  그대로 쓰지 않고 `data.brain_mask_policy`에 데이터셋별 정책을 등록·검증해야 한다.
- Brain-free slice는 학습에서 제외 — 모델은 뇌 없는 입력에서 정의되지 않은 출력을 낼 수 있으며,
  gating이 이를 0으로 만든다(공식 파이프라인 밖에서 모델을 단독 사용하지 말 것).
- 실패 패턴: 큰 병변의 약한 DWI 신호(0083: Dice 0.12, 111 mL 중 7.5 mL만 검출 — 아급성기
  pseudo-normalization 의심), 작은 다발성 병변의 FP/FN, 예측 component가 GT보다 많음(389 vs 209).
- Phase 2 limitation 유지: XY 1.146 mm 그룹의 1–2 voxel component 손실, 원본 공간 Dice 상한.
- 작은 component 자동 제거 없음 (원칙 유지).
- 일부 CUDA 연산 비결정성으로 bitwise 재현성 미보장.

## 19. Phase 4 (2.5D) 비교 시 고정해야 할 조건

Phase 4 config는 `2d_baseline.yaml`을 `extends`하고 **`data.context_slices: 0 → 1`(in_channels 2 → 6)
만 변경**한다. 고정: processed/v1, split_v1_seed42, normalization(clip_zscore, case별, brain mask),
brain_mask_policy, exclude_brain_free_slices(중심 slice 기준), sampler(0.5, epoch당 사용 slice 수),
augmentation, UNet(channels/strides/num_res_units=0/instance), DiceCELoss, AdamW/cosine, lr·wd,
batch 32, max 100 / patience 20, **fp32**, gradient guard, brain-mask gating, threshold 0.5, 체크포인트
선택 규칙, seed 42, 평가 엔진(원본 공간, positive/negative 분리). 2.5D 채널 순서는
`[DWI z-1, z, z+1, ADC z-1, z, z+1]`, 볼륨 경계는 edge slice 반복.
