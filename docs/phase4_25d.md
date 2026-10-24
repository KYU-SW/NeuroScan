# Phase 4 — 2.5D U-Net (DWI + ADC, z±1) 및 2D 비교

작성 2026-10-24 · 기준 커밋 `9208000` (Phase 3 미커밋 작업 트리 위) · 공식 run
`results/phase4/25d_baseline_20261024-192617_34d18df104da` · 비교 기준 2D run
`results/phase3/2d_baseline_20261017-133753_38dfcea89a50` (results/, models/ 는 Git 미포함) ·
**Test Split 37명 미사용 (두 run 모두 접근 기록 0명)**

## 1. 목적

인접 slice(z−1, z+1)를 입력에 추가하는 것만으로 2D baseline 대비 환자 단위 분할 성능이 달라지는지
같은 조건에서 비교한다. 최고 성능 경쟁이 아니라 **단일 변수 비교**가 목적이다.

## 2. 설정 — 변경은 `context_slices` 하나

`configs/25d_baseline.yaml`은 `2d_baseline.yaml`을 `extends`한다. 2D와 다른 키는 아래 4개뿐이며,
`tests/test_phase4.py`가 그 외 키가 하나라도 다르면 실패하고 `scripts/compare_runs.py`도 비교를 거부한다.

| 키 | 2D | 2.5D | 성격 |
|---|---|---|---|
| `data.context_slices` | 0 | **1** | **유일한 실험 변수** |
| `run_name` | 2d_baseline | 25d_baseline | 식별자 |
| `checkpoint.dir` | models/phase3 | models/phase4 | 출력 경로 (Phase 3 덮어쓰기 방지) |
| `output.results_dir` | results/phase3 | results/phase4 | 출력 경로 |

입력 `[DWI z−1, z, z+1, ADC z−1, z, z+1]` (6채널, modality-major), 볼륨 경계는 edge slice 반복.
고정(상속): processed/v1, split_v1_seed42, clip_zscore 정규화(case별, brain mask), brain-mask 정책,
brain-free slice 학습 제외(**중심 slice 기준**), positive-ratio sampler 0.5, augmentation, MONAI UNet
(32–512, `num_res_units=0`, instance), DiceCELoss, AdamW 1e-3 / cosine, batch 32, 256², max 100 /
patience 20, **fp32**, fail-fast guard, brain-mask gating, threshold 0.5, checkpoint 선택 규칙, seed 42,
Phase 2 평가 엔진(원본 공간, positive/negative 분리).

파라미터 2,638,153 → **2,639,305** (+1,152, 첫 conv 입력 2→6). 용량 차이는 무시 가능하다.

코드 변경: 학습·추론·데이터 코드는 Phase 3에서 이미 `context_slices`를 지원하여 **수정 없음**.
신규는 config 2개, 테스트, 비교 모듈/스크립트뿐이다 (§9).

## 3. 테스트 (`tests/test_phase4.py`, 15개)

공정성(config diff = 허용 4키, 우발적 변경 검출, phase3 경로 미사용) · 6채널 순서와 모든 slice의
edge replication · 6채널 dataset shape와 중심 slice label 정렬 · brain-free 제외가 중심 slice 기준 ·
`predict_case(context=1)`가 경계/내부 slice에서 직접 model 호출과 일치 · 추론 batch 크기 불변 ·
gating이 중심 slice brain 밖을 0으로 · augmentation이 6채널+label에 동일 공간 변환 · seed 고정 시
augmentation 결정성 · 쌍대 비교 통계(case 집합 불일치 오류, 개선/악화/동률, seeded bootstrap).
전체 suite **139 passed**, ruff check/format 통과.

## 4. Tiny Overfit (2D와 동일 3명·기준·40 epoch, augmentation 끔)

`results/phase4/25d_tiny_overfit_20261024-192418_f06c80a9fee9` — **1회 시도 통과**.

| 기준 | 값 | 결과 |
|---|---|---|
| 평균 Dice ≥ 0.80 | **0.9986** | 통과 |
| loss 비 ≤ 0.5 | 0.0077 | 통과 |
| 최대 확률 ≥ 0.5 | 1.0 | 통과 |
| 예측/GT 부피 ≤ 3 | 1.00 | 통과 |
| checkpoint reload 일치 | True | 통과 |

## 5. 학습

| 항목 | 2D (Phase 3) | 2.5D (Phase 4) |
|---|---|---|
| Epoch (실행 / best) | 62 / 42 | 69 / **49** |
| Best val positive mean Dice (학습 중) | 0.7309 | **0.7428** |
| Epoch 시간 평균 (min–max) | 60.7 s | **111.2 s** (100–134) |
| 총 학습 시간 | 3,767 s (63분) | **7,674 s (128분)** |
| Peak VRAM (torch) | 1.17 GB | **1.20 GB** |
| Gradient norm 최대 | 6.07 | 6.21 |
| Guard 발동 | 없음 | 없음 |
| Checkpoint reload 일치 | True | True |

Best checkpoint: `models/phase4/25d_baseline_20261024-192617_34d18df104da/best.pt`.
Epoch 시간이 2D의 약 1.8배인 이유는 CPU 측 6채널 slice stacking과 augmentation(`num_workers=0`,
GPU 사용률 ~15–20%)이다. `num_workers`는 재현성 조건이므로 바꾸지 않았다. 후반 epoch(>60)의
시간 증가(~127 s)는 원인을 확인하지 못했다 (같은 PC의 다른 프로세스와의 CPU 경합 가능성).

## 6. Validation 결과 (Val 37명 = positive 36 + negative 1, 원본 공간, threshold 0.5)

| Metric (positive 36명) | 2D mean | 2.5D mean ± std | 2.5D median [IQR] |
|---|---|---|---|
| **Dice** | 0.731 | **0.743 ± 0.195** | 0.817 [0.704–0.865] |
| IoU | 0.612 | 0.622 ± 0.210 | 0.691 [0.544–0.762] |
| Sensitivity | 0.697 | 0.710 ± 0.233 | 0.767 [0.616–0.883] |
| Precision | 0.843 | 0.851 ± 0.134 | 0.852 [0.810–0.936] |
| Volume error (mL, signed) | −3.09 | −2.62 ± 18.7 | −0.30 |
| Absolute volume error (mL) | 6.65 | 6.49 ± 17.7 | 0.89 |

과소분할 경향(Precision > Sensitivity, 부호 있는 부피 오차 음수)은 2.5D에서도 유지된다.
Best checkpoint를 이 validation으로 골랐으므로 두 모델 모두 수치가 낙관적이다.

## 7. 2D vs 2.5D 쌍대 비교 (positive 36명, Δ = 2.5D − 2D)

`results/phase4/comparison/25d_baseline_..._vs_2d_baseline_.../` (`summary.json`,
`paired_metrics_positive.csv`, `paired_metrics_negative.csv`, `dice_comparison.png`).
Bootstrap: 환자 재표본 10,000회, seed 42, percentile 95% CI. Wilcoxon signed-rank 양측.
개선/악화 판정 허용오차 |Δ| ≤ 0.001 = 동률.

| Metric | 2D | 2.5D | mean Δ | 95% CI | median Δ | Wilcoxon p | 개선 / 악화 / 동률 |
|---|---|---|---|---|---|---|---|
| **Dice** | 0.7309 | 0.7428 | **+0.0119** | [−0.0017, +0.0284] | +0.0056 | 0.286 | 20 / 16 / 0 |
| IoU | 0.6120 | 0.6222 | +0.0102 | [−0.0030, +0.0243] | +0.0084 | 0.338 | 20 / 16 / 0 |
| Sensitivity | 0.6967 | 0.7104 | +0.0137 | [−0.0060, +0.0337] | +0.0177 | 0.105 | 23 / 12 / 1 |
| Precision | 0.8431 | 0.8510 | +0.0079 | [−0.0179, +0.0350] | −0.0076 | 0.81 | 16 / 20 / 0 |
| Abs. volume error (mL) | 6.65 | 6.49 | −0.16 | [−1.04, +0.93] | | 0.78 | |
| 예측 component 수 (평균) | 10.8 | 10.0 | −0.78 | [−1.89, +0.31] | | 0.25 | |

- **Dice 95% CI가 0을 포함하고 Wilcoxon p = 0.29 → 통계적으로 유의한 차이 없음.**
- 평균 개선의 상당 부분이 두 환자에서 나온다: 0181 (16.2 mL, 0.314 → 0.504, +0.19), 0010
  (3.4 mL, 0.197 → 0.338, +0.14). 두 명을 빼면 mean ΔDice는 +0.003이다.
- 가장 크게 나빠진 환자: 0074 (−0.069), 0044 (−0.046), 0173 (−0.039). 모두 thin.
- 2D 최악 사례 0083 (111 mL, 아급성기 의심)은 0.121 → 0.160으로 여전히 실패.
- 예측 component 총합 389 → 361 (GT 209): z 방향 문맥으로 흩어진 FP가 약간 줄었다.

### Subgroup (탐색적, n 작음 — 결론 아님)

| 군 | n | 2D | 2.5D | mean ΔDice | 개선 / 악화 |
|---|---|---|---|---|---|
| Thin (2 mm) | 28 | 0.718 | 0.732 | **+0.015** | 17 / 11 |
| Thick (4.4–5 mm) | 8 | 0.777 | 0.779 | +0.002 | 3 / 5 |
| < 1 mL | 5 | 0.765 | 0.771 | +0.006 | 2 / 3 |
| 1–10 mL | 15 | 0.687 | 0.691 | +0.005 | 8 / 7 |
| ≥ 10 mL | 16 | 0.762 | 0.782 | **+0.020** | 10 / 6 |
| ADC mm²/s | 18 | 0.729 | 0.748 | +0.019 | 10 / 8 |
| ADC 10⁻³ mm²/s | 8 | 0.751 | 0.754 | +0.003 | 5 / 3 |
| ADC 10⁻⁶ mm²/s | 10 | 0.718 | 0.724 | +0.007 | 5 / 5 |

설계 시 예상과 방향이 일치한다: 인접 slice가 실제로 가까운 thin 군에서 개선, 5 mm 간격의 thick
군에서는 차이 없음. 큰 병변에서 개선이 더 크다.

### Negative patient (sub-strokecase0151)

두 모델 모두 `false_positive_case`. FP 0.112 mL (14 voxel) → **0.216 mL (27 voxel)**, component 4 → 4.
1명이므로 비율 지표로 일반화하지 않는다.

## 8. 결론

- 2.5D는 2D 대비 **평균 Dice +0.012 (0.731 → 0.743)**, IoU·Sensitivity·Precision 평균도 모두 소폭
  상승했다. 그러나 **95% CI [−0.002, +0.028]이 0을 포함하고 Wilcoxon p = 0.29**로, seed 1회·
  validation 36명 기준에서 **2.5D의 우월성은 통계적으로 입증되지 않았다.** 사전 판정 기준(설계:
  "CI 하한 > 0이면 2.5D 우세") 기준 판정은 **"차이 없음 (2.5D 쪽으로 기운 경향)"**이다.
- 개선은 thin-slice·큰 병변에서 두드러지고 소수 환자에 집중된다. 이는 z 문맥 정보 가설과 일치하지만
  탐색적 관찰이다.
- 비용: 학습 시간 약 2배(CPU 병목), VRAM은 동일 수준, 파라미터 동일 수준.
- Test는 열지 않았다. Validation으로 best를 골랐으므로 두 수치 모두 낙관적이다.

## 9. 신규/변경 파일

| 파일 | 내용 |
|---|---|
| `configs/25d_baseline.yaml`, `configs/25d_tiny_overfit.yaml` | 2D config를 extends, context_slices 1 + 출력 경로 |
| `ai/neuroscan_ai/evaluation/comparison.py` | config diff, 쌍대 표, seeded bootstrap CI, Wilcoxon, 개선/악화 집계 |
| `scripts/compare_runs.py` | 두 run 비교 CLI. Test 접근·환자 불일치·허용 외 config 차이 시 거부, 기존 결과 덮어쓰기 금지 |
| `tests/test_phase4.py` | §3 |
| `scripts/train_2d.py` | docstring만 (2D/2.5D 공용 명시). 로직 변경 없음 |

## 10. Limitations

- **Seed 1회.** 0.01 수준 차이는 seed 변동과 구분되지 않는다 (CUDA 일부 비결정성 포함).
- Validation으로 checkpoint 선택 → 수치 낙관적. 독립 평가는 Test에서만 (아직 봉인).
- **강도 augmentation의 채널별 독립 적용 (R1).** `RandScaleIntensityd/RandShiftIntensityd`
  (`channel_wise=True`)가 2.5D에서는 같은 modality의 z−1/z/z+1에도 서로 다른 배율·shift를 준다. 2D에서는
  채널 = modality라 문제가 없었다. "context_slices만 변경" 원칙을 지키기 위해 코드를 바꾸지 않았으며,
  이 때문에 2.5D 이점이 과소평가되었을 가능성이 있다.
- Thick-slice(5 mm) 환자는 인접 slice 정보가 약해 2.5D 이점이 작다 (n = 8).
- Negative 환자 1명뿐 — FP 경향 판단 불가.
- Phase 2/3 limitation(XY 1.146 mm 그룹의 작은 component 손실, 원본 공간 Dice 상한, 작은 component
  자동 제거 없음) 유지.

## 11. 다음 단계

1. Phase 3·4 커밋 (현재 미커밋).
2. Seed 반복 (2D·2.5D 각각 seed 43, 44 추가, 약 6시간) — 차이가 seed 변동보다 큰지 확인.
3. (선택 ablation) modality 단위로 묶은 강도 augmentation으로 2.5D 재학습 — R1 영향 측정.
   2D에서는 동작이 동일해야 하며, 공식 비교와 분리해 보고.
4. 최종 모델 확정 후 **사용자 승인 하에** Test 37명 1회 평가 (2D·2.5D 모두, checkpoint 해시 고정 후).
   Phase 3 문서 §12·§18의 "Phase 5 Test" 표현은 이 정책으로 대체한다.
5. 원본 NIfTI → 예측 mask·정량 결과를 내는 독립 추론 진입점 (Backend 전제).
