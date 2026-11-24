# v2 2D Baseline 결과 (개발용 동결, 2026-11-24)

> Validation 결과다. Validation은 Checkpoint와 Threshold 선택에 함께 쓰였으므로 낙관적으로 편향될 수
> 있다. Test는 접근하지 않았다. 임상 성능 주장에 사용할 수 없다.

- 모델 `neuroscan_v2_2d_baseline_v1`(채널 36·72·144·288, 파라미터 2,439,577), 전처리
  `preprocess_v2_isles_1mm_inplane_v1`, Commit `2969fed`, Cache v2(Index Seal `7207ffad…`).
- 공식 Run: `v2-2d-baseline-seed42-multiseed-v1`, `v2-2d-baseline-seed43-multiseed-v1`,
  `v2-2d-baseline-seed44-multiseed-v1`. Commit `1901ce7`의 Seed 42 Run(r3·Run B)은 재현성 증거로만
  보존하며 집계에 넣지 않았다.
- 데이터: Train 176 / Validation 37(양성 36, 음성 1) / Test 0 / Final Test 0 / SOOP 0.
- 기록: `results/v2_multiseed_analysis/v2_2d_baseline_v1/`(git 제외, `files.sha256` 봉인).

## 1. 공식 Validation 대표값 (Threshold 0.50, Ungated, 평균 ± 표본 SD ddof=1, n=3)

| 지표 | 평균 ± SD | Seed 42 / 43 / 44 |
|---|---|---|
| Dice (양성 사례 평균) | 0.7786 ± 0.0089 | 0.7768 / 0.7707 / 0.7883 |
| Lesion-wise Recall | 0.8022 ± 0.0292 | 0.8278 / 0.7703 / 0.8086 |
| <0.1 mL Recall | 0.644 ± 0.039 | 0.680 / 0.602 / 0.650 |
| <0.5 mL Recall | 0.735 ± 0.039 | 0.769 / 0.692 / 0.744 |
| <1.0 mL Recall | 0.755 ± 0.036 | 0.787 / 0.716 / 0.763 |
| FP Volume (mL) | 118.81 ± 4.95 | 124.20 / 114.48 / 117.75 |
| No-GT-overlap Component 수 | 100.0 ± 3.0 | 103 / 97 / 100 |
| No-GT-overlap Component 부피 (mL) | 6.33 ± 1.58 | 7.56 / 4.54 / 6.89 |
| Oversegmentation (mL) | 112.48 ± 3.63 | 116.64 / 109.93 / 110.86 |
| Missed GT Component | 41.3 ± 6.1 | 36 / 48 / 40 |
| 음성 사례 FP (mL) | 0.099 ± 0.089 | 0.064 / 0.200 / 0.032 |
| Thick Dice | 0.816 ± 0.007 | 0.814 / 0.823 / 0.809 |
| Thick Recall | 0.725 ± 0.066 | 0.775 / 0.650 / 0.750 |
| Thick <0.1 mL Recall | 0.412 ± 0.102 | 0.471 / 0.294 / 0.471 |

Missed Lesion(Threshold 0.50, GT Component 209개, 겹침 1 voxel 이상을 탐지로 보는 탐색용 규칙):
세 Seed 공통 Missed 28개(그중 <0.1 mL 26개), 한 Seed라도 Missed 56개, 모든 Missed는 0.37 mL 이하
(최대 0.368 mL)다.

## 2. Threshold 결정

| 후보 | selected | 기록 |
|---|---|---|
| 0.50 | true | `frozen_for_development: true`, 정책 `threshold_v2_2d_baseline_fixed_0_50_v1`, source `multiseed_validation_stability_selected` |
| 0.45 | false | `rejected_as_frozen_threshold: true`, `rejection_reason: negligible_dice_gain_with_consistent_fp_increase` |

0.45 기각 근거: 3-Seed 평균 Dice 이득 +0.00072(Seed 간 Dice SD 0.008945), 평균 FP 증가 +5.463 mL,
평균 Oversegmentation 증가 +5.319 mL, No-GT-overlap Component 평균 +2개, Recall 이득은 Seed 43의 GT
Component 1개뿐이고, 같은 Validation 37건을 Checkpoint와 Threshold 선정에 함께 썼다.
0.50은 개발용 Threshold이며 임상 최종 Threshold가 아니다. Checkpoint 선택 Threshold도 0.50이다.
공식 Prediction은 Ungated, Gated는 비교용이다.

| 0.45 − 0.50 | Seed 42 | Seed 43 | Seed 44 | 평균 |
|---|---|---|---|---|
| Dice | +0.0003 | +0.0009 | +0.0009 | +0.0007 |
| FP (mL) | +4.99 | +6.06 | +5.34 | +5.46 |
| Oversegmentation (mL) | +4.87 | +5.92 | +5.17 | +5.32 |
| No-GT-overlap Component | +1 | +2 | +3 | +2 |
| Recall | 0 | +1 병변 | 0 | — |

비교는 저장된 Probability Map에 0.45와 0.50만 적용해 했다(모델 추론·재탐색 없음). 0.50 재계산은 세
Seed 모두 Run 기록과 정확히 같았다.

## 3. 한계

- Validation은 Checkpoint와 Threshold 선택에 공동 사용됐다. Validation 결과는 낙관적으로 편향될 수 있다.
- 음성 Validation 사례는 1건뿐이다.
- Test, Final Test, SOOP는 접근하지 않았다. 외부 검증은 완료되지 않았다.
- 임상 성능 주장에 사용할 수 없다.
- Thick Slice의 작은 병변 Recall이 주요 약점이다: Thick <0.1 mL Recall 0.412 ± 0.102(17개 Component,
  8개 사례).
- 세 Seed 공통 Missed GT Component는 28개이고, 그중 26개가 0.1 mL 미만이다. 모든 Missed Lesion은
  0.37 mL 이하다.
- FP의 약 95%는 병변 주변 Oversegmentation이다.
- 2D Architecture·Hyperparameter·Threshold는 추가 Validation 튜닝 없이 동결한다.

## 4. 동결의 의미

현재 개발 기준(모델·전처리·Threshold 0.50)을 더 이상 Validation 결과에 맞춰 바꾸지 않는다는 뜻이다.
Test 성능을 보장하지 않는다. `test_evaluated: false`, `final_test_accessed: false`,
`clinical_model: false`, `external_validation_completed: false`.

## 5. Cache Commit·Config 영향

- Cache v2 생성 Commit은 `2969fed`다. Cache Gate는 Cache 생성 Git Commit뿐 아니라 Config canonical
  SHA-256도 검증한다.
- 이번 문서·Config 변경이 Commit되면 Cache v2는 신규 학습 Gate에 쓸 수 없다. Config 내용이 바뀐 것만으로도
  (새 Commit 전이라도) `config_canonical_sha256_mismatch`로 거부되고, Dirty Working Tree도
  `working_tree_not_clean`으로 거부된다. 이는 구현된 Gate 동작이며 의도된 결과다. 세부 거부 조건은
  `docs/v2_2d_baseline_run_protocol.md` §15.
- 기존 완료 Run의 검증(verify_run, Seal, Hash)에는 영향이 없다.
- 신규 실제 학습을 하려면 Clean Commit에서 생성한 새 Cache Version이 필요하다. 현재 단계에서는 Cache를
  재생성하지 않는다.

## 6. 2.5D 설계 목표

Primary safety targets:

1. Thick Slice의 <0.1 mL Recall 개선 (2D: 0.412 ± 0.102)
2. 세 Seed 공통 Missed Lesion 28개 감소
3. 전체 Small-lesion Recall 개선 (2D: 0.644 / 0.735 / 0.755)
4. 병변 주변 Oversegmentation 감소 (2D: 112.48 ± 3.63 mL)

금지 사항:

- Test 결과를 보고 2.5D 설계를 변경하는 것
- 2D Threshold 재탐색
- 기존 2D Run 수정
- SOOP 자동 통합
- 2.5D 구현·학습 자동 시작
- 2D 결과를 보고 2.5D Threshold를 사후 선택하는 것
