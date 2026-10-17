# Phase 1 — Dataset QA, Preprocessing, Patient Split

작성일 2026-10-17 · 데이터: ISLES 2022 (250 cases, CC BY 4.0) · 결과 파일: `results/qa/`

## 1. 재현 방법

```powershell
.venv\Scripts\python.exe scripts\run_qa.py --dataset isles2022 --root data\raw\ISLES2022
.venv\Scripts\python.exe scripts\analyze_spacing_policy.py --dataset isles2022 --root data\raw\ISLES2022
.venv\Scripts\python.exe scripts\preprocess.py --config configs\preprocess_v1.yaml
.venv\Scripts\python.exe scripts\make_split.py --out data\splits\split_v1_seed42.json
.venv\Scripts\python.exe -m pytest            # -m data : 실제 데이터 통합 테스트만
```

| 산출물 | 내용 |
|---|---|
| `results/qa/case_summary.csv` | 환자별 QA 한 줄 (geometry, dtype, 강도 통계, mask, 병변 부피·component·Slice 범위) |
| `results/qa/dataset_summary.json` | 데이터셋 전체 요약 |
| `results/qa/raw_manifest.json` | raw 1,688개 파일 SHA-256 (raw 불변 검증용) |
| `results/qa/spacing_policy_comparison.json` | Resampling 정책 비교 실험 결과 |
| `data/processed/v1/<case>/{dwi,adc,mask}.nii.gz, meta.json` | 전처리 결과 |
| `data/processed/v1/cases.csv`, `preprocess_config.yaml` | 전처리 색인과 설정 사본 |
| `data/splits/split_v1_seed42.json` | 환자 단위 Split |

## 2. QA 결과

| 항목 | 결과 |
|---|---|
| Case | 250명 (`sub-strokecase0001`~`0250`), 누락 파일 0, 읽기 오류 0 |
| DWI / ADC / GT 격자 | 250명 모두 같음 (shape 일치, affine 허용오차 1e-3 mm) |
| FLAIR 격자 | 250명 모두 DWI와 다름 → 정합 필요 (MVP 제외, QA 통계만 기록) |
| Header zooms와 affine spacing | 250명 모두 일치 |
| Orientation | DWI/ADC/GT 모두 LAS (FLAIR: RAS 152, LAS 98) |
| Slice 축 | 250명 모두 3번째 배열 축이 가장 두꺼움 (axial 촬영) |
| 원본 dtype | DWI/ADC float64 191명, int16 59명 · GT uint16 181명, int16 47명, float64 22명 |
| GT가 정확히 0/1이 아닌 Case | 47명 (1이 0.9999999997로 저장됨) → `mask > 0.5`로 이진화 후 {0,1} 확인 |
| GT가 빈 Case | 3명: 0150, 0151, 0170 → `has_lesion=false`로 유지 (삭제하지 않음) |
| NaN | 0 |
| 병변 부피 (mL) | 중앙값 6.66, 평균 23.4, 최대 482.2 · 0: 3명, (0,1): 43명, [1,10): 95명, [10,50): 79명, ≥50: 30명 |
| 병변 component (26-연결) | 총 2,349개, 환자당 중앙값 5개, 최대 126개 |

### Spacing 분포 (DWI)

| spacing (mm) | 환자 수 | 대표 shape |
|---|---|---|
| 2.0 × 2.0 × 2.0 | 193 | 112×112×72~73 |
| 1.797 × 1.797 × 4.8 | 29 | 128×128×25 |
| 1.146 × 1.146 × 4.8 | 23 | 192×192×30 |
| 0.875 × 0.875 × 2.0 | 2 | 256×256×73 |
| 1.967 × 1.967 × 2.0 | 1 | 122×122×63 |
| 1.353 × 1.353 × 4.4 | 1 | 170×170×33 |
| 2.0 × 2.0 × 5.0 | 1 | 115×115×28 |

Slice 두께 3mm 초과(thick) 54명, 2mm(thin) 196명. 1 mL 미만 병변 43명과 Negative 3명은 모두
2mm 등방 그룹에 속합니다.

### ⚠️ ADC 강도 단위가 Case마다 다름

| ADC 중앙값 수준 | 환자 수 | 추정 단위 |
|---|---|---|
| ~0.001 | 106 | mm²/s |
| ~0.9 | 68 | 10⁻³ mm²/s |
| ~920 | 76 | 10⁻⁶ mm²/s (thick 53명 포함) |

ADC 값의 크기가 최대 10⁶배 차이 납니다. processed/v1은 강도를 바꾸지 않으므로, Phase 3 입력 파이프라인에서
**Case별 뇌 영역(nonzero) 기준 정규화**가 필수입니다. 고정된 ADC 임계값(예: 620×10⁻⁶ mm²/s)을 그대로
쓰면 안 됩니다. ADC 단위 그룹은 사실상 기관(center)을 나타내는 지표로도 볼 수 있습니다.

## 3. Resampling 정책 결정

GT를 정책 격자로 보냈다가 원래 격자로 되돌린 뒤 원본과 비교했습니다 (`analyze_spacing_policy.py`,
linear 보간 후 ≥ 0.5).

| 정책 | spacing | 2.5D z±1 | 보간으로 만든 Slice | 왕복 Dice (전체 / thick) | 사라진 component | 총 voxel |
|---|---|---|---|---|---|---|
| A 1mm 등방 | 1×1×1 | 1mm, 대부분 보간 | 56% | 0.991 / 0.997 | 187 | 1,795M |
| B XY 2mm, Z 유지 | 2×2×원본 | 실제 Slice (2 또는 4.4–5mm) | 0% | 0.985 / 0.934 | 224 | 198M |
| C 중앙값 2mm 등방 | 2×2×2 | 2mm, thick은 58% 보간 | 13% | 0.986 / 0.939 | 283 | 224M |
| **B' XY 1mm, Z 유지 (채택)** | 1×1×원본 | 실제 Slice | 0% | **0.9993 / 0.9975** | **68** | 792M |
| (참고) 변환 없음 | 원본 | 실제 Slice | 0% | 1.0 / 1.0 | 0 | 226M |

**채택: B' — XY 1.0 mm, Z 원본 유지, RAS 방향**

- **2.5D 인접 Slice 의미:** z-1/z/z+1이 항상 실제로 촬영된 Slice입니다. 다만 물리적 간격은 2mm(196명)
  또는 4.4–5mm(54명)로 다릅니다. Phase 4에서 thin/thick 그룹별 결과를 따로 보고합니다.
- **작은 병변 보존:** XY를 낮추는 일이 없어서 손실이 가장 적습니다(사라진 component 68/2349, 병변이
  완전히 사라진 Case 0).
- **연산량:** B의 약 4배이지만 2D 입력 약 224×224, 전체 float32 캐시 약 7GB로 RTX 5060 Ti 16GB와
  RAM 32GB에서 감당할 수 있습니다.
- **부피 계산:** 공식 병변 부피는 **원본 공간 GT**로 계산합니다(meta `original_volume_ml`).
  processed 부피는 참고값입니다(평균 상대오차 0.29%, 최대 7.1%).
- **원본 공간 복원:** Z가 voxel 단위로 그대로이므로 processed Slice k가 원본 Slice k와 정확히 대응합니다
  (통합 테스트로 검증). XY만 `resample_to_grid`로 원래 격자에 되돌리면 되고, 왕복 Dice 0.9993이
  복원 손실의 상한입니다.

## 4. processed/v1 내용

순서: GT 이진화(`> 0.5`, 원본 공간) → LAS→RAS 방향 변환(무손실) → XY 1.0mm resampling(DWI/ADC는 linear,
mask는 linear 후 ≥ 0.5, 첫 voxel 모서리 정렬, Z는 voxel 그대로 복사) → dtype 변환(영상 float32,
mask uint8). 강도 정규화는 하지 않습니다(Phase 3 입력 파이프라인에서 수행).

`meta.json`에는 case ID, 원본(shape/spacing/affine/orientation/dtype), lesion(원본·processed
voxel 수와 부피, `has_lesion`, `lost_in_processing`), processed(shape/spacing/affine/orientation/dtype/
files), preprocessing_version, 적용한 transform 목록, 소프트웨어 버전을 기록합니다.

결과: 250명 생성, `lost_in_processing` 0명, Negative 3명 `has_lesion=false`.

## 5. 환자 단위 Split (seed 42)

전략 (`neuroscan_ai/data/split.py`):
1. Negative 3명은 seed로 섞은 뒤 test → val → train 순서로 1명씩 배정합니다.
2. Positive는 (thin / thick Slice) × (병변 부피 사분위: 1.83 / 7.17 / 22.6 mL)의 8개 층으로 나눕니다.
3. 층별로 섞은 목록을 차례로, 목표 인원 대비 가장 덜 채워진 Split에 배정합니다. 크기는
   val = test = floor(250×0.15) = 37명, 나머지 176명은 train입니다.

| Split | n | Negative | thick | <1 mL | 부피 중앙값 (mL) | ADC 단위 그룹 (1e-3 / 1 / 1000) |
|---|---|---|---|---|---|---|
| train | 176 | 0150 | 38 | 31 | 6.60 | 70 / 50 / 56 |
| val | 37 | 0151 | 8 | 5 | 5.47 | 18 / 9 / 10 |
| test | 37 | 0170 | 8 | 7 | 7.75 | 18 / 9 / 10 |

Split 간 중복 0, 전체 포함 100%, 결정적(같은 seed, 입력 순서와 무관), QA CSV에서 다시 생성해도 동일합니다.
**Test 환자 37명은 학습, 튜닝, 임계값 선택, 모델 선택에 사용하지 않습니다.**

## 6. External Validation 확장 구조

- ISLES에만 해당하는 코드는 `neuroscan_ai/datasets/isles2022.py`(파일 탐색) 하나뿐입니다.
- QA(`qa/`), 전처리(`preprocessing/`), Split(`data/`)은 `CaseRecord`만 받는 범용 코드입니다.
- 외부 데이터셋은 `data/raw/external/<dataset>/`에 두고, `datasets/<dataset>.py` 어댑터를
  `ADAPTERS`에 등록한 뒤 같은 설정(`target_spacing: [1.0, 1.0, null]`)으로 전처리합니다. GT가 없는
  데이터는 `label=None`으로 처리할 수 있습니다.
- 원칙: 외부 검증에서는 ISLES로 학습한 모델을 다시 학습하지 않고 그대로 적용합니다. 외부 데이터 전체가
  Test 역할이므로 Split을 만들지 않습니다.

## 7. Phase 2로 넘기는 사항

- Empty GT 3명의 metric 정책은 `meta.lesion.has_lesion=false`를 이용해 evaluation engine에서
  명시적으로 정의합니다.
- 평가는 원본 공간 GT 기준으로 합니다(processed 예측 → `resample_to_grid` → 원본 격자).
- 작은 component는 지우지 않습니다(원본 Prediction 보존 원칙).
- Phase 3에서 ADC 단위 차이를 해결하는 Case별 강도 정규화가 필요합니다.
