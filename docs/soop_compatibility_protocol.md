# SOOP 호환성 프로토콜 (Format Adapter · 메타데이터 집계)

> 연구 및 교육 목적의 프로토타입이며, 임상 사용이 승인된 의료기기가 아닙니다.
> 이 문서는 공개 데이터셋 SOOP을 NeuroScan 입력 형식으로 **형식만 변환**하는 규칙과, 공개 메타데이터
> 집계 결과를 정리합니다. 전처리·Brain Mask·ADC 변환 정책은 정의하지 않으며, 모델 평가 문서가 아닙니다.
> 사례별 수치와 Alias는 포함하지 않습니다.

관련 문서: [외부 데이터셋 요구조건](external_dataset_requirements.md) ·
[실패 사례 정성 분석 요약](v1_failure_analysis_summary.md) · [고정 내부 평가 보고서](final_test_report.md)

## 1. 데이터셋과 고정 버전

| 항목 | 값 |
|---|---|
| 데이터셋 | Stroke Outcome Optimization Project (SOOP), OpenNeuro ds004889 |
| 버전 | **1.1.2** (DOI 10.18112/openneuro.ds004889.v1.1.2, Git 태그 commit `a62344db73f18f384a95b452f4a1a423087e059b`) |
| 라이선스 | CC0 (`dataset_description.json`) |
| 참고 논문 | Absher et al., Scientific Data 2024 (doi 10.1038/s41597-024-03667-5) |
| 파일 고정 | OpenNeuro snapshot 1.1.2 URL → S3 object `versionId`; NIfTI는 git-annex SHA-256, 텍스트는 git blob SHA-1로 검증 |

다른 버전의 파일을 섞지 않는다. 버전이 바뀌면 이 문서와 집계를 새로 만든다.

## 2. 파일럿 목적과 영구 제외

- 목적: SOOP의 TRACE·ADC·Acute 마스크가 NeuroScan 입력 구조와 **형식상** 호환되는지 확인.
  학습·성능 평가가 아니다.
- 파일럿 5건은 공개 파일 트리의 적격 후보 중 `SHA-256("neuroscan-soop-pilot-v1:" + subject_id)`가 작은
  순서로 사전 선정했다(선정 전 영상 미열람).
- 파일럿 5건은 **Adapter·QA 개발과 회귀 검증 전용**으로 영구 고정한다. 외부검증 성능 집계, v1 호환성 연구의
  성능 수치, v2 Train·Validation·보류 세트, v2 Final Evaluation에서 모두 제외한다.

## 3. 입력 파일과 Ground Truth

| 역할 | BIDS 경로 |
|---|---|
| DWI (모델 입력 `dwi`) | `sub-*/dwi/sub-*_rec-TRACE_dwi.nii.gz` + `.json` |
| ADC (모델 입력 `adc`) | `sub-*/dwi/sub-*_rec-ADC_dwi.nii.gz` + `.json` |
| **주 GT** | `derivatives/lesion_masks/sub-*/dwi/sub-*_space-TRACE_desc-lesionAcute_mask.nii.gz` |
| 선택 마스크 | `..._desc-lesion_mask.nii.gz`, `..._desc-lesionChronic_mask.nii.gz` |
| 데이터셋 공통 | `dwi.bval`, `dwi.bvec` (선택) |

- 주 GT는 `desc-lesionAcute`이다. NeuroScan v1이 급성 DWI 병변 분할을 대상으로 학습·평가했기 때문이다.
- `desc-lesion`의 공식 정의는 논문에 없다. 파일럿에서 Acute와 Chronic의 합집합과 일치한 사례가 있었으나
  관찰일 뿐이며 정의로 사용하지 않는다.
- 필수 5개 파일 중 하나라도 없으면 적격에서 제외하고 사유만 기록한다. **마스크가 없는 사례를 Negative로
  해석하지 않는다.** Negative 정책은 별도로 정한다.

## 4. Format Adapter (`ai/neuroscan_ai/datasets/soop.py`)

### 4.1 하는 일

- BIDS 경로 규칙으로 필수·선택 파일 탐색
- 마지막 singleton 축 제거: `(X, Y, Z, 1)` → `(X, Y, Z)` (4.3)
- TRACE 기준 ADC·마스크의 shape, affine(허용오차 1e-3 mm), spacing, orientation 검증 — 불일치 시 거부
- 마스크가 정확히 {0, 1}인지 검증 — 아니면 거부
- ADC Storage Group과 ADC Content Status **표시** (5, 6)
- 출력: `data/interim/soop_format_v1/<alias>/{dwi,adc,label}.nii.gz, metadata.json, audit.json`
  (`<alias>.part`에 쓴 뒤 이름 변경, 기존 폴더 덮어쓰기 거부)
- 감사 기록: 원본 SHA-256(작업 전후 비교), 파생 SHA-256, 파일별 처리 방식, 보존 검증 결과, 미수행 작업 목록,
  소프트웨어 버전. 출력에는 Alias와 해시만 기록하고 원본 ID·원본 경로는 넣지 않는다.

### 4.2 하지 않는 일

Resampling, Skull stripping, Brain extraction, 강도 정규화, ADC 값·단위 변환, 음수값 제거, 배경 코드 변경,
마스크 이진화·수정·병합, component 제거, 다른 영상으로 대체. Adapter는 전처리·Brain Mask·학습·추론 코드를
import하지 않는다(자동 테스트로 확인).

### 4.3 Singleton 축 제거 조건

허용:

- 헤더 `dim[0] = 3`이고 공간축에 크기 1이 없는 3D 입력 → 바이트 그대로 복사
- 헤더 `dim[0] = 4`, `dim[4] = 1`, 실제 shape `(X, Y, Z, 1)`, 공간축에 크기 1 없음 → 마지막 축 제거

거부: 마지막 축이 2 이상, 공간축이 1인 경우, 5D 이상, 헤더 dims와 실제 배열 shape 불일치.

### 4.4 Raw·Scaled Voxel 보존 검증

SOOP 영상은 int16 저장값과 NIfTI `scl_slope` / `scl_inter`(공개 전 SPM 기반 비식별화 과정에서 다시 쓰인
것으로 판단)를 가진다. NiBabel은 저장 과정에서 scaling을 다시 계산할 수 있으므로 헤더 복사만으로 보존을
판단하지 않는다.

| 방법 | 합성 비교 결과 (NiBabel 5.4.2) |
|---|---|
| 1. int16 raw 저장값 + 원본 scl_slope·scl_inter 유지 | raw, float64 분석값, float32 분석값 모두 비트 일치 |
| 2. 분석값을 float32로 저장(slope 1, inter 0) | float64 분석값 기준 최대 약 0.0019 차이 |

**방법 1을 사용한다.** 파생 파일을 다시 열어 다음을 모두 비교하고, 하나라도 다르면 파일을 지우고 사례를 거부한다.

- raw 저장값, float64·float32 분석값, dtype, 실제 적용 scaling
- 공간 affine(비트 일치), qform·sform 코드와 quaternion·srow 필드, `xyzt_units`, `pixdim[0:4]`, orientation
- 공간 3축 shape (4D 입력에서 빠지는 것은 `pixdim[4]`뿐이며 기록한다)

분석에 쓰이는 scaled voxel 값의 동일성이 최종 기준이다.

## 5. ADC Storage Group

> **고지: P / S / N은 파일럿 5건에서 관찰한 저장 패턴 이름이며, 물리 ADC 단위가 아니다.**

| 그룹 | 정의 (모든 필드가 정확히 일치해야 함) |
|---|---|
| P | SeriesDescription·ProtocolName `ddADC`, PhilipsRescaleSlope 9.66995e-4, PhilipsRescaleIntercept 0, PhilipsRWVSlope 9.66995e-7, PhilipsRWVIntercept 0, SoftwareVersions `5.6.1\5.6.1.0`, UsePhilipsFloatNotDisplayScaling 1 |
| S | `dADC`, RescaleSlope 0.966995, RescaleIntercept 0, RWVSlope 0.966995, RWVIntercept 0, SoftwareVersions `3.2.3`, UsePhilipsFloat… 1 |
| N | `dADC`, RescaleSlope 1.27773e-6, RescaleIntercept −0.00193634, RWVSlope 1.27773e-6, RWVIntercept −0.00193634, SoftwareVersions `5.6.1\5.6.1.0`, UsePhilipsFloat… 1 |
| unknown | 위와 하나라도 다르거나 필드가 없는 경우 |

- 숫자는 상대오차 1e-6, 문자열은 정확히 일치해야 한다. PhilipsScaleSlope는 같은 그룹 안에서도 촬영마다 달라
  "존재하고 0보다 큰 숫자"를 조건으로 한다.
- 근거: dcm2niix 공식 문서의 Philips 변환식(`DV = SV·RS + RI`, `FP = DV/(RS·SS)`)과 파일럿 저장값의 값
  간격(lattice) 일치(P·N은 FP, S는 DV). 물리 단위·변환식 적용 여부는 데이터 제공자 확인 전까지 확정하지 않는다.
- Storage Group은 ADC 값 변환에 사용하지 않는다. 비슷해 보이는 조합을 근거 없이 같은 그룹으로 묶지 않는다.

## 6. ADC Content Status

`rec-ADC` 파일이 실제로 ADC인지 사이드카만으로 보수적으로 판정한다.

| 상태 | 조건 |
|---|---|
| `confirmed_adc` | ImageType에 ADC가 있고, SeriesDescription과 ProtocolName이 모두 ADC 계열 이름이며, B0 표기가 없음 |
| `probable_adc` | ImageType에 ADC가 있고, 두 이름 모두 ADC 계열이 아니지만 B0 표기도 없음(예: DWI 촬영 이름을 딴 파생 시리즈) |
| `suspicious_non_adc` | ImageType에 ADC가 없고, 두 이름 모두 B0·b-value·DWI·diffusion 계열 |
| `unknown` | 필드 누락 또는 모순(예: 이름은 ADC인데 ImageType이 DIFFUSION/MIXED, ImageType은 ADC인데 이름이 B0) |

### 6.1 v2 데이터 풀 정책 (확정)

| Content Status | 풀 상태 (`v2_pool_status`) | 처리 |
|---|---|---|
| `confirmed_adc` | `eligible_candidate` | v2 적격 **후보** (분할 전 별도 QA·층화 설계 필요) |
| `probable_adc` | `review_required` | **자동 Train / Validation 포함 금지.** 후속 영상값 QA를 통과한 경우에만 `confirmed_adc`로 승격 가능 |
| `suspicious_non_adc` | `auto_excluded` | 자동 제외 |
| `unknown` | `auto_excluded` | 자동 제외 |
| 파일럿 5건 (모든 상태) | `pilot_permanently_excluded` | Adapter·QA·회귀 테스트 전용. Train, Validation, 보류 평가, R1, R3 및 모든 성능 집계에서 영구 제외 |

- 제외되거나 검토 대기인 사례를 근거 없이 포함하지 않는다. 승격 기준과 영상값 QA는 별도 승인 후 정한다.
- 이 정책은 풀 상태 표시일 뿐이며, 이 단계에서 v2 Split을 만들지 않는다.
- Adapter는 이 판정으로 데이터를 수정하거나 다른 영상으로 대체하지 않는다.

## 7. 공개 메타데이터 집계 결과 (1.1.2, 적격 1,451건)

출처: 적격 사례의 TRACE·ADC 사이드카 2,902개(파일럿 5건 재사용, 나머지는 텍스트만 다운로드)와 공통 `dwi.bval`.
영상은 추가로 받지 않았다. 사전 확인에서 응답이 없던 2개는 1회 재시도로 확보해 `metadata_incomplete`는 0건이다.

| 항목 | 결과 |
|---|---|
| 피험자 폴더 / Acute 마스크 없음 / 적격 | 1,715 / 264 / 1,451 |
| ADC Storage Group | P 369 · S 209 · N 223 · unknown 650 |
| ADC Content Status | confirmed 1,403 · probable 27 · suspicious_non_adc 19 · unknown 2 |
| v2 풀 상태 (content 기준) | eligible_candidate 1,403 · review_required 27 · auto_excluded 21 |
| 파일럿 제외 후 eligible_candidate | 1,398 (파일럿 5건은 모두 confirmed_adc) |
| Slice 두께 / 간격 | 두께 5 mm 1,449 · 4 mm 2 / 간격 **6 mm 1,445** · 6.2 mm 2 · 6.071 mm 1 · 5 mm 3 |
| 자장 세기 | 1.5 T 1,154 · 3 T 297 |
| 제조사 | Philips 1,427 · GE 16 · Siemens 8 |
| 장비 모델 | Ingenia Ambition X 796 · Achieva 366 · Ingenia Elition X 176 · Intera 89 · Signa HDxt 16 · Symphony 8 |
| 변환 소프트웨어 | dcm2niix v1.0.20201102 (전부) |
| Philips scaling 필드 누락률 | 약 1.7–1.9% (GE·Siemens 사례) |
| `Units` 필드 / b-value 필드 | 전부 없음 / 없음 |
| `dwi.bval` | 데이터셋 공통 파일 1개, 값 `0` 하나; 사례별 bval 없음 → b-value는 메타데이터로 확인 불가 |

Storage Group이 unknown인 650건 중 602건은 content가 `confirmed_adc`이다. 즉 대부분은 ADC로 보이지만 저장
방식을 아직 설명하지 못한 경우이다.

## 8. v1 범위와의 차이

| 항목 | v1 (ISLES 2022 학습 범위) | SOOP 관찰 |
|---|---|---|
| Slice 간격 | 2–5 mm | 사실상 전부 6 mm (5 mm는 3건) |
| Skull stripping | 되어 있음 (`skull_stripped_background`) | 되어 있지 않음, 두피 포함 공개, Brain Mask 미제공 |
| 영상 차원 | 3D | TRACE·ADC가 `(X, Y, Z, 1)` |
| ADC 저장 | 세 단위 그룹 | 촬영 소프트웨어·시리즈에 따라 다른 Philips scaling, 일부 음수 배경 코드 |
| 제조사 | 여러 기관 | Philips 약 98%, 단일 기관 |

따라서 SOOP은 동결된 v1 정책을 그대로 적용하는 외부검증 데이터가 아니다.

## 9. 미구현 정책 (별도 승인 대상)

- **Brain Mask 정책: 미구현.** v1 배경 기반 Mask는 SOOP에서 두피를 포함한다. Brain extraction 도구 비교와
  정책 등록은 별도 승인 후 진행하며, ISLES와 SOOP에 같은 정책을 적용하는 v2 정책으로만 정의한다.
- **ADC 변환 정책: 미구현.** 공식 근거(데이터 제공자 확인 포함) 없이 clip, abs, offset, 단위 변환,
  음수 0 대체를 하지 않는다.
- Slice 두께 정책(원본 Z 유지와 mm 기준 context 비교 등)은 v2 설계에서 정한다.

## 10. 권장 역할

| 역할 | 판정 |
|---|---|
| v1 원형 외부검증 | **부적합** (Skull stripping 없음, 학습 범위 밖 6 mm, ADC 저장 방식 미확정) |
| v2 Final Evaluation | **부적합** (단일 기관, 기관 독립성 없음) |
| v2 Train / Validation | **후보** (환자 단위 층화 분할, 파일럿 5건 제외, content 기준 통과 사례만) |
| v1 호환성 연구 | 별도 프로토콜과 승인이 있을 때만, 결과는 "외부검증"으로 부르지 않음 |

## 11. 저장 위치와 공개 범위

| 내용 | 위치 | Git |
|---|---|---|
| 파일럿 원본 영상·사이드카, Alias 대응표, 다운로드 기록 | `data/raw/soop_pilot/` (`_internal/`) | 제외 |
| 공개 사이드카, 파일 트리, 계획·다운로드 기록 | `data/raw/soop_meta_v1_1_2/` (`_internal/`) | 제외 |
| 형식 변환 파생 파일 | `data/interim/soop_format_v1/` | 제외 |
| 집계 결과 | `results/soop_compatibility/v1/` | 제외 |

- 공개 출력에는 Alias와 집계만 사용한다. 원본 Subject ID는 `_internal/`의 로컬 파일에만 둔다.
- 집계 출력에 원본 ID가 섞이면 결과를 쓰지 않고 거부한다.

## 12. 실행

```
.venv\Scripts\python.exe scripts\build_soop_format.py          # 파일럿 형식 변환
.venv\Scripts\python.exe scripts\soop_sidecars.py plan         # 트리·적격 목록·URL 사전 확인
.venv\Scripts\python.exe scripts\soop_sidecars.py download     # 사이드카 다운로드 (텍스트만)
.venv\Scripts\python.exe scripts\soop_sidecars.py retry        # 실패 파일 1회 재시도 (재실행 거부)
.venv\Scripts\python.exe scripts\soop_sidecars.py aggregate    # 집계만 생성
```

자동 테스트: `tests/test_soop_adapter.py` (합성 NIfTI·JSON만 사용).
