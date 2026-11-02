# NeuroScan 외부 데이터셋 요구조건 및 역할 분리 정책

> 연구 및 교육 목적의 프로토타입이며, 임상 사용이 승인된 의료기기가 아닙니다.
> 이 문서는 **특정 데이터셋을 선정하는 문서가 아닙니다.** 외부 데이터 후보를 평가하기 위한 사전 요구조건이며,
> 데이터에 접근하기 전에 역할(R1 / R2 / R3)과 보호 절차를 확정하기 위한 정책입니다.

관련 문서: [고정 내부 평가 보고서](final_test_report.md) ·
[실패 사례 정성 분석 요약](v1_failure_analysis_summary.md) · [Final Test 프로토콜](final_test_protocol.md)

## 1. 목적

- 동결된 NeuroScan v1이 ISLES 2022 밖의 기관·장비·프로토콜에서도 비슷하게 동작하는지 확인
- v1 실패 분석에서 나온 가설 후보(thick slice와 context, 작은 병변, ADC 단위, negative 사례)를 검증할 수 있는
  데이터 분포 확보
- NeuroScan v2의 개발 데이터와 독립 평가 데이터를 처음부터 분리

## 2. 데이터 역할

| 역할 | 용도 | 대상 모델 | 접근 시점 |
|---|---|---|---|
| **R1** | NeuroScan v1 External Validation | 동결된 v1 (`neuroscan-v1.0.0-test-locked`) | 프로토콜 동결 후 1회 |
| **R2** | NeuroScan v2 Train / Validation | 개발 중인 v2 | v2 개발 기간 |
| **R3** | NeuroScan v2 Final Evaluation | 동결된 v2 | v2 동결 후 1회 |

## 3. 역할 분리 원칙

- 가능하면 R1, R2, R3에 서로 다른 데이터셋을 쓴다.
- 하나의 큰 데이터셋을 나눠야 한다면 **환자·기관 단위로 먼저 분리**한 뒤 역할을 지정한다(같은 기관이 여러 역할에
  걸치지 않는 것을 우선).
- 역할은 데이터 접근 전에 문서와 split 해시로 고정하며, **지정 후 변경하지 않는다.**
- 역할 간 환자, 기관, 영상 해시 중복을 검사한다(환자 ID가 없으면 영상 해시와 메타데이터로 확인).
- **R3는 v2 개발 중 봉인**하며, 코드 수준 봉인(SplitGuard와 동등)과 접근 감사 기록을 둔다.
- R1 결과는 v2 설계의 참고 정보일 뿐이며, v2 모델 선택 기준으로 쓰지 않는다.
- ISLES 2022 Test는 이미 평가와 분석에 사용되었으므로 **R3로 재사용하지 않는다.** ISLES 2022 Train·Validation은
  R2 후보가 될 수 있다.

## 4. 필수 입력

- DWI
- ADC
- Binary lesion Ground Truth
- 환자 단위 구분(한 환자의 영상이 여러 역할에 흩어지지 않도록)
- Affine, voxel spacing, 영상 방향, shape 및 격자 정보
- DWI·ADC·GT의 공간 대응(같은 격자이거나 문서화된 정합 관계)

## 5. Ground Truth 문서화

다음이 문서로 확인되어야 한다.

- 주석 작성자 수와 숙련도 범주
- 주석에 사용한 기준 영상(DWI, ADC, 기타)
- 여러 작성자가 있을 때의 합의 방식
- 주석 수정·검수 과정
- Binary label 정의(포함·제외 기준)
- Negative case 정의
- 가능하면 관찰자 간 변동 정보

## 6. 기하 조건

- DWI·ADC·GT shape 확인
- Affine 일치 또는 정합 관계 문서화
- Orientation, obliquity
- XY spacing, Z spacing, slice 수
- Thick / Thin 분류(v1 기준과 같은 방식으로 기록)
- 물리적 좌표(world mm) 복원 가능 여부

v1 입력 검증 범위(spacing, slice 수, 뇌 비율 등) 밖의 사례는 제외하지 말고 별도로 집계해 보고한다.

## 7. 우선 확보 조건

v1 실패 분석의 관찰을 검증할 수 있도록 다음 순서로 우선한다.

1. 문서화된 GT 품질
2. ISLES 2022와 다른 기관·장비
3. 1 mL 미만 병변
4. Thick slice 사례
5. Negative 사례
6. 여러 ADC 단위(저장 형식)
7. 여러 component로 이루어진 병변
8. 다양한 obliquity
9. 가능하면 장비 제조사와 자장 세기 정보

## 8. 혼재 변수 분리

v1 분석에서는 ADC 단위와 slice 두께가 같은 사례에 겹쳐 두 효과를 분리할 수 없었다. 가능하면 다음 조합을 모두
확보한다.

| | ADC 1e−6 | ADC 1e−3 |
|---|---|---|
| Thin slice | 확보 | 확보 |
| Thick slice | 확보 | 확보 |

ADC 단위와 slice 두께가 완전히 겹치는 데이터만으로는 두 효과를 분리하기 어렵다. 기관이 두 변수와 함께 바뀌는
경우에도 같은 한계가 있으므로, 기관 정보를 함께 기록한다.

## 9. R1 요구사항 (동결된 v1 External Validation)

- v1 threshold 0.5 유지
- v1 preprocessing(processed/v1) 유지
- v1 brain mask 정책(`skull_stripped_background`) 유지
- 후처리 없음, 작은 component 제거 없음
- 원본 공간(original space) 평가
- v1 Final Test의 manifest·lock 절차와 동등한 평가 보호(사전 등록, 해시 동결, 승인, 1회 실행, 감사 기록)
- 결과로 v1을 다시 조정하지 않음

Skull stripping이 되지 않은 데이터는 R1 우선 후보에서 분리해 별도 검토 대상으로 둔다. 새 brain mask 정책이 필요하다면
그 평가는 동결된 v1과 동일한 평가가 아닐 수 있으므로, R1 실행 전에 별도 프로토콜로 고정하고 결과도 구분해 보고한다.

## 10. R2 요구사항 (v2 Train / Validation)

- 환자 단위 Train / Validation 분할
- 기관·장비·병변 크기·slice 두께·ADC 단위로 층화
- 작은 병변 사례를 충분히 포함
- Negative 사례 포함
- 다중 seed 실험이 가능한 규모
- Modality-consistent augmentation ablation이 가능한 규모
- Thick slice와 context 정책 실험이 가능한 분포

## 11. R3 요구사항 (동결된 v2 Final Evaluation)

- R1, R2, ISLES 2022와 환자 중복 없음
- 가능하면 기관도 분리
- v2 동결 전 접근 금지
- Primary endpoint, 통계 방법, subgroup을 첫 접근 전에 사전 고정
- 1회 평가 보호(manifest, 해시 동결, 승인, lock, 감사 기록)
- 환자별 결과 비공개, 집계 결과만 보고

## 12. Dataset QA

각 데이터셋은 역할 지정 전에 다음 QA를 통과해야 하며, 결과는 집계만 문서화한다.

- 파일 존재와 손상 검사
- DWI·ADC·GT pairing
- 빈 영상 검사
- Shape·affine·spacing
- Orientation·obliquity
- Intensity 범위
- ADC 단위 그룹 판정(판정 불가 사례 표시)
- Binary GT 검사
- GT의 brain mask 밖 voxel
- Lesion volume 분포
- Component count 분포
- Negative 비율
- Brain mask 비율(v1 정책 기준)
- 비식별화 상태
- DICOM / NIfTI header 검사(개인정보 문자열 잔존 여부)
- 기관·장비 메타데이터 확보 여부
- 기존 데이터(ISLES 2022, 다른 역할)와의 중복 검사
- License와 사용 승인 확인

## 13. Dataset Adapter

- `ai/neuroscan_ai/datasets/` 아래 데이터셋별 adapter로 구현한다.
- Adapter는 경로 탐색과 case mapping만 담당하고, 기존 전처리 코드를 우회하지 않는다.
- 합성 데이터 테스트를 포함한다.
- QA 집계 보고서를 생성한다.
- 원본 데이터와 결과는 Git에서 제외한다(`data/`, `results/` ignore 규칙).
- 원본 식별자는 UI나 공개 문서에 노출하지 않는다(alias 사용).

## 14. 표본 수

특정 숫자를 임의로 확정하지 않는다. 역할별로 다음을 고려해 별도로 계산하고, 계산 근거를 데이터 접근 전에 문서화한다.

- Primary endpoint
- 기대 효과 크기
- 환자별 차이의 표준편차
- 원하는 신뢰구간 폭
- Negative 비율
- Subgroup별 최소 n(v1 프로토콜은 n < 10이면 기술통계만 보고)
- 기관 수
- 병변 크기 분포
- Thick / Thin 비율
- 탈락 또는 사용 불가 사례 비율

v1 고정 내부 평가의 환자별 Dice 차이 표준편차(약 0.074)는 후속 비교 설계의 참고값일 뿐이며, 외부검증 표본 수를
자동으로 결정하는 값이 아니다.

## 15. 라이선스·윤리

데이터 사용 전에 다음을 확인하고 기록한다.

- 데이터 라이선스
- 재배포 가능 여부
- 상업적 사용 제한
- 인용 요구사항
- IRB 필요 여부
- 비식별화 책임 주체
- 데이터 사용 승인 문서
- 원본 저장 위치와 접근 권한
- 접근 감사 로그
- 삭제·보존 정책
