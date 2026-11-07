# Brain Extraction 파일럿 프로토콜

> 연구 및 교육 목적의 프로토타입이며, 임상 사용이 승인된 의료기기가 아닙니다.
> 이 문서는 Brain Mask 도구를 **비교·선정하는 절차**를 정의합니다. 현재까지 실행 래퍼, QA 코드, SynthStrip
> Docker 실행기, 합성 테스트가 구현되어 있고, SynthStrip 1.8 CPU 이미지로 **합성 영상 1건의 Smoke Test**만
> 수행했습니다. 실제 SOOP·ISLES MRI 처리와 Brain Mask 생성, HD-BET 설치는 하지 않았습니다.
> 사례별 수치와 Alias 대응은 포함하지 않습니다.

관련 문서: [SOOP 호환성 프로토콜](soop_compatibility_protocol.md) ·
[외부 데이터셋 요구조건](external_dataset_requirements.md)

## 1. 목적과 필요성

- SOOP 영상은 두피와 뇌 외부 조직을 포함하고 공식 Brain Mask가 없다.
- NeuroScan v1의 `skull_stripped_background` 규칙은 Skull-stripped 영상(ISLES 2022)을 전제로 하므로 SOOP에서는
  두피·외부 조직까지 Mask에 포함한다.
- v2에서 ISLES와 SOOP에 **같은 Brain Mask 정책**을 적용하기 위해 도구를 비교하고, 품질 기준을 통과한 도구만
  정책으로 고정한다.

## 2. 최우선 원칙

1. **병변 보존**: Acute GT Voxel이 하나라도 Brain Mask 밖이면 `gt_preservation_failure = true` → 실패.
2. **뇌 조직 보존**: 뇌 조직을 잘라내는 Mask는 두피 제거가 깨끗해도 채택하지 않는다.
3. Mask를 자동 팽창·수정·보정하지 않는다. 격자가 맞지 않으면 Resampling하지 않고 실패로 기록한다.
4. 가장 작은 Mask나 가장 깨끗해 보이는 Mask를 고르지 않는다.
5. **모델 성능으로 도구를 선택하지 않는다.** 도구는 모델 추론 전에 Mask 품질만으로 고른다.

## 3. 입력 정책 — Policy A (확정)

- SOOP와 ISLES 모두 **TRACE / DWI**에서 Brain Mask를 만든다(SOOP은 Format Adapter의 3D 파생본).
- 같은 격자의 ADC에 **동일 Mask**를 적용한다. ADC에서 별도 Mask를 만들지 않는다.
- T1w·FLAIR는 사용하지 않는다(추가 정합과 추가 다운로드가 필요하므로).

## 4. 비교 대상 (1차)

| 대상 | 버전 | 기본 설정 (사례별 조정 금지) |
|---|---|---|
| v1 배경 Mask (기준선) | v1 동결 코드 `skull_stripped_background` (읽기 전용 import) | — |
| SynthStrip | 1.8 (`freesurfer/synthstrip:1.8`, 모델 `synthstrip.1.pt`) | `-m`, 경계 `-b 1` mm, CSF 포함(`--no-csf` 미사용), CPU 또는 GPU 모드 기록 |
| HD-BET | 2.0.1 | `--save_bet_mask`, TTA 기본값(켬), 장치 기록, 입력 방향 표준화 → 실행 → 원래 방향·공간 복원 |

- **FSL BET은 1차 비교에서 제외**한다. 공식 문서상 Windows는 WSL로만 지원되고, 라이선스가 비상업 용도로
  제한되며, 확산 영상에서는 b0 영상 대상 관례만 문서화되어 있다. SynthStrip과 HD-BET이 모두 품질 기준을
  충족하지 못할 때에만 별도 승인 후 예비 후보로 검토한다.
- 같은 도구는 SOOP와 ISLES에 **같은 설정**을 적용한다. 결과를 보고 사례마다 설정을 바꾸지 않는다.

### 4.1 공식 근거 요약

| 항목 | SynthStrip | HD-BET |
|---|---|---|
| DWI 근거 | 논문에서 DWI(단일 확산 방향)와 5–6 mm thick-slice 임상 영상 평가. TRACE·ADC 평가는 없음 | T1·T1c·T2·FLAIR로 학습. DWI 근거 없음 |
| 공간 처리 | 내부적으로 1 mm·LIA로 변환 후 Mask를 원래 입력 공간으로 되돌림 | 입력이 MNI152 방향과 맞아야 함(`fslreorient2std` 권장). 3D만 지원 |
| Windows | 공식 지원 문구 확인 필요 (Docker·Apptainer·FreeSurfer 경로 제공) | "should work" 수준, 확인 필요 |
| 라이선스 | FreeSurfer Software License (연구용, 임상 사용 비권장). 가중치는 MIT 또는 CC BY 4.0 | Apache-2.0 |
| 인용 | Hoopes et al., NeuroImage 2022, doi 10.1016/j.neuroimage.2022.119474 | Isensee et al., Hum Brain Mapp 2019, doi 10.1002/hbm.24750 (+ nnU-Net) |

TRACE·ADC 적용 가능성은 공식 근거가 없으므로 이 파일럿으로 직접 검증한다.

## 5. 구현 (`ai/neuroscan_ai/brain_extraction/`)

| 모듈 | 역할 |
|---|---|
| `interfaces.py` | `RunRequest` / `RunResult`, 도구 이름(`v1_background`, `synthstrip`, `hd_bet`), 상태, Alias 규칙 |
| `runners.py` | v1 기준선 Runner, SynthStrip·HD-BET 명령 명세와 Runner, 무손실 방향 변환, Mock Runner |
| `qa.py` | 기하·값·구조·병변 보존·기준선 비교 측정과 상태 판정 |
| `figures.py` | Panel A–D 정적 QA 그림과 자동 Layout 검사 |
| `audit.py` | 감사 기록, 해시, 원본 ID 차단, 덮어쓰기 금지 JSON 쓰기 |
| `selection.py` | SOOP 파일럿 고정 목록, ISLES 결정적 선정, 도구 선택 규칙 |
| `executors.py` | SynthStrip Docker 실행기(고정 정책 `bm_synthstrip_1_8_cpu_pilot`). 프로세스를 실행하는 유일한 모듈 |

CLI: `scripts/run_brain_extraction_pilot.py` (`specs`: 명령 명세 출력, `run`: 현재 단계에서는 외부 도구 거부).

- `executors.py`만 `subprocess`를 사용하며(항상 argv 목록, `shell=False`), 나머지 모듈은 프로세스·네트워크·
  모델 라이브러리를 import하지 않는다(자동 테스트로 확인). Runner는 실행기를 import하지 않고 주입받는다.
- 외부 도구 Runner는 실행기(executor)를 주입해야만 동작하며, 기본 실행기는 실행을 거부한다.
  현재 구현된 실제 실행기는 SynthStrip Docker 실행기 하나이며(13장), HD-BET 실행기는 없다.
- 출력: `<out>/<alias>/<tool>/{brain_mask.nii.gz, qa.json, audit.json, figure_info.json, run.log}`.
  `.part` 폴더에 쓴 뒤 이름을 바꾸며, 기존 폴더는 덮어쓰지 않는다. 저장되는 Mask는 TRACE 격자·qform/sform
  코드를 가진 uint8 0/1 영상이고, 도구 원출력도 같은 폴더에 보존한다.
- 감사 기록: 도구·버전, 명세(SynthStrip: container digest·model hash·인자·장치 / HD-BET: weight hash·인자·
  TTA·장치), 실행 명령, 외부 실행 여부, Mock 여부, 방향 변환 기록, 입력·출력 SHA-256, 미수행 작업 목록.
  SynthStrip은 digest·model hash가 고정 정책에 기록되어 있다(13장). HD-BET은 설치 전이라 비어 있다.
- 공개 감사 기록에는 Host 절대경로를 남기지 않는다(13.3).

## 6. 자동 QA와 판정

| 구분 | 항목 |
|---|---|
| 기하 | TRACE와 shape 일치, affine 차이 ≤ 1e-3 mm, orientation 일치, spacing 일치, qform·sform 코드 |
| 값 | 정확히 {0, 1}, NaN·Inf 없음, 빈 Mask 아님, 전체 1 아님 |
| 구조 | Voxel 수, 부피 mL, 영상 대비 비율, 26-연결 component 수, 최대 component 비율, 영상 경계 접촉 비율, Bounding Box, 축별 포함 slice 수 |
| 병변 보존 | Acute GT 중 Mask 밖 Voxel 수·비율, GT component별 Mask 밖 Voxel |
| 기준선 비교 | v1 배경 Mask 대비 Dice, 부피 비율, 축별 포함 slice 손실 |

| 상태 | 조건 |
|---|---|
| `fail` | GT preservation failure, shape·affine·orientation·spacing 불일치, binary 아님, 빈 Mask, 전체 1 Mask, 실행 실패, Mask 없음 |
| `review_required` | Mask 비율 이상, component 조각화, 최대 component 비율 낮음, 경계 접촉 과다, 기준선 대비 slice 손실·부피 차이, qform/sform 코드 차이 |
| `pass` | 위 어느 것에도 해당하지 않음 |

- 검토 기준의 **수치 임계값은 아직 고정하지 않는다**(`ReviewThresholds` 기본값은 미평가). 파일럿 분포와
  육안 검토 후 고정하며, 고정 전에는 해당 규칙을 "not evaluated"로 기록한다.
- qform·sform **코드** 차이는 Voxel 위치를 바꾸지 않으므로 실패가 아닌 검토 대상이다.

## 7. 육안 QA

- Panel A: Axial 중간 slice + Mask 윤곽선
- Panel B: GT 최대 slice + GT 채움 + Mask 윤곽선
- Panel C: Axial / Coronal / Sagittal Mask 윤곽선 (GT 중심 기준)
- Panel D: 같은 slice에서 도구별 Mask 비교
- 색상: v1 배경 회색 점선, SynthStrip 파랑(#0072B2), HD-BET 주황(#E69F00), GT 자홍 반투명. 범례 항상 표시.
- 제목에는 Alias와 범주만 쓴다. SOOP와 ISLES는 같은 레이아웃을 쓴다.
- 제목·라벨이 다른 그림과 겹치는지 자동으로 검사한다(Panel C 포함).
- 사용자가 사례·도구별로 뇌 누락, 두피·안구 잔존, 병변 절단, 방향 이상을 기록한다.

## 8. 파일럿 구성

- **SOOP**: 기존 파일럿 5건만 사용한다(추가 다운로드 없음). 5건은 이미 모든 성능 평가와 v2 Split에서 영구
  제외되어 있으므로 Brain Mask 도구 개발·QA에만 사용한다.
- **ISLES**: **Train만** 사용한다(Validation·Test 금지, Train이 아닌 행이 들어오면 선정 자체를 거부).
  - 범주: 작은 병변(0 < 부피 < 1 mL), 큰 병변(≥ 50 mL), thick slice(> 3 mm), 높은 기울기(≥ 10°).
  - 결정적 선정: `SHA-256("neuroscan-bet-isles-v1:" + case_id)` 순서로, 범주 순서(thick → oblique →
    small → large)대로 아직 2건을 채우지 못한 범주를 가장 많이 채우는 사례를 고르며 최대 6건.
  - Alias `isles-bet-01..`, 대응표는 로컬에만 둔다.
  - ISLES는 이미 Skull-stripped이므로, 새 도구가 기존 뇌 영역을 지나치게 줄이거나 넓히지 않는지 확인하는
    대조군이다.

## 9. 도구 선택 규칙

순서: 1 GT 병변 보존 → 2 뇌 조직 누락 방지 → 3 두피·안구·외부 조직 제거 → 4 DWI·ADC 격자 보존 →
5 SOOP·ISLES 일관성 → 6 자동 처리 안정성 → 7 오프라인 실행 → 8 설치·운영 복잡도 → 9 실행 시간·GPU →
10 라이선스.

- 한 사례라도 `fail`(GT 보존 실패 포함)인 도구는 적격이 아니다. 모든 사례의 육안 QA 승인이 필요하다.
- **어느 도구도 통과하지 못하면 Brain Mask 정책을 확정하지 않는다.** Mask 팽창 같은 사후 보정은 별도 승인
  없이는 하지 않는다.
- 선택 후 도구 버전, container digest 또는 패키지 버전, 가중치 hash, 실행 인자를 정책 이름으로 고정한다.
- 선택하지 않은 도구의 결과도 삭제하지 않고 로컬에 보존한다.

## 10. v1 동결 영역

- v1 코드·설정·체크포인트·Final Test manifest·Lock Tag는 변경하지 않는다.
- v1 배경 Mask는 기준선으로만 계산하며(읽기 전용 import), v1 정책을 바꾸지 않는다.
- 새 Brain Mask 정책은 v2 정책으로만 정의한다.

## 11. 별도 승인이 필요한 단계

- 실제 도구 설치(WSL2 기반 별도 환경 검토), Docker image pull, 가중치 다운로드와 hash 기록
- 실제 SOOP·ISLES 영상 처리와 Brain Mask 생성
- 검토 임계값 고정, 도구 최종 선택과 정책 고정
- FSL BET 예비 검토
- 현재 NeuroScan Python 환경은 변경하지 않는다.

## 12. 저장 위치

| 내용 | 위치 | Git |
|---|---|---|
| 실행 결과·QA·그림 | `results/brain_extraction_pilot/` | 제외 |
| SOOP Brain Mask | `data/interim/soop_brain_masks/` | 제외 |
| ISLES Brain Mask | `data/interim/isles_brain_masks/` | 제외 |
| Runner·QA 코드, 합성 테스트, 이 문서 | 저장소 | 포함 |

Brain Mask NIfTI, MRI, GT, QA PNG, 환자별 JSON·CSV, Alias 대응표, 가중치, 환자 단위 실행 로그는 Git에서
제외한다. 공개 출력에 원본 ID가 섞이면 결과를 쓰지 않고 거부한다.

## 13. SynthStrip 1.8 CPU 실행 정책과 합성 Smoke Test

### 13.1 고정 정책 `bm_synthstrip_1_8_cpu_pilot`

| 항목 | 값 |
|---|---|
| Image | `freesurfer/synthstrip@sha256:ebbc177221194371f16362513ace68312a22922bb581bdfa618ac7ff9c1d2c06` (CPU, linux/amd64) |
| Model | `synthstrip.1.pt`, SHA-256 `37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33` (이미지 내장) |
| 장치 / Thread | CPU / 4 |
| 경계 | 1 mm (`-b 1`) |
| CSF | 포함 (`--no-csf` 미사용) |
| Network | none |

- 값은 코드(`SynthStripPolicy`, frozen)에 고정되어 있으며, 다른 값으로 만든 정책·명령은 실행기가 거부한다.
  사례별 변경은 불가능하다.
- 실행 명령: `docker run --rm --pull never --network none --name <임의 이름> -v <WORKDIR>:/data <digest 고정 Image>
  -i /data/tool_input.nii.gz -m /data/tool_mask.nii.gz -b 1 -t 4`
- 실행기는 이미지를 받지 않는다(`--pull never`, 로컬 이미지 digest 확인). Docker Engine 상태 확인, 시간 제한
  (초과 시 컨테이너 강제 삭제), 종료 코드·stdout·stderr(길이 제한)·실행 시간 기록, 출력 Mask 확인, 실행 후 남은
  컨테이너 확인을 수행한다.

### 13.2 허용·금지 실행 조건

- Mount는 Runner의 격리 작업 폴더(`<alias>/<tool>.part`) 하나만 허용한다. 이 폴더는 승인된 출력 경로
  (`results/brain_extraction_pilot/`, `data/interim/soop_brain_masks/`, `data/interim/isles_brain_masks/`) 아래에
  있어야 하고, 실행 시점에 도구 입력 파일 하나만 들어 있어야 한다.
- 금지: Tag만으로 실행, 다른 Digest, GPU 옵션, privileged, host network, 추가 Mount, 저장소 전체·`data/raw`
  Mount, Docker socket Mount, 환경변수 전달, device·capability 추가, 셸 실행(`shell=True`), 셸 메타문자를 포함한
  인자.

### 13.3 감사 기록의 경로 정제

- **내부 실행 기록**(Host 경로 포함)은 로컬 전용이며 승인된 출력 경로 아래 `_internal/`에만 저장한다(Git 제외).
- **공개 감사 기록**(`audit.json`, `qa.json`, `run.log`, `figure_info.json`)에는 Host 작업 경로를 `<WORKDIR>`로,
  그 밖의 Host 경로를 `[REDACTED]`로 바꾸고 컨테이너 경로(`/data/...`)만 남긴다. 드라이브 문자 경로, 사용자 홈
  경로, UNC 경로, 로컬 계정명, 원본 Subject ID가 남아 있으면 기록을 쓰지 않고 거부한다.

### 13.4 합성 Smoke Test 결과

- 합성 3D NIfTI 1건(배경 0, 뇌·두피 타원체, 합성 GT, 약 5.7° 기울어진 affine, 1.5×1.5×3.0 mm, qform·sform)으로
  수행했다. **실제 MRI 접근은 0건**이다.
- 첫 실행과 정식 실행기 재실행 모두 종료 코드 0, `--network none`(컨테이너 안에는 loopback만 존재),
  shape·affine(차이 0)·orientation·spacing·qform/sform 일치, 값 {0, 1}, component 1개, 경계 접촉 0,
  합성 GT 보존 100%(밖 voxel 0개)였다. 두 실행의 Mask는 비트 단위로 동일했다.
- 합성 팬텀에서 두피·두개골은 Mask에 포함되지 않았고, **합성 뇌 영역의 82.1%만 Mask에 포함**되었다.
- **합성 팬텀은 실제 해부 구조가 아니므로 이 결과는 실제 영상에서의 성능 근거가 아니다.** 82.1%라는 값은
  뇌 조직 누락 가능성을 실제 MRI의 기준선 비교와 육안 QA로 반드시 확인해야 한다는 기술적 관찰로만 사용한다.

### 13.5 다음 단계

실제 SOOP 사례 실행(파일럿 1건부터)은 별도 승인 대상이다. HD-BET 설치와 GPU 이미지는 이 정책에 포함되지 않는다.
