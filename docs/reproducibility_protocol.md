# 실행 재현성 프로토콜 (NeuroScan v2)

> 상태: Manifest 기록·비교 도구만 구현. 모델 실행 코드에는 아직 연결하지 않았다.
> 코드: `ai/neuroscan_ai/reproducibility/` (`hashing`·`environment`·`manifest`·`validation`),
> CLI: `scripts/capture_run_manifest.py`, 테스트: `tests/test_reproducibility_manifest.py`(합성만).

## 1. 왜 필요한가 (v1에서 얻은 교훈)

동결된 Checkpoint·Config·Split·입력이 모두 그대로였는데도, 나중에 같은 모델을 다시 실행한 결과가
예전에 저장된 Validation 예측과 Voxel 단위로 일치하지 않은 경험이 있다. 현재 환경에서는 반복
실행이 결정적이었으므로, 차이는 당시 실행 상태에 있었지만 원인을 확정할 수 없었다.

- 당시 실행 코드가 실행 **후에** 처음 커밋되어, 실행에 쓰인 코드 버전을 특정할 수 없었다.
- 실행 기록에 Git Commit, PyTorch·CUDA·cuDNN·GPU Driver 버전, 결정론 설정이 없었다.
- Probability Map을 저장하지 않아, 나중에 Confidence·Ungated 분석을 하려면 모델을 다시 실행해야
  했고, 그 재실행은 원래 결과와 같다고 보장할 수 없었다.
- Gating 전·후 결과와 Brain Mask를 저장하지 않아, Mask 밖 예측을 사후에 볼 수 없었다.

결론: 재현성은 실행 **전에** 기록해야 하며, 나중에 쓸 수 있는 중간 산출물은 실행할 때 함께 저장한다.

## 2. v2 실행의 필수 규칙

1. 실행 전 Working Tree가 깨끗하고 실행 코드가 커밋되어 있어야 한다. Dirty 상태 실행은
   `exact_match`가 될 수 없다.
2. 실행 전 **Pre-run Manifest**, 실행 후 **Post-run Manifest**를 남긴다.
3. 다음 산출물을 실행과 동시에 저장한다.
   - Ungated Probability Map (Gating·Threshold 전, Clip·재정규화 없음)
   - Ungated Binary Prediction, Gated Binary Prediction
   - 실제로 사용한 Brain Mask
   - 사례별 Metric, 집계 결과
4. 모든 산출물의 SHA-256을 Post-run Manifest에 기록한다.
5. Final Test 접근 여부와 Split별 접근 사례 수를 기록한다. 허용되지 않은 Split 접근이 계획에
   있으면 Pre-run Manifest 생성 단계에서 거부한다.

## 3. Manifest 내용

| 영역 | 항목 |
|---|---|
| Identity | run_id, stage, created_at_utc, purpose, dataset_id, split_name, case_count |
| Source control | repository(자격증명 제거), branch, git_commit, working_tree_clean, tracked_diff_present, untracked_files_present, lock_tag, lock_tag_target |
| Configuration | config·참조 config·split의 정제 경로, 원본 SHA-256, LF 정규화 SHA-256, 정책 ID(전처리·추론·Threshold·Brain Mask·결합) |
| Model | architecture ID, checkpoint 정제 경로·SHA-256·metadata, 입력 채널 정책, 출력 정의, sigmoid/softmax, 고정 Threshold |
| Environment | OS, Python, PyTorch, CUDA runtime, cuDNN, GPU 이름, GPU Driver, MONAI·NumPy·NiBabel·SciPy, 패키지 lock 파일 LF SHA-256 |
| Determinism | seed, deterministic algorithms(+warn_only), cudnn deterministic·benchmark, TF32 matmul·convolution, precision, AMP, worker 수, batch size |
| Inputs | 정렬 alias 목록 Hash, 사례 수, Case Key 정책, 입력 Manifest Hash, Case Key별 입력 파일 Hash(DWI·ADC·GT·Brain Mask) |
| Outputs (post-run) | Case Key별 Probability·Ungated·Gated·Brain Mask·사례 Metric Hash, 일반 이름의 집계 결과 Hash |
| Access | Split별 사례 수, Final Test 접근, 허용 Split, 비허용 Split 접근 여부 |

모르는 값은 추측하지 않고 `null`로 남긴다(예: CUDA가 없거나 Driver를 읽을 수 없는 환경).

## 4. Schema, Hash와 정규화

- Schema 이름은 `neuroscan-run-manifest-v1`로 고정한다. 생성·쓰기·연결 검사·비교 모두 이 값을
  확인하며, Schema가 없거나 다르면(Pre-run과 Post-run 사이 불일치 포함) 거부한다.
- 바이너리(Checkpoint, NIfTI): 원본 바이트 SHA-256.
- 텍스트(Config, Split, Package lock): 원본 SHA-256과 **CRLF→LF 정규화 SHA-256**을 함께 기록한다
  (lock 파일은 정규화 Hash만). **Config·Split·Package lock 비교에는 정규화 Hash를 쓴다.** 줄바꿈만
  다르고 내용이 같으면 같은 Hash이고, 의존성 내용이 다르면 다른 Hash다.
- Pre-run Manifest는 정렬된 키의 Canonical JSON SHA-256으로 Post-run Manifest에 연결된다.
- 사례 alias는 평문으로 기록하지 않는다. Case Key 정책 `sorted_alias_position_v1`에 따라:
  - `alias_list_sha256` = SHA-256(정책 이름 + 정렬된 alias 목록). 같은 alias 집합이라도 정렬 규칙이
    다르면 Hash가 달라진다.
  - 사례별 입력·출력 Hash는 정렬 순서의 위치 Key `case_0001`, `case_0002`, ... 아래에 둔다.
  - 두 Manifest의 Case Key는 `alias_list_sha256`와 정책이 같을 때만 서로 비교할 수 있다.
  - 기록되는 경로에 alias가 들어 있으면 해당 Case Key로 바꾼다.

## 5. 비교 수준

| 수준 | 의미 |
|---|---|
| `exact_match` | 비교한 모든 항목이 있고 같으며, 두 Working Tree가 모두 깨끗함 |
| `compatible_but_not_exact` | 결과 정의 항목은 같지만 Git Commit·PyTorch·CUDA·cuDNN·Driver·GPU·Package lock·결정론 설정이 다르거나 Working Tree가 깨끗하지 않음 |
| `unavailable` | 차이는 없지만 필요한 항목이 한쪽 이상에서 없거나 `null`임 |
| `mismatch` | 결과 정의 항목(Checkpoint·Config·Split·입력 alias 목록·입력·출력 Hash)이 다름 |

Git Commit, Checkpoint, Config, Split, 입력, PyTorch, CUDA, cuDNN, Driver, GPU, Package lock, 결정론
설정, 출력 Hash 중 하나라도 다르면 `exact_match`가 될 수 없다. Package lock 차이는 환경 차이로
분류하며(결과 Hash 불일치로 보지 않음), 한쪽이라도 `null`이면 `unavailable`이다. 재현 주장은
`exact_match`일 때만 한다.
CLI: `capture`(현재 상태 기록), `validate`(저장 Manifest와 현재 상태 비교), `compare`(두 Manifest 비교).

### 단일 Manifest 비교와 Run Bundle 비교

- `compare_manifests()`는 같은 종류의 Manifest 두 개의 구조 비교다. Post-run Manifest에는 Source Control·
  Config가 없으므로 Post-run 단독 비교로는 실행 재현성을 주장하지 않는다.
- 독립 실행 두 건의 **공식 재현성 판정은 Run Bundle 비교**(`run_control.bundle.compare_run_bundles`)를
  쓴다. 각 Run을 Pre-run + Post-run 한 묶음으로 보고, 먼저 무결성(Schema, Pre/Post 연결, Seal, 기록된
  모든 출력 Hash와 실제 파일)을 확인한 뒤 결과 정의 항목과 환경 항목을 비교한다.
- **감사 필드와 재현성 필드는 분리한다.** run_id, 생성·시작·종료 시각, Pre-run Manifest Hash 자체,
  Seal 값, 출력 폴더명은 독립 실행마다 달라야 정상이므로 재현성 판정에서 제외한다(Manifest에서는
  삭제하지 않고 감사기록으로 보존). 따라서 run_id와 시간이 달라도 `exact_match`가 될 수 있다.
- 필드 상태: `value`(값 있음), `unavailable`(필요하지만 수집되지 않음), `not_applicable`(그 단계에
  적용되지 않음). Training Run의 입력 Checkpoint는 양쪽 모두 없으면 `not_applicable`로 일치 처리하고,
  한쪽만 없으면 `mismatch`다. Validation·Inference·External validation의 입력 Checkpoint는 필수이며
  없으면 `exact_match`가 될 수 없다.
- 입력 파일 Hash(`inputs.file_sha256`)는 사례가 있는 실행에서 실제로 기록돼 있어야 한다. 비어 있으면
  `unavailable`이다(비교할 근거가 없음).

## 6. 원본 ID와 경로 보호

- 공개 Manifest에 허용: 토큰 경로(`<WORKDIR>`, `<CHECKPOINT>`, `<DATASET>` + 상대경로), Hash, 버전,
  정책 ID, 위치 Case Key(`case_0001` 등).
- 거부: alias 평문, 원본 Subject ID처럼 보이는 문자열(원본 ID 형태 alias 포함), Host 절대경로,
  계정 이름, 자격증명·토큰·개인키 패턴, 비밀성 환경변수 값. 선언한 Root 밖의 경로는 기록을 거부한다.
- Alias ↔ Case Key 대응표가 필요하면 Git 제외 경로 `results/reproducibility/_internal/` 아래에만
  쓴다. 공개 Manifest와 공개 로그에는 넣지 않는다.
- Manifest는 Git 제외 경로(`results/reproducibility/`)에만 쓰고 덮어쓰지 않는다.

## 7. Final Test

Final Test 실행은 동결 절차를 따르며, 그 실행의 Pre-run·Post-run Manifest에 Final Test 접근과
Lock Tag 대상 Commit을 기록한다. Final Test 결과로 모델·Threshold·후처리를 바꾸지 않는다.

## 8. 아직 없는 것

이 도구는 모델 실행 코드에 연결되지 않았다. v2 학습·추론 실행기가 Pre-run Manifest 생성 →
실행 → 산출물 저장 → Post-run Manifest 생성을 강제하도록 연결하는 것은 별도 승인 단계다.
