# v2 실행 통제 프로토콜 (Run Control)

> 상태: 실행 흐름과 Gate만 구현. Runner는 합성 배열을 쓰는 **Mock Runner뿐**이며 실제 모델·
> Checkpoint·MRI를 다루지 않는다. 실제 v2 학습·추론 Runner 연결은 별도 승인 대상이다.
> 코드: `ai/neuroscan_ai/run_control/` (`plan`·`gate`·`runner`·`outputs`·`lifecycle`),
> CLI: `scripts/run_v2_controlled.py`(`--mock` 전용), 테스트: `tests/test_v2_run_control.py`(합성만).
> Manifest 규칙은 `docs/reproducibility_protocol.md`(Schema `neuroscan-run-manifest-v1`)를 따른다.

## 1. 원칙

- **Manifest 없는 실행 금지.** Pre-run 승인이 Runner보다 먼저다. Runner는 승인된 Gate 결과를 받아야
  하고, 디스크의 Pre-run Manifest가 승인 이후 바뀌지 않았는지(Canonical JSON Hash) 확인한 뒤에만
  시작한다. Gate가 없거나 거부됐거나 Manifest가 바뀌었으면 Runner는 아무것도 쓰지 않고 거부한다.
- v1 Runner·Config·실행 경로와 완전히 분리한다. 이 패키지는 v1 학습·추론·전처리 코드, PyTorch,
  MONAI를 import하지 않는다.

## 2. 실행 순서

1. RunPlan 검증 (생성 시 거부 규칙 적용)
2. Pre-run Manifest 생성
3. Pre-run Manifest Hash 고정 (`<run_dir>/pre_run_manifest.json`)
4. Runner 실행
5. 필수 산출물 존재 확인
6. 산출물 SHA-256 기록
7. Post-run Manifest 생성 (`<run_dir>/post_run_manifest.json`)
8. Pre/Post 연결 검증
9. 상태 확정: `completed`, `failed_incomplete_outputs`, `failed_output_validation`,
   `failed_runner_error`, `failed_manifest`

## 3. RunPlan과 Split 접근 통제

RunPlan: run_id, purpose, stage(`train`·`validation`·`inference`·`external_validation`), dataset_id,
split_name, 사례 alias 목록, allowed_case_count, model_id, checkpoint, config, split file, 정책 ID 5종
(전처리·추론·Threshold·Brain Mask·결합), 기대 Hash, 출력 폴더.

- Train은 `train`만, Validation은 `val`만 접근한다.
- Inference·External validation은 명시 승인된 Split만 접근한다.
- `test`와 `final_test`는 기본 거부한다(승인 목록에 있어도 거부). Final Test는 별도 동결 절차를 따른다.
- SOOP 파일럿 사례(`soop-pilot-NN`)와 명시 제외 사례는 모든 학습·성능 실행에서 거부한다.
- 사례가 선언한 Split에 없거나 다른 Split에도 속하면 거부한다. 사례 수가 allowed_case_count와
  다르면 거부한다.

## 4. Pre-run Gate

다음이 모두 통과해야 `pre_run_status = "approved"`이다. 하나라도 실패하면 `rejected`이고 Runner를
호출하지 않는다.

- Git Commit 기록, Working Tree clean
- 기대 Hash 선언과 일치: Config·Split·Package lock(LF 정규화), Checkpoint, Alias 목록
- 사례 수
- 기대 환경 Version·결정론 설정(선언한 경우)과 일치
- 출력 경로 Git 제외, 같은 run_id 폴더 없음
- Pre-run Manifest 생성 시 개인정보·경로 검사 통과

거부도 감사기록이다. 출력 폴더가 아직 없으면 새 run 폴더에 `gate_rejection.json`(사유 코드만)을 남긴다.

## 5. 필수 산출물

Validation·Inference: 사례별 Probability Map, Ungated Binary, Brain Mask, Gated Binary, 사례 Metric
JSON, 실행 단위 Aggregate JSON, Post-run Manifest. **Probability·Ungated·Gated·Brain Mask는 같은
실행에서 함께 저장한다.** 하나라도 없으면 `failed_incomplete_outputs`.

Train(아직 미구현): Checkpoint, 학습 History, Validation Metric, 선택 Threshold 기록, Post-run Manifest.

## 6. 출력 관계 검사

사례마다 Shape·Affine 일치, Probability 유한값·[0, 1], Brain Mask·Prediction Binary,
`ungated == (probability >= threshold)`, `gated == ungated AND brain_mask`, `gated ≤ ungated`.
검사는 읽기만 하며, 값을 고쳐서 통과시키지 않는다. 위반은 `failed_output_validation`.

## 7. 실패 기록과 재실행

- 실패해도 Post-run Manifest를 쓴다: 상태, 종료 코드, 실패 단계, 오류 유형, 정제된 오류 메시지,
  성공·실패 사례 수, 생성·누락 산출물, Pre-run Manifest Hash, 실제 접근 Split.
- 오류 메시지는 절대경로·사용자명·원본 ID·alias·자격증명 검사에 걸리면 `[redacted]`로 기록한다.
- 같은 run_id의 출력이 있으면 거부한다. 재실행은 새 run_id로 한다. 실패한 Run 폴더를 지우고 같은
  ID를 다시 쓰지 않으며, Manifest·Output Hash를 수정하지 않는다. 실패한 Run도 보존한다.

## 8. 결정론과 사후 검증

- Runner가 실제로 쓰는 Seed를 Pre-run·Post-run Manifest의 `determinism.seed`에 기록한다. 호출자가
  선언한 Seed가 Runner Seed와 다르면 Gate 전에 실행을 거부한다.
- Mock Runner는 사례마다 명시적 RNG(`default_rng(seed + 사례 위치)`)를 쓴다. 같은 RunPlan·같은
  Seed·같은 합성 입력이면 Probability와 모든 출력 Hash가 정확히 같고, Seed가 다르면 달라진다.
- Aggregate 결과는 실행마다 하나(`aggregate_results/aggregate.json`)만 만든다.
- 실행이 끝나면 Post-run Manifest의 Canonical JSON SHA-256을 `post_run_manifest.sha256`에 기록한다.
- `verify_run(run_dir)`은 다음 중 하나라도 어긋나면 실패한다: Pre/Post 연결, Post-run Seal,
  Pre/Post의 identity·access·alias 목록 Hash 일치, 기록된 모든 출력 Hash와 디스크 파일. 실행 후
  Manifest나 필수 출력을 고치면 검증에 실패한다. 실패한 Run도 같은 방법으로 감사기록을 검증한다.

## 9. 원본 ID와 경로 보호

출력 파일과 Manifest는 alias가 아니라 위치 Case Key(`case_0001` 등)를 쓴다. Manifest에는 alias 평문,
원본 ID, Host 절대경로, 사용자명, 자격증명을 넣지 않는다. 출력은 Git 제외 경로
`results/v2_runs/<run_id>/`에만 쓴다.

## 10. Mock Runner

합성 배열(무작위 Probability, 타원체 Brain Mask)로 위 흐름을 검증하는 테스트 전용 Runner다. 결과는
임상·성능 결과가 아니다. 실제 v2 Runner를 연결하려면 같은 Gate·산출물·관계 검사·Post-run 규칙을
통과해야 하며, 연결 자체가 별도 승인 대상이다.
