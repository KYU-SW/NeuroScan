# NeuroScan-ai

**2.5D 딥러닝 기반 허혈성 뇌졸중 병변 분할 및 3D 진단 보조 시스템** (연구용 프로토타입)

> ⚠️ **연구 및 교육 목적의 프로토타입이며 진단도구가 아닙니다.**
>
> - 임상 진단 또는 치료 결정에 사용해서는 안 됩니다. 확정 진단, 치료 추천, 예후 예측을 하지 않습니다.
> - 임상 사용 승인을 받지 않았습니다.
> - 모든 결과는 영상 검토를 보조하는 참고 정보이며, **최종 판단은 반드시 의료진이 해야 합니다.**
> - 아래 성능 수치는 모두 **Validation 결과**입니다. Test Split은 봉인되어 있으며, 최종 내부 Test 성능과
>   외부 데이터셋 검증은 아직 측정하지 않았습니다 ([§3](#3-성능-현황-validation-결과)).

## 1. 목적

ISLES 2022 MRI(DWI + ADC)에서 허혈성 병변을 분할하고, 2D U-Net Baseline과 인접 Slice를 활용한
2.5D U-Net을 환자 단위로 비교합니다. Slice별 예측은 원래 MRI의 Z 위치에 다시 적층하여 환자 단위
3D 병변 Volume으로 재구성하고, 병변 위치·범위·부피를 정량화하여 Web 2D/3D Viewer에서 검토할 수
있게 합니다.

```
MRI (DWI/ADC) → 전처리 → 2D / 2.5D 입력 → MONAI U-Net → Slice Prediction
  → 3D Volume 재구성 → GT 비교 (Patient-wise Dice 등) → 정량 분석 → Web Viewer
```

| 항목 | 결정 |
|---|---|
| 데이터셋 | ISLES 2022 (CC BY 4.0, Git에 포함하지 않음) |
| 입력 | DWI + ADC (FLAIR는 MVP 이후 추가 실험) |
| Baseline | DWI(z) + ADC(z), 2채널 2D U-Net |
| Proposed | DWI(z-1, z, z+1) + ADC(z-1, z, z+1), 6채널 2.5D U-Net |
| 모델 구현 | MONAI `UNet` (외부 U-Net 구현 코드를 복사하지 않음) |
| 평가 | Primary: Patient-wise Dice / Secondary: IoU, Sensitivity, Precision, Lesion Volume Error |
| 연구 목표 | 독립 Test 환자 평균 Dice ≥ 0.70 (프로그램의 합격/불합격 조건이 아닌 연구 목표) |
| 분할 원칙 | 환자 단위 Train / Validation / Test |
| 후처리 | 초기 Prediction은 작은 Connected Component를 지우지 않고 원본 그대로 보존 |
| 전처리 v1 | RAS, XY 1.0 mm / Z 원본 유지 (2.5D 인접 Slice = 실제 촬영 Slice), 강도 정규화는 입력 파이프라인에서 |
| Split v1 | seed 42, 176 / 37 / 37, Negative 3명 분산, thin/thick × 부피 사분위 층화 |
| Viewer | React + TypeScript + NiiVue (MVP, Phase 6). vtk.js는 고급 3D 확장, Cornerstone3D는 DICOM 연동 시 검토 |
| Backend/DB | FastAPI + **SQLite**(WAL, 현재) + 독립 단일 GPU worker. MySQL은 향후 운영 확장 대상. DB에는 메타데이터와 상대 경로만 저장 |
| MVP 제외 | nnU-Net 비교, FLAIR |

## 2. 현재 제공 기능

- DWI + ADC NIfTI 업로드 (localhost Web UI)
- 2.5D U-Net 추론 (Phase 4 모델, threshold 0.5)
- Slice Prediction을 환자 단위 3D Volume으로 재구성하고 원본 영상 공간(Original Space)으로 복원
- 병변 부피와 Connected Component(연결 영역) 분석
- Axial / Coronal / Sagittal / MPR / 3D Viewer (React + NiiVue)
- Component 선택·강조·숨기기
- Oblique(비스듬히 촬영된) 영상에서도 voxel 기반으로 정확한 위치 이동
- Research Mode (기본 비활성): Validation case의 Ground Truth 비교, TP / FP / FN 차이 지도,
  Patient-level Dice·IoU·Sensitivity·Precision, Component Matching

## 3. 성능 현황 (Validation 결과)

| 모델 | Validation positive-patient mean Dice |
|---|---|
| 2D U-Net Baseline | 약 **0.731** |
| 2.5D U-Net (현재 추론 모델) | 약 **0.743** |

- 2.5D와 2D의 차이(ΔDice +0.012, 95% CI [−0.002, +0.028], Wilcoxon p = 0.29)는 **통계적으로 유의하지
  않았습니다.** 상세: [docs/phase4_25d.md](docs/phase4_25d.md)
- Validation 데이터는 checkpoint 선택에 사용되었으므로 결과가 낙관적일 수 있습니다.
- **Test Split은 아직 봉인 상태**이며, 최종 내부 Test 성능은 아직 측정하지 않았습니다.
- **외부 데이터셋 검증은 아직 수행하지 않았습니다.**
- Validation 성능을 실제 임상 성능으로 해석해서는 안 됩니다.

## 4. 디렉터리 구조

```
NeuroScan-ai/
├── ai/neuroscan_ai/   # Python 패키지 (Web과 독립)
│   ├── datasets/      #   데이터셋별 파일 탐색 어댑터 (isles2022.py; 외부 데이터셋 추가 지점)
│   ├── io/            #   NIfTI 로드/저장, geometry
│   ├── qa/            #   범용 Dataset QA
│   ├── preprocessing/ #   mask 이진화, 방향, spacing, case 파이프라인
│   ├── data/          #   환자 단위 Split, subgroup metadata
│   ├── reconstruction/#   Slice→Volume 재구성, processed→original 복원
│   ├── evaluation/    #   환자 단위 metric, Negative 정책, report
│   ├── analysis/      #   병변 정량, connected component
│   ├── inference/     #   독립 단일 환자 추론 service (Phase 5A; Web·DB 비의존)
│   └── research/      #   Validation 연구 비교: TP/FP/FN, 지표, Component Matching (Phase 7-B)
├── configs/           # 실험 설정 (YAML)
├── scripts/           # CLI 진입점 (check_env.py 등)
├── tests/             # 단위 테스트
├── docs/              # 설계 문서·실험 기록
├── backend/           # FastAPI 백엔드 (neuroscan_api, Phase 5B)
├── frontend/          # React + TypeScript + NiiVue Viewer (Phase 6)
├── data/              # 로컬 전용: raw/ processed/ splits/   (Git 제외)
├── models/            # 학습된 checkpoint                    (Git 제외)
├── results/           # 연구 평가 결과·예측 NIfTI            (Git 제외)
├── outputs/           # 독립 추론 결과 (Phase 5A)            (Git 제외)
└── storage/           # FastAPI 업로드·결과·SQLite (Phase 5B) (Git 제외)
```

## 5. 실행 환경

| 요구 사항 | 버전 |
|---|---|
| Python | **>=3.11,<3.12** (`pyproject.toml`) |
| Node.js | **>=24.15.0** (`frontend/package.json`) |
| DB | SQLite (Backend 현재 사용, 별도 설치 불필요). MySQL은 향후 운영 확장 대상 |
| CUDA / PyTorch | 아래 검증 환경 참고 (CPU 추론도 가능하나 느림) |

현재 검증한 환경 (Phase 0 검증 완료, 2026-10-17):

| 구성 | 버전 |
|---|---|
| OS | Windows 11 Pro |
| GPU | NVIDIA GeForce RTX 5060 Ti 16 GB (Blackwell, sm_120), Driver 617.42 |
| Python | 3.11.9 (프로젝트 전용 `.venv`) |
| PyTorch | 2.14.1+cu130 (CUDA runtime 13.0, cuDNN 9.24) |
| MONAI | 1.6.1 |
| NiBabel / SimpleITK | 5.4.2 / 2.5.6 |
| NumPy / SciPy | 2.4.6 / 1.17.1 |

PyTorch 2.14의 공식 Stable CUDA 빌드는 12.6 / 13.0 / 13.2입니다. CUDA 12.6 빌드는 Blackwell(sm_120)을
지원하지 않으므로 **CUDA 13.0 빌드**를 사용합니다. GPU 드라이버는 CUDA 13.x를 지원해야 합니다.

## 6. 환경 구성

PowerShell 기준:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
# 1) PyTorch: 공식 CUDA 13.0 인덱스에서 설치
.venv\Scripts\python.exe -m pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
# 2) 프로젝트와 나머지 의존성 (editable)
.venv\Scripts\python.exe -m pip install -e ".[dev]"
# 3) 환경 검증
.venv\Scripts\python.exe scripts\check_env.py
```

검증된 정확한 버전으로 재현하려면 다음을 사용합니다.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

`scripts/check_env.py`는 Python 버전, CUDA 사용 가능 여부와 GPU 아키텍처 지원, GPU 연산(matmul, AMP conv),
MONAI 2D UNet의 GPU forward(2채널·6채널), NiBabel NIfTI 왕복, NiBabel↔SimpleITK 좌표계(RAS↔LPS) 일치를
확인합니다.

## 7. API · Worker · Frontend 실행 (localhost)

PowerShell 창 3개에서 각각 실행합니다. API와 worker는 `configs/api_v1.yaml`을 사용하며, 학습된
checkpoint가 `models/` 아래에 있어야 추론할 수 있습니다 (checkpoint는 저장소에 포함되지 않습니다).

```powershell
# 1) API
cd C:\Projects\NeuroScan-ai
$env:PYTHONPATH = "backend"
.venv\Scripts\python.exe -m neuroscan_api
```

```powershell
# 2) Worker (단일 GPU 추론 작업 처리)
cd C:\Projects\NeuroScan-ai
$env:PYTHONPATH = "backend"
.venv\Scripts\python.exe -m neuroscan_api.worker
```

```powershell
# 3) Frontend (Vite dev server, http://127.0.0.1:5173 — /api 요청은 API로 proxy)
cd C:\Projects\NeuroScan-ai\frontend
npm ci
npm run dev
```

경로 `C:\Projects\NeuroScan-ai`는 예시이며 저장소를 받은 위치로 바꿔 사용합니다. 상세:
[docs/phase5b_fastapi_backend.md](docs/phase5b_fastapi_backend.md),
[docs/phase6_viewer.md](docs/phase6_viewer.md).

## 8. 데이터 · 저장소 정책과 데이터 라이선스

다음은 저장소(Git)에 **포함되지 않습니다** (`.gitignore`):

- ISLES 2022 원본 데이터 (`data/raw/`)와 processed 데이터 (`data/processed/`)
- 모델 Checkpoint (`models/`, `*.pt`, `*.pth`)
- Prediction NIfTI와 Ground Truth NIfTI (`*.nii`, `*.nii.gz`)
- `results/`, `outputs/`
- `storage/` (업로드·추론 결과·연구 case 산출물)와 SQLite DB 및 그 백업
- `.env`, `.env.*` 및 모든 인증정보

저장소에는 소스 코드, 설정, 테스트, 문서와 환자 단위 분할 정의(`data/splits/`, 공개 데이터의 익명
case ID와 요약 통계만 포함)만 있습니다.

- ISLES 2022 데이터 라이선스: **CC BY 4.0** (데이터의 `dataset_description.json`, `LICENSE` 기준).
  사용 시 아래 논문을 인용해야 합니다.
- 데이터 인용: Hernandez Petzsche, M.R., de la Rosa, E., Hanning, U. et al. ISLES 2022: A multi-center
  magnetic resonance imaging stroke lesion segmentation dataset. *Scientific Data* 9, 762 (2022).
  DOI: [10.1038/s41597-022-01875-5](https://doi.org/10.1038/s41597-022-01875-5)
- 외부 프로젝트 코드는 복사하지 않습니다. 라이브러리(PyTorch, MONAI, NiBabel, SimpleITK 등)는 각
  라이선스에 따라 의존성으로만 사용합니다.

## 9. 개발 단계

| Phase | 내용 | 상태 |
|---|---|---|
| 0 | 개발환경 및 프로젝트 기반 | ✅ 완료 |
| 1 | Dataset QA, 전처리(processed/v1), 환자 단위 Split — [docs/phase1_data_qa.md](docs/phase1_data_qa.md) | ✅ 완료 |
| 2 | Slice→Volume 재구성, 원본 공간 복원, 평가·정량 엔진 — [docs/phase2_evaluation_reconstruction.md](docs/phase2_evaluation_reconstruction.md) | ✅ 완료 |
| 3 | 2D U-Net Baseline (Val positive mean Dice 0.731) — [docs/phase3_2d_baseline.md](docs/phase3_2d_baseline.md) | ✅ 완료 |
| 4 | 2.5D U-Net 및 비교 (Val positive mean Dice 0.743, ΔDice +0.012, 95% CI [−0.002, +0.028] — 유의차 없음) — [docs/phase4_25d.md](docs/phase4_25d.md) | ✅ 완료 |
| 5A | 독립 단일 환자 추론 (DWI+ADC NIfTI → mask·정량·manifest, CLI/service) — [docs/phase5a_independent_inference.md](docs/phase5a_independent_inference.md) | ✅ 완료 |
| 5B | FastAPI 백엔드 MVP (업로드·작업 큐·단일 GPU worker·결과·Viewer 파일, localhost) — [docs/phase5b_fastapi_backend.md](docs/phase5b_fastapi_backend.md) | ✅ 완료 |
| 6 | React + NiiVue 2D/3D Viewer MVP (DWI/ADC·AI 예측 mask overlay·Axial/Coronal/Sagittal/MPR/3D·부피·연결 영역 정량, localhost, 인증 없음) — [docs/phase6_viewer.md](docs/phase6_viewer.md) | ✅ 완료 |
| 7-A | Connected Component Review(선택·강조·숨기기)와 동기화된 voxel 기반 위치 이동 — [docs/phase7_component_review.md](docs/phase7_component_review.md) | ✅ 완료 |
| 7-B | Research Validation Mode (Validation case GT 비교, TP/FP/FN 차이 지도, 환자 단위 지표, Component Matching) — [docs/phase7_research_validation.md](docs/phase7_research_validation.md) | ✅ 완료 |
| 7-C | 선택적 Viewer 고도화 후보 (surface mesh, 일반 Viewer 레이아웃 정리 등) | 후보 |
| Final Test Evaluation | 봉인된 Test Split으로 최종 내부 성능 1회 측정 | 예정 |
| External Validation | 외부 데이터셋 검증 | 예정 |
| Additional Data Training | 추가 데이터 학습 | 예정 |

## 10. 독립 추론 (Phase 5A)

skull-strip된 DWI + ADC NIfTI 한 쌍(같은 격자)을 입력하면 Phase 4 2.5D U-Net으로 병변 영역을 분할하고
원본 DWI 공간의 mask와 정량 결과를 저장합니다. **연구 및 의료영상 검토 보조용이며 진단을 확정하지
않습니다.** 출력 점수는 보정되지 않은 모델 점수이며 질병 확률이 아닙니다.

```powershell
.venv\Scripts\python.exe scripts\infer_case.py --dwi <dwi.nii.gz> --adc <adc.nii.gz> --output outputs
# 선택: --case-id <비식별 ID> --device auto|cuda|cpu --strict --json --no-export-inputs
```

결과: `outputs/<case_id>/<run_id>/` — `prediction_mask.nii.gz`(uint8 0/1, 원본 DWI shape·affine),
`quantification.json`, `inference_metadata.json`, `manifest.json`, `inputs/`(header를 비운 사본, 공유
기본 대상 아님), `logs/`. 기존 폴더는 덮어쓰지 않으며, 실패하면 `.<run_id>.failed/`에 metadata와 로그만
남습니다. 입력 파일명·절대 경로·NIfTI header 문자열은 기록하지 않습니다. Exit code와 계약 상세는
[docs/phase5a_independent_inference.md](docs/phase5a_independent_inference.md) 참고.

## 11. Research Mode (기본 비활성)

공개 데이터 Validation case에서 AI 예측을 Ground Truth와 비교하는 **localhost 연구·검증 전용** 화면입니다.

- 기본값은 **비활성**입니다 (`configs/api_v1.yaml`의 `research_enabled: false`). 비활성 상태에서는 연구
  API 경로가 등록되지 않습니다(404).
- 환경변수로 명시적으로 활성화합니다: API를 실행할 PowerShell 창에서 `$env:NEUROSCAN_RESEARCH_ENABLED = "1"`.
- **Validation case만** CLI로 등록합니다. Test Split case는 파일을 열기 전에 거부되며 Research Mode에
  등록하지 않습니다. 화면·URL·API에는 원본 case ID 대신 alias(`val-001` 등)만 표시됩니다.

  ```powershell
  $env:PYTHONPATH = "backend"
  .venv\Scripts\python.exe scripts\build_research_case.py --case <validation case key> --alias val-001
  ```

- 일반 사용자 업로드 화면(일반 Inference Viewer)에는 Ground Truth와 환자별 Dice가 표시되지 않습니다.
- 상세: [docs/phase7_research_validation.md](docs/phase7_research_validation.md)

## 12. 알려진 제한

- 외부 데이터셋 검증을 수행하지 않았습니다.
- 봉인된 Test Split의 최종 내부 Test 평가를 수행하지 않았습니다.
- 인증·권한과 HTTPS가 없습니다. 현재 localhost 연구용으로만 실행합니다.
- DICOM / PACS 연동이 없습니다 (NIfTI 입력만 지원).
- 의료진의 Mask 편집 기능이 없습니다.
- 실제 임상 사용 승인을 받지 않았습니다.

## 13. 프로젝트 코드 라이선스

현재 저장소에는 프로젝트 코드 라이선스(`LICENSE`) 파일이 없습니다. 라이선스는 아직 정하지 않았습니다.
ISLES 2022 데이터의 라이선스(CC BY 4.0)는 [§8](#8-데이터--저장소-정책과-데이터-라이선스)을 참고하세요.
