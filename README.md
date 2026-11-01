# NeuroScan-ai

**2.5D 딥러닝 기반 허혈성 뇌졸중 병변 분할 및 3D 진단 보조 시스템** (연구용 프로토타입)

> ⚠️ 이 프로젝트는 연구·교육 목적의 프로토타입입니다. 확정 진단, 치료 추천, 예후 예측을 하지 않으며
> 임상 사용이 승인되지 않았습니다. 모든 결과는 의료진의 영상 검토를 보조하기 위한 참고 정보입니다.

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
| Backend/DB | FastAPI + MySQL, AI 파이프라인 검증 이후 구현 (DB에는 메타데이터와 경로만 저장) |
| MVP 제외 | nnU-Net 비교, FLAIR |

## 2. 디렉터리 구조

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
│   └── inference/     #   독립 단일 환자 추론 service (Phase 5A; Web·DB 비의존)
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

## 3. 실행 환경 (Phase 0 검증 완료, 2026-10-17)

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

## 4. 환경 구성

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

## 5. 데이터와 라이선스

- ISLES 2022 원본과 가공 데이터는 `data/` 아래에만 두며 Git에 포함하지 않습니다 (`.gitignore`).
- `*.nii`, `*.nii.gz`, checkpoint(`*.pt`, `*.pth`), `models/`, `results/`도 Git에서 제외합니다.
- ISLES 2022 데이터 라이선스: **CC BY 4.0** (데이터의 `dataset_description.json`, `LICENSE` 기준).
  사용 시 아래 논문을 인용해야 합니다.
- 데이터 인용: Hernandez Petzsche, M.R., de la Rosa, E., Hanning, U. et al. ISLES 2022: A multi-center
  magnetic resonance imaging stroke lesion segmentation dataset. *Scientific Data* 9, 762 (2022).
  DOI: [10.1038/s41597-022-01875-5](https://doi.org/10.1038/s41597-022-01875-5)
- 외부 프로젝트 코드는 복사하지 않습니다. 라이브러리(PyTorch, MONAI, NiBabel, SimpleITK 등)는 각
  라이선스에 따라 의존성으로만 사용합니다.

## 6. 개발 단계

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
| 7 | Viewer 고도화 (vtk.js surface mesh·component별 3D 색상·DICOM·E2E 등, GT overlay·환자별 Dice는 공개 데이터 연구/검증 모드로 분리) | 예정 |

## 7. 독립 추론 (Phase 5A)

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
