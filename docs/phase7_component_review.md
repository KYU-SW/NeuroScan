# Phase 7-A — Connected Component Review

작성 2026-11-01 · 기준 커밋 `3c92092` (Phase 6) · 상태: 구현·수동 smoke 완료, 정책 승인 (2026-11-01) ·
**Test Split 미사용**

> ⚠️ 연구 및 의료영상 검토 보조용 프로토타입이며 의료기기가 아닙니다. 영역 구분·강조·숨김은 검토
> 편의를 위한 표시 기능이며 진단·위험도·모델 확신도를 뜻하지 않습니다.

## 1. 목적

AI 예측 mask 안의 공간적으로 분리된 영역(26-connectivity connected component)을 하나씩 검토한다.
작은 분리 영역은 자동 삭제하지 않고, 정렬·위치 이동·강조·표시 숨김으로 확인을 돕는다. Ground Truth,
Difference Map, 환자별 Dice, Research DB/API(Phase 7-B)는 이번 범위가 아니다.

## 2. Component Labelmap 계약 (Phase 5A 출력 확장)

| 파일 | 역할 |
|---|---|
| `prediction_mask.nii.gz` | **공식 AI 원본 결과** (uint8 0/1). 의미·형식 변경 없음 |
| `prediction_components.nii.gz` (신규) | **검토용 파생 산출물**. 같은 shape·affine·orientation의 uint8 labelmap |

- 값: 0 = background, 1~254 = 부피 내림차순 display label(= `size_rank`), 255 = 254번째 이후 영역 전체
  ("기타"). 영역을 삭제·병합하지 않으며 JSON에는 255번에 속한 영역도 각각 그대로 남는다.
- `analysis.components.component_analysis()` **한 번의 labelling**으로 JSON 목록과 labelmap을 함께 만든다
  (`inference.quantify.quantify_with_labelmap`). 기존 `components()` 필드는 그대로이며 새 필드만 추가된다.
- 저장 전 검증(`check_labelmap`, 실패 시 `E_OUTPUT_VERIFY`): labelmap foreground = mask foreground,
  label별 voxel 수 = JSON 합계, 대표 voxel이 자기 label 안, mask bytes SHA-256이 분석 전후 동일.
  저장 후에는 기존 `verify_nifti`(배열 완전 일치·dtype·affine·header 텍스트 없음)를 통과해야 한다.
- manifest: `kind = derived_review_labelmap`, `derived_from = prediction_mask`, `same_grid_as = dwi`,
  공유 기본 목록 포함. Backend FileResource role `prediction_components` — mask와 같은 servable/shareable
  정책, 다운로드 이름 `prediction_components.nii.gz`.

## 3. Component 정보 (API `components[]`, 추가 필드는 이전 결과에서 `null`)

기존: `label`(scipy label, 의미 불변), `voxel_count`, `volume_ml`, `centroid_voxel`, `centroid_world_mm`,
`bbox_min_voxel`/`bbox_max_voxel`, `hemisphere_estimate`.
추가: `display_label`, `size_rank`, `representative_voxel`, `representative_world_mm`,
`slice_range {axis, first, last}`.

- **representative voxel**: component 안의 voxel 중 centroid와의 거리(mm, spacing 반영)가 가장 가까운
  것(동점은 C 순서 첫 번째). centroid가 영역 밖(고리·곡선·조각 형태)이어도 항상 영역 내부다.
  world mm는 실제 NIfTI affine으로 계산한다.
- **slice_range**: 영상의 S/I 배열 축 기준 첫·마지막 voxel index. 화면에는 축상면 slider와 같은 1부터
  세는 번호로 보여 주며, 축이 I 방향이면 뒤집어 맞춘다.

## 4. 화면 (Result Viewer 오른쪽 "연결 영역 검토")

- 목록: 영역 번호(크기 순위) · 부피 · slice 범위 · 환자 좌측/우측/정중선 부근/추정 불가 · 색 견본 ·
  중립 표시 "작은 분리 영역".
- **"작은 분리 영역" 기준**: 가장 큰 영역이 아니면서 그 부피의 10% 미만(`SMALL_REGION_RATIO`). 정렬·
  표시용 구분일 뿐 의료적 의미는 없으며 Test 평가 전 동결 대상이다. 위양성·의심·위험 등 의료 용어는
  쓰지 않는다(금지 문구 DOM 검사에 추가).
- 정렬: 부피 큰 순(기본) / 부피 작은 순. 목록 보기: 전체 / 선택한 영역만 / 숨긴 영역 제외.
- 선택: 고유 색 + 전체 밝기, 나머지는 **회색 흐림**(원래 색을 회색 #606060 쪽으로 80% 혼합, 불투명).
  같은 항목을 다시 누르면 선택 유지, "선택 해제"로 모든 색 복원. 숨긴 영역을 선택하면 다시 표시된다.
- 숨기기(영역별) / "선택 영역만 표시" / "전체 다시 표시": LUT alpha 0으로 **표시에서만** 제외. 선택된
  영역을 숨기면 선택이 해제된다.
- 안내 문구: "선택하지 않은 영역은 검토 편의를 위해 흐리게 표시됩니다. AI 원본 예측 결과는 변경되지
  않습니다." / "흐린 표시는 모델 점수, 위험도 또는 의료적 판단을 뜻하지 않습니다." / "숨기기는 화면 표시에서만
  제외합니다. 원본 예측 파일과 정량값은 그대로입니다."
- 영역 색: "영역별 색"(기본) / "단색"(Phase 6 팔레트 색, 강조·숨김은 동일하게 동작).
- 선택·숨김·정렬 상태는 브라우저 메모리(Zustand)에만 있고 새로고침 시 초기화된다. DB 저장, 파일 수정,
  정량 재계산은 없다.
- Phase 7 이전 결과(labelmap 없음): 목록·위치 이동은 동작(대표 voxel 대신 centroid mm를 실제 affine 역변환),
  영역별 색·강조·숨김은 비활성화하고 안내를 표시한다.

### 회색 흐림을 쓰는 이유

Phase 7-0 spike에서 NiiVue drawing layer의 label별 부분 alpha(0.25)는 2D에서는 정확했지만 3D volume
rendering에서는 두께가 있는 영역이 오히려 밝아졌다(ray 누적, `docs/phase7_spike.md` §5). 2D·3D에 같은 규칙을
쓰기 위해 부분 alpha는 강조 정책에서 제외하고 불투명 회색 혼합을 쓴다. 숨김(alpha 0)은 2D·3D 모두 정확하다.

## 5. Oblique 위치 이동 오류와 수정

Phase 6은 병변 중심·영역 이동에 `jumpToWorldMm(world mm)`를 썼고, NiiVue `convertMM2Frac(mm)` 기본값은 축 정렬
근사(`frac2mmOrtho`)라 oblique 영상에서 다른 voxel로 이동했다(Validation 0089, 31.2°: 최대 20 voxel). 이제 모든
이동은 voxel 기준이다.

```
representative_voxel (또는 center_of_mass_voxel)
  → engine.jumpToVoxel()  (voxel → 실제 affine mm → convertMM2Frac(mm, true))
  → NiiVue crosshair → location event → viewerStore.crosshair → 축상/관상/시상 slider
```

`jumpToWorldMm`는 인터페이스에 남아 있지만 결과 화면은 사용하지 않는다(테스트로 호출 0회 확인).

## 6. Crosshair / Slider 흐름 (engine = 위치의 기준)

- Slider → `engine.setSliceVoxel(axis, index)` → location event → store → 세 slider 모두 다시 그림.
- Canvas 클릭 → NiiVue `locationChange` → type guard → store → slider.
- 영역 선택 → `engine.jumpToVoxel` → event → store → slider.
- store → engine 위치 반영 경로 없음. 같은 정수 voxel은 갱신 생략, requestAnimationFrame 단위 묶음,
  unmount 시 listener 해제(`connectCrosshair`). slider는 항상 축상·관상·시상 3개를 표시한다.

## 7. 검증 결과

**자동 검사**: typecheck · lint · format · production build 통과, **Vitest 90 passed** (신규
`componentReview.test.tsx` 15 + 7-0 갱신), spike 회귀 **Phase 6 64/64 · Phase 7-0 33/34**(실패 1 = 문서화된
3D 부분 alpha 제한), Python **Non-Test suite 275 passed, 1 skipped**, ruff 통과. Test Split을 읽는
`tests/test_phase2_data.py` 4개는 봉인 정책으로 제외.

신규·변경 Python 테스트: `tests/test_component_labelmap.py` 8개(JSON/labelmap 일치, foreground 보존, 입력 mask
불변, centroid가 밖인 고리 영역의 대표 voxel, slice 범위, 1 voxel 영역, 254개 / 300개(255 "기타"), oblique
affine 왕복, 불일치 검출, 실제 엔진 실행 산출물), 출력 계약·API 테스트 갱신(labelmap 다운로드·SHA·label별 voxel
수 검증 포함).

**Validation smoke** (동결된 Phase 5A engine, GPU, 임시 storage, production build + headless Chrome):

| 항목 | 0086 | 0089 (oblique 31.2°) |
|---|---|---|
| 총 부피 / 영역 수 / 최대 영역 / slice 수 | 48.98 mL / 9 / 48.50 mL / 29 (Phase 6과 동일) | 1.66 mL / 5 / 1.49 mL / 14 |
| mask vs Phase 4 예측 불일치 voxel | 0 / 915,712 | 0 / 915,712 |
| labelmap 검증 (foreground·label별 voxel·대표 voxel) | 통과 | 통과 |
| 영역 선택 후 세 slider = 대표 voxel | 9/9 | 5/5 (1 voxel 영역 2개 포함) |
| 열자마자 slider = 병변 중심 voxel | 일치 | 일치 |
| 숨김·복원 후 result JSON / mask SHA-256 | 불변 / 불변 | 불변 / 불변 |
| 요청 경로 | `/api/v1`만 | `/api/v1`만 |
| 금지 문구·파일명·case ID | 0건 | 0건 |

0089의 slider 값(예: 영역 1 voxel 69,48,37 → 42,48,37)은 Phase 7-0 spike의 NiiVue 측정과 같다.

### 7.1 사용자 수동 smoke (Chrome/Edge, 실제 GPU) 및 승인 (2026-11-01)

| 항목 | 결과 |
|---|---|
| 0086: 총 부피 48.98 mL · 9개 · 최대 48.50 mL | 확인 (자동 smoke와 동일) |
| 0086: 영역 선택·위치 이동, 세 slider 즉시 동기화, 선택 강조, 비선택 회색 흐림, 숨기기·전체 복원, 2D/3D 상태 일치 | 정상 |
| 0089: 5개 영역, oblique 위치 이동, 작은 영역 유지, crosshair·slider 동기화 | 정상 |

확정된 정책:

1. "작은 분리 영역" = 가장 큰 영역이 아니면서 그 부피의 10% 미만. Viewer 우선 검토용 UI 분류이며 의료적
   판정·위양성 기준이 아니다. 자동 삭제·정량 제외·원본 변경 없음, 위양성·의심·위험 표현 금지. **Test Split
   최종 평가 전 동결.**
2. 비선택 영역은 2D·3D 모두 회색 흐림. 부분 alpha는 공식 강조 방식으로 쓰지 않는다.
3. 병변 중심·영역 이동은 voxel 기반 경로가 공식 정책. `jumpToWorldMm`(축 정렬 근사)는 쓰지 않는다.

## 8. 알려진 제한

- 3D에서 부분 alpha로는 흐려지지 않는다(NiiVue 제한) → 회색 흐림으로 대체.
- label은 최대 255개(R8 drawing layer). 254번째 이후 영역은 "기타" 하나로 함께 강조·숨김된다. API 목록은 기존대로
  상위 100개.
- 3D 카메라 자동 이동, surface mesh, 확률 heatmap, 영역별 점수는 없다(범위 외).
- "작은 분리 영역" 10% 기준은 화면 구분용 규칙이며 Test 전에 동결한다.
- slice 범위의 화면 번호는 nibabel 축 코드 기준으로 축상면 slider와 맞춘다. 축이 45° 근처로 기운 영상에서는
  NiiVue와 축 판정이 다를 수 있다.
- 실행 중인 API/worker는 재시작해야 새 출력을 만든다. 이전 결과는 labelmap 없이 표시된다.

## 9. Phase 7-B로 넘어갈 조건

1. ✅ 계약·정책 승인(§7.1). 10% 기준은 Test 전 동결 유지.
2. ✅ Phase 7-0 + 7-A 커밋.
3. ✅ 사용자 Chrome/Edge 수동 확인(실제 GPU).
4. 7-B는 Validation 전용 CLI 등록·별도 storage/table/route·`research_enabled: false` 기본값으로 시작.
