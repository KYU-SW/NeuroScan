# Phase 7-0 — NiiVue 기능 검증 spike

작성 2026-11-01 · 기준 커밋 `3c92092` (Phase 6) · 상태: spike 완료 (미커밋) · **Test Split 미사용**

Phase 7-A/B 본 구현 전에 현재 NiiVue 0.69.0 구조로 component labelmap 표시, label별 강조·숨김, crosshair
위치 이벤트, 코드 기반 이동, event loop 방지, oblique affine 왕복이 가능한지 검증했다. 패키지·백엔드·DB·
Phase 5A 출력 계약·Phase 6 화면 동작은 바꾸지 않았다.

## 1. 추가·변경 (프론트엔드만)

| 파일 | 내용 |
|---|---|
| `src/features/viewer/labelLut.ts` (신규) | label 0 투명, 1~254 고유 색(RGB 격자, 채널 간 최소 차 35, 회색·어두운 색 제외), 255 = "기타"(회색 #999999). 선택(alpha 1)·비선택(alpha 0.25 또는 `dimStyle: "gray"`)·숨김(alpha 0) LUT 생성, 잘못된 label 거부 |
| `src/features/viewer/location.ts` (신규) | locationChange payload type guard(unknown → 복사본), file voxel·world mm·frac·slider 정규화, affine 역행렬, frame 단위 묶음 + 같은 정수 voxel 생략 sink, `connectCrosshair` (engine → store 단방향) |
| `ViewerEngine.ts` | 추가만: `setComponentDisplay`, `onLocationChange`, `getCrosshair`, `jumpToVoxel`, `setSliceVoxel` |
| `NiivueEngine.ts` | 위 메서드 구현. 공개 `addEventListener("locationChange")`, `convertMM2Frac(mm, true)`(실제 affine), `convertVox2Frac`, `dimsRAS`, `hdr.affine`, `drawLut`+`cmapper.makeDrawLut`만 사용. 구독자가 없으면 이벤트 처리 생략, `setComponentDisplay`를 부르지 않으면 Phase 6 단색 LUT 그대로 |
| `FakeViewerEngine.ts` | 같은 인터페이스(항등 affine, 64×64×20), `emitRawLocation`으로 NiiVue payload 모의 |
| `src/spike/runSpike7.ts` (신규), `SpikePage.tsx`, `synthetic.ts` | `/__spike?suite=p7&autorun=1` 자동 검증 34항목, 수동 확인 모드, 임의 격자 NIfTI 생성 함수 |
| `tests/phase7spike.test.ts`, `tests/spikeIsolation.test.ts` (신규) | 아래 §4 |

Phase 6 화면은 새 메서드를 호출하지 않는다. `jumpToWorldMm`·`setSliceFraction`은 동작이 같고, 구독자가
있을 때만 위치 알림을 보낸다.

## 2. 실행 방법

```powershell
cd frontend
npm run dev            # http://127.0.0.1:5173
# 자동: http://127.0.0.1:5173/__spike?suite=p7&autorun=1   (Phase 6 회귀: ?suite=p6&autorun=1)
# 수동: 같은 페이지에서 "Phase 7-0 수동 확인" → canvas 클릭·버튼 조작
```

`oblique` 쿼리(선택)는 base64url JSON `{dims, affine(3×4), points:[{label, vox, mm}]}` — **숫자만** 전달하며
영상 데이터·case ID는 포함하지 않는다. 이번 검증은 Validation 0089(SplitGuard로 val 확인, obliquity 31.2°,
LAS)의 header 격자와 Phase 4 Validation 예측 component 5개의 대표 voxel(component 내부, centroid 최근접)을
사용했다. 값은 scratchpad에만 두고 저장소에 넣지 않았다.

## 3. Spike 결과 (headless Chrome, SwiftShader WebGL2)

**Phase 7-0: 33/34 통과** · Phase 6 spike 회귀 **64/64 통과**

| 항목 | 결과 |
|---|---|
| draw LUT = `buildComponentLut` 바이트 일치, label 0 alpha 0 | ✅ |
| 2D: label 1~255 전부 고유 색으로 표시 (축상 한 단면에 0~255 격자) | ✅ 255/255 |
| 2D nearest: overlay 제외 252,895 픽셀 중 label 색이 아닌 픽셀 0 | ✅ (방향 글자·crosshair overlay 64픽셀 제외) |
| `setComponentDisplay(null)` → Phase 6 단색 LUT(#E69F00) 복귀 | ✅ |
| 2D 선택 label 정확한 색 / 비선택 alpha 0.25 혼합식 일치 / 숨김 = 원래 영상 / 복원 = 초기 화면 | ✅ 3584/3584 · 1.000 · 0 · diff 0 |
| 2D 대안(gray): 비선택 = 회색 혼합 색 정확 | ✅ |
| coronal·sagittal label 색 정확 | ✅ |
| 3D label별 색 구분(영역 평균 색도 → 자기 label) | ✅ 4/4 |
| 3D 선택 label 밝기 유지 | ✅ 1.000 |
| **3D 비선택 alpha 0.25** | ❌ 밝기 비율 1.25 / 1.26 / 1.08 (어두워지지 않음) |
| 3D 대안(gray): 비선택 채도 0.21 / 0.20 / 0.00배, 선택 밝기 1.000 | ✅ |
| 3D 숨김 (잔여 0/2400) · 복원 (diff 0) | ✅ |
| `jumpToVoxel` → 이벤트 vox 일치, mm = affine·vox, slider(LAS: 63−i) | ✅ |
| `setSliceVoxel(axial, 5)` → slice만 변경 | ✅ |
| NiiVue `locationChange`(moveCrosshairInVox) 수신, type guard 거부 0 | ✅ |
| canvas mousedown/up → NiiVue 이벤트 → 정규화 위치 | ✅ |
| event loop: 같은 voxel 200회 → store 1회, 이후 변화 3회 → 3회 | ✅ |
| LAS 왕복 200개 | ✅ 200/200 |
| oblique(합성 30°+20°) / oblique(Validation 0089 격자, 31.2°): 이동 정확·mm 오차 | ✅ 5/5 (1.1e-5 / 4.8e-5 mm) |
| 같은 두 격자: crosshair 위치 labelmap 값 = 해당 component | ✅ 5/5 · 5/5 |
| 같은 두 격자: slider 값 재설정 시 voxel 불변 | ✅ 5/5 · 5/5 |
| 같은 두 격자: voxel↔mm 왕복 300개 | ✅ 300/300 · 300/300 |
| 정보: Phase 6 `jumpToWorldMm`의 oblique voxel 오차 | 합성 최대 50 voxel, **0089 최대 20 voxel** |

## 4. 자동 테스트 (Vitest)

`tests/phase7spike.test.ts` 14개: label 0 투명, 1~254 고유·유효(최소 채널 차 35, 회색 제외), 255 예약,
선택 alpha 1, 비선택 alpha 0.25 / gray, 숨김 0(선택보다 우선), 전체 복원, 잘못된 label·dimStyle 거부,
payload type guard(복사·NaN·범위·DataView 거부), 같은 voxel 생략, slider 명령 → engine → 이벤트,
engine 이벤트 → store(store → engine 호출 없음), 100회 반복·drag 10프레임 → 필요한 갱신만, LAS·oblique
voxel↔mm 왕복(500개, 오차 < 1e-9), oblique 위치 → file voxel·slider.
`tests/spikeIsolation.test.ts` 2개: production build(in-memory, `NODE_ENV=production`)에 spike 문자열 없음 +
React production build 확인, 대조군으로 development build에는 spike가 있음.

## 5. 발견한 NiiVue 제한과 대응

1. **3D에서 label별 부분 alpha로 흐려지지 않음.** alpha 0(숨김)은 정확하지만 0.25는 두꺼운 영역에서 오히려
   밝아졌다(ray 누적 방식으로 보임). 2D는 정확하다. 대안 `dimStyle: "gray"`(불투명, 회색 80% 혼합)는 2D·3D
   모두 비선택 영역의 채도를 0.2배 이하로 낮추고 선택 영역은 그대로 둔다 → **7-A 기본 정책 후보**.
2. **Phase 6 `jumpToWorldMm`이 oblique 영상에서 다른 voxel로 이동한다.** `convertMM2Frac(mm)` 기본값이
   직교 근사(`frac2mmOrtho`)라 API의 실제 affine mm와 어긋난다(0089에서 최대 20 voxel). 새 `jumpToVoxel`은
   `convertMM2Frac(mm, true)`로 정확하다(5/5, 300/300). Phase 6 동작 변경 금지에 따라 이번에는 고치지
   않았다 → **7-A에서 병변 중심·component 이동을 `jumpToVoxel`(또는 `isForceSliceMM=true`)로 교체 권고**.
3. locationChange payload 타입이 `unknown`이고, `vox`는 file voxel이 아니라 NiiVue 표시 격자(RAS) index다.
   `frac`은 NiiVue 내부 crosshair 배열 참조다 → type guard + 복사, voxel은 실제 affine 역변환으로 계산.
4. 코드로 `scene.crosshairPos`를 바꾸면 이벤트가 발생하지 않는다(마우스·`moveCrosshairInVox`만 발생) →
   engine 명령이 같은 형식으로 직접 알리고, 중복은 sink가 제거.
5. drawing layer는 R8 → label 최대 255 (254 + 기타).
6. 방향 글자·crosshair는 slice 위에 그려지므로 픽셀 비교 시 제외해야 한다(검증 방법 문제, 표시 문제 아님).

## 6. Test Split

Test Split 영상·GT·예측을 열지 않았다. oblique 검증은 SplitGuard로 Validation임을 확인한 0089의 header와
Phase 4 Validation 예측만 사용했다. Python 회귀는 Test GT를 읽는 `tests/test_phase2_data.py`를 제외했다.

## 7. 이후 결정 (2026-11-01, Phase 7-A)

- 비선택 영역 강조는 2D·3D 모두 **회색 흐림**(`dimStyle` 기본값 `"gray"`)으로 확정. 부분 alpha(`"alpha"`)는 이
  spike의 비교 항목(`3d_dimmed`, 알려진 제한으로 실패 유지)에서만 명시적으로 사용한다.
- Phase 6 `jumpToWorldMm` oblique 오류는 Phase 7-A에서 voxel 기반 이동(`jumpToVoxel`)으로 수정했다
  (`docs/phase7_component_review.md` §5).
