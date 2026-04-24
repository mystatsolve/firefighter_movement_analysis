# 듀얼 카메라 소방관 포즈 분석 시스템

OAK-D 카메라 2대를 활용한 실시간 소방관 자세 분석 및 부상 위험도 예측 시스템입니다.

## 시스템 개요

기존 단일 카메라 시스템은 카메라 각도에 따라 관절 감지 정확도가 크게 좌우되었습니다.
본 시스템은 **2대의 OAK-D 카메라**와 **양측(좌/우) 관절 분석** + **카메라간 신뢰도 가중 퓨전**을 적용하여 정확도를 대폭 향상시킵니다.

### 정확도 향상 원리

```
기존 시스템 (단일 카메라, 왼쪽만):
  카메라 → MediaPipe → 왼쪽 관절 6개 → 분석
  문제: 카메라 각도에 따라 정확도 ±30% 변동

본 시스템 (듀얼 카메라, 양측):
  카메라1 ─→ MediaPipe ─→ 좌/우 12개 관절 ─┐
                                             ├→ 카메라간 퓨전 → 스무딩 → 분석
  카메라2 ─→ MediaPipe ─→ 좌/우 12개 관절 ─┘
  개선: visibility 가중 평균으로 안정적 추정
```

### 퓨전 수식

각 관절 각도에 대해:

1. **양측 퓨전** (카메라 내부): visibility가 높은 쪽에 가중치를 더 줌
   ```
   w_left = vis_left / (vis_left + vis_right + ε)
   angle_bilateral = w_left × angle_left + w_right × angle_right
   ```

2. **카메라간 퓨전**: 두 카메라의 같은 관절을 퓨전
   ```
   w1 = vis_cam1 / (vis_cam1 + vis_cam2 + ε)
   angle_fused = w1 × angle_cam1 + w2 × angle_cam2
   ```

3. **이동 평균 스무딩**: 5프레임 윈도우로 프레임간 노이즈 제거

## 설치

### 필수 요구사항

- Python 3.8 이상
- OAK-D 카메라 1~2대 (USB3 권장)

### 패키지 설치

```bash
pip install depthai opencv-python mediapipe numpy scipy matplotlib
```

### 모델 파일

`pose_landmarker_full.task` 파일이 `skel_analysis/` 디렉토리에 필요합니다.
이미 포함되어 있으므로 별도 다운로드는 불필요합니다.

## 실행 방법

### 기본 실행 (듀얼 카메���)

```bash
cd two_camera/skel_analysis
python main.py
```

### 단일 카메라 모드

카메라가 1대만 연결되면 자동으로 단일 모드로 전환됩니다.
강제로 단일 모드를 사용하려면:

```bash
python main.py --single
```

### 하중/작업 유형 지정

소방관이 장비를 다루는 상황을 시뮬레이션할 수 있습니다:

```bash
# 20kg 장비 들기
python main.py --load 20 --task lift

# 15kg 호스 끌기
python main.py --load 15 --task pull

# 25kg 환자 운반
python main.py --load 25 --task carry

# 장비 밀기
python main.py --load 10 --task push
```

### 체중 설정

기본 체중은 75kg입니다. 변경하려면:

```bash
python main.py --body-mass 80
```

### 시작과 동시에 녹화

```bash
python main.py --record
```

## 키 조작

| 키 | 기능 |
|---|---|
| `q` 또는 `ESC` | 프로그램 종료 |
| `r` | 녹화 시작/중지 (CSV + 차트 자동 저장) |
| `s` | 스냅샷 (현재 분석 보고서 저장) |
| `SPACE` | 일시정지/재개 |

## 화면 구성 (HUD)

```
+------------------+------------------+------------------+
|   카메라 1        |   카메라 2        |   분석 패널      |
|   포즈 스켈레톤   |   포즈 스켈레톤   |   퓨전 각도      |
|   + 각도 라벨    |   + 각도 라벨    |   confidence 바  |
|                  |                  |   부위별 위험도   |
|                  |                  |   종합 위험도    |
+------------------+------------------+------------------+
|  FPS | 카메라수 | 감지율1 | 감지율2 | REC | 분석횟수    |
+------------------------------------------------------+
```

## 파일 구조

```
skel_analysis/
├── main.py                  # 진입점 - 메인 루프 + CLI
├── config.py                # 전체 설정 상수 (카메라, 분석, 생체역학)
├── dual_camera.py           # OAK-D 2대 동시 관리
├── pose_analyzer.py         # MediaPipe 양측(Bilateral) 포즈 감지
├── angle_fusion.py          # 2카메라 신뢰도 가중 퓨전 (핵심 신규)
├── biomech_engine.py        # Hill-type 근육 + 인대 시뮬레이션
├── injury_predictor.py      # 위험도 분류 + 보고서 생성
├── realtime_display.py      # 듀얼 카메라 HUD
├── data_export.py           # CSV/차트 내보내기
├── pose_landmarker_full.task # MediaPipe 모델 파일
└── README.md                # 이 문서
```

## 아키텍처

### 데이터 플로우

```
DualCameraManager          # OAK-D 2대 프레임 획득
    ↓ [frame1, frame2]
PoseAnalyzer (×2)          # 카메라별 독립 포즈 분석
    ↓ [angles+vis ×2]
AngleFusionEngine          # 카메라간 가중 퓨전 + 스무딩
    ↓ [fused_angles]
BiomechEngine (background) # Hill-type 근육 + 인대 시뮬레이션
    ↓ [muscle_risks, ligament_risks]
InjuryPredictor            # 위험도 분류 + 보고서
    ↓
RealtimeDisplay            # 실시간 HUD 표시
DataExporter               # CSV + 차트 저장
```

### 비동기 처리

- **메인 루프**: 30fps로 프레임 획득 + 포즈 감지 + 퓨전 + 디스플레이
- **백그라운드**: 2초마다(60프레임) 생체역학 시뮬레이션 실행 (~100-200ms)
- ThreadPoolExecutor(max_workers=1)로 메인 루프 차단 없이 분석

## 출력 파일

`r` 키로 녹화를 중지하거나 프로그램 종료 시 `output/` 디렉토리에 저장됩니다:

| 파일 | 설명 |
|---|---|
| `joint_angles_YYYYMMDD_HHMMSS.csv` | 프레임별 관절 각도 + confidence |
| `analysis_YYYYMMDD_HHMMSS.csv` | 근육 스트레스 + 인대 변형률 |
| `angles_chart_YYYYMMDD_HHMMSS.png` | 관절 각도 시계열 그래프 |
| `muscle_stress_YYYYMMDD_HHMMSS.png` | 근육 스트레스 바 차트 |
| `body_risks_YYYYMMDD_HHMMSS.png` | 부위별 위험도 차트 |
| `report_YYYYMMDD_HHMMSS.txt` | 텍스트 분석 보고서 (`s` 키) |

## 생체역학 모델

### 근육 모델 (DeGrooteFregly2016Muscle)

OpenSim의 Hill-type 3요소 근육 모델을 구현합니다:
- **수축 요소 (CE)**: 신경 활성화에 의한 능동적 힘 생성
- **병렬 탄성 요소 (PE)**: 근막/결합조직의 수동 저항
- **직렬 탄성 요소 (SE)**: 건(tendon)의 탄성

분석 대상 8개 근육: 대퇴사두근, 대퇴이두근, 비복근, 대둔근, 척추기립근, 삼각근, 상완이두근, 광배근

### 인대 모델 (Blankevoort1991Ligament)

3구간 비선형 힘-변형률 관계를 모델링합니다:
- **이완 구간**: 변형률 ≤ 0, 힘 = 0
- **전이 구간**: 포물선 (콜라겐이 점진적으로 펴짐)
- **선형 구간**: 후크 법칙 유사 (콜라겐이 완전히 펴짐)

분석 대상 6개 인대: ACL, PCL, 슬개건, 극상인대, 관절와상완인대, 요추인대

### 위험도 등급

| 등급 | 근육 스트레스 | 인대 변형률 | 의미 |
|---|---|---|---|
| Normal | < 100 kPa | < 3% | 안전 |
| Low | 100~250 kPa | 3~6% | 피로 누적 가능 |
| Medium | 250~400 kPa | 6~10% | 자세 교정 필요 |
| High | 400~600 kPa | 10~15% | 즉시 자세 변경 |
| Critical | > 600 kPa | > 15% | 부상 위험, 즉시 중단 |

## 문제 해결

### 카메라가 감지되지 않음

1. USB 케이블이 제대로 연결되었는지 확인
2. USB 3.0 포트 사용 권장 (USB 2.0에서도 동작하지만 속도 저하)
3. 다른 프로그램이 카메라를 사용 중인지 확인

### 포즈가 감지되지 않음

1. 카메라 앞에 사람이 전신이 보이도록 위치
2. 조명이 충분한지 확인
3. `--single` 모드로 먼저 테스트

### FPS가 낮음

1. USB 3.0 포트 사용 확인
2. 다른 무거운 프로그램 종료
3. 해상도 조정: `config.py`에서 `OAKD_RESOLUTION_W/H` 변경
