# OAK-D Real-Time Firefighter Pose Analysis System

## 소방관 실시간 동작 분석 및 부상 위험도 예측 시스템

OAK-D 깊이 카메라와 MediaPipe Pose를 결합하여 소방관의 실시간 동작을 분석하고,
OpenSim 기반 생체역학 시뮬레이션으로 근골격계 부상 위험도를 예측하는 시스템입니다.

---

## 시스템 개요

### 목적
소방관은 현장 활동 중 무거운 장비를 운반하고 극한 자세를 취합니다.
이 시스템은 실시간으로 관절 각도를 모니터링하고, 근육 스트레스와 인대 변형률을
시뮬레이션하여 과도한 부하나 위험한 자세를 조기에 감지합니다.

### 핵심 기능
- **실시간 포즈 감지**: MediaPipe Tasks API로 33개 신체 랜드마크 추출
- **6개 관절 각도 계산**: 무릎, 고관절, 발목, 어깨, 팔꿈치, 몸통 기울기
- **생체역학 시뮬레이션**: Hill-type 근육 모델 + Blankevoort 인대 모델
- **부상 위험도 평가**: 근육 스트레스 / 인대 변형률 기반 5단계 분류
- **외부 하중 반영**: 장비 무게에 따른 추가 근육/인대 부하 계산
- **실시간 시각화**: HUD 오버레이로 각도, 위험도 바, 부위별 리스크 표시
- **데이터 녹화**: CSV(관절 각도) + MP4(비디오) 동시 저장

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                    메인 루프 (30fps)                                  │
│                                                                     │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐              │
│  │ OAK-D    │    │ MediaPipe     │    │ Angle        │              │
│  │ Camera   │───▶│ PoseAnalyzer  │───▶│ Buffer       │              │
│  │          │    │               │    │ (60 frames)  │              │
│  └──────────┘    └───────────────┘    └──────┬───────┘              │
│       BGR             angles                  │                     │
│       frame           + skeleton              │ (매 15프레임)       │
│                                               ▼                     │
│  ┌──────────┐                          ┌──────────────┐             │
│  │ Realtime │◀─────────────────────────│ Biomech      │             │
│  │ Display  │         result           │ Engine       │             │
│  │ (HUD)    │                          │ (Background) │             │
│  └──────────┘                          └──────────────┘             │
│                                         ThreadPool                  │
│                                         ~100-200ms                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 데이터 흐름
1. **프레임 획득**: OAK-D RGB 카메라 또는 웹캠에서 BGR 프레임 수신
2. **포즈 감지**: MediaPipe PoseLandmarker로 33개 랜드마크 추출
3. **각도 계산**: 6개 관절 각도를 벡터 내적으로 계산 (0-180°)
4. **버퍼 축적**: 최근 60프레임(2초)의 각도 데이터를 롤링 윈도우로 유지
5. **분석 제출**: 15프레임마다 윈도우 데이터를 백그라운드 스레드에 제출
6. **시뮬레이션**: Hill-type 근육 모델 ODE + 인대 힘-변형 모델 실행
7. **위험도 평가**: 피크 스트레스/변형률을 임계값과 비교하여 5단계 분류
8. **시각화**: HUD 오버레이로 실시간 결과 표시

---

## 파일 구조

```
fire_pose_detection/oakd_realtime/
├── main.py               # 진입점: CLI 인자 파싱 + 실시간 루프 오케스트레이션
├── oakd_camera.py        # OAK-D DepthAI 파이프라인 + 웹캠 폴백
├── pose_analyzer.py      # MediaPipe Tasks API 래퍼 (포즈 감지 + 각도 계산)
├── biomech_engine.py     # 비동기 생체역학 시뮬레이션 엔진
├── angle_buffer.py       # 롤링 윈도우 관절 각도 버퍼
├── realtime_display.py   # OpenCV 기반 실시간 HUD 오버레이
├── config.py             # 모든 설정 상수 (모델 파라미터, 임계값 등)
├── requirements.txt      # Python 패키지 의존성
├── .gitignore            # Git 제외 파일 목록
└── README.md             # 이 문서
```

### 각 모듈 역할

| 모듈 | 줄 수 | 역할 |
|------|-------|------|
| `config.py` | ~300 | 카메라/분석/모델/디스플레이 설정, 근육/인대 파라미터 |
| `oakd_camera.py` | ~250 | DepthAI 파이프라인, XLink 스트리밍, 웹캠 폴백 |
| `pose_analyzer.py` | ~700 | MediaPipe 포즈 감지, 관절 각도 계산, 스켈레톤 렌더링 |
| `biomech_engine.py` | ~550 | Hill-type 근육, Blankevoort 인대, 비동기 분석 |
| `angle_buffer.py` | ~210 | deque 기반 롤링 버퍼, 녹화 모드 |
| `realtime_display.py` | ~350 | 사이드 패널, 위험도 바, 하단 상태 바 |
| `main.py` | ~300 | CLI, 메인 루프, 키보드 처리, 녹화/스냅샷 |

---

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
depthai>=2.24.0
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
```

### 2. MediaPipe 모델 파일 다운로드

```bash
# pose_landmarker_full.task 모델 다운로드 (약 9MB)
# oakd_realtime/ 폴더에 배치
curl -L -o pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
```

### 3. 실행

```bash
# OAK-D 카메라 (기본)
python main.py

# 웹캠 모드
python main.py --webcam
python main.py --webcam --webcam-id 1

# OAK-D + 스테레오 깊이
python main.py --depth

# 외부 하중 + 작업 유형 지정
python main.py --load 25 --task lift --body-mass 80

# 시작과 동시에 녹화
python main.py --record --load 25 --task lift
```

### CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--webcam` | False | 웹캠 모드 (OAK-D 대신) |
| `--webcam-id` | 0 | 웹캠 장치 번호 |
| `--depth` | False | OAK-D 스테레오 깊이 활성화 |
| `--record` | False | 시작과 동시에 녹화 |
| `--load` | 0.0 | 외부 하중 (kg) |
| `--task` | none | 작업 유형: none/lift/pull/carry/push |
| `--body-mass` | 75.0 | 작업자 체중 (kg) |

### 실시간 키보드 조작

| 키 | 동작 |
|----|------|
| `q` / `ESC` | 프로그램 종료 |
| `r` | 녹화 시작/중지 (CSV + MP4) |
| `s` | 현재 화면 스냅샷 (PNG) |
| `Space` | 일시정지/재개 |

---

## 생체역학 모델 상세

### Hill-type 근육 모델 (DeGrooteFregly2016Muscle)

De Groote & Fregly (2016) 논문 기반의 Hill-type 근육 모델입니다.

**총 근력 계산식:**
```
F_total = F_max × [activation × f_active(l) × f_v(v)] + F_max × f_passive(l) + F_damping
```

구성 요소:
- **활성화 역학 (Activation Dynamics)**: 신경 흥분 → 근육 활성화 ODE
  - tau_a = 15ms (활성화 시간 상수)
  - tau_d = 60ms (비활성화 시간 상수)
- **능동적 힘-길이 관계**: 3개 비대칭 가우시안의 합
  - nfl=1.0 (최적 길이)에서 최대 힘 출력
- **수동적 힘-길이 관계**: 지수적 탄성
  - 근섬유가 최적 길이 이상으로 늘어나면 수동 저항 발생
- **힘-속도 관계**: 역쌍곡사인 기반
  - 단축성: 속도↑ → 힘↓
  - 신장성: 속도↑ → 힘↑ (최대 1.8배)

**분석 대상 근육:**
| 근육 | 최대 등척성 힘 | 작용 관절 |
|------|--------------|----------|
| Quadriceps (대퇴사두근) | 6000 N | 무릎 신전 |
| Hamstrings (햄스트링) | 3500 N | 무릎 굴곡 |
| Gluteus Maximus (대둔근) | 3000 N | 고관절 신전 |
| Erector Spinae (척추기립근) | 4000 N | 몸통 신전 |
| Gastrocnemius (비복근) | 2500 N | 발목 저굴 |
| Deltoid (삼각근) | 1500 N | 어깨 외전 |

### Blankevoort 인대 모델 (Blankevoort1991Ligament)

3영역 비선형 힘-변형률 관계:
1. **Slack 영역** (strain ≤ 0): 힘 = 0 (이완 상태)
2. **Toe 영역** (0 < strain < 6%): F = ½k/ε_t × strain² (포물선)
3. **Linear 영역** (strain ≥ 6%): F = k × (strain - ε_t/2) (선형)

**분석 대상 인대:**
| 인대 | 선형 강성 | 작용 관절 |
|------|----------|----------|
| ACL (전방십자인대) | 5000 N | 무릎 |
| PCL (후방십자인대) | 6000 N | 무릎 |
| MCL (내측측부인대) | 4000 N | 무릎 |
| Achilles (아킬레스건) | 8000 N | 발목 |

### 위험도 분류 기준

**근육 스트레스 임계값:**
| 등급 | 스트레스 (kPa) | 의미 |
|------|---------------|------|
| Normal | < 100 | 정상 범위 |
| Low | 100~200 | 경미한 부하 |
| Moderate | 200~350 | 주의 필요 |
| High | 350~500 | 높은 부상 위험 |
| Critical | > 500 | 즉시 중단 필요 |

**인대 변형률 임계값:**
| 등급 | 변형률 (%) | 의미 |
|------|-----------|------|
| Normal | < 2% | 정상 범위 |
| Low | 2~4% | 경미한 신장 |
| Moderate | 4~6% | 주의 (토우 영역 상한) |
| High | 6~10% | 미세 손상 가능 |
| Critical | > 10% | 파열 위험 |

---

## 외부 하중 모델

소방관이 장비를 운반할 때의 추가 역학적 부하를 반영합니다.

### 작업 유형별 부하 프로파일

| 작업 유형 | 설명 | 주요 부하 부위 |
|----------|------|--------------|
| `lift` | 장비 들어올리기 | 허리(척추기립근), 무릎(대퇴사두근) |
| `pull` | 호스 당기기 | 어깨(삼각근), 허리 |
| `carry` | 장비 운반 | 전신 균등 |
| `push` | 문 밀기 | 어깨, 무릎 |

### 부하 계산

```
추가 excitation = (하중력 × 모멘트암 × 자세계수 × 하중계수) / 최대 토크
```

- 하중력 = load_kg × 9.81 [N]
- 모멘트암 = 0.05 [m] (평균적 관절-하중 거리)
- 자세계수 = sin(관절각도/2) (각도에 따른 모멘트 변화)
- 하중계수 = 작업유형별 근육 기여도 (config.py LOAD_TASK_PROFILES)

---

## 하드웨어 요구사항

### 권장 사양
- **카메라**: Luxonis OAK-D (또는 OAK-D Lite, OAK-D Pro)
- **USB**: USB3 포트 (USB2도 가능하지만 대역폭 제한)
- **CPU**: Intel i5/AMD Ryzen 5 이상 (MediaPipe + scipy 연산)
- **RAM**: 8GB 이상
- **OS**: Windows 10/11, Ubuntu 20.04+

### 웹캠 대체
OAK-D가 없는 경우 일반 USB 웹캠으로도 동작합니다:
- 깊이 데이터 사용 불가
- 포즈 분석 + 생체역학 시뮬레이션은 정상 동작

---

## 출력 파일

모든 출력은 `output/` 디렉토리에 저장됩니다.

| 파일 형식 | 생성 시점 | 내용 |
|----------|----------|------|
| `snapshot_YYYYMMDD_HHMMSS.png` | `s` 키 입력 | 현재 HUD 화면 캡처 |
| `recording_YYYYMMDD_HHMMSS.csv` | `r` 키로 녹화 종료 | 관절 각도 시계열 (ms 단위) |
| `recording_YYYYMMDD_HHMMSS.mp4` | `r` 키로 녹화 종료 | 스켈레톤 오버레이 비디오 |

### CSV 파일 형식
```csv
Time(ms),Knee,Hip,Ankle,Shoulder,Elbow,Trunk
0,145.2,160.1,88.3,45.0,130.5,170.2
33,144.8,159.5,87.9,44.8,131.0,169.8
66,144.5,158.9,87.5,44.5,131.5,169.5
...
```

---

## 성능 지표

테스트 환경: Intel i7-11700, 16GB RAM, OAK-D

| 지표 | 값 |
|------|------|
| 처리 FPS | 28-30 fps |
| 포즈 감지율 | 80-96% |
| 분석 시간 | ~100-200 ms/회 |
| 분석 주기 | ~0.5초 (15프레임마다) |
| 메모리 사용 | ~200-300 MB |

---

## 참조 문헌

1. De Groote, F., Kinney, A.L., Rao, A.V., Fregly, B.J. (2016).
   "Evaluation of Direct Collocation Optimal Control Problem Formulations
   for Solving the Muscle Redundancy Problem."
   *Annals of Biomedical Engineering*, 44(10), 2922-2936.

2. Blankevoort, L., Kuiper, J.H., Huiskes, R., Grootenboer, H.J. (1991).
   "Articular Contact in a Three-Dimensional Model of the Knee."
   *Journal of Biomechanics*, 24(11), 1019-1031.

3. Lugaresi, C., et al. (2019).
   "MediaPipe: A Framework for Building Perception Pipelines."
   *arXiv preprint arXiv:1906.08172*.

4. Seth, A., et al. (2018).
   "OpenSim: Simulating musculoskeletal dynamics and neuromuscular control
   to study human and animal movement."
   *PLoS Computational Biology*, 14(7), e1006223.

---

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 개발되었습니다.

---

## 개발 환경

- Python 3.9+
- MediaPipe 0.10.33+ (Tasks API)
- DepthAI 2.24+ (OAK-D SDK)
- OpenCV 4.8+
- NumPy 1.24+
- SciPy 1.11+
