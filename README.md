# Firefighter Movement Analysis - 소방관 동작 분석 기반 부상 예측 시스템

## 개요

소방관의 현장 활동 동작을 분석하여 **근육 내부 압력(kPa)**과 **인대 장력(N)**을 수치화하고, 이를 기반으로 **부상 위험도를 예측**하는 시스템입니다.

Stanford 대학교의 오픈소스 근골격계 시뮬레이션 프레임워크인 [OpenSim](https://github.com/opensim-org/opensim-core)의 핵심 생체역학 모델을 Python으로 구현하였으며, OpenSim 설치 없이 독립 실행이 가능합니다.

---

## 시스템 구성

본 시스템은 **3가지 입력 모드**를 지원합니다:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    입력 모드 3가지                                    │
├─────────────────┬──────────────────┬────────────────────────────────┤
│  Mode 1         │  Mode 2          │  Mode 3                        │
│  가상 시나리오    │  Kinovea CSV     │  동영상 자동 분석               │
│  (run_simulation)│  (run_kinovea)   │  (video_injury_predictor)      │
│                 │                  │                                │
│  소방 활동 5종의 │  Kinovea에서     │  mp4 동영상을 입력하면          │
│  가상 운동 패턴  │  측정한 관절 각도 │  MediaPipe Pose로 자동         │
│  으로 시뮬레이션  │  CSV 데이터를    │  관절 각도 추출 후 분석         │
│                 │  입력으로 사용    │  + 외부 하중(중량물) 지원       │
└────────┬────────┴────────┬─────────┴──────────────┬─────────────────┘
         │                 │                        │
         └────────────────┐│┌───────────────────────┘
                          │││
                    ┌─────▼▼▼──────┐
                    │  생체역학 모델  │
                    │              │
                    │ ● Hill-type  │
                    │   근육 모델   │
                    │ ● 인대 모델   │
                    │ ● 부상 예측   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   출력물       │
                    │              │
                    │ ● 그래프 PNG  │
                    │ ● 데이터 CSV  │
                    │ ● 보고서 TXT  │
                    │ ● 영상 MP4   │
                    └──────────────┘
```

---

## 핵심 생체역학 모델

### 1. 근육 모델: DeGrooteFregly2016Muscle (Hill-type)

OpenSim의 [DeGrooteFregly2016Muscle](https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1DeGrooteFregly2016Muscle.html) 모델을 Python으로 충실히 재현하였습니다.

**핵심 수식:**
```
총 섬유력 = 활성 섬유력 + 수동 섬유력 + 감쇠력

활성 섬유력 = 최대등척성힘 × 활성화 × f(길이) × f(속도)
  - f(길이): 3개 가우시안 함수의 합 (활성 힘-길이 곡선)
  - f(속도): 로그 함수 (힘-속도 곡선)

근육 내부 압력(스트레스) = 총 섬유력(N) ÷ 생리학적 단면적(m²)  → [Pa]
```

**구현된 8개 근육:**

| 근육명 | 최대 등척성 힘 | PCSA | 관련 관절 |
|--------|-------------|------|----------|
| 대퇴사두근 (Quadriceps) | 6,000 N | 75 cm² | 무릎 |
| 비복근 (Gastrocnemius) | 1,600 N | 30 cm² | 발목 |
| 대둔근 (Gluteus Maximus) | 1,500 N | 40 cm² | 고관절 |
| 척추기립근 (Erector Spinae) | 2,500 N | 50 cm² | 허리 |
| 삼각근 (Deltoid) | 1,100 N | 20 cm² | 어깨 |
| 상완이두근 (Biceps) | 600 N | 12 cm² | 팔꿈치 |
| 광배근 (Latissimus Dorsi) | 1,200 N | 25 cm² | 어깨/등 |
| 대퇴이두근 (Hamstrings) | 900 N | 18 cm² | 무릎 |

### 2. 인대 모델: Blankevoort1991Ligament

OpenSim의 [Blankevoort1991Ligament](https://simtk.org/api_docs/opensim/api_docs24/classOpenSim_1_1Ligament.html) 모델을 구현하였습니다.

**3구간 비선형 힘-변형률 관계:**
```
변형률 ≤ 0%     : 힘 = 0 (이완 상태, 인대가 느슨함)
0% < 변형률 < 6% : 힘 = 0.5 × k/ε_t × 변형률²  (발끝 구간, 점진적 저항)
변형률 ≥ 6%      : 힘 = k × (변형률 - ε_t/2)    (선형 구간, 강한 저항)
```

**구현된 6개 인대:**

| 인대명 | 이완 길이 | 강성 | 추정 파괴력 |
|--------|---------|------|-----------|
| 전방십자인대 (ACL) | 32 mm | 5,000 N | 2,160 N |
| 후방십자인대 (PCL) | 38 mm | 9,000 N | 3,000 N |
| 슬개건 (Patellar) | 50 mm | 15,000 N | 10,000 N |
| 극상인대 (Supraspinous) | 45 mm | 3,000 N | 1,500 N |
| 관절와상완인대 (GH) | 25 mm | 2,000 N | 800 N |
| 요추인대 (Lumbar) | 55 mm | 4,000 N | 2,000 N |

### 3. 부상 위험도 분류 기준

**근육 스트레스 임계값:**
| 위험도 | 스트레스 | 의미 |
|--------|---------|------|
| 정상 | < 100 kPa | 일상 활동 수준 |
| 낮음 | 100~250 kPa | 정상 운동 범위 |
| 중간 | 250~400 kPa | 피로 누적 주의 |
| 높음 | 400~600 kPa | 부상 위험 상승 |
| 매우높음 | > 600 kPa | 조직 손상 가능 |

**인대 변형률 임계값:**
| 위험도 | 변형률 | 의미 |
|--------|-------|------|
| 낮음 | 3~6% | 정상 생리학적 범위 |
| 중간 | 6~10% | 발끝-선형 전환점 |
| 높음 | 10~15% | 파괴 접근 |
| 매우높음 | > 15% | 파열 위험 |

---

## 디렉토리 구조

```
firefighter_movement_analysis/
│
├── README.md                          ← 이 파일
├── requirements.txt                   ← Python 패키지 의존성
│
├── biomech/                           ← Mode 1 & 2: 시뮬레이션 기반 분석
│   ├── firefighter_biomech.py         ← 핵심 라이브러리
│   │   ├── DeGrooteFregly2016Muscle   : Hill-type 근육 모델 클래스
│   │   ├── Blankevoort1991Ligament    : 인대 모델 클래스
│   │   ├── InjuryPredictor            : 부상 위험도 분류 엔진
│   │   ├── FirefighterScenario        : 소방관 활동 시나리오 5종
│   │   ├── SimulationEngine           : 시뮬레이션 실행 엔진
│   │   ├── JointAngleToMuscle         : 관절 각도→섬유 길이 변환
│   │   └── KinoveaInput               : Kinovea CSV 파서
│   │
│   ├── run_simulation.py              ← Mode 1: 가상 시나리오 실행
│   ├── run_kinovea.py                 ← Mode 2: Kinovea CSV 입력 실행
│   └── output/                        ← 결과 저장 폴더
│
├── biotech_cam2/                      ← Mode 3: 동영상 기반 분석
│   ├── video_injury_predictor.py      ← 동영상→포즈→부상예측 전체 파이프라인
│   │   ├── PoseAnalyzer               : MediaPipe Pose 관절 각도 추출
│   │   ├── VideoProcessor             : 동영상 프레임 처리
│   │   ├── DeGrooteFregly2016Muscle   : 근육 모델 (독립 구현)
│   │   ├── Blankevoort1991Ligament    : 인대 모델 (독립 구현)
│   │   ├── LOAD_TASK_PROFILES         : 들기/끌기/운반/밀기 하중 모델
│   │   └── 시각화 + CSV 출력 함수들
│   │
│   ├── create_test_video.py           ← 테스트용 스켈레톤 영상 생성
│   └── output/                        ← 결과 저장 폴더
│
└── sample_data/                       ← 샘플 데이터
    └── sample_kinovea_data.csv        ← Kinovea 형식 샘플 CSV
```

---

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

또는 개별 설치:
```bash
# Mode 1 & 2 (시뮬레이션/Kinovea)
pip install numpy matplotlib scipy

# Mode 3 (동영상 분석) - 추가 설치
pip install mediapipe opencv-python
```

### 2. Mode 1: 가상 시나리오 시뮬레이션

소방관 활동 5종(계단 오르기, 장비 운반, 호스 당기기, 사다리 오르기, 요구조자 구출)에 대한 가상 시뮬레이션을 실행합니다.

```bash
cd biomech
python run_simulation.py
```

**출력물:**
- 시나리오별 근육 힘/스트레스/인대 장력 그래프 (PNG)
- 부상 위험도 히트맵 (신체 부위 × 시나리오)
- 종합 대시보드
- 모델 특성 곡선 (힘-길이, 힘-속도)
- CSV 결과 파일 3개

### 3. Mode 2: Kinovea 데이터 입력

Kinovea로 측정한 관절 각도 CSV 파일을 입력으로 사용합니다.

```bash
cd biomech
python run_kinovea.py <CSV파일경로> [시나리오이름]

# 예시
python run_kinovea.py ../sample_data/sample_kinovea_data.csv "계단 오르기 실측"
```

**Kinovea CSV 형식:**
```csv
Time(ms), Knee_Angle, Hip_Angle, Ankle_Angle, Shoulder_Angle, Elbow_Angle, Trunk_Angle
0,        170.2,      165.3,     95.1,         30.5,           155.2,       175.0
33,       168.5,      163.1,     93.8,         32.1,           153.0,       173.5
```

모든 관절이 필수가 아닙니다 - 측정한 관절만 자동으로 분석됩니다.

### 4. Mode 3: 동영상 자동 분석

동영상에서 MediaPipe Pose로 자동 관절 각도를 추출하고 부상을 예측합니다.

```bash
cd biotech_cam2

# 기본 (맨몸 동작)
python video_injury_predictor.py 소방활동.mp4 "현장활동"

# 30kg 물체 들기
python video_injury_predictor.py 소방활동.mp4 "장비들기" 30 lift

# 50kg 호스 끌기
python video_injury_predictor.py 소방활동.mp4 "호스당기기" 50 pull

# 25kg 장비 운반
python video_injury_predictor.py 소방활동.mp4 "장비운반" 25 carry

# 40kg 문 밀기 (체중 80kg 지정)
python video_injury_predictor.py 소방활동.mp4 "문밀기" 40 push 80

# 웹캠 실시간 분석
python video_injury_predictor.py 0 "실시간"
```

**명령어 구조:**
```
python video_injury_predictor.py <동영상> [시나리오명] [하중kg] [동작유형] [체중kg]
```

| 인자 | 설명 | 기본값 |
|------|------|-------|
| 동영상 | mp4/avi/mov 파일 경로 또는 `0`(웹캠) | 필수 |
| 시나리오명 | 분석 이름 | "동영상 분석" |
| 하중kg | 외부 하중 무게 | 0 (맨몸) |
| 동작유형 | lift / pull / carry / push | none |
| 체중kg | 대상자 체중 | 75 |

**출력물:**
- `pose_annotated.mp4` - 포즈 랜드마크가 표시된 영상
- `joint_angles.png` - 관절 각도 시계열 그래프
- `muscle_stress.png` - 근육 내부 압력 + 위험 임계값
- `ligament_tension.png` - 인대 장력/변형률
- `body_risk_chart.png` - 신체 부위별 위험도 차트
- `video_joint_angles.csv` - 관절 각도 CSV (Kinovea 호환)
- `video_muscle_results.csv` - 근육 분석 결과
- `video_ligament_results.csv` - 인대 분석 결과
- `video_body_risks.csv` - 신체 부위별 위험도

---

## 외부 하중 모델 (역동역학 기반)

Mode 3에서 중량물을 들거나 끌거나 할 때의 추가 근육/인대 부하를 반영합니다.

**원리:**
```
추가 관절 토크 = 외부 하중(N) × 모멘트 암(m) × 자세 계수 × 분배 계수
추가 근육 흥분 = 추가 관절 토크 ÷ 최대 근력 토크
```

**동작 유형별 근육 부하 분배:**

| 동작 | 가장 부하가 큰 근육 | 가장 부하가 큰 인대 |
|------|------------------|------------------|
| **들기 (lift)** | 척추기립근(45%), 대퇴사두근(35%) | 요추인대(50%), 극상인대(40%) |
| **끌기 (pull)** | 광배근(40%), 상완이두근(35%) | 관절와상완인대(35%), 요추인대(35%) |
| **운반 (carry)** | 척추기립근(35%), 대퇴사두근(30%) | 요추인대(40%), 슬개건(30%) |
| **밀기 (push)** | 삼각근(35%), 대퇴사두근(30%) | 관절와상완인대(30%), 요추인대(30%) |

---

## 관절 각도 → 근육 섬유 길이 변환

동영상이나 Kinovea에서 측정된 관절 각도를 근육 섬유 길이로 변환하는 매핑:

| 관절 (CSV 컬럼) | 관련 근육 | 관련 인대 |
|----------------|---------|---------|
| `Knee_Angle` (무릎) | 대퇴사두근, 대퇴이두근 | ACL, PCL, 슬개건 |
| `Hip_Angle` (고관절) | 대둔근 | - |
| `Ankle_Angle` (발목) | 비복근 | - |
| `Shoulder_Angle` (어깨) | 삼각근, 광배근 | 관절와상완인대 |
| `Elbow_Angle` (팔꿈치) | 상완이두근 | - |
| `Trunk_Angle` (몸통) | 척추기립근 | 극상인대, 요추인대 |

---

## 참고 문헌

- De Groote, F., et al. (2016). "Evaluation of Direct Collocation Optimal Control Problem Formulations for Solving the Muscle Redundancy Problem." *Annals of Biomedical Engineering*, 44(10), 2922-2936.
- Blankevoort, L. & Huiskes, R. (1991). "Ligament-bone interaction in a three-dimensional model of the knee." *J Biomech Eng*, 113(3), 263-269.
- Seth, A., et al. (2018). "OpenSim: Simulating musculoskeletal dynamics and neuromuscular control to study human and animal movement." *PLOS Computational Biology*, 14(7), e1006223.
- Ward, S.R., et al. (2009). "Are current measurements of lower extremity muscle architecture accurate?" *Clinical Orthopaedics and Related Research*, 467(4), 1074-1082.

---

## 라이선스

본 프로젝트는 연구 목적으로 개발되었습니다.
OpenSim 참조: [Apache License 2.0](https://github.com/opensim-org/opensim-core/blob/main/LICENSE)
