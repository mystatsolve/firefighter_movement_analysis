"""
config.py - 전체 시스템 설정 상수 모듈
=======================================

이 파일은 OAK-D 실시간 소방관 포즈 분석 시스템의 모든 설정값을 중앙 관리합니다.
각 모듈(카메라, 포즈 감지, 생체역학 분석, 디스플레이)이 이 파일에서 설정을 임포트하여 사용합니다.

구성 섹션:
    1. OAK-D 카메라 설정 ─ 해상도, FPS, 깊이(Depth) 활성화 여부
    2. MediaPipe Pose 설정 ─ 모델 복잡도, 감지/추적 신뢰도 임계값
    3. 분석 설정 ─ 롤링 윈도우 크기, 분석 주기, 기본 체중/하중
    4. 근육 파라미터 (MUSCLE_PARAMS) ─ OpenSim 문헌 기반 8개 근육
    5. 인대 파라미터 (LIGAMENT_PARAMS) ─ OpenSim 문헌 기반 6개 인대
    6. 관절 각도→근육/인대 매핑 ─ 각도에서 섬유 길이/변형률 변환
    7. 하중 작업 프로파일 ─ lift/pull/carry/push 4가지 동작 유형
    8. 위험도 임계값 ─ 근육 스트레스 / 인대 변형률 기준
    9. 디스플레이 설정 ─ 창 이름, 패널 크기, 색상 정의
"""

import numpy as np


# =============================================================================
# 1. OAK-D 카메라 설정 (OAK-D Camera Settings)
# =============================================================================
# OAK-D의 ColorCamera 노드에서 preview 출력으로 사용할 해상도입니다.
# 640x480은 MediaPipe Pose의 입력으로 적합하며, USB 대역폭을 고려한 최적 크기입니다.
# 해상도를 높이면(예: 1280x720) 포즈 감지 정확도가 올라가지만, 처리 속도가 감소합니다.
OAKD_RESOLUTION_W = 640   # 가로 해상도 (pixels)
OAKD_RESOLUTION_H = 480   # 세로 해상도 (pixels)
OAKD_FPS = 30             # 초당 프레임 수 (frames per second)
OAKD_PREVIEW_SIZE = (OAKD_RESOLUTION_W, OAKD_RESOLUTION_H)  # (width, height) 튜플

# Stereo Depth 관련 설정
# --depth 플래그로 활성화할 수 있습니다. 양안(좌/우 모노 카메라)으로 깊이맵을 생성합니다.
# 현재 버전에서는 깊이 데이터를 분석에 직접 사용하지 않지만, 향후 3D 포즈 보정에 활용 가능합니다.
OAKD_ENABLE_DEPTH = False           # 기본적으로 깊이 비활성화
OAKD_DEPTH_ALIGN_TO_RGB = True      # 깊이맵을 RGB 카메라 좌표계로 정렬


# =============================================================================
# 2. MediaPipe Pose 설정 (MediaPipe Pose Settings)
# =============================================================================
# MediaPipe PoseLandmarker의 감지/추적 신뢰도 임계값입니다.
# 값이 높을수록 정확하지만 감지율(Detection Rate)이 떨어질 수 있습니다.
# 소방관 활동처럼 빠른 동작에서는 0.5 정도가 적절합니다.
POSE_MODEL_COMPLEXITY = 1            # 0=Lite(빠름), 1=Full(균형), 2=Heavy(정확)
POSE_MIN_DETECTION_CONF = 0.5       # 포즈 감지 최소 신뢰도 (0.0 ~ 1.0)
POSE_MIN_TRACKING_CONF = 0.5        # 포즈 추적 최소 신뢰도 (0.0 ~ 1.0)


# =============================================================================
# 3. 분석 설정 (Analysis Settings)
# =============================================================================
# 롤링 윈도우: 최근 N 프레임의 관절 각도를 유지합니다.
# 60프레임 = 30fps에서 약 2초 분량입니다.
# 이 윈도우 데이터가 생체역학 시뮬레이션의 입력이 됩니다.
ROLLING_WINDOW_SIZE = 60             # 버퍼 최대 프레임 수

# 분석 제출 간격: 매 N 프레임마다 백그라운드 생체역학 분석을 요청합니다.
# 60프레임 = 약 2초 간격으로 분석 결과가 갱신됩니다.
# 분석 소요 시간은 약 100~200ms이므로 30fps 루프에 영향을 주지 않습니다.
ANALYSIS_INTERVAL = 60               # 분석 제출 간격 (프레임 단위)
ANALYSIS_MAX_WORKERS = 1             # 백그라운드 분석 스레드 수

# 기본 대상자 정보
DEFAULT_BODY_MASS_KG = 75.0          # 기본 체중 (kg)
DEFAULT_LOAD_KG = 0.0                # 기본 외부 하중 (kg), 0 = 맨몸
DEFAULT_TASK_TYPE = 'none'           # 기본 동작 유형 (none = 하중 미적용)

# 신경 흥분(Excitation) 변환 계수
# 관절 각속도(deg/s) → 근육 신경 흥분 수준으로 변환할 때 사용됩니다.
#   excitation = EXCITATION_BASE + EXCITATION_GAIN × |angular_velocity|
# EXCITATION_BASE: 정지 상태에서의 기저 근육 활성화 수준 (근긴장도)
# EXCITATION_GAIN: 각속도 1 deg/s 당 추가되는 흥분량
EXCITATION_GAIN = 0.008              # 각속도 → 흥분 변환 이득
EXCITATION_BASE = 0.15               # 기저 흥분 수준 (0.0 ~ 1.0)


# =============================================================================
# 4. 근육 파라미터 (Muscle Parameters)
# =============================================================================
# OpenSim의 DeGrooteFregly2016Muscle 모델에서 사용하는 근육 생리학적 파라미터입니다.
# 각 파라미터의 의미:
#   - max_isometric_force (N): 최대 등척성 힘 ─ 근육이 최적 길이에서 낼 수 있는 최대 힘
#   - optimal_fiber_length (m): 최적 섬유 길이 ─ 최대 힘을 발생시키는 근섬유 길이
#   - tendon_slack_length (m): 건(tendon) 이완 길이 ─ 힘이 0인 상태의 건 길이
#   - pennation_angle_at_optimal (rad): 우각 ─ 최적 길이에서 근섬유와 건의 각도
#   - pcsa (m²): 생리학적 단면적 ─ 근육 내부 압력(스트레스) 계산에 사용
#   - name_kr: 한국어 근육명 (보고서/표시용)
#
# 참고문헌:
#   - De Groote et al. (2016). Annals of Biomedical Engineering, 44(10), 2922-2936.
#   - Arnold et al. (2010). Ann Biomed Eng, 38(2), 269-279.
MUSCLE_PARAMS = {
    'Quadriceps': {  # 대퇴사두근: 무릎 신전(펴기)의 주동근, 보행/계단/스쿼트의 핵심
        'name_kr': '대퇴사두근',
        'max_isometric_force': 6000,       # 6000 N - 인체에서 가장 큰 힘을 내는 근육
        'optimal_fiber_length': 0.084,     # 84 mm
        'tendon_slack_length': 0.10,       # 100 mm
        'pennation_angle_at_optimal': np.radians(15),  # 15도
        'pcsa': 75e-4,                     # 75 cm² → m² 변환
    },
    'Hamstrings': {  # 대퇴이두근: 무릎 굴곡(구부리기) + 고관절 신전 보조
        'name_kr': '대퇴이두근',
        'max_isometric_force': 900,
        'optimal_fiber_length': 0.11,
        'tendon_slack_length': 0.32,
        'pennation_angle_at_optimal': np.radians(12),
        'pcsa': 18e-4,
    },
    'Gastrocnemius': {  # 비복근: 발목 저측굴곡(발끝으로 서기), 보행 추진력
        'name_kr': '비복근',
        'max_isometric_force': 1600,
        'optimal_fiber_length': 0.055,
        'tendon_slack_length': 0.39,       # 아킬레스건이 길어서 건 이완 길이가 큼
        'pennation_angle_at_optimal': np.radians(17),
        'pcsa': 30e-4,
    },
    'GluteusMax': {  # 대둔근: 고관절 신전, 물체 들기/계단 오르기에서 핵심
        'name_kr': '대둔근',
        'max_isometric_force': 1500,
        'optimal_fiber_length': 0.14,
        'tendon_slack_length': 0.05,
        'pennation_angle_at_optimal': np.radians(5),
        'pcsa': 40e-4,
    },
    'ErectorSpinae': {  # 척추기립근: 몸통 신전(허리 세우기), 들기 동작에서 최대 부하
        'name_kr': '척추기립근',
        'max_isometric_force': 2500,
        'optimal_fiber_length': 0.12,
        'tendon_slack_length': 0.03,
        'pennation_angle_at_optimal': np.radians(0),  # 평행 섬유
        'pcsa': 50e-4,
    },
    'Deltoid': {  # 삼각근: 어깨 외전/굴곡, 물체 밀기/들어올리기
        'name_kr': '삼각근',
        'max_isometric_force': 1100,
        'optimal_fiber_length': 0.10,
        'tendon_slack_length': 0.04,
        'pennation_angle_at_optimal': np.radians(15),
        'pcsa': 20e-4,
    },
    'Biceps': {  # 상완이두근: 팔꿈치 굴곡, 물체 잡기/당기기
        'name_kr': '상완이두근',
        'max_isometric_force': 600,
        'optimal_fiber_length': 0.116,
        'tendon_slack_length': 0.24,
        'pennation_angle_at_optimal': np.radians(0),  # 평행 섬유
        'pcsa': 12e-4,
    },
    'LatissimusDorsi': {  # 광배근: 어깨 신전/내전, 당기기 동작의 핵심
        'name_kr': '광배근',
        'max_isometric_force': 1200,
        'optimal_fiber_length': 0.25,
        'tendon_slack_length': 0.05,
        'pennation_angle_at_optimal': np.radians(20),
        'pcsa': 25e-4,
    },
}


# =============================================================================
# 5. 인대 파라미터 (Ligament Parameters)
# =============================================================================
# OpenSim의 Blankevoort1991Ligament 모델에서 사용하는 인대 기계적 특성입니다.
# 각 파라미터의 의미:
#   - slack_length (m): 이완 길이 ─ 인대에 장력이 0인 상태의 길이
#   - linear_stiffness (N/strain): 선형 강성 ─ 선형 구간에서의 힘/변형률 비율
#   - transition_strain: 발끝→선형 전환 변형률 ─ 이 지점에서 비선형→선형 전환
#   - damping_coefficient (N·s/strain): 감쇠 계수 ─ 동적 하중 시 에너지 흡수
#   - estimated_failure_force (N): 추정 파괴력 ─ 인대가 파열되는 추정 힘
#
# 참고문헌:
#   - Blankevoort & Huiskes (1991). J Biomech Eng, 113(3), 263-269.
#   - Woo et al. (1991). Am J Sports Med, 19(3), 217-225.
LIGAMENT_PARAMS = {
    'ACL': {  # 전방십자인대: 무릎 전방 안정성, 급격한 방향 전환 시 손상 위험
        'name_kr': '전방십자인대(ACL)',
        'slack_length': 0.032,             # 32 mm
        'linear_stiffness': 5000,          # 5000 N/strain
        'transition_strain': 0.06,         # 6%에서 발끝→선형 전환
        'damping_coefficient': 0.003,
        'estimated_failure_force': 2160,   # 약 2160 N에서 파열
    },
    'PCL': {  # 후방십자인대: 무릎 후방 안정성, ACL보다 강함
        'name_kr': '후방십자인대(PCL)',
        'slack_length': 0.038,
        'linear_stiffness': 9000,          # ACL보다 높은 강성
        'transition_strain': 0.06,
        'damping_coefficient': 0.003,
        'estimated_failure_force': 3000,
    },
    'PatellarTendon': {  # 슬개건: 대퇴사두근의 힘을 경골에 전달, 점프/착지 시 고부하
        'name_kr': '슬개건',
        'slack_length': 0.050,
        'linear_stiffness': 15000,         # 매우 높은 강성 (체중의 수 배를 전달)
        'transition_strain': 0.05,
        'damping_coefficient': 0.005,
        'estimated_failure_force': 10000,  # 약 1 톤 힘에서 파열
    },
    'SupraspinousLig': {  # 극상인대: 척추 후방 안정성, 허리 굽힘 시 보호
        'name_kr': '극상인대',
        'slack_length': 0.045,
        'linear_stiffness': 3000,
        'transition_strain': 0.06,
        'damping_coefficient': 0.002,
        'estimated_failure_force': 1500,
    },
    'GlenohumeralLig': {  # 관절와상완인대: 어깨 관절 안정성, 과도한 외회전 시 손상
        'name_kr': '관절와상완인대',
        'slack_length': 0.025,
        'linear_stiffness': 2000,
        'transition_strain': 0.06,
        'damping_coefficient': 0.002,
        'estimated_failure_force': 800,
    },
    'LumbarLig': {  # 요추인대: 요추부 안정성, 무거운 물체 들기 시 최대 부하
        'name_kr': '요추인대',
        'slack_length': 0.055,
        'linear_stiffness': 4000,
        'transition_strain': 0.07,
        'damping_coefficient': 0.003,
        'estimated_failure_force': 2000,
    },
}


# =============================================================================
# 6. 관절 각도 → 근육/인대 매핑 (Angle-to-Tissue Mapping)
# =============================================================================
# MediaPipe에서 추출한 관절 각도(도)를 근육 섬유 길이(정규화)와
# 인대 변형률로 변환하는 선형 매핑 테이블입니다.
#
# 근육 매핑 (ANGLE_TO_MUSCLE):
#   joint: 이 근육과 연관된 관절 각도 이름
#   theta_range: 관절 각도의 작동 범위 (도) ─ 이 범위를 [0,1]로 정규화
#   nfl_range: 정규화 섬유 길이(NFL) 범위 ─ 1.0이 최적 길이
#     - nfl < 1.0: 근육이 수축된 상태 (짧음)
#     - nfl > 1.0: 근육이 신장된 상태 (길음, 수동 장력 발생)
#     - 예: 대퇴사두근은 무릎 60도(깊은 굽힘)에서 0.7(짧음),
#            180도(완전 신전)에서 1.15(길음)
ANGLE_TO_MUSCLE = {
    'Quadriceps':      {'joint': 'Knee_Angle',     'theta_range': (60, 180), 'nfl_range': (0.7, 1.15)},
    'Hamstrings':      {'joint': 'Knee_Angle',     'theta_range': (60, 180), 'nfl_range': (1.2, 0.8)},  # 길항근: 반대 방향
    'Gastrocnemius':   {'joint': 'Ankle_Angle',    'theta_range': (60, 120), 'nfl_range': (1.15, 0.8)},
    'GluteusMax':      {'joint': 'Hip_Angle',      'theta_range': (90, 180), 'nfl_range': (1.15, 0.85)},
    'ErectorSpinae':   {'joint': 'Trunk_Angle',    'theta_range': (120, 180), 'nfl_range': (1.2, 0.9)},
    'Deltoid':         {'joint': 'Shoulder_Angle', 'theta_range': (0, 180), 'nfl_range': (0.8, 1.2)},
    'Biceps':          {'joint': 'Elbow_Angle',    'theta_range': (40, 170), 'nfl_range': (0.7, 1.2)},
    'LatissimusDorsi': {'joint': 'Shoulder_Angle', 'theta_range': (0, 180), 'nfl_range': (1.15, 0.8)},  # 삼각근의 길항근
}

# 인대 매핑 (ANGLE_TO_LIGAMENT):
#   strain_range: 변형률 범위 ─ 관절 각도에 따른 인대 늘어남 정도
#     - 0.01 = 1% 변형 (정상), 0.06 = 6% 변형 (전환점), 0.15 = 15% (파열 위험)
ANGLE_TO_LIGAMENT = {
    'ACL':              {'joint': 'Knee_Angle',     'theta_range': (60, 180), 'strain_range': (0.01, 0.05)},
    'PCL':              {'joint': 'Knee_Angle',     'theta_range': (60, 180), 'strain_range': (0.05, 0.01)},  # ACL과 반대 패턴
    'PatellarTendon':   {'joint': 'Knee_Angle',     'theta_range': (60, 180), 'strain_range': (0.06, 0.02)},
    'SupraspinousLig':  {'joint': 'Trunk_Angle',    'theta_range': (120, 180), 'strain_range': (0.06, 0.01)},
    'GlenohumeralLig':  {'joint': 'Shoulder_Angle', 'theta_range': (0, 180), 'strain_range': (0.01, 0.06)},
    'LumbarLig':        {'joint': 'Trunk_Angle',    'theta_range': (120, 180), 'strain_range': (0.08, 0.01)},
}


# =============================================================================
# 7. 하중 작업 프로파일 (Load Task Profiles)
# =============================================================================
# 역동역학(Inverse Dynamics) 기반 외부 하중 모델입니다.
# 소방관이 장비/환자를 다루는 4가지 기본 동작 유형별로
# 각 근육/인대에 분배되는 추가 부하 비율을 정의합니다.
#
# muscle_load_factor (0~1):
#   해당 동작에서 이 근육이 받는 외부 하중의 비율.
#   값이 클수록 해당 근육이 더 큰 추가 부담을 받습니다.
#   예: lift 시 척추기립근 0.45 = 외부 하중의 45%가 이 근육에 추가 토크로 작용
#
# ligament_strain_factor (0~1):
#   해당 동작에서 이 인대에 추가되는 변형률 비율.
#   추가 변형률 = 기본 변형률 × (하중/체중) × strain_factor
LOAD_TASK_PROFILES = {
    'lift': {  # 들기: 바닥에서 물체를 들어올리기 (예: 장비, 환자)
        'name': 'Lifting',
        'muscle_load_factor': {
            'Quadriceps': 0.35,       # 무릎 신전 ─ 들기의 주요 동원근
            'Hamstrings': 0.20,       # 무릎 굴곡/고관절 신전 보조
            'Gastrocnemius': 0.10,    # 발목 안정화
            'GluteusMax': 0.30,       # 고관절 신전 ─ 들기 핵심
            'ErectorSpinae': 0.45,    # 몸통 신전 ─ 들기 시 최대 부하 ★
            'Deltoid': 0.15,          # 어깨 ─ 물체 잡기
            'Biceps': 0.25,           # 팔꿈치 ─ 물체 당기기
            'LatissimusDorsi': 0.20,  # 등 ─ 물체를 몸쪽으로
        },
        'ligament_strain_factor': {
            'ACL': 0.3, 'PCL': 0.2, 'PatellarTendon': 0.35,
            'SupraspinousLig': 0.4, 'GlenohumeralLig': 0.15,
            'LumbarLig': 0.5,        # 요추 ─ 들기 시 최대 부하 ★
        },
    },
    'pull': {  # 끌기: 호스, 장비 등을 몸 쪽으로 당기기
        'name': 'Pulling',
        'muscle_load_factor': {
            'Quadriceps': 0.25, 'Hamstrings': 0.15, 'Gastrocnemius': 0.15,
            'GluteusMax': 0.20, 'ErectorSpinae': 0.30, 'Deltoid': 0.20,
            'Biceps': 0.35,           # 팔꿈치 굴곡 ─ 당기기 핵심 ★
            'LatissimusDorsi': 0.40,  # 등 ─ 당기기 핵심 ★
        },
        'ligament_strain_factor': {
            'ACL': 0.2, 'PCL': 0.15, 'PatellarTendon': 0.2,
            'SupraspinousLig': 0.25,
            'GlenohumeralLig': 0.35,  # 어깨 인대 ─ 당기기 시 부하 ★
            'LumbarLig': 0.35,
        },
    },
    'carry': {  # 운반: 장비/환자를 들고 이동하기
        'name': 'Carrying',
        'muscle_load_factor': {
            'Quadriceps': 0.30, 'Hamstrings': 0.15,
            'Gastrocnemius': 0.20,    # 보행 + 추가 하중
            'GluteusMax': 0.25, 'ErectorSpinae': 0.35, 'Deltoid': 0.25,
            'Biceps': 0.20, 'LatissimusDorsi': 0.15,
        },
        'ligament_strain_factor': {
            'ACL': 0.25, 'PCL': 0.2, 'PatellarTendon': 0.3,
            'SupraspinousLig': 0.3, 'GlenohumeralLig': 0.2, 'LumbarLig': 0.4,
        },
    },
    'push': {  # 밀기: 장비, 문 등을 앞으로 밀어내기
        'name': 'Pushing',
        'muscle_load_factor': {
            'Quadriceps': 0.30, 'Hamstrings': 0.10, 'Gastrocnemius': 0.20,
            'GluteusMax': 0.25, 'ErectorSpinae': 0.25,
            'Deltoid': 0.35,          # 어깨 ─ 밀기 핵심 ★
            'Biceps': 0.15, 'LatissimusDorsi': 0.10,
        },
        'ligament_strain_factor': {
            'ACL': 0.2, 'PCL': 0.15, 'PatellarTendon': 0.25,
            'SupraspinousLig': 0.2,
            'GlenohumeralLig': 0.3,   # 어깨 인대 ─ 밀기 시 부하 ★
            'LumbarLig': 0.3,
        },
    },
}


# =============================================================================
# 8. 위험도 임계값 (Risk Thresholds)
# =============================================================================
# 근육 스트레스 임계값 (Pa 단위):
#   근육 내부 압력(스트레스) = 총 섬유력 / 생리학적 단면적 (PCSA)
#   100 kPa 이하: 정상 운동 범위
#   250 kPa 이상: 상승된 위험 (반복 시 피로 누적)
#   400 kPa 이상: 높은 위험 (근섬유 미세 손상 가능)
#   600 kPa 이상: 매우 위험 (조직 손상, 급성 부상 가능)
MUSCLE_STRESS_THRESHOLDS = {
    'Low': 100e3,       # 100 kPa ─ 정상 운동 범위 상한
    'Medium': 250e3,    # 250 kPa ─ 피로 누적 시작
    'High': 400e3,      # 400 kPa ─ 미세 손상 위험
    'Critical': 600e3,  # 600 kPa ─ 급성 손상 위험
}

# 인대 변형률 임계값 (무차원, 0~1):
#   인대 변형률 = (현재 길이 - 이완 길이) / 이완 길이
#   3% 이하: 정상 생리학적 범위
#   6%: 발끝→선형 전환점 (인대가 본격적으로 힘을 받기 시작)
#   10%: 파괴 접근 구간
#   15%: 파열 가능 구간
LIGAMENT_STRAIN_THRESHOLDS = {
    'Low': 0.03,        # 3% ─ 정상
    'Medium': 0.06,     # 6% ─ 전환점
    'High': 0.10,       # 10% ─ 파괴 접근
    'Critical': 0.15,   # 15% ─ 파열 위험
}

# 위험도 레벨 순서 (인덱스가 점수로 사용됨)
RISK_LEVELS = ['Normal', 'Low', 'Medium', 'High', 'Critical']

# 근육/인대 → 신체 부위 매핑 (종합 위험도 계산용)
# 같은 신체 부위에 속한 조직들의 위험도 중 최대값이 해당 부위의 위험도가 됩니다.
REGION_MAP = {
    # 근육 → 부위
    'Quadriceps': 'Knee', 'Hamstrings': 'Knee/Thigh', 'Gastrocnemius': 'Ankle/Calf',
    'GluteusMax': 'Hip', 'ErectorSpinae': 'Lumbar', 'Deltoid': 'Shoulder',
    'Biceps': 'Elbow', 'LatissimusDorsi': 'Shoulder/Back',
    # 인대 → 부위
    'ACL': 'Knee', 'PCL': 'Knee', 'PatellarTendon': 'Knee',
    'SupraspinousLig': 'Lumbar', 'GlenohumeralLig': 'Shoulder', 'LumbarLig': 'Lumbar',
}


# =============================================================================
# 9. 디스플레이 설정 (Display Settings)
# =============================================================================
# 실시간 HUD (Heads-Up Display) 창의 레이아웃 설정입니다.
# 화면 구성: [카메라 피드 (640x480)] [사이드 패널 (320px)] / [하단 상태 바 (40px)]
WINDOW_NAME = 'OAK-D Firefighter Pose Analysis'  # OpenCV 창 이름
SIDE_PANEL_WIDTH = 320    # 우측 분석 패널 너비 (px)
BOTTOM_BAR_HEIGHT = 40    # 하단 상태 바 높이 (px)

# 위험도 색상 맵 (BGR 형식 ─ OpenCV는 RGB가 아닌 BGR 사용)
# 각 위험도 레벨에 대응하는 색상으로, 바 차트와 텍스트에 사용됩니다.
RISK_COLORS = {
    'Normal':   (0, 180, 0),     # 초록 ─ 안전
    'Low':      (0, 220, 100),   # 연두 ─ 낮은 위험
    'Medium':   (0, 200, 255),   # 노란/주황 ─ 중간 위험
    'High':     (0, 100, 255),   # 주황 ─ 높은 위험
    'Critical': (0, 0, 255),     # 빨강 ─ 매우 위험
}

# 관절 각도 표시 라벨 (영문)
# cv2.putText는 한글을 지원하지 않으므로 영문 라벨을 사용합니다.
JOINT_LABELS = {
    'Knee_Angle': 'Knee',           # 무릎
    'Hip_Angle': 'Hip',             # 고관절
    'Ankle_Angle': 'Ankle',         # 발목
    'Shoulder_Angle': 'Shoulder',   # 어깨
    'Elbow_Angle': 'Elbow',         # 팔꿈치
    'Trunk_Angle': 'Trunk',         # 몸통
}
