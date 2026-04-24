"""
config.py - 듀얼 카메라 소방관 포즈 분석 시스템 설정 상수 모듈
================================================================

이 파일은 듀얼 카메라(OAK-D-LITE + OAK-D) 기반 소방관 자세 분석 시스템의
모든 설정값을 중앙 관리합니다.

기존 oakd_realtime/config.py 대비 주요 변경:
    - BILATERAL_JOINT_ANGLES: 좌/우 양측 관절 각도 정의 추가
    - MIN_VISIBILITY_THRESHOLD: 관절 가시성 최소 임계값 추가
    - SMOOTHING_WINDOW: 이동 평균 스무딩 윈도우 크기 추가
    - 디스플레이 설정: 듀얼 카메라 레이아웃 대응

구성 섹션:
    1. OAK-D 카메라 설정
    2. MediaPipe Pose 설정
    3. 양측(Bilateral) 분석 설정 (신규)
    4. 분석 설정
    5. 근육 파라미터 (MUSCLE_PARAMS)
    6. 인대 파라미터 (LIGAMENT_PARAMS)
    7. 관절 각도 → 근육/인대 매핑
    8. 하중 작업 프로파일
    9. 위험도 임계값
    10. 디스플레이 설정
"""

import numpy as np


# =============================================================================
# 1. OAK-D 카메라 설정 (OAK-D Camera Settings)
# =============================================================================
# OAK-D의 ColorCamera 노드에서 preview 출력으로 사용할 해상도입니다.
# 640x480은 MediaPipe Pose의 입력으로 적합하며, USB 대역폭을 고려한 최적 크기입니다.
OAKD_RESOLUTION_W = 640   # 가로 해상도 (pixels)
OAKD_RESOLUTION_H = 480   # 세로 해상도 (pixels)
OAKD_FPS = 30             # 초당 프레임 수 (frames per second)
OAKD_PREVIEW_SIZE = (OAKD_RESOLUTION_W, OAKD_RESOLUTION_H)  # (width, height) 튜플

# Stereo Depth 관련 설정
OAKD_ENABLE_DEPTH = False           # 기본적으로 깊이 비활성화
OAKD_DEPTH_ALIGN_TO_RGB = True      # 깊이맵을 RGB 카메라 좌표계로 정렬


# =============================================================================
# 2. MediaPipe Pose 설정 (MediaPipe Pose Settings)
# =============================================================================
# MediaPipe PoseLandmarker의 감지/추적 신뢰도 임계값입니다.
POSE_MODEL_COMPLEXITY = 1            # 0=Lite, 1=Full, 2=Heavy
POSE_MIN_DETECTION_CONF = 0.5       # 포즈 감지 최소 신뢰도
POSE_MIN_TRACKING_CONF = 0.5        # 포즈 추적 최소 신뢰도


# =============================================================================
# 3. 양측(Bilateral) 분석 설정 (신규)
# =============================================================================
# 기존 단일(왼쪽만) 분석 대비, 좌/우 모두 추출하여 visibility 가중 평균으로
# 정확도를 대폭 향상시킵니다. 카메라 각도와 무관하게 안정적인 각도 추정이 가능합니다.
#
# MIN_VISIBILITY_THRESHOLD: 이 값 미만의 visibility를 가진 관절은 무시합니다.
#   - MediaPipe의 visibility는 0~1 범위 (1이면 완전히 보임)
#   - 0.3 미만이면 해당 관절이 가려져서 신뢰할 수 없음
MIN_VISIBILITY_THRESHOLD = 0.3

# SMOOTHING_WINDOW: 이동 평균 스무딩에 사용할 프레임 수
#   - 프레임 간 노이즈(떨림)를 제거하여 안정적인 각도 제공
#   - 5프레임 = 30fps에서 약 167ms (실시간성 유지하면서 노이즈 제거)
SMOOTHING_WINDOW = 5

# BILATERAL_JOINT_ANGLES: 좌/우 양측 관절 각도 정의
# 각 관절에 대해 left/right 랜드마크를 모두 정의합니다.
# 퓨전 시 양쪽의 visibility를 비교하여 가중 평균을 계산합니다.
#
# 형식: {
#   '관절명': {
#       'left': ('점A_left', '점B_left', '점C_left'),   # 왼쪽 관절
#       'right': ('점A_right', '점B_right', '점C_right'), # 오른쪽 관절
#       'label': '표시 라벨',
#       'is_trunk': True/False  # 몸통 특수 계산 여부
#   }
# }
BILATERAL_JOINT_ANGLES = {
    'Knee_Angle': {
        'left': ('left_hip', 'left_knee', 'left_ankle'),
        'right': ('right_hip', 'right_knee', 'right_ankle'),
        'label': 'Knee',
    },
    'Hip_Angle': {
        'left': ('left_shoulder', 'left_hip', 'left_knee'),
        'right': ('right_shoulder', 'right_hip', 'right_knee'),
        'label': 'Hip',
    },
    'Ankle_Angle': {
        'left': ('left_knee', 'left_ankle', 'left_foot_index'),
        'right': ('right_knee', 'right_ankle', 'right_foot_index'),
        'label': 'Ankle',
    },
    'Shoulder_Angle': {
        'left': ('left_hip', 'left_shoulder', 'left_elbow'),
        'right': ('right_hip', 'right_shoulder', 'right_elbow'),
        'label': 'Shoulder',
    },
    'Elbow_Angle': {
        'left': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'label': 'Elbow',
    },
    'Trunk_Angle': {
        'left': ('left_shoulder', 'left_hip', 'left_knee'),
        'right': ('right_shoulder', 'right_hip', 'right_knee'),
        'label': 'Trunk',
        'is_trunk': True,  # 특수 계산: 수직 기준 각도
    },
}


# =============================================================================
# 4. 분석 설정 (Analysis Settings)
# =============================================================================
# 롤링 윈도우: 최근 N 프레임의 관절 각도를 유지합니다.
# 60프레임 = 30fps에서 약 2초 분량입니다.
ROLLING_WINDOW_SIZE = 60             # 버퍼 최대 프레임 수

# 분석 제출 간격: 매 N 프레임마다 백그라운드 생체역학 분석을 요청합니다.
ANALYSIS_INTERVAL = 60               # 분석 제출 간격 (프레임 단위)
ANALYSIS_MAX_WORKERS = 1             # 백그라운드 분석 스레드 수

# 기본 대상자 정보
DEFAULT_BODY_MASS_KG = 75.0          # 기본 체중 (kg)
DEFAULT_LOAD_KG = 0.0                # 기본 외부 하중 (kg)
DEFAULT_TASK_TYPE = 'none'           # 기본 동작 유형

# 신경 흥분(Excitation) 변환 계수
EXCITATION_GAIN = 0.008              # 각속도 → 흥분 변환 이득
EXCITATION_BASE = 0.15               # 기저 흥분 수준


# =============================================================================
# 5. 근육 파라미터 (Muscle Parameters)
# =============================================================================
# OpenSim의 DeGrooteFregly2016Muscle 모델에서 사용하는 근육 생리학적 파라미터입니다.
MUSCLE_PARAMS = {
    'Quadriceps': {
        'name_kr': '대퇴사두근',
        'max_isometric_force': 6000,
        'optimal_fiber_length': 0.084,
        'tendon_slack_length': 0.10,
        'pennation_angle_at_optimal': np.radians(15),
        'pcsa': 75e-4,
    },
    'Hamstrings': {
        'name_kr': '대퇴이두근',
        'max_isometric_force': 900,
        'optimal_fiber_length': 0.11,
        'tendon_slack_length': 0.32,
        'pennation_angle_at_optimal': np.radians(12),
        'pcsa': 18e-4,
    },
    'Gastrocnemius': {
        'name_kr': '비복근',
        'max_isometric_force': 1600,
        'optimal_fiber_length': 0.055,
        'tendon_slack_length': 0.39,
        'pennation_angle_at_optimal': np.radians(17),
        'pcsa': 30e-4,
    },
    'GluteusMax': {
        'name_kr': '대둔근',
        'max_isometric_force': 1500,
        'optimal_fiber_length': 0.14,
        'tendon_slack_length': 0.05,
        'pennation_angle_at_optimal': np.radians(5),
        'pcsa': 40e-4,
    },
    'ErectorSpinae': {
        'name_kr': '척추기립근',
        'max_isometric_force': 2500,
        'optimal_fiber_length': 0.12,
        'tendon_slack_length': 0.03,
        'pennation_angle_at_optimal': np.radians(0),
        'pcsa': 50e-4,
    },
    'Deltoid': {
        'name_kr': '삼각근',
        'max_isometric_force': 1100,
        'optimal_fiber_length': 0.10,
        'tendon_slack_length': 0.04,
        'pennation_angle_at_optimal': np.radians(15),
        'pcsa': 20e-4,
    },
    'Biceps': {
        'name_kr': '상완이두근',
        'max_isometric_force': 600,
        'optimal_fiber_length': 0.116,
        'tendon_slack_length': 0.24,
        'pennation_angle_at_optimal': np.radians(0),
        'pcsa': 12e-4,
    },
    'LatissimusDorsi': {
        'name_kr': '광배근',
        'max_isometric_force': 1200,
        'optimal_fiber_length': 0.25,
        'tendon_slack_length': 0.05,
        'pennation_angle_at_optimal': np.radians(20),
        'pcsa': 25e-4,
    },
}


# =============================================================================
# 6. 인대 파라미터 (Ligament Parameters)
# =============================================================================
# OpenSim의 Blankevoort1991Ligament 모델에서 사용하는 인대 기계적 특성입니다.
LIGAMENT_PARAMS = {
    'ACL': {
        'name_kr': '전방십자인대(ACL)',
        'slack_length': 0.032,
        'linear_stiffness': 5000,
        'transition_strain': 0.06,
        'damping_coefficient': 0.003,
        'estimated_failure_force': 2160,
    },
    'PCL': {
        'name_kr': '후방십자인대(PCL)',
        'slack_length': 0.038,
        'linear_stiffness': 9000,
        'transition_strain': 0.06,
        'damping_coefficient': 0.003,
        'estimated_failure_force': 3000,
    },
    'PatellarTendon': {
        'name_kr': '슬개건',
        'slack_length': 0.050,
        'linear_stiffness': 15000,
        'transition_strain': 0.05,
        'damping_coefficient': 0.005,
        'estimated_failure_force': 10000,
    },
    'SupraspinousLig': {
        'name_kr': '극상인대',
        'slack_length': 0.045,
        'linear_stiffness': 3000,
        'transition_strain': 0.06,
        'damping_coefficient': 0.002,
        'estimated_failure_force': 1500,
    },
    'GlenohumeralLig': {
        'name_kr': '관절와상완인대',
        'slack_length': 0.025,
        'linear_stiffness': 2000,
        'transition_strain': 0.06,
        'damping_coefficient': 0.002,
        'estimated_failure_force': 800,
    },
    'LumbarLig': {
        'name_kr': '요추인대',
        'slack_length': 0.055,
        'linear_stiffness': 4000,
        'transition_strain': 0.07,
        'damping_coefficient': 0.003,
        'estimated_failure_force': 2000,
    },
}


# =============================================================================
# 7. 관절 각도 → 근육/인대 매핑 (Angle-to-Tissue Mapping)
# =============================================================================
ANGLE_TO_MUSCLE = {
    'Quadriceps':      {'joint': 'Knee_Angle',     'theta_range': (60, 180), 'nfl_range': (0.7, 1.15)},
    'Hamstrings':      {'joint': 'Knee_Angle',     'theta_range': (60, 180), 'nfl_range': (1.2, 0.8)},
    'Gastrocnemius':   {'joint': 'Ankle_Angle',    'theta_range': (60, 120), 'nfl_range': (1.15, 0.8)},
    'GluteusMax':      {'joint': 'Hip_Angle',      'theta_range': (90, 180), 'nfl_range': (1.15, 0.85)},
    'ErectorSpinae':   {'joint': 'Trunk_Angle',    'theta_range': (120, 180), 'nfl_range': (1.2, 0.9)},
    'Deltoid':         {'joint': 'Shoulder_Angle', 'theta_range': (0, 180), 'nfl_range': (0.8, 1.2)},
    'Biceps':          {'joint': 'Elbow_Angle',    'theta_range': (40, 170), 'nfl_range': (0.7, 1.2)},
    'LatissimusDorsi': {'joint': 'Shoulder_Angle', 'theta_range': (0, 180), 'nfl_range': (1.15, 0.8)},
}

ANGLE_TO_LIGAMENT = {
    'ACL':              {'joint': 'Knee_Angle',     'theta_range': (60, 180), 'strain_range': (0.01, 0.05)},
    'PCL':              {'joint': 'Knee_Angle',     'theta_range': (60, 180), 'strain_range': (0.05, 0.01)},
    'PatellarTendon':   {'joint': 'Knee_Angle',     'theta_range': (60, 180), 'strain_range': (0.06, 0.02)},
    'SupraspinousLig':  {'joint': 'Trunk_Angle',    'theta_range': (120, 180), 'strain_range': (0.06, 0.01)},
    'GlenohumeralLig':  {'joint': 'Shoulder_Angle', 'theta_range': (0, 180), 'strain_range': (0.01, 0.06)},
    'LumbarLig':        {'joint': 'Trunk_Angle',    'theta_range': (120, 180), 'strain_range': (0.08, 0.01)},
}


# =============================================================================
# 8. 하중 작업 프로파일 (Load Task Profiles)
# =============================================================================
LOAD_TASK_PROFILES = {
    'lift': {
        'name': 'Lifting',
        'muscle_load_factor': {
            'Quadriceps': 0.35, 'Hamstrings': 0.20, 'Gastrocnemius': 0.10,
            'GluteusMax': 0.30, 'ErectorSpinae': 0.45, 'Deltoid': 0.15,
            'Biceps': 0.25, 'LatissimusDorsi': 0.20,
        },
        'ligament_strain_factor': {
            'ACL': 0.3, 'PCL': 0.2, 'PatellarTendon': 0.35,
            'SupraspinousLig': 0.4, 'GlenohumeralLig': 0.15, 'LumbarLig': 0.5,
        },
    },
    'pull': {
        'name': 'Pulling',
        'muscle_load_factor': {
            'Quadriceps': 0.25, 'Hamstrings': 0.15, 'Gastrocnemius': 0.15,
            'GluteusMax': 0.20, 'ErectorSpinae': 0.30, 'Deltoid': 0.20,
            'Biceps': 0.35, 'LatissimusDorsi': 0.40,
        },
        'ligament_strain_factor': {
            'ACL': 0.2, 'PCL': 0.15, 'PatellarTendon': 0.2,
            'SupraspinousLig': 0.25, 'GlenohumeralLig': 0.35, 'LumbarLig': 0.35,
        },
    },
    'carry': {
        'name': 'Carrying',
        'muscle_load_factor': {
            'Quadriceps': 0.30, 'Hamstrings': 0.15, 'Gastrocnemius': 0.20,
            'GluteusMax': 0.25, 'ErectorSpinae': 0.35, 'Deltoid': 0.25,
            'Biceps': 0.20, 'LatissimusDorsi': 0.15,
        },
        'ligament_strain_factor': {
            'ACL': 0.25, 'PCL': 0.2, 'PatellarTendon': 0.3,
            'SupraspinousLig': 0.3, 'GlenohumeralLig': 0.2, 'LumbarLig': 0.4,
        },
    },
    'push': {
        'name': 'Pushing',
        'muscle_load_factor': {
            'Quadriceps': 0.30, 'Hamstrings': 0.10, 'Gastrocnemius': 0.20,
            'GluteusMax': 0.25, 'ErectorSpinae': 0.25, 'Deltoid': 0.35,
            'Biceps': 0.15, 'LatissimusDorsi': 0.10,
        },
        'ligament_strain_factor': {
            'ACL': 0.2, 'PCL': 0.15, 'PatellarTendon': 0.25,
            'SupraspinousLig': 0.2, 'GlenohumeralLig': 0.3, 'LumbarLig': 0.3,
        },
    },
}


# =============================================================================
# 9. 위험도 임계값 (Risk Thresholds)
# =============================================================================
# 근육 스트레스 임계값 (Pa 단위)
MUSCLE_STRESS_THRESHOLDS = {
    'Low': 100e3,       # 100 kPa
    'Medium': 250e3,    # 250 kPa
    'High': 400e3,      # 400 kPa
    'Critical': 600e3,  # 600 kPa
}

# 인대 변형률 임계값 (무차원)
LIGAMENT_STRAIN_THRESHOLDS = {
    'Low': 0.03,        # 3%
    'Medium': 0.06,     # 6%
    'High': 0.10,       # 10%
    'Critical': 0.15,   # 15%
}

# 위험도 레벨 순서
RISK_LEVELS = ['Normal', 'Low', 'Medium', 'High', 'Critical']

# 근육/인대 → 신체 부위 매핑
REGION_MAP = {
    'Quadriceps': 'Knee', 'Hamstrings': 'Knee/Thigh', 'Gastrocnemius': 'Ankle/Calf',
    'GluteusMax': 'Hip', 'ErectorSpinae': 'Lumbar', 'Deltoid': 'Shoulder',
    'Biceps': 'Elbow', 'LatissimusDorsi': 'Shoulder/Back',
    'ACL': 'Knee', 'PCL': 'Knee', 'PatellarTendon': 'Knee',
    'SupraspinousLig': 'Lumbar', 'GlenohumeralLig': 'Shoulder', 'LumbarLig': 'Lumbar',
}


# =============================================================================
# 10. 디스플레이 설정 (Display Settings)
# =============================================================================
# 듀얼 카메라 HUD 레이아웃
# [카메라1 (640x480)] [카메라2 (640x480)] [사이드 패널 (320px)]
# [                    하단 상태 바 (40px)                      ]
WINDOW_NAME = 'Dual-Camera Firefighter Pose Analysis'
SIDE_PANEL_WIDTH = 320    # 우측 분석 패널 너비 (px)
BOTTOM_BAR_HEIGHT = 40    # 하단 상태 바 높이 (px)

# 위험도 색상 맵 (BGR)
RISK_COLORS = {
    'Normal':   (0, 180, 0),     # 초록
    'Low':      (0, 220, 100),   # 연두
    'Medium':   (0, 200, 255),   # 주황
    'High':     (0, 100, 255),   # 진주황
    'Critical': (0, 0, 255),     # 빨강
}

# 관절 각도 표시 라벨 (영문 - cv2.putText는 한글 미지원)
JOINT_LABELS = {
    'Knee_Angle': 'Knee',
    'Hip_Angle': 'Hip',
    'Ankle_Angle': 'Ankle',
    'Shoulder_Angle': 'Shoulder',
    'Elbow_Angle': 'Elbow',
    'Trunk_Angle': 'Trunk',
}
