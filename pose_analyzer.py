"""
MediaPipe Pose Wrapper - Uses MediaPipe Tasks API (0.10.33+)
Detects 33 body landmarks and computes 6 joint angles per frame.

==============================================================================
[모듈 목적 및 역할]
==============================================================================
이 모듈은 MediaPipe Tasks API를 사용하여 실시간 영상에서 인체의 33개 랜드마크를
검출하고, 이를 기반으로 6개의 주요 관절 각도를 계산하는 래퍼(wrapper) 클래스를
제공합니다.

본 모듈은 biotech_cam2/video_injury_predictor.py 프로젝트와 연계되어 동작합니다.
video_injury_predictor.py가 녹화된 영상을 분석하여 부상 위험도를 예측하는 반면,
이 모듈(pose_analyzer.py)은 OAK-D 카메라를 통한 실시간 포즈 분석을 담당합니다.
두 모듈 모두 동일한 관절 각도 정의(무릎, 엉덩이, 발목, 어깨, 팔꿈치, 몸통)를
사용하므로, 실시간 분석 결과와 사후 영상 분석 결과를 직접 비교할 수 있습니다.

==============================================================================
[MediaPipe Tasks API vs Legacy mp.solutions API 마이그레이션 설명]
==============================================================================
기존의 mp.solutions.pose API (일명 "legacy API")는 MediaPipe 0.10.x 이후
공식적으로 deprecated(폐지 예정) 상태입니다.

Legacy API 방식 (더 이상 권장하지 않음):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, ...)
    results = pose.process(rgb_frame)

Tasks API 방식 (현재 권장, 이 모듈에서 사용):
    options = PoseLandmarkerOptions(base_options=BaseOptions(...), ...)
    landmarker = PoseLandmarker.create_from_options(options)
    result = landmarker.detect_for_video(mp_image, timestamp_ms=...)

Tasks API의 장점:
1. 모델 파일(.task)을 직접 로드하여 버전 관리가 용이함
2. RunningMode(IMAGE, VIDEO, LIVE_STREAM)를 명시적으로 지정 가능
3. GPU 가속, 모델 교체 등 확장성이 뛰어남
4. 멀티 포즈 검출(num_poses) 파라미터 지원
5. 랜드마크 신뢰도(presence_confidence) 별도 설정 가능
==============================================================================
"""

import os
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, Optional, Tuple

# config.py에서 포즈 검출 관련 설정값을 가져옴
# - POSE_MIN_DETECTION_CONF: 포즈 검출 최소 신뢰도 (기본값 0.5 수준)
# - POSE_MIN_TRACKING_CONF: 포즈 추적 최소 신뢰도 (기본값 0.5 수준)
# - JOINT_LABELS: 관절 각도 이름을 화면 표시용 짧은 라벨로 매핑하는 딕셔너리
from config import (
    POSE_MIN_DETECTION_CONF, POSE_MIN_TRACKING_CONF,
    JOINT_LABELS,
)

# ==============================================================================
# MediaPipe Tasks API 네임스페이스 축약
# ==============================================================================
# mp.tasks.vision: PoseLandmarker, PoseLandmarkerOptions, RunningMode 등 포함
# mp.tasks.BaseOptions: 모델 파일 경로 또는 바이너리 데이터를 지정하는 기본 옵션
_vision = mp.tasks.vision
_BaseOptions = mp.tasks.BaseOptions

# ==============================================================================
# [모델 파일 경로 설정 및 바이너리 로딩 이유]
# ==============================================================================
# pose_landmarker_full.task 파일은 MediaPipe의 전체 포즈 랜드마커 모델입니다.
# "full" 모델은 "lite" 모델보다 정확도가 높지만 연산량이 더 많습니다.
#
# 중요: 모델 파일을 경로(path)로 직접 전달하지 않고, 바이트(bytes)로 읽어서
# model_asset_buffer로 전달하는 이유:
#
# Windows 환경에서 파일 경로에 한글, 특수문자, 유니코드가 포함된 경우
# (예: "F:\++++++++++++++++++ 신규사업\1_1_1_develop_pro\...")
# MediaPipe의 내부 C++ 바인딩(protobuf/TFLite)이 유니코드 경로를 올바르게
# 처리하지 못하여 FileNotFoundError 또는 인코딩 오류가 발생할 수 있습니다.
#
# 이를 우회하기 위해 Python의 open() 함수로 먼저 파일을 바이너리로 읽은 후,
# model_asset_buffer 파라미터를 통해 메모리에서 직접 모델을 로드합니다.
# 이 방식은 경로에 어떤 문자가 포함되어 있든 안정적으로 동작합니다.
# ==============================================================================
_MODEL_FILENAME = 'pose_landmarker_full.task'
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODEL_FILENAME)


class PoseAnalyzer:
    """
    MediaPipe PoseLandmarker 기반 관절 각도 추출기.

    이 클래스는 다음 기능을 제공합니다:
    1. MediaPipe Tasks API를 사용한 실시간 포즈 랜드마크 검출 (33개 키포인트)
    2. 검출된 랜드마크로부터 6개 주요 관절 각도 계산
    3. 프레임 위에 스켈레톤(골격) 및 각도 라벨 오버레이 시각화

    사용 예시:
        analyzer = PoseAnalyzer()
        angles, annotated_frame = analyzer.process_frame(bgr_frame)
        if angles is not None:
            print(f"무릎 각도: {angles['Knee_Angle']:.1f}도")
        analyzer.close()
    """

    # ==========================================================================
    # [MediaPipe 33개 랜드마크 인덱스 매핑]
    # ==========================================================================
    # MediaPipe Pose는 총 33개의 신체 랜드마크를 검출합니다.
    # 각 랜드마크는 정규화된 3D 좌표 (x, y, z)를 가집니다:
    #   - x: 이미지 너비 기준 0.0~1.0 (왼쪽→오른쪽)
    #   - y: 이미지 높이 기준 0.0~1.0 (위→아래)
    #   - z: 깊이 (엉덩이 중심 기준, 카메라에 가까울수록 음수)
    #
    # 전체 33개 랜드마크 목록 (본 클래스에서 사용하는 것에 * 표시):
    #   [0]  코(nose) *
    #   [1]  왼쪽 눈 안쪽 (left_eye_inner)
    #   [2]  왼쪽 눈 (left_eye)
    #   [3]  왼쪽 눈 바깥쪽 (left_eye_outer)
    #   [4]  오른쪽 눈 안쪽 (right_eye_inner)
    #   [5]  오른쪽 눈 (right_eye)
    #   [6]  오른쪽 눈 바깥쪽 (right_eye_outer)
    #   [7]  왼쪽 귀 (left_ear)
    #   [8]  오른쪽 귀 (right_ear)
    #   [9]  입 왼쪽 (mouth_left)
    #   [10] 입 오른쪽 (mouth_right)
    #   [11] 왼쪽 어깨 (left_shoulder) *
    #   [12] 오른쪽 어깨 (right_shoulder) *
    #   [13] 왼쪽 팔꿈치 (left_elbow) *
    #   [14] 오른쪽 팔꿈치 (right_elbow) *
    #   [15] 왼쪽 손목 (left_wrist) *
    #   [16] 오른쪽 손목 (right_wrist) *
    #   [17] 왼쪽 새끼손가락 (left_pinky)
    #   [18] 오른쪽 새끼손가락 (right_pinky)
    #   [19] 왼쪽 검지 (left_index)
    #   [20] 오른쪽 검지 (right_index)
    #   [21] 왼쪽 엄지 (left_thumb)
    #   [22] 오른쪽 엄지 (right_thumb)
    #   [23] 왼쪽 엉덩이/골반 (left_hip) *
    #   [24] 오른쪽 엉덩이/골반 (right_hip) *
    #   [25] 왼쪽 무릎 (left_knee) *
    #   [26] 오른쪽 무릎 (right_knee) *
    #   [27] 왼쪽 발목 (left_ankle) *
    #   [28] 오른쪽 발목 (right_ankle) *
    #   [29] 왼쪽 뒤꿈치 (left_heel) *
    #   [30] 오른쪽 뒤꿈치 (right_heel) *
    #   [31] 왼쪽 발끝/엄지발가락 (left_foot_index) *
    #   [32] 오른쪽 발끝/엄지발가락 (right_foot_index) *
    #
    # 본 분석에서는 관절 각도 계산에 필요한 주요 랜드마크만 선별하여 사용합니다.
    # 얼굴 관련 랜드마크(1~10)와 손가락 랜드마크(17~22)는 관절 각도 계산에
    # 직접 사용되지 않으므로 제외하였습니다.
    # ==========================================================================
    LANDMARKS = {
        'nose': 0,                  # 코 - 얼굴 방향 참조용
        'left_shoulder': 11,        # 왼쪽 어깨 - 어깨각도, 몸통각도 계산에 사용
        'right_shoulder': 12,       # 오른쪽 어깨 - 양측 비교 시 사용 가능
        'left_elbow': 13,           # 왼쪽 팔꿈치 - 팔꿈치각도 계산의 꼭짓점
        'right_elbow': 14,          # 오른쪽 팔꿈치
        'left_wrist': 15,           # 왼쪽 손목 - 팔꿈치각도 계산의 끝점
        'right_wrist': 16,          # 오른쪽 손목
        'left_hip': 23,             # 왼쪽 엉덩이(골반) - 엉덩이각도 계산의 꼭짓점
        'right_hip': 24,            # 오른쪽 엉덩이(골반)
        'left_knee': 25,            # 왼쪽 무릎 - 무릎각도 계산의 꼭짓점
        'right_knee': 26,           # 오른쪽 무릎
        'left_ankle': 27,           # 왼쪽 발목 - 발목각도 계산의 꼭짓점
        'right_ankle': 28,          # 오른쪽 발목
        'left_heel': 29,            # 왼쪽 뒤꿈치 - 발 방향 참조
        'right_heel': 30,           # 오른쪽 뒤꿈치
        'left_foot_index': 31,      # 왼쪽 발끝 - 발목각도 계산의 끝점
        'right_foot_index': 32,     # 오른쪽 발끝
    }

    # ==========================================================================
    # [6개 관절 각도 정의 - 생체역학적(biomechanical) 의미]
    # ==========================================================================
    # 관절 각도는 3개의 랜드마크(점 A, B, C)로 정의됩니다.
    # 중간 점(B)이 관절의 꼭짓점이 되며, A-B-C가 이루는 각도를 측정합니다.
    #
    # 각 관절 각도의 생체역학적 의미:
    #
    # 1. Knee_Angle (무릎 각도):
    #    - 점: 엉덩이(hip) → 무릎(knee) → 발목(ankle)
    #    - 의미: 무릎의 굴곡(flexion)/신전(extension) 정도
    #    - 180도 = 다리가 완전히 펴진 상태
    #    - 90도 이하 = 깊은 스쿼트 또는 무릎 과굴곡 (부상 위험 증가)
    #    - 소방관의 무릎 부상 위험 평가에 핵심 지표
    #
    # 2. Hip_Angle (엉덩이/고관절 각도):
    #    - 점: 어깨(shoulder) → 엉덩이(hip) → 무릎(knee)
    #    - 의미: 고관절의 굴곡(flexion)/신전(extension) 정도
    #    - 180도 = 상체와 하체가 일직선 (직립 자세)
    #    - 90도 = 상체가 90도 앞으로 숙여진 상태
    #    - 허리 부상 위험과 밀접한 관련
    #
    # 3. Ankle_Angle (발목 각도):
    #    - 점: 무릎(knee) → 발목(ankle) → 발끝(foot_index)
    #    - 의미: 발목의 배굴(dorsiflexion)/저굴(plantarflexion)
    #    - 90도 = 정강이와 발이 직각
    #    - 발목 가동범위(ROM) 평가에 사용
    #
    # 4. Shoulder_Angle (어깨 각도):
    #    - 점: 엉덩이(hip) → 어깨(shoulder) → 팔꿈치(elbow)
    #    - 의미: 어깨의 굴곡(flexion)/외전(abduction) 정도
    #    - 0도 = 팔이 몸통에 붙어있는 상태
    #    - 180도 = 팔을 머리 위로 완전히 올린 상태
    #    - 소방관이 장비를 들어올릴 때의 어깨 부하 평가
    #
    # 5. Elbow_Angle (팔꿈치 각도):
    #    - 점: 어깨(shoulder) → 팔꿈치(elbow) → 손목(wrist)
    #    - 의미: 팔꿈치의 굴곡(flexion)/신전(extension)
    #    - 180도 = 팔이 완전히 펴진 상태
    #    - 90도 = 팔꿈치가 직각으로 구부러진 상태
    #
    # 6. Trunk_Angle (몸통/체간 각도) - 특수 계산:
    #    - 점: 어깨(shoulder) → 엉덩이(hip) → 무릎(knee) (정의상)
    #    - 실제 계산: 어깨-엉덩이 벡터와 수직선(vertical) 사이의 각도
    #    - 의미: 상체의 전방 기울기(forward lean) 정도
    #    - 0도에 가까움 = 과도한 전방 굴곡 (허리 부상 고위험)
    #    - 180도에 가까움 = 완전 직립 (몸통이 수직)
    #    - 소방관이 호스를 끌거나 무거운 장비를 운반할 때 중요
    #
    # 참고: 현재는 왼쪽(left) 측 랜드마크만 사용합니다.
    # 카메라가 피험자의 측면(sagittal plane)에서 촬영한다고 가정합니다.
    # 양측 분석이 필요한 경우 right_* 랜드마크를 추가하여 확장 가능합니다.
    # ==========================================================================
    JOINT_ANGLES = {
        'Knee_Angle': {
            'points': ('left_hip', 'left_knee', 'left_ankle'),
            'label': 'Knee',
        },
        'Hip_Angle': {
            'points': ('left_shoulder', 'left_hip', 'left_knee'),
            'label': 'Hip',
        },
        'Ankle_Angle': {
            'points': ('left_knee', 'left_ankle', 'left_foot_index'),
            'label': 'Ankle',
        },
        'Shoulder_Angle': {
            'points': ('left_hip', 'left_shoulder', 'left_elbow'),
            'label': 'Shoulder',
        },
        'Elbow_Angle': {
            'points': ('left_shoulder', 'left_elbow', 'left_wrist'),
            'label': 'Elbow',
        },
        'Trunk_Angle': {
            'points': ('left_shoulder', 'left_hip', 'left_knee'),
            'label': 'Trunk',
            'is_trunk': True,  # 특수 계산 플래그: 수직 기준 각도 사용
        },
    }

    # ==========================================================================
    # [스켈레톤(골격) 연결선 정의]
    # ==========================================================================
    # MediaPipe Tasks API에서 제공하는 표준 포즈 연결선(connections) 정보입니다.
    # 이 연결선은 랜드마크 간의 뼈대(bone)를 나타내며,
    # 시각화 시 랜드마크 점들을 선분으로 연결하여 인체 스켈레톤을 표현합니다.
    # 예: (11, 13) = 왼쪽 어깨와 왼쪽 팔꿈치를 연결하는 위팔뼈(상완골)
    #
    # Legacy API에서는 mp.solutions.pose.POSE_CONNECTIONS를 사용했지만,
    # Tasks API에서는 _vision.PoseLandmarksConnections.POSE_LANDMARKS를 사용합니다.
    # ==========================================================================
    _POSE_CONNECTIONS = _vision.PoseLandmarksConnections.POSE_LANDMARKS

    def __init__(self, min_detection_confidence=None, min_tracking_confidence=None):
        """
        PoseAnalyzer 초기화.

        MediaPipe PoseLandmarker를 생성하고 VIDEO 모드로 설정합니다.

        Parameters
        ----------
        min_detection_confidence : float, optional
            포즈 검출 최소 신뢰도 (0.0~1.0).
            None이면 config.py의 POSE_MIN_DETECTION_CONF 사용.
            이 값이 높을수록 오탐(false positive)은 줄지만 미탐(miss)이 증가.

        min_tracking_confidence : float, optional
            포즈 추적 최소 신뢰도 (0.0~1.0).
            None이면 config.py의 POSE_MIN_TRACKING_CONF 사용.
            이 값이 높을수록 추적 안정성은 높지만,
            빠른 동작 시 추적이 끊어질 수 있음.
        """
        # 설정값이 명시적으로 전달되지 않으면 config.py의 기본값을 사용
        det = min_detection_confidence if min_detection_confidence is not None else POSE_MIN_DETECTION_CONF
        trk = min_tracking_confidence if min_tracking_confidence is not None else POSE_MIN_TRACKING_CONF

        # ==================================================================
        # [모델 파일을 바이트(bytes)로 로딩 - 유니코드 경로 우회]
        # ==================================================================
        # Windows에서 한글/특수문자가 포함된 경로에서 MediaPipe가 모델 파일을
        # 직접 읽지 못하는 문제를 우회하기 위해 Python의 open()으로 먼저
        # 바이너리를 읽어 메모리에 올립니다.
        #
        # model_asset_path (경로 전달) 대신 model_asset_buffer (바이트 전달)를
        # 사용하면 MediaPipe 내부의 C++ 파일 시스템 호출을 완전히 우회합니다.
        # ==================================================================
        with open(_MODEL_PATH, 'rb') as f:
            model_data = f.read()

        # ==================================================================
        # [PoseLandmarkerOptions 설정]
        # ==================================================================
        # running_mode 옵션 설명:
        #   - IMAGE: 단일 이미지 처리 (detect() 메서드 사용)
        #   - VIDEO: 비디오 프레임 순차 처리 (detect_for_video() 메서드 사용)
        #            → 반드시 단조 증가하는 timestamp_ms를 제공해야 함
        #            → 내부적으로 이전 프레임 정보를 활용한 추적(tracking) 수행
        #   - LIVE_STREAM: 비동기 콜백 기반 실시간 처리 (detect_async() 메서드 사용)
        #
        # VIDEO 모드를 선택한 이유:
        # - OAK-D 카메라에서 프레임을 동기적으로 가져오므로 VIDEO 모드가 적합
        # - 이전 프레임의 포즈 정보를 활용하여 추적 성능이 향상됨
        # - LIVE_STREAM 모드는 콜백 기반이라 프레임 처리 순서 보장이 어려움
        #
        # timestamp_ms 요구사항:
        # - VIDEO 모드에서는 각 프레임에 대해 밀리초 단위의 타임스탬프를 제공해야 함
        # - 타임스탬프는 반드시 단조 증가(monotonically increasing)해야 함
        # - 동일하거나 감소하는 타임스탬프를 전달하면 에러 발생
        # - 30fps 기준으로 약 33ms 간격을 사용 (1000ms / 30fps ≈ 33ms)
        #
        # num_poses=1:
        # - 한 프레임에서 최대 1명의 포즈만 검출
        # - 소방관 개인의 자세 분석이 목적이므로 1명으로 제한
        # - 다중 인원 분석이 필요하면 이 값을 증가시킬 수 있음
        #
        # min_pose_detection_confidence:
        # - 포즈 검출기(detector)의 최소 신뢰도
        # - 새로운 포즈를 최초 감지할 때 적용되는 임계값
        #
        # min_pose_presence_confidence:
        # - 포즈 존재 여부 판단의 최소 신뢰도
        # - 이 값 미만이면 "포즈가 없다"고 판단하고 다시 검출기를 실행
        #
        # min_tracking_confidence:
        # - 프레임 간 랜드마크 추적의 최소 신뢰도
        # - 이 값 미만이면 추적을 포기하고 검출기를 다시 실행
        # ==================================================================
        options = _vision.PoseLandmarkerOptions(
            base_options=_BaseOptions(model_asset_buffer=model_data),
            running_mode=_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=det,
            min_pose_presence_confidence=det,
            min_tracking_confidence=trk,
        )
        # PoseLandmarker 인스턴스 생성
        # create_from_options()는 모델을 로드하고 추론 엔진을 초기화함
        self._landmarker = _vision.PoseLandmarker.create_from_options(options)

        # 스켈레톤 시각화를 위한 드로잉 유틸리티
        # Tasks API에서는 mp.tasks.vision.drawing_utils를 사용
        # (Legacy API의 mp.solutions.drawing_utils와는 다른 모듈)
        self._draw_utils = _vision.drawing_utils

        # ==================================================================
        # [프레임 타임스탬프 카운터]
        # ==================================================================
        # VIDEO 모드에서 detect_for_video()를 호출할 때마다
        # 단조 증가하는 타임스탬프를 제공해야 합니다.
        # 실제 카메라 타임스탬프 대신 33ms씩 증가하는 가상 타임스탬프를 사용합니다.
        # 이는 30fps를 가정한 값이며, 실제 FPS와 정확히 일치할 필요는 없습니다.
        # MediaPipe 내부에서는 이 값을 프레임 순서 판단에만 사용합니다.
        # ==================================================================
        self._frame_ts_ms = 0

    @staticmethod
    def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        세 점 A, B, C로 정의되는 관절 각도를 계산합니다.

        수학적 원리 (벡터 내적과 역코사인):
        ==================================
        벡터 BA = A - B (꼭짓점 B에서 A를 향하는 벡터)
        벡터 BC = C - B (꼭짓점 B에서 C를 향하는 벡터)

        두 벡터 사이의 각도 theta는 내적(dot product) 공식으로 계산:
            cos(theta) = (BA . BC) / (|BA| * |BC|)

        여기서:
            BA . BC = BA_x*BC_x + BA_y*BC_y + BA_z*BC_z  (내적/스칼라곱)
            |BA| = sqrt(BA_x^2 + BA_y^2 + BA_z^2)        (벡터 크기/노름)

        그런 다음 역코사인(arccos)으로 각도를 구합니다:
            theta = arccos(cos(theta))

        결과는 라디안에서 도(degrees)로 변환됩니다.

        수치 안정성 처리:
        ================
        - 분모에 1e-10을 더하여 제로 디비전(0으로 나누기) 방지
          → 두 점이 겹쳐서 벡터 크기가 0이 되는 극단적 경우 대비
        - np.clip으로 cos 값을 [-1.0, 1.0] 범위로 클리핑
          → 부동소수점 오차로 인해 1.0을 미세하게 초과할 수 있음
          → arccos의 정의역은 [-1, 1]이므로 범위 밖 값은 NaN 발생

        Parameters
        ----------
        a : np.ndarray
            첫 번째 점의 3D 좌표 [x, y, z] (관절의 한쪽 끝)
        b : np.ndarray
            꼭짓점(vertex)의 3D 좌표 [x, y, z] (각도를 측정하는 관절 중심)
        c : np.ndarray
            세 번째 점의 3D 좌표 [x, y, z] (관절의 다른 쪽 끝)

        Returns
        -------
        float
            계산된 관절 각도 (도, degrees). 범위: 0도 ~ 180도
            - 0도 = 세 점이 접혀있는 상태 (A와 C가 같은 방향)
            - 180도 = 세 점이 일직선 (A-B-C가 완전히 펴진 상태)
        """
        # 꼭짓점 B에서 A, C를 향하는 두 벡터 계산
        ba = a - b  # B → A 방향 벡터
        bc = c - b  # B → C 방향 벡터

        # 코사인 유사도 계산 (내적 / 두 벡터 크기의 곱)
        # 1e-10은 벡터 크기가 0일 때(두 점이 동일 위치) 나눗셈 오류 방지
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)

        # 부동소수점 오차로 인해 [-1, 1] 범위를 벗어나는 것을 방지
        # arccos 함수는 입력이 [-1, 1] 범위 밖이면 NaN을 반환하므로 필수
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # arccos로 라디안 각도를 구한 후, degrees로 변환하여 반환
        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def calc_trunk_angle(shoulder: np.ndarray, hip: np.ndarray) -> float:
        """
        몸통(trunk)의 전방 기울기 각도를 계산합니다.

        일반 관절 각도와 달리, 몸통 각도는 "수직선(vertical)"을 기준으로 합니다.
        이는 절대적인 자세 기울기를 측정하기 위함입니다.

        계산 원리:
        =========
        1. 몸통 벡터(trunk_vec) = 어깨 좌표 - 엉덩이 좌표
           → 엉덩이에서 어깨를 향하는 방향 = 몸통의 방향

        2. 수직 기준 벡터(vertical) = [0, -1, 0]
           → MediaPipe 좌표계에서 Y축은 아래로 증가하므로,
             위를 향하는 벡터는 [0, -1, 0]입니다.

        3. 코사인 각도 = (trunk_vec . vertical) / |trunk_vec|
           → 수직 벡터의 크기는 1이므로 분모에서 생략 가능

        4. 최종 각도 = 180도 - arccos(cos_angle)
           → 직립 시: trunk_vec이 [0,-1,0]과 같은 방향 → cos=1 → arccos=0 → 180-0=180
           → 완전 전방 굴곡(수평): trunk_vec이 수직과 90도 → arccos=90 → 180-90=90
           → 값이 작을수록 몸통이 앞으로 기울어진 상태를 의미

        왜 세 점이 아닌 두 점만 사용하는가:
        ================================
        일반 관절(무릎, 팔꿈치 등)은 인접한 세 점으로 상대적 각도를 측정하지만,
        몸통 기울기는 중력 방향(수직선)에 대한 절대적 각도이므로
        어깨와 엉덩이 두 점만으로 몸통 벡터를 정의한 후
        가상의 수직선과 비교합니다.

        Parameters
        ----------
        shoulder : np.ndarray
            어깨의 3D 좌표 [x, y, z]
        hip : np.ndarray
            엉덩이(골반)의 3D 좌표 [x, y, z]

        Returns
        -------
        float
            몸통 전방 기울기 각도 (도, degrees).
            - 180도에 가까움 = 완전 직립 자세 (몸통이 수직)
            - 90도 = 상체가 완전히 수평으로 숙여진 상태
            - 값이 작을수록 과도한 전방 굴곡으로 허리 부상 위험 증가
        """
        # 엉덩이에서 어깨를 향하는 벡터 = 몸통의 방향 벡터
        trunk_vec = shoulder - hip

        # 수직 기준 벡터 (MediaPipe 좌표계에서 "위" 방향)
        # MediaPipe에서 y좌표는 위→아래로 증가하므로,
        # 위를 향하는 단위 벡터는 [0, -1, 0]
        vertical = np.array([0, -1, 0])

        # 몸통 벡터와 수직 벡터 사이의 코사인 값
        # vertical의 노름이 1이므로 분모는 trunk_vec의 노름만 필요
        cos_angle = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # 180도에서 빼는 변환:
        # - 직립 시: cos_angle ≈ 1 → arccos ≈ 0도 → 180 - 0 = 180도 (직립)
        # - 전방 90도 굴곡: cos_angle ≈ 0 → arccos ≈ 90도 → 180 - 90 = 90도
        # 이 변환을 통해 "직립=180, 전방굴곡=감소"라는 직관적 해석이 가능
        return 180.0 - float(np.degrees(np.arccos(cos_angle)))

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[Dict[str, float]], np.ndarray]:
        """
        단일 비디오 프레임을 처리하여 관절 각도와 시각화 이미지를 반환합니다.

        처리 파이프라인:
        1. BGR → RGB 색공간 변환 (MediaPipe는 RGB 입력을 기대)
        2. MediaPipe Image 객체 생성
        3. PoseLandmarker로 포즈 검출 (VIDEO 모드, 타임스탬프 증가)
        4. 검출된 랜드마크에서 3D 좌표 추출
        5. 6개 관절 각도 계산
        6. 스켈레톤 및 각도 라벨을 프레임에 오버레이
        7. (각도 딕셔너리, 시각화 프레임) 튜플 반환

        Parameters
        ----------
        frame : np.ndarray
            BGR 형식의 입력 프레임 (OpenCV 기본 색공간).
            shape: (height, width, 3), dtype: uint8

        Returns
        -------
        Tuple[Optional[Dict[str, float]], np.ndarray]
            - angles: 6개 관절 각도 딕셔너리 {'Knee_Angle': float, ...}
                      포즈가 검출되지 않으면 None
            - annotated: 스켈레톤과 각도 라벨이 그려진 프레임 사본
                         포즈 미검출 시에도 원본 복사본 반환 (빈 프레임 아님)
        """
        # ==================================================================
        # [색공간 변환: BGR → RGB]
        # ==================================================================
        # OpenCV는 기본적으로 BGR 형식을 사용하지만,
        # MediaPipe Tasks API는 RGB 형식의 입력을 기대합니다.
        # 이 변환을 생략하면 색상 채널이 뒤바뀌어 검출 성능이 저하됩니다.
        # ==================================================================
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ==================================================================
        # [MediaPipe Image 객체 생성]
        # ==================================================================
        # Tasks API는 자체 Image 클래스를 사용합니다.
        # SRGB 포맷을 지정하여 올바른 색공간 해석을 보장합니다.
        # data 파라미터에 NumPy 배열을 직접 전달합니다.
        # ==================================================================
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ==================================================================
        # [타임스탬프 증가 및 포즈 검출]
        # ==================================================================
        # VIDEO 모드에서는 timestamp_ms가 반드시 이전 호출보다 커야 합니다.
        # 33ms 증가는 약 30fps를 가정한 값입니다.
        # 실제 프레임 레이트와 정확히 맞출 필요는 없으며,
        # 중요한 것은 "단조 증가(monotonically increasing)"라는 조건입니다.
        #
        # detect_for_video()는 내부적으로:
        # 1. 첫 프레임이거나 추적 실패 시: 전체 이미지에서 포즈 검출(detection)
        # 2. 이전 프레임 포즈가 있으면: 이전 위치 근처에서 추적(tracking)
        # → 추적 모드가 검출 모드보다 훨씬 빠름 (매 프레임 전체 탐색 불필요)
        # ==================================================================
        self._frame_ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms=self._frame_ts_ms)

        # 원본 프레임의 복사본을 생성 (시각화용, 원본 훼손 방지)
        annotated = frame.copy()

        # ==================================================================
        # [포즈 미검출 처리]
        # ==================================================================
        # result.pose_landmarks가 비어있으면 프레임에서 사람이 감지되지 않은 것.
        # 이 경우 각도는 None, 시각화는 원본 그대로 반환합니다.
        # 호출측에서 None 체크 후 해당 프레임을 건너뛰거나 기본값 처리 가능.
        # ==================================================================
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None, annotated

        # ==================================================================
        # [랜드마크 좌표 추출]
        # ==================================================================
        # result.pose_landmarks[0]: 첫 번째(유일한) 검출된 포즈의 랜드마크 리스트
        # 각 랜드마크는 NormalizedLandmark 객체로, x/y/z 속성을 가짐
        # - x: 이미지 너비 기준 정규화 좌표 (0.0 = 왼쪽, 1.0 = 오른쪽)
        # - y: 이미지 높이 기준 정규화 좌표 (0.0 = 위, 1.0 = 아래)
        # - z: 깊이 (엉덩이 중심 기준 상대값, 카메라에 가까울수록 음수)
        # ==================================================================
        landmarks = result.pose_landmarks[0]

        # 필요한 랜드마크의 3D 좌표를 NumPy 배열로 변환하여 딕셔너리에 저장
        # 이후 관절 각도 계산에서 벡터 연산을 위해 NumPy 배열 형태가 필요
        coords = {}
        for name, idx in self.LANDMARKS.items():
            lm = landmarks[idx]
            coords[name] = np.array([lm.x, lm.y, lm.z])

        # ==================================================================
        # [6개 관절 각도 계산]
        # ==================================================================
        # JOINT_ANGLES 딕셔너리에 정의된 각 관절에 대해 각도를 계산합니다.
        # 'is_trunk' 플래그가 True인 경우 특수 계산(수직 기준)을 적용하고,
        # 그 외에는 일반적인 세 점 각도 계산을 수행합니다.
        # ==================================================================
        angles = {}
        for angle_name, config in self.JOINT_ANGLES.items():
            pt_names = config['points']
            if config.get('is_trunk'):
                # 몸통 각도: 어깨-엉덩이 벡터와 수직선 사이의 각도
                angle = self.calc_trunk_angle(
                    coords['left_shoulder'], coords['left_hip']
                )
            else:
                # 일반 관절 각도: 세 점 A, B(꼭짓점), C의 사이각
                a = coords[pt_names[0]]  # 관절의 한쪽 끝 (예: 엉덩이)
                b = coords[pt_names[1]]  # 꼭짓점/관절 중심 (예: 무릎)
                c = coords[pt_names[2]]  # 관절의 다른 끝 (예: 발목)
                angle = self.calc_angle(a, b, c)
            angles[angle_name] = angle

        # ==================================================================
        # [스켈레톤(골격) 시각화 - Tasks API의 draw_landmarks 사용]
        # ==================================================================
        # Tasks API의 drawing_utils.draw_landmarks()는 Legacy API와 유사하지만,
        # 입력 형식이 다릅니다:
        # - Legacy: mp.solutions.drawing_utils.draw_landmarks(
        #               image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # - Tasks:  _vision.drawing_utils.draw_landmarks(
        #               image, landmarks_list, connections, ...)
        #
        # 시각화 설정:
        # - landmark_drawing_spec: 각 관절 점(빨간색, 두께 2, 반지름 2)
        # - connection_drawing_spec: 뼈대 연결선(초록색, 두께 2, 반지름 1)
        #
        # 이 시각화는 디버깅 및 실시간 모니터링에 유용하며,
        # 관절 위치와 추적 품질을 시각적으로 확인할 수 있습니다.
        # ==================================================================
        self._draw_utils.draw_landmarks(
            annotated,
            landmarks,
            self._POSE_CONNECTIONS,
            landmark_drawing_spec=self._draw_utils.DrawingSpec(
                color=(0, 0, 255),       # BGR: 빨간색 - 관절 점
                thickness=2,             # 선 두께 (픽셀)
                circle_radius=2          # 점 반지름 (픽셀)
            ),
            connection_drawing_spec=self._draw_utils.DrawingSpec(
                color=(0, 255, 0),       # BGR: 초록색 - 뼈대 연결선
                thickness=2,             # 선 두께 (픽셀)
                circle_radius=1          # (연결선에서는 미사용이나 필수 파라미터)
            ),
        )

        # ==================================================================
        # [관절 각도 라벨 오버레이 (화면 표시)]
        # ==================================================================
        # 계산된 각도 값을 해당 관절 위치 근처에 텍스트로 표시합니다.
        # 정규화 좌표(0~1)를 실제 픽셀 좌표로 변환한 후,
        # 약간의 오프셋(+10 픽셀)을 주어 관절 점과 텍스트가 겹치지 않도록 합니다.
        #
        # Trunk_Angle은 별도의 위치 매핑이 없으므로 화면에 표시되지 않습니다.
        # (필요 시 angle_positions에 추가하여 표시 가능)
        # ==================================================================
        h, w = frame.shape[:2]  # 프레임의 높이, 너비 (픽셀 좌표 변환용)

        # 각 관절 각도를 화면에 표시할 위치 매핑
        # 해당 관절의 정규화 좌표를 사용하여 텍스트 위치 결정
        angle_positions = {
            'Knee_Angle': coords['left_knee'],        # 무릎 위치에 무릎 각도 표시
            'Hip_Angle': coords['left_hip'],          # 엉덩이 위치에 고관절 각도 표시
            'Ankle_Angle': coords['left_ankle'],      # 발목 위치에 발목 각도 표시
            'Shoulder_Angle': coords['left_shoulder'],# 어깨 위치에 어깨 각도 표시
            'Elbow_Angle': coords['left_elbow'],      # 팔꿈치 위치에 팔꿈치 각도 표시
        }

        for angle_name, angle_val in angles.items():
            if angle_name in angle_positions:
                pos = angle_positions[angle_name]
                # 정규화 좌표(0.0~1.0)를 픽셀 좌표로 변환
                # x좌표에 이미지 너비를 곱하고, y좌표에 이미지 높이를 곱함
                px, py = int(pos[0] * w), int(pos[1] * h)
                # config.py의 JOINT_LABELS에서 짧은 표시 라벨을 가져옴
                # 예: 'Knee_Angle' → 'Knee', 'Hip_Angle' → 'Hip'
                label = JOINT_LABELS.get(angle_name, angle_name)
                # 텍스트 오버레이: "라벨:각도값" 형식 (예: "Knee:142")
                # 위치를 x+10 픽셀만큼 오른쪽으로 이동하여 관절 점과 겹침 방지
                # 노란색(0, 255, 255) 텍스트로 가독성 확보 (어두운 배경에서 잘 보임)
                cv2.putText(annotated, f'{label}:{angle_val:.0f}',
                            (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)

        # 계산된 관절 각도 딕셔너리와 시각화된 프레임을 함께 반환
        return angles, annotated

    def close(self):
        """
        MediaPipe PoseLandmarker 리소스를 해제합니다.

        내부적으로 할당된 GPU/CPU 메모리, 추론 세션 등을 정리합니다.
        프로그램 종료 시 또는 더 이상 포즈 분석이 필요하지 않을 때 호출해야 합니다.
        close()를 호출하지 않으면 메모리 누수가 발생할 수 있습니다.

        사용 예시:
            analyzer = PoseAnalyzer()
            try:
                while running:
                    angles, frame = analyzer.process_frame(camera_frame)
            finally:
                analyzer.close()  # 반드시 해제
        """
        self._landmarker.close()
