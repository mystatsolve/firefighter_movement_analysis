"""
pose_analyzer.py - MediaPipe 양측(Bilateral) 포즈 감지 모듈
=============================================================

기존 oakd_realtime/pose_analyzer.py를 확장하여 좌/우 양측 관절 각도와
각 관절의 visibility 점수를 함께 반환합니다.

핵심 변경점 (기존 대비):
    1. left + right 양측 관절 각도 동시 추출
    2. 각 관절별 visibility 점수 반환 (가중 퓨전에 사용)
    3. 양측 visibility 가중 평균으로 단일 관절 각도 산출
    4. 듀얼 카메라 시스템에서 각 카메라가 독립적인 PoseAnalyzer 인스턴스를 가짐

사용 예시:
    analyzer = PoseAnalyzer()
    result = analyzer.process_frame(bgr_frame)
    if result is not None:
        angles, visibility, annotated = result
        # angles: {'Knee_Angle': 145.2, ...}  (양측 가중 평균)
        # visibility: {'Knee_Angle': 0.85, ...}  (평균 가시성)
"""

import os
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, Optional, Tuple

from config import (
    POSE_MIN_DETECTION_CONF, POSE_MIN_TRACKING_CONF,
    JOINT_LABELS, BILATERAL_JOINT_ANGLES, MIN_VISIBILITY_THRESHOLD,
)

# MediaPipe Tasks API 네임스페이스
_vision = mp.tasks.vision
_BaseOptions = mp.tasks.BaseOptions

# 모델 파일 경로 (바이너리 로딩으로 유니코드 경로 문제 우회)
_MODEL_FILENAME = 'pose_landmarker_full.task'
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODEL_FILENAME)


class PoseAnalyzer:
    """
    MediaPipe PoseLandmarker 기반 양측(Bilateral) 관절 각도 추출기.

    기존 단일(left만) 분석을 확장하여 좌/우 모두 분석하고,
    각 관절의 visibility를 함께 반환합니다.
    """

    # 33개 랜드마크 인덱스 매핑 (좌/우 모두 포함)
    LANDMARKS = {
        'nose': 0,
        'left_shoulder': 11,  'right_shoulder': 12,
        'left_elbow': 13,     'right_elbow': 14,
        'left_wrist': 15,     'right_wrist': 16,
        'left_hip': 23,       'right_hip': 24,
        'left_knee': 25,      'right_knee': 26,
        'left_ankle': 27,     'right_ankle': 28,
        'left_heel': 29,      'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32,
    }

    # 스켈레톤 연결선 (시각화용)
    _POSE_CONNECTIONS = _vision.PoseLandmarksConnections.POSE_LANDMARKS

    def __init__(self, min_detection_confidence=None, min_tracking_confidence=None):
        """
        PoseAnalyzer 초기화.

        Parameters
        ----------
        min_detection_confidence : float, optional
            포즈 검출 최소 신뢰도. None이면 config 기본값 사용.
        min_tracking_confidence : float, optional
            포즈 추적 최소 신뢰도. None이면 config 기본값 사용.
        """
        det = min_detection_confidence if min_detection_confidence is not None else POSE_MIN_DETECTION_CONF
        trk = min_tracking_confidence if min_tracking_confidence is not None else POSE_MIN_TRACKING_CONF

        # 모델 바이너리 로딩 (유니코드 경로 우회)
        with open(_MODEL_PATH, 'rb') as f:
            model_data = f.read()

        options = _vision.PoseLandmarkerOptions(
            base_options=_BaseOptions(model_asset_buffer=model_data),
            running_mode=_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=det,
            min_pose_presence_confidence=det,
            min_tracking_confidence=trk,
        )
        self._landmarker = _vision.PoseLandmarker.create_from_options(options)
        self._draw_utils = _vision.drawing_utils

        # VIDEO 모드 타임스탬프 (단조 증가 필수)
        self._frame_ts_ms = 0

    @staticmethod
    def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        세 점 A, B(꼭짓점), C로 정의되는 관절 각도를 계산합니다.

        벡터 내적과 역코사인으로 0~180도 범위의 각도를 반환합니다.
        수치 안정성을 위해 분모에 epsilon 추가, cos값 클리핑을 적용합니다.
        """
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def calc_trunk_angle(shoulder: np.ndarray, hip: np.ndarray) -> float:
        """
        몸통(trunk) 전방 기울기 각도 계산.

        어깨-엉덩이 벡터와 수직선(vertical) 사이의 각도를 측정합니다.
        180도 = 완전 직립, 90도 = 수평 굴곡.
        """
        trunk_vec = shoulder - hip
        vertical = np.array([0, -1, 0])
        cos_angle = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return 180.0 - float(np.degrees(np.arccos(cos_angle)))

    def process_frame(self, frame: np.ndarray) -> Optional[Tuple[Dict[str, float], Dict[str, float], np.ndarray]]:
        """
        단일 프레임에서 양측(Bilateral) 관절 각도 + visibility를 추출합니다.

        처리 파이프라인:
        1. BGR → RGB 변환 → MediaPipe 포즈 검출
        2. 각 관절의 left/right 양측 각도 계산
        3. visibility 가중 평균으로 최종 각도 산출
        4. 스켈레톤 + 각도 라벨 시각화

        Parameters
        ----------
        frame : np.ndarray
            BGR 형식의 입력 프레임 (shape: H x W x 3)

        Returns
        -------
        Optional[Tuple[Dict, Dict, np.ndarray]]
            포즈 미검출 시 None.
            검출 시 (angles, visibility, annotated_frame) 튜플:
            - angles: {'Knee_Angle': float, ...} 양측 가중 평균 각도
            - visibility: {'Knee_Angle': float, ...} 평균 가시성 점수
            - annotated_frame: 스켈레톤이 그려진 프레임
        """
        # BGR → RGB 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 타임스탬프 증가 및 포즈 검출
        self._frame_ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms=self._frame_ts_ms)

        annotated = frame.copy()

        # 포즈 미검출
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = result.pose_landmarks[0]

        # 랜드마크 좌표 + visibility 추출
        coords = {}
        vis_scores = {}
        for name, idx in self.LANDMARKS.items():
            lm = landmarks[idx]
            coords[name] = np.array([lm.x, lm.y, lm.z])
            vis_scores[name] = lm.visibility if hasattr(lm, 'visibility') else 1.0

        # ====================================================================
        # 양측(Bilateral) 관절 각도 계산
        # ====================================================================
        # 각 관절에 대해 left/right 양측의 각도와 visibility를 구한 후,
        # visibility 가중 평균으로 최종 각도를 결정합니다.
        #
        # 퓨전 수식:
        #   w_left = vis_left / (vis_left + vis_right + epsilon)
        #   w_right = vis_right / (vis_left + vis_right + epsilon)
        #   angle = w_left * angle_left + w_right * angle_right
        #
        # 한쪽만 보이면 (vis < threshold) 보이는 쪽만 사용합니다.
        # ====================================================================
        angles = {}
        visibility = {}

        for angle_name, config in BILATERAL_JOINT_ANGLES.items():
            is_trunk = config.get('is_trunk', False)

            # 왼쪽 각도 + visibility 계산
            left_pts = config['left']
            left_vis = min(vis_scores.get(left_pts[0], 0),
                          vis_scores.get(left_pts[1], 0),
                          vis_scores.get(left_pts[2], 0) if not is_trunk else vis_scores.get(left_pts[1], 0))
            left_angle = None
            if left_vis >= MIN_VISIBILITY_THRESHOLD:
                if is_trunk:
                    left_angle = self.calc_trunk_angle(coords[left_pts[0]], coords[left_pts[1]])
                else:
                    left_angle = self.calc_angle(
                        coords[left_pts[0]], coords[left_pts[1]], coords[left_pts[2]])

            # 오른쪽 각도 + visibility 계산
            right_pts = config['right']
            right_vis = min(vis_scores.get(right_pts[0], 0),
                           vis_scores.get(right_pts[1], 0),
                           vis_scores.get(right_pts[2], 0) if not is_trunk else vis_scores.get(right_pts[1], 0))
            right_angle = None
            if right_vis >= MIN_VISIBILITY_THRESHOLD:
                if is_trunk:
                    right_angle = self.calc_trunk_angle(coords[right_pts[0]], coords[right_pts[1]])
                else:
                    right_angle = self.calc_angle(
                        coords[right_pts[0]], coords[right_pts[1]], coords[right_pts[2]])

            # visibility 가중 평균
            eps = 1e-6
            if left_angle is not None and right_angle is not None:
                # 양쪽 모두 유효 → 가중 평균
                w_left = left_vis / (left_vis + right_vis + eps)
                w_right = right_vis / (left_vis + right_vis + eps)
                fused_angle = w_left * left_angle + w_right * right_angle
                fused_vis = (left_vis + right_vis) / 2.0
            elif left_angle is not None:
                # 왼쪽만 유효
                fused_angle = left_angle
                fused_vis = left_vis
            elif right_angle is not None:
                # 오른쪽만 유효
                fused_angle = right_angle
                fused_vis = right_vis
            else:
                # 양쪽 모두 불가 → 기본값 (직립 자세 가정)
                fused_angle = 180.0 if not is_trunk else 170.0
                fused_vis = 0.0

            angles[angle_name] = fused_angle
            visibility[angle_name] = fused_vis

        # ====================================================================
        # 스켈레톤 시각화
        # ====================================================================
        self._draw_utils.draw_landmarks(
            annotated,
            landmarks,
            self._POSE_CONNECTIONS,
            landmark_drawing_spec=self._draw_utils.DrawingSpec(
                color=(0, 0, 255), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=self._draw_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=1
            ),
        )

        # 관절 각도 라벨 오버레이
        h, w = frame.shape[:2]
        # 왼쪽 관절 위치에 각도 표시 (기존 호환)
        angle_positions = {
            'Knee_Angle': coords.get('left_knee'),
            'Hip_Angle': coords.get('left_hip'),
            'Ankle_Angle': coords.get('left_ankle'),
            'Shoulder_Angle': coords.get('left_shoulder'),
            'Elbow_Angle': coords.get('left_elbow'),
        }

        for angle_name, angle_val in angles.items():
            pos = angle_positions.get(angle_name)
            if pos is not None:
                px, py = int(pos[0] * w), int(pos[1] * h)
                label = JOINT_LABELS.get(angle_name, angle_name)
                vis_val = visibility.get(angle_name, 0)
                # 색상: visibility에 따라 노란(높음) → 빨강(낮음)
                color = (0, 255, 255) if vis_val > 0.5 else (0, 165, 255)
                cv2.putText(annotated, f'{label}:{angle_val:.0f}',
                            (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1, cv2.LINE_AA)

        return angles, visibility, annotated

    def close(self):
        """MediaPipe PoseLandmarker 리소스 해제."""
        self._landmarker.close()
