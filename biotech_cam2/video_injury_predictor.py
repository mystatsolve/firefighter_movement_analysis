"""
소방관 동영상 기반 근골격계 부상 예측 시스템
===========================================

동영상에서 MediaPipe Pose로 신체 랜드마크를 감지하고,
관절 각도를 실시간 추출하여 근육 내부 압력 및 인대 장력을
수치화한 뒤 부상 위험도를 예측합니다.

파이프라인:
  동영상 → MediaPipe Pose 감지 → 33개 랜드마크 추출
  → 6개 관절 각도 계산 → 근육 섬유 길이/속도 변환
  → Hill-type 근육 모델 (DeGrooteFregly2016) → 근육 내부 압력(kPa)
  → Blankevoort 인대 모델 → 인대 장력(N)
  → 부상 위험도 분류 (낮음/중간/높음/매우높음)

사용법:
  python video_injury_predictor.py <동영상경로> [시나리오이름]
  python video_injury_predictor.py firefighter_stair.mp4 "계단 오르기"
  python video_injury_predictor.py 0    (웹캠 실시간)

의존성:
  pip install mediapipe opencv-python numpy matplotlib scipy

참고: https://github.com/opensim-org/opensim-core
"""

import os
import sys
import csv
import time as time_module
import numpy as np
from scipy.integrate import solve_ivp
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Dict, List, Tuple, Optional

# 한글 폰트
for font_name in ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']:
    try:
        rcParams['font.family'] = font_name
        break
    except:
        continue
rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


###############################################################################
# 1. MediaPipe 포즈 감지 및 관절 각도 추출
###############################################################################

class PoseAnalyzer:
    """MediaPipe Pose 기반 관절 각도 추출기"""

    # MediaPipe Pose 랜드마크 인덱스
    # https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
    LANDMARKS = {
        'nose': 0,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
        'left_heel': 29, 'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32,
    }

    # 관절 각도 정의: (관절명, 점A, 점B_관절중심, 점C)
    # 각도 = A-B-C 사이의 각도
    JOINT_ANGLES = {
        'Knee_Angle': {
            'points': ('left_hip', 'left_knee', 'left_ankle'),
            'label': '무릎',
        },
        'Hip_Angle': {
            'points': ('left_shoulder', 'left_hip', 'left_knee'),
            'label': '고관절',
        },
        'Ankle_Angle': {
            'points': ('left_knee', 'left_ankle', 'left_foot_index'),
            'label': '발목',
        },
        'Shoulder_Angle': {
            'points': ('left_hip', 'left_shoulder', 'left_elbow'),
            'label': '어깨',
        },
        'Elbow_Angle': {
            'points': ('left_shoulder', 'left_elbow', 'left_wrist'),
            'label': '팔꿈치',
        },
        'Trunk_Angle': {
            # 몸통 기울기: 어깨-고관절-수직선 (수직 기준으로 계산)
            'points': ('left_shoulder', 'left_hip', 'left_knee'),
            'label': '몸통',
            'is_trunk': True,
        },
    }

    def __init__(self, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        세 점 A, B(관절 중심), C 사이의 각도 계산 (도).

        Parameters
        ----------
        a, b, c : 3D 좌표 (x, y, z)

        Returns
        -------
        float : 각도 (도, 0~180)
        """
        ba = a - b
        bc = c - b

        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle

    @staticmethod
    def calc_trunk_angle(shoulder: np.ndarray, hip: np.ndarray) -> float:
        """
        몸통 각도 계산: 어깨-고관절 벡터와 수직선 사이의 각도.
        직립 = 180도, 전방 굴곡 = 감소
        """
        trunk_vec = shoulder - hip
        vertical = np.array([0, -1, 0])  # MediaPipe Y축: 위가 음수

        cos_angle = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        # 직립=180으로 보정
        return 180.0 - angle

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[dict], np.ndarray]:
        """
        단일 프레임에서 포즈 감지 및 관절 각도 계산.

        Parameters
        ----------
        frame : BGR 이미지 (OpenCV 형식)

        Returns
        -------
        (angles_dict or None, annotated_frame)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        annotated = frame.copy()

        if not results.pose_landmarks:
            return None, annotated

        landmarks = results.pose_landmarks.landmark

        # 랜드마크를 3D 좌표 배열로 변환
        coords = {}
        for name, idx in self.LANDMARKS.items():
            lm = landmarks[idx]
            coords[name] = np.array([lm.x, lm.y, lm.z])

        # 관절 각도 계산
        angles = {}
        for angle_name, config in self.JOINT_ANGLES.items():
            pt_names = config['points']
            a = coords[pt_names[0]]
            b = coords[pt_names[1]]
            c = coords[pt_names[2]]

            if config.get('is_trunk'):
                angle = self.calc_trunk_angle(
                    coords['left_shoulder'], coords['left_hip']
                )
            else:
                angle = self.calc_angle(a, b, c)

            angles[angle_name] = angle

        # 랜드마크 시각화
        self.mp_draw.draw_landmarks(
            annotated, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style()
        )

        # 관절 각도 텍스트 표시
        h, w = frame.shape[:2]
        angle_positions = {
            'Knee_Angle': coords['left_knee'],
            'Hip_Angle': coords['left_hip'],
            'Ankle_Angle': coords['left_ankle'],
            'Shoulder_Angle': coords['left_shoulder'],
            'Elbow_Angle': coords['left_elbow'],
        }

        for angle_name, angle_val in angles.items():
            if angle_name in angle_positions:
                pos = angle_positions[angle_name]
                px, py = int(pos[0] * w), int(pos[1] * h)
                label = self.JOINT_ANGLES[angle_name]['label']
                cv2.putText(annotated, f'{label}:{angle_val:.0f}',
                            (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)

        return angles, annotated

    def close(self):
        self.pose.close()


###############################################################################
# 2. 동영상 처리 및 관절 각도 시계열 추출
###############################################################################

class VideoProcessor:
    """동영상 파일에서 전체 관절 각도 시계열 추출"""

    def __init__(self, video_path: str):
        """
        Parameters
        ----------
        video_path : 동영상 파일 경로 또는 '0' (웹캠)
        """
        if video_path.isdigit():
            self.cap = cv2.VideoCapture(int(video_path))
            self.is_webcam = True
        else:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"동영상 파일을 찾을 수 없습니다: {video_path}")
            self.cap = cv2.VideoCapture(video_path)
            self.is_webcam = False

        if not self.cap.isOpened():
            raise RuntimeError(f"동영상을 열 수 없습니다: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.total_frames > 0 else 0

        print(f"    해상도: {self.width} x {self.height}")
        print(f"    FPS: {self.fps:.1f}")
        print(f"    총 프레임: {self.total_frames}")
        print(f"    영상 길이: {self.duration:.1f}초")

    def extract_angles(self, analyzer: PoseAnalyzer,
                       save_annotated_video: bool = True,
                       output_video_path: str = None,
                       max_frames: int = None) -> dict:
        """
        전체 동영상에서 관절 각도 시계열 추출.

        Parameters
        ----------
        analyzer : PoseAnalyzer 인스턴스
        save_annotated_video : 랜드마크 표시된 영상 저장 여부
        output_video_path : 출력 영상 경로
        max_frames : 최대 처리 프레임 수 (None=전체)

        Returns
        -------
        dict : {'time': np.ndarray, 'joints': {관절명: np.ndarray}, 'fps': float}
        """
        time_list = []
        joints_data = {name: [] for name in PoseAnalyzer.JOINT_ANGLES}

        # 출력 영상 설정
        writer = None
        if save_annotated_video:
            if output_video_path is None:
                output_video_path = os.path.join(OUTPUT_DIR, 'pose_annotated.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, self.fps,
                                     (self.width, self.height))

        frame_count = 0
        detected_count = 0
        process_limit = max_frames if max_frames else self.total_frames

        print(f"    포즈 분석 시작...")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            angles, annotated = analyzer.process_frame(frame)

            current_time = frame_count / self.fps

            if angles is not None:
                time_list.append(current_time)
                for joint_name in PoseAnalyzer.JOINT_ANGLES:
                    joints_data[joint_name].append(angles.get(joint_name, 0.0))
                detected_count += 1

            if writer:
                # 프레임에 진행 상태 표시
                progress = frame_count / max(process_limit, 1) * 100
                cv2.putText(annotated, f'Frame: {frame_count}/{process_limit} ({progress:.0f}%)',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 위험도 실시간 표시 (간이 계산)
                if angles:
                    y_offset = 60
                    for joint_name, angle_val in angles.items():
                        label = PoseAnalyzer.JOINT_ANGLES[joint_name]['label']
                        cv2.putText(annotated, f'{label}: {angle_val:.1f} deg',
                                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1)
                        y_offset += 25

                writer.write(annotated)

            frame_count += 1

            if frame_count % 100 == 0:
                pct = frame_count / max(process_limit, 1) * 100
                print(f"    진행: {frame_count}/{process_limit} ({pct:.0f}%)")

        if writer:
            writer.release()
            print(f"    포즈 표시 영상 저장: {output_video_path}")

        self.cap.release()

        # 결과 정리
        detection_rate = detected_count / max(frame_count, 1) * 100
        print(f"    완료: {frame_count} 프레임 처리, {detected_count} 프레임 감지 ({detection_rate:.1f}%)")

        if detected_count == 0:
            print("    [경고] 포즈를 감지할 수 없었습니다. 데모 데이터로 대체합니다.")
            print("    (실제 사람이 촬영된 영상을 사용하면 실측 분석이 가능합니다)")
            return self._generate_demo_data(frame_count)

        # numpy 배열로 변환
        time_arr = np.array(time_list)
        joints = {}
        for joint_name, data in joints_data.items():
            if data:
                joints[joint_name] = np.array(data)

        # 스무딩 (노이즈 제거)
        joints = self._smooth_angles(joints, window=5)

        return {
            'time': time_arr,
            'joints': joints,
            'fps': self.fps,
            'total_frames': frame_count,
            'detected_frames': detected_count,
        }

    def _generate_demo_data(self, frame_count: int) -> dict:
        """포즈 감지 실패 시 데모 데이터 생성 (스쿼트 동작 시뮬레이션)"""
        n = max(frame_count, 90)
        time_arr = np.arange(n) / self.fps
        freq = 0.5  # 0.5Hz 스쿼트

        joints = {
            'Knee_Angle': 145 + 25 * np.sin(2 * np.pi * freq * time_arr),
            'Hip_Angle': 150 + 20 * np.sin(2 * np.pi * freq * time_arr + 0.3),
            'Ankle_Angle': 85 + 10 * np.sin(2 * np.pi * freq * time_arr + 0.1),
            'Shoulder_Angle': 50 + 20 * np.sin(2 * np.pi * freq * time_arr * 0.5),
            'Elbow_Angle': 130 + 20 * np.sin(2 * np.pi * freq * time_arr * 0.5 + 0.5),
            'Trunk_Angle': 160 + 15 * np.sin(2 * np.pi * freq * time_arr + 0.2),
        }

        return {
            'time': time_arr, 'joints': joints,
            'fps': self.fps, 'total_frames': n, 'detected_frames': n,
        }

    @staticmethod
    def _smooth_angles(joints: dict, window: int = 5) -> dict:
        """이동 평균 필터로 관절 각도 스무딩 (측정 노이즈 제거)"""
        smoothed = {}
        for name, data in joints.items():
            if len(data) < window:
                smoothed[name] = data
                continue
            kernel = np.ones(window) / window
            # 가장자리 처리
            padded = np.pad(data, (window // 2, window // 2), mode='edge')
            smoothed[name] = np.convolve(padded, kernel, mode='valid')[:len(data)]
        return smoothed


###############################################################################
# 3. 생체역학 모델 (Hill-type 근육 + 인대)
###############################################################################

class DeGrooteFregly2016Muscle:
    """OpenSim DeGrooteFregly2016Muscle 모델"""

    _b11, _b21, _b31, _b41 = 0.8150671134243542, 1.055033428970575, 0.162384573599574, 0.063303448465465
    _b12, _b22, _b32, _b42 = 0.433004984392647, 0.716775413397760, -0.029947116970696, 0.200356847296188
    _b13, _b23, _b33, _b43 = 0.1, 1.0, 0.353553390593274, 0.0
    _kPE = 4.0
    _c1, _c2, _c3 = 0.200, 1.0, 0.200
    _d1, _d2, _d3, _d4 = -0.3211346127989808, -8.149, -0.374, 0.8825327733249912
    MIN_NORM_FIBER_LENGTH = 0.2
    MAX_NORM_FIBER_LENGTH = 1.8

    def __init__(self, name, max_isometric_force, optimal_fiber_length,
                 tendon_slack_length, pennation_angle_at_optimal, pcsa,
                 activation_time_constant=0.015, deactivation_time_constant=0.060,
                 fiber_damping=0.01, passive_fiber_strain_at_one_norm_force=0.6,
                 tendon_strain_at_one_norm_force=0.049):
        self.name = name
        self.max_isometric_force = max_isometric_force
        self.optimal_fiber_length = optimal_fiber_length
        self.tendon_slack_length = tendon_slack_length
        self.pennation_angle_at_optimal = pennation_angle_at_optimal
        self.pcsa = pcsa
        self.tau_a = activation_time_constant
        self.tau_d = deactivation_time_constant
        self.fiber_damping = fiber_damping
        self._e0 = passive_fiber_strain_at_one_norm_force
        self._kT = np.log((1.0 + self._c3) / self._c1) / tendon_strain_at_one_norm_force

    @staticmethod
    def _gaussian_like(x, b1, b2, b3, b4):
        denom = b3 + b4 * x
        if abs(denom) < 1e-12:
            denom = 1e-12
        return b1 * np.exp(-0.5 * ((x - b2) / denom) ** 2)

    def calc_active_fl(self, nfl):
        nfl = np.atleast_1d(nfl)
        fl = np.zeros_like(nfl, dtype=float)
        for i, lm in enumerate(nfl):
            fl[i] = (self._gaussian_like(lm, self._b11, self._b21, self._b31, self._b41) +
                     self._gaussian_like(lm, self._b12, self._b22, self._b32, self._b42) +
                     self._gaussian_like(lm, self._b13, self._b23, self._b33, self._b43))
        return fl

    def calc_passive_fl(self, nfl):
        nfl = np.atleast_1d(nfl)
        num = np.exp(self._kPE * (nfl - 1.0) / self._e0)
        denom = np.exp(self._kPE)
        offset = np.exp(self._kPE * (self.MIN_NORM_FIBER_LENGTH - 1.0) / self._e0)
        return np.maximum((num - offset) / (denom - offset), 0.0)

    def calc_fv(self, nfv):
        v = np.atleast_1d(nfv)
        inner = self._d2 * v + self._d3
        return np.clip(self._d1 * np.log(inner + np.sqrt(inner**2 + 1.0)) + self._d4, 0.0, 1.8)

    def activation_derivative(self, a, e):
        f = 0.5 * np.tanh(10.0 * (e - a))
        z = 0.5 + 1.5 * a
        return ((f + 0.5) / (self.tau_a * z) + (-f + 0.5) * z / self.tau_d) * (e - a)

    def simulate(self, time_arr, excitation, nfl_profile, nfv_profile):
        n = len(time_arr)

        def ode(t, y):
            exc = np.interp(t, time_arr, excitation)
            return [self.activation_derivative(y[0], exc)]

        sol = solve_ivp(ode, [time_arr[0], time_arr[-1]], [0.05],
                        t_eval=time_arr, method='RK45', max_step=0.01)
        activations = np.clip(sol.y[0], 0.01, 1.0)

        cos_penn = np.cos(self.pennation_angle_at_optimal)
        f_act_fl = self.calc_active_fl(nfl_profile)
        f_pass_fl = self.calc_passive_fl(nfl_profile)
        f_fv = self.calc_fv(nfv_profile)

        active = self.max_isometric_force * activations * f_act_fl * f_fv
        passive = self.max_isometric_force * f_pass_fl
        damp = self.max_isometric_force * self.fiber_damping * nfv_profile
        total = active + passive + damp
        stress = total / self.pcsa

        return {
            'time': time_arr, 'activation': activations,
            'active_fiber_force': active, 'passive_fiber_force': passive,
            'total_fiber_force': total, 'force_along_tendon': total * cos_penn,
            'muscle_stress_Pa': stress, 'muscle_stress_kPa': stress / 1000.0,
        }


class Blankevoort1991Ligament:
    """OpenSim Blankevoort1991Ligament 모델"""

    def __init__(self, name, slack_length, linear_stiffness,
                 transition_strain=0.06, damping_coefficient=0.003):
        self.name = name
        self.slack_length = slack_length
        self.linear_stiffness = linear_stiffness
        self.transition_strain = transition_strain
        self.damping_coefficient = damping_coefficient

    def calc_force(self, strain, strain_rate):
        # 스프링
        if strain <= 0:
            sf = 0.0
        elif strain < self.transition_strain:
            sf = 0.5 * self.linear_stiffness / self.transition_strain * strain**2
        else:
            sf = self.linear_stiffness * (strain - self.transition_strain / 2.0)
        # 감쇠
        df = 0.0
        if strain > 0 and strain_rate > 0:
            df = self.damping_coefficient * strain_rate * 0.5 * (1.0 + np.tanh(strain / 0.005))
        return max(0.0, sf + df), sf, df

    def simulate(self, time_arr, length_profile, velocity_profile):
        n = len(time_arr)
        results = {
            'time': time_arr, 'strain': np.zeros(n), 'strain_percent': np.zeros(n),
            'spring_force': np.zeros(n), 'damping_force': np.zeros(n),
            'total_force': np.zeros(n),
        }
        for i in range(n):
            strain = length_profile[i] / self.slack_length - 1.0
            strain_rate = velocity_profile[i] / self.slack_length
            total, spring, damp = self.calc_force(strain, strain_rate)
            results['strain'][i] = strain
            results['strain_percent'][i] = strain * 100.0
            results['spring_force'][i] = spring
            results['damping_force'][i] = damp
            results['total_force'][i] = total
        return results


###############################################################################
# 4. 관절 각도 → 근육/인대 변환
###############################################################################

MUSCLE_PARAMS = {
    '대퇴사두근': {'max_isometric_force': 6000, 'optimal_fiber_length': 0.084,
                'tendon_slack_length': 0.10, 'pennation_angle_at_optimal': np.radians(15), 'pcsa': 75e-4},
    '대퇴이두근': {'max_isometric_force': 900, 'optimal_fiber_length': 0.11,
                'tendon_slack_length': 0.32, 'pennation_angle_at_optimal': np.radians(12), 'pcsa': 18e-4},
    '비복근': {'max_isometric_force': 1600, 'optimal_fiber_length': 0.055,
             'tendon_slack_length': 0.39, 'pennation_angle_at_optimal': np.radians(17), 'pcsa': 30e-4},
    '대둔근': {'max_isometric_force': 1500, 'optimal_fiber_length': 0.14,
             'tendon_slack_length': 0.05, 'pennation_angle_at_optimal': np.radians(5), 'pcsa': 40e-4},
    '척추기립근': {'max_isometric_force': 2500, 'optimal_fiber_length': 0.12,
               'tendon_slack_length': 0.03, 'pennation_angle_at_optimal': np.radians(0), 'pcsa': 50e-4},
    '삼각근': {'max_isometric_force': 1100, 'optimal_fiber_length': 0.10,
             'tendon_slack_length': 0.04, 'pennation_angle_at_optimal': np.radians(15), 'pcsa': 20e-4},
    '상완이두근': {'max_isometric_force': 600, 'optimal_fiber_length': 0.116,
              'tendon_slack_length': 0.24, 'pennation_angle_at_optimal': np.radians(0), 'pcsa': 12e-4},
    '광배근': {'max_isometric_force': 1200, 'optimal_fiber_length': 0.25,
             'tendon_slack_length': 0.05, 'pennation_angle_at_optimal': np.radians(20), 'pcsa': 25e-4},
}

LIGAMENT_PARAMS = {
    '전방십자인대(ACL)': {'slack_length': 0.032, 'linear_stiffness': 5000, 'transition_strain': 0.06,
                      'damping_coefficient': 0.003, 'estimated_failure_force': 2160},
    '후방십자인대(PCL)': {'slack_length': 0.038, 'linear_stiffness': 9000, 'transition_strain': 0.06,
                      'damping_coefficient': 0.003, 'estimated_failure_force': 3000},
    '슬개건': {'slack_length': 0.050, 'linear_stiffness': 15000, 'transition_strain': 0.05,
            'damping_coefficient': 0.005, 'estimated_failure_force': 10000},
    '극상인대': {'slack_length': 0.045, 'linear_stiffness': 3000, 'transition_strain': 0.06,
             'damping_coefficient': 0.002, 'estimated_failure_force': 1500},
    '관절와상완인대': {'slack_length': 0.025, 'linear_stiffness': 2000, 'transition_strain': 0.06,
               'damping_coefficient': 0.002, 'estimated_failure_force': 800},
    '요추인대': {'slack_length': 0.055, 'linear_stiffness': 4000, 'transition_strain': 0.07,
             'damping_coefficient': 0.003, 'estimated_failure_force': 2000},
}

# 관절 각도 → 근육 섬유 길이 매핑
ANGLE_TO_MUSCLE = {
    '대퇴사두근': {'joint': 'Knee_Angle', 'theta_range': (60, 180), 'nfl_range': (0.7, 1.15)},
    '대퇴이두근': {'joint': 'Knee_Angle', 'theta_range': (60, 180), 'nfl_range': (1.2, 0.8)},
    '비복근': {'joint': 'Ankle_Angle', 'theta_range': (60, 120), 'nfl_range': (1.15, 0.8)},
    '대둔근': {'joint': 'Hip_Angle', 'theta_range': (90, 180), 'nfl_range': (1.15, 0.85)},
    '척추기립근': {'joint': 'Trunk_Angle', 'theta_range': (120, 180), 'nfl_range': (1.2, 0.9)},
    '삼각근': {'joint': 'Shoulder_Angle', 'theta_range': (0, 180), 'nfl_range': (0.8, 1.2)},
    '상완이두근': {'joint': 'Elbow_Angle', 'theta_range': (40, 170), 'nfl_range': (0.7, 1.2)},
    '광배근': {'joint': 'Shoulder_Angle', 'theta_range': (0, 180), 'nfl_range': (1.15, 0.8)},
}

ANGLE_TO_LIGAMENT = {
    '전방십자인대(ACL)': {'joint': 'Knee_Angle', 'theta_range': (60, 180), 'strain_range': (0.01, 0.05)},
    '후방십자인대(PCL)': {'joint': 'Knee_Angle', 'theta_range': (60, 180), 'strain_range': (0.05, 0.01)},
    '슬개건': {'joint': 'Knee_Angle', 'theta_range': (60, 180), 'strain_range': (0.06, 0.02)},
    '극상인대': {'joint': 'Trunk_Angle', 'theta_range': (120, 180), 'strain_range': (0.06, 0.01)},
    '관절와상완인대': {'joint': 'Shoulder_Angle', 'theta_range': (0, 180), 'strain_range': (0.01, 0.06)},
    '요추인대': {'joint': 'Trunk_Angle', 'theta_range': (120, 180), 'strain_range': (0.08, 0.01)},
}


###############################################################################
# 4-1. 역동역학 기반 외부 하중 모델
###############################################################################

# 동작 유형별 관절 토크 분배 계수
# 외부 하중이 각 관절에 얼마나 토크를 발생시키는지 정의
# 값 = 하중(N) × 모멘트 암(m) 비율 → 관절 토크 기여도
LOAD_TASK_PROFILES = {
    'lift': {   # 들기 (바닥에서 물체를 들어올리기)
        'name': '들기 (Lifting)',
        'description': '바닥 또는 낮은 위치에서 물체를 들어올리는 동작',
        # 각 근육이 받는 하중 기여 계수 (0~1)
        # 체중 대비 외부 하중의 추가 부담 비율
        'muscle_load_factor': {
            '대퇴사두근': 0.35,    # 무릎 신전: 들기 시 주요 동원
            '대퇴이두근': 0.20,    # 무릎 굴곡/고관절 신전 보조
            '비복근': 0.10,       # 발목 안정화
            '대둔근': 0.30,       # 고관절 신전: 들기 핵심
            '척추기립근': 0.45,    # 몸통 신전: 들기 시 최대 부하
            '삼각근': 0.15,       # 어깨: 물체 잡기
            '상완이두근': 0.25,    # 팔꿈치: 물체 잡기/당기기
            '광배근': 0.20,       # 등: 물체 몸쪽으로 당기기
        },
        # 인대 추가 변형률 계수
        'ligament_strain_factor': {
            '전방십자인대(ACL)': 0.3,
            '후방십자인대(PCL)': 0.2,
            '슬개건': 0.35,
            '극상인대': 0.4,      # 허리 인대: 들기 시 고위험
            '관절와상완인대': 0.15,
            '요추인대': 0.5,      # 요추: 들기 시 최대 부하
        },
    },
    'pull': {   # 끌기 (호스, 장비 등)
        'name': '끌기 (Pulling)',
        'description': '물체를 몸쪽으로 끌거나 당기는 동작',
        'muscle_load_factor': {
            '대퇴사두근': 0.25,
            '대퇴이두근': 0.15,
            '비복근': 0.15,
            '대둔근': 0.20,
            '척추기립근': 0.30,
            '삼각근': 0.20,
            '상완이두근': 0.35,    # 팔꿈치 굴곡: 당기기 핵심
            '광배근': 0.40,       # 등: 당기기 핵심
        },
        'ligament_strain_factor': {
            '전방십자인대(ACL)': 0.2,
            '후방십자인대(PCL)': 0.15,
            '슬개건': 0.2,
            '극상인대': 0.25,
            '관절와상완인대': 0.35, # 어깨 인대: 당기기 시 부하
            '요추인대': 0.35,
        },
    },
    'carry': {  # 운반 (장비, 환자 등)
        'name': '운반 (Carrying)',
        'description': '물체를 들고 이동하는 동작',
        'muscle_load_factor': {
            '대퇴사두근': 0.30,
            '대퇴이두근': 0.15,
            '비복근': 0.20,       # 보행 + 추가 하중
            '대둔근': 0.25,
            '척추기립근': 0.35,
            '삼각근': 0.25,
            '상완이두근': 0.20,
            '광배근': 0.15,
        },
        'ligament_strain_factor': {
            '전방십자인대(ACL)': 0.25,
            '후방십자인대(PCL)': 0.2,
            '슬개건': 0.3,
            '극상인대': 0.3,
            '관절와상완인대': 0.2,
            '요추인대': 0.4,
        },
    },
    'push': {   # 밀기 (장비, 문 등)
        'name': '밀기 (Pushing)',
        'description': '물체를 앞으로 밀어내는 동작',
        'muscle_load_factor': {
            '대퇴사두근': 0.30,
            '대퇴이두근': 0.10,
            '비복근': 0.20,
            '대둔근': 0.25,
            '척추기립근': 0.25,
            '삼각근': 0.35,       # 어깨: 밀기 핵심
            '상완이두근': 0.15,
            '광배근': 0.10,
        },
        'ligament_strain_factor': {
            '전방십자인대(ACL)': 0.2,
            '후방십자인대(PCL)': 0.15,
            '슬개건': 0.25,
            '극상인대': 0.2,
            '관절와상완인대': 0.3,
            '요추인대': 0.3,
        },
    },
}


def calc_load_excitation(joint_angles_deg: np.ndarray, load_kg: float,
                         body_mass_kg: float, load_factor: float,
                         max_isometric_force: float) -> np.ndarray:
    """
    역동역학 기반 외부 하중에 의한 추가 흥분 계산.

    원리:
      관절 토크 = 외부 하중(N) × 모멘트 암(m) × 자세 계수
      추가 흥분 = 관절 토크 / 최대 근력

    자세 계수: 관절 각도에 따라 불리한 자세일수록 토크 증가
      - 깊은 굴곡(무릎 90도 이하) → 모멘트 암 증가 → 토크 상승
      - 직립 자세(180도) → 최소 토크

    Parameters
    ----------
    joint_angles_deg : 관절 각도 시계열 (도)
    load_kg : 외부 하중 (kg)
    body_mass_kg : 체중 (kg)
    load_factor : 해당 근육의 하중 분배 계수
    max_isometric_force : 해당 근육의 최대 등척성 힘 (N)

    Returns
    -------
    np.ndarray : 하중에 의한 추가 흥분 [0, 1]
    """
    g = 9.81  # m/s²
    load_force = load_kg * g  # N

    # 평균 모멘트 암 (m) - 관절 각도에 따라 변화
    # 각도가 작을수록(깊은 굴곡) 모멘트 암이 커져 토크 증가
    # 180도(직립)에서 최소, 90도에서 최대
    angle_rad = np.radians(joint_angles_deg)
    posture_factor = np.sin(angle_rad / 2.0)  # 0~1, 굴곡 시 증가
    posture_factor = np.clip(posture_factor, 0.3, 1.0)

    avg_moment_arm = 0.05  # 평균 모멘트 암 5cm

    # 관절 토크 = 하중 × 모멘트암 × 자세계수 × 하중분배계수
    joint_torque = load_force * avg_moment_arm * posture_factor * load_factor

    # 추가 흥분 = 토크 / (최대 근력 × 최적 모멘트 암)
    max_torque = max_isometric_force * avg_moment_arm
    additional_excitation = joint_torque / max_torque

    return np.clip(additional_excitation, 0.0, 0.8)


def calc_load_ligament_strain(base_strain: np.ndarray, load_kg: float,
                              body_mass_kg: float, strain_factor: float) -> np.ndarray:
    """
    외부 하중에 의한 인대 추가 변형률 계산.

    원리: 하중이 체중에 비례하여 인대에 추가 부담
    추가 변형률 = 기본 변형률 × (하중/체중) × 분배계수
    """
    load_ratio = load_kg / body_mass_kg
    additional_strain = base_strain * load_ratio * strain_factor
    return additional_strain


def convert_angles_to_scenario(angle_data: dict, scenario_name: str,
                               excitation_gain: float = 0.008,
                               excitation_base: float = 0.15,
                               load_kg: float = 0.0,
                               body_mass_kg: float = 75.0,
                               task_type: str = 'none') -> dict:
    """
    관절 각도 시계열 → 시뮬레이션 시나리오 변환.

    Parameters
    ----------
    angle_data : 관절 각도 데이터 (PoseAnalyzer 출력)
    scenario_name : 시나리오 이름
    excitation_gain : 각속도 → 흥분 변환 계수
    excitation_base : 기저 흥분 수준
    load_kg : 외부 하중 (kg), 0이면 무부하
    body_mass_kg : 대상자 체중 (kg)
    task_type : 동작 유형 ('lift', 'pull', 'carry', 'push', 'none')
    """
    time = angle_data['time']
    joints = angle_data['joints']
    dt = np.mean(np.diff(time)) if len(time) > 1 else 1.0 / 30.0

    # 하중 프로파일 가져오기
    task_profile = LOAD_TASK_PROFILES.get(task_type, None)
    has_load = load_kg > 0 and task_profile is not None

    if has_load:
        print(f"\n    [하중 모델 적용]")
        print(f"      동작 유형: {task_profile['name']}")
        print(f"      외부 하중: {load_kg} kg ({load_kg * 9.81:.1f} N)")
        print(f"      체중: {body_mass_kg} kg")
        print(f"      하중/체중 비율: {load_kg/body_mass_kg*100:.1f}%")

    muscles = {}
    for muscle_name, mapping in ANGLE_TO_MUSCLE.items():
        joint = mapping['joint']
        if joint not in joints:
            continue
        angles = joints[joint]
        theta_min, theta_max = mapping['theta_range']
        nfl_min, nfl_max = mapping['nfl_range']

        t = np.clip((angles - theta_min) / (theta_max - theta_min), 0, 1)
        nfl = nfl_min + t * (nfl_max - nfl_min)
        nfv = np.gradient(nfl, time) / 10.0

        # 기본 흥분 (각속도 기반)
        ang_vel = np.gradient(angles, dt)
        excitation = excitation_base + excitation_gain * np.abs(ang_vel)

        # 외부 하중에 의한 추가 흥분
        if has_load:
            load_factor = task_profile['muscle_load_factor'].get(muscle_name, 0.1)
            max_force = MUSCLE_PARAMS[muscle_name]['max_isometric_force']
            load_exc = calc_load_excitation(
                angles, load_kg, body_mass_kg, load_factor, max_force
            )
            excitation = excitation + load_exc

        excitation = np.clip(excitation, 0.01, 1.0)

        muscles[muscle_name] = {
            'excitation': excitation,
            'fiber_length': (nfl, nfv),
        }

    ligaments = {}
    for lig_name, mapping in ANGLE_TO_LIGAMENT.items():
        joint = mapping['joint']
        if joint not in joints:
            continue
        angles = joints[joint]
        theta_min, theta_max = mapping['theta_range']
        strain_min, strain_max = mapping['strain_range']
        params = LIGAMENT_PARAMS[lig_name]

        t = np.clip((angles - theta_min) / (theta_max - theta_min), 0, 1)
        strain = strain_min + t * (strain_max - strain_min)

        # 외부 하중에 의한 추가 변형률
        if has_load:
            strain_factor = task_profile['ligament_strain_factor'].get(lig_name, 0.1)
            additional = calc_load_ligament_strain(
                strain, load_kg, body_mass_kg, strain_factor
            )
            strain = strain + additional

        length = params['slack_length'] * (1.0 + strain)
        velocity = np.gradient(length, time)

        ligaments[lig_name] = (length, velocity)

    return {
        'name': scenario_name,
        'time': time,
        'muscles': muscles,
        'ligaments': ligaments,
        'duration': time[-1] - time[0] if len(time) > 1 else 0,
        'load_kg': load_kg,
        'task_type': task_type,
    }


###############################################################################
# 5. 부상 예측 엔진
###############################################################################

MUSCLE_STRESS_THRESHOLDS = {
    '낮음': 100e3, '중간': 250e3, '높음': 400e3, '매우높음': 600e3,
}
LIGAMENT_STRAIN_THRESHOLDS = {
    '낮음': 0.03, '중간': 0.06, '높음': 0.10, '매우높음': 0.15,
}


def classify_risk(value, thresholds):
    levels = list(thresholds.keys())
    vals = list(thresholds.values())
    for i in range(len(vals) - 1, -1, -1):
        if value >= vals[i]:
            return levels[i]
    return '정상'


def run_biomech_simulation(scenario: dict) -> dict:
    """시나리오 데이터로 생체역학 시뮬레이션 실행"""
    time = scenario['time']
    dt = np.mean(np.diff(time)) if len(time) > 1 else 1.0 / 30.0

    muscle_results = {}
    muscle_risks = []

    for muscle_name, profiles in scenario['muscles'].items():
        params = MUSCLE_PARAMS[muscle_name]
        muscle = DeGrooteFregly2016Muscle(name=muscle_name, **params)
        nfl, nfv = profiles['fiber_length']
        result = muscle.simulate(time, profiles['excitation'], nfl, nfv)
        muscle_results[muscle_name] = result

        peak_stress = np.max(result['muscle_stress_Pa'])
        mean_stress = np.mean(result['muscle_stress_Pa'])
        risk = classify_risk(peak_stress, MUSCLE_STRESS_THRESHOLDS)
        fatigue = np.trapz(np.maximum(result['muscle_stress_Pa'] / 250e3, 0)**2, dx=dt)

        muscle_risks.append({
            'muscle_name': muscle_name,
            'peak_stress_kPa': peak_stress / 1000,
            'mean_stress_kPa': mean_stress / 1000,
            'risk_level': risk,
            'fatigue_index': fatigue,
        })

    ligament_results = {}
    ligament_risks = []

    for lig_name, (length, velocity) in scenario['ligaments'].items():
        params = LIGAMENT_PARAMS[lig_name]
        lig = Blankevoort1991Ligament(
            name=lig_name, slack_length=params['slack_length'],
            linear_stiffness=params['linear_stiffness'],
            transition_strain=params['transition_strain'],
            damping_coefficient=params['damping_coefficient'],
        )
        result = lig.simulate(time, length, velocity)
        ligament_results[lig_name] = result

        peak_strain = np.max(result['strain'])
        peak_force = np.max(result['total_force'])
        strain_risk = classify_risk(peak_strain, LIGAMENT_STRAIN_THRESHOLDS)

        ligament_risks.append({
            'ligament_name': lig_name,
            'peak_strain_pct': peak_strain * 100,
            'peak_force_N': peak_force,
            'strain_risk': strain_risk,
        })

    # 신체 부위별 위험도
    region_map = {
        '대퇴사두근': '무릎', '대퇴이두근': '무릎/허벅지', '비복근': '발목/종아리',
        '대둔근': '고관절', '척추기립근': '허리', '삼각근': '어깨',
        '상완이두근': '팔꿈치', '광배근': '어깨/등',
        '전방십자인대(ACL)': '무릎', '후방십자인대(PCL)': '무릎',
        '슬개건': '무릎', '극상인대': '허리',
        '관절와상완인대': '어깨', '요추인대': '허리',
    }
    risk_levels = ['정상', '낮음', '중간', '높음', '매우높음']
    region_scores = {}
    for mr in muscle_risks:
        r = region_map.get(mr['muscle_name'], '기타')
        s = risk_levels.index(mr['risk_level']) if mr['risk_level'] in risk_levels else 0
        region_scores.setdefault(r, []).append(s)
    for lr in ligament_risks:
        r = region_map.get(lr['ligament_name'], '기타')
        s = risk_levels.index(lr['strain_risk']) if lr['strain_risk'] in risk_levels else 0
        region_scores.setdefault(r, []).append(s)

    body_risks = {}
    for region, scores in region_scores.items():
        ms = max(scores)
        body_risks[region] = {'risk_level': risk_levels[ms], 'risk_score': ms}

    return {
        'muscle_results': muscle_results, 'ligament_results': ligament_results,
        'muscle_risks': muscle_risks, 'ligament_risks': ligament_risks,
        'body_risks': body_risks,
    }


###############################################################################
# 6. 시각화
###############################################################################

def plot_joint_angles(angle_data: dict, save_path: str):
    """추출된 관절 각도 시계열"""
    time = angle_data['time']
    joints = angle_data['joints']
    n = len(joints)

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), squeeze=False)
    fig.suptitle('동영상 추출 관절 각도 시계열 (MediaPipe Pose)', fontsize=14, fontweight='bold')

    joint_labels = {
        'Knee_Angle': '무릎 각도', 'Hip_Angle': '고관절 각도',
        'Ankle_Angle': '발목 각도', 'Shoulder_Angle': '어깨 각도',
        'Elbow_Angle': '팔꿈치 각도', 'Trunk_Angle': '몸통 각도',
    }
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    for i, (jname, data) in enumerate(joints.items()):
        ax = axes[i, 0]
        label = joint_labels.get(jname, jname)
        ax.plot(time, data, color=colors[i % len(colors)], linewidth=1)
        ax.set_ylabel(f'{label} (deg)')
        ax.grid(True, alpha=0.3)
        ax.text(0.98, 0.95,
                f'min={data.min():.1f}  max={data.max():.1f}  ROM={data.max()-data.min():.1f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[-1, 0].set_xlabel('시간 (초)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_muscle_stress(sim_results: dict, scenario_name: str, save_path: str):
    """근육 스트레스 시계열 + 위험 임계값"""
    mr = sim_results['muscle_results']
    n = len(mr)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), squeeze=False)
    fig.suptitle(f'근육 내부 압력(스트레스) - {scenario_name}', fontsize=14, fontweight='bold')

    thresh_kpa = {k: v / 1000 for k, v in MUSCLE_STRESS_THRESHOLDS.items()}
    colors_th = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

    for i, (name, data) in enumerate(mr.items()):
        ax = axes[i, 0]
        ax.plot(data['time'], data['muscle_stress_kPa'], 'k-', linewidth=0.7)
        y_max = max(np.max(data['muscle_stress_kPa']) * 1.2, list(thresh_kpa.values())[-1] * 1.1)

        prev = 0
        for j, (lbl, val) in enumerate(thresh_kpa.items()):
            ax.axhspan(prev, val, alpha=0.1, color=colors_th[j])
            ax.axhline(y=val, color=colors_th[j], linestyle='--', linewidth=0.8,
                       label=f'{lbl}: {val:.0f}kPa')
            prev = val

        ax.set_ylabel('스트레스 (kPa)')
        ax.set_title(name, fontsize=11)
        ax.legend(loc='upper right', fontsize=7)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('시간 (초)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ligament_tension(sim_results: dict, scenario_name: str, save_path: str):
    """인대 장력/변형률"""
    lr = sim_results['ligament_results']
    if not lr:
        return
    n = len(lr)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'인대 장력 - {scenario_name}', fontsize=14, fontweight='bold')

    colors_th = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    for i, (name, data) in enumerate(lr.items()):
        axes[i, 0].plot(data['time'], data['total_force'], 'b-', linewidth=0.8)
        axes[i, 0].set_ylabel('장력 (N)')
        axes[i, 0].set_title(f'{name} - 장력')
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(data['time'], data['strain_percent'], 'r-', linewidth=0.8)
        for j, (lbl, val) in enumerate(LIGAMENT_STRAIN_THRESHOLDS.items()):
            axes[i, 1].axhline(y=val * 100, color=colors_th[j], linestyle='--',
                               linewidth=0.8, label=f'{lbl}: {val*100:.0f}%')
        axes[i, 1].set_ylabel('변형률 (%)')
        axes[i, 1].set_title(f'{name} - 변형률')
        axes[i, 1].legend(fontsize=7)
        axes[i, 1].grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('시간 (초)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_body_risk_chart(sim_results: dict, scenario_name: str, save_path: str):
    """신체 부위별 위험도 + 근육 스트레스 막대"""
    body_risks = sim_results['body_risks']
    muscle_risks = sim_results['muscle_risks']

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(body_risks) * 0.8)))
    fig.suptitle(f'부상 위험도 종합 - {scenario_name}', fontsize=14, fontweight='bold')

    # 신체 부위
    regions = list(body_risks.keys())
    scores = [body_risks[r]['risk_score'] for r in regions]
    rc = {0: '#27ae60', 1: '#2ecc71', 2: '#f1c40f', 3: '#e67e22', 4: '#e74c3c'}
    axes[0].barh(regions, scores, color=[rc[s] for s in scores], edgecolor='gray')
    axes[0].set_xlim(0, 5)
    axes[0].set_xlabel('위험 점수')
    axes[0].set_title('신체 부위별 위험도')
    rl = ['정상', '낮음', '중간', '높음', '매우높음']
    for i, (r, s) in enumerate(zip(regions, scores)):
        axes[0].text(s + 0.1, i, rl[s], va='center', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='x')

    # 근육 스트레스
    names = [m['muscle_name'] for m in muscle_risks]
    peaks = [m['peak_stress_kPa'] for m in muscle_risks]
    tv = list(MUSCLE_STRESS_THRESHOLDS.values())
    sc = []
    for p in peaks:
        pp = p * 1000
        if pp >= tv[3]: sc.append('#e74c3c')
        elif pp >= tv[2]: sc.append('#e67e22')
        elif pp >= tv[1]: sc.append('#f1c40f')
        elif pp >= tv[0]: sc.append('#2ecc71')
        else: sc.append('#27ae60')
    axes[1].barh(names, peaks, color=sc, edgecolor='gray')
    axes[1].set_xlabel('최대 스트레스 (kPa)')
    axes[1].set_title('근육별 최대 내부 압력')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(scenario_name: str, sim_results: dict) -> str:
    """한국어 보고서"""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  부상 위험도 분석 보고서 - {scenario_name}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("[근육 내부 압력 분석]")
    lines.append(f"{'근육명':<15} {'최대스트레스':>12} {'평균스트레스':>12} {'위험도':>8} {'피로지수':>8}")
    lines.append("-" * 55)
    for mr in sim_results['muscle_risks']:
        lines.append(
            f"{mr['muscle_name']:<15} "
            f"{mr['peak_stress_kPa']:>10.1f}kPa "
            f"{mr['mean_stress_kPa']:>10.1f}kPa "
            f"{mr['risk_level']:>8} "
            f"{mr['fatigue_index']:>8.2f}"
        )
    lines.append("")

    lines.append("[인대 장력 분석]")
    lines.append(f"{'인대명':<20} {'최대변형률':>8} {'최대장력':>10} {'위험도':>8}")
    lines.append("-" * 46)
    for lr in sim_results['ligament_risks']:
        lines.append(
            f"{lr['ligament_name']:<20} "
            f"{lr['peak_strain_pct']:>6.2f}% "
            f"{lr['peak_force_N']:>8.1f}N "
            f"{lr['strain_risk']:>8}"
        )
    lines.append("")

    lines.append("[신체 부위별 종합 위험도]")
    lines.append("-" * 40)
    rl = ['정상', '낮음', '중간', '높음', '매우높음']
    for region, info in sim_results['body_risks'].items():
        bar = "#" * info['risk_score'] + "-" * (4 - info['risk_score'])
        lines.append(f"  {region:<12} [{bar}] {info['risk_level']}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def export_csv(sim_results: dict, scenario_name: str, output_dir: str):
    """결과 CSV 저장"""
    fp = os.path.join(output_dir, 'video_muscle_results.csv')
    with open(fp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['시나리오', '근육명', '최대_스트레스(kPa)', '평균_스트레스(kPa)',
                     '위험도', '피로누적지수'])
        for mr in sim_results['muscle_risks']:
            w.writerow([scenario_name, mr['muscle_name'],
                        f"{mr['peak_stress_kPa']:.1f}", f"{mr['mean_stress_kPa']:.1f}",
                        mr['risk_level'], f"{mr['fatigue_index']:.4f}"])

    fp = os.path.join(output_dir, 'video_ligament_results.csv')
    with open(fp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['시나리오', '인대명', '최대_변형률(%)', '최대_장력(N)', '위험도'])
        for lr in sim_results['ligament_risks']:
            w.writerow([scenario_name, lr['ligament_name'],
                        f"{lr['peak_strain_pct']:.2f}", f"{lr['peak_force_N']:.1f}",
                        lr['strain_risk']])

    fp = os.path.join(output_dir, 'video_body_risks.csv')
    with open(fp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['시나리오', '신체부위', '위험도', '위험점수'])
        for region, info in sim_results['body_risks'].items():
            w.writerow([scenario_name, region, info['risk_level'], info['risk_score']])

def export_joint_angles_csv(angle_data: dict, output_dir: str):
    """관절 각도 시계열 CSV 저장 (Kinovea 호환 형식)"""
    fp = os.path.join(output_dir, 'video_joint_angles.csv')
    time = angle_data['time']
    joints = angle_data['joints']
    joint_names = list(joints.keys())

    with open(fp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Time(ms)'] + joint_names)
        for i in range(len(time)):
            row = [f"{time[i] * 1000:.0f}"]
            for jn in joint_names:
                row.append(f"{joints[jn][i]:.1f}")
            w.writerow(row)
    print(f"  관절 각도 CSV 저장: {fp}")


###############################################################################
# 7. 메인 실행
###############################################################################

def main():
    print("=" * 70)
    print("  소방관 동영상 기반 부상 예측 시스템")
    print("  MediaPipe Pose + OpenSim 생체역학 모델")
    print("  (역동역학 기반 외부 하중 지원)")
    print("=" * 70)

    if len(sys.argv) < 2:
        print("\n사용법:")
        print("  python video_injury_predictor.py <동영상> [시나리오명] [하중kg] [동작유형] [체중kg]")
        print("\n예시:")
        print('  python video_injury_predictor.py fire.mp4 "장비운반" 25 carry')
        print('  python video_injury_predictor.py fire.mp4 "호스당기기" 30 pull')
        print('  python video_injury_predictor.py fire.mp4 "환자들기" 80 lift 70')
        print('  python video_injury_predictor.py fire.mp4 "문밀기" 50 push')
        print('  python video_injury_predictor.py fire.mp4 "맨몸동작"          (하중 없음)')
        print("  python video_injury_predictor.py 0                             (웹캠)")
        print("\n동작 유형:")
        print("  lift  : 들기 (바닥에서 물체 들어올리기)")
        print("  pull  : 끌기 (호스, 장비 당기기)")
        print("  carry : 운반 (물체 들고 이동)")
        print("  push  : 밀기 (장비, 문 밀기)")
        print("\n지원 형식: mp4, avi, mov, mkv 등")
        sys.exit(0)

    video_path = sys.argv[1]
    scenario_name = sys.argv[2] if len(sys.argv) >= 3 else '동영상 분석'
    load_kg = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.0
    task_type = sys.argv[4] if len(sys.argv) >= 5 else 'none'
    body_mass_kg = float(sys.argv[5]) if len(sys.argv) >= 6 else 75.0

    print(f"\n  입력: {video_path}")
    print(f"  시나리오: {scenario_name}")
    if load_kg > 0:
        task_info = LOAD_TASK_PROFILES.get(task_type, {})
        print(f"  외부 하중: {load_kg} kg ({load_kg * 9.81:.1f} N)")
        print(f"  동작 유형: {task_info.get('name', task_type)}")
        print(f"  체중: {body_mass_kg} kg")
    else:
        print(f"  외부 하중: 없음 (맨몸)")

    # [1] 동영상 로드
    print(f"\n[1/5] 동영상 로드...")
    processor = VideoProcessor(video_path)

    # [2] 포즈 분석 및 관절 각도 추출
    print(f"\n[2/5] MediaPipe Pose 분석...")
    analyzer = PoseAnalyzer(model_complexity=1)
    angle_data = processor.extract_angles(
        analyzer,
        save_annotated_video=True,
        output_video_path=os.path.join(OUTPUT_DIR, 'pose_annotated.mp4'),
    )
    analyzer.close()

    # 관절 각도 CSV 저장 (Kinovea 호환)
    export_joint_angles_csv(angle_data, OUTPUT_DIR)

    # [3] 관절 각도 → 생체역학 시뮬레이션
    print(f"\n[3/5] 생체역학 시뮬레이션...")
    scenario = convert_angles_to_scenario(
        angle_data, scenario_name,
        load_kg=load_kg, body_mass_kg=body_mass_kg, task_type=task_type
    )
    print(f"    근육 {len(scenario['muscles'])}개, 인대 {len(scenario['ligaments'])}개 변환 완료")

    sim_results = run_biomech_simulation(scenario)
    print(f"    시뮬레이션 완료")

    # 시나리오 이름에 하중 정보 추가
    display_name = scenario_name
    if load_kg > 0:
        task_info = LOAD_TASK_PROFILES.get(task_type, {})
        display_name = f"{scenario_name} ({task_info.get('name', task_type)}, {load_kg:.0f}kg)"

    # [4] 시각화
    print(f"\n[4/5] 시각화 생성...")
    plot_joint_angles(angle_data, os.path.join(OUTPUT_DIR, 'joint_angles.png'))
    print(f"  저장: joint_angles.png")
    plot_muscle_stress(sim_results, display_name,
                       os.path.join(OUTPUT_DIR, 'muscle_stress.png'))
    print(f"  저장: muscle_stress.png")
    plot_ligament_tension(sim_results, display_name,
                          os.path.join(OUTPUT_DIR, 'ligament_tension.png'))
    print(f"  저장: ligament_tension.png")
    plot_body_risk_chart(sim_results, display_name,
                         os.path.join(OUTPUT_DIR, 'body_risk_chart.png'))
    print(f"  저장: body_risk_chart.png")

    # [5] CSV 출력
    print(f"\n[5/5] CSV 결과 저장...")
    export_csv(sim_results, display_name, OUTPUT_DIR)

    # 보고서
    report = generate_report(display_name, sim_results)
    print("\n" + report)

    print(f"\n모든 결과가 {OUTPUT_DIR} 에 저장되었습니다.")
    print("=" * 70)
    print("  출력 파일 목록:")
    print("    - pose_annotated.mp4   : 포즈 랜드마크 표시된 영상")
    print("    - joint_angles.png     : 관절 각도 시계열 그래프")
    print("    - muscle_stress.png    : 근육 내부 압력 그래프")
    print("    - ligament_tension.png : 인대 장력/변형률 그래프")
    print("    - body_risk_chart.png  : 신체 부위별 위험도 차트")
    print("    - video_joint_angles.csv    : 관절 각도 CSV (Kinovea 호환)")
    print("    - video_muscle_results.csv  : 근육 분석 결과")
    print("    - video_ligament_results.csv: 인대 분석 결과")
    print("    - video_body_risks.csv      : 신체 부위별 위험도")
    print("=" * 70)


if __name__ == '__main__':
    main()
