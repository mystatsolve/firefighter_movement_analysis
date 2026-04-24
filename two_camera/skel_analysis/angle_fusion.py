"""
angle_fusion.py - 듀얼 카메라 신뢰도 가중 퓨전 엔진 (핵심 신규 모듈)
======================================================================

2대의 카메라에서 독립적으로 추출한 관절 각도를 visibility(신뢰도) 기반
가중 평균으로 퓨전하여 최종 관절 각도를 산출합니다.

정확도 향상 원리:
    1단계 - 양측(Bilateral) 퓨전:
        각 카메라 내에서 left/right 관절을 가중 평균 (pose_analyzer.py에서 처리)

    2단계 - 카메라간(Inter-Camera) 퓨전:
        카메라 1과 카메라 2의 같은 관절 각도를 visibility 가중 평균
        수식: fused = w1 * angle_cam1 + w2 * angle_cam2
              w1 = vis1 / (vis1 + vis2 + epsilon)

    3단계 - 이동 평균 스무딩:
        퓨전된 각도에 N프레임 이동 평균을 적용하여 프레임간 노이즈 제거

추가 기능:
    - 롤링 버퍼 (deque, 60프레임): 생체역학 분석용 시계열 데이터 유지
    - 녹화 기능: 전체 세션의 퓨전 결과를 메모리에 저장

사용 예시:
    engine = AngleFusionEngine()
    fused = engine.fuse(angles_cam1, vis_cam1, angles_cam2, vis_cam2)
    # fused: {'Knee_Angle': 142.5, 'Hip_Angle': 165.3, ...}
    window = engine.get_window_data()  # 생체역학 분석용 시계열
"""

import time
import numpy as np
from collections import deque
from typing import Dict, Optional, List, Tuple

from config import (
    ROLLING_WINDOW_SIZE, SMOOTHING_WINDOW,
    JOINT_LABELS,
)


class AngleFusionEngine:
    """
    듀얼 카메라 관절 각도 퓨전 엔진.

    2대 카메라의 관절 각도를 visibility 가중 평균으로 퓨전하고,
    이동 평균 스무딩을 적용하여 최종 관절 각도를 산출합니다.

    Attributes
    ----------
    frame_count : int
        처리된 프레임 수
    is_recording : bool
        녹화 중 여부
    """

    def __init__(self, smoothing_window=None, rolling_window_size=None):
        """
        Parameters
        ----------
        smoothing_window : int, optional
            이동 평균 윈도우 크기. None이면 config 기본값 사용.
        rolling_window_size : int, optional
            롤링 버퍼 크기. None이면 config 기본값 사용.
        """
        self._smooth_n = smoothing_window or SMOOTHING_WINDOW
        self._window_size = rolling_window_size or ROLLING_WINDOW_SIZE

        # 이동 평균용 히스토리 버퍼 (관절별)
        # {angle_name: deque(maxlen=smooth_n)}
        self._smooth_buffers: Dict[str, deque] = {}

        # 생체역학 분석용 롤링 윈도우 (시계열)
        # {'time': deque, 'joints': {name: deque}}
        self._rolling_time = deque(maxlen=self._window_size)
        self._rolling_joints: Dict[str, deque] = {}

        # 녹화 데이터
        self._recording = False
        self._record_data: List[Dict] = []

        # 카운터
        self._frame_count = 0
        self._start_time = time.time()

        # 최신 퓨전 결과 캐시
        self._latest_fused: Optional[Dict[str, float]] = None
        self._latest_confidence: Optional[Dict[str, float]] = None

    @property
    def frame_count(self) -> int:
        """처리된 프레임 수."""
        return self._frame_count

    @property
    def is_recording(self) -> bool:
        """녹화 중 여부."""
        return self._recording

    def fuse(self,
             angles_cam1: Optional[Dict[str, float]],
             vis_cam1: Optional[Dict[str, float]],
             angles_cam2: Optional[Dict[str, float]] = None,
             vis_cam2: Optional[Dict[str, float]] = None,
             ) -> Optional[Dict[str, float]]:
        """
        2대 카메라의 관절 각도를 퓨전합니다.

        퓨전 과정:
        1. 카메라간 visibility 가중 평균
        2. 이동 평균 스무딩
        3. 롤링 버퍼에 저장
        4. (녹화 중이면) 녹화 데이터에 추가

        Parameters
        ----------
        angles_cam1 : Optional[Dict[str, float]]
            카메라 1의 관절 각도. None이면 카메라 1 미검출.
        vis_cam1 : Optional[Dict[str, float]]
            카메라 1의 관절별 visibility (0~1).
        angles_cam2 : Optional[Dict[str, float]]
            카메라 2의 관절 각도. None이면 단일 카메라 모드.
        vis_cam2 : Optional[Dict[str, float]]
            카메라 2의 관절별 visibility.

        Returns
        -------
        Optional[Dict[str, float]]
            퓨전+스무딩된 최종 관절 각도. 두 카메라 모두 미검출이면 None.
        """
        # 두 카메라 모두 미검출 → None
        if angles_cam1 is None and angles_cam2 is None:
            return None

        # ================================================================
        # 카메라간 퓨전 (Inter-Camera Fusion)
        # ================================================================
        # 수식: fused = w1 * angle1 + w2 * angle2
        #        w1 = vis1 / (vis1 + vis2 + eps)
        #        w2 = vis2 / (vis1 + vis2 + eps)
        # ================================================================
        eps = 1e-6
        fused = {}
        confidence = {}

        # 모든 관절 이름 수집
        all_joints = set()
        if angles_cam1:
            all_joints.update(angles_cam1.keys())
        if angles_cam2:
            all_joints.update(angles_cam2.keys())

        for joint in all_joints:
            a1 = angles_cam1.get(joint) if angles_cam1 else None
            a2 = angles_cam2.get(joint) if angles_cam2 else None
            v1 = vis_cam1.get(joint, 0.0) if vis_cam1 and a1 is not None else 0.0
            v2 = vis_cam2.get(joint, 0.0) if vis_cam2 and a2 is not None else 0.0

            if a1 is not None and a2 is not None:
                # 양쪽 카메라 모두 유효 → 가중 평균
                w1 = v1 / (v1 + v2 + eps)
                w2 = v2 / (v1 + v2 + eps)
                fused[joint] = w1 * a1 + w2 * a2
                confidence[joint] = (v1 + v2) / 2.0
            elif a1 is not None:
                fused[joint] = a1
                confidence[joint] = v1
            elif a2 is not None:
                fused[joint] = a2
                confidence[joint] = v2

        if not fused:
            return None

        # ================================================================
        # 이동 평균 스무딩 (Moving Average Smoothing)
        # ================================================================
        # 최근 N프레임의 퓨전 결과를 평균하여 프레임간 떨림(jitter) 제거
        # ================================================================
        smoothed = {}
        for joint, val in fused.items():
            if joint not in self._smooth_buffers:
                self._smooth_buffers[joint] = deque(maxlen=self._smooth_n)
            self._smooth_buffers[joint].append(val)
            smoothed[joint] = float(np.mean(self._smooth_buffers[joint]))

        # ================================================================
        # 롤링 윈도우에 저장 (생체역학 분석용)
        # ================================================================
        current_time = time.time() - self._start_time
        self._rolling_time.append(current_time)
        for joint, val in smoothed.items():
            if joint not in self._rolling_joints:
                self._rolling_joints[joint] = deque(maxlen=self._window_size)
            self._rolling_joints[joint].append(val)

        # 녹화 중이면 데이터 저장
        if self._recording:
            record = {
                'time': current_time,
                'frame': self._frame_count,
            }
            record.update(smoothed)
            # confidence도 저장
            for joint, conf in confidence.items():
                record[f'{joint}_conf'] = conf
            self._record_data.append(record)

        # 캐시 업데이트
        self._latest_fused = smoothed
        self._latest_confidence = confidence
        self._frame_count += 1

        return smoothed

    def get_window_data(self) -> Optional[Dict]:
        """
        생체역학 분석용 롤링 윈도우 데이터를 반환합니다.

        BiomechEngine.submit_analysis()에 전달할 수 있는 형식으로
        시계열 데이터를 변환합니다.

        Returns
        -------
        Optional[Dict]
            {'time': np.ndarray, 'joints': {name: np.ndarray}}
            데이터가 부족하면 None.
        """
        if len(self._rolling_time) < 2:
            return None

        time_arr = np.array(self._rolling_time)
        joints = {}
        for name, buf in self._rolling_joints.items():
            if len(buf) == len(self._rolling_time):
                joints[name] = np.array(buf)

        if not joints:
            return None

        return {'time': time_arr, 'joints': joints}

    def get_latest(self) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """
        최신 퓨전 결과와 confidence를 반환합니다.

        Returns
        -------
        Tuple[Optional[Dict], Optional[Dict]]
            (fused_angles, confidence_scores)
        """
        return self._latest_fused, self._latest_confidence

    def start_recording(self):
        """녹화 시작. 기존 데이터 초기화."""
        self._recording = True
        self._record_data = []
        print("[AngleFusion] 녹화 시작")

    def stop_recording(self) -> List[Dict]:
        """
        녹화 중지 및 데이터 반환.

        Returns
        -------
        List[Dict]
            녹화된 프레임별 데이터 리스트
        """
        self._recording = False
        data = list(self._record_data)
        print(f"[AngleFusion] 녹화 중지 ({len(data)} 프레임)")
        return data

    def get_recording_data(self) -> List[Dict]:
        """현재까지 녹화된 데이터 반환 (녹화 중단 없이)."""
        return list(self._record_data)

    def get_stats(self) -> Dict:
        """
        퓨전 엔진 통계 정보 반환.

        Returns
        -------
        Dict
            {'frame_count', 'buffer_size', 'recording', 'record_frames', 'elapsed'}
        """
        return {
            'frame_count': self._frame_count,
            'buffer_size': len(self._rolling_time),
            'recording': self._recording,
            'record_frames': len(self._record_data),
            'elapsed': time.time() - self._start_time,
        }
