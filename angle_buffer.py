"""
angle_buffer.py - 롤링 윈도우 관절 각도 버퍼
=============================================

이 모듈은 실시간 포즈 분석에서 시간 순서대로 수집된 관절 각도 데이터를
고정 크기 버퍼(deque)에 저장하고 관리합니다.

핵심 개념:
    롤링 윈도우(Rolling Window)란 최근 N개의 데이터만 유지하는 자료구조입니다.
    새 데이터가 들어오면 가장 오래된 데이터가 자동으로 제거됩니다.
    Python의 collections.deque(maxlen=N)을 사용하면 O(1) 시간에 이 동작이 수행됩니다.

    예: maxlen=60, 30fps 카메라
        → 항상 최근 2초간의 관절 각도 데이터를 유지
        → 이 2초 분량의 데이터가 생체역학 시뮬레이션의 입력이 됨

데이터 흐름:
    1. PoseAnalyzer가 프레임에서 6개 관절 각도를 추출
    2. main.py가 push(timestamp, angles)로 버퍼에 추가
    3. 버퍼가 가득 차면 get_window()로 numpy 배열 형태로 추출
    4. BiomechEngine이 이 윈도우 데이터로 시뮬레이션 실행

녹화 모드:
    start_recording() 호출 시 버퍼에 들어오는 모든 데이터를 별도 리스트에도 저장합니다.
    stop_recording() 후 get_full_history()로 전체 세션 데이터를 추출하여
    CSV로 저장할 수 있습니다.
"""

import time
import numpy as np
from collections import deque
from typing import Optional, Dict, List

from config import ROLLING_WINDOW_SIZE


class AngleBuffer:
    """
    deque 기반 고정 크기 관절 각도 버퍼.

    내부 구조:
        _buffer: deque(maxlen=N) ─ 최근 N개 프레임의 {timestamp, angles} 딕셔너리
        _full_history: list ─ 녹화 모드에서 전체 세션의 모든 프레임 저장
        _recording: bool ─ 녹화 모드 활성화 여부

    사용 예시:
        buf = AngleBuffer(max_size=60)
        buf.push(0.033, {'Knee_Angle': 145.2, 'Hip_Angle': 160.1, ...})
        buf.push(0.066, {'Knee_Angle': 144.8, 'Hip_Angle': 159.5, ...})
        ...
        if buf.is_full():
            window = buf.get_window()
            # window = {'time': np.array([...]), 'joints': {'Knee_Angle': np.array([...]), ...}}
    """

    def __init__(self, max_size: int = ROLLING_WINDOW_SIZE):
        """
        Parameters
        ----------
        max_size : int
            버퍼 최대 크기 (프레임 수).
            기본값은 config.ROLLING_WINDOW_SIZE (60프레임 = 30fps에서 2초).
            값이 클수록 더 긴 시간 구간을 분석하지만, 메모리 사용량이 증가하고
            시뮬레이션 시간이 길어집니다.
        """
        self._max_size = max_size
        # deque의 maxlen 속성: 크기 초과 시 가장 오래된 항목을 자동 제거
        self._buffer = deque(maxlen=max_size)
        self._full_history = []   # 녹화 모드에서 전체 이력 저장용
        self._recording = False   # 녹화 모드 플래그

    def push(self, timestamp: float, angles: Dict[str, float]):
        """
        한 프레임의 관절 각도 데이터를 버퍼에 추가합니다.

        Parameters
        ----------
        timestamp : float
            프레임 타임스탬프 (초 단위). 세션 시작부터의 경과 시간.
            예: time.time() - session_start
        angles : Dict[str, float]
            관절 각도 딕셔너리. PoseAnalyzer.process_frame()의 반환값.
            예: {'Knee_Angle': 145.2, 'Hip_Angle': 160.1, 'Ankle_Angle': 88.3,
                 'Shoulder_Angle': 45.0, 'Elbow_Angle': 130.5, 'Trunk_Angle': 170.2}

        동작 과정:
            1. angles.copy()로 딕셔너리 복사 (원본 변경 방지)
            2. deque에 append (maxlen 초과 시 가장 오래된 항목 자동 제거)
            3. 녹화 모드이면 _full_history에도 추가 (이쪽은 크기 제한 없음)
        """
        entry = {'timestamp': timestamp, 'angles': angles.copy()}
        self._buffer.append(entry)
        if self._recording:
            self._full_history.append(entry)

    def get_window(self) -> Optional[dict]:
        """
        현재 롤링 윈도우의 데이터를 numpy 배열 형태로 반환합니다.

        Returns
        -------
        dict 또는 None
            버퍼에 2개 이상의 프레임이 있으면:
            {
                'time': np.ndarray ─ 타임스탬프 배열 (shape: [N])
                'joints': {
                    'Knee_Angle': np.ndarray,    ─ 각 관절의 각도 시계열 (shape: [N])
                    'Hip_Angle': np.ndarray,
                    ...
                }
            }
            버퍼가 비어있거나 1개 이하이면 None.

        Note:
            최소 2개 프레임이 필요한 이유: np.gradient()로 각속도를 계산하려면
            최소 2개 시점이 필요하기 때문입니다.
        """
        if len(self._buffer) < 2:
            return None

        # deque를 리스트로 변환하여 인덱싱 가능하게 함
        entries = list(self._buffer)
        time_arr = np.array([e['timestamp'] for e in entries])

        # 가장 최근 프레임의 관절 이름을 기준으로 배열 생성
        # (중간에 감지 실패로 관절 이름이 바뀔 수 있으므로 최근 기준)
        joint_names = list(entries[-1]['angles'].keys())
        joints = {}
        for jn in joint_names:
            # 각 관절의 전체 시계열을 numpy 배열로 변환
            # 특정 프레임에서 해당 관절이 없으면 0.0으로 대체
            joints[jn] = np.array([
                e['angles'].get(jn, 0.0) for e in entries
            ])

        return {'time': time_arr, 'joints': joints}

    def is_full(self) -> bool:
        """버퍼가 최대 크기에 도달했는지 여부. True이면 분석 제출 가능."""
        return len(self._buffer) >= self._max_size

    @property
    def size(self) -> int:
        """현재 버퍼에 저장된 프레임 수."""
        return len(self._buffer)

    @property
    def max_size(self) -> int:
        """버퍼 최대 크기."""
        return self._max_size

    # =========================================================================
    # 녹화 모드 (Recording Mode)
    # =========================================================================
    # 녹화 모드는 롤링 윈도우와 별개로 전체 세션 데이터를 저장합니다.
    # 롤링 윈도우는 최근 N프레임만 유지하지만, 녹화 모드에서는
    # 시작부터 종료까지의 모든 프레임이 _full_history에 축적됩니다.

    def start_recording(self):
        """
        녹화 시작. 현재 버퍼의 내용을 초기 이력으로 복사하고,
        이후 push()되는 모든 데이터를 _full_history에도 추가합니다.
        """
        self._recording = True
        # 현재 버퍼에 이미 있는 데이터도 이력에 포함
        self._full_history = list(self._buffer)

    def stop_recording(self):
        """녹화 중지. _full_history는 유지되어 get_full_history()로 접근 가능."""
        self._recording = False

    @property
    def is_recording(self) -> bool:
        """녹화 모드 활성화 여부."""
        return self._recording

    def get_full_history(self) -> Optional[dict]:
        """
        녹화된 전체 세션 데이터를 numpy 배열 형태로 반환합니다.

        Returns
        -------
        dict 또는 None
            get_window()와 동일한 형식이지만, 전체 녹화 구간의 데이터를 포함합니다.
            녹화 데이터가 없으면 None.

        Note:
            이 데이터는 CSV로 저장되어 Kinovea, Excel 등에서 분석할 수 있습니다.
        """
        if not self._full_history:
            return None

        entries = self._full_history
        time_arr = np.array([e['timestamp'] for e in entries])
        joint_names = list(entries[-1]['angles'].keys())
        joints = {}
        for jn in joint_names:
            joints[jn] = np.array([
                e['angles'].get(jn, 0.0) for e in entries
            ])
        return {'time': time_arr, 'joints': joints}

    def clear(self):
        """버퍼와 녹화 이력을 모두 초기화합니다."""
        self._buffer.clear()
        self._full_history = []
        self._recording = False
