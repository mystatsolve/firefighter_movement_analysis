"""
realtime_display.py - 듀얼 카메라 실시간 HUD 디스플레이 모듈
=============================================================

2대 카메라의 분할 화면 + 퓨전 관절 각도 + 위험도를 실시간으로 표시합니다.

화면 레이아웃:
    +------------------+------------------+------------------+
    |   카메라 1        |   카메라 2        |   분석 패널      |
    |   (640x480)      |   (640x480)      |   (320x480)     |
    |   스켈레톤+각도   |   스켈레톤+각도   |   위험도 바차트  |
    +------------------+------------------+------------------+
    |        하단 상태 바: FPS, 카메라수, 감지율, 녹화 상태   |
    +--------------------------------------------------------+

단일 카메라 모드:
    +------------------------------------+------------------+
    |   카메라 1 (640x480)               |   분석 패널      |
    +------------------------------------+------------------+
    |            하단 상태 바                                |
    +-------------------------------------------------------+
"""

import cv2
import numpy as np
import time
from typing import Dict, Optional, List

from config import (
    OAKD_RESOLUTION_W, OAKD_RESOLUTION_H,
    SIDE_PANEL_WIDTH, BOTTOM_BAR_HEIGHT,
    WINDOW_NAME, RISK_COLORS, RISK_LEVELS,
    JOINT_LABELS,
)


class RealtimeDisplay:
    """
    듀얼 카메라 실시간 HUD 디스플레이.

    OpenCV 기반으로 카메라 영상, 포즈 스켈레톤, 퓨전 각도, 위험도를
    실시간으로 시각화합니다.
    """

    def __init__(self, is_dual=True):
        """
        Parameters
        ----------
        is_dual : bool
            듀얼 카메라 모드 여부. False면 단일 카메라 레이아웃.
        """
        self._is_dual = is_dual
        self._cam_w = OAKD_RESOLUTION_W
        self._cam_h = OAKD_RESOLUTION_H
        self._panel_w = SIDE_PANEL_WIDTH
        self._bar_h = BOTTOM_BAR_HEIGHT

        # 전체 캔버스 크기 계산
        num_cams = 2 if is_dual else 1
        self._total_w = self._cam_w * num_cams + self._panel_w
        self._total_h = self._cam_h + self._bar_h

        # FPS 계산용
        self._fps_time = time.time()
        self._fps_count = 0
        self._current_fps = 0.0

    def _update_fps(self):
        """FPS 업데이트 (매 프레임 호출)."""
        self._fps_count += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 1.0:
            self._current_fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_time = time.time()

    def _draw_side_panel(self, panel: np.ndarray,
                         fused_angles: Optional[Dict[str, float]],
                         confidence: Optional[Dict[str, float]],
                         analysis_result: Optional[Dict]):
        """
        우측 분석 패널을 그립니다.

        패널 구성:
        - 상단: 퓨전된 관절 각도 + confidence 바
        - 중앙: 부위별 위험도 바 차트
        - 하단: 종합 위험도 표시

        Parameters
        ----------
        panel : np.ndarray
            사이드 패널 캔버스 (panel_w x cam_h)
        fused_angles : Dict
            퓨전된 관절 각도
        confidence : Dict
            관절별 신뢰도 점수
        analysis_result : Dict
            생체역학 분석 결과
        """
        # 배경: 어두운 회색
        panel[:] = (30, 30, 30)
        y = 25

        # === 제목 ===
        cv2.putText(panel, "Fused Joint Angles", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 5
        cv2.line(panel, (10, y), (self._panel_w - 10, y), (100, 100, 100), 1)
        y += 20

        # === 퓨전 관절 각도 ===
        if fused_angles:
            for angle_name, angle_val in fused_angles.items():
                label = JOINT_LABELS.get(angle_name, angle_name)
                conf = confidence.get(angle_name, 0.0) if confidence else 0.0

                # 관절 이름 + 각도값
                cv2.putText(panel, f"{label}:", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(panel, f"{angle_val:.1f}", (100, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

                # confidence 바 (최대 100px 너비)
                bar_x = 170
                bar_w = int(conf * 100)
                bar_color = (0, 200, 0) if conf > 0.6 else (0, 200, 255) if conf > 0.3 else (0, 0, 200)
                cv2.rectangle(panel, (bar_x, y - 10), (bar_x + bar_w, y - 2), bar_color, -1)
                cv2.rectangle(panel, (bar_x, y - 10), (bar_x + 100, y - 2), (80, 80, 80), 1)
                cv2.putText(panel, f"{conf:.0%}", (bar_x + 105, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
                y += 22
        else:
            cv2.putText(panel, "No pose detected", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
            y += 22

        y += 10
        cv2.line(panel, (10, y), (self._panel_w - 10, y), (100, 100, 100), 1)
        y += 20

        # === 부위별 위험도 ===
        cv2.putText(panel, "Body Region Risks", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 25

        if analysis_result and 'body_risks' in analysis_result:
            for region, info in analysis_result['body_risks'].items():
                risk_level = info['risk_level']
                risk_score = info['risk_score']
                color = RISK_COLORS.get(risk_level, (128, 128, 128))

                # 부위 이름
                display_name = region[:12]
                cv2.putText(panel, f"{display_name}:", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

                # 위험도 바 (점수 기반, 최대 4)
                bar_x = 120
                bar_max_w = 120
                bar_w = int(risk_score / 4.0 * bar_max_w) if risk_score > 0 else 2
                cv2.rectangle(panel, (bar_x, y - 10), (bar_x + bar_w, y - 2), color, -1)
                cv2.rectangle(panel, (bar_x, y - 10), (bar_x + bar_max_w, y - 2), (80, 80, 80), 1)

                # 위험 레벨 텍스트
                cv2.putText(panel, risk_level, (bar_x + bar_max_w + 5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                y += 22
        else:
            cv2.putText(panel, "Analyzing...", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1, cv2.LINE_AA)
            y += 22

        y += 10
        cv2.line(panel, (10, y), (self._panel_w - 10, y), (100, 100, 100), 1)
        y += 20

        # === 종합 위험도 ===
        if analysis_result:
            overall = analysis_result.get('overall_risk', 'Normal')
            color = RISK_COLORS.get(overall, (128, 128, 128))
            cv2.putText(panel, "Overall Risk:", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 30
            cv2.putText(panel, overall, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    def _draw_bottom_bar(self, bar: np.ndarray,
                         num_cameras: int,
                         detection_rates: List[float],
                         is_recording: bool,
                         analysis_count: int):
        """
        하단 상태 바를 그립니다.

        표시 항목: FPS | 카메라 수 | 양쪽 감지율 | 녹화 상태 | 분석 횟수

        Parameters
        ----------
        bar : np.ndarray
            하단 바 캔버스
        num_cameras : int
            활성 카메라 수
        detection_rates : List[float]
            각 카메라의 감지율 (0~1)
        is_recording : bool
            녹화 중 여부
        analysis_count : int
            누적 분석 횟수
        """
        bar[:] = (40, 40, 40)
        y = 28

        # FPS
        cv2.putText(bar, f"FPS: {self._current_fps:.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

        # 카메라 수
        cam_color = (0, 255, 0) if num_cameras >= 2 else (0, 200, 255)
        cv2.putText(bar, f"Cam: {num_cameras}", (130, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cam_color, 1, cv2.LINE_AA)

        # 감지율
        x = 230
        for i, rate in enumerate(detection_rates):
            color = (0, 200, 0) if rate > 0.7 else (0, 200, 255) if rate > 0.3 else (0, 0, 200)
            cv2.putText(bar, f"Det{i + 1}: {rate:.0%}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            x += 120

        # 녹화 상태
        if is_recording:
            # 깜빡이는 빨간 점
            blink = int(time.time() * 2) % 2 == 0
            if blink:
                cv2.circle(bar, (x + 10, y - 5), 6, (0, 0, 255), -1)
            cv2.putText(bar, "REC", (x + 22, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            x += 70

        # 분석 횟수
        cv2.putText(bar, f"Analysis: {analysis_count}", (x + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def render(self,
               frames: List[Optional[np.ndarray]],
               annotated_frames: List[Optional[np.ndarray]],
               fused_angles: Optional[Dict[str, float]],
               confidence: Optional[Dict[str, float]],
               analysis_result: Optional[Dict],
               num_cameras: int = 1,
               detection_rates: Optional[List[float]] = None,
               is_recording: bool = False,
               analysis_count: int = 0,
               ) -> np.ndarray:
        """
        전체 HUD 화면을 렌더링합니다.

        Parameters
        ----------
        frames : List[Optional[np.ndarray]]
            원본 카메라 프레임 리스트
        annotated_frames : List[Optional[np.ndarray]]
            스켈레톤이 그려진 프레임 리스트
        fused_angles : Dict
            퓨전된 관절 각도
        confidence : Dict
            관절별 신뢰도
        analysis_result : Dict
            생체역학 분석 결과
        num_cameras : int
            활성 카메라 수
        detection_rates : List[float]
            각 카메라 감지율
        is_recording : bool
            녹화 중 여부
        analysis_count : int
            누적 분석 횟수

        Returns
        -------
        np.ndarray
            렌더링된 전체 HUD 이미지
        """
        self._update_fps()

        if detection_rates is None:
            detection_rates = [0.0] * num_cameras

        # 전체 캔버스 생성 (검은색 배경)
        canvas = np.zeros((self._total_h, self._total_w, 3), dtype=np.uint8)

        # === 카메라 영상 배치 ===
        for i in range(2 if self._is_dual else 1):
            x_offset = i * self._cam_w
            # annotated_frame 우선, 없으면 원본, 없으면 빈 화면
            frame = None
            if i < len(annotated_frames) and annotated_frames[i] is not None:
                frame = annotated_frames[i]
            elif i < len(frames) and frames[i] is not None:
                frame = frames[i]

            if frame is not None:
                # 크기 맞추기
                if frame.shape[:2] != (self._cam_h, self._cam_w):
                    frame = cv2.resize(frame, (self._cam_w, self._cam_h))
                canvas[0:self._cam_h, x_offset:x_offset + self._cam_w] = frame
            else:
                # 빈 화면에 카메라 번호 표시
                cv2.putText(canvas, f"Camera {i + 1}: No Signal",
                            (x_offset + 150, self._cam_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA)

            # 카메라 라벨 오버레이
            label_color = (0, 255, 0) if i == 0 else (255, 100, 0)
            cv2.putText(canvas, f"Cam {i + 1}", (x_offset + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2, cv2.LINE_AA)

        # === 사이드 패널 ===
        panel_x = self._cam_w * (2 if self._is_dual else 1)
        panel = canvas[0:self._cam_h, panel_x:panel_x + self._panel_w]
        self._draw_side_panel(panel, fused_angles, confidence, analysis_result)

        # === 하단 상태 바 ===
        bar = canvas[self._cam_h:self._cam_h + self._bar_h, 0:self._total_w]
        self._draw_bottom_bar(bar, num_cameras, detection_rates,
                              is_recording, analysis_count)

        return canvas

    def show(self, canvas: np.ndarray) -> int:
        """
        렌더링된 캔버스를 화면에 표시하고 키 입력을 반환합니다.

        Parameters
        ----------
        canvas : np.ndarray
            render()의 반환값

        Returns
        -------
        int
            눌린 키 코드 (-1이면 없음)
        """
        cv2.imshow(WINDOW_NAME, canvas)
        return cv2.waitKey(1)

    def destroy(self):
        """OpenCV 창을 닫습니다."""
        cv2.destroyAllWindows()
