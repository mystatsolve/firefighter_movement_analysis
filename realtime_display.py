"""
realtime_display.py - 실시간 HUD 오버레이 디스플레이
===================================================

이 모듈은 OpenCV를 사용하여 실시간 분석 결과를 시각적으로 표시합니다.
카메라 피드 위에 사이드 패널, 하단 상태 바를 합성하여 전문적인 모니터링 화면을 구성합니다.

디스플레이 레이아웃:
    ┌─────────────────────────────────┬────────────────────┐
    │                                 │   POSE ANALYSIS    │
    │                                 │   Buffer: ██░ 75%  │
    │     카메라 피드                  │                    │
    │     (스켈레톤 오버레이 포함)       │   Joint Angles     │
    │                                 │   Knee:  145.2°    │
    │     640 x 480                   │   Hip:   160.1°    │
    │                                 │   Ankle:  88.3°    │
    │                                 │                    │
    │                                 │   Risk Analysis    │
    │                                 │   Overall: Normal  │
    │                                 │   Knee ██░ Low     │
    │                                 │   Back ███ High    │
    │                                 │                    │
    │                                 │   Muscle Stress    │
    │                                 │   Quad  ██░ 120kPa│
    │                                 │   Glute ███ 250kPa│
    ├─────────────────────────────────┴────────────────────┤
    │ FPS:30.0 | Detect:95% | Analyses:12 | ● REC | [keys]│
    └──────────────────────────────────────────────────────┘

구현 특이사항:
    - cv2.putText()는 CJK(한/중/일) 폰트를 지원하지 않으므로
      모든 텍스트를 영문으로 표시합니다.
    - 한글 텍스트가 필요하면 PIL(Pillow)의 ImageDraw를 사용해야 하지만,
      성능 오버헤드가 크므로 실시간 시스템에서는 영문을 사용합니다.
    - 색상은 BGR 형식 (OpenCV 기본) 입니다.
"""

import cv2
import numpy as np
from typing import Optional, Dict

from config import (
    SIDE_PANEL_WIDTH, BOTTOM_BAR_HEIGHT, WINDOW_NAME,
    RISK_COLORS, JOINT_LABELS, RISK_LEVELS,
    MUSCLE_STRESS_THRESHOLDS,
)


class RealtimeDisplay:
    """
    OpenCV 기반 실시간 HUD(Head-Up Display) 오버레이 클래스.

    이 클래스는 카메라 프레임, 관절 각도, 위험도 분석 결과를 합성하여
    하나의 큰 디스플레이 프레임을 생성합니다. 생성된 프레임은 cv2.imshow()로 표시됩니다.

    구조:
        - 왼쪽 상단: 카메라 피드 (스켈레톤 + 각도 라벨 오버레이 포함)
        - 오른쪽: 사이드 패널 (버퍼 상태, 관절 각도, 위험도 분석)
        - 하단: 상태 바 (FPS, 감지율, 녹화 상태, 키보드 안내)

    렌더링 파이프라인:
        1. render() 호출 → 빈 캔버스(검정) 생성
        2. 카메라 프레임을 왼쪽 상단에 복사
        3. _draw_side_panel()로 사이드 패널 렌더링
        4. _draw_bottom_bar()로 하단 바 렌더링
        5. 완성된 캔버스 반환

    Attributes
    ----------
    _window_created : bool
        OpenCV 창이 이미 생성되었는지 추적합니다.
        중복 생성을 방지하기 위해 사용됩니다.
    """

    def __init__(self):
        """디스플레이 초기화. 창은 show() 첫 호출 시 생성됩니다."""
        self._window_created = False

    def render(self, camera_frame: np.ndarray,
               angles: Optional[Dict[str, float]],
               analysis_result: Optional[dict],
               fps: float,
               detection_rate: float,
               analysis_count: int,
               is_recording: bool,
               is_paused: bool,
               buffer_fill: float,
               is_analyzing: bool) -> np.ndarray:
        """
        전체 디스플레이 프레임을 합성합니다.

        이 메서드는 메인 루프에서 매 프레임마다 호출됩니다.
        30fps에서 ~33ms마다 호출되므로 빠르게 실행되어야 합니다.

        Parameters
        ----------
        camera_frame : np.ndarray
            PoseAnalyzer가 스켈레톤을 그린 BGR 프레임.
            shape: (480, 640, 3), dtype: uint8
        angles : Dict[str, float] 또는 None
            현재 프레임의 관절 각도 딕셔너리.
            포즈가 감지되지 않으면 None.
            예: {'Knee_Angle': 145.2, 'Hip_Angle': 160.1, ...}
        analysis_result : dict 또는 None
            BiomechEngine의 최신 분석 결과.
            분석이 아직 완료되지 않았으면 None.
            구조: {
                'overall_risk': str,     # 전체 위험 등급
                'body_risks': dict,      # 부위별 위험도
                'muscle_risks': list,    # 근육별 상세 분석
                'ligament_risks': list,  # 인대별 상세 분석
            }
        fps : float
            현재 처리 프레임 레이트 (초당 프레임 수).
            25fps 이상이면 녹색, 미만이면 주황색으로 표시.
        detection_rate : float
            포즈 감지 성공률 (0-100%).
            80% 이상이면 녹색, 미만이면 주황색으로 표시.
        analysis_count : int
            지금까지 완료된 생체역학 분석 횟수.
        is_recording : bool
            녹화 모드 활성화 여부. True이면 빨간 원(●) + "REC" 표시.
        is_paused : bool
            일시정지 상태 여부. True이면 "PAUSED" 표시.
        buffer_fill : float
            각도 버퍼 채움 비율 (0.0~1.0).
            0.0=비어있음, 1.0=가득참(분석 제출 가능).
        is_analyzing : bool
            현재 백그라운드에서 분석이 실행 중인지 여부.
            True이면 "ANALYZING..." 텍스트 표시.

        Returns
        -------
        np.ndarray
            합성된 전체 디스플레이 프레임 (BGR).
            shape: (cam_h + BOTTOM_BAR_HEIGHT, cam_w + SIDE_PANEL_WIDTH, 3)
        """
        cam_h, cam_w = camera_frame.shape[:2]
        total_w = cam_w + SIDE_PANEL_WIDTH     # 전체 너비 = 카메라 + 사이드 패널
        total_h = cam_h + BOTTOM_BAR_HEIGHT    # 전체 높이 = 카메라 + 하단 바

        # 빈 캔버스 생성 (검정 배경)
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # 카메라 프레임을 왼쪽 상단에 복사
        canvas[0:cam_h, 0:cam_w] = camera_frame

        # 오른쪽 사이드 패널 렌더링
        self._draw_side_panel(canvas, cam_w, cam_h, angles, analysis_result,
                              buffer_fill, is_analyzing)

        # 하단 상태 바 렌더링
        self._draw_bottom_bar(canvas, total_w, cam_h, total_h,
                              fps, detection_rate, analysis_count,
                              is_recording, is_paused)

        return canvas

    def _draw_side_panel(self, canvas, x_start, panel_h, angles, result,
                         buffer_fill, is_analyzing):
        """
        오른쪽 사이드 패널을 렌더링합니다.

        사이드 패널 구성:
            1. 헤더: "POSE ANALYSIS" 제목
            2. 버퍼 상태 바: 채움 비율을 시각적 바로 표시
            3. 관절 각도 섹션: 6개 관절의 현재 각도값
            4. 위험도 분석 섹션:
               - 전체 위험 등급 (Overall)
               - 부위별 위험도 바 (Body regions)
               - 근육 스트레스 바 (Muscle stress in kPa)

        Parameters
        ----------
        canvas : np.ndarray
            그릴 대상 캔버스 (in-place 수정)
        x_start : int
            사이드 패널 시작 x 좌표 (= 카메라 프레임 너비)
        panel_h : int
            사이드 패널 높이 (= 카메라 프레임 높이)
        angles : dict 또는 None
            현재 관절 각도
        result : dict 또는 None
            생체역학 분석 결과
        buffer_fill : float
            버퍼 채움 비율 (0.0~1.0)
        is_analyzing : bool
            분석 진행 중 여부
        """
        x = x_start
        w = SIDE_PANEL_WIDTH

        # ─── 패널 배경 (짙은 회색) ───
        canvas[0:panel_h, x:x+w] = (30, 30, 30)

        # ─── 헤더 ───
        cv2.putText(canvas, "POSE ANALYSIS", (x + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (x + 5, 32), (x + w - 5, 32), (80, 80, 80), 1)

        # ─── 버퍼 채움 상태 바 ───
        # 버퍼가 가득 차야(100%) 분석을 제출할 수 있으므로 진행도를 시각화합니다.
        y = 45
        cv2.putText(canvas, f"Buffer: {buffer_fill*100:.0f}%", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        bar_x = x + 120
        bar_w = w - 135
        # 배경 바 (짙은 회색)
        cv2.rectangle(canvas, (bar_x, y - 10), (bar_x + bar_w, y), (60, 60, 60), -1)
        # 채움 바 (녹색=가득참, 주황색=진행중)
        fill_w = int(bar_w * buffer_fill)
        if fill_w > 0:
            color = (0, 200, 0) if buffer_fill >= 1.0 else (0, 180, 255)
            cv2.rectangle(canvas, (bar_x, y - 10), (bar_x + fill_w, y), color, -1)

        # 분석 진행 중 표시
        if is_analyzing:
            cv2.putText(canvas, "ANALYZING...", (x + 10, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

        # ─── 관절 각도 섹션 ───
        # 현재 프레임에서 감지된 6개 관절의 각도를 표시합니다.
        y = 80
        cv2.putText(canvas, "Joint Angles (deg)", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y += 5
        cv2.line(canvas, (x + 5, y), (x + w - 5, y), (60, 60, 60), 1)
        y += 18

        if angles:
            for angle_name, angle_val in angles.items():
                # JOINT_LABELS로 사람이 읽기 쉬운 이름으로 변환
                label = JOINT_LABELS.get(angle_name, angle_name)
                cv2.putText(canvas, f"{label:>10s}: {angle_val:6.1f}",
                            (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (220, 220, 220), 1, cv2.LINE_AA)
                y += 20
        else:
            # 포즈가 감지되지 않은 경우
            cv2.putText(canvas, "  No pose detected", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)
            y += 20

        # ─── 위험도 분석 섹션 ───
        y += 10
        cv2.putText(canvas, "Risk Analysis", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y += 5
        cv2.line(canvas, (x + 5, y), (x + w - 5, y), (60, 60, 60), 1)
        y += 18

        if result is None:
            # 아직 분석 결과가 없음 (버퍼가 차기를 기다리는 중)
            cv2.putText(canvas, "  Waiting for data...", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)
            return

        # ─── 전체 위험 등급 (Overall Risk) ───
        # 모든 근육/인대의 위험도 중 최대값
        overall = result.get('overall_risk', 'Normal')
        color = RISK_COLORS.get(overall, (180, 180, 180))
        # High/Critical이면 굵은 글씨(thickness=2)로 강조
        cv2.putText(canvas, f"Overall: {overall}", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2 if overall in ('High', 'Critical') else 1,
                    cv2.LINE_AA)
        y += 25

        # ─── 부위별 위험도 바 (Body Region Risks) ───
        # 각 신체 부위(무릎, 허리, 어깨 등)의 위험 등급을 색상 바로 표시
        body_risks = result.get('body_risks', {})
        for region, info in body_risks.items():
            risk = info['risk_level']
            score = info['risk_score']  # 0~4 (Normal=0, Low=1, Moderate=2, High=3, Critical=4)
            rc = RISK_COLORS.get(risk, (180, 180, 180))

            # 부위 이름
            cv2.putText(canvas, f"{region:>14s}", (x + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

            # 위험도 바: score/4 비율로 채움
            bar_x = x + 140
            bar_w = 80
            bar_h = 12
            cv2.rectangle(canvas, (bar_x, y - bar_h + 2), (bar_x + bar_w, y + 2),
                          (50, 50, 50), -1)  # 배경
            fill = int(bar_w * score / 4)
            if fill > 0:
                cv2.rectangle(canvas, (bar_x, y - bar_h + 2), (bar_x + fill, y + 2),
                              rc, -1)  # 채움 (위험 등급 색상)

            # 위험 등급 텍스트
            cv2.putText(canvas, risk, (bar_x + bar_w + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, rc, 1, cv2.LINE_AA)
            y += 18

        # ─── 근육 스트레스 상세 바 (Muscle Stress kPa) ───
        # 각 근육의 피크 스트레스를 바 형태로 표시
        y += 8
        cv2.putText(canvas, "Muscle Stress (kPa)", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 5
        cv2.line(canvas, (x + 5, y), (x + w - 5, y), (60, 60, 60), 1)
        y += 15

        muscle_risks = result.get('muscle_risks', [])
        # 스트레스 바의 최대 범위 (Critical 임계값)
        thresh_max = max(MUSCLE_STRESS_THRESHOLDS.values()) / 1000  # Pa → kPa 변환
        for mr in muscle_risks:
            name = mr.get('display_name', mr['muscle_name'])
            peak = mr['peak_stress_kPa']       # 피크 스트레스 (kPa)
            risk = mr['risk_level']             # 위험 등급
            rc = RISK_COLORS.get(risk, (180, 180, 180))

            # 이름이 길면 8자로 잘라서 표시 (사이드 패널 공간 한계)
            display = name[:8] if len(name) > 8 else name
            cv2.putText(canvas, f"{display}", (x + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

            # 스트레스 바: peak/thresh_max 비율로 채움
            bar_x2 = x + 80
            bar_w2 = 120
            cv2.rectangle(canvas, (bar_x2, y - 10), (bar_x2 + bar_w2, y + 1),
                          (40, 40, 40), -1)  # 배경
            fill_ratio = min(peak / thresh_max, 1.0)  # 1.0 = Critical 임계값 도달
            fill_w = int(bar_w2 * fill_ratio)
            if fill_w > 0:
                cv2.rectangle(canvas, (bar_x2, y - 10), (bar_x2 + fill_w, y + 1),
                              rc, -1)  # 채움 (위험 등급 색상)

            # 수치 표시 (kPa 단위)
            cv2.putText(canvas, f"{peak:.0f}", (bar_x2 + bar_w2 + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, rc, 1, cv2.LINE_AA)
            y += 16

            # 패널 높이 초과 방지
            if y > panel_h - 20:
                break

    def _draw_bottom_bar(self, canvas, total_w, y_start, total_h,
                         fps, detection_rate, analysis_count,
                         is_recording, is_paused):
        """
        하단 상태 바를 렌더링합니다.

        표시 항목 (왼쪽→오른쪽):
            1. FPS: 현재 프레임 처리 속도
            2. Detect: 포즈 감지 성공률
            3. Analyses: 완료된 분석 횟수
            4. REC: 녹화 중 표시 (빨간 원)
            5. PAUSED: 일시정지 표시
            6. Controls: 키보드 단축키 안내 (우측 정렬)

        Parameters
        ----------
        canvas : np.ndarray
            그릴 대상 캔버스
        total_w : int
            전체 캔버스 너비
        y_start : int
            하단 바 시작 y 좌표
        total_h : int
            전체 캔버스 높이
        fps : float
            현재 FPS
        detection_rate : float
            감지 성공률 (0-100%)
        analysis_count : int
            완료된 분석 횟수
        is_recording : bool
            녹화 중 여부
        is_paused : bool
            일시정지 여부
        """
        # 배경 (매우 짙은 회색)
        canvas[y_start:total_h, 0:total_w] = (25, 25, 25)

        y = y_start + 25  # 텍스트 y 위치 (바 내부 중앙)
        x = 10            # 시작 x 위치

        # ─── FPS 표시 ───
        # 25fps 이상이면 녹색(실시간 유지), 미만이면 주황색(성능 저하)
        cv2.putText(canvas, f"FPS: {fps:.1f}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if fps >= 25 else (0, 200, 255), 1, cv2.LINE_AA)
        x += 120

        # ─── 감지율 표시 ───
        # 80% 이상이면 녹색(양호), 미만이면 주황색(카메라 각도 조정 필요)
        cv2.putText(canvas, f"Detect: {detection_rate:.0f}%", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if detection_rate >= 80 else (0, 200, 255), 1, cv2.LINE_AA)
        x += 140

        # ─── 분석 횟수 표시 ───
        cv2.putText(canvas, f"Analyses: {analysis_count}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        x += 150

        # ─── 녹화 표시 (빨간 원 + "REC") ───
        if is_recording:
            cv2.circle(canvas, (x + 5, y - 5), 6, (0, 0, 255), -1)  # 빨간 원
            cv2.putText(canvas, "REC", (x + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            x += 80

        # ─── 일시정지 표시 ───
        if is_paused:
            cv2.putText(canvas, "PAUSED", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
            x += 100

        # ─── 키보드 단축키 안내 (우측 정렬) ───
        hint = "Q:Quit  R:Record  S:Snap  Space:Pause"
        hint_w = len(hint) * 8  # 대략적인 텍스트 너비 추정
        cv2.putText(canvas, hint, (total_w - hint_w - 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

    def show(self, frame: np.ndarray):
        """
        합성된 프레임을 OpenCV 창에 표시합니다.

        첫 호출 시 cv2.namedWindow()로 창을 생성합니다.
        WINDOW_NORMAL 플래그로 사용자가 창 크기를 조절할 수 있습니다.

        Parameters
        ----------
        frame : np.ndarray
            render()가 반환한 합성 프레임 (BGR)
        """
        if not self._window_created:
            # 첫 프레임에서 창 생성 (크기 조절 가능)
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self._window_created = True
        cv2.imshow(WINDOW_NAME, frame)

    def destroy(self):
        """
        OpenCV 창을 닫고 자원을 해제합니다.

        프로그램 종료 시 호출됩니다.
        호출하지 않으면 Windows에서 창이 응답 없음 상태가 될 수 있습니다.
        """
        if self._window_created:
            cv2.destroyWindow(WINDOW_NAME)
            self._window_created = False
