"""
main.py - OAK-D 실시간 소방관 포즈 분석 시스템 진입점
=====================================================

이 모듈은 전체 시스템의 진입점(Entry Point)으로, CLI 인자 파싱과
실시간 메인 루프 오케스트레이션을 담당합니다.

시스템 목적:
    소방관의 실시간 동작을 카메라로 촬영하고, MediaPipe로 관절 각도를 추출한 후,
    생체역학 시뮬레이션으로 근육/인대 부상 위험도를 예측합니다.
    이를 통해 현장 활동 중 과도한 부하나 위험한 자세를 조기에 감지할 수 있습니다.

실행 방법:
    python main.py                     # OAK-D 카메라 (기본)
    python main.py --webcam            # 웹캠 폴백 모드
    python main.py --webcam-id 1       # 특정 웹캠 장치 사용
    python main.py --depth             # OAK-D 스테레오 깊이맵 활성화
    python main.py --record            # 시작과 동시에 녹화
    python main.py --load 25           # 외부 하중 25kg (장비 무게)
    python main.py --task lift         # 작업 유형: lift/pull/carry/push
    python main.py --body-mass 80      # 작업자 체중 80kg

    복합 예시 (전형적인 소방 훈련 분석):
    python main.py --load 25 --task lift --body-mass 80

실시간 키보드 조작:
    q 또는 ESC  - 프로그램 종료
    r           - 녹화 시작/중지 토글 (CSV + MP4 저장)
    s           - 현재 화면 스냅샷 저장 (PNG)
    Space       - 일시정지/재개 토글

메인 루프 데이터 흐름:
    ┌───────────────────────────────────────────────────────────┐
    │                    메인 루프 (30fps)                       │
    │                                                           │
    │  1. OAK-D/웹캠 → BGR 프레임 획득                          │
    │  2. MediaPipe → 6개 관절 각도 추출 + 스켈레톤 오버레이      │
    │  3. AngleBuffer → 최근 2초(60프레임) 각도 데이터 축적       │
    │  4. (버퍼 가득 & N프레임마다) → 생체역학 분석 제출           │
    │  5. BiomechEngine → 백그라운드에서 비동기 시뮬레이션         │
    │  6. 최신 분석 결과 폴링 → 화면에 위험도 표시                │
    │  7. RealtimeDisplay → HUD 렌더링 + cv2.imshow             │
    │  8. 키보드 입력 처리                                       │
    │                                                           │
    └───────────────────────────────────────────────────────────┘

구성 모듈:
    - config.py          : 모든 설정 상수 (해상도, 임계값, 모델 파라미터)
    - oakd_camera.py     : OAK-D/웹캠 프레임 획득
    - pose_analyzer.py   : MediaPipe 포즈 감지 + 각도 계산
    - angle_buffer.py    : 롤링 윈도우 각도 버퍼
    - biomech_engine.py  : 비동기 생체역학 시뮬레이션 엔진
    - realtime_display.py: 실시간 HUD 오버레이
"""

import os
import sys
import time
import argparse
import csv

import cv2
import numpy as np

from config import (
    OAKD_FPS, ANALYSIS_INTERVAL, ROLLING_WINDOW_SIZE, WINDOW_NAME,
    DEFAULT_BODY_MASS_KG, DEFAULT_LOAD_KG, DEFAULT_TASK_TYPE,
    LOAD_TASK_PROFILES, JOINT_LABELS,
)
from oakd_camera import OAKDCamera
from pose_analyzer import PoseAnalyzer
from angle_buffer import AngleBuffer
from biomech_engine import BiomechEngine
from realtime_display import RealtimeDisplay


def parse_args():
    """
    CLI 인자를 파싱합니다.

    사용 가능한 인자:
        --webcam      : 웹캠 모드 (OAK-D 대신)
        --webcam-id   : 웹캠 장치 번호 (기본: 0)
        --depth       : OAK-D 스테레오 깊이 활성화
        --record      : 시작과 동시에 녹화
        --load        : 외부 하중 (kg). 장비, 호스, 구조물 무게.
        --task        : 작업 유형 (none/lift/pull/carry/push)
                        각 유형별로 근육/인대 부하 계수가 다릅니다.
        --body-mass   : 작업자 체중 (kg). 역학 계산의 기준.

    Returns
    -------
    argparse.Namespace
        파싱된 인자 객체
    """
    parser = argparse.ArgumentParser(
        description='OAK-D Real-Time Firefighter Pose Analysis'
    )
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam instead of OAK-D')
    parser.add_argument('--webcam-id', type=int, default=0,
                        help='Webcam device ID (default: 0)')
    parser.add_argument('--depth', action='store_true',
                        help='Enable OAK-D stereo depth')
    parser.add_argument('--record', action='store_true',
                        help='Start with recording enabled')
    parser.add_argument('--load', type=float, default=DEFAULT_LOAD_KG,
                        help=f'External load in kg (default: {DEFAULT_LOAD_KG})')
    parser.add_argument('--task', type=str, default=DEFAULT_TASK_TYPE,
                        choices=['none', 'lift', 'pull', 'carry', 'push'],
                        help=f'Task type (default: {DEFAULT_TASK_TYPE})')
    parser.add_argument('--body-mass', type=float, default=DEFAULT_BODY_MASS_KG,
                        help=f'Body mass in kg (default: {DEFAULT_BODY_MASS_KG})')
    return parser.parse_args()


def save_snapshot(frame, output_dir):
    """
    현재 디스플레이 프레임을 PNG 이미지로 저장합니다.

    Parameters
    ----------
    frame : np.ndarray
        저장할 프레임 (BGR). render()가 반환한 전체 HUD 화면.
    output_dir : str
        저장 디렉토리 경로. 없으면 자동 생성.

    저장 형식:
        output/snapshot_YYYYMMDD_HHMMSS.png
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(output_dir, f'snapshot_{ts}.png')
    cv2.imwrite(path, frame)
    print(f"[OK] Snapshot saved: {path}")


def save_recording(angle_buffer, output_dir, fps):
    """
    녹화된 관절 각도 데이터를 CSV 파일로 저장합니다.

    CSV 형식:
        Time(ms), Knee, Hip, Ankle, Shoulder, Elbow, Trunk
        0, 145.2, 160.1, 88.3, 45.0, 130.5, 170.2
        33, 144.8, 159.5, 87.9, 44.8, 131.0, 169.8
        ...

    Parameters
    ----------
    angle_buffer : AngleBuffer
        녹화 데이터를 포함하는 버퍼 객체.
        get_full_history()로 전체 세션 데이터를 추출합니다.
    output_dir : str
        CSV 저장 디렉토리 경로.
    fps : float
        녹화 시 FPS (파일명에는 사용하지 않지만 향후 확장용).

    저장 형식:
        output/recording_YYYYMMDD_HHMMSS.csv

    Note
    ----
    CSV는 UTF-8 BOM(utf-8-sig)으로 인코딩됩니다.
    이렇게 하면 Excel에서 열었을 때 한글이 깨지지 않습니다.
    """
    history = angle_buffer.get_full_history()
    if history is None or len(history['time']) == 0:
        print("[WARN] No data to save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(output_dir, f'recording_{ts}.csv')

    time_arr = history['time']       # numpy 배열: 타임스탬프 (초)
    joints = history['joints']       # dict: {관절이름: numpy 배열}
    joint_names = list(joints.keys())

    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # 헤더: JOINT_LABELS를 사용하여 읽기 쉬운 이름으로 변환
        writer.writerow(['Time(ms)'] + [JOINT_LABELS.get(j, j) for j in joint_names])
        for i in range(len(time_arr)):
            # 시간을 밀리초로 변환하여 저장
            row = [f"{time_arr[i] * 1000:.0f}"]
            for jn in joint_names:
                row.append(f"{joints[jn][i]:.1f}")
            writer.writerow(row)

    print(f"[OK] Recording saved: {path} ({len(time_arr)} frames)")


def main():
    """
    메인 함수: 시스템 초기화 + 실시간 루프 실행.

    실행 흐름:
        1. CLI 인자 파싱
        2. 출력 디렉토리 설정
        3. 시스템 정보 출력
        4. 구성 요소 초기화 (카메라, 분석기, 버퍼, 엔진, 디스플레이)
        5. 카메라 시작
        6. 메인 루프 진입
        7. 종료 시 정리(cleanup)
    """
    args = parse_args()

    # ─── 출력 디렉토리 설정 ───
    # 스냅샷, 녹화 파일이 저장되는 위치
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # ─── 시스템 정보 출력 ───
    print("=" * 60)
    print("  OAK-D Real-Time Firefighter Pose Analysis")
    print("  MediaPipe Pose + OpenSim Biomechanics Model")
    print("=" * 60)

    mode = "Webcam" if args.webcam else "OAK-D"
    print(f"  Camera: {mode}")
    if args.load > 0:
        task_info = LOAD_TASK_PROFILES.get(args.task, {})
        print(f"  Load: {args.load} kg ({args.load * 9.81:.1f} N)")
        print(f"  Task: {task_info.get('name', args.task)}")
    else:
        print(f"  Load: None (bodyweight only)")
    print(f"  Body mass: {args.body_mass} kg")
    print(f"  Analysis interval: every {ANALYSIS_INTERVAL} frames")
    print("=" * 60)

    # ─── 구성 요소 초기화 ───
    # 각 모듈이 독립적으로 동작하며, main.py가 이들을 조율합니다.

    # (1) 카메라: OAK-D 또는 웹캠에서 BGR 프레임을 제공
    camera = OAKDCamera(
        use_webcam=args.webcam,
        enable_depth=args.depth,
        webcam_id=args.webcam_id,
    )

    # (2) 포즈 분석기: MediaPipe로 33개 랜드마크 감지 → 6개 관절 각도 계산
    analyzer = PoseAnalyzer()

    # (3) 각도 버퍼: 최근 60프레임(2초)의 관절 각도를 롤링 윈도우로 유지
    angle_buffer = AngleBuffer(max_size=ROLLING_WINDOW_SIZE)

    # (4) 생체역학 엔진: Hill-type 근육 모델 + 인대 모델로 부상 위험도 계산
    #     ThreadPoolExecutor로 백그라운드 비동기 실행 (메인 루프 30fps 유지)
    engine = BiomechEngine(
        load_kg=args.load,
        body_mass_kg=args.body_mass,
        task_type=args.task,
    )

    # (5) 디스플레이: 카메라 피드 + 사이드 패널 + 하단 바를 합성
    display = RealtimeDisplay()

    # ─── 카메라 시작 ───
    try:
        camera.start()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    # ─── 녹화 모드 (--record 옵션) ───
    if args.record:
        angle_buffer.start_recording()
        print("[OK] Recording started.")

    # ─── 비디오 녹화용 VideoWriter ───
    # 'r' 키로 녹화 시작 시 생성됩니다.
    video_writer = None

    # =========================================================================
    # 메인 루프 상태 변수
    # =========================================================================
    frame_count = 0           # 총 처리 프레임 수
    detected_count = 0        # 포즈 감지 성공 프레임 수
    paused = False            # 일시정지 상태
    t_start = time.time()     # 세션 시작 시각 (경과 시간 계산용)
    fps_counter = 0           # FPS 계산용 프레임 카운터
    fps_time = time.time()    # FPS 계산용 시간 기준점
    current_fps = 0.0         # 현재 FPS (1초마다 갱신)
    angles = None             # 현재 프레임의 관절 각도 (또는 None)
    annotated = None          # 스켈레톤 오버레이된 프레임 (또는 None)
    analysis_result = None    # 최신 생체역학 분석 결과 (또는 None)
    detection_rate = 0.0      # 포즈 감지 성공률 (%)

    print("\n[OK] Starting real-time loop... Press 'q' to quit.\n")

    # =========================================================================
    # 메인 루프
    # =========================================================================
    # 30fps (OAK-D) 또는 웹캠 FPS에 맞추어 무한 루프를 실행합니다.
    # cv2.waitKey(1)로 최소 1ms 대기하면서 키보드 입력을 처리합니다.
    try:
        while True:
            # ─── 프레임 획득 및 처리 (비일시정지 상태) ───
            if not paused:
                # (1) 카메라에서 프레임 획득
                bgr, depth = camera.get_frame()
                if bgr is None:
                    # OAK-D tryGet()이 None 반환 (아직 프레임 없음)
                    time.sleep(0.001)
                    continue

                frame_count += 1
                # 세션 시작부터의 경과 시간 (타임스탬프로 사용)
                current_time = time.time() - t_start

                # (2) MediaPipe 포즈 감지 + 관절 각도 계산
                # angles: {'Knee_Angle': 145.2, ...} 또는 None (감지 실패)
                # annotated: 스켈레톤 + 각도 라벨이 그려진 BGR 프레임
                angles, annotated = analyzer.process_frame(bgr)

                if angles is not None:
                    detected_count += 1
                    # (3) 각도 데이터를 롤링 버퍼에 추가
                    angle_buffer.push(current_time, angles)

                    # (4) 생체역학 분석 제출 조건:
                    #     - 버퍼가 가득 참 (60프레임 = 2초 데이터)
                    #     - ANALYSIS_INTERVAL 프레임마다 (기본 15프레임 = 0.5초)
                    if angle_buffer.is_full() and frame_count % ANALYSIS_INTERVAL == 0:
                        window = angle_buffer.get_window()
                        if window is not None:
                            # submit_analysis(): ThreadPool에 시뮬레이션 제출
                            # 이전 분석이 아직 실행 중이면 skip됨
                            engine.submit_analysis(window)

                # (5) 최신 분석 결과 폴링 (완료된 경우에만 갱신)
                analysis_result = engine.get_latest_result()

                # (6) 비디오 녹화 중이면 프레임 저장
                if video_writer is not None:
                    video_writer.write(annotated)

            else:
                # ─── 일시정지 상태 ───
                # 프레임을 새로 획득하지 않고 마지막 프레임을 계속 표시합니다.
                time.sleep(0.033)  # ~30fps 속도로 대기 (CPU 과부하 방지)
                if annotated is None:
                    continue  # 아직 한 번도 프레임을 받지 못한 경우

            # ─── FPS 계산 ───
            # 1초마다 처리된 프레임 수로 FPS를 갱신합니다.
            fps_counter += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_time = time.time()

            # ─── 감지율 계산 ───
            # (감지 성공 프레임 수 / 총 프레임 수) × 100
            detection_rate = (detected_count / max(frame_count, 1)) * 100

            # ─── 디스플레이 렌더링 ───
            # 카메라 피드 + 사이드 패널 + 하단 바를 합성하여 표시
            display_frame = display.render(
                camera_frame=annotated,
                angles=angles if not paused else angles,
                analysis_result=analysis_result,
                fps=current_fps,
                detection_rate=detection_rate,
                analysis_count=engine.analysis_count,
                is_recording=angle_buffer.is_recording,
                is_paused=paused,
                buffer_fill=angle_buffer.size / angle_buffer.max_size,
                is_analyzing=engine.is_analyzing,
            )
            display.show(display_frame)

            # ─── 키보드 입력 처리 ───
            # cv2.waitKey(1): 1ms 대기하며 키 입력 확인
            # & 0xFF: Windows에서 상위 비트 제거 (8비트 ASCII만 사용)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' 또는 ESC → 종료
                break

            elif key == ord('r'):  # 'r' → 녹화 토글
                if angle_buffer.is_recording:
                    # 녹화 중지 → CSV 저장
                    angle_buffer.stop_recording()
                    save_recording(angle_buffer, output_dir, OAKD_FPS)
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    print("[OK] Recording stopped.")
                else:
                    # 녹화 시작 → CSV 축적 + MP4 비디오 녹화
                    angle_buffer.start_recording()
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    vid_path = os.path.join(output_dir, f'recording_{ts}.mp4')
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱
                    video_writer = cv2.VideoWriter(vid_path, fourcc, OAKD_FPS, (w, h))
                    print(f"[OK] Recording started: {vid_path}")

            elif key == ord('s'):  # 's' → 스냅샷 저장
                save_snapshot(display_frame, output_dir)

            elif key == ord(' '):  # Space → 일시정지/재개
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        # =========================================================================
        # 정리(Cleanup)
        # =========================================================================
        # 녹화 중이면 저장 후 종료
        if angle_buffer.is_recording:
            angle_buffer.stop_recording()
            save_recording(angle_buffer, output_dir, OAKD_FPS)
        if video_writer is not None:
            video_writer.release()

        # 각 모듈의 자원 해제
        analyzer.close()        # MediaPipe PoseLandmarker 해제
        engine.shutdown()       # ThreadPoolExecutor 종료
        camera.stop()           # OAK-D/웹캠 연결 종료
        display.destroy()       # OpenCV 창 닫기

        # ─── 세션 요약 출력 ───
        elapsed_total = time.time() - t_start
        print(f"\n{'=' * 60}")
        print(f"  Session Summary")
        print(f"  Duration: {elapsed_total:.1f}s")
        print(f"  Frames: {frame_count}")
        print(f"  Detections: {detected_count} ({detection_rate:.1f}%)")
        print(f"  Analyses: {engine.analysis_count}")
        print(f"  Output: {output_dir}")
        print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
