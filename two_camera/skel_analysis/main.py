"""
main.py - 듀얼 카메라 소방관 포즈 분석 시스템 진입점
=====================================================

OAK-D 카메라 2대를 사용하여 소방관의 실시간 자세를 분석하고
부상 위험도를 예측하는 메인 루프입니다.

시스템 파이프라인:
    카메라1,2 프레임 획득
    → MediaPipe 양측 포즈 감지 (카메라별 독립)
    → 듀얼 카메라 신뢰도 가중 퓨전
    → 이동 평균 스무딩
    → 생체역학 시뮬레이션 (백그라운드)
    → 위험도 분류 + HUD 표시

실행 방법:
    python main.py                          # 기본 실행 (듀얼 카메라)
    python main.py --single                 # 단일 카메라 모드
    python main.py --load 20 --task lift    # 20kg 들기 작업
    python main.py --body-mass 80           # 체중 80kg
    python main.py --record                 # 시작 시 바로 녹화

키 조작:
    q / ESC : 종료
    r       : 녹화 시작/중지 (CSV + 차트 자동 저장)
    s       : 스냅샷 (현재 분석 보고서 저장)
    SPACE   : 일시정지/재개
"""

import argparse
import sys
import time

from config import (
    DEFAULT_BODY_MASS_KG, DEFAULT_LOAD_KG, DEFAULT_TASK_TYPE,
    ANALYSIS_INTERVAL,
)
from dual_camera import DualCameraManager
from pose_analyzer import PoseAnalyzer
from angle_fusion import AngleFusionEngine
from biomech_engine import BiomechEngine
from injury_predictor import InjuryPredictor
from realtime_display import RealtimeDisplay
from data_export import DataExporter, TimeSeriesLogger


def parse_args():
    """커맨드라인 인자 파싱."""
    parser = argparse.ArgumentParser(
        description='듀얼 카메라 소방관 포즈 분석 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py                          기본 실행 (듀얼 카메라)
  python main.py --single                 단일 카메라 모드
  python main.py --load 20 --task lift    20kg 들기 작업
  python main.py --record                 시작과 동시에 녹화

키 조작:
  q/ESC  종료  |  r 녹화  |  s 스냅샷  |  SPACE 일시정지
        """,
    )
    parser.add_argument('--load', type=float, default=DEFAULT_LOAD_KG,
                        help=f'외부 하중 (kg, 기본: {DEFAULT_LOAD_KG})')
    parser.add_argument('--task', type=str, default=DEFAULT_TASK_TYPE,
                        choices=['none', 'lift', 'pull', 'carry', 'push'],
                        help=f'작업 유형 (기본: {DEFAULT_TASK_TYPE})')
    parser.add_argument('--body-mass', type=float, default=DEFAULT_BODY_MASS_KG,
                        help=f'체중 (kg, 기본: {DEFAULT_BODY_MASS_KG})')
    parser.add_argument('--single', action='store_true',
                        help='단일 카메라 모드 강제')
    parser.add_argument('--record', action='store_true',
                        help='시작과 동시에 녹화 시작')
    return parser.parse_args()


def main():
    """메인 루프."""
    args = parse_args()

    print("=" * 60)
    print("  듀얼 카메라 소방관 포즈 분석 시스템")
    print("  Dual-Camera Firefighter Pose Analysis System")
    print("=" * 60)
    print(f"  체중: {args.body_mass}kg | 하중: {args.load}kg | 작업: {args.task}")
    print(f"  모드: {'단일' if args.single else '듀얼'} 카메라")
    print("=" * 60)

    # ================================================================
    # 모듈 초기화
    # ================================================================
    camera = None
    analyzers = []
    fusion = None
    engine = None
    predictor = None
    display = None
    exporter = None
    logger = None

    try:
        # 1. 카메라 연결
        print("\n[1/6] 카메라 연결 중...")
        camera = DualCameraManager(force_single=args.single)
        num_cameras = camera.start()

        # 2. 포즈 분석기 (카메라별 독립 인스턴스)
        print("[2/6] MediaPipe 포즈 분석기 초기화...")
        for i in range(num_cameras):
            analyzer = PoseAnalyzer()
            analyzers.append(analyzer)
            print(f"  포즈 분석기 {i + 1} 준비 완료")

        # 3. 퓨전 엔진
        print("[3/6] 각도 퓨전 엔진 초기화...")
        fusion = AngleFusionEngine()

        # 4. 생체역학 엔진
        print("[4/6] 생체역학 엔진 초기화...")
        engine = BiomechEngine(
            load_kg=args.load,
            body_mass_kg=args.body_mass,
            task_type=args.task,
        )

        # 5. 부상 예측기
        print("[5/6] 부상 예측기 초기화...")
        predictor = InjuryPredictor()

        # 6. 디스플레이 + 내보내기 + 시계열 로거
        print("[6/7] 디스플레이 초기화...")
        display = RealtimeDisplay(is_dual=camera.is_dual)
        exporter = DataExporter()

        # 7. 시계열 로거 (모든 데이터를 프레임별/분석별로 CSV 저장)
        print("[7/7] 시계열 로거 초기화...")
        logger = TimeSeriesLogger()

        # 녹화 자동 시작
        if args.record:
            fusion.start_recording()

        print("\n" + "=" * 60)
        print("  시스템 준비 완료! 실시간 분석을 시작합니다.")
        print("  q/ESC=종료 | r=녹화 | s=스냅샷 | SPACE=일시정지")
        print("=" * 60 + "\n")

        # ================================================================
        # 메인 루프
        # ================================================================
        paused = False
        frame_counter = 0
        detection_counts = [0] * num_cameras
        total_frames = [0] * num_cameras

        # 최신 유효 프레임 캐시 (프레임 드롭 대응)
        last_frames = [None] * num_cameras
        last_annotated = [None] * num_cameras

        while True:
            # 일시정지 상태
            if paused:
                key = display.show(
                    display.render(
                        last_frames, last_annotated,
                        *fusion.get_latest(),
                        engine.get_latest_result(),
                        num_cameras=num_cameras,
                        detection_rates=[dc / max(tf, 1) for dc, tf in zip(detection_counts, total_frames)],
                        is_recording=fusion.is_recording,
                        analysis_count=engine.analysis_count,
                    )
                )
                if key == ord(' '):
                    paused = False
                    print("[시스템] 재개")
                elif key == 27 or key == ord('q'):
                    break
                continue

            # 프레임 획득
            frames = camera.get_frames()

            # 카메라별 포즈 분석
            angles_list = [None] * num_cameras
            vis_list = [None] * num_cameras

            for i in range(num_cameras):
                frame = frames[i] if i < len(frames) else None
                if frame is None:
                    continue

                last_frames[i] = frame
                total_frames[i] += 1

                result = analyzers[i].process_frame(frame)
                if result is not None:
                    angles_list[i], vis_list[i], last_annotated[i] = result
                    detection_counts[i] += 1
                else:
                    last_annotated[i] = frame.copy()

            # 듀얼 카메라 퓨전
            fused = fusion.fuse(
                angles_list[0], vis_list[0],
                angles_list[1] if num_cameras > 1 else None,
                vis_list[1] if num_cameras > 1 else None,
            )

            # 시계열 로거에 프레임 데이터 기록
            fused_angles, confidence = fusion.get_latest()
            logger.log_frame(
                fused_angles, confidence,
                angles_list[0], vis_list[0],
                angles_list[1] if num_cameras > 1 else None,
                vis_list[1] if num_cameras > 1 else None,
            )

            # 생체역학 분석 제출 (ANALYSIS_INTERVAL 프레임마다)
            frame_counter += 1
            if frame_counter % ANALYSIS_INTERVAL == 0:
                window = fusion.get_window_data()
                if window is not None:
                    engine.submit_analysis(window)

            # 분석 결과 폴링
            analysis_result = engine.get_latest_result()

            # 새 분석 결과가 도착하면 시계열 로거에 기록
            if analysis_result and engine.analysis_count > logger.analysis_count:
                logger.log_analysis(analysis_result)

            # 경고 출력
            if analysis_result:
                warnings = predictor.get_warnings(analysis_result)
                for w in warnings:
                    pass  # HUD에서 표시, 콘솔 스팸 방지

            # 디스플레이 렌더링
            canvas = display.render(
                last_frames, last_annotated,
                fused_angles, confidence,
                analysis_result,
                num_cameras=num_cameras,
                detection_rates=[dc / max(tf, 1) for dc, tf in zip(detection_counts, total_frames)],
                is_recording=fusion.is_recording,
                analysis_count=engine.analysis_count,
            )

            key = display.show(canvas)

            # ============================================================
            # 키 입력 처리
            # ============================================================
            if key == 27 or key == ord('q'):
                # 종료
                break

            elif key == ord('r'):
                # 녹화 토글
                if fusion.is_recording:
                    record_data = fusion.stop_recording()
                    analysis = engine.get_latest_result()
                    exporter.export_all(record_data, analysis)
                else:
                    fusion.start_recording()

            elif key == ord('s'):
                # 스냅샷 (현재 분석 보고서 저장)
                if analysis_result:
                    report = predictor.generate_report(analysis_result)
                    print(report)
                    predictor.save_report(
                        analysis_result,
                        exporter._generate_filename('report', 'txt')
                    )
                    # 분석 CSV도 저장
                    exporter.export_analysis_csv(analysis_result)
                else:
                    print("[스냅샷] 아직 분석 결과가 없습니다.")

            elif key == ord(' '):
                # 일시정지
                paused = True
                print("[시스템] 일시정지 (SPACE로 재개)")

    except KeyboardInterrupt:
        print("\n[시스템] Ctrl+C 감지. 종료합니다...")

    except Exception as e:
        print(f"\n[오류] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ================================================================
        # 정리 (cleanup)
        # ================================================================
        print("\n[시스템] 종료 중...")

        # 시계열 로거 내보내기 (항상 저장 - 녹화 여부 무관)
        if logger and logger.frame_count > 0:
            logger.export_all()

        # 녹화 중이면 저장
        if fusion and fusion.is_recording:
            record_data = fusion.stop_recording()
            if exporter and record_data:
                analysis = engine.get_latest_result() if engine else None
                exporter.export_all(record_data, analysis)

        # 리소스 해제
        if engine:
            engine.shutdown()
        for analyzer in analyzers:
            analyzer.close()
        if camera:
            camera.stop()
        if display:
            display.destroy()

        print("[시스템] 종료 완료")


if __name__ == '__main__':
    main()
