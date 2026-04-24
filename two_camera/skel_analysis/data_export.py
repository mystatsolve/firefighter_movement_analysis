"""
data_export.py - CSV 내보내기 + Matplotlib 차트 + 시계열 로깅 모듈
===================================================================

녹화된 데이터를 CSV 파일로 내보내고, Matplotlib를 사용하여
분석 차트를 생성합니다.

제공 기능:
    1. TimeSeriesLogger: 모든 데이터를 시계열로 누적 저장 (핵심 추가)
       - 프레임별: 퓨전 각도, confidence, 카메라별 원시 각도
       - 분석별: 근육 스트레스, 인대 변형률, 부위별 위험도
       - 내보내기: 통합 CSV, 근육 시계열 CSV, 인대 시계열 CSV, 부위 위험도 CSV
    2. DataExporter: 기존 CSV/차트 내보내기 (호환 유지)

사용 예시:
    logger = TimeSeriesLogger()
    # 매 프레임마다
    logger.log_frame(fused, conf, angles_cam1, vis1, angles_cam2, vis2)
    # 분석 결과 도착 시
    logger.log_analysis(analysis_result)
    # 종료 시 전체 내보내기
    logger.export_all()
"""

import os
import csv
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import (
    JOINT_LABELS, RISK_LEVELS,
    MUSCLE_PARAMS, LIGAMENT_PARAMS, REGION_MAP,
)


# =============================================================================
# 관절/근육/인대 이름 상수 (CSV 헤더 생성용)
# =============================================================================
_JOINT_NAMES = list(JOINT_LABELS.keys())  # ['Knee_Angle', 'Hip_Angle', ...]
_MUSCLE_NAMES = list(MUSCLE_PARAMS.keys())  # ['Quadriceps', 'Hamstrings', ...]
_LIGAMENT_NAMES = list(LIGAMENT_PARAMS.keys())  # ['ACL', 'PCL', ...]
# 부위 목록 (REGION_MAP의 고유 값)
_REGION_NAMES = sorted(set(REGION_MAP.values()))


class TimeSeriesLogger:
    """
    모든 데이터를 시계열로 누적 저장하는 로거.

    매 프레임마다 관절 각도/confidence/카메라별 원시 데이터를 기록하고,
    생체역학 분석 결과가 도착할 때마다 근육/인대/부위 위험도를 기록합니다.

    내보내기 시 다음 CSV 파일을 생성합니다:
        1. full_timeseries.csv    - 프레임별 통합 데이터 (각도 + 최신 분석 forward-fill)
        2. muscle_timeseries.csv  - 분석별 8개 근육 스트레스 시계열
        3. ligament_timeseries.csv - 분석별 6개 인대 변형률 시계열
        4. region_risk_timeseries.csv - 분석별 부위 위험도 시계열

    사용 예시:
        logger = TimeSeriesLogger(output_dir='output')
        # 메인 루프에서
        logger.log_frame(fused, conf, cam1_angles, cam1_vis, cam2_angles, cam2_vis)
        if new_analysis:
            logger.log_analysis(analysis_result)
        # 종료 시
        logger.export_all()
    """

    def __init__(self, output_dir: str = 'output'):
        """
        Parameters
        ----------
        output_dir : str
            출력 파일 저장 디렉토리. 없으면 자동 생성.
        """
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._start_time = time.time()
        self._frame_count = 0

        # === 프레임 레벨 데이터 (매 프레임 1행) ===
        self._frame_rows: List[Dict] = []

        # === 분석 레벨 데이터 (분석 완료 시 1행) ===
        self._muscle_rows: List[Dict] = []      # 근육 스트레스 시계열
        self._ligament_rows: List[Dict] = []     # 인대 변형률 시계열
        self._region_rows: List[Dict] = []       # 부위 위험도 시계열

        # === 최신 분석 결과 캐시 (forward-fill용) ===
        self._latest_analysis_flat: Dict = {}

        # 세션 타임스탬프 (파일명용)
        self._session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    def _elapsed(self) -> float:
        """시작 시점으로부터 경과 시간 (초)."""
        return time.time() - self._start_time

    def log_frame(self,
                  fused_angles: Optional[Dict[str, float]],
                  confidence: Optional[Dict[str, float]],
                  cam1_angles: Optional[Dict[str, float]] = None,
                  cam1_vis: Optional[Dict[str, float]] = None,
                  cam2_angles: Optional[Dict[str, float]] = None,
                  cam2_vis: Optional[Dict[str, float]] = None):
        """
        매 프레임의 데이터를 기록합니다.

        Parameters
        ----------
        fused_angles : Dict - 퓨전된 관절 각도
        confidence : Dict - 관절별 신뢰도 (0~1)
        cam1_angles : Dict - 카메라1 원시 각도 (None이면 미검출)
        cam1_vis : Dict - 카메라1 visibility
        cam2_angles : Dict - 카메라2 원시 각도 (None이면 미검출)
        cam2_vis : Dict - 카메라2 visibility
        """
        row = {
            'time': round(self._elapsed(), 4),
            'frame': self._frame_count,
        }

        # 퓨전된 각도
        for jn in _JOINT_NAMES:
            row[jn] = round(fused_angles.get(jn, 0.0), 2) if fused_angles else ''
            row[f'{jn}_conf'] = round(confidence.get(jn, 0.0), 4) if confidence else ''

        # 카메라1 원시 각도 + visibility
        for jn in _JOINT_NAMES:
            row[f'cam1_{jn}'] = round(cam1_angles.get(jn, 0.0), 2) if cam1_angles else ''
            row[f'cam1_{jn}_vis'] = round(cam1_vis.get(jn, 0.0), 4) if cam1_vis else ''

        # 카메라2 원시 각도 + visibility
        for jn in _JOINT_NAMES:
            row[f'cam2_{jn}'] = round(cam2_angles.get(jn, 0.0), 2) if cam2_angles else ''
            row[f'cam2_{jn}_vis'] = round(cam2_vis.get(jn, 0.0), 4) if cam2_vis else ''

        # 최신 분석 결과 forward-fill (분석이 아직 없으면 빈 값)
        row.update(self._latest_analysis_flat)

        self._frame_rows.append(row)
        self._frame_count += 1

    def log_analysis(self, analysis_result: Dict):
        """
        생체역학 분석 결과를 기록합니다.

        BiomechEngine.get_latest_result()가 새 결과를 반환할 때마다 호출합니다.
        내부적으로 3가지 시계열(근육/인대/부위)에 각각 1행씩 추가하고,
        프레임 레벨 forward-fill용 캐시를 갱신합니다.

        Parameters
        ----------
        analysis_result : Dict
            BiomechEngine.get_latest_result()의 반환값
        """
        if not analysis_result:
            return

        t = round(self._elapsed(), 4)
        frame = self._frame_count

        # ── 근육 시계열 1행 ──
        muscle_row = {'time': t, 'frame': frame, 'overall_risk': analysis_result.get('overall_risk', ''),
                      'overall_score': analysis_result.get('overall_score', 0)}
        for mr in analysis_result.get('muscle_risks', []):
            name = mr['muscle_name']
            muscle_row[f'{name}_peak_kPa'] = round(mr['peak_stress_kPa'], 2)
            muscle_row[f'{name}_mean_kPa'] = round(mr['mean_stress_kPa'], 2)
            muscle_row[f'{name}_risk'] = mr['risk_level']
            muscle_row[f'{name}_fatigue'] = round(mr.get('fatigue_index', 0), 4)
        self._muscle_rows.append(muscle_row)

        # ── 인대 시계열 1행 ──
        lig_row = {'time': t, 'frame': frame, 'overall_risk': analysis_result.get('overall_risk', ''),
                   'overall_score': analysis_result.get('overall_score', 0)}
        for lr in analysis_result.get('ligament_risks', []):
            name = lr['ligament_name']
            lig_row[f'{name}_strain_pct'] = round(lr['peak_strain_pct'], 3)
            lig_row[f'{name}_force_N'] = round(lr['peak_force_N'], 2)
            lig_row[f'{name}_risk'] = lr['strain_risk']
        self._ligament_rows.append(lig_row)

        # ── 부위 위험도 시계열 1행 ──
        region_row = {'time': t, 'frame': frame, 'overall_risk': analysis_result.get('overall_risk', ''),
                      'overall_score': analysis_result.get('overall_score', 0)}
        for region, info in analysis_result.get('body_risks', {}).items():
            safe_name = region.replace('/', '_')
            region_row[f'{safe_name}_risk'] = info['risk_level']
            region_row[f'{safe_name}_score'] = info['risk_score']
        self._region_rows.append(region_row)

        # ── forward-fill 캐시 갱신 ──
        # 프레임 레벨 CSV에 분석 컬럼을 포함시키기 위해 최신 값을 캐시
        flat = {
            'overall_risk': analysis_result.get('overall_risk', ''),
            'overall_score': analysis_result.get('overall_score', 0),
        }
        for mr in analysis_result.get('muscle_risks', []):
            name = mr['muscle_name']
            flat[f'{name}_stress_kPa'] = round(mr['peak_stress_kPa'], 2)
            flat[f'{name}_risk'] = mr['risk_level']
        for lr in analysis_result.get('ligament_risks', []):
            name = lr['ligament_name']
            flat[f'{name}_strain_pct'] = round(lr['peak_strain_pct'], 3)
            flat[f'{name}_lig_risk'] = lr['strain_risk']
        for region, info in analysis_result.get('body_risks', {}).items():
            safe_name = region.replace('/', '_')
            flat[f'{safe_name}_region_risk'] = info['risk_level']
        self._latest_analysis_flat = flat

    def _write_csv(self, rows: List[Dict], filepath: str) -> str:
        """딕셔너리 리스트를 CSV로 저장. 모든 행의 키 합집합을 헤더로 사용."""
        if not rows:
            return ""
        # 모든 행의 키 합집합 (순서 유지)
        all_keys = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, '') for k in all_keys})
        return filepath

    def export_all(self) -> Dict[str, str]:
        """
        누적된 모든 시계열 데이터를 CSV 파일로 내보냅니다.

        생성 파일:
            - full_timeseries_*.csv      프레임별 통합 (각도+conf+카메라별+분석 forward-fill)
            - muscle_timeseries_*.csv    분석별 근육 스트레스 시계열
            - ligament_timeseries_*.csv  분석별 인대 변형률 시계열
            - region_risk_timeseries_*.csv 분석별 부위 위험도 시계열

        Returns
        -------
        Dict[str, str]
            {파일종류: 저장경로} 딕셔너리
        """
        results = {}
        ts = self._session_ts

        if self._frame_rows:
            path = os.path.join(self._output_dir, f'full_timeseries_{ts}.csv')
            self._write_csv(self._frame_rows, path)
            results['full_timeseries'] = path
            print(f"[TimeSeriesLogger] 통합 시계열 CSV: {path} ({len(self._frame_rows)} 프레임)")

        if self._muscle_rows:
            path = os.path.join(self._output_dir, f'muscle_timeseries_{ts}.csv')
            self._write_csv(self._muscle_rows, path)
            results['muscle_timeseries'] = path
            print(f"[TimeSeriesLogger] 근육 스트레스 CSV: {path} ({len(self._muscle_rows)} 분석)")

        if self._ligament_rows:
            path = os.path.join(self._output_dir, f'ligament_timeseries_{ts}.csv')
            self._write_csv(self._ligament_rows, path)
            results['ligament_timeseries'] = path
            print(f"[TimeSeriesLogger] 인대 변형률 CSV: {path} ({len(self._ligament_rows)} 분석)")

        if self._region_rows:
            path = os.path.join(self._output_dir, f'region_risk_timeseries_{ts}.csv')
            self._write_csv(self._region_rows, path)
            results['region_risk_timeseries'] = path
            print(f"[TimeSeriesLogger] 부위 위험도 CSV: {path} ({len(self._region_rows)} 분석)")

        total = sum(1 for v in results.values() if v)
        print(f"\n[TimeSeriesLogger] 총 {total}개 시계열 CSV 내보내기 완료")
        return results

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def analysis_count(self) -> int:
        return len(self._muscle_rows)


class DataExporter:
    """
    데이터 내보내기 및 차트 생성기.

    AngleFusionEngine의 녹화 데이터와 BiomechEngine의 분석 결과를
    CSV 파일과 Matplotlib 차트로 변환합니다.
    """

    def __init__(self, output_dir: str = 'output'):
        """
        Parameters
        ----------
        output_dir : str
            출력 파일 저장 디렉토리. 없으면 자동 생성.
        """
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _generate_filename(self, prefix: str, extension: str) -> str:
        """타임스탬프 기반 파일명 생성."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(self._output_dir, f'{prefix}_{timestamp}.{extension}')

    def export_csv(self, record_data: List[Dict], filepath: str = None) -> str:
        """
        녹화 데이터를 CSV 파일로 내보냅니다.

        CSV 컬럼: time, frame, Knee_Angle, Hip_Angle, ..., Knee_Angle_conf, ...

        Parameters
        ----------
        record_data : List[Dict]
            AngleFusionEngine.stop_recording()의 반환값
        filepath : str, optional
            저장 경로. None이면 자동 생성.

        Returns
        -------
        str
            저장된 파일 경로
        """
        if not record_data:
            print("[DataExporter] 내보낼 데이터가 없습니다.")
            return ""

        if filepath is None:
            filepath = self._generate_filename('joint_angles', 'csv')

        # 컬럼 헤더 추출 (첫 번째 레코드의 키)
        headers = list(record_data[0].keys())

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in record_data:
                writer.writerow(row)

        print(f"[DataExporter] CSV 저장: {filepath} ({len(record_data)} 행)")
        return filepath

    def export_analysis_csv(self, analysis_result: Dict, filepath: str = None) -> str:
        """
        생체역학 분석 결과를 CSV로 내보냅니다.

        Parameters
        ----------
        analysis_result : Dict
            BiomechEngine.get_latest_result()의 반환값
        filepath : str, optional
            저장 경로. None이면 자동 생성.

        Returns
        -------
        str
            저장된 파일 경로
        """
        if not analysis_result:
            print("[DataExporter] 분석 결과가 없습니다.")
            return ""

        if filepath is None:
            filepath = self._generate_filename('analysis', 'csv')

        rows = []

        # 근육 데이터
        for mr in analysis_result.get('muscle_risks', []):
            rows.append({
                'type': 'muscle',
                'name': mr['muscle_name'],
                'display_name': mr['display_name'],
                'peak_value': mr['peak_stress_kPa'],
                'mean_value': mr['mean_stress_kPa'],
                'risk_level': mr['risk_level'],
                'fatigue_index': mr.get('fatigue_index', 0),
            })

        # 인대 데이터
        for lr in analysis_result.get('ligament_risks', []):
            rows.append({
                'type': 'ligament',
                'name': lr['ligament_name'],
                'display_name': lr['display_name'],
                'peak_value': lr['peak_strain_pct'],
                'mean_value': 0,
                'risk_level': lr['strain_risk'],
                'fatigue_index': 0,
            })

        if not rows:
            return ""

        headers = list(rows[0].keys())
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"[DataExporter] 분석 CSV 저장: {filepath}")
        return filepath

    def plot_angles(self, record_data: List[Dict], filepath: str = None) -> str:
        """
        관절 각도 시계열 차트를 생성합니다.

        Parameters
        ----------
        record_data : List[Dict]
            녹화 데이터
        filepath : str, optional
            저장 경로. None이면 자동 생성.

        Returns
        -------
        str
            저장된 파일 경로
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # 비GUI 백엔드
            import matplotlib.pyplot as plt
        except ImportError:
            print("[DataExporter] matplotlib가 설치되지 않았습니다.")
            return ""

        if not record_data:
            return ""

        if filepath is None:
            filepath = self._generate_filename('angles_chart', 'png')

        # 시간 배열 추출
        times = [r['time'] for r in record_data]

        # 관절별 각도 추출
        angle_names = [k for k in record_data[0].keys()
                       if k.endswith('_Angle') and not k.endswith('_conf')]

        fig, axes = plt.subplots(len(angle_names), 1, figsize=(12, 3 * len(angle_names)),
                                 sharex=True)
        if len(angle_names) == 1:
            axes = [axes]

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']

        for i, name in enumerate(angle_names):
            values = [r.get(name, 0) for r in record_data]
            label = JOINT_LABELS.get(name, name)
            color = colors[i % len(colors)]

            axes[i].plot(times, values, color=color, linewidth=1.5, label=label)
            axes[i].set_ylabel(f'{label} (deg)')
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 200)

        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('Joint Angle Time Series (Dual Camera Fused)', fontsize=14)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[DataExporter] 각도 차트 저장: {filepath}")
        return filepath

    def plot_muscle_stress(self, analysis_result: Dict, filepath: str = None) -> str:
        """
        근육 스트레스 바 차트를 생성합니다.

        Parameters
        ----------
        analysis_result : Dict
            생체역학 분석 결과
        filepath : str, optional
            저장 경로

        Returns
        -------
        str
            저장된 파일 경로
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[DataExporter] matplotlib가 설치되지 않았습니다.")
            return ""

        if not analysis_result or 'muscle_risks' not in analysis_result:
            return ""

        if filepath is None:
            filepath = self._generate_filename('muscle_stress', 'png')

        muscle_risks = analysis_result['muscle_risks']
        names = [mr['display_name'] for mr in muscle_risks]
        peak_values = [mr['peak_stress_kPa'] for mr in muscle_risks]
        risk_levels = [mr['risk_level'] for mr in muscle_risks]

        # 위험도별 색상
        risk_color_map = {
            'Normal': '#4CAF50', 'Low': '#8BC34A', 'Medium': '#FF9800',
            'High': '#FF5722', 'Critical': '#F44336',
        }
        colors = [risk_color_map.get(r, '#9E9E9E') for r in risk_levels]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, peak_values, color=colors, edgecolor='white', linewidth=0.5)

        # 임계값 표시선
        ax.axvline(x=100, color='green', linestyle='--', alpha=0.5, label='Low (100 kPa)')
        ax.axvline(x=250, color='orange', linestyle='--', alpha=0.5, label='Medium (250 kPa)')
        ax.axvline(x=400, color='red', linestyle='--', alpha=0.5, label='High (400 kPa)')

        ax.set_xlabel('Peak Muscle Stress (kPa)')
        ax.set_title('Muscle Stress Analysis')
        ax.legend(loc='lower right')
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[DataExporter] 근육 스트레스 차트 저장: {filepath}")
        return filepath

    def plot_body_risks(self, analysis_result: Dict, filepath: str = None) -> str:
        """
        부위별 위험도 차트를 생성합니다.

        Parameters
        ----------
        analysis_result : Dict
            생체역학 분석 결과
        filepath : str, optional
            저장 경로

        Returns
        -------
        str
            저장된 파일 경로
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[DataExporter] matplotlib가 설치되지 않았습니다.")
            return ""

        if not analysis_result or 'body_risks' not in analysis_result:
            return ""

        if filepath is None:
            filepath = self._generate_filename('body_risks', 'png')

        body_risks = analysis_result['body_risks']
        regions = list(body_risks.keys())
        scores = [body_risks[r]['risk_score'] for r in regions]
        risk_levels_list = [body_risks[r]['risk_level'] for r in regions]

        risk_color_map = {
            'Normal': '#4CAF50', 'Low': '#8BC34A', 'Medium': '#FF9800',
            'High': '#FF5722', 'Critical': '#F44336',
        }
        colors = [risk_color_map.get(r, '#9E9E9E') for r in risk_levels_list]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(regions, scores, color=colors, edgecolor='white', linewidth=0.5)

        # 레벨 라벨
        for bar, level in zip(bars, risk_levels_list):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    level, va='center', fontsize=10)

        ax.set_xlabel('Risk Score (0-4)')
        ax.set_xlim(0, 5)
        ax.set_title('Body Region Risk Assessment')
        ax.set_xticks(range(5))
        ax.set_xticklabels(RISK_LEVELS)
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[DataExporter] 부위별 위험도 차트 저장: {filepath}")
        return filepath

    def export_all(self, record_data: List[Dict], analysis_result: Optional[Dict]) -> Dict[str, str]:
        """
        모든 데이터를 한 번에 내보냅니다.

        Parameters
        ----------
        record_data : List[Dict]
            녹화 데이터
        analysis_result : Dict
            분석 결과

        Returns
        -------
        Dict[str, str]
            {'csv': filepath, 'analysis_csv': filepath, 'angles_chart': filepath, ...}
        """
        results = {}

        if record_data:
            results['csv'] = self.export_csv(record_data)
            results['angles_chart'] = self.plot_angles(record_data)

        if analysis_result:
            results['analysis_csv'] = self.export_analysis_csv(analysis_result)
            results['muscle_chart'] = self.plot_muscle_stress(analysis_result)
            results['risk_chart'] = self.plot_body_risks(analysis_result)

        print(f"\n[DataExporter] 총 {len(results)}개 파일 내보내기 완료:")
        for key, path in results.items():
            if path:
                print(f"  {key}: {path}")

        return results
