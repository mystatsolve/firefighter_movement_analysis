"""
injury_predictor.py - 위험도 분류 및 부상 예측 보고서 모듈
==========================================================

생체역학 엔진(biomech_engine.py)의 시뮬레이션 결과를 받아서
위험도 분류, 신체 부위 매핑, 텍스트 보고서를 생성합니다.

biomech_engine.py에서 분리된 모듈로, 다음 기능을 제공합니다:
    1. 분석 결과 해석 및 요약
    2. 부위별 위험도 텍스트 보고서 생성
    3. 위험 경고 메시지 생성
    4. 보고서를 파일로 저장

사용 예시:
    predictor = InjuryPredictor()
    result = engine.get_latest_result()
    if result:
        report = predictor.generate_report(result)
        warnings = predictor.get_warnings(result)
"""

import time
from datetime import datetime
from typing import Dict, List, Optional

from config import (
    RISK_LEVELS, RISK_COLORS, REGION_MAP,
    MUSCLE_PARAMS, LIGAMENT_PARAMS,
)


class InjuryPredictor:
    """
    부상 위험도 분류 및 보고서 생성기.

    생체역학 분석 결과를 인간이 읽을 수 있는 보고서로 변환합니다.
    """

    # 위험도별 한국어 설명
    RISK_DESCRIPTIONS_KR = {
        'Normal': '정상 범위 - 안전한 자세입니다.',
        'Low': '낮은 위험 - 장시간 유지 시 피로 누적 가능.',
        'Medium': '중간 위험 - 자세 교정이 필요합니다.',
        'High': '높은 위험 - 즉시 자세 변경 권장.',
        'Critical': '매우 위험 - 부상 가능성이 높습니다. 즉시 중단하세요.',
    }

    # 부위별 한국어 이름
    REGION_NAMES_KR = {
        'Knee': '무릎',
        'Knee/Thigh': '무릎/허벅지',
        'Ankle/Calf': '발목/종아리',
        'Hip': '고관절',
        'Lumbar': '허리(요추)',
        'Shoulder': '어깨',
        'Shoulder/Back': '어깨/등',
        'Elbow': '팔꿈치',
    }

    def __init__(self):
        """InjuryPredictor 초기화."""
        self._report_count = 0

    def get_risk_summary(self, analysis_result: Dict) -> Dict:
        """
        분석 결과에서 주요 위험 요약 정보를 추출합니다.

        Parameters
        ----------
        analysis_result : Dict
            BiomechEngine.get_latest_result()의 반환값

        Returns
        -------
        Dict
            'overall_risk': 전체 위험도 문자열
            'overall_score': 위험 점수 (0~4)
            'high_risk_regions': 위험 부위 목록
            'top_muscle_risk': 최고 위험 근육 정보
            'top_ligament_risk': 최고 위험 인대 정보
        """
        if not analysis_result:
            return {
                'overall_risk': 'Normal',
                'overall_score': 0,
                'high_risk_regions': [],
                'top_muscle_risk': None,
                'top_ligament_risk': None,
            }

        # 위험 부위 추출 (Medium 이상)
        high_risk_regions = []
        for region, info in analysis_result.get('body_risks', {}).items():
            if info['risk_score'] >= 2:  # Medium 이상
                kr_name = self.REGION_NAMES_KR.get(region, region)
                high_risk_regions.append({
                    'region': region,
                    'region_kr': kr_name,
                    'risk_level': info['risk_level'],
                    'risk_score': info['risk_score'],
                })
        high_risk_regions.sort(key=lambda x: x['risk_score'], reverse=True)

        # 최고 위험 근육
        top_muscle = None
        max_muscle_score = 0
        for mr in analysis_result.get('muscle_risks', []):
            score = RISK_LEVELS.index(mr['risk_level']) if mr['risk_level'] in RISK_LEVELS else 0
            if score > max_muscle_score:
                max_muscle_score = score
                top_muscle = mr

        # 최고 위험 인대
        top_ligament = None
        max_lig_score = 0
        for lr in analysis_result.get('ligament_risks', []):
            score = RISK_LEVELS.index(lr['strain_risk']) if lr['strain_risk'] in RISK_LEVELS else 0
            if score > max_lig_score:
                max_lig_score = score
                top_ligament = lr

        return {
            'overall_risk': analysis_result.get('overall_risk', 'Normal'),
            'overall_score': analysis_result.get('overall_score', 0),
            'high_risk_regions': high_risk_regions,
            'top_muscle_risk': top_muscle,
            'top_ligament_risk': top_ligament,
        }

    def get_warnings(self, analysis_result: Dict) -> List[str]:
        """
        분석 결과에서 경고 메시지 목록을 생성합니다.

        Parameters
        ----------
        analysis_result : Dict
            BiomechEngine.get_latest_result()의 반환값

        Returns
        -------
        List[str]
            경고 메시지 리스트 (비어있으면 경고 없음)
        """
        warnings = []
        if not analysis_result:
            return warnings

        overall = analysis_result.get('overall_risk', 'Normal')
        if overall in ('High', 'Critical'):
            warnings.append(f"[{overall}] {self.RISK_DESCRIPTIONS_KR.get(overall, '')}")

        # 위험 부위별 경고
        for region, info in analysis_result.get('body_risks', {}).items():
            if info['risk_score'] >= 3:  # High 이상
                kr_name = self.REGION_NAMES_KR.get(region, region)
                warnings.append(f"  - {kr_name}: {info['risk_level']}")

        return warnings

    def generate_report(self, analysis_result: Dict) -> str:
        """
        분석 결과로부터 텍스트 보고서를 생성합니다.

        Parameters
        ----------
        analysis_result : Dict
            BiomechEngine.get_latest_result()의 반환값

        Returns
        -------
        str
            포맷된 텍스트 보고서
        """
        if not analysis_result:
            return "[보고서] 분석 데이터 없음"

        self._report_count += 1
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        summary = self.get_risk_summary(analysis_result)

        lines = [
            "=" * 60,
            f"  소방관 부상 위험도 분석 보고서 #{self._report_count}",
            f"  생성 시각: {now}",
            "=" * 60,
            "",
            f"  [종합 위험도] {summary['overall_risk']} "
            f"(점수: {summary['overall_score']}/4)",
            f"  {self.RISK_DESCRIPTIONS_KR.get(summary['overall_risk'], '')}",
            "",
        ]

        # 부위별 위험도
        lines.append("  --- 신체 부위별 위험도 ---")
        for region, info in analysis_result.get('body_risks', {}).items():
            kr_name = self.REGION_NAMES_KR.get(region, region)
            lines.append(f"    {kr_name:12s}: {info['risk_level']}")
        lines.append("")

        # 근육 상세
        lines.append("  --- 근육 스트레스 ---")
        for mr in sorted(analysis_result.get('muscle_risks', []),
                         key=lambda x: x['peak_stress_kPa'], reverse=True):
            lines.append(
                f"    {mr['display_name']:8s}: "
                f"피크 {mr['peak_stress_kPa']:6.1f} kPa, "
                f"평균 {mr['mean_stress_kPa']:6.1f} kPa "
                f"[{mr['risk_level']}]"
            )
        lines.append("")

        # 인대 상세
        lines.append("  --- 인대 변형률 ---")
        for lr in sorted(analysis_result.get('ligament_risks', []),
                         key=lambda x: x['peak_strain_pct'], reverse=True):
            lines.append(
                f"    {lr['display_name']:12s}: "
                f"피크 {lr['peak_strain_pct']:5.2f}%, "
                f"힘 {lr['peak_force_N']:7.1f} N "
                f"[{lr['strain_risk']}]"
            )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def save_report(self, analysis_result: Dict, filepath: str):
        """
        보고서를 파일로 저장합니다.

        Parameters
        ----------
        analysis_result : Dict
            BiomechEngine.get_latest_result()의 반환값
        filepath : str
            저장할 파일 경로
        """
        report = self.generate_report(analysis_result)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"[InjuryPredictor] 보고서 저장: {filepath}")
