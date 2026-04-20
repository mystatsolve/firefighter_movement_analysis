"""
소방관 근골격계 부상 예측 시스템
- OpenSim DeGrooteFregly2016Muscle 모델 기반 근육 내부 압력 수치화
- OpenSim Blankevoort1991Ligament 모델 기반 인대 장력 수치화
- 소방관 활동별 부상 위험도 예측

참고: https://github.com/opensim-org/opensim-core
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ============================================================================
# 1. DeGrooteFregly2016Muscle - OpenSim 기반 Hill-type 근육 모델
# ============================================================================

class DeGrooteFregly2016Muscle:
    """
    OpenSim DeGrooteFregly2016Muscle 모델의 Python 구현.
    Hill-type 근육 모델로 활성/수동 섬유력, 건 힘, 근육 내부 압력(스트레스)을 계산.

    Reference:
        De Groote et al. (2016). Evaluation of Direct Collocation Optimal Control
        Problem Formulations for Solving the Muscle Redundancy Problem.
        Annals of Biomedical Engineering, 44(10), 2922-2936.
    """

    # --- 모델 상수 (OpenSim DeGrooteFregly2016Muscle.h 기준) ---

    # 활성 힘-길이 곡선 (Active Force-Length) - 3개 가우시안
    _b11, _b21, _b31, _b41 = 0.8150671134243542, 1.055033428970575, 0.162384573599574, 0.063303448465465
    _b12, _b22, _b32, _b42 = 0.433004984392647, 0.716775413397760, -0.029947116970696, 0.200356847296188
    _b13, _b23, _b33, _b43 = 0.1, 1.0, 0.353553390593274, 0.0

    # 수동 힘-길이 곡선 (Passive Force-Length)
    _kPE = 4.0

    # 건 힘-길이 곡선 (Tendon Force-Length)
    _c1 = 0.200
    _c2 = 1.0
    _c3 = 0.200

    # 힘-속도 곡선 (Force-Velocity)
    _d1 = -0.3211346127989808
    _d2 = -8.149
    _d3 = -0.374
    _d4 = 0.8825327733249912

    # 섬유 길이 범위
    MIN_NORM_FIBER_LENGTH = 0.2
    MAX_NORM_FIBER_LENGTH = 1.8

    def __init__(self, name: str, max_isometric_force: float,
                 optimal_fiber_length: float, tendon_slack_length: float,
                 pennation_angle_at_optimal: float, pcsa: float,
                 activation_time_constant: float = 0.015,
                 deactivation_time_constant: float = 0.060,
                 fiber_damping: float = 0.01,
                 passive_fiber_strain_at_one_norm_force: float = 0.6,
                 tendon_strain_at_one_norm_force: float = 0.049):
        """
        Parameters
        ----------
        name : 근육 이름
        max_isometric_force : 최대 등척성 힘 (N)
        optimal_fiber_length : 최적 섬유 길이 (m)
        tendon_slack_length : 건 이완 길이 (m)
        pennation_angle_at_optimal : 최적 길이에서의 우각 (rad)
        pcsa : 생리학적 단면적 (m^2)
        activation_time_constant : 활성화 시간 상수 (s)
        deactivation_time_constant : 비활성화 시간 상수 (s)
        fiber_damping : 섬유 감쇠 계수
        passive_fiber_strain_at_one_norm_force : 1 정규화 힘에서의 수동 변형률
        tendon_strain_at_one_norm_force : 1 정규화 힘에서의 건 변형률
        """
        self.name = name
        self.max_isometric_force = max_isometric_force
        self.optimal_fiber_length = optimal_fiber_length
        self.tendon_slack_length = tendon_slack_length
        self.pennation_angle_at_optimal = pennation_angle_at_optimal
        self.pcsa = pcsa
        self.tau_a = activation_time_constant
        self.tau_d = deactivation_time_constant
        self.fiber_damping = fiber_damping

        # 수동 힘-길이 곡선 파라미터 계산
        self._e0 = passive_fiber_strain_at_one_norm_force

        # 건 힘-길이 곡선 kT 계산
        self._kT = np.log((1.0 + self._c3) / self._c1) / tendon_strain_at_one_norm_force

    # --- 힘-길이-속도 곡선 ---

    @staticmethod
    def _gaussian_like(x: float, b1: float, b2: float, b3: float, b4: float) -> float:
        """가우시안 유사 함수 (OpenSim 기준)"""
        denom = b3 + b4 * x
        if abs(denom) < 1e-12:
            denom = 1e-12
        return b1 * np.exp(-0.5 * ((x - b2) / denom) ** 2)

    def calc_active_force_length_multiplier(self, norm_fiber_length: np.ndarray) -> np.ndarray:
        """활성 힘-길이 배율 (3개 가우시안 합)"""
        fl = np.zeros_like(norm_fiber_length, dtype=float)
        for i, lm in enumerate(np.atleast_1d(norm_fiber_length)):
            fl_val = (self._gaussian_like(lm, self._b11, self._b21, self._b31, self._b41) +
                      self._gaussian_like(lm, self._b12, self._b22, self._b32, self._b42) +
                      self._gaussian_like(lm, self._b13, self._b23, self._b33, self._b43))
            fl[i] = fl_val
        return fl

    def calc_passive_force_length_multiplier(self, norm_fiber_length: np.ndarray) -> np.ndarray:
        """수동 힘-길이 배율 (지수함수 기반)"""
        nfl = np.atleast_1d(norm_fiber_length)
        num = np.exp(self._kPE * (nfl - 1.0) / self._e0)
        denom = np.exp(self._kPE)
        offset_num = np.exp(self._kPE * (self.MIN_NORM_FIBER_LENGTH - 1.0) / self._e0)
        result = (num - offset_num) / (denom - offset_num)
        result = np.maximum(result, 0.0)
        return result

    def calc_force_velocity_multiplier(self, norm_fiber_velocity: np.ndarray) -> np.ndarray:
        """힘-속도 배율 (De Groote 2016 로그 함수)"""
        v = np.atleast_1d(norm_fiber_velocity)
        inner = self._d2 * v + self._d3
        fv = self._d1 * np.log(inner + np.sqrt(inner ** 2 + 1.0)) + self._d4
        return np.clip(fv, 0.0, 1.8)

    def calc_tendon_force_multiplier(self, norm_tendon_length: np.ndarray) -> np.ndarray:
        """건 힘-길이 배율"""
        ntl = np.atleast_1d(norm_tendon_length)
        ft = self._c1 * np.exp(self._kT * (ntl - self._c2)) - self._c3
        return np.maximum(ft, 0.0)

    # --- 근육 상태 계산 ---

    def compute_muscle_state(self, activation: float,
                             norm_fiber_length: float,
                             norm_fiber_velocity: float) -> dict:
        """
        주어진 활성화, 섬유 길이, 섬유 속도에서 근육 상태 계산.

        Returns
        -------
        dict : 활성력, 수동력, 감쇠력, 총 섬유력, 건 방향 힘, 근육 스트레스(내부 압력)
        """
        nfl = np.atleast_1d(np.array([norm_fiber_length]))
        nfv = np.atleast_1d(np.array([norm_fiber_velocity]))

        f_act_fl = self.calc_active_force_length_multiplier(nfl)[0]
        f_pass_fl = self.calc_passive_force_length_multiplier(nfl)[0]
        f_fv = self.calc_force_velocity_multiplier(nfv)[0]

        cos_penn = np.cos(self.pennation_angle_at_optimal)

        active_force = self.max_isometric_force * activation * f_act_fl * f_fv
        passive_force = self.max_isometric_force * f_pass_fl
        damping_force = self.max_isometric_force * self.fiber_damping * norm_fiber_velocity
        total_fiber_force = active_force + passive_force + damping_force
        force_along_tendon = total_fiber_force * cos_penn

        # 근육 내부 압력 (스트레스) = 힘 / 단면적
        muscle_stress = total_fiber_force / self.pcsa  # Pa

        return {
            'active_fiber_force': active_force,
            'passive_fiber_force': passive_force,
            'damping_force': damping_force,
            'total_fiber_force': total_fiber_force,
            'force_along_tendon': force_along_tendon,
            'muscle_stress_Pa': muscle_stress,
            'muscle_stress_kPa': muscle_stress / 1000.0,
            'f_active_fl': f_act_fl,
            'f_passive_fl': f_pass_fl,
            'f_fv': f_fv,
        }

    # --- 활성화 동역학 ---

    def activation_derivative(self, activation: float, excitation: float) -> float:
        """
        활성화 동역학 ODE (OpenSim smoothed first-order dynamics).
        da/dt = [(f+0.5)/(tau_a*z) + (-f+0.5)*z/tau_d] * (e - a)
        """
        smoothing = 10.0
        f = 0.5 * np.tanh(smoothing * (excitation - activation))
        z = 0.5 + 1.5 * activation
        da_dt = ((f + 0.5) / (self.tau_a * z) +
                 (-f + 0.5) * z / self.tau_d) * (excitation - activation)
        return da_dt

    def simulate(self, time_array: np.ndarray,
                 excitation_profile: np.ndarray,
                 norm_fiber_length_profile: np.ndarray,
                 norm_fiber_velocity_profile: np.ndarray) -> dict:
        """
        시간에 따른 근육 시뮬레이션 실행.

        Parameters
        ----------
        time_array : 시간 배열 (s)
        excitation_profile : 신경 흥분 프로파일 [0, 1]
        norm_fiber_length_profile : 정규화 섬유 길이 프로파일
        norm_fiber_velocity_profile : 정규화 섬유 속도 프로파일

        Returns
        -------
        dict : 시간 시계열 결과
        """
        n = len(time_array)

        # 활성화 동역학 적분
        def ode_func(t, y):
            exc = np.interp(t, time_array, excitation_profile)
            return [self.activation_derivative(y[0], exc)]

        sol = solve_ivp(ode_func, [time_array[0], time_array[-1]],
                        [0.05], t_eval=time_array, method='RK45',
                        max_step=0.005)
        activations = np.clip(sol.y[0], 0.01, 1.0)

        # 각 시점에서 근육 상태 계산
        results = {
            'time': time_array,
            'activation': activations,
            'excitation': excitation_profile,
            'norm_fiber_length': norm_fiber_length_profile,
            'norm_fiber_velocity': norm_fiber_velocity_profile,
            'active_fiber_force': np.zeros(n),
            'passive_fiber_force': np.zeros(n),
            'damping_force': np.zeros(n),
            'total_fiber_force': np.zeros(n),
            'force_along_tendon': np.zeros(n),
            'muscle_stress_Pa': np.zeros(n),
            'muscle_stress_kPa': np.zeros(n),
        }

        for i in range(n):
            state = self.compute_muscle_state(
                activations[i],
                norm_fiber_length_profile[i],
                norm_fiber_velocity_profile[i]
            )
            for key in ['active_fiber_force', 'passive_fiber_force', 'damping_force',
                        'total_fiber_force', 'force_along_tendon',
                        'muscle_stress_Pa', 'muscle_stress_kPa']:
                results[key][i] = state[key]

        return results


# ============================================================================
# 2. Blankevoort1991Ligament - OpenSim 기반 인대 모델
# ============================================================================

class Blankevoort1991Ligament:
    """
    OpenSim Blankevoort1991Ligament 모델의 Python 구현.
    비선형 힘-변형률 관계로 인대 장력을 계산.

    Reference:
        Blankevoort, L. and Huiskes, R. (1991). Ligament-bone interaction in a
        three-dimensional model of the knee. J Biomech Eng, 113(3), 263-269.
    """

    def __init__(self, name: str, slack_length: float,
                 linear_stiffness: float,
                 transition_strain: float = 0.06,
                 damping_coefficient: float = 0.003):
        """
        Parameters
        ----------
        name : 인대 이름
        slack_length : 이완 길이 (m)
        linear_stiffness : 선형 강성 (N/strain)
        transition_strain : 발끝-선형 전환 변형률
        damping_coefficient : 감쇠 계수 (N·s/strain)
        """
        self.name = name
        self.slack_length = slack_length
        self.linear_stiffness = linear_stiffness
        self.transition_strain = transition_strain
        self.damping_coefficient = damping_coefficient

    def calc_strain(self, length: float) -> float:
        """변형률 계산"""
        return length / self.slack_length - 1.0

    def calc_strain_rate(self, lengthening_speed: float) -> float:
        """변형률 속도 계산"""
        return lengthening_speed / self.slack_length

    def calc_spring_force(self, strain: float) -> float:
        """
        스프링 힘 계산 (3구간 비선형).
        - strain <= 0: 0 (이완)
        - 0 < strain < e_t: 0.5 * k / e_t * strain^2 (발끝 구간)
        - strain >= e_t: k * (strain - e_t/2) (선형 구간)
        """
        if strain <= 0:
            return 0.0
        elif strain < self.transition_strain:
            return 0.5 * self.linear_stiffness / self.transition_strain * strain ** 2
        else:
            return self.linear_stiffness * (strain - self.transition_strain / 2.0)

    def calc_damping_force(self, strain: float, strain_rate: float) -> float:
        """감쇠력 계산 (인장 상태에서만)"""
        if strain <= 0 or strain_rate <= 0:
            return 0.0
        # 부드러운 전환 함수
        smooth = 0.5 * (1.0 + np.tanh(strain / 0.005))
        return self.damping_coefficient * strain_rate * smooth

    def calc_total_force(self, strain: float, strain_rate: float) -> float:
        """총 인대 장력 (인장만 허용)"""
        spring_f = self.calc_spring_force(strain)
        damp_f = self.calc_damping_force(strain, strain_rate)
        return max(0.0, spring_f + damp_f)

    def simulate(self, time_array: np.ndarray,
                 length_profile: np.ndarray,
                 velocity_profile: np.ndarray) -> dict:
        """
        시간에 따른 인대 시뮬레이션 실행.

        Parameters
        ----------
        time_array : 시간 배열 (s)
        length_profile : 인대 길이 프로파일 (m)
        velocity_profile : 인대 신장 속도 프로파일 (m/s)

        Returns
        -------
        dict : 시간 시계열 결과
        """
        n = len(time_array)
        results = {
            'time': time_array,
            'length': length_profile,
            'velocity': velocity_profile,
            'strain': np.zeros(n),
            'strain_rate': np.zeros(n),
            'spring_force': np.zeros(n),
            'damping_force': np.zeros(n),
            'total_force': np.zeros(n),
            'strain_percent': np.zeros(n),
        }

        for i in range(n):
            strain = self.calc_strain(length_profile[i])
            strain_rate = self.calc_strain_rate(velocity_profile[i])
            results['strain'][i] = strain
            results['strain_rate'][i] = strain_rate
            results['spring_force'][i] = self.calc_spring_force(strain)
            results['damping_force'][i] = self.calc_damping_force(strain, strain_rate)
            results['total_force'][i] = self.calc_total_force(strain, strain_rate)
            results['strain_percent'][i] = strain * 100.0

        return results


# ============================================================================
# 3. InjuryPredictor - 부상 예측 엔진
# ============================================================================

class InjuryPredictor:
    """근육 스트레스 및 인대 변형률 기반 부상 위험도 예측"""

    # 근육 스트레스 임계값 (Pa)
    MUSCLE_STRESS_THRESHOLDS = {
        '낮음': 100e3,     # 100 kPa - 정상 운동 범위
        '중간': 250e3,     # 250 kPa - 상승된 위험
        '높음': 400e3,     # 400 kPa - 높은 위험
        '매우높음': 600e3,  # 600 kPa - 조직 손상 가능
    }

    # 인대 변형률 임계값
    LIGAMENT_STRAIN_THRESHOLDS = {
        '낮음': 0.03,      # 3% - 정상 생리학적 범위
        '중간': 0.06,      # 6% - 발끝-선형 전환점
        '높음': 0.10,      # 10% - 파괴 접근
        '매우높음': 0.15,   # 15% - 파열 구간
    }

    # 인대 힘 비율 임계값 (추정 파괴력 대비)
    LIGAMENT_FORCE_RATIO_THRESHOLDS = {
        '낮음': 0.25,
        '중간': 0.50,
        '높음': 0.75,
        '매우높음': 0.90,
    }

    @staticmethod
    def classify_risk(value: float, thresholds: dict) -> str:
        """위험도 분류"""
        levels = list(thresholds.keys())
        values = list(thresholds.values())

        for i in range(len(values) - 1, -1, -1):
            if value >= values[i]:
                return levels[i]
        return '정상'

    def compute_muscle_risk(self, muscle_name: str,
                            stress_series: np.ndarray,
                            dt: float) -> dict:
        """
        근육 부상 위험도 계산.

        Parameters
        ----------
        muscle_name : 근육 이름
        stress_series : 근육 스트레스 시계열 (Pa)
        dt : 시간 간격 (s)

        Returns
        -------
        dict : 위험도 분석 결과
        """
        peak_stress = np.max(stress_series)
        mean_stress = np.mean(stress_series)
        risk_level = self.classify_risk(peak_stress, self.MUSCLE_STRESS_THRESHOLDS)

        # 피로 누적 지수: 시간-가중 누적 스트레스 적분
        threshold = self.MUSCLE_STRESS_THRESHOLDS['중간']
        fatigue_integrand = np.maximum(stress_series / threshold, 0.0) ** 2
        fatigue_index = np.trapz(fatigue_integrand, dx=dt)

        # 위험 구간 시간 비율
        time_above_medium = np.sum(stress_series >= self.MUSCLE_STRESS_THRESHOLDS['중간']) * dt
        time_above_high = np.sum(stress_series >= self.MUSCLE_STRESS_THRESHOLDS['높음']) * dt
        total_time = len(stress_series) * dt

        return {
            'muscle_name': muscle_name,
            'peak_stress_kPa': peak_stress / 1000.0,
            'mean_stress_kPa': mean_stress / 1000.0,
            'risk_level': risk_level,
            'fatigue_index': fatigue_index,
            'time_above_medium_pct': time_above_medium / total_time * 100 if total_time > 0 else 0,
            'time_above_high_pct': time_above_high / total_time * 100 if total_time > 0 else 0,
        }

    def compute_ligament_risk(self, ligament_name: str,
                              strain_series: np.ndarray,
                              force_series: np.ndarray,
                              estimated_failure_force: float,
                              dt: float) -> dict:
        """
        인대 부상 위험도 계산.

        Parameters
        ----------
        ligament_name : 인대 이름
        strain_series : 변형률 시계열
        force_series : 장력 시계열 (N)
        estimated_failure_force : 추정 파괴력 (N)
        dt : 시간 간격 (s)

        Returns
        -------
        dict : 위험도 분석 결과
        """
        peak_strain = np.max(strain_series)
        peak_force = np.max(force_series)
        mean_force = np.mean(force_series)
        force_ratio = peak_force / estimated_failure_force if estimated_failure_force > 0 else 0

        strain_risk = self.classify_risk(peak_strain, self.LIGAMENT_STRAIN_THRESHOLDS)
        force_risk = self.classify_risk(force_ratio, self.LIGAMENT_FORCE_RATIO_THRESHOLDS)

        # 종합 위험도 (더 높은 쪽 기준)
        risk_levels = ['정상', '낮음', '중간', '높음', '매우높음']
        combined_idx = max(
            risk_levels.index(strain_risk) if strain_risk in risk_levels else 0,
            risk_levels.index(force_risk) if force_risk in risk_levels else 0
        )
        combined_risk = risk_levels[combined_idx]

        # 순환 하중 피로
        threshold = self.LIGAMENT_STRAIN_THRESHOLDS['중간']
        cycles_above = np.sum(np.diff(np.sign(strain_series - threshold)) > 0)

        return {
            'ligament_name': ligament_name,
            'peak_strain_pct': peak_strain * 100,
            'peak_force_N': peak_force,
            'mean_force_N': mean_force,
            'force_ratio_pct': force_ratio * 100,
            'strain_risk': strain_risk,
            'force_risk': force_risk,
            'combined_risk': combined_risk,
            'loading_cycles': int(cycles_above),
        }

    def compute_body_region_risks(self, muscle_risks: List[dict],
                                  ligament_risks: List[dict]) -> dict:
        """신체 부위별 종합 위험도 산출"""
        # 근육/인대를 신체 부위로 매핑
        region_map = {
            '대퇴사두근': '무릎', '비복근': '발목/종아리', '대둔근': '고관절',
            '척추기립근': '허리', '삼각근': '어깨', '상완이두근': '팔꿈치',
            '광배근': '어깨/등', '대퇴이두근': '무릎/허벅지',
            '전방십자인대(ACL)': '무릎', '후방십자인대(PCL)': '무릎',
            '슬개건': '무릎', '극상인대': '허리',
            '관절와상완인대': '어깨', '요추인대': '허리',
        }

        risk_levels = ['정상', '낮음', '중간', '높음', '매우높음']
        region_scores = {}

        for mr in muscle_risks:
            region = region_map.get(mr['muscle_name'], '기타')
            score = risk_levels.index(mr['risk_level']) if mr['risk_level'] in risk_levels else 0
            if region not in region_scores:
                region_scores[region] = []
            region_scores[region].append(score)

        for lr in ligament_risks:
            region = region_map.get(lr['ligament_name'], '기타')
            score = risk_levels.index(lr['combined_risk']) if lr['combined_risk'] in risk_levels else 0
            if region not in region_scores:
                region_scores[region] = []
            region_scores[region].append(score)

        body_risks = {}
        for region, scores in region_scores.items():
            max_score = max(scores) if scores else 0
            body_risks[region] = {
                'risk_level': risk_levels[max_score],
                'risk_score': max_score,
                'num_tissues': len(scores),
            }

        return body_risks

    def generate_report(self, scenario_name: str,
                        muscle_risks: List[dict],
                        ligament_risks: List[dict],
                        body_risks: dict) -> str:
        """한국어 부상 예측 보고서 생성"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"  부상 위험도 분석 보고서 - {scenario_name}")
        lines.append("=" * 70)
        lines.append("")

        # 근육 분석
        lines.append("■ 근육 내부 압력(스트레스) 분석")
        lines.append("-" * 50)
        lines.append(f"{'근육명':<15} {'최대 스트레스':>12} {'평균 스트레스':>12} {'위험도':>8} {'피로지수':>8}")
        lines.append("-" * 50)
        for mr in muscle_risks:
            lines.append(
                f"{mr['muscle_name']:<15} "
                f"{mr['peak_stress_kPa']:>10.1f}kPa "
                f"{mr['mean_stress_kPa']:>10.1f}kPa "
                f"{mr['risk_level']:>8} "
                f"{mr['fatigue_index']:>8.2f}"
            )
        lines.append("")

        # 인대 분석
        lines.append("■ 인대 장력 분석")
        lines.append("-" * 60)
        lines.append(f"{'인대명':<20} {'최대변형률':>8} {'최대장력':>10} {'변형위험':>8} {'장력위험':>8}")
        lines.append("-" * 60)
        for lr in ligament_risks:
            lines.append(
                f"{lr['ligament_name']:<20} "
                f"{lr['peak_strain_pct']:>6.2f}% "
                f"{lr['peak_force_N']:>8.1f}N "
                f"{lr['strain_risk']:>8} "
                f"{lr['force_risk']:>8}"
            )
        lines.append("")

        # 신체 부위별 위험도
        lines.append("■ 신체 부위별 종합 위험도")
        lines.append("-" * 40)
        for region, info in body_risks.items():
            risk_bar = "#" * info['risk_score'] + "-" * (4 - info['risk_score'])
            lines.append(f"  {region:<12} [{risk_bar}] {info['risk_level']}")
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


# ============================================================================
# 4. FirefighterScenario - 소방관 활동 시나리오
# ============================================================================

# 근육 생리학적 파라미터 (문헌 기반)
MUSCLE_PARAMS = {
    '대퇴사두근': {
        'max_isometric_force': 6000.0,
        'optimal_fiber_length': 0.084,
        'tendon_slack_length': 0.10,
        'pennation_angle_at_optimal': np.radians(15),
        'pcsa': 75e-4,  # 75 cm^2 -> m^2
    },
    '비복근': {
        'max_isometric_force': 1600.0,
        'optimal_fiber_length': 0.055,
        'tendon_slack_length': 0.39,
        'pennation_angle_at_optimal': np.radians(17),
        'pcsa': 30e-4,
    },
    '대둔근': {
        'max_isometric_force': 1500.0,
        'optimal_fiber_length': 0.14,
        'tendon_slack_length': 0.05,
        'pennation_angle_at_optimal': np.radians(5),
        'pcsa': 40e-4,
    },
    '척추기립근': {
        'max_isometric_force': 2500.0,
        'optimal_fiber_length': 0.12,
        'tendon_slack_length': 0.03,
        'pennation_angle_at_optimal': np.radians(0),
        'pcsa': 50e-4,
    },
    '삼각근': {
        'max_isometric_force': 1100.0,
        'optimal_fiber_length': 0.10,
        'tendon_slack_length': 0.04,
        'pennation_angle_at_optimal': np.radians(15),
        'pcsa': 20e-4,
    },
    '상완이두근': {
        'max_isometric_force': 600.0,
        'optimal_fiber_length': 0.116,
        'tendon_slack_length': 0.24,
        'pennation_angle_at_optimal': np.radians(0),
        'pcsa': 12e-4,
    },
    '광배근': {
        'max_isometric_force': 1200.0,
        'optimal_fiber_length': 0.25,
        'tendon_slack_length': 0.05,
        'pennation_angle_at_optimal': np.radians(20),
        'pcsa': 25e-4,
    },
    '대퇴이두근': {
        'max_isometric_force': 900.0,
        'optimal_fiber_length': 0.11,
        'tendon_slack_length': 0.32,
        'pennation_angle_at_optimal': np.radians(12),
        'pcsa': 18e-4,
    },
}

# 인대 파라미터 (문헌 기반)
LIGAMENT_PARAMS = {
    '전방십자인대(ACL)': {
        'slack_length': 0.032,
        'linear_stiffness': 5000.0,
        'transition_strain': 0.06,
        'damping_coefficient': 0.003,
        'estimated_failure_force': 2160.0,  # N
    },
    '후방십자인대(PCL)': {
        'slack_length': 0.038,
        'linear_stiffness': 9000.0,
        'transition_strain': 0.06,
        'damping_coefficient': 0.003,
        'estimated_failure_force': 3000.0,
    },
    '슬개건': {
        'slack_length': 0.050,
        'linear_stiffness': 15000.0,
        'transition_strain': 0.05,
        'damping_coefficient': 0.005,
        'estimated_failure_force': 10000.0,
    },
    '극상인대': {
        'slack_length': 0.045,
        'linear_stiffness': 3000.0,
        'transition_strain': 0.06,
        'damping_coefficient': 0.002,
        'estimated_failure_force': 1500.0,
    },
    '관절와상완인대': {
        'slack_length': 0.025,
        'linear_stiffness': 2000.0,
        'transition_strain': 0.06,
        'damping_coefficient': 0.002,
        'estimated_failure_force': 800.0,
    },
    '요추인대': {
        'slack_length': 0.055,
        'linear_stiffness': 4000.0,
        'transition_strain': 0.07,
        'damping_coefficient': 0.003,
        'estimated_failure_force': 2000.0,
    },
}


class FirefighterScenario:
    """소방관 활동별 운동 시나리오 정의"""

    @staticmethod
    def _generate_cyclic_excitation(time: np.ndarray, freq: float,
                                    base: float, amplitude: float,
                                    phase: float = 0.0) -> np.ndarray:
        """주기적 신경 흥분 프로파일 생성"""
        exc = base + amplitude * 0.5 * (1.0 + np.sin(2 * np.pi * freq * time + phase))
        return np.clip(exc, 0.01, 1.0)

    @staticmethod
    def _generate_fiber_trajectory(time: np.ndarray, freq: float,
                                   mean_length: float, amplitude: float,
                                   phase: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """정규화 섬유 길이 및 속도 프로파일 생성"""
        length = mean_length + amplitude * np.sin(2 * np.pi * freq * time + phase)
        velocity = amplitude * 2 * np.pi * freq * np.cos(2 * np.pi * freq * time + phase)
        # 정규화 속도 (최대 수축 속도 = 10 * optimal_fiber_length/s 기준)
        velocity_norm = velocity / 10.0
        return length, velocity_norm

    @staticmethod
    def _generate_ligament_trajectory(time: np.ndarray, slack_length: float,
                                      freq: float, strain_amplitude: float,
                                      mean_strain: float = 0.02,
                                      phase: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """인대 길이 및 속도 프로파일 생성"""
        strain = mean_strain + strain_amplitude * np.abs(np.sin(2 * np.pi * freq * time + phase))
        length = slack_length * (1.0 + strain)
        velocity = np.gradient(length, time)
        return length, velocity

    @staticmethod
    def stair_climbing() -> dict:
        """계단 오르기 시나리오 (25kg 장비 착용)"""
        dt = 0.005
        duration = 60.0
        time = np.arange(0, duration, dt)
        freq = 1.0  # 1 Hz 보행 주기

        muscles = {
            '대퇴사두근': {
                'excitation': FirefighterScenario._generate_cyclic_excitation(
                    time, freq, base=0.3, amplitude=0.5, phase=0),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=1.0, amplitude=0.15, phase=0),
            },
            '비복근': {
                'excitation': FirefighterScenario._generate_cyclic_excitation(
                    time, freq, base=0.2, amplitude=0.4, phase=np.pi / 3),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=0.95, amplitude=0.12, phase=np.pi / 3),
            },
            '대둔근': {
                'excitation': FirefighterScenario._generate_cyclic_excitation(
                    time, freq, base=0.25, amplitude=0.45, phase=np.pi / 6),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=1.05, amplitude=0.10, phase=np.pi / 6),
            },
        }

        ligaments = {
            '전방십자인대(ACL)': FirefighterScenario._generate_ligament_trajectory(
                time, LIGAMENT_PARAMS['전방십자인대(ACL)']['slack_length'],
                freq, strain_amplitude=0.03, mean_strain=0.02),
            '슬개건': FirefighterScenario._generate_ligament_trajectory(
                time, LIGAMENT_PARAMS['슬개건']['slack_length'],
                freq, strain_amplitude=0.04, mean_strain=0.03),
        }

        return {'name': '계단 오르기', 'time': time, 'muscles': muscles,
                'ligaments': ligaments, 'duration': duration, 'load_kg': 25}

    @staticmethod
    def equipment_carry() -> dict:
        """장비 운반 시나리오 (30kg)"""
        dt = 0.005
        duration = 120.0
        time = np.arange(0, duration, dt)
        freq = 0.8

        muscles = {
            '척추기립근': {
                'excitation': FirefighterScenario._generate_cyclic_excitation(
                    time, freq * 0.5, base=0.5, amplitude=0.3, phase=0),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq * 0.5, mean_length=0.95, amplitude=0.08, phase=0),
            },
            '삼각근': {
                'excitation': FirefighterScenario._generate_cyclic_excitation(
                    time, freq, base=0.35, amplitude=0.35, phase=np.pi / 4),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=1.0, amplitude=0.10, phase=np.pi / 4),
            },
            '상완이두근': {
                'excitation': FirefighterScenario._generate_cyclic_excitation(
                    time, freq, base=0.3, amplitude=0.3, phase=np.pi / 2),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=1.05, amplitude=0.12, phase=np.pi / 2),
            },
        }

        ligaments = {
            '극상인대': FirefighterScenario._generate_ligament_trajectory(
                time, LIGAMENT_PARAMS['극상인대']['slack_length'],
                freq * 0.5, strain_amplitude=0.04, mean_strain=0.03),
            '관절와상완인대': FirefighterScenario._generate_ligament_trajectory(
                time, LIGAMENT_PARAMS['관절와상완인대']['slack_length'],
                freq, strain_amplitude=0.03, mean_strain=0.02),
        }

        return {'name': '장비 운반', 'time': time, 'muscles': muscles,
                'ligaments': ligaments, 'duration': duration, 'load_kg': 30}

    @staticmethod
    def hose_pulling() -> dict:
        """호스 당기기 시나리오 (50~150N 저항)"""
        dt = 0.005
        duration = 30.0
        time = np.arange(0, duration, dt)
        freq = 0.5  # 느린 반복 당기기

        # 점진적 증가하는 흥분
        ramp = np.clip(time / duration, 0, 1)

        muscles = {
            '광배근': {
                'excitation': np.clip(0.3 + 0.5 * ramp +
                    0.15 * np.sin(2 * np.pi * freq * time), 0.01, 1.0),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=0.9, amplitude=0.15, phase=0),
            },
            '상완이두근': {
                'excitation': np.clip(0.25 + 0.45 * ramp +
                    0.15 * np.sin(2 * np.pi * freq * time + np.pi / 4), 0.01, 1.0),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=0.85, amplitude=0.18, phase=np.pi / 4),
            },
            '대퇴사두근': {
                'excitation': np.clip(0.2 + 0.3 * ramp +
                    0.1 * np.sin(2 * np.pi * freq * time + np.pi / 2), 0.01, 1.0),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=1.05, amplitude=0.10, phase=np.pi / 2),
            },
        }

        ligaments = {
            '관절와상완인대': FirefighterScenario._generate_ligament_trajectory(
                time, LIGAMENT_PARAMS['관절와상완인대']['slack_length'],
                freq, strain_amplitude=0.05, mean_strain=0.03),
            '전방십자인대(ACL)': FirefighterScenario._generate_ligament_trajectory(
                time, LIGAMENT_PARAMS['전방십자인대(ACL)']['slack_length'],
                freq, strain_amplitude=0.02, mean_strain=0.015),
        }

        return {'name': '호스 당기기', 'time': time, 'muscles': muscles,
                'ligaments': ligaments, 'duration': duration, 'load_kg': 0}

    @staticmethod
    def ladder_climb() -> dict:
        """사다리 오르기 시나리오"""
        dt = 0.005
        duration = 45.0
        time = np.arange(0, duration, dt)
        freq = 0.7  # 느린 오르기 주기

        muscles = {
            '삼각근': {
                'excitation': FirefighterScenario._generate_cyclic_excitation(
                    time, freq, base=0.4, amplitude=0.4, phase=0),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=1.0, amplitude=0.12, phase=0),
            },
            '대퇴사두근': {
                'excitation': FirefighterScenario._generate_cyclic_excitation(
                    time, freq, base=0.35, amplitude=0.45, phase=np.pi / 2),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=1.0, amplitude=0.14, phase=np.pi / 2),
            },
            '비복근': {
                'excitation': FirefighterScenario._generate_cyclic_excitation(
                    time, freq, base=0.2, amplitude=0.35, phase=np.pi),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=0.95, amplitude=0.10, phase=np.pi),
            },
        }

        ligaments = {
            '관절와상완인대': FirefighterScenario._generate_ligament_trajectory(
                time, LIGAMENT_PARAMS['관절와상완인대']['slack_length'],
                freq, strain_amplitude=0.04, mean_strain=0.025),
            '전방십자인대(ACL)': FirefighterScenario._generate_ligament_trajectory(
                time, LIGAMENT_PARAMS['전방십자인대(ACL)']['slack_length'],
                freq, strain_amplitude=0.025, mean_strain=0.02),
        }

        return {'name': '사다리 오르기', 'time': time, 'muscles': muscles,
                'ligaments': ligaments, 'duration': duration, 'load_kg': 25}

    @staticmethod
    def victim_rescue() -> dict:
        """요구조자 구출 시나리오 (80kg)"""
        dt = 0.005
        duration = 60.0
        time = np.arange(0, duration, dt)
        freq = 0.3  # 느린 끌기 동작

        # 높은 부하, 불규칙한 패턴
        burst = np.where((time % 10) < 5, 1.0, 0.6)  # 5초 작업 / 5초 조정

        muscles = {
            '척추기립근': {
                'excitation': np.clip(0.5 * burst +
                    0.2 * np.sin(2 * np.pi * freq * time), 0.01, 1.0),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=0.85, amplitude=0.12, phase=0),
            },
            '대퇴사두근': {
                'excitation': np.clip(0.4 * burst +
                    0.25 * np.sin(2 * np.pi * freq * time + np.pi / 3), 0.01, 1.0),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=0.95, amplitude=0.15, phase=np.pi / 3),
            },
            '대퇴이두근': {
                'excitation': np.clip(0.35 * burst +
                    0.2 * np.sin(2 * np.pi * freq * time + np.pi / 2), 0.01, 1.0),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=1.1, amplitude=0.13, phase=np.pi / 2),
            },
            '대둔근': {
                'excitation': np.clip(0.45 * burst +
                    0.15 * np.sin(2 * np.pi * freq * time + np.pi), 0.01, 1.0),
                'fiber_length': FirefighterScenario._generate_fiber_trajectory(
                    time, freq, mean_length=1.0, amplitude=0.10, phase=np.pi),
            },
        }

        ligaments = {
            '요추인대': FirefighterScenario._generate_ligament_trajectory(
                time, LIGAMENT_PARAMS['요추인대']['slack_length'],
                freq, strain_amplitude=0.06, mean_strain=0.04),
            '전방십자인대(ACL)': FirefighterScenario._generate_ligament_trajectory(
                time, LIGAMENT_PARAMS['전방십자인대(ACL)']['slack_length'],
                freq, strain_amplitude=0.035, mean_strain=0.025),
            '후방십자인대(PCL)': FirefighterScenario._generate_ligament_trajectory(
                time, LIGAMENT_PARAMS['후방십자인대(PCL)']['slack_length'],
                freq, strain_amplitude=0.03, mean_strain=0.02),
        }

        return {'name': '요구조자 구출', 'time': time, 'muscles': muscles,
                'ligaments': ligaments, 'duration': duration, 'load_kg': 80}


# ============================================================================
# 5. SimulationEngine - 시뮬레이션 실행 엔진
# ============================================================================

class SimulationEngine:
    """근골격계 시뮬레이션 실행 및 부상 예측 통합 엔진"""

    def __init__(self):
        self.predictor = InjuryPredictor()
        self.scenarios = {
            '계단 오르기': FirefighterScenario.stair_climbing,
            '장비 운반': FirefighterScenario.equipment_carry,
            '호스 당기기': FirefighterScenario.hose_pulling,
            '사다리 오르기': FirefighterScenario.ladder_climb,
            '요구조자 구출': FirefighterScenario.victim_rescue,
        }

    def run_scenario(self, scenario_func) -> dict:
        """단일 시나리오 시뮬레이션 실행"""
        scenario = scenario_func()
        time = scenario['time']
        dt = time[1] - time[0]

        # 근육 시뮬레이션
        muscle_results = {}
        muscle_risks = []

        for muscle_name, profiles in scenario['muscles'].items():
            params = MUSCLE_PARAMS[muscle_name]
            muscle = DeGrooteFregly2016Muscle(
                name=muscle_name, **params
            )

            fiber_length, fiber_velocity = profiles['fiber_length']
            result = muscle.simulate(time, profiles['excitation'],
                                     fiber_length, fiber_velocity)
            muscle_results[muscle_name] = result

            risk = self.predictor.compute_muscle_risk(
                muscle_name, result['muscle_stress_Pa'], dt
            )
            muscle_risks.append(risk)

        # 인대 시뮬레이션
        ligament_results = {}
        ligament_risks = []

        for lig_name, (length_profile, velocity_profile) in scenario['ligaments'].items():
            params = LIGAMENT_PARAMS[lig_name]
            ligament = Blankevoort1991Ligament(
                name=lig_name,
                slack_length=params['slack_length'],
                linear_stiffness=params['linear_stiffness'],
                transition_strain=params['transition_strain'],
                damping_coefficient=params['damping_coefficient'],
            )

            result = ligament.simulate(time, length_profile, velocity_profile)
            ligament_results[lig_name] = result

            risk = self.predictor.compute_ligament_risk(
                lig_name, result['strain'], result['total_force'],
                params['estimated_failure_force'], dt
            )
            ligament_risks.append(risk)

        # 신체 부위별 위험도
        body_risks = self.predictor.compute_body_region_risks(
            muscle_risks, ligament_risks
        )

        # 보고서 생성
        report = self.predictor.generate_report(
            scenario['name'], muscle_risks, ligament_risks, body_risks
        )

        return {
            'scenario': scenario,
            'muscle_results': muscle_results,
            'ligament_results': ligament_results,
            'muscle_risks': muscle_risks,
            'ligament_risks': ligament_risks,
            'body_risks': body_risks,
            'report': report,
        }

    def run_scenario_from_data(self, scenario_data: dict) -> dict:
        """직접 구성된 시나리오 데이터로 시뮬레이션 실행 (Kinovea 입력용)"""
        time = scenario_data['time']
        dt = time[1] - time[0]

        muscle_results = {}
        muscle_risks = []

        for muscle_name, profiles in scenario_data['muscles'].items():
            params = MUSCLE_PARAMS[muscle_name]
            muscle = DeGrooteFregly2016Muscle(name=muscle_name, **params)

            fiber_length, fiber_velocity = profiles['fiber_length']
            result = muscle.simulate(time, profiles['excitation'],
                                     fiber_length, fiber_velocity)
            muscle_results[muscle_name] = result

            risk = self.predictor.compute_muscle_risk(
                muscle_name, result['muscle_stress_Pa'], dt
            )
            muscle_risks.append(risk)

        ligament_results = {}
        ligament_risks = []

        for lig_name, (length_profile, velocity_profile) in scenario_data['ligaments'].items():
            params = LIGAMENT_PARAMS[lig_name]
            ligament = Blankevoort1991Ligament(
                name=lig_name,
                slack_length=params['slack_length'],
                linear_stiffness=params['linear_stiffness'],
                transition_strain=params['transition_strain'],
                damping_coefficient=params['damping_coefficient'],
            )

            result = ligament.simulate(time, length_profile, velocity_profile)
            ligament_results[lig_name] = result

            risk = self.predictor.compute_ligament_risk(
                lig_name, result['strain'], result['total_force'],
                params['estimated_failure_force'], dt
            )
            ligament_risks.append(risk)

        body_risks = self.predictor.compute_body_region_risks(
            muscle_risks, ligament_risks
        )

        report = self.predictor.generate_report(
            scenario_data['name'], muscle_risks, ligament_risks, body_risks
        )

        return {
            'scenario': scenario_data,
            'muscle_results': muscle_results,
            'ligament_results': ligament_results,
            'muscle_risks': muscle_risks,
            'ligament_risks': ligament_risks,
            'body_risks': body_risks,
            'report': report,
        }

    def run_all_scenarios(self) -> dict:
        """전체 시나리오 시뮬레이션 실행"""
        all_results = {}
        for name, func in self.scenarios.items():
            print(f"  시뮬레이션 실행 중: {name}...")
            all_results[name] = self.run_scenario(func)
            print(f"  완료: {name}")
        return all_results


# ============================================================================
# 6. KinoveaInput - Kinovea CSV 데이터 입력 및 변환
# ============================================================================

class JointAngleToMuscle:
    """
    관절 각도 → 정규화 근육 섬유 길이/속도 변환.

    근골격 기하학 모델 기반. 각 근육은 특정 관절 각도에 의존하며,
    관절 각도 변화에 따라 근육-건 단위 길이가 변화합니다.

    변환 원리:
        정규화 섬유 길이 = f(관절각도)
        - 관절이 굴곡될수록 굴곡근은 짧아지고, 신전근은 길어짐
        - 선형 근사: norm_length = a * theta + b
          (a, b는 근육별 해부학적 모멘트 암에서 도출)

    Reference:
        - Manal & Buchanan (2004). Subject-specific estimates of tendon
          moment arms from musculoskeletal models.
        - Arnold et al. (2010). A model of the lower limb for analysis
          of human movement. Annals of Biomedical Engineering.
    """

    # 관절각도(도) → 정규화 섬유 길이 매핑 테이블
    # 형식: { 근육명: (관절명, theta_min, theta_max, nfl_at_min, nfl_at_max) }
    # theta_min/max: Kinovea에서 측정되는 관절 각도 범위 (도)
    # nfl_at_min/max: 해당 각도에서의 정규화 섬유 길이
    #
    # 예: 무릎 완전 신전(170도) → 대퇴사두근 nfl=1.1 (늘어남)
    #     무릎 완전 굴곡(60도)  → 대퇴사두근 nfl=0.7 (짧아짐)

    MAPPING = {
        '대퇴사두근': {
            'joint': 'Knee_Angle',
            'theta_range': (60.0, 180.0),    # 무릎 굴곡~신전
            'nfl_range': (0.7, 1.15),        # 굴곡 시 짧고, 신전 시 긺
            'description': '무릎 신전근 - 무릎각도에 비례하여 길어짐',
        },
        '대퇴이두근': {
            'joint': 'Knee_Angle',
            'theta_range': (60.0, 180.0),
            'nfl_range': (1.2, 0.8),         # 굴곡 시 길고, 신전 시 짧음 (길항근)
            'description': '무릎 굴곡근 - 무릎각도에 반비례',
        },
        '비복근': {
            'joint': 'Ankle_Angle',
            'theta_range': (60.0, 120.0),    # 발목 배굴~저굴
            'nfl_range': (1.15, 0.8),        # 배굴 시 늘어남, 저굴 시 짧아짐
            'description': '발목 저굴근 - 발목각도에 반비례',
        },
        '대둔근': {
            'joint': 'Hip_Angle',
            'theta_range': (90.0, 180.0),    # 고관절 굴곡~신전
            'nfl_range': (1.15, 0.85),       # 굴곡 시 늘어남
            'description': '고관절 신전근 - 고관절각도에 반비례',
        },
        '척추기립근': {
            'joint': 'Trunk_Angle',
            'theta_range': (120.0, 180.0),   # 몸통 굴곡~직립
            'nfl_range': (1.2, 0.9),         # 굴곡 시 늘어남
            'description': '척추 신전근 - 몸통각도에 반비례',
        },
        '삼각근': {
            'joint': 'Shoulder_Angle',
            'theta_range': (0.0, 180.0),     # 어깨 내림~올림
            'nfl_range': (0.8, 1.2),         # 거상 시 늘어남
            'description': '어깨 거상근 - 어깨각도에 비례',
        },
        '상완이두근': {
            'joint': 'Elbow_Angle',
            'theta_range': (40.0, 170.0),    # 팔꿈치 굴곡~신전
            'nfl_range': (0.7, 1.2),         # 신전 시 늘어남
            'description': '팔꿈치 굴곡근 - 팔꿈치각도에 비례',
        },
        '광배근': {
            'joint': 'Shoulder_Angle',
            'theta_range': (0.0, 180.0),
            'nfl_range': (1.15, 0.8),        # 어깨 거상 시 짧아짐 (길항근)
            'description': '어깨 내림근 - 어깨각도에 반비례',
        },
    }

    # 관절각도 → 인대 변형률 매핑
    # 인대 길이 = slack_length * (1 + strain)
    LIGAMENT_MAPPING = {
        '전방십자인대(ACL)': {
            'joint': 'Knee_Angle',
            'theta_range': (60.0, 180.0),
            'strain_range': (0.01, 0.05),    # 굴곡 시 낮음, 신전 시 높음
            'description': 'ACL은 무릎 신전 시 긴장 증가',
        },
        '후방십자인대(PCL)': {
            'joint': 'Knee_Angle',
            'theta_range': (60.0, 180.0),
            'strain_range': (0.05, 0.01),    # 굴곡 시 높음, 신전 시 낮음
            'description': 'PCL은 무릎 굴곡 시 긴장 증가',
        },
        '슬개건': {
            'joint': 'Knee_Angle',
            'theta_range': (60.0, 180.0),
            'strain_range': (0.06, 0.02),    # 깊은 굴곡 시 높은 변형률
            'description': '슬개건은 무릎 굴곡 시 긴장 증가',
        },
        '극상인대': {
            'joint': 'Trunk_Angle',
            'theta_range': (120.0, 180.0),
            'strain_range': (0.06, 0.01),    # 몸통 굴곡 시 긴장
            'description': '극상인대는 몸통 굴곡 시 긴장 증가',
        },
        '관절와상완인대': {
            'joint': 'Shoulder_Angle',
            'theta_range': (0.0, 180.0),
            'strain_range': (0.01, 0.06),    # 어깨 거상 시 긴장
            'description': '어깨 인대는 거상 시 긴장 증가',
        },
        '요추인대': {
            'joint': 'Trunk_Angle',
            'theta_range': (120.0, 180.0),
            'strain_range': (0.08, 0.01),    # 몸통 굴곡 시 높은 긴장
            'description': '요추인대는 몸통 굴곡 시 긴장 증가',
        },
    }

    @staticmethod
    def angle_to_norm_fiber_length(angle_deg: np.ndarray, mapping: dict) -> np.ndarray:
        """
        관절 각도(도) → 정규화 섬유 길이 변환 (선형 보간).

        Parameters
        ----------
        angle_deg : 관절 각도 시계열 (도)
        mapping : MAPPING 딕셔너리의 근육 항목

        Returns
        -------
        np.ndarray : 정규화 섬유 길이
        """
        theta_min, theta_max = mapping['theta_range']
        nfl_min, nfl_max = mapping['nfl_range']

        # 선형 보간 (범위 밖은 클램핑)
        t = np.clip((angle_deg - theta_min) / (theta_max - theta_min), 0.0, 1.0)
        nfl = nfl_min + t * (nfl_max - nfl_min)
        return nfl

    @staticmethod
    def angle_to_ligament_strain(angle_deg: np.ndarray, mapping: dict) -> np.ndarray:
        """관절 각도(도) → 인대 변형률 변환"""
        theta_min, theta_max = mapping['theta_range']
        strain_min, strain_max = mapping['strain_range']

        t = np.clip((angle_deg - theta_min) / (theta_max - theta_min), 0.0, 1.0)
        strain = strain_min + t * (strain_max - strain_min)
        return strain

    @staticmethod
    def estimate_excitation_from_angle_change(angle_deg: np.ndarray,
                                              dt: float,
                                              gain: float = 0.01,
                                              base: float = 0.1) -> np.ndarray:
        """
        관절 각도 변화율로부터 근육 흥분(excitation) 추정.

        원리: 관절이 빠르게 움직일수록 근육 활성화가 높음.
        excitation = base + gain * |d(angle)/dt|

        Parameters
        ----------
        angle_deg : 관절 각도 시계열 (도)
        dt : 시간 간격 (s)
        gain : 각속도→흥분 변환 계수
        base : 기저 흥분 수준

        Returns
        -------
        np.ndarray : 추정 흥분 프로파일 [0, 1]
        """
        angular_velocity = np.gradient(angle_deg, dt)
        excitation = base + gain * np.abs(angular_velocity)
        return np.clip(excitation, 0.01, 1.0)


class KinoveaInput:
    """
    Kinovea CSV 데이터를 읽어 시뮬레이션 입력으로 변환.

    Kinovea CSV 형식:
        Time(ms), Knee_Angle, Hip_Angle, Ankle_Angle, Shoulder_Angle, ...

    지원 관절 컬럼명:
        Knee_Angle    : 무릎 각도 (도)
        Hip_Angle     : 고관절 각도 (도)
        Ankle_Angle   : 발목 각도 (도)
        Shoulder_Angle: 어깨 각도 (도)
        Elbow_Angle   : 팔꿈치 각도 (도)
        Trunk_Angle   : 몸통 각도 (도)
    """

    SUPPORTED_JOINTS = [
        'Knee_Angle', 'Hip_Angle', 'Ankle_Angle',
        'Shoulder_Angle', 'Elbow_Angle', 'Trunk_Angle'
    ]

    @staticmethod
    def load_csv(filepath: str) -> dict:
        """
        Kinovea CSV 파일 로드.

        Parameters
        ----------
        filepath : CSV 파일 경로

        Returns
        -------
        dict : {'time': np.ndarray (초), 'joints': {관절명: np.ndarray (도)}}
        """
        import csv

        time_ms = []
        joints = {}

        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # 시간 컬럼 감지
            time_col = None
            for h in headers:
                h_clean = h.strip()
                if h_clean.lower().startswith('time') or h_clean.lower() == 't':
                    time_col = h
                    break

            if time_col is None:
                raise ValueError("CSV에서 시간 컬럼(Time)을 찾을 수 없습니다. "
                                 "첫 번째 컬럼에 'Time' 또는 'Time(ms)'를 포함시켜 주세요.")

            # 관절 컬럼 감지
            joint_cols = {}
            for h in headers:
                h_clean = h.strip()
                for supported in KinoveaInput.SUPPORTED_JOINTS:
                    if supported.lower() in h_clean.lower():
                        joint_cols[supported] = h
                        joints[supported] = []
                        break

            if not joint_cols:
                raise ValueError(
                    f"CSV에서 관절 각도 컬럼을 찾을 수 없습니다.\n"
                    f"지원 컬럼명: {KinoveaInput.SUPPORTED_JOINTS}\n"
                    f"발견된 컬럼: {headers}"
                )

            for row in reader:
                # 시간 파싱 (쉼표 소수점 처리)
                time_val = row[time_col].strip().replace(',', '.')
                time_ms.append(float(time_val))

                for joint_name, col_name in joint_cols.items():
                    val = row[col_name].strip().replace(',', '.')
                    joints[joint_name].append(float(val))

        # 밀리초 → 초 변환
        time_arr = np.array(time_ms)
        if time_arr.max() > 1000:  # 밀리초로 판단
            time_arr = time_arr / 1000.0

        for joint_name in joints:
            joints[joint_name] = np.array(joints[joint_name])

        print(f"  Kinovea 데이터 로드 완료:")
        print(f"    시간 범위: {time_arr[0]:.3f} ~ {time_arr[-1]:.3f} 초")
        print(f"    데이터 포인트: {len(time_arr)}")
        print(f"    관절: {list(joints.keys())}")

        return {'time': time_arr, 'joints': joints}

    @staticmethod
    def convert_to_scenario(kinovea_data: dict,
                            scenario_name: str = 'Kinovea 측정',
                            excitation_gain: float = 0.008,
                            excitation_base: float = 0.15) -> dict:
        """
        Kinovea 관절 각도 데이터 → 시뮬레이션 시나리오 변환.

        Parameters
        ----------
        kinovea_data : load_csv() 반환값
        scenario_name : 시나리오 이름
        excitation_gain : 각속도→흥분 변환 계수 (클수록 민감)
        excitation_base : 기저 흥분 수준

        Returns
        -------
        dict : SimulationEngine.run_scenario_from_data()에 전달할 시나리오
        """
        time = kinovea_data['time']
        joints = kinovea_data['joints']
        dt = np.mean(np.diff(time))

        muscles = {}
        ligaments = {}

        # 관절 각도 → 근육 프로파일 변환
        for muscle_name, mapping in JointAngleToMuscle.MAPPING.items():
            joint_name = mapping['joint']
            if joint_name not in joints:
                continue

            angle_deg = joints[joint_name]

            # 정규화 섬유 길이
            nfl = JointAngleToMuscle.angle_to_norm_fiber_length(angle_deg, mapping)

            # 정규화 섬유 속도 (길이의 시간 미분)
            nfl_velocity = np.gradient(nfl, time) / 10.0  # 최대 수축 속도로 정규화

            # 흥분 프로파일 추정
            excitation = JointAngleToMuscle.estimate_excitation_from_angle_change(
                angle_deg, dt, gain=excitation_gain, base=excitation_base
            )

            muscles[muscle_name] = {
                'excitation': excitation,
                'fiber_length': (nfl, nfl_velocity),
            }

        # 관절 각도 → 인대 프로파일 변환
        for lig_name, mapping in JointAngleToMuscle.LIGAMENT_MAPPING.items():
            joint_name = mapping['joint']
            if joint_name not in joints:
                continue

            angle_deg = joints[joint_name]
            params = LIGAMENT_PARAMS[lig_name]

            # 변형률 계산
            strain = JointAngleToMuscle.angle_to_ligament_strain(angle_deg, mapping)

            # 인대 길이 = slack_length * (1 + strain)
            length = params['slack_length'] * (1.0 + strain)
            velocity = np.gradient(length, time)

            ligaments[lig_name] = (length, velocity)

        if not muscles:
            raise ValueError(
                "CSV 데이터에서 매핑 가능한 관절 각도를 찾을 수 없습니다.\n"
                f"필요한 관절: {set(m['joint'] for m in JointAngleToMuscle.MAPPING.values())}\n"
                f"CSV 관절: {list(joints.keys())}"
            )

        print(f"\n  변환 완료:")
        print(f"    근육 {len(muscles)}개: {list(muscles.keys())}")
        print(f"    인대 {len(ligaments)}개: {list(ligaments.keys())}")

        return {
            'name': scenario_name,
            'time': time,
            'muscles': muscles,
            'ligaments': ligaments,
            'duration': time[-1] - time[0],
            'load_kg': 0,
        }
