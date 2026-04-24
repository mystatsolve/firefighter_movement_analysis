"""
biomech_engine.py - Hill-type 근육 모델 + 인대 모델 + 비동기 분석 엔진
======================================================================

oakd_realtime/biomech_engine.py를 거의 그대로 재사용합니다.
입력 형식은 angle_fusion.py의 get_window_data() 출력과 동일합니다.

구성요소:
    1. DeGrooteFregly2016Muscle: Hill-type 3요소 근육 모델
       - 수축 요소(CE) + 병렬 탄성(PE) + 직렬 탄성(SE=건)
       - 활성 힘-길이, 수동 힘-길이, 힘-속도 관계
       - 활성화 역학 ODE (De Groote 2016 매끄러운 전환)

    2. Blankevoort1991Ligament: 3구간 비선형 인대 모델
       - 이완/전이(toe)/선형 구간 + 점성 감쇠

    3. BiomechEngine: ThreadPoolExecutor 기반 비동기 분석 래퍼
       - submit/poll 패턴으로 메인 루프 차단 없이 시뮬레이션

참고문헌:
    - De Groote, F., et al. (2016). Annals of Biomedical Engineering, 44(10), 2922-2936.
    - Blankevoort, L., et al. (1991). J. Biomechanics, 24(11), 1019-1031.
    - Hill, A.V. (1938). Proceedings of the Royal Society B, 126(843), 136-195.
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Dict

from config import (
    ANALYSIS_MAX_WORKERS,
    MUSCLE_PARAMS, LIGAMENT_PARAMS,
    ANGLE_TO_MUSCLE, ANGLE_TO_LIGAMENT,
    LOAD_TASK_PROFILES,
    MUSCLE_STRESS_THRESHOLDS, LIGAMENT_STRAIN_THRESHOLDS,
    RISK_LEVELS, REGION_MAP,
    EXCITATION_GAIN, EXCITATION_BASE,
)


# =============================================================================
# DeGrooteFregly2016Muscle - Hill-type 근육 모델
# =============================================================================

class DeGrooteFregly2016Muscle:
    """
    De Groote & Fregly (2016) Hill-type 근육 모델.

    OpenSim 생체역학 소프트웨어의 표준 근육 모델을 구현합니다.
    근섬유의 역학적 특성(힘-길이, 힘-속도, 활성화 역학)을 수학적으로
    모델링하여 관절 움직임으로부터 근육 스트레스를 예측합니다.
    """

    # 능동적 힘-길이 곡선의 3개 가우시안 피팅 계수
    _b11, _b21, _b31, _b41 = 0.8150671134243542, 1.055033428970575, 0.162384573599574, 0.063303448465465
    _b12, _b22, _b32, _b42 = 0.433004984392647, 0.716775413397760, -0.029947116970696, 0.200356847296188
    _b13, _b23, _b33, _b43 = 0.1, 1.0, 0.353553390593274, 0.0

    # 수동적 힘-길이 곡선 계수
    _kPE = 4.0

    # 건 관련 계수
    _c1, _c2, _c3 = 0.200, 1.0, 0.200

    # 힘-속도 곡선 계수 (역쌍곡사인 기반)
    _d1, _d2, _d3, _d4 = -0.3211346127989808, -8.149, -0.374, 0.8825327733249912

    # 최소 정규화 근섬유 길이
    MIN_NORM_FIBER_LENGTH = 0.2

    def __init__(self, name, max_isometric_force, optimal_fiber_length,
                 tendon_slack_length, pennation_angle_at_optimal, pcsa, **kwargs):
        """
        Hill-type 근육 모델 인스턴스 생성.

        Parameters
        ----------
        name : str - 근육 이름
        max_isometric_force : float - 최대 등척성 근력 [N]
        optimal_fiber_length : float - 최적 근섬유 길이 [m]
        tendon_slack_length : float - 건 이완 길이 [m]
        pennation_angle_at_optimal : float - 최적 길이에서의 깃각도 [rad]
        pcsa : float - 생리학적 횡단면적 [m^2]
        """
        self.name = name
        self.max_isometric_force = max_isometric_force
        self.optimal_fiber_length = optimal_fiber_length
        self.tendon_slack_length = tendon_slack_length
        self.pennation_angle_at_optimal = pennation_angle_at_optimal
        self.pcsa = pcsa
        self.tau_a = 0.015  # 활성화 시간상수 15ms
        self.tau_d = 0.060  # 비활성화 시간상수 60ms
        self.fiber_damping = 0.01
        self._e0 = 0.6
        self._kT = np.log((1.0 + self._c3) / self._c1) / 0.049

    @staticmethod
    def _gaussian_like(x, b1, b2, b3, b4):
        """비대칭 가우시안 유사 함수."""
        denom = b3 + b4 * x
        if abs(denom) < 1e-12:
            denom = 1e-12
        return b1 * np.exp(-0.5 * ((x - b2) / denom) ** 2)

    def calc_active_fl(self, nfl):
        """능동적 힘-길이 관계 (3-가우시안 근사)."""
        nfl = np.atleast_1d(nfl)
        fl = np.zeros_like(nfl, dtype=float)
        for i, lm in enumerate(nfl):
            fl[i] = (self._gaussian_like(lm, self._b11, self._b21, self._b31, self._b41) +
                     self._gaussian_like(lm, self._b12, self._b22, self._b32, self._b42) +
                     self._gaussian_like(lm, self._b13, self._b23, self._b33, self._b43))
        return fl

    def calc_passive_fl(self, nfl):
        """수동적 힘-길이 관계 (지수 함수 기반)."""
        nfl = np.atleast_1d(nfl)
        num = np.exp(self._kPE * (nfl - 1.0) / self._e0)
        denom = np.exp(self._kPE)
        offset = np.exp(self._kPE * (self.MIN_NORM_FIBER_LENGTH - 1.0) / self._e0)
        return np.maximum((num - offset) / (denom - offset), 0.0)

    def calc_fv(self, nfv):
        """힘-속도 관계 (역쌍곡사인 기반)."""
        v = np.atleast_1d(nfv)
        inner = self._d2 * v + self._d3
        return np.clip(self._d1 * np.log(inner + np.sqrt(inner**2 + 1.0)) + self._d4, 0.0, 1.8)

    def activation_derivative(self, a, e):
        """활성화 역학 ODE (De Groote 2016 매끄러운 전환)."""
        f = 0.5 * np.tanh(10.0 * (e - a))
        z = 0.5 + 1.5 * a
        return ((f + 0.5) / (self.tau_a * z) + (-f + 0.5) * z / self.tau_d) * (e - a)

    def simulate(self, time_arr, excitation, nfl_profile, nfv_profile):
        """
        근육 시뮬레이션 전체 파이프라인.

        Returns
        -------
        dict - 'time', 'activation', 'active_fiber_force', 'passive_fiber_force',
               'total_fiber_force', 'force_along_tendon',
               'muscle_stress_Pa', 'muscle_stress_kPa'
        """
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


# =============================================================================
# Blankevoort1991Ligament - 인대 모델
# =============================================================================

class Blankevoort1991Ligament:
    """
    Blankevoort et al. (1991) 비선형 인대 모델.
    3구간 힘-변형률 관계 (이완/전이/선형) + 점성 감쇠.
    """

    def __init__(self, name, slack_length, linear_stiffness,
                 transition_strain=0.06, damping_coefficient=0.003, **kwargs):
        self.name = name
        self.slack_length = slack_length
        self.linear_stiffness = linear_stiffness
        self.transition_strain = transition_strain
        self.damping_coefficient = damping_coefficient

    def calc_force(self, strain, strain_rate):
        """인대의 순간 힘 계산 (3구간 + 감쇠)."""
        if strain <= 0:
            sf = 0.0
        elif strain < self.transition_strain:
            sf = 0.5 * self.linear_stiffness / self.transition_strain * strain**2
        else:
            sf = self.linear_stiffness * (strain - self.transition_strain / 2.0)

        df = 0.0
        if strain > 0 and strain_rate > 0:
            df = self.damping_coefficient * strain_rate * 0.5 * (1.0 + np.tanh(strain / 0.005))

        return max(0.0, sf + df), sf, df

    def simulate(self, time_arr, length_profile, velocity_profile):
        """인대 시뮬레이션 실행."""
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


# =============================================================================
# 헬퍼 함수 (시나리오 변환 + 시뮬레이션 실행)
# =============================================================================

def classify_risk(value, thresholds):
    """임계값 기반 위험도 분류. 역순 검색으로 최고 매칭 반환."""
    levels = list(thresholds.keys())
    vals = list(thresholds.values())
    for i in range(len(vals) - 1, -1, -1):
        if value >= vals[i]:
            return levels[i]
    return 'Normal'


def _convert_window_to_scenario(window_data, load_kg, body_mass_kg, task_type):
    """
    관절 각도 윈도우 → 시뮬레이션 시나리오 변환.

    근육: 관절각도→정규화 섬유길이→속도→흥분도
    인대: 관절각도→변형률→길이→속도
    """
    time_arr = window_data['time']
    joints = window_data['joints']
    dt = np.mean(np.diff(time_arr)) if len(time_arr) > 1 else 1.0 / 30.0

    task_profile = LOAD_TASK_PROFILES.get(task_type, None)
    has_load = load_kg > 0 and task_profile is not None

    # 근육 시나리오
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
        nfv = np.gradient(nfl, time_arr) / 10.0

        ang_vel = np.gradient(angles, dt)
        excitation = EXCITATION_BASE + EXCITATION_GAIN * np.abs(ang_vel)

        if has_load:
            load_factor = task_profile['muscle_load_factor'].get(muscle_name, 0.1)
            max_force = MUSCLE_PARAMS[muscle_name]['max_isometric_force']
            angle_rad = np.radians(angles)
            posture_factor = np.clip(np.sin(angle_rad / 2.0), 0.3, 1.0)
            load_force = load_kg * 9.81
            joint_torque = load_force * 0.05 * posture_factor * load_factor
            max_torque = max_force * 0.05
            load_exc = np.clip(joint_torque / max_torque, 0.0, 0.8)
            excitation = excitation + load_exc

        excitation = np.clip(excitation, 0.01, 1.0)
        muscles[muscle_name] = {'excitation': excitation, 'fiber_length': (nfl, nfv)}

    # 인대 시나리오
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

        if has_load:
            strain_factor = task_profile['ligament_strain_factor'].get(lig_name, 0.1)
            load_ratio = load_kg / body_mass_kg
            strain = strain + strain * load_ratio * strain_factor

        length = params['slack_length'] * (1.0 + strain)
        velocity = np.gradient(length, time_arr)
        ligaments[lig_name] = (length, velocity)

    return {'time': time_arr, 'muscles': muscles, 'ligaments': ligaments}


def _run_simulation(scenario):
    """전체 생체역학 시뮬레이션 실행 (백그라운드 스레드)."""
    time_arr = scenario['time']
    dt = np.mean(np.diff(time_arr)) if len(time_arr) > 1 else 1.0 / 30.0

    # 근육 시뮬레이션
    muscle_risks = []
    for muscle_name, profiles in scenario['muscles'].items():
        params = MUSCLE_PARAMS[muscle_name]
        ctor_params = {k: v for k, v in params.items() if k != 'name_kr'}
        muscle = DeGrooteFregly2016Muscle(name=muscle_name, **ctor_params)
        nfl, nfv = profiles['fiber_length']
        result = muscle.simulate(time_arr, profiles['excitation'], nfl, nfv)

        peak_stress = np.max(result['muscle_stress_Pa'])
        mean_stress = np.mean(result['muscle_stress_Pa'])
        risk = classify_risk(peak_stress, MUSCLE_STRESS_THRESHOLDS)
        fatigue = np.trapz(np.maximum(result['muscle_stress_Pa'] / 250e3, 0)**2, dx=dt)

        muscle_risks.append({
            'muscle_name': muscle_name,
            'display_name': MUSCLE_PARAMS[muscle_name].get('name_kr', muscle_name),
            'peak_stress_kPa': peak_stress / 1000,
            'mean_stress_kPa': mean_stress / 1000,
            'risk_level': risk,
            'fatigue_index': fatigue,
        })

    # 인대 시뮬레이션
    ligament_risks = []
    for lig_name, (length, velocity) in scenario['ligaments'].items():
        params = LIGAMENT_PARAMS[lig_name]
        ctor_params = {k: v for k, v in params.items()
                       if k not in ('name_kr', 'estimated_failure_force')}
        lig = Blankevoort1991Ligament(name=lig_name, **ctor_params)
        result = lig.simulate(time_arr, length, velocity)

        peak_strain = np.max(result['strain'])
        peak_force = np.max(result['total_force'])
        strain_risk = classify_risk(peak_strain, LIGAMENT_STRAIN_THRESHOLDS)

        ligament_risks.append({
            'ligament_name': lig_name,
            'display_name': LIGAMENT_PARAMS[lig_name].get('name_kr', lig_name),
            'peak_strain_pct': peak_strain * 100,
            'peak_force_N': peak_force,
            'strain_risk': strain_risk,
        })

    # 신체 부위별 위험도 집계
    region_scores = {}
    for mr in muscle_risks:
        r = REGION_MAP.get(mr['muscle_name'], 'Other')
        s = RISK_LEVELS.index(mr['risk_level']) if mr['risk_level'] in RISK_LEVELS else 0
        region_scores.setdefault(r, []).append(s)
    for lr in ligament_risks:
        r = REGION_MAP.get(lr['ligament_name'], 'Other')
        s = RISK_LEVELS.index(lr['strain_risk']) if lr['strain_risk'] in RISK_LEVELS else 0
        region_scores.setdefault(r, []).append(s)

    body_risks = {}
    for region, scores in region_scores.items():
        ms = max(scores)
        body_risks[region] = {'risk_level': RISK_LEVELS[ms], 'risk_score': ms}

    # 전체 종합 위험도
    all_scores = []
    for mr in muscle_risks:
        idx = RISK_LEVELS.index(mr['risk_level']) if mr['risk_level'] in RISK_LEVELS else 0
        all_scores.append(idx)
    for lr in ligament_risks:
        idx = RISK_LEVELS.index(lr['strain_risk']) if lr['strain_risk'] in RISK_LEVELS else 0
        all_scores.append(idx)
    overall_score = max(all_scores) if all_scores else 0
    overall_risk = RISK_LEVELS[overall_score]

    return {
        'muscle_risks': muscle_risks,
        'ligament_risks': ligament_risks,
        'body_risks': body_risks,
        'overall_risk': overall_risk,
        'overall_score': overall_score,
        'timestamp': time.time(),
    }


# =============================================================================
# BiomechEngine - 비동기 래퍼 클래스
# =============================================================================

class BiomechEngine:
    """
    비동기 생체역학 분석 엔진.

    ThreadPoolExecutor를 사용하여 시뮬레이션을 백그라운드에서 실행합니다.
    submit/poll 패턴으로 메인 루프를 차단하지 않습니다.
    """

    def __init__(self, load_kg=0.0, body_mass_kg=75.0, task_type='none'):
        self._executor = ThreadPoolExecutor(max_workers=ANALYSIS_MAX_WORKERS)
        self._current_future: Optional[Future] = None
        self._latest_result: Optional[dict] = None
        self._analysis_count = 0
        self._load_kg = load_kg
        self._body_mass_kg = body_mass_kg
        self._task_type = task_type

    def submit_analysis(self, window_data: dict):
        """비차단 분석 요청 제출. 이전 분석 진행 중이면 건너뜀."""
        if self._current_future is not None and not self._current_future.done():
            return
        scenario = _convert_window_to_scenario(
            window_data, self._load_kg, self._body_mass_kg, self._task_type
        )
        self._current_future = self._executor.submit(_run_simulation, scenario)

    def get_latest_result(self) -> Optional[dict]:
        """최신 완료된 분석 결과 폴링 (non-blocking)."""
        if self._current_future is not None and self._current_future.done():
            try:
                result = self._current_future.result()
                self._latest_result = result
                self._analysis_count += 1
            except Exception as e:
                print(f"[WARN] Analysis error: {e}")
            self._current_future = None
        return self._latest_result

    @property
    def analysis_count(self) -> int:
        return self._analysis_count

    @property
    def is_analyzing(self) -> bool:
        return self._current_future is not None and not self._current_future.done()

    def shutdown(self):
        """엔진 종료. ThreadPoolExecutor 해제."""
        self._executor.shutdown(wait=False)
