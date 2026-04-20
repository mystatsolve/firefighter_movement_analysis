"""
Async Biomechanics Engine - Hill-type muscle model + ligament model
with ThreadPoolExecutor for non-blocking background analysis.

=============================================================================
비동기 생체역학 엔진 (Asynchronous Biomechanics Engine)
=============================================================================

본 모듈은 실시간 자세 추정(pose estimation) 데이터를 기반으로 근골격계 부하를
시뮬레이션하여 작업자의 부상 위험도를 평가하는 핵심 엔진입니다.

[모듈 구성요소]
1. DeGrooteFregly2016Muscle: Hill-type 근육 모델
   - 수축 요소(CE) + 병렬 탄성 요소(PE) + 직렬 탄성 요소(SE)
   - 활성 힘-길이, 수동 힘-길이, 힘-속도 관계 모델링
   - 활성화 역학 ODE (1차 비선형 미분방정식)

2. Blankevoort1991Ligament: 인대 역학 모델
   - 3구간 비선형 힘-변형률 관계 (이완/전이/선형)
   - 점성 감쇠를 포함한 동적 하중 모델링

3. BiomechEngine: ThreadPoolExecutor 기반 비동기 분석 래퍼
   - submit/poll 패턴으로 메인 루프 차단 없이 시뮬레이션 실행

[비동기 실행 구조]
    메인 루프(30fps):
      frame → MediaPipe → angles → buffer → engine.submit_analysis()
                                                    ↓ (백그라운드)
                                            _convert_window_to_scenario()
                                            _run_simulation() (~100-200ms)
                                                    ↓
      engine.get_latest_result() → 디스플레이 표시

[참고 문헌]
- De Groote, F., et al. (2016). Annals of Biomedical Engineering, 44(10), 2922-2936.
- Blankevoort, L., et al. (1991). J. Biomechanics, 24(11), 1019-1031.
- Hill, A.V. (1938). Proceedings of the Royal Society B, 126(843), 136-195.
=============================================================================
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Dict

# ============================================================================
# [설정 임포트] config.py에서 시뮬레이션에 필요한 모든 상수/매핑 테이블을 가져옴
#
# ANALYSIS_MAX_WORKERS: ThreadPoolExecutor 최대 워커 수 (보통 1)
# MUSCLE_PARAMS: 각 근육의 해부학적/기계적 파라미터 (F_max, PCSA, 등)
# LIGAMENT_PARAMS: 각 인대의 기계적 파라미터 (slack_length, stiffness, 등)
# ANGLE_TO_MUSCLE: 관절 각도 → 근섬유 길이 매핑 룩업 테이블
# ANGLE_TO_LIGAMENT: 관절 각도 → 인대 변형률 매핑 룩업 테이블
# LOAD_TASK_PROFILES: 작업 유형별(들기/밀기/운반) 부하 배분 프로파일
# MUSCLE_STRESS_THRESHOLDS: 근육 스트레스 위험도 임계값 (Pa 단위)
# LIGAMENT_STRAIN_THRESHOLDS: 인대 변형률 위험도 임계값 (무차원)
# RISK_LEVELS: 위험도 레벨 리스트 ['Normal', 'Low', 'Medium', 'High', 'Critical']
# REGION_MAP: 근육/인대 → 신체 부위(어깨/허리/무릎 등) 매핑
# EXCITATION_GAIN: 각속도 → 근육 흥분도 변환 비례 상수
# EXCITATION_BASE: 기저 근육 흥분도 (안정 시 최소 활성화 수준)
# ============================================================================
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
#
# [Hill-type 근육 모델 이론]
#
# A.V. Hill(1938)이 제안한 근육의 기계적 모델은 3개 요소로 구성됩니다:
#
#   1) 수축 요소 (Contractile Element, CE):
#      - 액틴-미오신 교차결합(cross-bridge)에 의한 능동적 힘 생성
#      - F_CE = F_max × activation × f_l(length) × f_v(velocity)
#      - 신경 명령(excitation)에 의해 제어됨
#
#   2) 병렬 탄성 요소 (Parallel Elastic Element, PE):
#      - 근막(fascia), 결합조직, 타이틴(titin)의 수동 탄성
#      - F_PE = F_max × f_passive(length)
#      - 신경 활성화와 무관하게 작용 (순수 기계적)
#
#   3) 직렬 탄성 요소 (Series Elastic Element, SE = Tendon):
#      - 건(힘줄)의 탄성
#      - 본 구현에서는 pennation angle로 건 방향 힘 변환
#      - F_tendon = F_total × cos(pennation_angle)
#
# [전체 힘 방정식]
#   F_total = F_active + F_passive + F_damping
#           = F_max×a×f_l×f_v + F_max×f_PE + F_max×β×v_norm
#
# =============================================================================

class DeGrooteFregly2016Muscle:
    """
    De Groote & Fregly (2016) Hill-type 근육 모델.

    OpenSim 생체역학 소프트웨어의 표준 근육 모델을 구현합니다.
    근섬유의 역학적 특성(힘-길이, 힘-속도, 활성화 역학)을 수학적으로
    모델링하여 관절 움직임으로부터 근육 스트레스를 예측합니다.
    """

    # ========================================================================
    # [클래스 상수] 능동적 힘-길이 곡선의 3개 가우시안 피팅 계수
    # ========================================================================
    # De Groote et al. (2016)은 활성 힘-길이 곡선을 3개의 가우시안 유사 함수의
    # 합으로 근사합니다. 각 가우시안: g(x) = b1 × exp(-0.5×((x-b2)/(b3+b4×x))²)
    #
    # 물리적 의미:
    #   b1 = 진폭(amplitude): 해당 구간의 최대 힘 기여도
    #   b2 = 중심(center): 최대 힘이 발생하는 정규화된 근섬유 길이
    #   b3 = 폭(width): 곡선의 넓이 (근육이 힘을 생성할 수 있는 길이 범위)
    #   b4 = 비대칭 계수(skewness): x에 따라 폭이 변하는 정도
    #
    # --- 가우시안 1: 주 피크 (최적 길이 근처의 주요 힘 생성) ---
    # _b11=0.815: 전체 힘의 ~81.5% 기여하는 주 피크 진폭
    # _b21=1.055: 최적 길이(1.0)보다 약간 긴 위치에서 피크
    # _b31=0.162: 비교적 좁은 활성 범위
    # _b41=0.063: 약간의 비대칭 (길어지는 쪽으로 확장)
    _b11, _b21, _b31, _b41 = 0.8150671134243542, 1.055033428970575, 0.162384573599574, 0.063303448465465

    # --- 가우시안 2: 보조 피크 (짧은 길이에서의 힘 기여) ---
    # _b12=0.433: ~43.3% 기여하는 보조 피크 진폭
    # _b22=0.717: 최적 길이보다 짧은 쪽에 위치한 중심
    # _b32=-0.030: 음의 폭 → 특수 형태 (수학적 최적화 결과)
    # _b42=0.200: 큰 비대칭 계수
    _b12, _b22, _b32, _b42 = 0.433004984392647, 0.716775413397760, -0.029947116970696, 0.200356847296188

    # --- 가우시안 3: 꼬리 보정 (넓은 범위의 미세 조정) ---
    # _b13=0.1: 작은 진폭 (미세 보정)
    # _b23=1.0: 최적 길이 중심
    # _b33=0.354: 넓은 폭 (= sqrt(2)/4, 대칭)
    # _b43=0.0: 완전 대칭 가우시안
    _b13, _b23, _b33, _b43 = 0.1, 1.0, 0.353553390593274, 0.0

    # ========================================================================
    # [클래스 상수] 수동적 힘-길이 (Passive Force-Length) 곡선 계수
    # ========================================================================
    # _kPE = 4.0: 수동 탄성의 지수 형상 계수 (exponential shape factor)
    #   - 값이 클수록 근육 신장 시 수동 저항력이 급격히 증가
    #   - 근막(fascia)과 결합조직의 강성을 반영
    #   - 일반적으로 3~5 범위의 값 사용
    _kPE = 4.0

    # ========================================================================
    # [클래스 상수] 건(Tendon) 관련 계수 (참조용)
    # ========================================================================
    # _c1, _c2, _c3: 건 힘-변형 곡선 파라미터
    #   - 본 구현에서는 건을 강체(rigid)로 가정하지만,
    #     _kT 계산 시 간접적으로 사용됨
    _c1, _c2, _c3 = 0.200, 1.0, 0.200

    # ========================================================================
    # [클래스 상수] 힘-속도 (Force-Velocity) 곡선 계수
    # ========================================================================
    # 역쌍곡사인(arcsinh) 기반 매끄러운 힘-속도 모델:
    # f_v(v) = d1 × log(d2×v + d3 + sqrt((d2×v+d3)² + 1)) + d4
    #
    # 물리적 의미:
    #   _d1 = -0.321: 곡선 기울기 스케일 (음수 → 단축 시 힘 감소)
    #   _d2 = -8.149: 속도 감도 계수 (클수록 속도에 민감)
    #   _d3 = -0.374: 수평 오프셋 (등척성 조건에서의 보정)
    #   _d4 = 0.883: 수직 오프셋 (등척성에서 f_v ≈ 1.0 보장)
    #
    # 특성:
    #   - 단축성 수축(concentric, v<0): 속도↑ → 힘↓
    #   - 등척성 수축(isometric, v=0): f_v = 1.0
    #   - 신장성 수축(eccentric, v>0): f_v > 1.0 (최대 ~1.8배)
    _d1, _d2, _d3, _d4 = -0.3211346127989808, -8.149, -0.374, 0.8825327733249912

    # ========================================================================
    # [클래스 상수] 최소 정규화 근섬유 길이
    # ========================================================================
    # 수동 힘-길이 계산의 수치 안정성을 위한 하한값
    # 근육이 최적 길이의 20% 미만으로 단축되는 것은 생리학적으로 불가능
    MIN_NORM_FIBER_LENGTH = 0.2

    def __init__(self, name, max_isometric_force, optimal_fiber_length,
                 tendon_slack_length, pennation_angle_at_optimal, pcsa, **kwargs):
        """
        Hill-type 근육 모델 인스턴스 생성.

        각 근육의 고유한 해부학적/기계적 특성을 파라미터로 받습니다.

        Parameters
        ----------
        name : str
            근육 이름 (예: 'erector_spinae', 'rectus_femoris')
        max_isometric_force : float
            최대 등척성 근력 [N]. 근육이 최적 길이에서 최대 활성화 시
            발생 가능한 최대 힘. (예: 대퇴사두근 ≈ 6000N)
        optimal_fiber_length : float
            최적 근섬유 길이 [m]. 액틴-미오신 중첩이 최대인 길이.
        tendon_slack_length : float
            건 이완 길이 [m]. 건에 힘이 없는 자연 길이.
        pennation_angle_at_optimal : float
            최적 길이에서의 깃각도 [rad]. 근섬유-건 사이 각도.
            건 방향 힘 = 근섬유 힘 × cos(α)
        pcsa : float
            생리학적 횡단면적 [m²]. 스트레스 = 힘/PCSA [Pa]
        **kwargs : dict
            무시되는 추가 파라미터 (name_kr 등)
        """
        self.name = name
        self.max_isometric_force = max_isometric_force
        self.optimal_fiber_length = optimal_fiber_length
        self.tendon_slack_length = tendon_slack_length
        self.pennation_angle_at_optimal = pennation_angle_at_optimal
        self.pcsa = pcsa

        # [활성화 역학 시간 상수]
        # 신경 흥분(excitation) → 근육 활성화(activation) 전이의 시간 지연.
        # Ca²⁺ 방출/재흡수 과정을 반영합니다.
        # tau_a = 15ms: 활성화 (Ca²⁺ 방출 → 교차결합 형성) - 빠름
        # tau_d = 60ms: 비활성화 (Ca²⁺ 재흡수 → 교차결합 해제) - 느림
        # 비활성화가 4배 느린 이유: Ca²⁺ 펌프(SERCA)의 능동 수송 필요
        self.tau_a = 0.015
        self.tau_d = 0.060

        # [근섬유 감쇠 계수]
        # 근섬유 내부의 점성 감쇠 (수치 안정성 + 물리적 현실성)
        # F_damp = F_max × 0.01 × v_normalized
        self.fiber_damping = 0.01

        # [수동 힘-길이 곡선 형상 파라미터]
        # _e0 = 0.6: nfl = 1.6일 때 수동력이 F_max와 같아지는 변형률
        self._e0 = 0.6

        # [건 곡선 스케일링 파라미터]
        # 건의 비선형→선형 전환 강성 계수 (0.049 = 건 변형률 기준점)
        self._kT = np.log((1.0 + self._c3) / self._c1) / 0.049

    @staticmethod
    def _gaussian_like(x, b1, b2, b3, b4):
        """
        비대칭 가우시안 유사 함수.

        활성 힘-길이 곡선을 구성하는 기본 단위 함수입니다.
        일반 가우시안과 달리 분모(width)가 x에 의존하여 비대칭 형태 가능.

        수식: g(x) = b1 × exp(-0.5 × ((x - b2) / (b3 + b4×x))²)

        Parameters
        ----------
        x : float - 정규화된 근섬유 길이
        b1 : float - 진폭 (피크 높이)
        b2 : float - 중심 (피크 위치)
        b3 : float - 기본 폭
        b4 : float - 비대칭 계수 (x에 따른 폭 변화)

        Returns
        -------
        float - 가우시안 유사 함수값 (0~b1 범위)
        """
        denom = b3 + b4 * x
        # 수치 안정성: 분모가 0에 매우 가까우면 최소값으로 대체
        if abs(denom) < 1e-12:
            denom = 1e-12
        return b1 * np.exp(-0.5 * ((x - b2) / denom) ** 2)

    def calc_active_fl(self, nfl):
        """
        능동적 힘-길이 관계 계산 (Active Force-Length).

        [생리학적 배경]
        근섬유 내 액틴-미오신 필라멘트의 중첩도에 의해 결정됩니다.
        - nfl ≈ 1.0 (최적 길이): 교차결합 수 최대 → f_active ≈ 1.0
        - nfl < 0.5: 필라멘트 간섭 → f_active → 0
        - nfl > 1.5: 중첩 부족 → f_active → 0

        [3-가우시안 근사]
        3개의 가우시안 유사 함수의 합으로 실험 데이터를 피팅합니다.
        다항식 대비 장점: 항상 양수, 미분 가능(smooth), 최적화에 유리.

        Parameters
        ----------
        nfl : float or array - 정규화된 근섬유 길이 (1.0 = 최적)

        Returns
        -------
        np.ndarray - 능동적 힘 스케일링 팩터 (0~1 범위)
        """
        nfl = np.atleast_1d(nfl)
        fl = np.zeros_like(nfl, dtype=float)
        for i, lm in enumerate(nfl):
            # 3개 가우시안의 합: 주 피크 + 보조 피크 + 꼬리 보정
            fl[i] = (self._gaussian_like(lm, self._b11, self._b21, self._b31, self._b41) +
                     self._gaussian_like(lm, self._b12, self._b22, self._b32, self._b42) +
                     self._gaussian_like(lm, self._b13, self._b23, self._b33, self._b43))
        return fl

    def calc_passive_fl(self, nfl):
        """
        수동적 힘-길이 관계 계산 (Passive Force-Length).

        [생리학적 배경]
        근막(fascia), 결합조직, 타이틴(titin)의 탄성에 의한 수동 저항력.
        신경 활성화와 무관하게 순수 기계적으로 발생합니다.

        [수학적 모델 - 지수 함수 기반]
        f_PE(l) = (exp(kPE×(l-1)/e0) - offset) / (exp(kPE) - offset)

        특성:
        - l < 1.0: ≈ 0 (근육이 짧으면 수동력 없음)
        - l = 1.0: ≈ 0 (최적 길이에서 수동력 미미)
        - l > 1.0: 지수적 증가 (늘어날수록 급격히 증가)
        - l = 1.6 (= 1+e0): f_PE = 1.0 (F_max와 같은 수동력)

        [생리학적 의의]
        - 과도한 신장을 방지하는 보호 메커니즘
        - 스트레칭 시 "뻣뻣함"의 물리적 원인
        - 관절 가동범위(ROM) 끝에서의 저항력

        Parameters
        ----------
        nfl : float or array - 정규화된 근섬유 길이

        Returns
        -------
        np.ndarray - 수동적 힘 스케일링 팩터 (≥ 0)
        """
        nfl = np.atleast_1d(nfl)
        num = np.exp(self._kPE * (nfl - 1.0) / self._e0)
        denom = np.exp(self._kPE)
        # 최소 길이에서의 오프셋 제거 (nfl=MIN에서 0이 되도록 보정)
        offset = np.exp(self._kPE * (self.MIN_NORM_FIBER_LENGTH - 1.0) / self._e0)
        return np.maximum((num - offset) / (denom - offset), 0.0)

    def calc_fv(self, nfv):
        """
        힘-속도 관계 계산 (Force-Velocity).

        [생리학적 배경]
        - 단축성 수축(concentric): 빠를수록 교차결합 형성 시간 부족 → 힘↓
        - 등척성(isometric, v=0): 기준 힘 (f_v = 1.0)
        - 신장성 수축(eccentric): 교차결합 강제 파괴 시 추가 저항 → 힘↑ (최대 1.8배)

        [수학적 모델 - 역쌍곡사인(arcsinh) 기반]
        f_v(v) = d1×log(d2×v + d3 + sqrt((d2×v+d3)² + 1)) + d4
        이는 arcsinh의 정의와 동일: log(x + sqrt(x²+1)) = arcsinh(x)

        장점: v=0에서 연속이고 미분 가능 (수치 최적화에 유리)

        Parameters
        ----------
        nfv : float or array - 정규화된 근섬유 속도
            양수=신장(eccentric), 음수=단축(concentric), 0=등척성

        Returns
        -------
        np.ndarray - 힘-속도 스케일링 팩터 (0~1.8 클리핑)
        """
        v = np.atleast_1d(nfv)
        inner = self._d2 * v + self._d3
        return np.clip(self._d1 * np.log(inner + np.sqrt(inner**2 + 1.0)) + self._d4, 0.0, 1.8)

    def activation_derivative(self, a, e):
        """
        활성화 역학 ODE (Activation Dynamics) - 1차 비선형 미분방정식.

        신경 흥분(excitation, e) → 근육 활성화(activation, a)의
        시간적 전이를 모델링합니다.

        [수학적 모델 - De Groote 2016의 매끄러운(smooth) 전환 공식]
        da/dt = [(f+0.5)/(τ_a×z) + (-f+0.5)×z/τ_d] × (e - a)

        여기서:
          f = 0.5×tanh(10×(e-a))  ← 활성화/비활성화 전환 스위칭 함수
            * e > a (활성화 중): f → +0.5 → τ_a(빠른) 시간상수 적용
            * e < a (비활성화 중): f → -0.5 → τ_d(느린) 시간상수 적용
            * tanh의 10배 스케일: 전환 구간의 급격함 조절

          z = 0.5 + 1.5×a  ← 활성도 의존 스케일링
            * 이미 활성화된 근육: 비활성화가 느려짐 (z 증가)
            * 이완 상태의 근육: 활성화가 빨라짐 (z 감소)

          (e - a) 인수: 평형점(a=e)에서 da/dt = 0 보장

        [장점 vs 전통적 if-else 모델]
        - 전체 구간에서 C∞ smooth (무한 미분 가능)
        - 경사 기반 최적화(gradient-based optimization)에 적합
        - if-else 분기로 인한 수치적 불연속 없음

        [생리학적 의미]
        - τ_a = 15ms: 운동뉴런 → Ca²⁺ 방출 → 교차결합 형성
        - τ_d = 60ms: Ca²⁺ 재흡수(SERCA 펌프) → 교차결합 해제

        Parameters
        ----------
        a : float - 현재 활성화 수준 (0~1)
        e : float - 현재 신경 흥분 수준 (0~1)

        Returns
        -------
        float - da/dt [1/초]
        """
        # 스위칭 함수: e>a → +0.5(활성화 경로), e<a → -0.5(비활성화 경로)
        f = 0.5 * np.tanh(10.0 * (e - a))
        # 활성도 의존 스케일링 (0.5~2.0 범위)
        z = 0.5 + 1.5 * a
        # 활성화 항 + 비활성화 항, (e-a)로 평형점 보장
        return ((f + 0.5) / (self.tau_a * z) + (-f + 0.5) * z / self.tau_d) * (e - a)

    def simulate(self, time_arr, excitation, nfl_profile, nfv_profile):
        """
        근육 시뮬레이션 전체 파이프라인 실행.

        [시뮬레이션 파이프라인]
        Step 1: 활성화 역학 ODE 적분 (excitation → activation)
          - solve_ivp(RK45): 적응형 Runge-Kutta 4(5)차
          - 초기조건: a(0)=0.05 (기저 근긴장도)
          - max_step=0.01s: τ_a(15ms)보다 작은 시간 간격 보장

        Step 2: 힘 구성요소 계산
          - F_active = F_max × a × f_l × f_v (능동적 수축력)
          - F_passive = F_max × f_PE (수동적 탄성력)
          - F_damp = F_max × β × v (점성 감쇠력)

        Step 3: 합력 및 스트레스
          - F_total = F_active + F_passive + F_damp
          - F_tendon = F_total × cos(α) (건 방향 투영)
          - σ = F_total / PCSA [Pa]

        Parameters
        ----------
        time_arr : np.ndarray - 시간 배열 [s]
        excitation : np.ndarray - 신경 흥분 프로파일 (0~1)
        nfl_profile : np.ndarray - 정규화 근섬유 길이 프로파일
        nfv_profile : np.ndarray - 정규화 근섬유 속도 프로파일

        Returns
        -------
        dict - 시뮬레이션 결과:
            'time', 'activation', 'active_fiber_force', 'passive_fiber_force',
            'total_fiber_force', 'force_along_tendon',
            'muscle_stress_Pa', 'muscle_stress_kPa'
        """
        n = len(time_arr)

        # --- Step 1: 활성화 역학 ODE 적분 ---
        def ode(t, y):
            # 현재 시각 t에서의 excitation을 선형 보간으로 획득
            exc = np.interp(t, time_arr, excitation)
            return [self.activation_derivative(y[0], exc)]

        # RK45: 4차 정확도, 5차 오차 추정의 적응형 시간 간격 적분기
        # 초기 활성화 0.05: 완전 이완에서도 미세한 근긴장도(tonus) 존재
        sol = solve_ivp(ode, [time_arr[0], time_arr[-1]], [0.05],
                        t_eval=time_arr, method='RK45', max_step=0.01)
        # [0.01, 1.0] 클리핑: 0은 수치 문제, 1 초과는 비생리적
        activations = np.clip(sol.y[0], 0.01, 1.0)

        # --- Step 2: 힘 구성요소 계산 ---
        # 깃각도 코사인: 근섬유→건 방향 힘 투영 계수
        cos_penn = np.cos(self.pennation_angle_at_optimal)
        # 각 시점의 힘-길이, 힘-속도 관계 값
        f_act_fl = self.calc_active_fl(nfl_profile)
        f_pass_fl = self.calc_passive_fl(nfl_profile)
        f_fv = self.calc_fv(nfv_profile)

        # 능동적 힘 = F_max × activation × f_l × f_v
        active = self.max_isometric_force * activations * f_act_fl * f_fv
        # 수동적 힘 = F_max × f_PE (활성화 무관)
        passive = self.max_isometric_force * f_pass_fl
        # 감쇠력 = F_max × β × v_normalized
        damp = self.max_isometric_force * self.fiber_damping * nfv_profile

        # --- Step 3: 합력 및 스트레스 ---
        total = active + passive + damp
        # 근육 스트레스 [Pa] = 힘/면적 (250kPa 이상 → 피로 위험)
        stress = total / self.pcsa

        return {
            'time': time_arr, 'activation': activations,
            'active_fiber_force': active, 'passive_fiber_force': passive,
            'total_fiber_force': total, 'force_along_tendon': total * cos_penn,
            'muscle_stress_Pa': stress, 'muscle_stress_kPa': stress / 1000.0,
        }


# =============================================================================
# Blankevoort1991Ligament - 인대(Ligament) 모델
# =============================================================================
#
# [인대 역학 모델 이론]
#
# 인대는 관절 안정성을 제공하는 수동적 결합조직(콜라겐 섬유)입니다.
# 힘-변형률 관계는 3개 구간으로 나뉩니다:
#
#   구간 1 - 이완 (Slack): strain ≤ 0
#     - 인대가 자연길이 이하, 콜라겐이 물결(crimp) 상태
#     - 힘 = 0 (인대는 압축력 전달 불가, 끈과 유사)
#
#   구간 2 - 전이 (Toe Region): 0 < strain < transition_strain
#     - 콜라겐 섬유가 점진적으로 펴짐 (uncrimping)
#     - 2차 포물선: F = 0.5 × k/ε_t × ε²
#     - 부드러운 저항감 구간
#
#   구간 3 - 선형 (Linear Region): strain ≥ transition_strain
#     - 콜라겐 완전히 펴짐, 직접 하중 전달
#     - 선형: F = k × (ε - ε_t/2)
#     - 이 구간 넘어서면 미세파열 → 완전파열 위험
#
# [감쇠력 (Damping)]
#   F_damp = c × ε_dot × 0.5 × (1 + tanh(ε/0.005))
#   - 인대가 당겨진 상태(strain>0)에서 더 늘어날 때(strain_rate>0)만 작용
#   - tanh로 매끄러운 on/off 전환
#
# =============================================================================

class Blankevoort1991Ligament:
    """
    Blankevoort et al. (1991) 비선형 인대 모델.

    인대의 3구간 힘-변형률 관계와 점성 감쇠를 모델링합니다.
    """

    def __init__(self, name, slack_length, linear_stiffness,
                 transition_strain=0.06, damping_coefficient=0.003, **kwargs):
        """
        인대 모델 초기화.

        Parameters
        ----------
        name : str
            인대 이름 (예: 'ACL', 'PCL', 'MCL')
        slack_length : float
            이완 길이 [m] - 인대에 힘이 없는 자연 상태 길이.
            이 길이를 기준으로 변형률 계산: ε = (L-L0)/L0
            이보다 짧으면 인대는 이완 (F=0)
        linear_stiffness : float
            선형 강성 [N/단위변형률] - 전이구간 이후의 기울기.
            값이 클수록 "딱딱한" 인대.
            예: ACL ≈ 5000, PCL ≈ 9000, MCL ≈ 3000
        transition_strain : float (기본 0.06 = 6%)
            전이 변형률 - toe→linear 전환 경계.
            콜라겐 crimp가 완전히 펴지는 지점.
            작을수록 인대가 빨리 팽팽해짐.
        damping_coefficient : float (기본 0.003)
            감쇠 계수 - 빠른 하중에서의 에너지 흡수.
            수치 안정성에도 기여 (고주파 진동 억제).
        **kwargs : dict
            무시되는 추가 파라미터
        """
        self.name = name
        self.slack_length = slack_length
        self.linear_stiffness = linear_stiffness
        self.transition_strain = transition_strain
        self.damping_coefficient = damping_coefficient

    def calc_force(self, strain, strain_rate):
        """
        인대의 순간 힘 계산 (3구간 모델 + 감쇠).

        [스프링 힘 - 3구간 비선형 모델]
        구간1 (Slack, ε≤0): F=0 (이완, 힘 전달 없음)
        구간2 (Toe, 0<ε<ε_t): F = k/(2ε_t) × ε² (포물선)
        구간3 (Linear, ε≥ε_t): F = k×(ε - ε_t/2) (직선)

        [감쇠 힘 - 조건부 점성 감쇠]
        strain>0 AND strain_rate>0일 때만:
        F_damp = c × ε_dot × 0.5 × (1 + tanh(ε/0.005))
        tanh 전환: ε>>0.005이면 완전 활성, ε≈0이면 비활성

        Parameters
        ----------
        strain : float - 변형률 (양수=늘어남, 음수=이완)
        strain_rate : float - 변형률 속도 [1/s] (양수=늘어나는 중)

        Returns
        -------
        tuple (total_force, spring_force, damping_force) [N]
        """
        # --- 탄성(스프링) 힘: 3구간 모델 ---
        if strain <= 0:
            # 구간 1: 이완 - 인대 느슨, 힘 없음
            sf = 0.0
        elif strain < self.transition_strain:
            # 구간 2: Toe - 포물선 (콜라겐 crimp 펴지는 과정)
            # ε=ε_t에서 접선기울기가 k와 일치하도록 설계
            sf = 0.5 * self.linear_stiffness / self.transition_strain * strain**2
        else:
            # 구간 3: 선형 - 후크 법칙 유사
            # ε_t/2를 빼서 구간2 끝과 C0 연속 보장
            sf = self.linear_stiffness * (strain - self.transition_strain / 2.0)

        # --- 감쇠 힘: 조건부 점성 ---
        df = 0.0
        if strain > 0 and strain_rate > 0:
            # 팽팽(strain>0) + 더 늘어나는 중(rate>0)일 때만
            # tanh: strain>>0.005이면 계수→1, strain≈0이면 계수→0
            df = self.damping_coefficient * strain_rate * 0.5 * (1.0 + np.tanh(strain / 0.005))

        # 인대는 tension만 전달 → 음수 클리핑
        return max(0.0, sf + df), sf, df

    def simulate(self, time_arr, length_profile, velocity_profile):
        """
        인대 시뮬레이션 실행.

        각 시간점에서:
        1) 변형률 계산: ε = L(t)/L0 - 1
        2) 변형률 속도: ε_dot = V(t)/L0
        3) calc_force()로 힘 계산

        Parameters
        ----------
        time_arr : np.ndarray - 시간 배열 [s]
        length_profile : np.ndarray - 인대 길이 [m]
        velocity_profile : np.ndarray - 길이 변화 속도 [m/s]

        Returns
        -------
        dict - 'time', 'strain', 'strain_percent',
               'spring_force', 'damping_force', 'total_force'
        """
        n = len(time_arr)
        results = {
            'time': time_arr, 'strain': np.zeros(n), 'strain_percent': np.zeros(n),
            'spring_force': np.zeros(n), 'damping_force': np.zeros(n),
            'total_force': np.zeros(n),
        }
        for i in range(n):
            # 변형률 = (현재길이/이완길이) - 1
            strain = length_profile[i] / self.slack_length - 1.0
            # 변형률 속도 = 길이변화속도 / 이완길이
            strain_rate = velocity_profile[i] / self.slack_length
            total, spring, damp = self.calc_force(strain, strain_rate)
            results['strain'][i] = strain
            results['strain_percent'][i] = strain * 100.0
            results['spring_force'][i] = spring
            results['damping_force'][i] = damp
            results['total_force'][i] = total
        return results


# =============================================================================
# 헬퍼 함수 (Helper Functions)
# =============================================================================

def classify_risk(value, thresholds):
    """
    임계값 기반 위험도 분류.

    [알고리즘]
    임계값 딕셔너리를 역순으로 순회하며 (가장 높은 위험도부터),
    측정값이 해당 임계값 이상인 첫 번째 수준을 반환합니다.

    예시:
      thresholds = {'Normal':0, 'Low':100k, 'Medium':250k, 'High':400k}
      value = 300000 → 'Medium' (300k ≥ 250k, < 400k)

    역순 검색 이유: 가장 높은 것부터 확인하면 첫 매칭이 정답.

    Parameters
    ----------
    value : float - 분류할 측정값 (스트레스 Pa 또는 변형률)
    thresholds : dict - {레벨명: 하한 임계값} (오름차순)

    Returns
    -------
    str - 위험도 수준. 아무것도 해당 안 되면 'Normal'.
    """
    levels = list(thresholds.keys())
    vals = list(thresholds.values())
    for i in range(len(vals) - 1, -1, -1):
        if value >= vals[i]:
            return levels[i]
    return 'Normal'


def _convert_window_to_scenario(window_data, load_kg, body_mass_kg, task_type):
    """
    관절 각도 윈도우 데이터 → 시뮬레이션 시나리오 변환.

    [변환 파이프라인]
    근육:
      1. 관절 각도 → 정규화 근섬유 길이 (nfl) [선형 매핑]
      2. nfl 시간미분 → 정규화 근섬유 속도 (nfv)
      3. 관절 각속도 → 신경 흥분도 (excitation)
      4. [부하 있으면] 외부 하중 기여분 추가

    인대:
      1. 관절 각도 → 변형률 (strain) [선형 매핑]
      2. [부하 있으면] 하중에 의한 추가 변형 가산
      3. 변형률 → 인대 길이 = slack_length × (1+strain)
      4. 길이 시간미분 → 속도

    [핵심 가정]
    - 관절 각도↔근섬유 길이: 1차 선형 근사
    - 흥분도 ∝ |각속도| (빠른 움직임 = 높은 근활성)
    - 외부 부하 → 관절 토크 → 추가 흥분 기여

    Parameters
    ----------
    window_data : dict - {'time': array, 'joints': {name: angles}}
    load_kg : float - 외부 하중 [kg]
    body_mass_kg : float - 작업자 체중 [kg]
    task_type : str - 작업 유형 ('lifting', 'carrying', 등)

    Returns
    -------
    dict - {'time': array, 'muscles': {...}, 'ligaments': {...}}
    """
    time_arr = window_data['time']
    joints = window_data['joints']
    # 평균 프레임 간격 (dt): 각속도 계산에 사용
    dt = np.mean(np.diff(time_arr)) if len(time_arr) > 1 else 1.0 / 30.0

    # 작업 유형별 부하 프로파일 조회
    task_profile = LOAD_TASK_PROFILES.get(task_type, None)
    has_load = load_kg > 0 and task_profile is not None

    # === 근육 시나리오 생성 ===
    muscles = {}
    for muscle_name, mapping in ANGLE_TO_MUSCLE.items():
        joint = mapping['joint']
        if joint not in joints:
            continue
        angles = joints[joint]  # 관절 각도 시계열 [도]
        theta_min, theta_max = mapping['theta_range']
        nfl_min, nfl_max = mapping['nfl_range']

        # [Step 1] 관절 각도 → 정규화 근섬유 길이 (선형 보간)
        # θ_min → nfl_min, θ_max → nfl_max
        t = np.clip((angles - theta_min) / (theta_max - theta_min), 0, 1)
        nfl = nfl_min + t * (nfl_max - nfl_min)

        # [Step 2] 근섬유 속도 = d(nfl)/dt / 10 (스케일링)
        nfv = np.gradient(nfl, time_arr) / 10.0

        # [Step 3] 관절 각속도 → 흥분도
        # 빠른 동작 = 높은 각속도 = 더 많은 근활성 필요
        ang_vel = np.gradient(angles, dt)
        excitation = EXCITATION_BASE + EXCITATION_GAIN * np.abs(ang_vel)

        # [Step 4] 외부 하중에 의한 추가 흥분
        if has_load:
            # load_factor: 이 근육이 부하를 지지하는 비율
            load_factor = task_profile['muscle_load_factor'].get(muscle_name, 0.1)
            max_force = MUSCLE_PARAMS[muscle_name]['max_isometric_force']
            angle_rad = np.radians(angles)
            # sin(θ/2): 각도 클수록 모멘트 암 증가 (토크 효과)
            posture_factor = np.clip(np.sin(angle_rad / 2.0), 0.3, 1.0)
            load_force = load_kg * 9.81  # 중력 [N]
            # 0.05m = 대략적 모멘트 암
            joint_torque = load_force * 0.05 * posture_factor * load_factor
            max_torque = max_force * 0.05
            # 추가 흥분 = 필요토크/최대토크 (0~0.8 제한)
            load_exc = np.clip(joint_torque / max_torque, 0.0, 0.8)
            excitation = excitation + load_exc

        # 최종 클리핑: [0.01, 1.0]
        excitation = np.clip(excitation, 0.01, 1.0)
        muscles[muscle_name] = {'excitation': excitation, 'fiber_length': (nfl, nfv)}

    # === 인대 시나리오 생성 ===
    ligaments = {}
    for lig_name, mapping in ANGLE_TO_LIGAMENT.items():
        joint = mapping['joint']
        if joint not in joints:
            continue
        angles = joints[joint]
        theta_min, theta_max = mapping['theta_range']
        strain_min, strain_max = mapping['strain_range']
        params = LIGAMENT_PARAMS[lig_name]

        # [Step 1] 관절 각도 → 인대 변형률 (선형 보간)
        t = np.clip((angles - theta_min) / (theta_max - theta_min), 0, 1)
        strain = strain_min + t * (strain_max - strain_min)

        # [Step 2] 외부 하중에 의한 추가 변형
        if has_load:
            strain_factor = task_profile['ligament_strain_factor'].get(lig_name, 0.1)
            load_ratio = load_kg / body_mass_kg  # 체중 대비 하중 비율
            strain = strain + strain * load_ratio * strain_factor

        # [Step 3] 변형률 → 길이/속도
        length = params['slack_length'] * (1.0 + strain)
        velocity = np.gradient(length, time_arr)
        ligaments[lig_name] = (length, velocity)

    return {'time': time_arr, 'muscles': muscles, 'ligaments': ligaments}


def _run_simulation(scenario):
    """
    전체 생체역학 시뮬레이션 실행 (백그라운드 스레드에서 호출).

    [실행 파이프라인]
    1) 각 근육: DeGrooteFregly2016Muscle.simulate()
       → 피크/평균 스트레스, 위험 등급, 피로도 지수

    2) 각 인대: Blankevoort1991Ligament.simulate()
       → 피크 변형률/힘, 위험 등급

    3) 신체 부위별 위험도 집계 (REGION_MAP 기반)
       → 각 부위의 최대 위험 점수

    4) 전체 종합 위험도 결정
       → "가장 약한 고리" 원칙: max(모든 점수)

    [피로도 지수 (Fatigue Index)]
    fatigue = ∫ max(σ/250kPa, 0)² dt
    - 기준 스트레스(250kPa) 초과 부분의 제곱을 시간 적분
    - 높은 스트레스에 가중치 부여 (2제곱)
    - 값이 클수록 누적 피로 위험 높음

    Parameters
    ----------
    scenario : dict - _convert_window_to_scenario()의 반환값

    Returns
    -------
    dict - {muscle_risks, ligament_risks, body_risks,
            overall_risk, overall_score, timestamp}
    """
    time_arr = scenario['time']
    dt = np.mean(np.diff(time_arr)) if len(time_arr) > 1 else 1.0 / 30.0

    # === 근육 시뮬레이션 및 위험도 평가 ===
    muscle_risks = []
    for muscle_name, profiles in scenario['muscles'].items():
        params = MUSCLE_PARAMS[muscle_name]
        # name_kr은 표시용이므로 생성자에서 제외
        ctor_params = {k: v for k, v in params.items() if k != 'name_kr'}
        muscle = DeGrooteFregly2016Muscle(name=muscle_name, **ctor_params)
        nfl, nfv = profiles['fiber_length']
        result = muscle.simulate(time_arr, profiles['excitation'], nfl, nfv)

        # 피크/평균 스트레스 추출
        peak_stress = np.max(result['muscle_stress_Pa'])
        mean_stress = np.mean(result['muscle_stress_Pa'])
        # 위험 등급 분류
        risk = classify_risk(peak_stress, MUSCLE_STRESS_THRESHOLDS)
        # 피로도 지수: (σ/250kPa)²의 시간 적분
        fatigue = np.trapz(np.maximum(result['muscle_stress_Pa'] / 250e3, 0)**2, dx=dt)

        muscle_risks.append({
            'muscle_name': muscle_name,
            'display_name': MUSCLE_PARAMS[muscle_name].get('name_kr', muscle_name),
            'peak_stress_kPa': peak_stress / 1000,
            'mean_stress_kPa': mean_stress / 1000,
            'risk_level': risk,
            'fatigue_index': fatigue,
        })

    # === 인대 시뮬레이션 및 위험도 평가 ===
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

    # === 신체 부위별 위험도 집계 ===
    # REGION_MAP으로 근육/인대를 부위에 매핑, 각 부위의 최대 점수 산출
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

    # === 전체 종합 위험도 ===
    # 모든 조직 중 최대 위험 점수 = 전체 위험도 ("가장 약한 고리" 원칙)
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
#
# [설계 패턴: Submit/Poll (비동기 논블로킹)]
#
# 실시간 시스템에서 무거운 계산(근육/인대 시뮬레이션, ~100-200ms)을
# 메인 스레드(카메라 30fps + GUI)를 차단하지 않고 수행해야 합니다.
#
# 해결: ThreadPoolExecutor + Future 객체
#   submit_analysis(): 작업 제출 → 즉시 반환 (O(1))
#   get_latest_result(): 완료 여부만 확인 → 대기하지 않음
#
# [max_workers=1인 이유]
# 1. Python GIL: 멀티스레드 CPU 작업은 진정한 병렬이 아님
# 2. 시간적 일관성: 한 번에 하나의 분석만 → 결과 순서 보장
# 3. 중복 방지: 이전 분석 진행 중이면 새 제출 건너뜀 (최신 데이터 우선)
# 4. 프레임률 여유: 30fps에서 3-4프레임마다 한 번 분석이면 충분
#
# [사용 패턴]
#   while running:
#       frame = camera.get()
#       angles = detect(frame)
#       engine.submit_analysis(window)  # 비차단 제출
#       result = engine.get_latest_result()  # 이전 결과 폴링
#       display(frame, result)
#
# =============================================================================

class BiomechEngine:
    """
    비동기 생체역학 분석 엔진.

    ThreadPoolExecutor를 사용하여 시뮬레이션을 백그라운드에서 실행합니다.
    메인 루프는 submit으로 제출, get_latest_result로 결과를 폴링합니다.
    """

    def __init__(self, load_kg=0.0, body_mass_kg=75.0, task_type='none'):
        """
        Parameters
        ----------
        load_kg : float - 외부 하중 [kg] (0이면 무부하)
        body_mass_kg : float - 작업자 체중 [kg] (부하비율 계산용)
        task_type : str - 작업 유형 ('lifting'/'carrying'/'pushing'/'none')
        """
        # 백그라운드 스레드풀 (max_workers=1: 위의 설계 근거 참조)
        self._executor = ThreadPoolExecutor(max_workers=ANALYSIS_MAX_WORKERS)
        # 현재 진행 중인 분석의 Future (None이면 대기 중인 작업 없음)
        self._current_future: Optional[Future] = None
        # 가장 최근 완료된 결과 (처음엔 None)
        self._latest_result: Optional[dict] = None
        # 누적 분석 완료 횟수 (모니터링용)
        self._analysis_count = 0
        # 시뮬레이션 파라미터
        self._load_kg = load_kg
        self._body_mass_kg = body_mass_kg
        self._task_type = task_type

    def submit_analysis(self, window_data: dict):
        """
        비차단(non-blocking) 분석 요청 제출.

        이전 분석이 진행 중이면 이번 요청을 건너뜁니다.
        이렇게 하면 분석 큐가 쌓이지 않고 항상 최신 데이터를 분석합니다.

        Parameters
        ----------
        window_data : dict - {'time': array, 'joints': {name: angles}}
        """
        # 이전 분석 진행 중 → 드롭 (큐 쌓임 방지)
        if self._current_future is not None and not self._current_future.done():
            return  # skip, previous analysis still running

        # 윈도우 → 시나리오 변환 (빠름, 메인 스레드에서 OK)
        scenario = _convert_window_to_scenario(
            window_data, self._load_kg, self._body_mass_kg, self._task_type
        )
        # 무거운 시뮬레이션을 백그라운드에 제출 (즉시 Future 반환)
        self._current_future = self._executor.submit(_run_simulation, scenario)

    def get_latest_result(self) -> Optional[dict]:
        """
        최신 완료된 분석 결과를 폴링 (non-blocking).

        Future가 done()이면 결과 추출, 아니면 이전 결과 반환.
        에러 시 경고 출력 후 이전 결과 유지 (서비스 연속성 보장).

        Returns
        -------
        Optional[dict] - 최신 분석 결과 또는 None
        """
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
        """누적 분석 완료 횟수."""
        return self._analysis_count

    @property
    def is_analyzing(self) -> bool:
        """현재 백그라운드 분석 진행 중 여부."""
        return self._current_future is not None and not self._current_future.done()

    def shutdown(self):
        """
        엔진 종료. ThreadPoolExecutor를 해제합니다.
        wait=False: 진행 중인 작업 완료를 기다리지 않고 즉시 종료.
        프로그램 종료 시 반드시 호출하여 스레드 자원 해제.
        """
        self._executor.shutdown(wait=False)
