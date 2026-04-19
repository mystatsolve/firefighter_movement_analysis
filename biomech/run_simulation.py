"""
소방관 근골격계 부상 예측 시뮬레이션 실행 스크립트
- 시나리오별 시뮬레이션 실행
- 시각화 (근육 힘, 스트레스, 인대 장력, 부상 위험도 히트맵)
- CSV 결과 출력
- 한국어 보고서 출력

실행: python run_simulation.py
의존성: pip install numpy matplotlib scipy
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지 저장
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정
for font_name in ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']:
    try:
        rcParams['font.family'] = font_name
        break
    except:
        continue
rcParams['axes.unicode_minus'] = False

from firefighter_biomech import (
    SimulationEngine, DeGrooteFregly2016Muscle, Blankevoort1991Ligament,
    InjuryPredictor, MUSCLE_PARAMS, LIGAMENT_PARAMS
)

# 출력 디렉토리
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 시각화 함수들
# ============================================================================

def plot_muscle_forces(results: dict, scenario_name: str, save_path: str):
    """근육 힘 시계열 그래프"""
    muscle_results = results['muscle_results']
    n_muscles = len(muscle_results)

    fig, axes = plt.subplots(n_muscles, 3, figsize=(18, 4 * n_muscles))
    if n_muscles == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'근육 힘 분석 - {scenario_name}', fontsize=16, fontweight='bold')

    for i, (name, data) in enumerate(muscle_results.items()):
        t = data['time']
        # 처음 5초만 표시 (디테일 확인용)
        mask = t <= 5.0

        # 활성/수동 섬유력
        axes[i, 0].plot(t[mask], data['active_fiber_force'][mask], 'r-', label='활성 섬유력', linewidth=1.5)
        axes[i, 0].plot(t[mask], data['passive_fiber_force'][mask], 'b-', label='수동 섬유력', linewidth=1.5)
        axes[i, 0].set_ylabel('힘 (N)')
        axes[i, 0].set_title(f'{name} - 섬유력 성분')
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)

        # 총 섬유력
        axes[i, 1].plot(t[mask], data['total_fiber_force'][mask], 'k-', linewidth=1.5)
        axes[i, 1].fill_between(t[mask], 0, data['total_fiber_force'][mask], alpha=0.2)
        axes[i, 1].set_ylabel('힘 (N)')
        axes[i, 1].set_title(f'{name} - 총 섬유력')
        axes[i, 1].grid(True, alpha=0.3)

        # 활성화 및 흥분
        axes[i, 2].plot(t[mask], data['excitation'][mask], 'g--', label='흥분(excitation)', linewidth=1)
        axes[i, 2].plot(t[mask], data['activation'][mask], 'r-', label='활성화(activation)', linewidth=1.5)
        axes[i, 2].set_ylabel('수준 (0-1)')
        axes[i, 2].set_title(f'{name} - 활성화 동역학')
        axes[i, 2].legend(fontsize=8)
        axes[i, 2].set_ylim(-0.05, 1.1)
        axes[i, 2].grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('시간 (초)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path}")


def plot_muscle_stress(results: dict, scenario_name: str, save_path: str):
    """근육 내부 압력(스트레스) 그래프 + 위험 임계값"""
    muscle_results = results['muscle_results']
    thresholds = InjuryPredictor.MUSCLE_STRESS_THRESHOLDS

    fig, axes = plt.subplots(len(muscle_results), 1,
                             figsize=(14, 4 * len(muscle_results)), squeeze=False)

    fig.suptitle(f'근육 내부 압력(스트레스) - {scenario_name}', fontsize=16, fontweight='bold')

    for i, (name, data) in enumerate(muscle_results.items()):
        ax = axes[i, 0]
        t = data['time']
        stress = data['muscle_stress_kPa']

        ax.plot(t, stress, 'k-', linewidth=0.8, label=f'{name} 스트레스')

        # 위험 구간 배경색
        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        labels_kr = list(thresholds.keys())
        vals = [v / 1000.0 for v in thresholds.values()]  # kPa
        y_max = max(np.max(stress) * 1.2, vals[-1] * 1.1)

        prev = 0
        for j, (val, color) in enumerate(zip(vals, colors)):
            ax.axhspan(prev, val, alpha=0.1, color=color)
            ax.axhline(y=val, color=color, linestyle='--', linewidth=1,
                       label=f'{labels_kr[j]}: {val:.0f} kPa')
            prev = val
        ax.axhspan(prev, y_max, alpha=0.1, color='#8e44ad')

        ax.set_ylabel('스트레스 (kPa)')
        ax.set_title(f'{name}', fontsize=12)
        ax.legend(loc='upper right', fontsize=7)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('시간 (초)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path}")


def plot_ligament_tension(results: dict, scenario_name: str, save_path: str):
    """인대 장력 및 변형률 그래프"""
    ligament_results = results['ligament_results']
    n_ligs = len(ligament_results)

    fig, axes = plt.subplots(n_ligs, 2, figsize=(14, 4 * n_ligs))
    if n_ligs == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'인대 장력 분석 - {scenario_name}', fontsize=16, fontweight='bold')

    strain_thresholds = InjuryPredictor.LIGAMENT_STRAIN_THRESHOLDS

    for i, (name, data) in enumerate(ligament_results.items()):
        t = data['time']

        # 인대 장력
        axes[i, 0].plot(t, data['total_force'], 'b-', linewidth=0.8, label='총 장력')
        axes[i, 0].plot(t, data['spring_force'], 'g--', linewidth=0.6, alpha=0.7, label='스프링 힘')
        axes[i, 0].set_ylabel('장력 (N)')
        axes[i, 0].set_title(f'{name} - 인대 장력')
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)

        # 변형률
        axes[i, 1].plot(t, data['strain_percent'], 'r-', linewidth=0.8)
        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        for j, (label, val) in enumerate(strain_thresholds.items()):
            axes[i, 1].axhline(y=val * 100, color=colors[j], linestyle='--',
                               linewidth=1, label=f'{label}: {val*100:.0f}%')
        axes[i, 1].set_ylabel('변형률 (%)')
        axes[i, 1].set_title(f'{name} - 변형률')
        axes[i, 1].legend(fontsize=7)
        axes[i, 1].grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('시간 (초)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path}")


def plot_injury_risk_heatmap(all_results: dict, save_path: str):
    """부상 위험도 히트맵 (신체 부위 × 시나리오)"""
    scenarios = list(all_results.keys())
    all_regions = set()
    for res in all_results.values():
        all_regions.update(res['body_risks'].keys())
    regions = sorted(all_regions)

    # 위험도 점수 행렬
    matrix = np.zeros((len(regions), len(scenarios)))
    for j, scenario in enumerate(scenarios):
        body_risks = all_results[scenario]['body_risks']
        for i, region in enumerate(regions):
            if region in body_risks:
                matrix[i, j] = body_risks[region]['risk_score']

    fig, ax = plt.subplots(figsize=(12, max(6, len(regions) * 0.8)))

    cmap = plt.cm.colors.ListedColormap(['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=10)

    # 셀에 위험도 텍스트 표시
    risk_labels = ['정상', '낮음', '중간', '높음', '매우높음']
    for i in range(len(regions)):
        for j in range(len(scenarios)):
            score = int(matrix[i, j])
            text_color = 'white' if score >= 3 else 'black'
            ax.text(j, i, risk_labels[score], ha='center', va='center',
                    fontsize=9, fontweight='bold', color=text_color)

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
    cbar.set_ticklabels(risk_labels)

    ax.set_title('소방관 활동별 신체 부위 부상 위험도 히트맵', fontsize=14, fontweight='bold')
    ax.set_xlabel('소방 활동 시나리오')
    ax.set_ylabel('신체 부위')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path}")


def plot_dashboard(all_results: dict, save_path: str):
    """종합 대시보드"""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('소방관 근골격계 부상 예측 시스템 - 종합 대시보드', fontsize=16, fontweight='bold')

    # 1. 시나리오별 최대 근육 스트레스 비교
    ax1 = fig.add_subplot(2, 2, 1)
    scenario_names = []
    muscle_peak_stresses = {}

    for scenario_name, res in all_results.items():
        scenario_names.append(scenario_name)
        for mr in res['muscle_risks']:
            if mr['muscle_name'] not in muscle_peak_stresses:
                muscle_peak_stresses[mr['muscle_name']] = []
            muscle_peak_stresses[mr['muscle_name']].append(mr['peak_stress_kPa'])

    x = np.arange(len(scenario_names))
    width = 0.8 / max(len(muscle_peak_stresses), 1)
    colors_muscle = plt.cm.Set2(np.linspace(0, 1, len(muscle_peak_stresses)))

    for i, (muscle, stresses) in enumerate(muscle_peak_stresses.items()):
        # 시나리오별로 해당 근육이 없으면 0
        padded = []
        for sn in scenario_names:
            found = False
            for mr in all_results[sn]['muscle_risks']:
                if mr['muscle_name'] == muscle:
                    padded.append(mr['peak_stress_kPa'])
                    found = True
                    break
            if not found:
                padded.append(0)
        ax1.bar(x + i * width - 0.4 + width / 2, padded, width,
                label=muscle, color=colors_muscle[i])

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, fontsize=8)
    ax1.set_ylabel('최대 스트레스 (kPa)')
    ax1.set_title('시나리오별 최대 근육 내부 압력')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    # 위험 기준선
    for label, val in InjuryPredictor.MUSCLE_STRESS_THRESHOLDS.items():
        ax1.axhline(y=val / 1000, color='red', linestyle=':', alpha=0.5, linewidth=0.8)

    # 2. 시나리오별 최대 인대 변형률 비교
    ax2 = fig.add_subplot(2, 2, 2)
    ligament_peak_strains = {}

    for scenario_name, res in all_results.items():
        for lr in res['ligament_risks']:
            if lr['ligament_name'] not in ligament_peak_strains:
                ligament_peak_strains[lr['ligament_name']] = {}
            ligament_peak_strains[lr['ligament_name']][scenario_name] = lr['peak_strain_pct']

    width2 = 0.8 / max(len(ligament_peak_strains), 1)
    colors_lig = plt.cm.Set1(np.linspace(0, 1, len(ligament_peak_strains)))

    for i, (lig, strain_dict) in enumerate(ligament_peak_strains.items()):
        padded = [strain_dict.get(sn, 0) for sn in scenario_names]
        ax2.bar(x + i * width2 - 0.4 + width2 / 2, padded, width2,
                label=lig, color=colors_lig[i])

    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_names, fontsize=8)
    ax2.set_ylabel('최대 변형률 (%)')
    ax2.set_title('시나리오별 최대 인대 변형률')
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    for label, val in InjuryPredictor.LIGAMENT_STRAIN_THRESHOLDS.items():
        ax2.axhline(y=val * 100, color='red', linestyle=':', alpha=0.5, linewidth=0.8)

    # 3. 피로 누적 지수 비교
    ax3 = fig.add_subplot(2, 2, 3)
    fatigue_data = {}
    for scenario_name, res in all_results.items():
        for mr in res['muscle_risks']:
            if mr['muscle_name'] not in fatigue_data:
                fatigue_data[mr['muscle_name']] = {}
            fatigue_data[mr['muscle_name']][scenario_name] = mr['fatigue_index']

    width3 = 0.8 / max(len(fatigue_data), 1)
    for i, (muscle, fdata) in enumerate(fatigue_data.items()):
        padded = [fdata.get(sn, 0) for sn in scenario_names]
        ax3.bar(x + i * width3 - 0.4 + width3 / 2, padded, width3,
                label=muscle, color=colors_muscle[i % len(colors_muscle)])

    ax3.set_xticks(x)
    ax3.set_xticklabels(scenario_names, fontsize=8)
    ax3.set_ylabel('피로 누적 지수')
    ax3.set_title('시나리오별 근육 피로 누적 지수')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 종합 위험도 요약 테이블
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    table_data = []
    for sn in scenario_names:
        res = all_results[sn]
        max_muscle_risk = max(res['muscle_risks'], key=lambda x: x['peak_stress_kPa'])
        max_lig_risk = max(res['ligament_risks'], key=lambda x: x['peak_strain_pct'])

        # 종합 위험 점수
        risk_levels = ['정상', '낮음', '중간', '높음', '매우높음']
        max_body_score = max(
            (info['risk_score'] for info in res['body_risks'].values()), default=0
        )

        table_data.append([
            sn,
            f"{max_muscle_risk['muscle_name']}\n({max_muscle_risk['peak_stress_kPa']:.0f} kPa)",
            max_muscle_risk['risk_level'],
            f"{max_lig_risk['ligament_name']}\n({max_lig_risk['peak_strain_pct']:.1f}%)",
            max_lig_risk['combined_risk'],
            risk_levels[max_body_score],
        ])

    col_labels = ['시나리오', '최대 스트레스 근육', '근육\n위험도',
                  '최대 변형률 인대', '인대\n위험도', '종합\n위험도']
    table = ax4.table(cellText=table_data, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)

    # 위험도에 따라 셀 배경색 설정
    risk_colors = {'정상': '#27ae60', '낮음': '#2ecc71', '중간': '#f1c40f',
                   '높음': '#e67e22', '매우높음': '#e74c3c'}
    for i, row in enumerate(table_data):
        for j, val in enumerate(row):
            if val in risk_colors:
                table[i + 1, j].set_facecolor(risk_colors[val] + '40')

    ax4.set_title('시나리오별 부상 위험도 요약', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path}")


def plot_model_curves(save_path: str):
    """모델 검증용 기본 커브 (force-length, force-velocity, tendon)"""
    muscle = DeGrooteFregly2016Muscle(
        name='참조 근육', max_isometric_force=1000,
        optimal_fiber_length=0.1, tendon_slack_length=0.2,
        pennation_angle_at_optimal=0.0, pcsa=10e-4
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('OpenSim DeGrooteFregly2016 근육 모델 특성 곡선', fontsize=14, fontweight='bold')

    # 활성 힘-길이
    nfl = np.linspace(0.3, 1.7, 300)
    f_act_fl = muscle.calc_active_force_length_multiplier(nfl)
    axes[0, 0].plot(nfl, f_act_fl, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('정규화 섬유 길이')
    axes[0, 0].set_ylabel('활성 힘-길이 배율')
    axes[0, 0].set_title('활성 힘-길이 곡선 (Active Force-Length)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='최적 길이')
    axes[0, 0].legend()

    # 수동 힘-길이
    f_pass_fl = muscle.calc_passive_force_length_multiplier(nfl)
    axes[0, 1].plot(nfl, f_pass_fl, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('정규화 섬유 길이')
    axes[0, 1].set_ylabel('수동 힘-길이 배율')
    axes[0, 1].set_title('수동 힘-길이 곡선 (Passive Force-Length)')
    axes[0, 1].grid(True, alpha=0.3)

    # 힘-속도
    nfv = np.linspace(-1.0, 1.0, 300)
    f_fv = muscle.calc_force_velocity_multiplier(nfv)
    axes[1, 0].plot(nfv, f_fv, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('정규화 섬유 속도 (음=단축, 양=신장)')
    axes[1, 0].set_ylabel('힘-속도 배율')
    axes[1, 0].set_title('힘-속도 곡선 (Force-Velocity)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)

    # 건 힘-길이
    ntl = np.linspace(0.95, 1.1, 300)
    f_tendon = muscle.calc_tendon_force_multiplier(ntl)
    axes[1, 1].plot(ntl, f_tendon, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('정규화 건 길이')
    axes[1, 1].set_ylabel('건 힘-길이 배율')
    axes[1, 1].set_title('건 힘-길이 곡선 (Tendon Force-Length)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path}")


def plot_ligament_model_curves(save_path: str):
    """인대 모델 특성 곡선"""
    ligament = Blankevoort1991Ligament(
        name='참조 인대', slack_length=0.03,
        linear_stiffness=5000.0, transition_strain=0.06
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('OpenSim Blankevoort1991 인대 모델 특성 곡선', fontsize=14, fontweight='bold')

    # 힘-변형률 곡선
    strains = np.linspace(-0.02, 0.15, 500)
    forces = np.array([ligament.calc_spring_force(s) for s in strains])

    axes[0].plot(strains * 100, forces, 'b-', linewidth=2)
    axes[0].axvline(x=ligament.transition_strain * 100, color='r', linestyle='--',
                    alpha=0.7, label=f'전환 변형률 ({ligament.transition_strain*100:.0f}%)')
    axes[0].set_xlabel('변형률 (%)')
    axes[0].set_ylabel('스프링 힘 (N)')
    axes[0].set_title('인대 힘-변형률 곡선')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 위험 구간 표시
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    for (label, val), color in zip(InjuryPredictor.LIGAMENT_STRAIN_THRESHOLDS.items(), colors):
        axes[0].axvline(x=val * 100, color=color, linestyle=':', alpha=0.5,
                        label=f'{label}: {val*100:.0f}%')
    axes[0].legend(fontsize=8)

    # 3구간 설명 (이완, 발끝, 선형)
    axes[1].fill_between([-2, 0], 0, 1, alpha=0.2, color='green', label='이완 구간 (Slack)')
    axes[1].fill_between([0, 6], 0, 1, alpha=0.2, color='yellow', label='발끝 구간 (Toe)')
    axes[1].fill_between([6, 15], 0, 1, alpha=0.2, color='orange', label='선형 구간 (Linear)')
    axes[1].fill_between([15, 20], 0, 1, alpha=0.2, color='red', label='파열 구간 (Failure)')

    axes[1].plot(strains * 100, forces / max(forces.max(), 1), 'k-', linewidth=2)
    axes[1].set_xlabel('변형률 (%)')
    axes[1].set_ylabel('정규화 힘')
    axes[1].set_title('인대 변형 구간 분류')
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(-2, 20)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path}")


# ============================================================================
# CSV 출력 함수
# ============================================================================

def export_muscle_csv(all_results: dict, filepath: str):
    """근육 시뮬레이션 결과 CSV 출력 (요약 데이터)"""
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            '시나리오', '근육명', '최대_활성력(N)', '최대_수동력(N)',
            '최대_총섬유력(N)', '최대_스트레스(kPa)', '평균_스트레스(kPa)',
            '위험도', '피로누적지수', '중간위험_초과시간(%)', '높은위험_초과시간(%)'
        ])

        for scenario_name, res in all_results.items():
            for mr in res['muscle_risks']:
                muscle_data = res['muscle_results'][mr['muscle_name']]
                writer.writerow([
                    scenario_name,
                    mr['muscle_name'],
                    f"{np.max(muscle_data['active_fiber_force']):.1f}",
                    f"{np.max(muscle_data['passive_fiber_force']):.1f}",
                    f"{np.max(muscle_data['total_fiber_force']):.1f}",
                    f"{mr['peak_stress_kPa']:.1f}",
                    f"{mr['mean_stress_kPa']:.1f}",
                    mr['risk_level'],
                    f"{mr['fatigue_index']:.2f}",
                    f"{mr['time_above_medium_pct']:.1f}",
                    f"{mr['time_above_high_pct']:.1f}",
                ])

    print(f"  저장: {filepath}")


def export_ligament_csv(all_results: dict, filepath: str):
    """인대 시뮬레이션 결과 CSV 출력"""
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            '시나리오', '인대명', '최대_변형률(%)', '최대_장력(N)',
            '평균_장력(N)', '파괴력_대비(%)', '변형위험도', '장력위험도',
            '종합위험도', '하중반복횟수'
        ])

        for scenario_name, res in all_results.items():
            for lr in res['ligament_risks']:
                writer.writerow([
                    scenario_name,
                    lr['ligament_name'],
                    f"{lr['peak_strain_pct']:.2f}",
                    f"{lr['peak_force_N']:.1f}",
                    f"{lr['mean_force_N']:.1f}",
                    f"{lr['force_ratio_pct']:.1f}",
                    lr['strain_risk'],
                    lr['force_risk'],
                    lr['combined_risk'],
                    lr['loading_cycles'],
                ])

    print(f"  저장: {filepath}")


def export_injury_summary_csv(all_results: dict, filepath: str):
    """부상 위험도 종합 요약 CSV"""
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['시나리오', '신체부위', '위험도', '위험점수', '관련조직수'])

        for scenario_name, res in all_results.items():
            for region, info in res['body_risks'].items():
                writer.writerow([
                    scenario_name, region,
                    info['risk_level'], info['risk_score'], info['num_tissues']
                ])

    print(f"  저장: {filepath}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("=" * 70)
    print("  소방관 근골격계 부상 예측 시스템")
    print("  OpenSim DeGrooteFregly2016Muscle / Blankevoort1991Ligament 기반")
    print("=" * 70)
    print()

    # 모델 특성 곡선 생성
    print("[1/4] 모델 특성 곡선 생성...")
    plot_model_curves(os.path.join(OUTPUT_DIR, 'model_muscle_curves.png'))
    plot_ligament_model_curves(os.path.join(OUTPUT_DIR, 'model_ligament_curves.png'))

    # 시뮬레이션 실행
    print("\n[2/4] 시나리오 시뮬레이션 실행...")
    engine = SimulationEngine()
    all_results = engine.run_all_scenarios()

    # 시각화
    print("\n[3/4] 시각화 생성...")
    for scenario_name, results in all_results.items():
        safe_name = scenario_name.replace(' ', '_')
        plot_muscle_forces(results, scenario_name,
                           os.path.join(OUTPUT_DIR, f'{safe_name}_muscle_forces.png'))
        plot_muscle_stress(results, scenario_name,
                           os.path.join(OUTPUT_DIR, f'{safe_name}_muscle_stress.png'))
        plot_ligament_tension(results, scenario_name,
                              os.path.join(OUTPUT_DIR, f'{safe_name}_ligament_tension.png'))

    plot_injury_risk_heatmap(all_results,
                             os.path.join(OUTPUT_DIR, 'injury_risk_heatmap.png'))
    plot_dashboard(all_results,
                   os.path.join(OUTPUT_DIR, 'dashboard.png'))

    # CSV 출력
    print("\n[4/4] CSV 결과 저장...")
    export_muscle_csv(all_results, os.path.join(OUTPUT_DIR, 'muscle_forces.csv'))
    export_ligament_csv(all_results, os.path.join(OUTPUT_DIR, 'ligament_forces.csv'))
    export_injury_summary_csv(all_results, os.path.join(OUTPUT_DIR, 'injury_summary.csv'))

    # 보고서 출력
    print("\n" + "=" * 70)
    print("  부상 위험도 분석 보고서")
    print("=" * 70)
    for scenario_name, results in all_results.items():
        print(results['report'])
        print()

    print(f"\n모든 결과가 {OUTPUT_DIR} 에 저장되었습니다.")
    print("=" * 70)


if __name__ == '__main__':
    main()
