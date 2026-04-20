"""
Kinovea 관절 각도 데이터로 부상 예측 시뮬레이션 실행

사용법:
  python run_kinovea.py <CSV파일경로> [시나리오이름]

예시:
  python run_kinovea.py sample_kinovea_data.csv "계단 오르기 실측"
  python run_kinovea.py C:/data/firefighter_stair.csv

Kinovea CSV 형식:
  Time(ms), Knee_Angle, Hip_Angle, Ankle_Angle, Shoulder_Angle, Elbow_Angle, Trunk_Angle
  0,        170.2,      165.3,     95.1,         30.5,           155.2,       175.0
  33,       168.5,      163.1,     93.8,         32.1,           153.0,       173.5
  ...

  - 첫 번째 컬럼: 시간 (밀리초 또는 초)
  - 나머지 컬럼: 관절 각도 (도, degree)
  - 모든 관절이 필수는 아님 (있는 관절만 분석)

Kinovea에서 데이터 내보내기:
  1. Kinovea에서 영상 열기
  2. 관절에 각도 측정 도구 배치 (무릎, 고관절, 발목, 어깨, 팔꿈치, 몸통)
  3. 추적 실행 (마우스 우클릭 > Track Path)
  4. 도구 > 각도 운동학(Angular kinematics) > 내보내기(Export)
  5. CSV 파일로 저장
  6. 본 스크립트에 경로 전달

의존성: pip install numpy matplotlib scipy
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

for font_name in ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']:
    try:
        rcParams['font.family'] = font_name
        break
    except:
        continue
rcParams['axes.unicode_minus'] = False

from firefighter_biomech import (
    SimulationEngine, KinoveaInput, JointAngleToMuscle,
    InjuryPredictor, MUSCLE_PARAMS, LIGAMENT_PARAMS
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Kinovea 입력 데이터 시각화
# ============================================================================

def plot_kinovea_input(kinovea_data: dict, save_path: str):
    """Kinovea 입력 관절 각도 시계열 그래프"""
    time = kinovea_data['time']
    joints = kinovea_data['joints']
    n_joints = len(joints)

    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 3 * n_joints), squeeze=False)
    fig.suptitle('Kinovea 입력 데이터 - 관절 각도 시계열', fontsize=14, fontweight='bold')

    joint_labels = {
        'Knee_Angle': '무릎 각도',
        'Hip_Angle': '고관절 각도',
        'Ankle_Angle': '발목 각도',
        'Shoulder_Angle': '어깨 각도',
        'Elbow_Angle': '팔꿈치 각도',
        'Trunk_Angle': '몸통 각도',
    }

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    for i, (joint_name, angle_data) in enumerate(joints.items()):
        ax = axes[i, 0]
        label = joint_labels.get(joint_name, joint_name)
        ax.plot(time, angle_data, color=colors[i % len(colors)], linewidth=1.5)
        ax.set_ylabel(f'{label} (deg)')
        ax.set_title(f'{label} ({joint_name})')
        ax.grid(True, alpha=0.3)

        # 각도 범위 표시
        ax.text(0.98, 0.95,
                f'min={angle_data.min():.1f}  max={angle_data.max():.1f}  ROM={angle_data.max()-angle_data.min():.1f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[-1, 0].set_xlabel('시간 (초)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path}")


def plot_angle_to_muscle_conversion(kinovea_data: dict, scenario_data: dict,
                                    save_path: str):
    """관절 각도 → 근육 섬유 길이 변환 과정 시각화"""
    time = kinovea_data['time']
    joints = kinovea_data['joints']
    muscles = scenario_data['muscles']

    n_muscles = len(muscles)
    fig, axes = plt.subplots(n_muscles, 3, figsize=(18, 3.5 * n_muscles))
    if n_muscles == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('관절 각도 -> 근육 섬유 길이 변환 과정', fontsize=14, fontweight='bold')

    for i, (muscle_name, profiles) in enumerate(muscles.items()):
        mapping = JointAngleToMuscle.MAPPING[muscle_name]
        joint_name = mapping['joint']
        angle_data = joints[joint_name]
        nfl, nfv = profiles['fiber_length']
        excitation = profiles['excitation']

        # (1) 입력: 관절 각도
        axes[i, 0].plot(time, angle_data, 'b-', linewidth=1.2)
        axes[i, 0].set_ylabel(f'{joint_name} (deg)')
        axes[i, 0].set_title(f'{muscle_name} - 입력: {joint_name}')
        axes[i, 0].grid(True, alpha=0.3)

        # 변환 범위 표시
        theta_min, theta_max = mapping['theta_range']
        axes[i, 0].axhline(y=theta_min, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        axes[i, 0].axhline(y=theta_max, color='r', linestyle='--', alpha=0.5, linewidth=0.8)

        # (2) 변환: 정규화 섬유 길이
        axes[i, 1].plot(time, nfl, 'g-', linewidth=1.2)
        axes[i, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, linewidth=0.8,
                           label='최적 길이')
        axes[i, 1].set_ylabel('정규화 섬유 길이')
        axes[i, 1].set_title(f'{muscle_name} - 변환: 정규화 섬유 길이')
        axes[i, 1].legend(fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)

        # (3) 추정: 흥분 프로파일
        axes[i, 2].plot(time, excitation, 'r-', linewidth=1.2)
        axes[i, 2].set_ylabel('추정 흥분 (0-1)')
        axes[i, 2].set_title(f'{muscle_name} - 추정: 근육 흥분')
        axes[i, 2].set_ylim(-0.05, 1.1)
        axes[i, 2].grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('시간 (초)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path}")


def plot_muscle_stress_kinovea(results: dict, scenario_name: str, save_path: str):
    """근육 내부 압력(스트레스) 그래프"""
    muscle_results = results['muscle_results']
    thresholds = InjuryPredictor.MUSCLE_STRESS_THRESHOLDS

    fig, axes = plt.subplots(len(muscle_results), 1,
                             figsize=(14, 4 * len(muscle_results)), squeeze=False)
    fig.suptitle(f'근육 내부 압력(스트레스) - {scenario_name} [Kinovea 실측]',
                 fontsize=14, fontweight='bold')

    for i, (name, data) in enumerate(muscle_results.items()):
        ax = axes[i, 0]
        t = data['time']
        stress = data['muscle_stress_kPa']

        ax.plot(t, stress, 'k-', linewidth=0.8, label=f'{name} 스트레스')

        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        labels_kr = list(thresholds.keys())
        vals = [v / 1000.0 for v in thresholds.values()]
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


def plot_ligament_tension_kinovea(results: dict, scenario_name: str, save_path: str):
    """인대 장력 그래프"""
    ligament_results = results['ligament_results']
    if not ligament_results:
        return

    n_ligs = len(ligament_results)
    fig, axes = plt.subplots(n_ligs, 2, figsize=(14, 4 * n_ligs))
    if n_ligs == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'인대 장력 - {scenario_name} [Kinovea 실측]',
                 fontsize=14, fontweight='bold')

    strain_thresholds = InjuryPredictor.LIGAMENT_STRAIN_THRESHOLDS

    for i, (name, data) in enumerate(ligament_results.items()):
        t = data['time']

        axes[i, 0].plot(t, data['total_force'], 'b-', linewidth=0.8, label='총 장력')
        axes[i, 0].plot(t, data['spring_force'], 'g--', linewidth=0.6, alpha=0.7, label='스프링 힘')
        axes[i, 0].set_ylabel('장력 (N)')
        axes[i, 0].set_title(f'{name} - 인대 장력')
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)

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


def plot_body_risk_summary(results: dict, scenario_name: str, save_path: str):
    """신체 부위별 부상 위험도 요약 차트"""
    body_risks = results['body_risks']
    muscle_risks = results['muscle_risks']
    ligament_risks = results['ligament_risks']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'부상 위험도 종합 - {scenario_name} [Kinovea 실측]',
                 fontsize=14, fontweight='bold')

    # (1) 신체 부위별 위험도 막대 그래프
    regions = list(body_risks.keys())
    scores = [body_risks[r]['risk_score'] for r in regions]
    risk_colors_map = {0: '#27ae60', 1: '#2ecc71', 2: '#f1c40f', 3: '#e67e22', 4: '#e74c3c'}
    bar_colors = [risk_colors_map[s] for s in scores]

    axes[0].barh(regions, scores, color=bar_colors, edgecolor='gray')
    axes[0].set_xlim(0, 5)
    axes[0].set_xlabel('위험 점수')
    axes[0].set_title('신체 부위별 위험도')

    risk_labels = ['정상', '낮음', '중간', '높음', '매우높음']
    for i, (region, score) in enumerate(zip(regions, scores)):
        axes[0].text(score + 0.1, i, risk_labels[score], va='center', fontsize=10)

    axes[0].set_xticks([0, 1, 2, 3, 4])
    axes[0].set_xticklabels(risk_labels, fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='x')

    # (2) 근육 피크 스트레스 비교
    muscle_names = [mr['muscle_name'] for mr in muscle_risks]
    peak_stresses = [mr['peak_stress_kPa'] for mr in muscle_risks]

    stress_colors = []
    thresholds_vals = list(InjuryPredictor.MUSCLE_STRESS_THRESHOLDS.values())
    for ps in peak_stresses:
        ps_pa = ps * 1000
        if ps_pa >= thresholds_vals[3]:
            stress_colors.append('#e74c3c')
        elif ps_pa >= thresholds_vals[2]:
            stress_colors.append('#e67e22')
        elif ps_pa >= thresholds_vals[1]:
            stress_colors.append('#f1c40f')
        elif ps_pa >= thresholds_vals[0]:
            stress_colors.append('#2ecc71')
        else:
            stress_colors.append('#27ae60')

    axes[1].barh(muscle_names, peak_stresses, color=stress_colors, edgecolor='gray')
    axes[1].set_xlabel('최대 스트레스 (kPa)')
    axes[1].set_title('근육별 최대 내부 압력')
    axes[1].grid(True, alpha=0.3, axis='x')

    for label, val in InjuryPredictor.MUSCLE_STRESS_THRESHOLDS.items():
        axes[1].axvline(x=val / 1000, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path}")


def export_kinovea_results_csv(results: dict, scenario_name: str, output_dir: str):
    """Kinovea 분석 결과 CSV 출력"""
    # 근육 결과
    filepath = os.path.join(output_dir, f'kinovea_muscle_results.csv')
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            '시나리오', '근육명', '최대_활성력(N)', '최대_수동력(N)',
            '최대_총섬유력(N)', '최대_스트레스(kPa)', '평균_스트레스(kPa)',
            '위험도', '피로누적지수'
        ])
        for mr in results['muscle_risks']:
            md = results['muscle_results'][mr['muscle_name']]
            writer.writerow([
                scenario_name, mr['muscle_name'],
                f"{np.max(md['active_fiber_force']):.1f}",
                f"{np.max(md['passive_fiber_force']):.1f}",
                f"{np.max(md['total_fiber_force']):.1f}",
                f"{mr['peak_stress_kPa']:.1f}",
                f"{mr['mean_stress_kPa']:.1f}",
                mr['risk_level'],
                f"{mr['fatigue_index']:.4f}",
            ])
    print(f"  저장: {filepath}")

    # 인대 결과
    filepath = os.path.join(output_dir, f'kinovea_ligament_results.csv')
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            '시나리오', '인대명', '최대_변형률(%)', '최대_장력(N)',
            '종합위험도', '하중반복횟수'
        ])
        for lr in results['ligament_risks']:
            writer.writerow([
                scenario_name, lr['ligament_name'],
                f"{lr['peak_strain_pct']:.2f}",
                f"{lr['peak_force_N']:.1f}",
                lr['combined_risk'],
                lr['loading_cycles'],
            ])
    print(f"  저장: {filepath}")

    # 신체 부위별 위험도
    filepath = os.path.join(output_dir, f'kinovea_body_risks.csv')
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['시나리오', '신체부위', '위험도', '위험점수'])
        for region, info in results['body_risks'].items():
            writer.writerow([
                scenario_name, region, info['risk_level'], info['risk_score']
            ])
    print(f"  저장: {filepath}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("=" * 70)
    print("  소방관 부상 예측 시스템 - Kinovea 데이터 입력 모드")
    print("=" * 70)

    # CSV 파일 경로 처리
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    else:
        # 기본: 샘플 데이터 사용
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'sample_kinovea_data.csv')
        print(f"\n  CSV 파일이 지정되지 않아 샘플 데이터를 사용합니다.")
        print(f"  사용법: python run_kinovea.py <CSV파일경로> [시나리오이름]")

    scenario_name = sys.argv[2] if len(sys.argv) >= 3 else 'Kinovea 측정'

    if not os.path.exists(csv_path):
        print(f"\n  오류: 파일을 찾을 수 없습니다: {csv_path}")
        sys.exit(1)

    print(f"\n  입력 파일: {csv_path}")
    print(f"  시나리오: {scenario_name}")

    # 1단계: Kinovea CSV 로드
    print(f"\n[1/5] Kinovea CSV 데이터 로드...")
    kinovea_data = KinoveaInput.load_csv(csv_path)

    # 2단계: 관절 각도 → 시뮬레이션 데이터 변환
    print(f"\n[2/5] 관절 각도 -> 근육/인대 데이터 변환...")
    scenario_data = KinoveaInput.convert_to_scenario(
        kinovea_data, scenario_name=scenario_name
    )

    # 3단계: 시뮬레이션 실행
    print(f"\n[3/5] 시뮬레이션 실행...")
    engine = SimulationEngine()
    results = engine.run_scenario_from_data(scenario_data)

    # 4단계: 시각화
    print(f"\n[4/5] 시각화 생성...")
    plot_kinovea_input(kinovea_data,
                       os.path.join(OUTPUT_DIR, 'kinovea_input_angles.png'))
    plot_angle_to_muscle_conversion(kinovea_data, scenario_data,
                                    os.path.join(OUTPUT_DIR, 'kinovea_conversion.png'))
    plot_muscle_stress_kinovea(results, scenario_name,
                               os.path.join(OUTPUT_DIR, 'kinovea_muscle_stress.png'))
    plot_ligament_tension_kinovea(results, scenario_name,
                                  os.path.join(OUTPUT_DIR, 'kinovea_ligament_tension.png'))
    plot_body_risk_summary(results, scenario_name,
                           os.path.join(OUTPUT_DIR, 'kinovea_body_risk_summary.png'))

    # 5단계: CSV 출력
    print(f"\n[5/5] CSV 결과 저장...")
    export_kinovea_results_csv(results, scenario_name, OUTPUT_DIR)

    # 보고서 출력
    print("\n" + results['report'])

    print(f"\n모든 결과가 {OUTPUT_DIR} 에 저장되었습니다.")

    # 입력 데이터 매핑 안내
    print("\n" + "=" * 70)
    print("  관절-근육/인대 매핑 참조")
    print("=" * 70)
    print(f"\n  {'관절 컬럼':<20} {'관련 근육/인대':<60}")
    print("  " + "-" * 75)

    joint_to_tissues = {}
    for muscle, mapping in JointAngleToMuscle.MAPPING.items():
        j = mapping['joint']
        if j not in joint_to_tissues:
            joint_to_tissues[j] = {'muscles': [], 'ligaments': []}
        joint_to_tissues[j]['muscles'].append(muscle)
    for lig, mapping in JointAngleToMuscle.LIGAMENT_MAPPING.items():
        j = mapping['joint']
        if j not in joint_to_tissues:
            joint_to_tissues[j] = {'muscles': [], 'ligaments': []}
        joint_to_tissues[j]['ligaments'].append(lig)

    for joint, tissues in joint_to_tissues.items():
        parts = []
        if tissues['muscles']:
            parts.append(f"근육: {', '.join(tissues['muscles'])}")
        if tissues['ligaments']:
            parts.append(f"인대: {', '.join(tissues['ligaments'])}")
        print(f"  {joint:<20} {' | '.join(parts)}")

    print("=" * 70)


if __name__ == '__main__':
    main()
