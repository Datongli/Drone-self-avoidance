"""
=============================================================================
统计显著性检验脚本 (Statistical Significance Test)
=============================================================================

用途:
    根据审稿人意见 R2-Q4 / R3-Q9，对 MTrans-SAC 与各基线算法进行
    Welch's t-test (不等方差 t 检验)，验证性能差异的统计显著性。

    对应修订计划 §5.5 显著性检验提示。

使用方法:
    python statistical_significance_test.py
    python statistical_significance_test.py --csv baseline_benchmark_results.csv
    python statistical_significance_test.py --csv results.csv --level 10 --alpha 0.05

输出:
    - 控制台格式化显著性表格
    - p_values.csv
=============================================================================
"""

import csv
import re
import os
import sys
import argparse
import warnings
from collections import defaultdict

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


# =============================================================================
# 数据解析
# =============================================================================

def parse_benchmark_csv(filepath: str) -> dict:
    """
    解析基准测试 CSV 文件
    返回: dict[model_name][env_level] = {
        'success': [10次重复的成功次数],
        'collision': [...], 'power': [...], 'stepover': [...],
        'asr': [成功率百分比],
        'acr': [碰撞率百分比],
        'alr': [丢失率百分比(能量耗尽+超步)]
    }
    """
    data = defaultdict(lambda: defaultdict(lambda: {
        'success': [], 'collision': [], 'power': [], 'stepover': []
    }))

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if not row or len(row) < 6:
                continue
            # 跳过均值和标准差行
            if 'mean' in row[0].lower() or 'mean' in row[2].lower() or 'Mean' in row[2]:
                continue

            model = row[0].strip()
            level = int(row[1].strip())
            success = int(row[3].strip())
            collision = int(row[4].strip())
            power = int(row[5].strip())
            stepover = int(row[6].strip()) if len(row) > 6 else 0

            data[model][level]['success'].append(success)
            data[model][level]['collision'].append(collision)
            data[model][level]['power'].append(power)
            data[model][level]['stepover'].append(stepover)

    # 计算百分比（基于1000 episodes）
    for model in data:
        for level in data[model]:
            d = data[model][level]
            n = len(d['success'])
            if n == 0:
                continue
            d['asr'] = [s / 10.0 for s in d['success']]      # 成功率 %
            d['acr'] = [c / 10.0 for c in d['collision']]    # 碰撞率 %
            d['alr'] = [(p + o) / 10.0 for p, o in zip(d['power'], d['stepover'])]  # 丢失率 %

    return data


# =============================================================================
# 显著性检验
# =============================================================================

def welch_ttest(group_a: list, group_b: list, alpha: float = 0.05):
    """
    Welch's t-test (不等方差独立双样本 t 检验)
    H0: 两组均值相等
    H1: 两组均值不等（双尾检验）
    """
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    significant = p_value < alpha
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': significant,
        'alpha': alpha,
        'mean_a': np.mean(group_a),
        'std_a': np.std(group_a, ddof=1),
        'mean_b': np.mean(group_b),
        'std_b': np.std(group_b, ddof=1),
        'diff': np.mean(group_a) - np.mean(group_b),
    }


def run_all_tests(data: dict, reference_model: str = "SAC", levels=None, alpha=0.05):
    """
    对所有基线模型在指定难度等级上运行显著性检验
    reference_model: 参考模型（通常是 MTrans-SAC = "SAC"）
    """
    if levels is None:
        levels = [5, 8, 10, 15]

    results = []

    # 获取参考模型数据
    ref_data = data.get(reference_model, {})
    if not ref_data:
        print(f"[ERROR] 参考模型 '{reference_model}' 在数据中不存在!")
        print(f"  可用模型: {list(data.keys())}")
        return results

    for model_name in sorted(data.keys()):
        if model_name == reference_model:
            continue

        model_data = data[model_name]
        for level in levels:
            if level not in ref_data or level not in model_data:
                continue

            ref_asr = ref_data[level].get('asr', [])
            model_asr = model_data[level].get('asr', [])

            if len(ref_asr) < 2 or len(model_asr) < 2:
                print(f"  [WARN] {reference_model} vs {model_name} @ ξ={level}: "
                      f"样本不足 (n_ref={len(ref_asr)}, n_model={len(model_asr)})")
                continue

            result = welch_ttest(model_asr, ref_asr, alpha=alpha)
            result['reference'] = reference_model
            result['model'] = model_name
            result['level'] = level
            results.append(result)

    return results


# =============================================================================
# 输出格式化
# =============================================================================

def print_results(results: list, alpha: float = 0.05):
    """打印格式化的显著性检验结果"""
    print("\n" + "=" * 95)
    print(f"  Statistical Significance Analysis (Welch's t-test, α = {alpha})")
    print(f"  H0: No difference in ASR  |  H1: Significant difference (two-tailed)")
    print("=" * 95)

    # 按难度等级分组
    by_level = defaultdict(list)
    for r in results:
        by_level[r['level']].append(r)

    for level in sorted(by_level.keys()):
        print(f"\n  ┌─ ξ = {level} ───────────────────────────────────────────────────────┐")
        for r in by_level[level]:
            sig_mark = "✅ SIGNIFICANT" if r['significant'] else "  not significant"
            if r['significant']:
                p_str = f"p = {r['p_value']:.2e}"
            else:
                p_str = f"p = {r['p_value']:.4f}"

            direction = "↑" if r['diff'] > 0 else "↓"
            print(f"  │ {r['model']:<20} vs {r['reference']:<20} "
                  f"ΔASR = {r['diff']:+6.2f}{direction} pp  "
                  f"t = {r['t_statistic']:7.3f}  {p_str:<14} {sig_mark}")
        print(f"  └{'─' * 80}┘")

    # 汇总表
    print("\n" + "=" * 95)
    print(f"  {'Model':<20} {'ξ':>4} {'ΔASR (pp)':>10} {'t-stat':>8} {'p-value':>12} {'Significant?':>14}")
    print("  " + "-" * 70)
    for r in sorted(results, key=lambda x: (x['level'], x['model'])):
        sig = "YES p<{:.0e}".format(r['alpha']) if r['significant'] else "NO"
        print(f"  {r['model']:<20} {r['level']:>4} {r['diff']:>+10.2f} {r['t_statistic']:>8.3f} "
              f"{r['p_value']:>12.2e} {sig:>14}")
    print("=" * 95)


def save_csv(results: list, filepath: str = "p_values.csv"):
    """保存显著性检验结果到 CSV"""
    columns = ["Reference", "Model", "Level", "Mean_Ref_ASR", "Std_Ref_ASR",
               "Mean_Model_ASR", "Std_Model_ASR", "Delta_ASR_pp",
               "t_statistic", "p_value", "Significant"]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in results:
            writer.writerow([
                r['reference'], r['model'], r['level'],
                round(r['mean_a'], 4), round(r['std_a'], 4),
                round(r['mean_b'], 4), round(r['std_b'], 4),
                round(r['diff'], 4),
                round(r['t_statistic'], 4),
                f"{r['p_value']:.4e}",
                "YES" if r['significant'] else "NO"
            ])
    print(f"\n✅ p-values saved to: {filepath}")


# =============================================================================
# 手动数据输入支持（当CSV不可用时）
# =============================================================================

def manual_data_example():
    """使用修订计划中的已知数据进行示例检验"""
    # 来自 detailed_revision_plan.md §5.7 数据状态映射表
    # 这里用均值 ± std 反推合成 seed-level 数据（近似）
    # 实际使用时应从原始 seed 数据直接读取
    print("\n[INFO] 使用内置示例数据...")
    print("[WARN] 示例数据仅用于演示，提交论文请使用实际 10 seed 数据!")

    # 模拟10个seed的数据（基于已知的mean±std）
    np.random.seed(42)
    n_seeds = 10

    data = {}
    models_info = {
        "MTrans-SAC": {
            5: (87.32, 1.29), 8: (81.29, 1.85), 10: (79.24, 2.76), 15: (68.30, 3.03)
        },
        "DPRL": {
            5: (85.65, 1.55), 8: (78.78, 1.94), 10: (75.14, 2.33), 15: (64.66, 1.60)
        },
        "PointTransSAC": {
            5: (85.81, 1.57), 8: (80.39, 2.39), 10: (76.89, 1.75), 15: (67.58, 2.81)
        },
        "SetTransSAC": {
            5: (70.55, 2.69), 8: (64.81, 2.07), 10: (59.01, 2.95), 15: (52.24, 2.16)
        },
        "w/o Gating": {
            5: (61.92, 2.93), 8: (52.60, 2.65), 10: (49.73, 2.97), 15: (38.87, 2.83)
        },
        "w/o Trans+Gating": {
            5: (81.95, 1.32), 8: (74.40, 1.13), 10: (70.59, 1.96), 15: (59.49, 2.09)
        },
        "DDPG": {
            5: (73.88, 2.87), 8: (65.15, 3.09), 10: (60.17, 2.91), 15: (51.30, 3.06)
        },
        "Homo-Q": {
            5: (66.45, 3.44), 8: (57.28, 1.39), 10: (52.45, 1.76), 15: (41.30, 1.79)
        },
    }

    for model, levels in models_info.items():
        data[model] = {}
        for level, (mean_asr, std_asr) in levels.items():
            # 生成近似的 seed-level ASR 数据
            seeds_asr = np.random.normal(mean_asr, std_asr, n_seeds)
            # 限制在 [0, 100]
            seeds_asr = np.clip(seeds_asr, 0, 100)

            # 转换为 counts (/1000)
            success_counts = [round(s * 10) for s in seeds_asr]
            collision_counts = [round((100 - s) * 10 * 0.8) for s in seeds_asr]
            power_counts = [round((100 - s) * 10 * 0.15) for s in seeds_asr]
            stepover_counts = [round((100 - s) * 10 * 0.05) for s in seeds_asr]

            data[model][level] = {
                'success': success_counts,
                'collision': collision_counts,
                'power': power_counts,
                'stepover': stepover_counts,
                'asr': list(seeds_asr),
                'acr': [c / 10.0 for c in collision_counts],
                'alr': [(p + o) / 10.0 for p, o in zip(power_counts, stepover_counts)],
            }

    return data


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Statistical Significance Test for MTrans-SAC vs Baselines")
    parser.add_argument("--csv", type=str, default="baseline_benchmark_results.csv",
                        help="基准测试 CSV 文件路径")
    parser.add_argument("--reference", type=str, default="SAC",
                        help="参考模型名称（默认: SAC = MTrans-SAC Full）")
    parser.add_argument("--level", type=int, default=None,
                        help="仅测试指定难度等级")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="显著性水平 (默认: 0.05)")
    parser.add_argument("--output", type=str, default="p_values.csv",
                        help="输出 CSV 文件路径")
    parser.add_argument("--demo", action="store_true",
                        help="使用内置示例数据运行（无需CSV文件）")
    args = parser.parse_args()

    print("=" * 95)
    print("  STATISTICAL SIGNIFICANCE TEST (Welch's t-test)")
    print("  Reference Model: MTrans-SAC Full  |  α = {:.3f}".format(args.alpha))
    print("=" * 95)

    # 加载数据
    if args.demo:
        data = manual_data_example()
        args.reference = "MTrans-SAC"
    else:
        csv_path = args.csv
        if not os.path.isfile(csv_path):
            print(f"[ERROR] CSV 文件不存在: {csv_path}")
            print("[INFO] 可尝试 --demo 使用内置示例数据")
            sys.exit(1)
        print(f"\n[INFO] 读取数据: {csv_path}")
        data = parse_benchmark_csv(csv_path)

    # 显示可用模型
    print(f"\n[INFO] 可用模型 ({len(data)}): {', '.join(data.keys())}")

    # 确定参考模型名
    ref_name = args.reference
    if ref_name not in data:
        # 尝试模糊匹配
        candidates = [m for m in data.keys() if ref_name.lower() in m.lower()]
        if candidates:
            ref_name = candidates[0]
            print(f"[INFO] 自动匹配参考模型: {ref_name}")
        else:
            print(f"[ERROR] 参考模型 '{args.reference}' 不在数据中")
            sys.exit(1)

    # 选择测试等级
    levels = [args.level] if args.level else [5, 8, 10, 15]

    # 运行检验
    results = run_all_tests(data, reference_model=ref_name, levels=levels, alpha=args.alpha)

    if not results:
        print("\n[WARN] 没有可进行显著性检验的数据对")
        return

    # 输出结果
    print_results(results, alpha=args.alpha)

    # 保存
    save_csv(results, args.output)

    # 额外: 两两对比矩阵
    print("\n" + "=" * 95)
    print("  Pairwise Significance Matrix (ξ=10, α=0.05)")
    print("  ✓ = significant difference  |  ✗ = not significant")
    print("=" * 95)

    models_at_level10 = [m for m in data.keys() if 10 in data[m]]
    print(f"  {'':>20}", end="")
    for m in sorted(models_at_level10):
        print(f" {m[:12]:>12}", end="")
    print()
    for m1 in sorted(models_at_level10):
        print(f"  {m1:>20}", end="")
        for m2 in sorted(models_at_level10):
            if m1 == m2:
                print(f" {'---':>12}", end="")
                continue
            if 10 in data[m1] and 10 in data[m2]:
                asr1 = data[m1][10]['asr']
                asr2 = data[m2][10]['asr']
                if len(asr1) >= 2 and len(asr2) >= 2:
                    r = welch_ttest(asr1, asr2, args.alpha)
                    print(f" {'✓' if r['significant'] else '✗':>12}", end="")
                else:
                    print(f" {'N/A':>12}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
