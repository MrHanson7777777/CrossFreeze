"""
类别测试结果可视化与基线比较脚本

用于可视化实验的 per-class 准确率，支持单实验可视化或多实验对比

使用方法:
    # 可视化单个实验结果 (推荐)
    python visualize_class_accuracy.py --exp_dir ./results/ablation_D_cifar10_noniid_xxx
    
    # 对比两个实验
    python visualize_class_accuracy.py --exp_dir ./results/crossfreeze_xxx --baseline_dir ./results/baseline_xxx
    
    # 使用演示数据
    python visualize_class_accuracy.py --dataset cifar10 --demo
"""
import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from config import get_dataset_config


def load_experiment_results(result_dir):
    """
    加载实验结果，支持多种结果文件格式
    
    Args:
        result_dir: 结果目录
        
    Returns:
        results: 结果字典，包含 per_class_accuracy 等
    """
    results = {}
    
    # 尝试加载 metrics.json
    metrics_path = os.path.join(result_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)
    
    # 尝试加载 results.json (可能包含 per-class 准确率)
    results_path = os.path.join(result_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results['summary'] = json.load(f)
    
    # 尝试加载 per_class_accuracy.json
    per_class_path = os.path.join(result_dir, 'per_class_accuracy.json')
    if os.path.exists(per_class_path):
        with open(per_class_path, 'r') as f:
            results['per_class'] = json.load(f)
    
    if not results:
        print(f"警告: 在 {result_dir} 中未找到任何结果文件")
        return None
    
    return results


def extract_per_class_accuracy(results, num_classes):
    """
    从实验结果中提取 per-class 准确率
    
    Args:
        results: load_experiment_results 返回的结果字典
        num_classes: 类别数量
        
    Returns:
        per_class_acc: 各类别准确率列表，如果无法提取则返回 None
    """
    if results is None:
        return None
    
    # 优先从 per_class 字段提取 (独立的 per_class_accuracy.json)
    if 'per_class' in results:
        pc = results['per_class']
        if isinstance(pc, dict) and 'accuracy' in pc:
            return pc['accuracy']
        if isinstance(pc, dict) and 'per_class_accuracy' in pc:
            return pc['per_class_accuracy']
        elif isinstance(pc, list):
            return pc
    
    # 从 summary 中提取
    if 'summary' in results:
        summary = results['summary']
        if 'per_class_accuracy' in summary:
            return summary['per_class_accuracy']
        if 'class_accuracy' in summary:
            return summary['class_accuracy']
    
    # 从 metrics.json 的新格式中提取
    if 'metrics' in results:
        metrics = results['metrics']
        
        # CrossFreeze 格式: metrics['crossfreeze']['per_class_acc_sm'] 或 per_class_acc_m2
        if 'crossfreeze' in metrics:
            cf = metrics['crossfreeze']
            # 优先返回 Sm (全局模型) 的 per-class 准确率
            if 'per_class_acc_sm' in cf and cf['per_class_acc_sm']:
                return cf['per_class_acc_sm']
            if 'per_class_acc_m2' in cf and cf['per_class_acc_m2']:
                return cf['per_class_acc_m2']
        
        # Baseline 格式: metrics['baseline']['per_class_acc']
        if 'baseline' in metrics:
            bl = metrics['baseline']
            if 'per_class_acc' in bl and bl['per_class_acc']:
                return bl['per_class_acc']
        
        # 旧格式兼容: 检查是否有 cf_metrics (直接在 metrics 下)
        if 'cf_metrics' in metrics and 'per_class_acc' in metrics['cf_metrics']:
            pc_list = metrics['cf_metrics']['per_class_acc']
            if pc_list:
                return pc_list[-1]  # 返回最后一轮
        
        # 检查通用的 per_class_accuracy
        if 'per_class_accuracy' in metrics:
            pc_list = metrics['per_class_accuracy']
            if pc_list:
                return pc_list[-1] if isinstance(pc_list[0], list) else pc_list
    
    return None


def find_latest_result(base_dir, prefix):
    """
    查找最新的实验结果目录
    
    Args:
        base_dir: 基础目录
        prefix: 目录前缀 (如 'crossfreeze_cifar10' 或 'baseline_fedavg_cifar10')
        
    Returns:
        latest_dir: 最新的结果目录路径
    """
    if not os.path.exists(base_dir):
        return None
    
    matching_dirs = []
    for d in os.listdir(base_dir):
        if d.startswith(prefix):
            full_path = os.path.join(base_dir, d)
            if os.path.isdir(full_path):
                matching_dirs.append((full_path, os.path.getmtime(full_path)))
    
    if not matching_dirs:
        return None
    
    # 按修改时间排序，返回最新的
    matching_dirs.sort(key=lambda x: x[1], reverse=True)
    return matching_dirs[0][0]


def plot_single_experiment_accuracy(per_class_acc, class_names, save_path, 
                                     exp_name="Experiment", title=None):
    """
    绘制单个实验的 per-class 准确率条形图
    
    Args:
        per_class_acc: 各类别准确率列表
        class_names: 类别名称列表
        save_path: 保存路径
        exp_name: 实验名称
        title: 图表标题
    """
    num_classes = len(class_names)
    x = np.arange(num_classes)
    
    fig, ax = plt.subplots(figsize=(max(10, num_classes * 0.8), 6))
    
    # 根据准确率高低设置颜色
    colors = plt.cm.RdYlGn([acc / 100.0 for acc in per_class_acc])
    bars = ax.bar(x, per_class_acc, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title or f'{exp_name} - Per-Class Accuracy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    
    # 添加数值标注
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    # 添加平均值线
    avg_acc = np.mean(per_class_acc)
    ax.axhline(y=avg_acc, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(num_classes - 0.5, avg_acc + 2, f'Avg: {avg_acc:.1f}%', 
            fontsize=10, color='blue', ha='right')
    
    # 添加统计信息框
    stats_text = f'Average: {avg_acc:.1f}%\nMax: {max(per_class_acc):.1f}%\nMin: {min(per_class_acc):.1f}%\nStd: {np.std(per_class_acc):.1f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class 准确率图已保存到: {save_path}")


def plot_class_accuracy_comparison(crossfreeze_acc, baseline_acc, class_names, save_path, 
                                    title="Per-Class Accuracy Comparison"):
    """
    绘制类别准确率对比图
    
    Args:
        crossfreeze_acc: CrossFreeze各类别准确率
        baseline_acc: Baseline各类别准确率
        class_names: 类别名称
        save_path: 保存路径
        title: 标题
    """
    num_classes = len(class_names)
    x = np.arange(num_classes)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(max(10, num_classes * 0.8), 6))
    
    bars1 = ax.bar(x - width/2, crossfreeze_acc, width, label='CrossFreeze (Proposed)', color='#2196F3')
    bars2 = ax.bar(x + width/2, baseline_acc, width, label='FedAvg (Baseline)', color='#FF9800')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    
    # 添加数值标注
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=7, rotation=90)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=7, rotation=90)
    
    # 添加平均值线
    avg_cf = np.mean(crossfreeze_acc)
    avg_bl = np.mean(baseline_acc)
    ax.axhline(y=avg_cf, color='#2196F3', linestyle='--', alpha=0.5, label=f'CF Avg: {avg_cf:.1f}%')
    ax.axhline(y=avg_bl, color='#FF9800', linestyle='--', alpha=0.5, label=f'BL Avg: {avg_bl:.1f}%')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存到: {save_path}")


def plot_class_accuracy_radar(crossfreeze_acc, baseline_acc, class_names, save_path,
                               title="Per-Class Accuracy Radar Chart"):
    """
    绘制雷达图对比
    
    Args:
        crossfreeze_acc: CrossFreeze各类别准确率
        baseline_acc: Baseline各类别准确率
        class_names: 类别名称
        save_path: 保存路径
        title: 标题
    """
    num_classes = len(class_names)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
    
    # 闭合图形
    crossfreeze_acc_closed = crossfreeze_acc + [crossfreeze_acc[0]]
    baseline_acc_closed = baseline_acc + [baseline_acc[0]]
    angles_closed = angles + [angles[0]]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    ax.plot(angles_closed, crossfreeze_acc_closed, 'o-', linewidth=2, 
            label='CrossFreeze', color='#2196F3')
    ax.fill(angles_closed, crossfreeze_acc_closed, alpha=0.25, color='#2196F3')
    
    ax.plot(angles_closed, baseline_acc_closed, 'o-', linewidth=2, 
            label='FedAvg', color='#FF9800')
    ax.fill(angles_closed, baseline_acc_closed, alpha=0.25, color='#FF9800')
    
    ax.set_xticks(angles)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 100)
    ax.set_title(title, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"雷达图已保存到: {save_path}")


def plot_improvement_analysis(crossfreeze_acc, baseline_acc, class_names, save_path,
                               title="Accuracy Improvement Analysis"):
    """
    绘制改进分析图
    
    Args:
        crossfreeze_acc: CrossFreeze各类别准确率
        baseline_acc: Baseline各类别准确率
        class_names: 类别名称
        save_path: 保存路径
        title: 标题
    """
    num_classes = len(class_names)
    
    # 计算改进幅度
    improvements = [cf - bl for cf, bl in zip(crossfreeze_acc, baseline_acc)]
    
    # 按改进幅度排序
    sorted_indices = np.argsort(improvements)[::-1]
    sorted_improvements = [improvements[i] for i in sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(max(10, num_classes * 0.6), 6))
    
    colors = ['#4CAF50' if imp >= 0 else '#F44336' for imp in sorted_improvements]
    bars = ax.barh(range(num_classes), sorted_improvements, color=colors)
    
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Accuracy Improvement (percentage points)')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 添加数值标注
    for i, (bar, imp) in enumerate(zip(bars, sorted_improvements)):
        ax.annotate(f'{imp:+.1f}%',
                   xy=(imp, i),
                   xytext=(5 if imp >= 0 else -5, 0),
                   textcoords="offset points",
                   ha='left' if imp >= 0 else 'right',
                   va='center', fontsize=8)
    
    # 添加统计信息
    avg_improvement = np.mean(improvements)
    max_improvement = max(improvements)
    min_improvement = min(improvements)
    
    stats_text = f'Avg: {avg_improvement:+.1f}%\nMax: {max_improvement:+.1f}%\nMin: {min_improvement:+.1f}%'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"改进分析图已保存到: {save_path}")


def generate_comparison_report(crossfreeze_acc, baseline_acc, class_names, save_dir, dataset_name):
    """
    生成完整的对比分析报告
    
    Args:
        crossfreeze_acc: CrossFreeze各类别准确率
        baseline_acc: Baseline各类别准确率
        class_names: 类别名称
        save_dir: 保存目录
        dataset_name: 数据集名称
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算统计指标
    avg_cf = np.mean(crossfreeze_acc)
    avg_bl = np.mean(baseline_acc)
    improvements = [cf - bl for cf, bl in zip(crossfreeze_acc, baseline_acc)]
    
    print(f"\n{'='*60}")
    print(f"类别准确率对比报告 ({dataset_name.upper()})")
    print(f"{'='*60}")
    print(f"\nCrossFreeze (Proposed):")
    print(f"  - 平均准确率: {avg_cf:.2f}%")
    print(f"  - 最高: {max(crossfreeze_acc):.2f}% ({class_names[np.argmax(crossfreeze_acc)]})")
    print(f"  - 最低: {min(crossfreeze_acc):.2f}% ({class_names[np.argmin(crossfreeze_acc)]})")
    print(f"\nFedAvg (Baseline):")
    print(f"  - 平均准确率: {avg_bl:.2f}%")
    print(f"  - 最高: {max(baseline_acc):.2f}% ({class_names[np.argmax(baseline_acc)]})")
    print(f"  - 最低: {min(baseline_acc):.2f}% ({class_names[np.argmin(baseline_acc)]})")
    print(f"\n改进分析:")
    print(f"  - 平均改进: {np.mean(improvements):+.2f}%")
    print(f"  - 最大改进: {max(improvements):+.2f}% ({class_names[np.argmax(improvements)]})")
    print(f"  - 最小改进: {min(improvements):+.2f}% ({class_names[np.argmin(improvements)]})")
    print(f"  - 改进类别数: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}")
    print(f"{'='*60}\n")
    
    # 生成可视化
    title_prefix = f"{dataset_name.upper()} Non-IID"
    
    # 1. 条形图对比
    plot_class_accuracy_comparison(
        crossfreeze_acc, baseline_acc, class_names,
        os.path.join(save_dir, 'class_accuracy_comparison.png'),
        title=f"{title_prefix} - Per-Class Accuracy Comparison"
    )
    
    # 2. 雷达图
    plot_class_accuracy_radar(
        crossfreeze_acc, baseline_acc, class_names,
        os.path.join(save_dir, 'class_accuracy_radar.png'),
        title=f"{title_prefix} - Per-Class Accuracy Radar"
    )
    
    # 3. 改进分析
    plot_improvement_analysis(
        crossfreeze_acc, baseline_acc, class_names,
        os.path.join(save_dir, 'accuracy_improvement.png'),
        title=f"{title_prefix} - Accuracy Improvement Analysis"
    )
    
    # 4. 保存数据
    report = {
        'dataset': dataset_name,
        'class_names': class_names,
        'crossfreeze_acc': crossfreeze_acc,
        'baseline_acc': baseline_acc,
        'improvements': improvements,
        'summary': {
            'crossfreeze_avg': avg_cf,
            'baseline_avg': avg_bl,
            'avg_improvement': np.mean(improvements),
            'improved_classes': sum(1 for imp in improvements if imp > 0)
        }
    }
    
    with open(os.path.join(save_dir, 'comparison_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"报告已保存到: {os.path.join(save_dir, 'comparison_report.json')}")


def generate_single_experiment_report(per_class_acc, class_names, save_dir, exp_name, dataset_name):
    """
    生成单个实验的 per-class 准确率报告
    
    Args:
        per_class_acc: 各类别准确率列表
        class_names: 类别名称列表
        save_dir: 保存目录
        exp_name: 实验名称
        dataset_name: 数据集名称
    """
    os.makedirs(save_dir, exist_ok=True)
    
    avg_acc = np.mean(per_class_acc)
    
    print(f"\n{'='*60}")
    print(f"Per-Class 准确率报告 ({exp_name})")
    print(f"{'='*60}")
    print(f"数据集: {dataset_name.upper()}")
    print(f"平均准确率: {avg_acc:.2f}%")
    print(f"最高准确率: {max(per_class_acc):.2f}% ({class_names[np.argmax(per_class_acc)]})")
    print(f"最低准确率: {min(per_class_acc):.2f}% ({class_names[np.argmin(per_class_acc)]})")
    print(f"标准差: {np.std(per_class_acc):.2f}%")
    print(f"\n各类别准确率:")
    for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
        bar = '█' * int(acc / 5) + '░' * (20 - int(acc / 5))
        print(f"  {name:>12}: {bar} {acc:.1f}%")
    print(f"{'='*60}\n")
    
    # 绘制条形图
    plot_single_experiment_accuracy(
        per_class_acc, class_names,
        os.path.join(save_dir, 'per_class_accuracy.png'),
        exp_name=exp_name,
        title=f"{dataset_name.upper()} - {exp_name} Per-Class Accuracy"
    )
    
    # 保存数据
    report = {
        'experiment': exp_name,
        'dataset': dataset_name,
        'class_names': class_names,
        'per_class_accuracy': per_class_acc,
        'summary': {
            'average': float(avg_acc),
            'max': float(max(per_class_acc)),
            'min': float(min(per_class_acc)),
            'std': float(np.std(per_class_acc))
        }
    }
    
    report_path = os.path.join(save_dir, 'per_class_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"报告已保存到: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='类别准确率可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 可视化单个实验结果 (推荐)
  python visualize_class_accuracy.py --exp_dir ./results/ablation_D_cifar10_noniid_xxx

  # 对比两个实验
  python visualize_class_accuracy.py --exp_dir ./results/crossfreeze_xxx --baseline_dir ./results/baseline_xxx

  # 使用演示数据
  python visualize_class_accuracy.py --dataset cifar10 --demo
        """
    )
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100', 'pathmnist'],
                        help='数据集名称')
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='实验结果目录 (主要实验)')
    parser.add_argument('--baseline_dir', type=str, default=None,
                        help='基线结果目录 (可选，用于对比)')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='结果基础目录')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='保存目录 (默认为实验目录下)')
    parser.add_argument('--demo', action='store_true',
                        help='使用演示数据 (无需实际实验结果)')
    
    # 保留旧参数名兼容
    parser.add_argument('--crossfreeze_dir', type=str, default=None,
                        help='(兼容旧版) 等同于 --exp_dir')
    
    args = parser.parse_args()
    
    # 兼容旧参数名
    if args.crossfreeze_dir and not args.exp_dir:
        args.exp_dir = args.crossfreeze_dir
    
    # 获取数据集配置
    dataset_config = get_dataset_config(args.dataset)
    num_classes = dataset_config['num_classes']
    class_names = dataset_config.get('class_names', [f'Class {i}' for i in range(num_classes)])
    
    if args.demo:
        # 演示模式：使用模拟数据
        print("=" * 60)
        print("演示模式: 使用模拟数据")
        print("=" * 60)
        np.random.seed(42)
        
        # 生成模拟的准确率数据
        baseline_acc = np.random.uniform(40, 75, num_classes).tolist()
        crossfreeze_acc = [min(100, acc + np.random.uniform(5, 15)) for acc in baseline_acc]
        
        save_dir = os.path.join(args.results_dir, 'comparison_demo', args.dataset)
        generate_comparison_report(crossfreeze_acc, baseline_acc, class_names, save_dir, args.dataset)
        print("\n可视化完成! (演示数据)")
        return
    
    # 检查是否指定了实验目录
    if args.exp_dir is None:
        print("错误: 请指定实验结果目录 --exp_dir")
        print("\n使用示例:")
        print("  python visualize_class_accuracy.py --exp_dir ./results/ablation_D_cifar10_noniid_xxx")
        print("  python visualize_class_accuracy.py --demo  # 查看演示")
        return
    
    # 检查实验目录是否存在
    if not os.path.exists(args.exp_dir):
        print(f"错误: 实验目录不存在: {args.exp_dir}")
        return
    
    # 加载实验结果
    print(f"加载实验结果: {args.exp_dir}")
    results = load_experiment_results(args.exp_dir)
    
    if results is None:
        print("错误: 无法加载实验结果")
        return
    
    # 提取 per-class 准确率
    per_class_acc = extract_per_class_accuracy(results, num_classes)
    
    # 从实验目录名提取实验名称
    exp_name = os.path.basename(args.exp_dir)
    
    # 确定保存目录
    save_dir = args.save_dir or args.exp_dir
    
    if per_class_acc is not None:
        print(f"成功提取 per-class 准确率 ({len(per_class_acc)} 类)")
        
        # 如果有基线目录，进行对比
        if args.baseline_dir and os.path.exists(args.baseline_dir):
            print(f"加载基线结果: {args.baseline_dir}")
            baseline_results = load_experiment_results(args.baseline_dir)
            baseline_acc = extract_per_class_accuracy(baseline_results, num_classes)
            
            if baseline_acc is not None:
                print("生成对比报告...")
                comparison_save_dir = os.path.join(save_dir, 'comparison')
                generate_comparison_report(per_class_acc, baseline_acc, class_names, 
                                          comparison_save_dir, args.dataset)
            else:
                print("警告: 无法从基线结果提取 per-class 准确率，仅生成单实验报告")
                generate_single_experiment_report(per_class_acc, class_names, save_dir, 
                                                  exp_name, args.dataset)
        else:
            # 只有单个实验，生成单实验报告
            generate_single_experiment_report(per_class_acc, class_names, save_dir, 
                                              exp_name, args.dataset)
    else:
        print("警告: 实验结果中未找到 per-class 准确率数据")
        print("\n可能的原因:")
        print("  1. 训练时未记录 per-class 准确率")
        print("  2. 结果文件格式不兼容")
        print("\n解决方案:")
        print("  - 确保训练代码中记录了 per_class_accuracy")
        print("  - 或使用 --demo 参数查看演示效果")
        
        # 显示可用的结果文件
        print(f"\n实验目录中的文件:")
        for f in os.listdir(args.exp_dir):
            print(f"  - {f}")
    
    print("\n可视化完成!")


if __name__ == '__main__':
    main()