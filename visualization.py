"""
可视化工具 (未修改)
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import matplotlib
from matplotlib import font_manager, rcParams


# Ensure any dynamic text shown on figures is ASCII-only (strip non-ASCII)
def _to_ascii(text: str, fallback: str = "") -> str:
    s = ''.join(ch for ch in str(text) if ord(ch) < 128).strip()
    return s if s else fallback


def _ensure_cjk_font():
    """确保 Matplotlib 可用中文字体，避免 DejaVu Sans 缺字警告。

    策略：
    1) 在已安装字体中查找常见中文字体。
    2) 若找不到，则尝试从项目 ./fonts 目录动态注册若干常见字体文件。
    3) 若仍失败，则保留默认字体，但提示可能出现缺字。
    """
    preferred_names = [
        'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK',
        'Source Han Sans SC', 'Source Han Sans CN', 'WenQuanYi Micro Hei',
        'Arial Unicode MS'
    ]

    # 1) 尝试使用系统已安装字体
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred_names:
        if name in installed:
            rcParams['font.sans-serif'] = [name]
            rcParams['font.family'] = 'sans-serif'
            rcParams['axes.unicode_minus'] = False
            return

    # 2) 尝试从本地 fonts/ 目录加载常见字体文件
    repo_root = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(repo_root, 'fonts')
    candidate_files = [
        # Noto / 思源系列（按需自行放置其中之一到 fonts/ 目录）
        'NotoSansCJK-Regular.ttc',
        'NotoSansCJKsc-Regular.otf',
        'SourceHanSansSC-Regular.otf',
        'SourceHanSansCN-Regular.otf',
        'SimHei.ttf',
        'MSYH.TTC',  # Microsoft YaHei
    ]
    loaded_name = None
    if os.path.isdir(fonts_dir):
        for fn in candidate_files:
            fp = os.path.join(fonts_dir, fn)
            if os.path.exists(fp):
                try:
                    font_manager.fontManager.addfont(fp)
                    # 重新构建字体列表后再检测一次可用字体名
                    font_manager._rebuild()
                    installed = {f.name for f in font_manager.fontManager.ttflist}
                    for name in preferred_names:
                        if name in installed:
                            loaded_name = name
                            break
                    # 某些字体内部名称可能不同，若未匹配到则尝试用文件名指定
                    if loaded_name is None:
                        # 退一步使用 matplotlib 自动识别的家族名
                        loaded_name = font_manager.FontProperties(fname=fp).get_name()
                    if loaded_name:
                        rcParams['font.sans-serif'] = [loaded_name]
                        rcParams['font.family'] = 'sans-serif'
                        rcParams['axes.unicode_minus'] = False
                        return
                except Exception:
                    pass

    # 3) 仍未找到：保留默认字体（可能继续出现缺字警告）
    rcParams['axes.unicode_minus'] = False


# 全局初始化：风格与字体
_ensure_cjk_font()
sns.set_style("whitegrid")



def plot_accuracy_curve(metrics_recorder, save_path=None, title="Test Accuracy"):
    """Plot accuracy curve"""
    plt.figure(figsize=(10, 6))
    
    rounds = metrics_recorder.metrics['round']
    accuracy = metrics_recorder.metrics['test_accuracy']
    
    plt.plot(rounds, accuracy, marker='o', linewidth=2, markersize=4)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(_to_ascii(title) or "Test Accuracy", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.close()


def plot_loss_curve(metrics_recorder, save_path=None, title="Loss Curve"):
    """Plot loss curve"""
    plt.figure(figsize=(10, 6))
    
    rounds = metrics_recorder.metrics['round']
    
    # 适配 CrossFreeze
    if 'train_loss_s1' in metrics_recorder.metrics and len(metrics_recorder.metrics['train_loss_s1']) > 0:
        plt.plot(rounds, metrics_recorder.metrics['train_loss_s1'], 
                label='Train Loss S1 (M1+M2)', marker='o', linewidth=2, markersize=4, linestyle=':')
    
    if 'train_loss_s2' in metrics_recorder.metrics and len(metrics_recorder.metrics['train_loss_s2']) > 0:
        plt.plot(rounds, metrics_recorder.metrics['train_loss_s2'], 
                label='Train Loss S2 (Sm)', marker='x', linewidth=2, markersize=4, linestyle='--')

    if 'train_loss_even' in metrics_recorder.metrics and len(metrics_recorder.metrics['train_loss_even']) > 0:
        plt.plot(rounds, metrics_recorder.metrics['train_loss_even'], 
                label='Train Loss Even (M1)', marker='s', linewidth=2, markersize=4, linestyle='-.')

    # 兼容旧的 train_loss
    if 'train_loss' in metrics_recorder.metrics and len(metrics_recorder.metrics['train_loss']) > 0:
        plt.plot(rounds, metrics_recorder.metrics['train_loss'], 
                label='Train Loss (Legacy)', marker='o', linewidth=2, markersize=4)
    
    if len(metrics_recorder.metrics['test_loss']) > 0:
        plt.plot(rounds, metrics_recorder.metrics['test_loss'],
                label='Test Loss', marker='d', linewidth=2, markersize=4, color='black')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(_to_ascii(title) or "Loss Curve", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.close()


def plot_comparison_accuracy(comparison_metrics, save_path=None):
    """Plot accuracy comparison for multiple experiments"""
    plt.figure(figsize=(12, 6))
    
    for name, recorder in comparison_metrics.experiments.items():
        rounds = recorder.metrics['round']
        accuracy = recorder.metrics['test_accuracy']
        label = _to_ascii(name) or 'Experiment'
        plt.plot(rounds, accuracy, marker='o', linewidth=2, 
                markersize=4, label=label, alpha=0.8)
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.close()


def plot_comparison_communication(comparison_metrics, save_path=None):
    """Plot total communication cost comparison"""
    plt.figure(figsize=(10, 6))
    
    names = []
    costs = []
    
    for name, recorder in comparison_metrics.experiments.items():
        if len(recorder.metrics['communication_cost']) > 0:
            names.append(_to_ascii(name) or 'Experiment')
            total_cost = sum(recorder.metrics['communication_cost'])
            costs.append(total_cost)
    
    if len(names) > 0:
        bars = plt.bar(names, costs, alpha=0.7, edgecolor='black')
        
        # Value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2e}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Experiment', fontsize=12)
        plt.ylabel('Total Communication Cost (num parameters)', fontsize=12)
        plt.title('Communication Cost Comparison', fontsize=14)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.close()


def plot_compression_analysis(compression_rates, accuracies, communication_costs, 
                              save_path=None):
    """Plot compression-rate analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy vs Compression Rate
    ax1.plot(compression_rates, accuracies, marker='o', 
            linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Compression Rate', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs Compression Rate', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Communication Cost vs Compression Rate
    ax2.plot(compression_rates, communication_costs, marker='s',
            linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Compression Rate', fontsize=12)
    ax2.set_ylabel('Total Communication Cost (num parameters)', fontsize=12)
    ax2.set_title('Communication Cost vs Compression Rate', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.close()


def plot_accuracy_curves(metrics_recorder, save_path=None, title="Accuracy Curves"):
    """绘制平均训练和测试准确率曲线"""
    plt.figure(figsize=(12, 8))
    
    rounds = metrics_recorder.metrics['round']
    
    # 检查是否有 CrossFreeze 数据
    if hasattr(metrics_recorder, 'cf_metrics') and len(metrics_recorder.cf_metrics.get('test_acc_m2', [])) > 0:
        # M1+M2 曲线
        train_m2 = metrics_recorder.cf_metrics['train_acc_m2']
        test_m2 = metrics_recorder.cf_metrics['test_acc_m2']
        plt.plot(rounds, train_m2, marker='o', linewidth=2, markersize=4, 
                label='M1+M2 Train (Personalized)', color='#1f77b4', linestyle='-')
        plt.plot(rounds, test_m2, marker='s', linewidth=2, markersize=4, 
                label='M1+M2 Test (Personalized)', color='#1f77b4', linestyle='--')
        
        # M1+Sm 曲线
        train_sm = metrics_recorder.cf_metrics['train_acc_sm']
        test_sm = metrics_recorder.cf_metrics['test_acc_sm']
        plt.plot(rounds, train_sm, marker='^', linewidth=2, markersize=4, 
                label='M1+Sm Train (Global)', color='#ff7f0e', linestyle='-')
        plt.plot(rounds, test_sm, marker='v', linewidth=2, markersize=4, 
                label='M1+Sm Test (Global)', color='#ff7f0e', linestyle='--')
    # 兼容旧的 client_metrics 结构
    elif hasattr(metrics_recorder, 'client_metrics') and len(metrics_recorder.client_metrics.get('weighted_avg_test_m2', [])) > 0:
        # M1+M2 曲线
        train_m2 = metrics_recorder.client_metrics['weighted_avg_train_m2']
        test_m2 = metrics_recorder.client_metrics['weighted_avg_test_m2']
        plt.plot(rounds, train_m2, marker='o', linewidth=2, markersize=4, 
                label='M1+M2 Train (Personalized)', color='#1f77b4', linestyle='-')
        plt.plot(rounds, test_m2, marker='s', linewidth=2, markersize=4, 
                label='M1+M2 Test (Personalized)', color='#1f77b4', linestyle='--')
        
        # M1+Sm 曲线
        train_sm = metrics_recorder.client_metrics['weighted_avg_train_sm']
        test_sm = metrics_recorder.client_metrics['weighted_avg_test_sm']
        plt.plot(rounds, train_sm, marker='^', linewidth=2, markersize=4, 
                label='M1+Sm Train (Global)', color='#ff7f0e', linestyle='-')
        plt.plot(rounds, test_sm, marker='v', linewidth=2, markersize=4, 
                label='M1+Sm Test (Global)', color='#ff7f0e', linestyle='--')
    else:
        # 回退到基本准确率曲线
        test_accuracy = metrics_recorder.metrics.get('test_accuracy', [])
        if test_accuracy:
            plt.plot(rounds, test_accuracy, marker='o', linewidth=2, markersize=4, 
                    label='Test Accuracy', color='#1f77b4')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(_to_ascii(title) or "Accuracy Curves", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy curves saved: {save_path}")
    
    plt.close()


def plot_random_client_curves(metrics_recorder, save_path=None, title="Random Client Accuracy Curves"):
    """随机选择一个客户端绘制其训练和测试准确率曲线"""
    if not hasattr(metrics_recorder, 'client_metrics'):
        print("没有客户端详细数据，跳过个体客户端曲线绘制")
        return
        
    client_test_m2_data = metrics_recorder.client_metrics.get('client_test_m2_per_round', [])
    client_train_m2_data = metrics_recorder.client_metrics.get('client_train_m2_per_round', [])
    client_test_sm_data = metrics_recorder.client_metrics.get('client_test_sm_per_round', [])
    client_train_sm_data = metrics_recorder.client_metrics.get('client_train_sm_per_round', [])
    
    if not client_test_m2_data:
        print("没有客户端数据，跳过个体客户端曲线绘制")
        return
    
    # 随机选择一个客户端
    num_clients = len(client_test_m2_data[0]) if client_test_m2_data else 0
    if num_clients == 0:
        print("客户端数据为空，跳过个体客户端曲线绘制")
        return
        
    import random
    selected_client = random.randint(0, num_clients - 1)
    
    plt.figure(figsize=(12, 8))
    
    rounds = metrics_recorder.metrics['round']
    
    # 提取选中客户端的数据
    client_train_m2 = [round_data[selected_client] for round_data in client_train_m2_data]
    client_test_m2 = [round_data[selected_client] for round_data in client_test_m2_data]
    client_train_sm = [round_data[selected_client] for round_data in client_train_sm_data]
    client_test_sm = [round_data[selected_client] for round_data in client_test_sm_data]
    
    # 绘制曲线
    plt.plot(rounds, client_train_m2, marker='o', linewidth=2, markersize=4, 
            label=f'Client {selected_client} M1+M2 Train', color='#1f77b4', linestyle='-')
    plt.plot(rounds, client_test_m2, marker='s', linewidth=2, markersize=4, 
            label=f'Client {selected_client} M1+M2 Test', color='#1f77b4', linestyle='--')
    plt.plot(rounds, client_train_sm, marker='^', linewidth=2, markersize=4, 
            label=f'Client {selected_client} M1+Sm Train', color='#ff7f0e', linestyle='-')
    plt.plot(rounds, client_test_sm, marker='v', linewidth=2, markersize=4, 
            label=f'Client {selected_client} M1+Sm Test', color='#ff7f0e', linestyle='--')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(_to_ascii(title) or f"Client {selected_client} Accuracy Curves", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Random client curves saved: {save_path} (Client {selected_client})")
    
    plt.close()
    return selected_client


def plot_all_metrics(metrics_recorder, save_dir, experiment_name):
    """Plot all metric figures"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 平均准确率曲线
    plot_accuracy_curves(
        metrics_recorder,
        save_path=os.path.join(save_dir, f'{experiment_name}_accuracy_curves.png'),
        title=f'{_to_ascii(experiment_name) or "Experiment"} - Accuracy Curves'
    )
    
    # 随机客户端准确率曲线
    selected_client = plot_random_client_curves(
        metrics_recorder,
        save_path=os.path.join(save_dir, f'{experiment_name}_random_client_curves.png'),
        title=f'{_to_ascii(experiment_name) or "Experiment"} - Random Client Curves'
    )
    
    if selected_client is not None:
        print(f"随机选择的客户端: {selected_client}")
    
    # 原有的准确率曲线（兼容性）
    plot_accuracy_curve(
        metrics_recorder,
        save_path=os.path.join(save_dir, f'{experiment_name}_accuracy.png'),
        title=f'{_to_ascii(experiment_name) or "Experiment"} - Test Accuracy'
    )
    
    # Loss curve (已更新以支持 S1/S2/Even)
    plot_loss_curve(
        metrics_recorder,
        save_path=os.path.join(save_dir, f'{experiment_name}_loss.png'),
        title=f'{_to_ascii(experiment_name) or "Experiment"} - Loss Curve'
    )
    
    print(f"All figures saved to: {save_dir}")


def create_comparison_plots(comparison_metrics, save_dir):
    """Create all comparison figures"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Accuracy comparison
    plot_comparison_accuracy(
        comparison_metrics,
        save_path=os.path.join(save_dir, 'comparison_accuracy.png')
    )
    
    # Communication cost comparison
    plot_comparison_communication(
        comparison_metrics,
        save_path=os.path.join(save_dir, 'comparison_communication.png')
    )
    
    print(f"Comparison figures saved to: {save_dir}")