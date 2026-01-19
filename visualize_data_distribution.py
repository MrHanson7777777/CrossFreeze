"""
Non-IID 数据分布可视化脚本

用于验证联邦学习中各客户端的数据分布是否呈现Non-IID特性

使用方法:
    python visualize_data_distribution.py --dataset cifar10 --num_clients 20 --alpha 0.5
    python visualize_data_distribution.py --dataset cifar10 --num_clients 20 --alpha 0.1  # 更不均衡
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from config import get_dataset_config


def get_client_class_distribution(client_loaders, num_classes):
    """
    统计每个客户端各类别的样本数量
    
    Args:
        client_loaders: 客户端数据加载器列表
        num_classes: 类别数量
        
    Returns:
        distribution: (num_clients, num_classes) 的numpy数组
    """
    # client_loaders 可能是 dict（client_id: DataLoader）
    if isinstance(client_loaders, dict):
        num_clients = len(client_loaders)
        distribution = np.zeros((num_clients, num_classes), dtype=int)
        for client_id, loader in client_loaders.items():
            for data, target in loader:
                if target.dim() > 1:
                    target = target.squeeze()
                for label in target.numpy():
                    if label < num_classes:
                        distribution[client_id, label] += 1
        return distribution
    else:
        # 兼容列表类型
        num_clients = len(client_loaders)
        distribution = np.zeros((num_clients, num_classes), dtype=int)
        for client_id, loader in enumerate(client_loaders):
            for data, target in loader:
                if target.dim() > 1:
                    target = target.squeeze()
                for label in target.numpy():
                    if label < num_classes:
                        distribution[client_id, label] += 1
        return distribution


def plot_data_distribution_heatmap(distribution, class_names, save_path, title="Non-IID Data Distribution"):
    """
    绘制数据分布热力图
    
    Args:
        distribution: (num_clients, num_classes) 的numpy数组
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图表标题
    """
    num_clients, num_classes = distribution.shape
    
    fig, ax = plt.subplots(figsize=(max(12, num_classes * 0.8), max(8, num_clients * 0.4)))
    
    # 归一化到每个客户端的比例
    row_sums = distribution.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    normalized = distribution / row_sums * 100
    
    # 绘制热力图
    im = ax.imshow(normalized, cmap='YlOrRd', aspect='auto')
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Percentage (%)', rotation=-90, va="bottom")
    
    # 设置刻度
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_clients))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels([f'Client {i}' for i in range(num_clients)])
    
    # 添加数值标注
    for i in range(num_clients):
        for j in range(num_classes):
            value = distribution[i, j]
            if value > 0:
                text = ax.text(j, i, f'{value}',
                             ha="center", va="center", color="black" if normalized[i,j] < 50 else "white",
                             fontsize=7)
    
    ax.set_title(title)
    ax.set_xlabel('Class')
    ax.set_ylabel('Client')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"热力图已保存到: {save_path}")


def plot_data_distribution_bar(distribution, class_names, save_path, title="Samples per Client"):
    """
    绘制各客户端样本数量条形图
    
    Args:
        distribution: (num_clients, num_classes) 的numpy数组
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图表标题
    """
    num_clients, num_classes = distribution.shape
    
    fig, ax = plt.subplots(figsize=(max(10, num_clients * 0.5), 6))
    
    # 堆叠条形图
    x = np.arange(num_clients)
    bottom = np.zeros(num_clients)
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for class_id in range(num_classes):
        ax.bar(x, distribution[:, class_id], bottom=bottom, 
               label=class_names[class_id], color=colors[class_id])
        bottom += distribution[:, class_id]
    
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(num_clients)])
    ax.legend(loc='upper right', ncol=min(5, num_classes), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"条形图已保存到: {save_path}")


def plot_class_distribution_per_client(distribution, class_names, save_path, title="Class Distribution per Client"):
    """
    绘制每个客户端的类别分布饼图/环形图
    
    Args:
        distribution: (num_clients, num_classes) 的numpy数组
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图表标题
    """
    num_clients, num_classes = distribution.shape
    
    # 计算子图布局
    cols = min(5, num_clients)
    rows = (num_clients + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten() if num_clients > 1 else [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for client_id in range(num_clients):
        ax = axes[client_id]
        client_dist = distribution[client_id]
        
        # 只显示非零类别
        non_zero_mask = client_dist > 0
        if non_zero_mask.sum() > 0:
            sizes = client_dist[non_zero_mask]
            labels = [class_names[i] for i in range(num_classes) if non_zero_mask[i]]
            client_colors = [colors[i] for i in range(num_classes) if non_zero_mask[i]]
            
            ax.pie(sizes, labels=None, colors=client_colors, autopct='%1.0f%%',
                   pctdistance=0.75, textprops={'fontsize': 6})
        
        ax.set_title(f'Client {client_id}\n({client_dist.sum()} samples)', fontsize=9)
    
    # 隐藏多余的子图
    for idx in range(num_clients, len(axes)):
        axes[idx].axis('off')
    
    # 添加全局图例
    fig.legend(class_names, loc='upper center', ncol=min(10, num_classes), fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    
    plt.suptitle(title, fontsize=12, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"饼图已保存到: {save_path}")


def plot_noniid_summary(distribution, class_names, save_dir, dataset_name, alpha):
    """
    生成完整的Non-IID数据分布可视化报告
    
    Args:
        distribution: (num_clients, num_classes) 的numpy数组
        class_names: 类别名称列表
        save_dir: 保存目录
        dataset_name: 数据集名称
        alpha: Dirichlet参数
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_clients, num_classes = distribution.shape
    
    # 计算统计指标
    samples_per_client = distribution.sum(axis=1)
    classes_per_client = (distribution > 0).sum(axis=1)
    
    print(f"\n{'='*60}")
    print(f"Non-IID 数据分布统计 ({dataset_name}, alpha={alpha})")
    print(f"{'='*60}")
    print(f"客户端数量: {num_clients}")
    print(f"类别数量: {num_classes}")
    print(f"总样本数: {distribution.sum()}")
    print(f"\n每客户端样本数:")
    print(f"  - 最小: {samples_per_client.min()}")
    print(f"  - 最大: {samples_per_client.max()}")
    print(f"  - 平均: {samples_per_client.mean():.1f}")
    print(f"  - 标准差: {samples_per_client.std():.1f}")
    print(f"\n每客户端类别数:")
    print(f"  - 最小: {classes_per_client.min()}")
    print(f"  - 最大: {classes_per_client.max()}")
    print(f"  - 平均: {classes_per_client.mean():.1f}")
    print(f"{'='*60}\n")
    
    # 生成可视化图表
    title_prefix = f"{dataset_name.upper()} Non-IID (alpha={alpha})"
    
    # 1. 热力图
    plot_data_distribution_heatmap(
        distribution, class_names,
        os.path.join(save_dir, 'data_distribution_heatmap.png'),
        title=f"{title_prefix} - Client-Class Distribution"
    )
    
    # 2. 堆叠条形图
    plot_data_distribution_bar(
        distribution, class_names,
        os.path.join(save_dir, 'data_distribution_bar.png'),
        title=f"{title_prefix} - Samples per Client"
    )
    
    # 3. 饼图
    plot_class_distribution_per_client(
        distribution, class_names,
        os.path.join(save_dir, 'data_distribution_pie.png'),
        title=f"{title_prefix} - Class Distribution per Client"
    )
    
    # 4. 保存统计数据
    stats = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'total_samples': int(distribution.sum()),
        'samples_per_client': samples_per_client.tolist(),
        'classes_per_client': classes_per_client.tolist(),
        'distribution': distribution.tolist()
    }
    
    import json
    with open(os.path.join(save_dir, 'distribution_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"统计数据已保存到: {os.path.join(save_dir, 'distribution_stats.json')}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Non-IID数据分布可视化')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100', 'pathmnist'],
                        help='数据集名称')
    parser.add_argument('--num_clients', type=int, default=20,
                        help='客户端数量')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet参数 (越小越不均衡)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录')
    parser.add_argument('--save_dir', type=str, default='./results/data_distribution',
                        help='保存目录')
    
    args = parser.parse_args()
    
    # 获取数据集配置
    dataset_config = get_dataset_config(args.dataset)
    num_classes = dataset_config['num_classes']
    class_names = dataset_config.get('class_names', [f'Class {i}' for i in range(num_classes)])
    
    print(f"加载 {args.dataset} 数据集 (Non-IID, alpha={args.alpha})...")
    
    # 加载数据
    from data.data_loader import get_client_dataloaders
    client_loaders, test_loader = get_client_dataloaders(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        iid=False,  # Non-IID
        alpha=args.alpha,
        data_dir=args.data_dir
    )
    
    # 统计分布
    print("统计各客户端数据分布...")
    distribution = get_client_class_distribution(client_loaders, num_classes)
    
    # 生成可视化
    save_dir = os.path.join(args.save_dir, f"{args.dataset}_alpha{args.alpha}")
    plot_noniid_summary(distribution, class_names, save_dir, args.dataset, args.alpha)
    
    print("\n可视化完成!")


if __name__ == '__main__':
    main()
