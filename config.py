"""
配置文件 (已更新以移除压缩设置)
"""
import argparse
import torch


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CrossFreeze 联邦学习')
    
    # 实验设置
    parser.add_argument('--experiment', type=str, default='crossfreeze',
                        choices=['baseline', 'crossfreeze'],
                        help='实验类型: baseline(基准FedAvg), crossfreeze(交叉冻结)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='运行设备')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID (如 --gpu 0)')
    parser.add_argument('--use_cuda', type=int, default=1, help='是否使用CUDA (1: 是, 0: 否)')
    
    # 数据集设置
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'cifar100', 'pathmnist'],
                        help='数据集')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据集路径 (main.py的相对路径)')
    parser.add_argument('--iid', type=int, default=1,
                        help='是否IID分布 (1: IID, 0: Non-IID)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Non-IID的Dirichlet参数 (越小越不均衡)')
    
    # 联邦学习设置
    parser.add_argument('--num_clients', type=int, default=20,
                        help='客户端总数')
    parser.add_argument('--frac', type=float, default=0.5,
                        help='每轮参与训练的客户端比例')
    parser.add_argument('--epochs', type=int, default=50,
                        help='全局训练轮数')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='本地训练轮数 (用于S1, S2, Even)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--lr_decay_step', type=int, default=20,
                        help='学习率衰减步长 (每N轮衰减一次)')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5,
                        help='学习率衰减率')
    
    # CrossFreeze特有参数
    parser.add_argument('--mu', type=float, default=0.01,
                        help='近端正则化超参数（FedProx），控制M1漂移')
    
    # 【新增】一致性损失权重 Beta
    parser.add_argument('--consistency_beta', type=float, default=5.0,
                        help='M1一致性正则化损失的权重 (建议范围 1.0 - 10.0)')
    
    # 【新增】学习率调度器参数
    parser.add_argument('--min_lr', type=float, default=1e-5, 
                        help='CosineAnnealingLR的最小学习率 (CIFAR建议0.002, PathMNIST建议1e-5)')
    parser.add_argument('--gamma_sm', type=float, default=1.0, 
                        help='S1阶段全局损失的权重 (CIFAR建议2.0, PathMNIST建议0.1)')
    
    # 【新增】Gamma调度策略参数
    parser.add_argument('--gamma_scheduler', type=str, default='static',
                        choices=['static', 'dynamic'],
                        help='Gamma权重策略: static(固定), dynamic(动态Warm-up)')
    
    # 【新增】分离/交替优化模式参数
    parser.add_argument('--separate_loss', type=int, default=0,
                        help='If 1, S1 uses only M2, Even uses only Sm (Alternating Mode)')
    
    # CutMix数据增强参数
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='CutMix的Beta分布参数，控制混合强度 (0.2-2.0)')
    parser.add_argument('--cutmix_prob', type=float, default=0.8,
                        help='CutMix应用概率 (0.0-1.0)')
    parser.add_argument('--cutmix_min_ratio', type=float, default=0.1,
                        help='CutMix最小裁剪比例 (0.05-0.3)')
    parser.add_argument('--cutmix_max_ratio', type=float, default=0.8,
                        help='CutMix最大裁剪比例 (0.7-0.95)')
    parser.add_argument('--use_cutmix', type=int, default=1,
                        help='是否启用CutMix增强 (1: 启用, 0: 禁用)')
    parser.add_argument('--mixup_cutmix_ratio', type=float, default=0.5,
                        help='Mixup和CutMix混合使用的比例 (0.0: 全用Mixup, 1.0: 全用CutMix)')
    
    # 保存和日志
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='结果保存路径')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='日志记录间隔')
    parser.add_argument('--save_model', type=int, default=1,
                        help='是否保存模型 (1: 是, 0: 否)')
    
    # 可视化
    parser.add_argument('--plot', type=int, default=1,
                        help='是否生成可视化图表 (1: 是, 0: 否)')
    
    # 早停机制
    parser.add_argument('--early_stopping', type=int, default=1,
                        help='是否启用早停机制 (1: 启用, 0: 禁用)')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心轮数 (连续多少轮无改善时停止)')
    parser.add_argument('--min_delta', type=float, default=0.01,
                        help='早停最小改善阈值 (百分点, 例如 0.01 表示 0.01%)')
    
    args = parser.parse_args()
    
    # 处理GPU设置
    if args.use_cuda and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
        print(f"使用GPU: {args.device}")
    else:
        args.device = 'cpu'
        print("使用CPU")
    
    return args


# 数据集配置
DATASET_CONFIG = {
    'mnist': {
        'num_classes': 10,
        'input_channels': 1,
        'image_size': 28,
        'mean': [0.1307],
        'std': [0.3081],
        'class_names': [str(i) for i in range(10)]
    },
    'cifar10': {
        'num_classes': 10,
        'input_channels': 3,
        'image_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
        'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    },
    'cifar100': {
        'num_classes': 100,
        'input_channels': 3,
        'image_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761],
        'class_names': [f'Class {i}' for i in range(100)]
    },
    'pathmnist': {
        'num_classes': 9,
        'input_channels': 3,
        'image_size': 28,
        'mean': [0.5],  # MedMNIST 官方推荐的简单归一化
        'std': [0.5],
        'class_names': ['adipose', 'background', 'debris', 'lymphocytes', 
                       'mucus', 'smooth_muscle', 'normal_colon_mucosa',
                       'cancer-associated_stroma', 'colorectal_adenocarcinoma_epithelium']
    }
}


def get_dataset_config(dataset_name):
    """获取数据集配置"""
    return DATASET_CONFIG.get(dataset_name.lower(), DATASET_CONFIG['mnist'])