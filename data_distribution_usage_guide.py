#!/usr/bin/env python3
"""
数据分布可视化使用示例
"""

print("visualize_data_distribution.py 使用指南")
print("="*60)

print("\n1. 基本用法：")
print("python visualize_data_distribution.py --dataset cifar10 --num_clients 20 --alpha 0.5")

print("\n2. 不同参数示例：")

print("\n# 更不均衡的分布 (alpha越小越不均衡)")
print("python visualize_data_distribution.py --dataset cifar10 --num_clients 20 --alpha 0.1")

print("\n# 轻度不均衡")  
print("python visualize_data_distribution.py --dataset cifar10 --num_clients 20 --alpha 1.0")

print("\n# 不同数据集")
print("python visualize_data_distribution.py --dataset mnist --num_clients 10 --alpha 0.3")
print("python visualize_data_distribution.py --dataset cifar100 --num_clients 50 --alpha 0.2")
print("python visualize_data_distribution.py --dataset pathmnist --num_clients 15 --alpha 0.5")

print("\n# 不同客户端数量")
print("python visualize_data_distribution.py --dataset cifar10 --num_clients 10 --alpha 0.5")
print("python visualize_data_distribution.py --dataset cifar10 --num_clients 50 --alpha 0.5")

print("\n3. 输出文件：")
print("结果会保存在: ./results/data_distribution/{dataset}_alpha{alpha}/")
print("  - data_distribution_heatmap.png  # 热力图")
print("  - data_distribution_bar.png     # 堆叠条形图")  
print("  - data_distribution_pie.png     # 客户端饼图")
print("  - distribution_stats.json       # 统计数据")

print("\n4. alpha参数说明：")
print("  - alpha = 10.0   # 接近IID (均匀分布)")
print("  - alpha = 1.0    # 轻度Non-IID")
print("  - alpha = 0.5    # 中度Non-IID")
print("  - alpha = 0.1    # 重度Non-IID (极不均衡)")
print("  - alpha = 0.01   # 极重度Non-IID")

print("\n5. 注意事项：")
print("  - 该脚本独立运行，不依赖训练结果")
print("  - 用于验证数据分布是否符合Non-IID设定")
print("  - 可以在训练前验证数据划分效果")

print("\n" + "="*60)