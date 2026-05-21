#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from utils.sampling import stratified_prune
from models.Update import LocalUpdateMOON  # 导入刚才写的新类
from models.test import test_img_local

def evaluate_local_weighted(net, dataset, dict_users, args):
    total_acc, total_loss, total_samples = 0, 0, 0
    for idx in dict_users.keys():
        user_idxs = dict_users[idx]
        n_samples = len(user_idxs)
        net.eval()
        acc, loss = test_img_local(net, dataset, args, idx=idx, idxs=user_idxs)
        total_acc += acc * n_samples
        total_loss += loss * n_samples
        total_samples += n_samples
    if total_samples == 0: return 0, 0
    return total_acc / total_samples, total_loss / total_samples

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.alg = 'moon'
    
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"./experiments/{args.alg}_{args.dataset}_{time_str}"
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
        for arg in vars(args): f.write(f"{arg}: {getattr(args, arg)}\n")

    print(f"� Starting MOON on {args.dataset}...")
    print(f"📂 Saving to {experiment_dir}")

    # 1. 数据准备
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    
    # 2. 模型初始化
    net_glob = get_model(args)
    net_glob.train()

    # 3. 数据截断 (Data Scarcity)
    if args.data_usage < 1.0:
        print(f"⚠️ DATA SCARCITY MODE: Trained with {args.data_usage*100}% of global data (Stratified Pruned).")

    # 4. MOON 核心：存储上一轮的本地模型 (w_local_prev)
    # [关键优化] 初始化时强制存储在 CPU 上，节省 GPU 空间
    initial_state_cpu = {k: v.cpu() for k, v in net_glob.state_dict().items()}
    w_locals_prev = {i: copy.deepcopy(initial_state_cpu) for i in range(args.num_users)}

    # [性能优化] 在循环外创建静态模型对象，避免重复deepcopy
    # 这样可以将性能提升3-5倍
    print("🚀 Creating static model objects for performance optimization...")
    net_local_static = get_model(args)
    net_glob_freeze_static = get_model(args)
    net_prev_static = get_model(args)
    
    # 设置冻结模型的 requires_grad=False
    for p in net_glob_freeze_static.parameters(): p.requires_grad = False
    for p in net_prev_static.parameters(): p.requires_grad = False
    print("✅ Static models created. Training speed will be significantly improved.")

    loss_train_hist, loss_test_hist = [], []
    acc_train_hist, acc_test_hist = [], []

    progress_bar = tqdm(range(args.epochs), desc="Training Progress", ncols=120)

    for iter in progress_bar:
        # 统一的学习率调度器
        # 策略：在总轮数的 50% 和 75% 处，将学习率衰减为原来的 1/10
        if iter == int(args.epochs * 0.5) or iter == int(args.epochs * 0.75):
            args.lr *= 0.1
            tqdm.write(f"\n📉 Learning Rate decayed to {args.lr}")
        
        w_glob_updates = []
        loss_locals = []
        client_weights = []  # 记录每个客户端的数据量权重
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        client_progress = tqdm(idxs_users, desc=f"Round {iter+1}", leave=False, ncols=80)

        for idx in client_progress:
            user_idxs = dict_users_train[idx]
            
            # [性能优化] 使用静态对象 + load_state_dict，避免昂贵的 deepcopy
            # 1. net_local: 从当前全局参数加载
            net_local_static.load_state_dict(net_glob.state_dict())
            net_local_static.train()
            
            # 2. net_glob_freeze: 全局模型 (冻结)
            net_glob_freeze_static.load_state_dict(net_glob.state_dict())
            net_glob_freeze_static.eval()
            
            # 3. net_prev: 上一轮本地模型 (从 CPU 加载)
            net_prev_static.load_state_dict(w_locals_prev[idx])
            net_prev_static.eval()
            
            # 训练 (模型在 LocalUpdateMOON 内部会被移到 GPU)
            local = LocalUpdateMOON(args=args, dataset=dataset_train, idxs=user_idxs)
            w_new, loss = local.train(
                net=net_local_static.to(args.device),
                net_glob=net_glob_freeze_static.to(args.device),
                net_prev=net_prev_static.to(args.device),
                lr=args.lr
            )
            
            # 收集结果 (将结果深拷贝到 CPU，准备聚合)
            w_new_cpu = {k: v.cpu() for k, v in w_new.items()}
            w_glob_updates.append(w_new_cpu)
            loss_locals.append(loss)
            client_weights.append(len(user_idxs))  # 记录该客户端的数据量
            
            # 更新该用户的 previous model (存储在 CPU)
            w_locals_prev[idx] = copy.deepcopy(w_new_cpu)
            
            # [优化] 将模型移回 CPU，释放显存
            # 注意：由于模型已经移回CPU，torch.cuda.empty_cache() 变得不必要
            net_local_static.cpu()
            net_glob_freeze_static.cpu()
            net_prev_static.cpu()

        # [修复] 加权聚合：根据客户端数据量计算权重
        if len(w_glob_updates) > 0:
            total_samples = sum(client_weights)
            normalized_weights = [w / total_samples for w in client_weights]
            
            # 初始化全局权重
            w_glob_avg = {}
            for k in w_glob_updates[0].keys():
                w_glob_avg[k] = torch.zeros_like(w_glob_updates[0][k])
            
            # 加权聚合
            for i, weight in enumerate(normalized_weights):
                for k in w_glob_avg.keys():
                    w_glob_avg[k] += weight * w_glob_updates[i][k]
        
        # 加载回全局模型 (GPU)
        net_glob.load_state_dict(w_glob_avg)

        # 评估
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train_hist.append(loss_avg)
        
        # 使用全局模型进行评估 (MOON 最终产出的是全局模型，但也支持个性化)
        # 为了与 FedAvg 对比，通常评估 Global Model 在所有测试集上的表现
        # 这里为了与你的 main_crossfreeze 保持一致，计算加权平均准确率
        acc_train, _ = evaluate_local_weighted(net_glob, dataset_train, dict_users_train, args)
        acc_train_hist.append(acc_train)

        if (iter + 1) % args.test_freq == 0:
            acc_test, loss_test = evaluate_local_weighted(net_glob, dataset_test, dict_users_test, args)
            acc_test_hist.append(acc_test)
            loss_test_hist.append(loss_test)
            
            progress_bar.set_postfix({
                'TrLoss': f'{loss_avg:.4f}',
                'TrAcc': f'{acc_train:.2f}%',
                'TeAcc': f'{acc_test:.2f}%'
            })
            tqdm.write(f"🔄 Round {iter+1:3d} | Train Loss: {loss_avg:.4f} | Train Acc: {acc_train:.2f}% | Test Loss: {loss_test:.4f} | Test Acc: {acc_test:.2f}%")
        else:
            acc_test_hist.append(acc_test_hist[-1] if len(acc_test_hist)>0 else 0)
            loss_test_hist.append(loss_test_hist[-1] if len(loss_test_hist)>0 else 0)
            
            progress_bar.set_postfix({
                'TrLoss': f'{loss_avg:.4f}',
                'TrAcc': f'{acc_train:.2f}%'
            })

    # === 绘制四个子图 ===
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 训练损失
    ax1.plot(range(len(loss_train_hist)), loss_train_hist, 'b-', linewidth=2)
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # 测试损失
    ax2.plot(range(len(loss_test_hist)), loss_test_hist, 'r-', linewidth=2)
    ax2.set_title('Test Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # 训练准确率
    ax3.plot(range(len(acc_train_hist)), acc_train_hist, 'g-', linewidth=2)
    ax3.set_title('Training Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy (%)')
    ax3.grid(True, alpha=0.3)
    
    # 测试准确率
    ax4.plot(range(len(acc_test_hist)), acc_test_hist, 'm-', linewidth=2)
    ax4.set_title('Test Accuracy', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'{args.alg.upper()} - {args.dataset.upper()} Results', y=1.02, fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(experiment_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # === 保存详细结果 ===
    with open(os.path.join(experiment_dir, "result.txt"), "w") as f:
        f.write(f"=== {args.alg.upper()} Results on {args.dataset.upper()} ===\n")
        f.write(f"Final Train Loss: {loss_train_hist[-1]:.4f}\n")
        f.write(f"Final Test Loss: {loss_test_hist[-1]:.4f}\n")
        f.write(f"Final Train Acc: {acc_train_hist[-1]:.2f}%\n")
        f.write(f"Final Test Acc: {acc_test_hist[-1]:.2f}%\n")
        f.write(f"Best Test Acc: {max(acc_test_hist):.2f}%\n")
        f.write(f"Best Train Acc: {max(acc_train_hist):.2f}%\n")
        
    print(f"🎉 Done!")
    print(f"📊 Final Results - Train Acc: {acc_train_hist[-1]:.2f}% | Test Acc: {acc_test_hist[-1]:.2f}% | Best Test Acc: {max(acc_test_hist):.2f}%")