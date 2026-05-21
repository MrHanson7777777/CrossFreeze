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
from models.Update import LocalUpdate
from models.test import test_img_local

# 标准评估函数，直接评估传入的全局模型
def evaluate_local_weighted(net, dataset, dict_users, args):
    net.eval()
    total_acc, total_loss, total_samples = 0, 0, 0
    for idx in dict_users.keys():
        user_idxs = dict_users[idx]
        n_samples = len(user_idxs)
        
        # 原始 FedAvg 直接用全局模型 net 测试，不加载任何本地状态
        acc, loss = test_img_local(net, dataset, args, idx=idx, idxs=user_idxs)
        
        total_acc += acc * n_samples
        total_loss += loss * n_samples
        total_samples += n_samples
    return total_acc / total_samples, total_loss / total_samples

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.alg = 'fedavg'
    
    # 实验初始化
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"./experiments/{args.alg}_{args.dataset}_{time_str}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
        for arg in vars(args): f.write(f"{arg}: {getattr(args, arg)}\n")
            
    print(f"🚀 Starting Standard FedAvg on {args.dataset}...")
    print(f"📂 Results will be saved to: {experiment_dir}")

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    net_glob = get_model(args)
    net_glob.train()
    
    # [建议] 针对 FedAvg 这种容易崩的算法，强制降低 LR
    if args.data_usage <= 0.1:
        print("⚠️ Data Scarcity Detected: Lowering Learning Rate for FedAvg stability.")
        args.lr = 0.001 

    if args.data_usage < 1.0:
        print(f"⚠️ DATA SCARCITY MODE: Trained with {args.data_usage*100}% of global data (Stratified Pruned).")
    
    # 历史记录列表
    loss_train_hist = []
    loss_test_hist = []
    acc_train_hist = []
    acc_test_hist = []
    
    # 外层进度条
    progress_bar = tqdm(range(args.epochs), desc="Training Progress", ncols=120)
    
    for iter in progress_bar:
        # 统一的学习率调度器，在总轮数的 50% 和 75% 处，将学习率衰减为原来的 1/10
        if iter == int(args.epochs * 0.5) or iter == int(args.epochs * 0.75):
            args.lr *= 0.1
            tqdm.write(f"\n📉 Learning Rate decayed to {args.lr}")
        
        # === [修改后] 初始化列表 ===
        w_locals = []
        loss_locals = []
        client_sample_counts = []  # <--- 新增
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # 内层进度条
        client_desc = f"Round {iter+1}/{args.epochs} - Training clients"
        client_progress = tqdm(idxs_users, desc=client_desc, leave=False, ncols=80)

        for idx in client_progress:
            # 直接使用已经静态截断过的索引，不再进行随机采样
            user_idxs = list(dict_users_train[idx])
            
            # 传入 user_idxs 即可，因为 dict_users_train 已经被永久修剪过了
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=user_idxs)
            
            # 标准 FedAvg 发送全局模型副本给客户端
            w, loss, _ = local.train(
                net=copy.deepcopy(net_glob).to(args.device), 
                w_glob_keys=[], # FedAvg 更新所有参数
                lr=args.lr
            )
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            
            # === [新增] 记录该客户端实际参与训练的样本数 ===
            # 注意：user_idxs 是已经经过 m_tr 截断后的真实索引列表
            client_sample_counts.append(len(user_idxs))
            
            # 这里不再保存 w_heads_storage

        # === [修改后] 标准加权聚合 ===
        w_glob = copy.deepcopy(w_locals[0])
        
        # 1. 初始化为全 0
        for k in w_glob.keys():
            w_glob[k] = torch.zeros_like(w_glob[k])
            
        # 2. 计算本轮总样本数
        total_samples = sum(client_sample_counts)
        
        # 3. 加权累加
        for i in range(len(w_locals)):
            # 权重 = 该客户端样本数 / 总样本数
            weight = client_sample_counts[i] / total_samples
            
            for k in w_glob.keys():
                w_glob[k] += w_locals[i][k] * weight
        
        # 加载回全局模型
        net_glob.load_state_dict(w_glob)

        # 评估与记录
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train_hist.append(loss_avg)
        
        # 计算训练集准确率 (使用的是 net_glob 全局模型)
        acc_train, _ = evaluate_local_weighted(net_glob, dataset_train, dict_users_train, args)
        acc_train_hist.append(acc_train)

        # 测试与日志
        if (iter + 1) % args.test_freq == 0:
            # 计算测试集准确率 (使用的是 net_glob 全局模型)
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
            acc_test_hist.append(acc_test_hist[-1] if len(acc_test_hist) > 0 else 0)
            loss_test_hist.append(loss_test_hist[-1] if len(loss_test_hist) > 0 else 0)
            
            progress_bar.set_postfix({
                'TrLoss': f'{loss_avg:.4f}',
                'TrAcc': f'{acc_train:.2f}%'
            })

    # === 绘图与保存 ===
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1.plot(range(len(loss_train_hist)), loss_train_hist, 'b-', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    
    ax2.plot(range(len(loss_test_hist)), loss_test_hist, 'r-', linewidth=2)
    ax2.set_title('Test Loss')
    ax2.set_xlabel('Epochs')
    
    ax3.plot(range(len(acc_train_hist)), acc_train_hist, 'g-', linewidth=2)
    ax3.set_title('Training Accuracy (Global Model)')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy (%)')
    
    ax4.plot(range(len(acc_test_hist)), acc_test_hist, 'm-', linewidth=2)
    ax4.set_title('Test Accuracy (Global Model)')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'training_results.png'))
    plt.close()

    with open(os.path.join(experiment_dir, "result.txt"), "w") as f:
        f.write(f"=== {args.alg.upper()} Standard Results ===\n")
        f.write(f"Final Train Acc: {acc_train_hist[-1]:.2f}%\n")
        f.write(f"Final Test Acc: {acc_test_hist[-1]:.2f}%\n")
        f.write(f"Best Test Acc: {max(acc_test_hist):.2f}%\n")
    
    print(f"🎉 Done! Results saved to {experiment_dir}")
    print(f"📊 Final Results - Train Acc: {acc_train_hist[-1]:.2f}% | Test Acc: {acc_test_hist[-1]:.2f}% | Best Test Acc: {max(acc_test_hist):.2f}%")