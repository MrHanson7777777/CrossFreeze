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
from models.Update import LocalUpdate, LocalUpdateDitto
from models.test import test_img_local

# 评估函数 (复用标准评估逻辑)
def evaluate_local_weighted(net, dataset, dict_users, args, w_locals_storage):
    total_acc, total_loss, total_samples = 0, 0, 0
    for idx in dict_users.keys():
        user_idxs = dict_users[idx]
        n_samples = len(user_idxs)
        
        # Ditto 评估的是个性化模型 v_k
        if idx in w_locals_storage:
            net.load_state_dict(w_locals_storage[idx])
        else:
            # 如果该用户还没训练过，暂时使用全局模型 (Cold Start)
            pass 
            
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
    args.alg = 'ditto'
    
    # 初始化实验记录
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"./experiments/{args.alg}_{args.dataset}_{time_str}"
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
        for arg in vars(args): f.write(f"{arg}: {getattr(args, arg)}\n")

    print(f"👯 Starting Ditto on {args.dataset}...")
    print(f"   Lambda: {args.lam_ditto}")
    print(f"📂 Saving to {experiment_dir}")

    # 1. 加载数据
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    
    # 2. 初始化全局模型
    net_glob = get_model(args)
    net_glob.train()

    # 3. 应用数据稀缺截断 (Stratified Pruning)
    if args.data_usage < 1.0:
        print(f"⚠️ DATA SCARCITY MODE: Trained with {args.data_usage*100}% of global data (Stratified Pruned).")
    
    # 4. 初始化 Ditto 的存储结构
    # w_locals_storage 存储每个客户端的个性化模型 v_k
    w_locals_storage = {i: copy.deepcopy(net_glob.state_dict()) for i in range(args.num_users)}
    
    # 历史记录
    loss_train_hist, loss_test_hist = [], []
    acc_train_hist, acc_test_hist = [], []

    progress_bar = tqdm(range(args.epochs), desc="Training Progress", ncols=120)

    for iter in progress_bar:
        # 学习率衰减
        if iter == int(args.epochs * 0.5) or iter == int(args.epochs * 0.75):
            args.lr *= 0.1
            tqdm.write(f"\n📉 Learning Rate decayed to {args.lr}")

        loss_locals = []
        w_glob_updates = []
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        client_desc = f"Round {iter+1}/{args.epochs} - Training clients"
        client_progress = tqdm(idxs_users, desc=client_desc, leave=False, ncols=80)

        # 当前轮次的全局模型参数 (w^t)
        w_glob_current = net_glob.state_dict()

        for idx in client_progress:
            user_idxs = dict_users_train[idx]
            
            # === Task 1: 全局模型更新 (Standard FedAvg Step) ===
            # 目标：计算 w_k^{t+1} 用于聚合
            local_fedavg = LocalUpdate(args=args, dataset=dataset_train, idxs=user_idxs)
            net_temp = copy.deepcopy(net_glob) # 加载 w^t
            
            # 普通训练，不涉及 w_glob_keys 过滤（除非是 FedRep，这里假设 Ditto 全量更新）
            w_new_glob, loss, _ = local_fedavg.train(net=net_temp.to(args.device), w_glob_keys=[], lr=args.lr)
            w_glob_updates.append(w_new_glob)
            loss_locals.append(loss)
            
            # === Task 2: 个性化模型更新 (Ditto Step) ===
            # 目标：更新 v_k，使其在本地数据上 loss 最小，同时接近 w^t
            local_ditto = LocalUpdateDitto(args=args, dataset=dataset_train, idxs=user_idxs)
            
            net_local = copy.deepcopy(net_glob) 
            net_local.load_state_dict(w_locals_storage[idx]) # 加载上一次的 v_k
            
            # 训练 v_k，传入全局模型 w^t 作为正则项锚点 (w_ditto)
            w_new_local, _, _ = local_ditto.train(
                net=net_local.to(args.device), 
                w_ditto=w_glob_current, # 锚点是当前全局模型
                lam=args.lam_ditto,     # 正则系数
                lr=args.lr
            )
            
            # 保存更新后的 v_k
            w_locals_storage[idx] = copy.deepcopy(w_new_local)

        # === 全局聚合 ===
        w_glob_avg = copy.deepcopy(w_glob_updates[0])
        for k in w_glob_avg.keys():
            for i in range(1, len(w_glob_updates)):
                w_glob_avg[k] += w_glob_updates[i][k]
            w_glob_avg[k] = torch.div(w_glob_avg[k], len(w_glob_updates))
        net_glob.load_state_dict(w_glob_avg)

        # === 评估 (基于个性化模型 v_k) ===
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train_hist.append(loss_avg)
        
        acc_train, _ = evaluate_local_weighted(net_glob, dataset_train, dict_users_train, args, w_locals_storage)
        acc_train_hist.append(acc_train)

        if (iter + 1) % args.test_freq == 0:
            acc_test, loss_test = evaluate_local_weighted(net_glob, dataset_test, dict_users_test, args, w_locals_storage)
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
            progress_bar.set_postfix({'TrLoss': f'{loss_avg:.4f}', 'TrAcc': f'{acc_train:.2f}%'})

    # === 绘图与保存结果 (保持格式一致) ===
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    ax1.plot(range(len(loss_train_hist)), loss_train_hist, 'b-'); ax1.set_title('Training Loss')
    ax2.plot(range(len(loss_test_hist)), loss_test_hist, 'r-'); ax2.set_title('Test Loss')
    ax3.plot(range(len(acc_train_hist)), acc_train_hist, 'g-'); ax3.set_title('Training Acc (Personalized)')
    ax4.plot(range(len(acc_test_hist)), acc_test_hist, 'm-'); ax4.set_title('Test Acc (Personalized)')
    
    plt.tight_layout()
    plt.suptitle(f'{args.alg.upper()} - {args.dataset.upper()} Results', y=1.02)
    plt.savefig(os.path.join(experiment_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(experiment_dir, "result.txt"), "w") as f:
        f.write(f"Final Test Acc: {acc_test_hist[-1]:.2f}%\n")
        f.write(f"Best Test Acc: {max(acc_test_hist):.2f}%\n")
    
    print(f"🎉 Done! Final Test Acc: {acc_test_hist[-1]:.2f}%")