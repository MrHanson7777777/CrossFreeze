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
from tqdm import tqdm  # [新增]

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from utils.sampling import stratified_prune
from models.Update import LocalUpdate
from models.test import test_img_local

def evaluate_local_weighted(net, dataset, dict_users, args, w_heads_storage):
    total_acc, total_loss, total_samples = 0, 0, 0
    for idx in dict_users.keys():
        user_idxs = dict_users[idx]
        n_samples = len(user_idxs)
        if idx in w_heads_storage:
            net.fc.load_state_dict(w_heads_storage[idx])
        net.eval()
        acc, loss = test_img_local(net, dataset, args, idx=idx, idxs=user_idxs)
        total_acc += acc * n_samples
        total_loss += loss * n_samples
        total_samples += n_samples
    return total_acc / total_samples, total_loss / total_samples

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.alg = 'fedrep'
    
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"./experiments/{args.alg}_{args.dataset}_{time_str}"
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
        for arg in vars(args): f.write(f"{arg}: {getattr(args, arg)}\n")

    print(f"🧱 Starting FedRep on {args.dataset}...")
    print(f"📂 Saving to {experiment_dir}")

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    net_glob = get_model(args)
    net_glob.train()

    if args.data_usage < 1.0:
        print(f"⚠️ DATA SCARCITY MODE: Trained with {args.data_usage*100}% of global data (Stratified Pruned).")
    
    total_keys = list(net_glob.state_dict().keys())
    w_glob_keys = [k for k in total_keys if 'fc' not in k and 'sm_head' not in k]
    w_heads_storage = {i: copy.deepcopy(net_glob.fc.state_dict()) for i in range(args.num_users)}
    
    # 历史记录列表
    loss_train_hist = []
    loss_test_hist = []
    acc_train_hist = []
    acc_test_hist = []
    
    # 外层进度条
    progress_bar = tqdm(range(args.epochs), desc="Training Progress", ncols=120)
    
    for iter in progress_bar:
        # 统一的学习率调度器
        # 策略：在总轮数的 50% 和 75% 处，将学习率衰减为原来的 1/10
        if iter == int(args.epochs * 0.5) or iter == int(args.epochs * 0.75):
            args.lr *= 0.1
            tqdm.write(f"\n📉 Learning Rate decayed to {args.lr}")
        
        w_body_locals = []
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # 内层进度条
        client_desc = f"Round {iter+1}/{args.epochs} - Training clients"
        client_progress = tqdm(idxs_users, desc=client_desc, leave=False, ncols=80)

        client_weights = []  # 记录每个客户端的数据量权重
        
        for idx in client_progress:
            # 直接使用已经静态截断过的索引，不再进行随机采样
            user_idxs = dict_users_train[idx]
            
            # 传入 user_idxs 即可，因为 dict_users_train 已经被永久修剪过了
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=user_idxs)
            net_local = copy.deepcopy(net_glob)
            net_local.fc.load_state_dict(w_heads_storage[idx])
            
            w_new, loss, _ = local.train(net=net_local.to(args.device), w_glob_keys=w_glob_keys, lr=args.lr)  # <--- [新增]
            
            w_heads_storage[idx] = {k: v for k, v in net_local.fc.state_dict().items()}
            w_upload = {k: w_new[k] for k in w_glob_keys}
            w_body_locals.append(w_upload)
            loss_locals.append(loss)
            client_weights.append(len(user_idxs))  # 记录该客户端的数据量

        if len(w_body_locals) > 0:
            # 加权聚合：根据客户端数据量计算权重
            total_samples = sum(client_weights)
            normalized_weights = [w / total_samples for w in client_weights]
            
            # 初始化全局权重
            w_glob_body = {}
            for k in w_glob_keys:
                w_glob_body[k] = torch.zeros_like(w_body_locals[0][k])
            
            # 加权聚合
            for i, weight in enumerate(normalized_weights):
                for k in w_glob_keys:
                    w_glob_body[k] += weight * w_body_locals[i][k]
            
            net_glob.load_state_dict(w_glob_body, strict=False)

        # 计算训练和测试指标
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train_hist.append(loss_avg)
        
        # 计算训练准确率（个性化）
        acc_train, loss_train_eval = evaluate_local_weighted(net_glob, dataset_train, dict_users_train, args, w_heads_storage)
        acc_train_hist.append(acc_train)

        if (iter + 1) % args.test_freq == 0:
            acc_test, loss_test = evaluate_local_weighted(net_glob, dataset_test, dict_users_test, args, w_heads_storage)
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
    ax3.set_title('Training Accuracy (Personalized)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy (%)')
    ax3.grid(True, alpha=0.3)
    
    # 测试准确率
    ax4.plot(range(len(acc_test_hist)), acc_test_hist, 'm-', linewidth=2)
    ax4.set_title('Test Accuracy (Personalized)', fontsize=12, fontweight='bold')
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
        f.write(f"Final Personalized Train Acc: {acc_train_hist[-1]:.2f}%\n")
        f.write(f"Final Personalized Test Acc: {acc_test_hist[-1]:.2f}%\n")
        f.write(f"Best Personalized Test Acc: {max(acc_test_hist):.2f}%\n")
        f.write(f"Best Personalized Train Acc: {max(acc_train_hist):.2f}%\n")
        
    print(f"🎉 Done!")
    print(f"📊 Final Results - Train Acc: {acc_train_hist[-1]:.2f}% | Test Acc: {acc_test_hist[-1]:.2f}% | Best Test Acc: {max(acc_test_hist):.2f}%")