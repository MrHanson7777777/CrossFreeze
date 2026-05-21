#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch
import itertools
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

def evaluate_local_weighted(net, dataset, dict_users, args, w_locals_storage):
    total_acc, total_loss, total_samples = 0, 0, 0
    for idx in dict_users.keys():
        user_idxs = dict_users[idx]
        n_samples = len(user_idxs)
        if idx in w_locals_storage:
            net.load_state_dict(w_locals_storage[idx], strict=False)
        net.eval()
        acc, loss = test_img_local(net, dataset, args, idx=idx, idxs=user_idxs)
        total_acc += acc * n_samples
        total_loss += loss * n_samples
        total_samples += n_samples
    return total_acc / total_samples, total_loss / total_samples

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.alg = 'crossfreeze'
    
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"./experiments/{args.alg}_{args.dataset}_{time_str}"
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
        for arg in vars(args): f.write(f"{arg}: {getattr(args, arg)}\n")

    print(f"❄️ Starting CrossFreeze on {args.dataset}...")
    print(f"📂 Saving to {experiment_dir}")

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    net_glob = get_model(args)
    net_glob.train()

    if args.data_usage < 1.0:
        print(f"⚠️ DATA SCARCITY MODE: Trained with {args.data_usage*100}% of global data (Stratified Pruned).")
    
    w_glob_keys = list(itertools.chain.from_iterable(net_glob.weight_keys))
    w_locals_storage = {i: copy.deepcopy(net_glob.state_dict()) for i in range(args.num_users)}
    global_protos = {}

    # === 历史记录列表 ===
    loss_train_hist = []
    loss_test_hist = []
    acc_train_hist = []
    acc_test_hist = []

    # === 外层进度条 ===
    progress_bar = tqdm(range(args.epochs), desc="Training Progress", ncols=120)
    
    # 初始化全局 sm_head 参数
    w_glob_sm = {k: v for k, v in net_glob.state_dict().items() if k in w_glob_keys}

    for iter in progress_bar:
        # === [新增] 统一的学习率调度器 (MultiStepLR) ===
        # 策略：在总轮数的 50% 和 75% 处，将学习率衰减为原来的 1/10
        if iter == int(args.epochs * 0.5) or iter == int(args.epochs * 0.75):
            args.lr *= 0.1
            tqdm.write(f"\n📉 Learning Rate decayed to {args.lr}")
        
        w_sm_heads = []
        loss_locals = []
        client_sample_counts = []  # 统计各个用户的样本数
        collected_protos = {}
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # === 内层进度条 ===
        client_desc = f"Round {iter+1}/{args.epochs} - Training clients"
        client_progress = tqdm(idxs_users, desc=client_desc, leave=False, ncols=80)

        for idx in client_progress:
            # [修改] 直接使用已经静态截断过的索引，不再进行随机采样
            user_idxs = dict_users_train[idx]
            
            # 传入 user_idxs 即可，因为 dict_users_train 已经被永久修剪过了
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=user_idxs)
            net_local = copy.deepcopy(net_glob)
            local_state = w_locals_storage[idx]
            current_state = net_local.state_dict()
            for k, v in local_state.items():
                if 'sm_head' not in k:
                    current_state[k] = v
            net_local.load_state_dict(current_state)
            
            # 必须传入 w_glob=w_glob_sm
            w_new, loss, _, l_protos = local.train(
                net=net_local.to(args.device),
                w_glob_keys=w_glob_keys,
                w_glob=w_glob_sm,
                global_protos=global_protos,
                lr=args.lr,
                gamma=args.gamma, # 顺便确认gamma参数传入
                ind=iter  # 传入当前轮次 (0, 1, 2...)
            )
            
            w_locals_storage[idx] = copy.deepcopy(w_new)
            w_upload = {k: w_new[k] for k in w_glob_keys}
            w_sm_heads.append(w_upload)
            loss_locals.append(loss)
            client_sample_counts.append(len(user_idxs))# 收集样本数 (user_idxs 是已经经过 m_tr 截断后的真实索引)
            
            if l_protos:
                for y, p in l_protos.items():
                    if y not in collected_protos: collected_protos[y] = []
                    collected_protos[y].append(p)
        
        # 标准加权聚合
        if len(w_sm_heads) > 0:
            w_glob_sm = copy.deepcopy(w_sm_heads[0])
            
            # 1. 初始化累加器为全 0
            for k in w_glob_sm.keys():
                w_glob_sm[k] = torch.zeros_like(w_glob_sm[k])
                
            # 2. 计算总样本数
            total_samples = sum(client_sample_counts)

            # 3. 加权累加
            for i in range(len(w_sm_heads)):
                # 权重 = 该客户端实际参与训练的样本数 / 本轮总样本数
                weight = client_sample_counts[i] / total_samples
                for k in w_glob_sm.keys():
                    w_glob_sm[k] += w_sm_heads[i][k] * weight
            
            net_glob.load_state_dict(w_glob_sm, strict=False)
        
        for y in collected_protos:
            global_protos[y] = torch.stack(collected_protos[y]).mean(dim=0).detach()

        # 计算训练和测试指标
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train_hist.append(loss_avg)
        
        # 计算训练准确率（个性化）
        acc_train, loss_train_eval = evaluate_local_weighted(net_glob, dataset_train, dict_users_train, args, w_locals_storage)
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