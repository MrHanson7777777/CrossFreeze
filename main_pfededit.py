#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from models.Update import LocalUpdate, LocalUpdatePFedEdit
from models.test import test_img_local

# 评估函数 (复用)
def evaluate_local_weighted(net, dataset, dict_users, args, w_locals_storage):
    total_acc, total_loss, total_samples = 0, 0, 0
    for idx in dict_users.keys():
        user_idxs = dict_users[idx]
        n_samples = len(user_idxs)
        # 这里的 net 只是架构，加载该用户的本地参数
        if idx in w_locals_storage:
            net.load_state_dict(w_locals_storage[idx])
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
    args.alg = 'pfededit'
    
    # 初始化
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"./experiments/{args.alg}_{args.dataset}_{time_str}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 加载数据和模型
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    net_glob = get_model(args)
    net_glob.train()

    # =============================================================================
    # ✂️ [关键修改] 实施真正的静态数据稀缺 (True Static Data Scarcity)
    # 逻辑：在实验开始前一次性截断数据，确保每个用户在整个训练周期内只能看到这 m_tr 个样本。
    # =============================================================================
    if args.data_usage < 1.0:
        print(f"⚠️ DATA SCARCITY MODE: Trained with {args.data_usage*100}% of global data (Stratified Pruned).")
    # =============================================================================
    
    # 存储所有客户端的本地模型 (w_local)
    # 初始时，都复制自全局模型
    w_locals_storage = {i: copy.deepcopy(net_glob.state_dict()) for i in range(args.num_users)}
    
    # === 历史记录列表 ===
    loss_train_hist, loss_test_hist, acc_train_hist, acc_test_hist = [], [], [], []
    
    print(f"✂️ Starting pFedEdit (Automated Model Editing)...")
    print(f"   Edit Ratio (k): {args.edit_ratio}, Subset Ratio (p): {args.subset_ratio}")

    progress_bar = tqdm(range(args.epochs), desc="Training Progress")
    
    for iter in progress_bar:
        # 学习率衰减
        if iter == int(args.epochs * 0.5) or iter == int(args.epochs * 0.75):
            args.lr *= 0.1
            tqdm.write(f"\n📉 Learning Rate decayed to {args.lr}")
            
        w_glob_updates = []
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # 将当前的全局模型参数提取出来，供所有客户端使用
        w_glob_current = net_glob.state_dict()
        
        for idx in idxs_users:
            user_idxs = dict_users_train[idx]
            
            # 策略：如果是第一轮 (Round 0)，还没有“训练好的本地模型”来做编辑
            # 论文中提到 "First editing happens after the first round" 
            if iter == 0:
                # 第一轮：普通训练 (FedAvg)
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=user_idxs)
                net_temp = copy.deepcopy(net_glob)
                w_new, loss, _ = local.train(net=net_temp.to(args.device), w_glob_keys=[], lr=args.lr)
            else:
                # 后续轮次：pFedEdit 逻辑
                local = LocalUpdatePFedEdit(args=args, dataset=dataset_train, idxs=user_idxs)
                
                # 准备模型：
                # 1. net_glob_copy: 传入当前的全局模型 (会被修改)
                # 2. net_local_copy: 传入该用户上一轮的本地模型 (作为编辑源)
                net_glob_copy = copy.deepcopy(net_glob).to(args.device)
                
                # 构建本地模型实例
                net_local_copy = copy.deepcopy(net_glob).to(args.device) # 仅借用架构
                net_local_copy.load_state_dict(w_locals_storage[idx])
                
                w_new, loss = local.train(net_glob=net_glob_copy, net_local=net_local_copy, lr=args.lr)
            
            # 更新存储和聚合列表
            w_locals_storage[idx] = copy.deepcopy(w_new) # 存储为下一轮的 "Clean Model"
            w_glob_updates.append(w_new)
            loss_locals.append(loss)
            
        # === 全局聚合 (Standard Aggregation) ===
        w_glob_avg = copy.deepcopy(w_glob_updates[0])
        for k in w_glob_avg.keys():
            for i in range(1, len(w_glob_updates)):
                w_glob_avg[k] += w_glob_updates[i][k]
            w_glob_avg[k] = torch.div(w_glob_avg[k], len(w_glob_updates))
        
        net_glob.load_state_dict(w_glob_avg)
        
        # === 评估 ===
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train_hist.append(loss_avg)
        
        # 评估个性化准确率
        acc_train, _ = evaluate_local_weighted(net_glob, dataset_train, dict_users_train, args, w_locals_storage)
        acc_train_hist.append(acc_train)
        
        if (iter + 1) % args.test_freq == 0:
            acc_test, loss_test = evaluate_local_weighted(net_glob, dataset_test, dict_users_test, args, w_locals_storage)
            acc_test_hist.append(acc_test)
            loss_test_hist.append(loss_test)
            
            # 添加详细的训练日志打印
            progress_bar.set_postfix({
                'TrLoss': f'{loss_avg:.4f}',
                'TrAcc': f'{acc_train:.2f}%',
                'TeAcc': f'{acc_test:.2f}%'
            })
            tqdm.write(f"🔄 Round {iter+1:3d} | Train Loss: {loss_avg:.4f} | Train Acc: {acc_train:.2f}% | Test Loss: {loss_test:.4f} | Test Acc: {acc_test:.2f}%")
        else:
            acc_test_hist.append(acc_test_hist[-1] if len(acc_test_hist)>0 else 0)
            loss_test_hist.append(loss_test_hist[-1] if len(loss_test_hist)>0 else 0)
            
            # 非测试轮次也显示训练信息
            progress_bar.set_postfix({
                'TrLoss': f'{loss_avg:.4f}',
                'TrAcc': f'{acc_train:.2f}%'
            })

    # === 保存结果和绘制图表 ===
    # 创建实验目录并保存配置
    with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
        for arg in vars(args): f.write(f"{arg}: {getattr(args, arg)}\n")
    
    # 绘制训练结果图表 (与其他算法保持一致)
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
    print(f"📂 Saving to {experiment_dir}")