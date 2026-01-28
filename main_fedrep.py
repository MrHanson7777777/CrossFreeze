# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg), 
# FedAvg (--alg fedavg) and FedProx (--alg prox)

import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免GUI问题
from sklearn.manifold import TSNE
from datetime import datetime
import os

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate
from models.test import test_img_local_all

import time

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # === 创建基于时间的实验文件夹 ===
    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.alg}_{args.dataset}_{args.model}_{args.epochs}epochs_{args.num_users}users"
    experiment_dir = f"./experiments/{experiment_time}_{experiment_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 保存实验配置
    config_path = os.path.join(experiment_dir, "config.txt")
    with open(config_path, 'w') as f:
        f.write(f"Experiment Configuration\n")
        f.write(f"========================\n")
        f.write(f"Time: {experiment_time}\n")
        f.write(f"Algorithm: {args.alg}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Num Users: {args.num_users}\n")
        f.write(f"Fraction: {args.frac}\n")
        f.write(f"Local Epochs: {args.local_ep}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Local Batch Size: {args.local_bs}\n")
        if args.alg == 'fedproto':
            f.write(f"Lambda (ld): {args.ld}\n")
        elif args.alg == 'crossfreeze':
            f.write(f"Gamma: {getattr(args, 'gamma', 2.0)}\n")
    
    print(f"Experiment directory: {experiment_dir}")

    if 'cifar' in args.dataset or args.dataset == 'mnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        
        # === [新增] 动态计算每个用户的数据量 ===
        lens = []
        for idx in range(args.num_users):
            lens.append(len(dict_users_train[idx]))
            np.random.shuffle(dict_users_train[idx])
        lens = np.array(lens) # 转为 numpy 数组方便后续计算
        # ===================================
    else:
        if 'femnist' in args.dataset:
            train_path = './leaf-master/data/' + args.dataset + '/data/mytrain'
            test_path = './leaf-master/data/' + args.dataset + '/data/mytest'
        else:
            train_path = './leaf-master/data/' + args.dataset + '/data/train'
            test_path = './leaf-master/data/' + args.dataset + '/data/test'
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys()) 
        dict_users_test = list(dataset_test.keys())
        print(lens)
        print(clients)
        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    print(args.alg)

    # build model
    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    if args.alg == 'fedrep' or args.alg == 'fedper':
        if args.model == 'resnet':
            # FedRep 的逻辑是：聚合 Body，私有化 Head
            # 【重要】这里改为列表推导式生成嵌套列表 [[key1], [key2]...]
            # 这样后面的 itertools.chain 才能将其展平为 [key1, key2...]，而不是拆成字母
            w_glob_keys = [[key] for key in net_glob.state_dict().keys() if 'fc' not in key and 'sm_head' not in key]
        # ==============================
        elif 'cifar' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,3,4]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0,1,2,3,4,5]]
        else:
            w_glob_keys = net_keys[:-2]
    # === 新增 CrossFreeze 逻辑 ===
    elif args.alg == 'crossfreeze':
        # 【修正】必须用双层列表 [['...'], ['...']]
        # 否则后面的 itertools.chain 会把它拆成单个字母！
        w_glob_keys = [['sm_head.weight', 'sm_head.bias']]
    # === 新增 FedProto 逻辑 ===
    elif args.alg == 'fedproto':
        # FedProto 不聚合参数，只聚合原型，所以 w_glob_keys 为空
        w_glob_keys = []
    elif args.alg == 'lg':
        if 'cifar' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1,2]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [2,3]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0,6,7]]
        else:
            w_glob_keys = net_keys[total_num_layers - 2:]

    if args.alg == 'fedavg' or args.alg == 'prox' or args.alg == 'fedproto':
        w_glob_keys = []
    if 'sent140' not in args.dataset and w_glob_keys:  # 只在非空时调用 itertools.chain
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    
    print(total_num_layers)
    print(w_glob_keys)
    print(net_keys)
    if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'lg':
        num_param_glob = 0
        num_param_local = 0
        for key in net_glob.state_dict().keys():
            num_param_local += net_glob.state_dict()[key].numel()
            print(num_param_local)
            if key in w_glob_keys:
                num_param_glob += net_glob.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    
    # === [新增] 打印通信量占比 ===
    total_params = sum(p.numel() for p in net_glob.parameters())
    if args.alg == 'crossfreeze':
        # CrossFreeze 只传 Sm
        uploaded_params = sum(p.numel() for n, p in net_glob.named_parameters() if 'sm_head' in n)
        algo_label = "CrossFreeze (High Efficiency)"
    elif args.alg == 'fedrep':
        # FedRep 传 Body (除 fc 外所有)
        uploaded_params = sum(p.numel() for n, p in net_glob.named_parameters() if 'fc' not in n and 'sm_head' not in n)
        algo_label = "FedRep (Representation Learning)"
    elif args.alg == 'fedproto':
        # FedProto 传原型向量 + 分类头
        # 原型向量：num_classes * feature_dim (例如 10 * 512 = 5120)
        # 分类头：feature_dim * num_classes + num_classes (例如 512*10 + 10 = 5130)
        if args.model == 'resnet':
            feature_dim = 512  # ResNet18的特征维度
        elif args.model == 'cnn':
            feature_dim = 64   # CNN的特征维度
        else:
            feature_dim = 256  # 默认特征维度
        
        prototype_params = args.num_classes * feature_dim  # 原型向量
        classifier_params = feature_dim * args.num_classes + args.num_classes  # 分类头
        uploaded_params = prototype_params + classifier_params
        algo_label = "FedProto (Prototype-based)"
    elif args.alg == 'fedper':
        # FedPer 传部分参数
        uploaded_params = total_params  # 根据实际情况调整
        algo_label = "FedPer (Personalized)"
    elif args.alg == 'lg':
        # LG-FedAvg 传部分参数
        uploaded_params = total_params  # 根据实际情况调整
        algo_label = "LG-FedAvg (Local-Global)"
    elif args.alg == 'fedavg':
        # FedAvg 传所有
        uploaded_params = total_params
        algo_label = "FedAvg (Full Model)"
    else:
        # 其他算法默认传所有
        uploaded_params = total_params
        algo_label = f"{args.alg.upper()} (Full Model)"

    ratio = uploaded_params / total_params * 100
    print("\n" + "="*50)
    print(f"Communication Strategy: {algo_label}")
    print(f"   • Total Model Params:    {total_params:,}")
    print(f"   • Uploaded per Round:    {uploaded_params:,}")
    print(f"   • Communication Cost:    {ratio:.4f}% of total model")
    print("="*50 + "\n")
    
    print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] =net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    # training
    indd = None      # indices of embedding for sent140
    loss_train = []
    loss_test = []   # 新增：测试损失收集
    accs = []
    test_rounds = [] # 新增：测试轮次记录（用于绘图）
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    
    # === 初始化全局原型 ===
    global_protos = {}  # 统一初始化为空字典，CrossFreeze 也用这个
    
    # === 添加tqdm进度条 ===
    progress_bar = tqdm(range(args.epochs+1), desc="Training Progress", ncols=100)
    
    for iter in progress_bar:
        w_glob = {} # 这是给下一轮用的累加器，不要动它
        
        # === 准备 CrossFreeze 的锚点 ===
        w_glob_anchor = None
        if args.alg == 'crossfreeze':
            w_glob_anchor = {k: v.clone() for k, v in net_glob.state_dict().items() if k in w_glob_keys}
        elif args.alg == 'fedproto':
            # FedProto 依然用 w_glob_anchor 传原型
            w_glob_anchor = global_protos if iter > 0 else None

        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len=0
        
        # === 收集原型的容器 (CrossFreeze 和 FedProto 都需要) ===
        collected_protos = {} 
        w_glob_agg = {}
        
        # === 客户端训练进度条 ===
        client_desc = f"Round {iter:3d} - Training {m} clients"
        client_progress = tqdm(enumerate(idxs_users), total=len(idxs_users), 
                              desc=client_desc, leave=False, ncols=80)
        
        for ind, idx in client_progress:
            
            # === [新增] 健壮性检查：跳过没有数据的"倒霉"客户端 ===
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                # LEAF数据集的检查方式
                dataset_size = len(dataset_train[list(dataset_train.keys())[idx]])
            else:
                # 其他数据集的检查方式
                dataset_size = len(dict_users_train[idx])
            
            if dataset_size < args.local_bs:
                # 如果数据量为0，或者少于一个batch（因为drop_last=True），直接跳过
                tqdm.write(f"⚠️ Client {idx} skipped (Insufficient data: {dataset_size} samples < {args.local_bs} batch_size)")
                continue
            # ====================================================
            
            start_in = time.time()
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                if args.epochs == iter:
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_ft]], idxs=dict_users_train, indd=indd)
                else:
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]], idxs=dict_users_train, indd=indd)
            else:
                if args.epochs == iter:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])
                else:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            if args.alg != 'fedavg' and args.alg != 'prox':
                for k in w_locals[idx].keys():
                    if k not in w_glob_keys:
                        w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
            last = iter == args.epochs
            # === 训练并处理返回值 ===
            if args.alg == 'crossfreeze':
                # 【修改点 1】CrossFreeze 传入 global_protos，并接收 4 个返回值
                # w_glob 传的是 weights (Sm), global_protos 传的是特征原型
                w_local, loss, indd, local_protos = local.train(
                    net=net_local.to(args.device), 
                    idx=idx, 
                    w_glob_keys=w_glob_keys, 
                    lr=args.lr, 
                    last=last, 
                    w_glob=w_glob_anchor,     # 传 Head 权重
                    gamma=getattr(args, 'gamma', 2.0),
                    global_protos=global_protos # 传 特征原型
                )
                
                # 收集本地原型
                for y, proto in local_protos.items():
                    if y not in collected_protos:
                        collected_protos[y] = []
                    collected_protos[y].append(proto.to(args.device))
                    
            elif args.alg == 'fedproto':
                # FedProto 独立版逻辑，统一传入 global_protos
                w_local_tuple, loss, indd = local.train(
                    net=net_local.to(args.device), 
                    idx=idx, 
                    w_glob_keys=w_glob_keys, 
                    lr=args.lr, 
                    last=last, 
                    w_glob=w_glob_anchor, # 这里传的是原型 (原有逻辑)
                    global_protos=global_protos # 统一传入
                )
                local_protos, trained_weights = w_local_tuple
                w_locals[idx] = copy.deepcopy(trained_weights) # 保存私有模型
                
                # 收集用于聚合
                for y, proto in local_protos.items():
                    if y not in collected_protos:
                        collected_protos[y] = []
                    collected_protos[y].append(proto.to(args.device))
                    
                # 累加权重用于全局测试 (仅测试用)
                if len(w_glob_agg) == 0:
                    w_glob_agg = copy.deepcopy(trained_weights)
                    for k in w_glob_agg.keys(): w_glob_agg[k] = w_glob_agg[k] * lens[idx]
                else:
                    for k in w_glob_agg.keys(): w_glob_agg[k] += trained_weights[k] * lens[idx]
                    
            else:
                # 其他算法 (FedAvg, FedRep 等)
                w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx, w_glob_keys=w_glob_keys, lr=args.lr, last=last)
            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]

            # === 聚合权重 (仅针对 CrossFreeze/FedAvg/FedRep) ===
            if args.alg != 'fedproto':
                if len(w_glob) == 0:
                    w_glob = copy.deepcopy(w_local)
                    for k,key in enumerate(net_glob.state_dict().keys()):
                        w_glob[key] = w_glob[key]*lens[idx]
                        w_locals[idx][key] = w_local[key]
                else:
                    for k,key in enumerate(net_glob.state_dict().keys()):
                        if key in w_glob_keys:
                            w_glob[key] += w_local[key]*lens[idx]
                        else:
                            w_glob[key] += w_local[key]*lens[idx]
                        w_locals[idx][key] = w_local[key]

            times_in.append( time.time() - start_in )
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # === 循环结束后的聚合更新 ===
        
        # 1. 更新全局原型 (CrossFreeze 和 FedProto 共用)
        if args.alg in ['crossfreeze', 'fedproto']:
            new_global_protos = {}
            for y, proto_list in collected_protos.items():
                new_global_protos[y] = torch.stack(proto_list).mean(dim=0)
            global_protos = new_global_protos
        
        # 2. 更新全局模型参数
        if args.alg == 'fedproto':
            # FedProto 仅更新用于测试的临时全局模型
            for k in w_glob_agg.keys():
                w_glob_agg[k] = torch.div(w_glob_agg[k], total_len)
            net_glob.load_state_dict(w_glob_agg)
        else:
            # CrossFreeze / FedRep / FedAvg 正常更新参数
            for k in net_glob.state_dict().keys():
                w_glob[k] = torch.div(w_glob[k], total_len)
            w_local = net_glob.state_dict()
            for k in w_glob.keys():
                w_local[k] = w_glob[k]
            if args.epochs != iter:
                net_glob.load_state_dict(w_glob)

        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            
            # === [修改] 接收三个返回值 ===
            acc_test_val, acc_test_simple, loss_test_val = test_img_local_all(
                net_glob, args, dataset_test, dict_users_test,
                w_glob_keys=w_glob_keys, w_locals=w_locals, indd=indd,
                dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False
            )
            
            # 这里我们依然把"加权平均"作为主要的 accs 记录（因为它更稳健）
            # 或者你可以选择记录 accs.append(acc_test_simple) 看你更想由哪个指标主导
            accs.append(acc_test_val) 
            
            loss_test.append(loss_test_val)  # 收集测试损失
            test_rounds.append(iter)         # 记录测试轮次
            
            # === [修改] 日志打印，同时显示两个准确率 ===
            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            if iter != args.epochs:
                # 打印格式：Weighted (Simple)
                tqdm.write(f'🔄 Round {iter:3d} | Train Loss: {loss_avg:.4f} | Test Loss: {loss_test_val:.4f} | Test Acc: {acc_test_val:.2f}% (Avg: {acc_test_simple:.2f}%) | Time: {max(times_in):.2f}s')
                
                progress_bar.set_postfix({
                    'Train_Loss': f'{loss_avg:.4f}',
                    'Test_Acc': f'{acc_test_val:.2f}%', # 进度条只显示加权
                    'Test_Loss': f'{loss_test_val:.4f}'
                })
            else:
                # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                tqdm.write(f'🎯 Final Round | Train Loss: {loss_avg:.4f} | Test Loss: {loss_test_val:.4f} | Test Acc: {acc_test_val:.2f}% (Avg: {acc_test_simple:.2f}%)')
            
            # [修改后] 动态计算，不再死板除以 10
            # 确定我们要统计最后多少轮（如果有 150 轮就取最后 10 轮，如果只有 3 轮就取 3 轮）
            last_n_rounds = min(10, args.epochs)
            if iter >= args.epochs - last_n_rounds and iter != args.epochs:
                accs10 += acc_test_val / last_n_rounds

            # below prints the global accuracy of the single global model for the relevant algs
            if args.alg == 'fedavg' or args.alg == 'prox':
                acc_test_glob, acc_test_glob_simple, loss_test_glob = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                        w_locals=None,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
                if iter != args.epochs:
                    tqdm.write(f'🌍 Round {iter:3d} Global | Train Loss: {loss_avg:.4f} | Test Loss: {loss_test_glob:.4f} | Test Acc: {acc_test_glob:.2f}% (Avg: {acc_test_glob_simple:.2f}%)')
                else:
                    tqdm.write(f'🌍 Final Round Global | Train Loss: {loss_avg:.4f} | Test Loss: {loss_test_glob:.4f} | Test Acc: {acc_test_glob:.2f}% (Avg: {acc_test_glob_simple:.2f}%)')
                # [修改后] 同样使用动态计算
                last_n_rounds = min(10, args.epochs)
                if iter >= args.epochs - last_n_rounds and iter != args.epochs:
                    accs10_glob += acc_test_glob / last_n_rounds

            # === [新增] 全局泛化能力测试 ===
            if args.alg in ['crossfreeze', 'fedavg', 'fedproto']:
                from models.test import test_img_global
                
                acc_global, loss_global = test_img_global(net_glob, dataset_test, args)
                if iter != args.epochs:
                    tqdm.write(f"🌍 [Global Test] Round {iter} | Acc: {acc_global:.2f}% | Loss: {loss_global:.4f}")
                else:
                    tqdm.write(f"🌍 [Global Test] Final Round | Acc: {acc_global:.2f}% | Loss: {loss_global:.4f}")
                
                if args.alg == 'crossfreeze':
                    tqdm.write(f"   (Using Pre-trained Body + Aggregated Sm Head)")
            # ==============================

        if iter % args.save_every==args.save_every-1:
            model_save_path = './save/accs_'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
            torch.save(net_glob.state_dict(), model_save_path)

    # === 训练完成，关闭进度条并输出最终结果 ===
    progress_bar.close()
    
    print('\n' + '='*70)
    print(f'🎉 Training Completed! Algorithm: {args.alg.upper()}')
    print(f'📊 Final Results Summary:')
    print(f'   • Average accuracy (final 10 rounds): {accs10:.2f}%')
    if args.alg == 'fedavg' or args.alg == 'prox':
        print(f'   • Average global accuracy (final 10 rounds): {accs10_glob:.2f}%')
    
    total_time = time.time() - start
    print(f'⏱️  Total Training Time: {total_time:.2f}s ({total_time/60:.2f}min)')
    print(f'📈 Best Test Accuracy: {max(accs):.2f}%')
    print(f'📉 Final Test Accuracy: {accs[-1]:.2f}%')
    print('='*70)
    
    # === 绘制训练曲线 ===
    def plot_curves():
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 训练损失曲线
        ax1.plot(range(len(loss_train)), loss_train, 'b-', label='Train Loss', linewidth=2)
        ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 测试损失曲线
        if len(loss_test) > 0:
            ax2.plot(test_rounds, loss_test, 'r-', label='Test Loss', linewidth=2)
            ax2.set_title('Test Loss Curve', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 3. 测试准确率曲线
        if len(accs) > 0:
            ax3.plot(test_rounds, accs, 'g-', label='Test Accuracy', linewidth=2, marker='o', markersize=4)
            ax3.set_title('Test Accuracy Curve', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Accuracy (%)')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
        # 4. 训练和测试损失对比
        ax4.plot(range(len(loss_train)), loss_train, 'b-', label='Train Loss', linewidth=2)
        if len(loss_test) > 0:
            ax4.plot(test_rounds, loss_test, 'r-', label='Test Loss', linewidth=2)
        ax4.set_title('Train vs Test Loss Comparison', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Loss')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(experiment_dir, f'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    try:
        plot_path = plot_curves()
        print(f'📊 Training curves saved to: {plot_path}')
    except Exception as e:
        print(f'⚠️  Warning: Could not save plots - {e}')
    
    # === t-SNE 特征分布可视化 ===
    def extract_and_visualize_features():
        print("\n🎨 Generating t-SNE visualization for Global Model...")
        
        # 1. 设置采样数量 (2000个点足够看清分布，太多会跑很慢)
        num_samples = 2000
        
        # 2. 创建一个临时的 DataLoader，专门用于抽取测试集数据
        # shuffle=True 是关键！确保我们随机抽到所有类，而不是只抽到前几个类
        tsne_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=True)
        
        features_list = []
        labels_list = []
        count = 0
        
        net_glob.eval()
        
        with torch.no_grad():
            for batch_imgs, batch_labels in tsne_loader:
                batch_imgs = batch_imgs.to(args.device)
                
                # === 核心逻辑：提取特征 (Feature Extraction) ===
                if args.alg == 'crossfreeze':
                    # CrossFreeze: 返回 (output, features)
                    # mode='global' 确保使用 Sm 头 (虽然提取特征主要看 Body)
                    _, feats = net_glob(batch_imgs, mode='global')
                else:
                    # FedAvg / FedRep / FedProto
                    output = net_glob(batch_imgs)
                    # 兼容性处理：有的模型返回 tuple，有的直接返回 output
                    if isinstance(output, tuple):
                        _, feats = output
                    else:
                        # 如果模型没有返回特征层，通常取倒数第二层
                        # 这里假设你的 ResNet 修改版已经都能返回特征了
                        # 如果不能，暂时用 output (logits) 代替，虽然效果差一点但也能画
                        feats = output 
                
                # 收集数据
                features_list.append(feats.cpu().numpy())
                labels_list.append(batch_labels.numpy())
                
                count += len(batch_labels)
                if count >= num_samples:
                    break
        
        # 3. 数据拼接
        features = np.concatenate(features_list, axis=0)[:num_samples]
        labels = np.concatenate(labels_list, axis=0)[:num_samples]
        
        print(f"📊 t-SNE Data Shape: {features.shape}")
        print(f"📊 Unique Classes Found: {len(np.unique(labels))} (Should be 10)")
        
        # 4. 运行 t-SNE (降维: 512 -> 2)
        print("⏳ Running t-SNE algorithm (this may take a moment)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
        features_2d = tsne.fit_transform(features)
        
        # 5. 绘图
        plt.figure(figsize=(10, 8))
        # 使用 jet 或 tab10 颜色映射，保证10个类颜色区分明显
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=labels, cmap='tab10', alpha=0.6, s=20)
        
        plt.colorbar(scatter, ticks=range(10), label='Classes')
        plt.title(f't-SNE Visualization ({args.alg.upper()}) - Dirichlet (alpha={getattr(args, "alpha", "N/A")})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        
        # 6. 保存
        save_path = f'./experiments/{args.alg}_dirichlet_tsne.png'
        # 如果有 experiment_dir 变量，可以用 os.path.join(experiment_dir, 'tsne.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ t-SNE plot saved to: {save_path}")

    # 在脚本末尾调用 extract_and_visualize_features
    if __name__ == '__main__':
        # ... (训练代码) ...
        
        # 训练结束后调用
        try:
            extract_and_visualize_features()
        except Exception as e:
            print(f"⚠️ t-SNE plot failed: {e}")
