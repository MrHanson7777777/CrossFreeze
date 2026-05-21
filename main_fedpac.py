#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np
import torch
from tqdm import tqdm
from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdateFedPAC
from models.test import test_img_local

# 修复 6: 科学的加权评估函数 (Weighted Sample Accuracy)
def evaluate_pac_weighted(net, dataset, dict_users, args):
    net.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for idx, user_idxs in dict_users.items():
            # test_img_local 返回的是 (acc%, loss)
            acc, _ = test_img_local(net, dataset, args, idx=idx, idxs=user_idxs)
            # 反推正确样本数
            correct_samples = (acc / 100.0) * len(user_idxs)
            total_correct += correct_samples
            total_samples += len(user_idxs)
            
    return (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # 设置随机种子保证可复现
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    net_glob = get_model(args)
    
    global_protos = {}
    
    for iter in tqdm(range(args.epochs), desc="FedPAC Epochs"):
        w_locals, sample_counts = [], []
        collected_protos = {} 
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdateFedPAC(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            
            w, loss, l_protos = local.train(
                net=copy.deepcopy(net_glob).to(args.device), 
                global_protos=global_protos, 
                lr=args.lr
            )
            
            w_locals.append(copy.deepcopy(w))
            sample_counts.append(len(dict_users_train[idx]))
            
            for label, p in l_protos.items():
                if label not in collected_protos:
                    collected_protos[label] = []
                collected_protos[label].append(p)

        # 1. 模型聚合 (Weighted FedAvg)
        total_s = sum(sample_counts)
        w_glob = copy.deepcopy(w_locals[0])
        for k in w_glob.keys():
            w_glob[k] = torch.stack([
                w_locals[i][k] * (sample_counts[i] / total_s) for i in range(len(w_locals))
            ], dim=0).sum(dim=0)
        net_glob.load_state_dict(w_glob)

        # 2. 原型聚合 (Mean)
        global_protos = {} # 清空旧原型
        for label, p_list in collected_protos.items():
            global_protos[label] = torch.stack(p_list).mean(dim=0).detach()

        # 测试
        if (iter + 1) % args.test_freq == 0:
            acc_test = evaluate_pac_weighted(net_glob, dataset_test, dict_users_test, args)
            print(f"Round {iter+1} | Test Acc: {acc_test:.2f}%")