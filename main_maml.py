#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np
import torch
from tqdm import tqdm
from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdateMAML_Clean
from models.test import test_img_local

def evaluate_weighted(net, dataset, dict_users, args):
    net.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for idx, user_idxs in dict_users.items():
            acc, _ = test_img_local(net, dataset, args, idx=idx, idxs=user_idxs)
            total_correct += (acc / 100.0) * len(user_idxs)
            total_samples += len(user_idxs)
    return (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    net_glob = get_model(args)

    for iter in tqdm(range(args.epochs), desc="MAML Epochs"):
        w_locals, sample_counts = [], []
        idxs_users = np.random.choice(range(args.num_users), max(int(args.frac * args.num_users), 1), replace=False)

        for idx in idxs_users:
            local = LocalUpdateMAML_Clean(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            
            # MAML 训练
            w, _ = local.train(
                net=copy.deepcopy(net_glob).to(args.device), 
                lr_inner=args.alpha_maml, 
                lr_outer=args.beta_maml
            )
            w_locals.append(copy.deepcopy(w))
            sample_counts.append(len(dict_users_train[idx]))

        # 聚合
        total_s = sum(sample_counts)
        w_glob = copy.deepcopy(w_locals[0])
        for k in w_glob.keys():
            w_glob[k] = torch.stack([
                w_locals[i][k] * (sample_counts[i] / total_s) for i in range(len(w_locals))
            ], dim=0).sum(dim=0)
        net_glob.load_state_dict(w_glob)

        if (iter + 1) % args.test_freq == 0:
            acc_test = evaluate_weighted(net_glob, dataset_test, dict_users_test, args)
            print(f"Round {iter+1} | Test Acc: {acc_test:.2f}%")