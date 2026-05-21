# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/test.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time

# 条件导入 language_utils，仅在需要时导入
try:
    from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y
    LANGUAGE_UTILS_AVAILABLE = True
except ImportError:
    LANGUAGE_UTILS_AVAILABLE = False

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        image, label = self.dataset[d]
        return image, label

class DatasetSplit_leaf(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label

def test_img_local(net_g, dataset, args,idx=None,indd=None, user_idx=-1, idxs=None):
    net_g.eval()
    test_loss = 0
    correct = 0

    # put LEAF data into proper format
    if 'femnist' in args.dataset:
        leaf=True
        datatest_new = []
        usr = idx
        for j in range(len(dataset[usr]['x'])):
            datatest_new.append((torch.reshape(torch.tensor(dataset[idx]['x'][j]),(1,28,28)),torch.tensor(dataset[idx]['y'][j])))
    elif 'sent140' in args.dataset:
        leaf=True
        datatest_new = []
        for j in range(len(dataset[idx]['x'])):
            datatest_new.append((dataset[idx]['x'][j],dataset[idx]['y'][j]))
    else:
        leaf=False
    
    if leaf:
        data_loader = DataLoader(DatasetSplit_leaf(datatest_new,np.ones(len(datatest_new))), batch_size=args.local_bs, shuffle=False)
    else:
        data_loader = DataLoader(DatasetSplit(dataset,idxs), batch_size=args.local_bs,shuffle=False)
    if 'sent140' in args.dataset:
        hidden_train = net_g.init_hidden(args.local_bs)
    count = 0
    for idx, (data, target) in enumerate(data_loader):
        if 'sent140' in args.dataset:
            if not LANGUAGE_UTILS_AVAILABLE:
                raise ImportError("language_utils module is required for sent140 dataset")
            input_data, target_data = process_x(data, indd), process_y(target, indd)
            if args.local_bs != 1 and input_data.shape[0] != args.local_bs:
                break

            data, targets = torch.from_numpy(input_data).to(args.device), torch.from_numpy(target_data).to(args.device)
            net_g.zero_grad()

            hidden_train = repackage_hidden(hidden_train)
            output, feature = net_g(data, hidden_train)  # RNN现在返回(output, feature)
            hidden_train = net_g.last_hidden  # 获取更新后的hidden state

            loss = F.cross_entropy(output, torch.max(targets, 1)[1])  # output已经是.t()的结果
            _, pred_label = torch.max(output, 1)
            correct += (pred_label == torch.max(targets, 1)[1]).sum().item()
            count += args.local_bs
            test_loss += loss.item()

        else:
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            
            # --- 修复开始 ---
            # 统一处理所有算法的返回值
            if hasattr(args, 'alg') and args.alg == 'crossfreeze':
                 # CrossFreeze 本地测试通常用 mode='local'
                 output = net_g(data, mode='local')
            else:
                 output = net_g(data)

            # 拆包逻辑：如果是元组 (logits, features)，取第一个；如果是 Tensor，直接用
            if isinstance(output, tuple):
                log_probs = output[0]
            else:
                log_probs = output
            # --- 修复结束 ---
            
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()

    if 'sent140' not in args.dataset:
        count = len(data_loader.dataset)
    
    # === [修复] 防止分母为零崩溃 ===
    if count > 0:
        test_loss /= count
        accuracy = 100.00 * float(correct) / count
    else:
        test_loss = 0.0
        accuracy = 0.0
    # ============================
    
    return accuracy, test_loss

def test_img_local_all(net, args, dataset_test, dict_users_test, w_locals=None, w_glob_keys=None, indd=None, dataset_train=None, dict_users_train=None, return_all=False):
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    
    # 存储原始准确率，用于计算简单平均
    raw_acc_list = []
    
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(net)
        if w_locals is not None:
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
        net_local.eval()
        
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            a, b = test_img_local(net_local, dataset_test, args, idx=dict_users_test[idx], indd=indd, user_idx=idx)
            tot += len(dataset_test[dict_users_test[idx]]['x'])
            # 记录样本数用于加权
            num_samples = len(dataset_test[dict_users_test[idx]]['x'])
        else:
            a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx]) 
            tot += len(dict_users_test[idx])
            # 记录样本数用于加权
            num_samples = len(dict_users_test[idx])
            
        # a 是当前客户端的准确率 (0-100)
        # [新增] 收集纯准确率
        raw_acc_list.append(a)

        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            acc_test_local[idx] = a * num_samples
            loss_test_local[idx] = b * num_samples
        else:
            acc_test_local[idx] = a * num_samples
            loss_test_local[idx] = b * num_samples
        del net_local
    
    if return_all:
        return acc_test_local, loss_test_local

    # 1. 计算加权平均 (Weighted Average) - 更科学，反映整体性能
    weighted_acc = sum(acc_test_local) / tot
    
    # 2. [新增] 计算简单平均 (Simple Average) - 更符合部分论文直觉
    simple_acc = sum(raw_acc_list) / num_idxxs

    # 返回三个值：加权准确率，简单准确率，平均损失
    return weighted_acc, simple_acc, sum(loss_test_local)/tot


def test_img_global(net_g, dataset, args):
    """
    测试服务器端全局模型的泛化能力
    支持 CrossFreeze (Sm Head) 和 FedAvg/FedProto (Global Aggregated Model)
    """
    net_g.eval()
    test_loss = 0
    correct = 0
    
    # 使用 Test Batch Size
    data_loader = DataLoader(dataset, batch_size=args.bs)
    l = len(dataset)
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            
            # --- 关键：兼容 CrossFreeze 的双头切换 ---
            if args.alg == 'crossfreeze':
                # 强制使用 Sm 头 (Global Head) 进行预测
                log_probs, _ = net_g(data, mode='global')
            else:
                # FedAvg / FedRep / FedProto 使用默认 forward
                output = net_g(data)
                if isinstance(output, tuple):
                    log_probs = output[0]
                else:
                    log_probs = output
            
            # 计算指标
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()

    test_loss /= l
    accuracy = 100.00 * correct / l
    
    return accuracy, test_loss
