# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Update.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import numpy as np
import time
import copy
import itertools # 修复 2: 导入 itertools 用于 MAML 循环
# import FedProx  # 已删除：FedProx模块损坏且CrossFreeze不需要

# 条件导入 language_utils，仅在需要时导入
try:
    from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y
    LANGUAGE_UTILS_AVAILABLE = True
except ImportError:
    LANGUAGE_UTILS_AVAILABLE = False 

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.idxs[item]]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]),(1,28,28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label

# === FedPAC Update ===
class LocalUpdateFedPAC(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        # 训练用 loader (Shuffle)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)
        # 统计用 loader (No Shuffle, 保证确定性)
        self.ldr_proto = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=False)

    def train(self, net, global_protos, lr=0.1):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5, weight_decay=self.args.weight_decay)
        
        # [优化] 将 global_protos 转换为 Tensor 矩阵以便向量化计算
        if len(global_protos) > 0:
            proto_dim = next(iter(global_protos.values())).shape[0]
            max_cls = max(global_protos.keys()) + 1
            # 创建原型矩阵 [num_classes, feature_dim]
            proto_tensor = torch.zeros(max_cls, proto_dim).to(self.args.device)
            # 创建掩码，标记哪些类有原型 [num_classes] (一维)
            proto_mask = torch.zeros(max_cls).to(self.args.device)
            for k, v in global_protos.items():
                proto_tensor[k] = v.to(self.args.device)
                proto_mask[k] = 1.0
        else:
            proto_tensor = None
        
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                
                output = net(images)
                # 兼容性解包
                log_probs, features = output if isinstance(output, tuple) else (output, None)
                
                loss_ce = self.loss_func(log_probs, labels)
                loss_proto = torch.tensor(0.0).to(self.args.device)
                
                # [优化] 向量化计算 FedPAC 原型对齐损失
                if features is not None and proto_tensor is not None:
                    features_norm = F.normalize(features, dim=1)
                    
                    # 根据 labels 索引取出对应的原型 [batch_size, feature_dim]
                    batch_protos = F.embedding(labels, proto_tensor)
                    # 取出掩码 [batch_size] (一维)
                    batch_mask = F.embedding(labels, proto_mask)
                    
                    # 计算 MSE: ||f - p||^2 per sample
                    mse = (features_norm - batch_protos).pow(2).sum(dim=1)  # [batch]
                    
                    # 只计算有效原型的 Loss
                    valid_count = batch_mask.sum()
                    if valid_count > 0:
                        loss_proto = (mse * batch_mask).sum() / valid_count
                
                # 总损失
                loss = loss_ce + self.args.lam_pac * loss_proto
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
            
            avg_loss = np.mean(batch_loss) if batch_loss else 0.0
            epoch_loss.append(avg_loss)

        return net.state_dict(), np.mean(epoch_loss) if epoch_loss else 0.0, self.get_local_protos(net)

    def get_local_protos(self, net):
        net.eval()
        local_protos = {}
        counts = {}
        with torch.no_grad():
            for images, labels in self.ldr_proto:
                images = images.to(self.args.device)
                output = net(images)
                features = output[1] if isinstance(output, tuple) else None
                
                if features is not None:
                    features_norm = F.normalize(features, dim=1)
                    for i, y in enumerate(labels):
                        y_val = y.item()
                        if y_val not in local_protos:
                            local_protos[y_val] = torch.zeros_like(features_norm[i].cpu())
                            counts[y_val] = 0
                        local_protos[y_val] += features_norm[i].cpu()
                        counts[y_val] += 1
        
        # 归一化并返回
        return {y: (local_protos[y] / counts[y]).float() for y in local_protos}

# === DFSL Update ===
class LocalUpdateDFSL(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)

    def train(self, net, net_teacher, lr=0.1):
        net.train()
        net_teacher.eval()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5, weight_decay=self.args.weight_decay)
        
        epoch_loss = []
        T = self.args.temp_dfsl

        for iter in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                
                out_s = net(images)
                out_s = out_s[0] if isinstance(out_s, tuple) else out_s
                
                with torch.no_grad():
                    out_t = net_teacher(images)
                    out_t = out_t[0] if isinstance(out_t, tuple) else out_t

                loss_ce = self.loss_ce(out_s, labels)
                
                # 知识蒸馏 KL 散度
                p_s = F.log_softmax(out_s / T, dim=1)
                p_t = F.softmax(out_t / T, dim=1)
                loss_kd = self.loss_kl(p_s, p_t) * (T**2)

                loss = loss_ce + self.args.mu_dfsl * loss_kd
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            
            avg_loss = np.mean(batch_loss) if batch_loss else 0.0
            epoch_loss.append(avg_loss)
            
        return net.state_dict(), np.mean(epoch_loss) if epoch_loss else 0.0

# === MAML Update (修复版) ===
class LocalUpdateMAML_Clean(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        # MAML 需要严格区分 Support/Query，drop_last=True 防止形状不匹配
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True, drop_last=True)

    def train(self, net, lr_inner=0.01, lr_outer=0.001):
        from torch.nn.utils.stateless import functional_call
        net.train()
        
        # MAML 外层优化器
        optimizer = torch.optim.SGD(net.parameters(), lr=lr_outer, momentum=0.5)
        
        # [修复] 更安全的迭代器处理，防止一个 epoch 数据不够分两组
        data_iter = iter(self.ldr_train)
        
        epoch_loss = []
        for iter_num in range(self.args.local_ep):
            try:
                supp_x, supp_y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.ldr_train)
                supp_x, supp_y = next(data_iter)
                
            try:
                query_x, query_y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.ldr_train)
                query_x, query_y = next(data_iter)

            supp_x, supp_y = supp_x.to(self.args.device), supp_y.to(self.args.device)
            query_x, query_y = query_x.to(self.args.device), query_y.to(self.args.device)

            # --- Inner Loop (Meta-Train) ---
            params = dict(net.named_parameters())
            
            # 第一次前向传播 (Support Set)
            out_s = functional_call(net, params, supp_x)
            loss_s = self.loss_func(out_s[0] if isinstance(out_s, tuple) else out_s, supp_y)
            
            # 计算一阶梯度 (create_graph=True for second derivative in outer loop)
            # allow_unused=True 防止某些参数未参与前向计算时报错
            grads = torch.autograd.grad(loss_s, params.values(), create_graph=True, allow_unused=True)
            
            # 更新临时参数 (Fast Weights)
            fast_params = {}
            for (name, param), grad in zip(params.items(), grads):
                if grad is not None:
                    fast_params[name] = param - lr_inner * grad
                else:
                    fast_params[name] = param

            # --- Outer Loop (Meta-Test) ---
            out_q = functional_call(net, fast_params, query_x)
            loss_q = self.loss_func(out_q[0] if isinstance(out_q, tuple) else out_q, query_y)
            
            # --- Meta Update ---
            optimizer.zero_grad()
            loss_q.backward()
            optimizer.step()
            
            epoch_loss.append(loss_q.item())
            
        return net.state_dict(), np.mean(epoch_loss) if epoch_loss else 0.0

class LocalUpdateMAML(object):

    def __init__(self, args, dataset=None, idxs=None, optim=None,indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        else:
            data_len = len(idxs)
            
            # === [核心修复] 动态 drop_last ===
            if data_len >= self.args.local_bs:
                use_drop_last = True
            else:
                use_drop_last = False
            
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), 
                                      batch_size=self.args.local_bs, 
                                      shuffle=True, 
                                      drop_last=use_drop_last)
        self.optim = optim
        if 'sent140' in self.args.dataset and indd == None:
            if not LANGUAGE_UTILS_AVAILABLE:
                raise ImportError("language_utils module is required for sent140 dataset")
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None

    def train(self, net, c_list={}, idx=-1, lr=0.1,lr_in=0.0001, c=False):
        net.train()
        # train and update
        lr_in = lr*0.001
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay': self.args.weight_decay},
            {'params': bias_p, 'weight_decay': 0}
        ],
        lr=lr, momentum=0.5
        )
        
        local_eps = self.args.local_ep
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(2)
        for iter in range(local_eps):
            batch_loss = []
            if num_updates == self.args.local_updates:
                break
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)

                    split = self.args.local_bs 
                    sup_x, sup_y = data.to(self.args.device), targets.to(self.args.device)
                    targ_x, targ_y = data.to(self.args.device), targets.to(self.args.device)

                    param_dict = dict()
                    for name, param in net.named_parameters():
                        if param.requires_grad:
                            if "norm_layer" not in name:
                                param_dict[name] = param.to(device=self.args.device)
                    names_weights_copy = param_dict

                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    log_probs_sup = net(sup_x, hidden_train)
                    loss_sup = self.loss_func(log_probs_sup,sup_y)
                    grads = torch.autograd.grad(loss_sup, names_weights_copy.values(),
                                                    create_graph=True, allow_unused=True)
                    names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

                    for key, grad in names_grads_copy.items():
                        if grad is None:
                            print('Grads not found for inner loop parameter', key)
                        names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
                    for key in names_grads_copy.keys():
                        names_weights_copy[key] = names_weights_copy[key]- lr_in * names_grads_copy[key]

                    log_probs_targ = net(targ_x)
                    loss_targ = self.loss_func(log_probs_targ,targ_y)
                    loss_targ.backward()
                    optimizer.step()
                        
                    del log_probs_targ.grad
                    del loss_targ.grad
                    del loss_sup.grad
                    del log_probs_sup.grad
                    optimizer.zero_grad()
                    net.zero_grad()

                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    split = int(8* images.size()[0]/10)
                    sup_x, sup_y = images[:split].to(self.args.device), labels[:split].to(self.args.device)
                    targ_x, targ_y = images[split:].to(self.args.device), labels[split:].to(self.args.device)

                    param_dict = dict()
                    for name, param in net.named_parameters():
                        if param.requires_grad:
                            if "norm_layer" not in name:
                                param_dict[name] = param.to(device=self.args.device)
                    names_weights_copy = param_dict

                    net.zero_grad()
                    log_probs_sup = net(sup_x)
                    loss_sup = self.loss_func(log_probs_sup,sup_y)
                    if loss_sup != loss_sup:
                        continue
                    grads = torch.autograd.grad(loss_sup, names_weights_copy.values(),
                                                    create_graph=True, allow_unused=True)
                    names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
                        
                    for key, grad in names_grads_copy.items():
                        if grad is None:
                            print('Grads not found for inner loop parameter', key)
                        names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
                    for key in names_grads_copy.keys():
                        names_weights_copy[key] = names_weights_copy[key]- lr_in * names_grads_copy[key]
                        
                    loss_sup.backward(retain_graph=True)
                    log_probs_targ = net(targ_x)
                    loss_targ = self.loss_func(log_probs_targ,targ_y)
                    loss_targ.backward()
                    optimizer.step()
                    del log_probs_targ.grad
                    del loss_targ.grad
                    del loss_sup.grad
                    del log_probs_sup.grad
                    optimizer.zero_grad()
                    net.zero_grad()
 
                batch_loss.append(loss_sup.item())
                num_updates += 1
                if num_updates == self.args.local_updates:
                    break
                batch_loss.append(loss_sup.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # === [修复] 防止除以零错误 ===
        avg_loss = sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0
        # ===============================
        return net.state_dict(), avg_loss, self.indd#, num_updates


class LocalUpdateScaffold(object):

    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            data_len = len(idxs)
            
            # === [核心修复] 动态 drop_last ===
            if data_len >= self.args.local_bs:
                use_drop_last = True
            else:
                use_drop_last = False
            
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), 
                                      batch_size=self.args.local_bs, 
                                      shuffle=True, 
                                      drop_last=use_drop_last)
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None

    def train(self, net, c_list={}, idx=-1, lr=0.1, c=False):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay': self.args.weight_decay},
            {'params': bias_p, 'weight_decay': 0}
        ],
        lr=lr, momentum=0.5
        )
        
        local_eps = self.args.local_ep

        epoch_loss=[]
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            batch_loss = []
            if num_updates == self.args.local_updates:
                break
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()

                    hidden_train = repackage_hidden(hidden_train)
                    output, feature = net(data, hidden_train)  # RNN现在返回(output, feature)
                    hidden_train = net.last_hidden  # 获取更新后的hidden state
                    loss_fi = self.loss_func(output, torch.max(targets, 1)[1])  # output已经是.t()的结果
                    w = net.state_dict()
                    local_par_list = None
                    dif = None
                    for param in net.parameters():
                        if not isinstance(local_par_list, torch.Tensor):
                            local_par_list = param.reshape(-1)
                        else:
                            local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                    for k in c_list[idx].keys():
                        if not isinstance(dif, torch.Tensor):
                            dif = (-c_list[idx][k] +c_list[-1][k]).reshape(-1)
                        else:
                            dif = torch.cat((dif, (-c_list[idx][k]+c_list[-1][k]).reshape(-1)),0)
                    loss_algo = torch.sum(local_par_list * dif)
                    loss = loss_fi + loss_algo
                    
                    loss.backward()
                    optimizer.step()

                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)

                    log_probs = net(images)
                    loss_fi = self.loss_func(log_probs, labels)
                    w = net.state_dict()
                    local_par_list = None
                    dif = None
                    for param in net.parameters():
                        if not isinstance(local_par_list, torch.Tensor):
                            local_par_list = param.reshape(-1)
                        else:
                            local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                    for k in c_list[idx].keys():
                        if not isinstance(dif, torch.Tensor):
                            dif = (-c_list[idx][k] +c_list[-1][k]).reshape(-1)
                        else:
                            dif = torch.cat((dif, (-c_list[idx][k]+c_list[-1][k]).reshape(-1)),0)
                    loss_algo = torch.sum(local_par_list * dif)
                    loss = loss_fi + loss_algo
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)
                    optimizer.step()

                num_updates += 1
                if num_updates == self.args.local_updates:
                    break
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # === [修复] 防止除以零错误 ===
        avg_loss = sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0
        # ===============================
        return net.state_dict(), avg_loss, self.indd, num_updates

class LocalUpdateAPFL(object):

    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        else:
            data_len = len(idxs)
            
            # === [核心修复] 动态 drop_last ===
            if data_len >= self.args.local_bs:
                use_drop_last = True
            else:
                use_drop_last = False
            
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), 
                                      batch_size=self.args.local_bs, 
                                      shuffle=True, 
                                      drop_last=use_drop_last)
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None

    def train(self, net,ind=None,w_local=None, idx=-1, lr=0.1):
        net.train()
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay': self.args.weight_decay},
            {'params': bias_p, 'weight_decay': 0}
        ],
        lr=lr, momentum=0.5
        )
        
        # train and update
        local_eps = self.args.local_ep
        args = self.args
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            batch_loss = []
            if num_updates >= self.args.local_updates:
                break
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if  'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    w_loc_new = {}
                    w_glob = copy.deepcopy(net.state_dict())
                    for k in net.state_dict().keys():
                        w_loc_new[k] = self.args.alpha_apfl*w_local[k] + self.args.alpha_apfl*w_glob[k]

                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, feature = net(data, hidden_train)  # RNN现在返回(output, feature)
                    hidden_train = net.last_hidden  # 获取更新后的hidden state
                    loss = self.loss_func(output, torch.max(targets, 1)[1])  # output已经是.t()的结果
                    optimizer.zero_grad()
                    loss.backward()
                        
                    optimizer.step()
                    optimizer.zero_grad()
                    wt = copy.deepcopy(net.state_dict())
                    net.zero_grad()

                    del hidden_train
                    hidden_train = net.init_hidden(self.args.local_bs)

                    net.load_state_dict(w_loc_new)
                    output, feature = net(data, hidden_train)  # RNN现在返回(output, feature)
                    hidden_train = net.last_hidden  # 获取更新后的hidden state
                    loss = self.args.alpha_apfl*self.loss_func(output, torch.max(targets, 1)[1])  # output已经是.t()的结果
                    loss.backward()
                    optimizer.step()
                    w_local_bar = net.state_dict()
                    for k in w_local_bar.keys():
                        w_local[k] = w_local_bar[k] - w_loc_new[k] + w_local[k]

                    net.load_state_dict(wt)
                    optimizer.zero_grad()
                    del wt
                    del w_loc_new
                    del w_glob
                    del w_local_bar
                    
                else:
                        
                    w_loc_new = {} 
                    w_glob = copy.deepcopy(net.state_dict())
                    for k in net.state_dict().keys():
                        w_loc_new[k] = self.args.alpha_apfl*w_local[k] + self.args.alpha_apfl*w_glob[k]

                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                        
                    optimizer.step()
                    wt = copy.deepcopy(net.state_dict())

                    net.load_state_dict(w_loc_new)
                    log_probs = net(images)
                    loss = self.args.alpha_apfl*self.loss_func(log_probs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    w_local_bar = net.state_dict()
                    for k in w_local_bar.keys():
                        w_local[k] = w_local_bar[k] - w_loc_new[k] + w_local[k]

                    net.load_state_dict(wt)
                    optimizer.zero_grad()
                    del wt
                    del w_loc_new
                    del w_glob
                    del w_local_bar

                num_updates += 1
                if num_updates >= self.args.local_updates:
                    break

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # === [修复] 防止除以零错误 ===
        avg_loss = sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0
        # ===============================
        return net.state_dict(),w_local, avg_loss, self.indd

class LocalUpdateDitto(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        else:
            data_len = len(idxs)
            
            # === [核心修复] 动态 drop_last ===
            if data_len >= self.args.local_bs:
                use_drop_last = True
            else:
                use_drop_last = False
            
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), 
                                      batch_size=self.args.local_bs, 
                                      shuffle=True, 
                                      drop_last=use_drop_last)
        
        if 'sent140' in self.args.dataset and indd == None:
            from models.language_utils import get_word_emb_arr
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None
    
    def train(self, net, ind=None, w_ditto=None, lam=0, idx=-1, lr=0.1, last=False):
        net.train()
        
        # [关键点1] 优化器配置：过滤掉 BN/GroupNorm 层，防止 Weight Decay 破坏统计量
        decay_params = []
        no_decay_params = []
        for name, param in net.named_parameters():
            if not param.requires_grad: continue
            # 更安全的匹配：包含bias、bn(BatchNorm)、norm(GroupNorm/LayerNorm)
            if 'bias' in name or '.bn' in name or '.norm' in name or 'weight' in name and any(layer_type in name for layer_type in ['bn', 'norm']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # 使用传入的 lr，而不是 args.lr
        optimizer = torch.optim.SGD([
            {'params': decay_params, 'weight_decay': self.args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=lr, momentum=0.5)

        epoch_task_loss = [] # [修改] 改名，明确记录的是任务损失
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                
                # [关键点2] Tuple 拆包，兼容 ResNet 返回 (logits, features)
                output = net(images)
                if isinstance(output, tuple):
                    output = output[0]
                
                # 计算任务 Loss
                loss_task = self.loss_func(output, labels)
                
                # [关键点3] Ditto 正则项写入 Loss 函数
                # 让 Autograd 自动处理梯度，这样 LR 衰减时正则项也会同步衰减！
                loss_algo = 0.0
                if w_ditto is not None and lam > 0:
                    for name, param in net.named_parameters():
                        if param.requires_grad and name in w_ditto:
                            w_g = w_ditto[name].to(self.args.device)
                            # 0.5 * lambda * || w - w_global ||^2
                            loss_algo += 0.5 * lam * torch.norm(param - w_g) ** 2
                
                loss = loss_task + loss_algo
                loss.backward()
                optimizer.step()
                
                # [核心修改] 只记录 loss_task，不记录包含正则项的总 loss
                # 这样画出来的曲线就是下降的了
                batch_loss.append(loss_task.item()) 
            
            if len(batch_loss) > 0:
                epoch_task_loss.append(sum(batch_loss)/len(batch_loss))
            else:
                epoch_task_loss.append(0.0)

        # 返回的是纯净的分类损失
        return net.state_dict(), sum(epoch_task_loss) / len(epoch_task_loss), self.indd

# Generic local update class, implements local updates for FedRep, FedPer, LG-FedAvg, FedAvg, FedProx
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        else:
            data_len = len(idxs)
            
            # === [核心修复] 动态 drop_last ===
            # 如果数据量 < Batch Size (例如 1 个样本)，必须关闭 drop_last，否则 DataLoader 为空
            if data_len >= self.args.local_bs:
                use_drop_last = True  # 数据充足时，丢弃最后不完整的 batch 以稳定梯度
            else:
                use_drop_last = False # 数据极少时，强制保留所有数据
            
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), 
                                      batch_size=self.args.local_bs, 
                                      shuffle=True, 
                                      drop_last=use_drop_last)
            
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None        
        
        self.dataset=dataset
        self.idxs=idxs

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1, w_glob=None, gamma=2.0, global_protos=None):
        
        # 准备损失函数
        loss_mse = nn.MSELoss()

        # FedProto 独立版逻辑 (保持不动，防止冲突)
        if self.args.alg == 'fedproto':
            # FedProto 依然通过 w_glob 传原型，不动它
            global_protos_fedproto = w_glob

        # 优化器设置
        if self.args.alg == 'crossfreeze':
            # === [修复] CrossFreeze 也需要排除 bias/norm 的 weight_decay ===
            decay_params_cf = []
            no_decay_params_cf = []
            for name, param in net.named_parameters():
                # 1. 所有 bias 都不 decay
                if 'bias' in name:
                    no_decay_params_cf.append(param)
                # 2. Norm 层的 weight (scale) 不 decay
                elif 'bn' in name or 'norm' in name:
                    no_decay_params_cf.append(param)
                # 3. 其他权重 (conv weight, fc weight) 正常 decay
                else:
                    decay_params_cf.append(param)
            optimizer = torch.optim.SGD([
                {'params': decay_params_cf, 'weight_decay': self.args.weight_decay},
                {'params': no_decay_params_cf, 'weight_decay': 0.0}
            ], lr=lr, momentum=0.5)
        elif self.args.alg == 'fedproto':
            # FedProto 添加 weight_decay，防止过拟合
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5, weight_decay=self.args.weight_decay)
        # else:
        #     # 其他算法的优化器
        #     bias_p=[]
        #     weight_p=[]
        #     for name, p in net.named_parameters():
        #         if 'bias' in name:
        #             bias_p += [p]
        #         else:
        #             weight_p += [p]
        #     optimizer = torch.optim.SGD(
        #     [     
        #         {'params': weight_p, 'weight_decay':0.0001},
        #         {'params': bias_p, 'weight_decay':0}
        #     ],
        #     lr=lr, momentum=0.5
        #     )
        else:
            # === [修复] 针对 FedAvg/FedRep 的稳健优化器 ===
            # 排除 BatchNorm/GroupNorm 的 weight 参数不进行 decay
            decay_params = []
            no_decay_params = []
            
            for name, param in net.named_parameters():
                if not param.requires_grad:
                    continue
                
                # 1. 所有 bias 都不 decay
                if 'bias' in name:
                    no_decay_params.append(param)
                # 2. Norm 层的 weight (scale) 也不应该 decay
                elif 'bn' in name or 'norm' in name:
                    no_decay_params.append(param)
                # 3. 其他权重 (conv weight, fc weight) 正常 decay
                else:
                    decay_params.append(param)

            optimizer = torch.optim.SGD([
                {'params': decay_params, 'weight_decay': self.args.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ], lr=lr, momentum=0.5)

        # 确保接收到的全局参数被加载到 sm_head
        if self.args.alg == 'crossfreeze' and w_glob is not None:
            if 'sm_head.weight' in w_glob:
                net.sm_head.weight.data = w_glob['sm_head.weight'].clone().to(self.args.device)
                net.sm_head.bias.data = w_glob['sm_head.bias'].clone().to(self.args.device)

        # 在 if last 之前定义 local_eps，防止未定义变量错误
        local_eps = self.args.local_ep

        if last:
            if self.args.alg =='fedavg' or self.args.alg == 'prox':
                local_eps= 10
                net_keys = [*net.state_dict().keys()]
                # 针对不同模型类型的兼容性处理
                if hasattr(net, 'weight_keys') and len(net.weight_keys) > 4:
                    # 传统CNN模型（如MLP）有足够的weight_keys
                    if 'cifar' in self.args.dataset:
                        w_glob_keys = [net.weight_keys[i] for i in [0,1,3,4]]
                    elif 'mnist' in self.args.dataset:
                        w_glob_keys = [net.weight_keys[i] for i in [0,1,2]]
                    else:
                        w_glob_keys = net.weight_keys
                else:
                    # ResNet或其他模型，使用所有参数键
                    if 'sent140' in self.args.dataset:
                        w_glob_keys = [net_keys[i] for i in [0,1,2,3,4,5]]
                    else:
                        # 对于ResNet等模型，FedAvg聚合所有参数
                        w_glob_keys = [[key] for key in net_keys]
            elif 'maml' in self.args.alg:
                local_eps = 5
                w_glob_keys = []
            
            # CrossFreeze 保持原样，不要突然增加轮次
            elif self.args.alg == 'crossfreeze':
                local_eps = self.args.local_ep 
            
            else:
                # FedRep 等算法保留此逻辑
                local_eps =  max(10,local_eps-self.args.local_rep_ep)
        
        head_eps = local_eps-self.args.local_rep_ep
        
        # 1. 计算动态 Phase 阈值 (适配任意 local_ep，例如 10 轮 -> 2-6-2 分配)
        len_a = max(1, int(local_eps * 0.2)) 
        len_c = max(1, int(local_eps * 0.2))
        len_b = local_eps - len_a - len_c
        end_a = len_a
        end_b = len_a + len_b

        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)

        for iter in range(local_eps):
            done = False
            
            # 每轮强制进入 train 模式
            # 即使参数被冻结 (requires_grad=False)，BN 层的统计量 (running_mean/var) 仍需更新
            # 这是从头训练 (Training from Scratch) 成功的关键
            net.train()

            # 初始化 CrossFreeze 阶段标志
            is_phase_a = False
            is_phase_b = False
            is_phase_c = False

            # >>>>>>>>> 算法调度逻辑 <<<<<<<<<
            if self.args.alg == 'crossfreeze':
                if iter < end_a:
                    # Phase A: M2 预热 (20%)
                    is_phase_a = True
                    # 冻结 Body/Sm, 激活 M2
                    for p in net.body.parameters(): p.requires_grad = False
                    for p in net.fc.parameters(): p.requires_grad = True
                    for p in net.sm_head.parameters(): p.requires_grad = False
                    
                elif iter < end_b:
                    # Phase B: Body 对齐 (60%)
                    is_phase_b = True
                    # 激活 Body, 冻结 M2/Sm (作为锚点)
                    for p in net.body.parameters(): p.requires_grad = True
                    for p in net.fc.parameters(): p.requires_grad = False
                    for p in net.sm_head.parameters(): p.requires_grad = False
                    
                else:
                    # Phase C: Sm 校准 (20%)
                    is_phase_c = True
                    # 冻结 Body/M2, 激活 Sm
                    for p in net.body.parameters(): p.requires_grad = False
                    for p in net.fc.parameters(): p.requires_grad = False
                    for p in net.sm_head.parameters(): p.requires_grad = True

                # === [修复] Phase 切换边界时清零 SGD 动量缓存 ===
                # 避免上一 phase 的 momentum_buffer 污染新 phase 的优化方向
                if iter == 0 or iter == end_a or iter == end_b:
                    optimizer.state.clear()

            # FedRep 逻辑 (保持兼容)
            elif (iter < head_eps and self.args.alg == 'fedrep') or last:
                # 训练 Head，冻结 Body
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            
            # 使用 >= 确保后续轮次状态正确
            elif iter >= head_eps and self.args.alg == 'fedrep' and not last:
                # 训练 Body，冻结 Head
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            # 其他算法 (FedAvg, FedProto等) 所有参数都更新
            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                      param.requires_grad = True 
       
            # Batch 训练循环
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels,self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break
                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, feature = net(data, hidden_train)  # RNN现在返回(output, feature)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
                    optimizer.step()
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()

                    if self.args.alg == 'crossfreeze':
                        # CrossFreeze Loss 计算
                        out_local, feat_local = net(images, mode='local') # M2 path
                        out_glob, _ = net(images, mode='global')    # Sm path
                        
                        # 初始化原型 Loss
                        loss_proto = torch.tensor(0.0).to(self.args.device)
                        
                        # Phase B 且有原型时计算 Proto Loss
                        if (global_protos is not None) and is_phase_b:
                            proto_count = 0 
                            for i, label in enumerate(labels):
                                y = label.item()
                                if y in global_protos:
                                    target_proto = global_protos[y].to(self.args.device).detach()
                                    loss_proto += loss_mse(feat_local[i], target_proto)
                                    proto_count += 1
                            if proto_count > 0:
                                loss_proto = (loss_proto / proto_count) * self.args.ld
                        
                        # 根据 Phase 组合 Loss
                        if is_phase_a: 
                            loss = self.loss_func(out_local, labels)
                            ce_loss_for_plot = loss  # 只有 CE 损失
                        elif is_phase_b: 
                            loss_local = self.loss_func(out_local, labels)
                            loss_glob = self.loss_func(out_glob, labels)
                            
                            # === [修复] Gamma 渐变启动而非直接跳变 ===
                            warmup_rounds = self.args.epochs * 0.25
                            full_gamma = float(self.args.gamma)
                            current_gamma = full_gamma * min(1.0, ind / warmup_rounds)
                            
                            # 核心：Gamma 强约束 + 原型约束
                            loss = loss_local + current_gamma * loss_glob + loss_proto
                            ce_loss_for_plot = loss_local  # 只记录 M2 的 CE 损失
                        else: # is_phase_c
                            loss = self.loss_func(out_glob, labels)
                            ce_loss_for_plot = loss  # 只有 CE 损失
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
                        optimizer.step()

                    elif self.args.alg == 'fedproto':
                        # FedProto 逻辑
                        out = net(images)
                        if isinstance(out, tuple): log_probs, features = out
                        else: log_probs, features = out, None
                        
                        loss_ce = self.loss_func(log_probs, labels)
                        loss_proto = torch.tensor(0.0).to(self.args.device)
                        protos_to_use = global_protos if global_protos is not None else global_protos_fedproto

                        if protos_to_use is not None and features is not None:
                            proto_count = 0
                            for i, label in enumerate(labels):
                                y = label.item()
                                if y in protos_to_use:
                                    target = protos_to_use[y].to(self.args.device).detach()
                                    loss_proto += loss_mse(features[i], target)
                                    proto_count += 1
                            if proto_count > 0:
                                loss_proto = (loss_proto / proto_count) * self.args.ld
                        
                        loss = loss_ce + loss_proto
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
                        optimizer.step()

                    else:
                        # 其他算法 (FedAvg, FedRep等)
                        output = net(images)
                        if isinstance(output, tuple): log_probs = output[0]
                        else: log_probs = output
                        loss = self.loss_func(log_probs, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                num_updates += 1
                # === [修复] CrossFreeze 和 Ditto 记录纯 CE 损失用于绘图 ===
                if self.args.alg == 'crossfreeze':
                    batch_loss.append(ce_loss_for_plot.item())
                else:
                    batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            else:
                epoch_loss.append(0.0)
            
            if done:
                break
        
        # 计算本地原型 (CrossFreeze 和 FedProto 共用逻辑)
        local_protos = {}
        if self.args.alg in ['crossfreeze', 'fedproto']:
            net.eval()
            counts = {}
            with torch.no_grad():
                for images, labels in self.ldr_train:
                    images = images.to(self.args.device)
                    # CrossFreeze 需要 mode='local' 或者默认来拿特征，FedProto 只需要 call
                    if self.args.alg == 'crossfreeze':
                        _, features = net(images, mode='local')
                    else:
                        out = net(images)
                        if isinstance(out, tuple): _, features = out
                        else: features = None
                    
                    if features is not None:
                        for i, label in enumerate(labels):
                            y = label.item()
                            if y not in local_protos:
                                local_protos[y] = torch.zeros_like(features[i].cpu())
                                counts[y] = 0
                            local_protos[y] += features[i].cpu()
                            counts[y] += 1
            
            for y in local_protos:
                if counts[y] > 0:
                    local_protos[y] /= counts[y]

        avg_loss = sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0

        # 返回值处理
        if self.args.alg == 'fedproto':
            # FedProto 独立版返回格式 (保持不变)
            return (local_protos, net.state_dict()), avg_loss, self.indd
        elif self.args.alg == 'crossfreeze':
            # CrossFreeze 返回格式：正常权重 + 额外的 local_protos
            return net.state_dict(), avg_loss, self.indd, local_protos
        else:
            return net.state_dict(), avg_loss, self.indd

class LocalUpdateMTL(object):
    def __init__(self, args, dataset=None, idxs=None,indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        else:
            data_len = len(idxs)
            
            # === [核心修复] 动态 drop_last ===
            if data_len >= self.args.local_bs:
                use_drop_last = True
            else:
                use_drop_last = False
            
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), 
                                      batch_size=self.args.local_bs, 
                                      shuffle=True, 
                                      drop_last=use_drop_last)

        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        else:
            self.indd=indd

    def train(self, net, lr=0.1, omega=None, W_glob=None, idx=None, w_glob_keys=None):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name or name in w_glob_keys:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay': self.args.weight_decay},
            {'params': bias_p, 'weight_decay': 0}
        ],
        lr=lr, momentum=0.5
        )

        epoch_loss = []
        local_eps = self.args.local_ep
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels,self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()

                    hidden_train = repackage_hidden(hidden_train)
                    output, feature = net(data, hidden_train)  # RNN现在返回(output, feature)
                    hidden_train = net.last_hidden  # 获取更新后的hidden state
                    loss = self.loss_func(output, torch.max(targets, 1)[1])  # output已经是.t()的结果
                    W = W_glob.clone()
                    W_local = [net.state_dict(keep_vars=True)[key].flatten() for key in w_glob_keys]
                    W_local = torch.cat(W_local)
                    W[:, idx] = W_local

                    loss_regularizer = 0
                    loss_regularizer += W.norm() ** 2

                    k = 4000
                    for i in range(W.shape[0] // k):
                        x = W[i * k:(i+1) * k, :]
                        loss_regularizer += x.mm(omega).mm(x.T).trace()
                    f = (int)(math.log10(W.shape[0])+1) + 1
                    loss_regularizer *= 10 ** (-f)

                    loss = loss + loss_regularizer
                    loss.backward()
                    optimizer.step()
                
                else:
                
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    W = W_glob.clone().to(self.args.device)
                    W_local = [net.state_dict(keep_vars=True)[key].flatten() for key in w_glob_keys]
                    W_local = torch.cat(W_local)
                    W[:, idx] = W_local

                    loss_regularizer = 0
                    loss_regularizer += W.norm() ** 2

                    k = 4000
                    for i in range(W.shape[0] // k):
                        x = W[i * k:(i+1) * k, :]
                        loss_regularizer += x.mm(omega).mm(x.T).trace()
                    f = (int)(math.log10(W.shape[0])+1) + 1
                    loss_regularizer *= 10 ** (-f)

                    loss = loss + loss_regularizer
                    loss.backward()
                    optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # === [修复] 防止除以零错误 ===
        avg_loss = sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0
        # ===============================
        return net.state_dict(), avg_loss, self.indd

class LocalUpdatePFedEdit(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        
        # 1. 划分数据集：D_train (用于训练) 和 D_p (用于编辑评估)
        total_len = len(idxs)
        subset_len = int(self.args.subset_ratio * total_len)
        if subset_len < 1: subset_len = 1
        train_len = total_len - subset_len
        
        # 为了保证随机性，打乱 idxs
        idxs_list = list(idxs)
        np.random.shuffle(idxs_list)
        
        idxs_subset = idxs_list[:subset_len] # D_p
        idxs_train = idxs_list[subset_len:]  # D_train_star (剩余部分用于训练)
        
        # === [核心修复] 动态 drop_last 处理两个 DataLoader ===
        train_data_len = len(idxs_train)
        subset_data_len = len(idxs_subset)
        
        if train_data_len >= self.args.local_bs:
            use_drop_last_train = True
        else:
            use_drop_last_train = False
            
        if subset_data_len >= self.args.local_bs:
            use_drop_last_subset = True
        else:
            use_drop_last_subset = False
        
        self.ldr_val = DataLoader(DatasetSplit(dataset, idxs_subset), 
                                 batch_size=self.args.local_bs, 
                                 shuffle=False, 
                                 drop_last=use_drop_last_subset)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs_train), 
                                   batch_size=self.args.local_bs, 
                                   shuffle=True, 
                                   drop_last=use_drop_last_train)
        
        self.indd = indd

    def calculate_metrics(self, net, device):
        """
        在 D_p 上进行推理，收集每个样本的 (Prediction Correctness, Probability of GT class)
        """
        net.eval()
        metrics = []
        with torch.no_grad():
            for images, labels in self.ldr_val:
                images, labels = images.to(device), labels.to(device)
                
                # === [修复 1] 处理元组返回值 ===
                outputs = net(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0] # 只取 logits
                # ===========================
                
                probs = F.softmax(outputs, dim=1)
                
                # 获取 Ground Truth 类的概率 P(y|theta, x)
                gt_probs = probs.gather(1, labels.view(-1, 1)).squeeze()
                
                # 获取预测结果
                preds = outputs.argmax(dim=1)
                is_correct = preds.eq(labels)
                
                # 处理 batch size 为 1 的情况 (squeeze 后变成 0 维)
                if labels.dim() == 0 or (labels.dim() == 1 and len(labels) == 1):
                     metrics.append({
                        'gt_prob': gt_probs.item(),
                        'is_correct': is_correct.item(),
                    })
                else:
                    for i in range(len(labels)):
                        metrics.append({
                            'gt_prob': gt_probs[i].item(),
                            'is_correct': is_correct[i].item(),
                        })
        return metrics

    def train(self, net_glob, net_local, lr=0.1):
        """
        Block-wise pFedEdit (最终优化版)
        1. 修正 nn.Sequential 命名解析
        2. 排除无关的 sm_head
        """
        device = self.args.device
        
        # === Step 1: 评估基准 (Clean Global) ===
        base_metrics = self.calculate_metrics(net_glob, device)
        
        # === Step 2: 将参数按“块”分组 ===
        block_map = {}
        for name, _ in net_glob.named_parameters():
            
            # [优化] 排除 sm_head，因为它不影响本地评估指标
            if 'sm_head' in name:
                continue
                
            parts = name.split('.')
            
            # 处理 body 内部层 (body.0, body.3 等)
            if parts[0] == 'body' and len(parts) > 1:
                prefix = f"{parts[0]}.{parts[1]}" 
            else:
                prefix = parts[0]
            
            if prefix not in block_map:
                block_map[prefix] = []
            block_map[prefix].append(name)
            
        # print(f"DEBUG: Optimization Blocks: {list(block_map.keys())}") 

        candidates = []
        local_state = net_local.state_dict()
        global_state = net_glob.state_dict()
        
        # === Step 3: 遍历“块”进行评估 ===
        for block_name, param_names in block_map.items():
            
            if not all(k in local_state for k in param_names):
                continue
                
            # 1. 备份
            original_params_cache = {k: global_state[k].clone() for k in param_names}
            
            # 2. 替换
            for k in param_names:
                global_state[k].copy_(local_state[k])
            
            # 3. 评估
            edited_metrics = self.calculate_metrics(net_glob, device)
            
            # 4. 恢复
            for k in param_names:
                global_state[k].copy_(original_params_cache[k])
            
            # 5. 计算分数
            counts = [0, 0, 0, 0] 
            for i in range(len(base_metrics)):
                p_old = base_metrics[i]['gt_prob']
                p_new = edited_metrics[i]['gt_prob']
                is_correct = edited_metrics[i]['is_correct']
                te_positive = p_new > p_old
                
                if is_correct and te_positive: counts[0] += 1
                elif is_correct and not te_positive: counts[1] += 1
                elif not is_correct and te_positive: counts[2] += 1
                else: counts[3] += 1
            
            candidates.append((block_name, tuple(counts)))

        # === Step 4: 排序并选择 Top-k ===
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # edit_ratio 0.4 对于 block-wise 来说比较合适
        num_edit = int(self.args.edit_ratio * len(candidates))
        if num_edit < 1: num_edit = 1
        
        top_k_blocks = [x[0] for x in candidates[:num_edit]]
        
        # === Step 5: 应用编辑 ===
        for block_name in top_k_blocks:
            params_to_edit = block_map[block_name]
            for k in params_to_edit:
                global_state[k].copy_(local_state[k])
            
        # === Step 6: 本地微调 ===
        net_glob.train()
        bias_p, weight_p = [], []
        for name, p in net_glob.named_parameters():
            if 'bias' in name: bias_p += [p]
            else: weight_p += [p]
        optimizer = torch.optim.SGD(
            [{'params': weight_p, 'weight_decay': self.args.weight_decay},
             {'params': bias_p, 'weight_decay': 0}],
            lr=lr, momentum=0.5
        )
        
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(device), labels.to(device)
                net_glob.zero_grad()
                output = net_glob(images)
                if isinstance(output, tuple): output = output[0]
                loss = self.loss_func(output, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            else:
                epoch_loss.append(0.0)
            
        return net_glob.state_dict(), sum(epoch_loss)/len(epoch_loss) if len(epoch_loss)>0 else 0.0

class LocalUpdateMOON(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # 数据加载逻辑保持一致
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
             self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        else:
            data_len = len(idxs)
            
            # === [核心修复] 动态 drop_last ===
            if data_len >= self.args.local_bs:
                use_drop_last = True
            else:
                use_drop_last = False
            
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), 
                                      batch_size=self.args.local_bs, 
                                      shuffle=True, 
                                      drop_last=use_drop_last)
        self.indd = indd

    def train(self, net, net_glob, net_prev, lr=0.1):
        net.train()
        net_glob.eval()
        net_prev.eval()
        
        # 优化器
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5, weight_decay=self.args.weight_decay)

        epoch_loss = []
        
        # Cosine Similarity 用于对比损失
        cos = nn.CosineSimilarity(dim=-1)
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                
                # 1. 获取当前模型的输出和特征
                # 注意：这里假设 forward 返回 (logits, features)
                # 兼容 ResNet18_CrossFreeze 的 mode='local'
                if self.args.model == 'resnet':
                    out, pro1 = net(images, mode='local')
                else:
                    out, pro1 = net(images)

                # 2. 计算监督损失 (Cross Entropy)
                loss_sup = self.loss_func(out, labels)

                # 3. 获取全局模型特征 (Positive Key)
                with torch.no_grad():
                    if self.args.model == 'resnet':
                        _, pro2 = net_glob(images, mode='local')
                    else:
                        _, pro2 = net_glob(images)
                
                # 4. 获取上一轮本地模型特征 (Negative Key)
                with torch.no_grad():
                    if self.args.model == 'resnet':
                        _, pro3 = net_prev(images, mode='local')
                    else:
                        _, pro3 = net_prev(images)

                # 5. 计算 MOON 对比损失 (Model-Contrastive Loss)
                # L_con = -log( exp(sim(z, z_glob)/t) / (exp(sim(z, z_glob)/t) + exp(sim(z, z_prev)/t)) )
                
                pos_sim = cos(pro1, pro2)
                neg_sim = cos(pro1, pro3)
                
                logits = torch.cat((pos_sim.reshape(-1, 1), neg_sim.reshape(-1, 1)), dim=1)
                logits /= self.args.temperature
                
                # 对比损失的目标是让 index 0 (Global) 概率最大
                labels_con = torch.zeros(images.size(0)).long().to(self.args.device)
                loss_con = self.loss_func(logits, labels_con)

                # 6. 总损失
                loss = loss_sup + self.args.mu_moon * loss_con

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            else:
                epoch_loss.append(0.0)

        return net.state_dict(), sum(epoch_loss)/len(epoch_loss)