#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
import numpy as np
import torch

def noniid(dataset, num_users, shard_per_user, num_classes, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # === [新增] 健壮性检查 ===
    if shard_per_user == 0:
        raise ValueError("shard_per_user cannot be 0. Use Dirichlet distribution instead (set shard_per_user=0 and call dirichlet_split_noniid).")
    
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(shard_per_user * num_users / num_classes)
    samples_per_user = int( count/num_users )
    # whether to sample more test samples per user
    if (samples_per_user < 100):
        double = True
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            if (samples_per_user < 100):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        test.append(value)
    test = np.concatenate(test)

    return dict_users, rand_set_all

def dirichlet_split_noniid(train_labels, alpha, n_clients, return_dist=False, use_dist=None):
    '''
    参数:
        train_labels: 标签列表 (Numpy array)
        alpha: 分布系数
        n_clients: 客户端数
        return_dist: 是否返回生成的分布概率矩阵
        use_dist: 是否使用指定的分布概率矩阵
    '''
    if not isinstance(train_labels, np.ndarray):
        train_labels = np.array(train_labels)
    
    # 获取真实类别数
    classes = np.unique(train_labels)
    n_classes = len(classes)

    # 1. 生成或复用分布矩阵
    if use_dist is not None:
        label_distribution = use_dist
    else:
        # 生成分布矩阵(n_classes, n_clients),每一列代表一个客户端在各个类别上的概率分布
        label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten() for y in classes]
    client_id_map = {i: [] for i in range(n_clients)}
    
    # 2. 使用 Multinomial 分配 + Shuffle + 归一化保护
    for c, fracs in zip(class_idcs, label_distribution):
        # 重新归一化，防止浮点数精度问题导致 sum != 1 报错
        fracs = fracs / fracs.sum()
        
        # 先打乱，防止原始数据排序引入偏差
        np.random.shuffle(c)
        
        total_size = len(c)
        # 使用多项分布确定每个客户端分多少个该类样本，把每个类的样本数分配到各个客户端上，保证每个客户端的样本数符合 Dirichlet 分布
        num_samples_per_client = np.random.multinomial(total_size, fracs)
        
        start_idx = 0
        for i, count in enumerate(num_samples_per_client):
            if count > 0:
                client_id_map[i].append(c[start_idx : start_idx + count])
                start_idx += count

    # 3. 整理并再次打乱客户端内部数据
    for i in range(n_clients):
        if len(client_id_map[i]) > 0:
            client_id_map[i] = np.concatenate(client_id_map[i])
            # 如果 client_id_map[i] 里存的是多个数组, 拼接成一个一维数组
            np.random.shuffle(client_id_map[i])
        else:
            client_id_map[i] = np.array([], dtype=int)
        
    if return_dist:
        return client_id_map, label_distribution
    else:
        return client_id_map


# 分层截断
def stratified_prune(dict_users, dataset, m_tr, seed=None):
    """
    保持类别比例的静态截断，支持 ImageFolder 和普通 Dataset
    """
    if seed is not None:
        np.random.seed(seed)
        
    dict_users_pruned = {}
    
    # 健壮的 Label 获取逻辑
    if hasattr(dataset, 'targets'):
        all_targets = np.array(dataset.targets)
    elif hasattr(dataset, 'imgs'): 
        # Tiny-ImageNet (ImageFolder)
        all_targets = np.array([t for _, t in dataset.imgs])
    else:
        raise ValueError("Dataset format not supported for stratified pruning (no .targets or .imgs)")

    for idx, user_idxs in dict_users.items():
        if len(user_idxs) <= m_tr:
            dict_users_pruned[idx] = user_idxs
            continue
            
        user_idxs = np.array(user_idxs)
        user_labels = all_targets[user_idxs] # 获取该用户当前拥有的所有标签
        unique_labels, counts = np.unique(user_labels, return_counts=True)
        #对 user_labels 里的元素去重，并统计每个唯一值出现的次数
        
        original_size = len(user_idxs)
        selected_indices = []
        
        for label, count in zip(unique_labels, counts):
            ratio = count / original_size
            num_to_keep = int(round(ratio * m_tr))
            
            # 只要原来有，至少保留1个
            if num_to_keep == 0 and m_tr >= len(unique_labels):
                 num_to_keep = 1
            
            label_locations = user_idxs[user_labels == label]
            keep_idxs = np.random.choice(label_locations, min(num_to_keep, len(label_locations)), replace=False)
            selected_indices.extend(keep_idxs) #extend把列表里的元素逐个加入到目标列表,用append会嵌套
            
        selected_indices = np.array(selected_indices)
        
        # 修正总数误差, 这个修正只是为了解决“四舍五入”带来的整数误差, 其涉及的样本数量通常只有 1-2 个
        if len(selected_indices) > m_tr:
            selected_indices = np.random.choice(selected_indices, m_tr, replace=False)
        elif len(selected_indices) < m_tr:
            remaining = np.setdiff1d(user_idxs, selected_indices)
            num_needed = m_tr - len(selected_indices)
            if len(remaining) >= num_needed:
                fillers = np.random.choice(remaining, num_needed, replace=False)
                selected_indices = np.concatenate([selected_indices, fillers])
        
        dict_users_pruned[idx] = selected_indices.astype(int) # 确保最终保存的索引是整数类型
        
    return dict_users_pruned


def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# === [新增 1] 全局分层裁剪 ===
def global_stratified_prune(dataset, usage_ratio, seed=None):
    """
    按类别比例保留数据，使用局部随机种子确保可复现。
    """
    rng = np.random.default_rng(seed)
    
    if hasattr(dataset, 'targets'):
        all_labels = np.array(dataset.targets)
    elif hasattr(dataset, 'imgs'):  # ImageFolder
        all_labels = np.array([t for _, t in dataset.imgs])
    else:
        raise ValueError("Dataset must have .targets or .imgs")

    classes = np.unique(all_labels)
    kept_indices = []

    for c in classes:
        c_indices = np.where(all_labels == c)[0]
        count = len(c_indices)
        # 策略：每类至少保留 1 个样本，防止下游维度报错
        num_keep = int(np.ceil(count * usage_ratio))
        if num_keep < 1: num_keep = 1
        if num_keep > count: num_keep = count

        # 局部随机选择
        selected = rng.choice(c_indices, num_keep, replace=False)
        kept_indices.extend(selected)

    kept_indices = np.array(kept_indices)
    rng.shuffle(kept_indices)  # 打乱整体顺序
    
    return kept_indices

# === [新增 2] 最小样本强制检查 (只救0样本，不扶1样本) ===
def enforce_min_samples(dict_users, min_samples=1):
    """
    后处理：确保每个客户端至少有 min_samples 个样本。
    防止 Dirichlet 极端分布导致客户端分配到 0 个样本从而引发 Crash。
    """
    # 找出样本数不足的客户端
    poor_clients = [uid for uid, idxs in dict_users.items() if len(idxs) < min_samples]
    
    if len(poor_clients) == 0:
        return dict_users
    
    print(f"⚠️ Warning: {len(poor_clients)} clients have insufficient data (count < {min_samples}). Rebalancing...")

    # 预计算所有客户端长度，减少 len() 调用开销
    client_lens = {uid: len(idxs) for uid, idxs in dict_users.items()}

    for poor_uid in poor_clients:
        while len(dict_users[poor_uid]) < min_samples:
            # 找当前最富有的客户端 (动态查找)
            rich_uid = max(client_lens, key=client_lens.get)
            
            # 如果最富的也没有多余粮食（极其罕见），则抛出错误
            if client_lens[rich_uid] <= min_samples:
                 raise ValueError("Fatal Error: Global data scarcity is too extreme to satisfy min_samples constraint!")
            
            # 劫富：拿走最后一个样本
            transferred_idx = dict_users[rich_uid][-1]
            dict_users[rich_uid] = dict_users[rich_uid][:-1]
            client_lens[rich_uid] -= 1
            
            # 济贫 - [修复] 确保索引类型一致性
            dict_users[poor_uid] = np.concatenate([dict_users[poor_uid], [transferred_idx]]).astype(int)
            client_lens[poor_uid] += 1
            
    return dict_users