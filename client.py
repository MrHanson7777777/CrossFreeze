import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import numpy as np
import copy

from model_utils import (
    model_to_params_dict, params_dict_to_model,
    add_params_dict, subtract_params_dict, zero_params_dict
)

# --- Mixup / CutMix 辅助函数保持不变 ---
def mixup_data(x, y, alpha=1.0, device='cpu'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0, min_ratio=0.1, max_ratio=0.8, prob=1.0, device='cpu'):
    if np.random.random() > prob:
        return x, y, y, 1.0
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    lam = np.clip(lam, min_ratio, max_ratio)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    y_a, y_b = y, y[index]
    W = x.size(3)
    H = x.size(2)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return mixed_x, y_a, y_b, lam

def set_module_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable

class CrossFreezeClient:
    """CrossFreeze 客户端 (OOD 优化版)"""
    
    def __init__(self, client_id, model, train_loader, test_loader, 
                 dataset_name, lr=0.01, local_epochs=5, device='cpu', momentum=0.9, weight_decay=1e-4,
                 lr_decay_step=20, lr_decay_gamma=0.5, mu=0.01,
                 total_epochs=50, cutmix_alpha=1.0, cutmix_prob=0.8,
                 cutmix_min_ratio=0.1, cutmix_max_ratio=0.8, use_cutmix=True,
                 mixup_cutmix_ratio=0.5, consistency_beta=5.0, min_lr=1e-5, gamma_sm=1.0):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = device
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.mu = mu
        
        # 增强参数
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        self.cutmix_min_ratio = cutmix_min_ratio
        self.cutmix_max_ratio = cutmix_max_ratio
        self.use_cutmix = use_cutmix
        self.mixup_cutmix_ratio = mixup_cutmix_ratio
        
        # 正则化参数
        self.consistency_beta = consistency_beta
        self.consistency_criterion = nn.MSELoss()
        
        # 学习率和权重
        self.min_lr = min_lr
        self.gamma_sm = gamma_sm  
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 数据增强策略
        if 'pathmnist' in dataset_name.lower():
            self.transform_train = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(15),
                T.RandomCrop(28, padding=4),
            ])
        elif 'cifar' in dataset_name.lower():
            self.transform_train = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(32, padding=4),
                # ColorJitter 对 CIFAR OOD 很有帮助，增加一点扰动
                T.RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
            ])
        else:
            self.transform_train = T.Compose([])

        # 优化器
        self.optimizer_m1_m2 = torch.optim.SGD(
            list(self.model.m1.parameters()) + list(self.model.m2.parameters()),
            lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        self.optimizer_sm = torch.optim.SGD(
            self.model.sm.parameters(),
            lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        self.optimizer_m1 = torch.optim.SGD(
            self.model.m1.parameters(),
            lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        sched_epochs = int(total_epochs * 1.5) # 保持较大学习率更久
        self.scheduler_m1_m2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_m1_m2, T_max=sched_epochs, eta_min=self.min_lr
        )
        self.scheduler_sm = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_sm, T_max=sched_epochs, eta_min=self.min_lr
        )
        self.scheduler_m1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_m1, T_max=sched_epochs, eta_min=self.min_lr
        )
        
    def set_sm_parameters(self, global_sm_state_dict):
        self.model.sm.load_state_dict(global_sm_state_dict)
    
    def get_sm_parameters(self):
        return copy.deepcopy(self.model.sm.state_dict())
    
    def get_num_samples(self):
        try:
            return len(self.train_loader.dataset)
        except Exception:
            return 0

    def apply_data_augmentation(self, data, target):
        if not self.use_cutmix:
            return mixup_data(data, target, alpha=self.cutmix_alpha, device=self.device)
        if np.random.random() < self.mixup_cutmix_ratio:
            return cutmix_data(data, target, alpha=self.cutmix_alpha, min_ratio=self.cutmix_min_ratio, max_ratio=self.cutmix_max_ratio, prob=self.cutmix_prob, device=self.device)
        else:
            return mixup_data(data, target, alpha=self.cutmix_alpha, device=self.device)

    def _get_dynamic_gamma(self, round_idx, total_rounds):
        """
        【关键优化】软热身 (Soft Warm-up) 策略
        逻辑：
        1. 前 20% 轮次 (warmup_steps) 内，给予一个极小的权重 (0.1)。
           这允许本地模型自由学习，但保持微弱的全局连接，防止彻底跑偏。
        2. 之后，线性增加到目标 gamma_sm。
        """
        if total_rounds is None:
            return self.gamma_sm

        warmup_ratio = 0.20 
        warmup_steps = int(warmup_ratio * total_rounds)
        
        # 基础约束力，防止特征空间完全崩塌
        base_gamma = 0.1 

        if round_idx < warmup_steps:
            # 阶段一: 弱约束
            return base_gamma
        else:
            # 阶段二: 线性增加
            # 从 base_gamma 平滑过渡到 target gamma
            progress = (round_idx - warmup_steps) / (total_rounds - warmup_steps)
            return base_gamma + (self.gamma_sm - base_gamma) * progress 

    def train(self, round_idx, total_rounds):
        if (round_idx + 1) % 2 == 1:
            # S1 (M1+M2) -> S2 (Sm)
            avg_loss_s1 = self.train_s1(round_idx, total_rounds)
            avg_loss_s2, hard_samples_count = self.train_s2(round_idx, total_rounds)
            sm_params = self.get_sm_parameters()
            self.scheduler_m1_m2.step()
            self.scheduler_sm.step()
            return sm_params, avg_loss_s1, avg_loss_s2, hard_samples_count
        else:
            # Even (M1)
            avg_loss_even = self.train_even(round_idx, total_rounds)
            self.scheduler_m1.step()
            return None, avg_loss_even, 0.0, 0

    def train_s1(self, round_idx=None, total_rounds=None):
        """S1: 训练 M1 和 M2，使用动态全局约束"""
        self.model.train()
        set_module_trainable(self.model.m1, True)
        set_module_trainable(self.model.m2, True)
        set_module_trainable(self.model.sm, False)
        
        total_loss = 0.0
        num_batches = 0
        
        # 获取动态权重
        current_gamma_sm = self._get_dynamic_gamma(round_idx, total_rounds)

        for epoch in range(self.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                if data.size(0) > 1:
                    with torch.no_grad():
                        data = self.transform_train(data)
                
                if target.dim() > 1: target = target.squeeze().long()
                else: target = target.long()

                # 一致性正则化
                loss_cons = 0.0
                if self.consistency_beta > 0:
                    if data.size(0) > 1:
                        with torch.no_grad():
                            inputs_aug_cons = self.transform_train(data)
                    else:
                        inputs_aug_cons = data
                    
                    feat_strong = self.model.m1(inputs_aug_cons)
                    with torch.no_grad():
                        feat_weak = self.model.m1(data)
                    loss_cons = self.consistency_criterion(feat_strong, feat_weak)

                # 分类任务
                inputs, targets_a, targets_b, lam = self.apply_data_augmentation(data, target)
                self.optimizer_m1_m2.zero_grad()
                
                features = self.model.m1(inputs)
                output_m2 = self.model.m2(features)
                output_sm = self.model.sm(features)
                
                loss_m2 = mixup_criterion(self.criterion, output_m2, targets_a, targets_b, lam)
                loss_sm = mixup_criterion(self.criterion, output_sm, targets_a, targets_b, lam)

                # 动态加权 Loss
                loss = loss_m2 + current_gamma_sm * loss_sm + self.consistency_beta * loss_cons
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.m1.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.model.m2.parameters(), max_norm=1.0)
                self.optimizer_m1_m2.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches if num_batches > 0 else 0

    def train_s2(self, round_idx, total_rounds):
        """S2: 全量训练 Sm"""
        train_loader = self.train_loader 
        if round_idx == 0: 
            self.model.sm.load_state_dict(self.model.m2.state_dict())
        
        self.model.train()
        set_module_trainable(self.model.m1, False)
        set_module_trainable(self.model.m2, False)
        set_module_trainable(self.model.sm, True)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            for data, target in train_loader:
                if data.size(0) <= 1: continue 
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                if data.size(0) > 1:
                    with torch.no_grad():
                        data = self.transform_train(data)
                
                if target.dim() > 1: target = target.squeeze().long()
                else: target = target.long()

                inputs, targets_a, targets_b, lam = self.apply_data_augmentation(data, target)
                
                self.optimizer_sm.zero_grad()
                with torch.no_grad():
                    features = self.model.m1(inputs)
                output_sm = self.model.sm(features)
                loss = mixup_criterion(self.criterion, output_sm, targets_a, targets_b, lam)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.sm.parameters(), max_norm=1.0)
                self.optimizer_sm.step()
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0, len(train_loader.dataset)

    def train_even(self, round_idx=None, total_rounds=None):
        """Even: 训练 M1 (对齐全局 + 本地)"""
        self.model.train()
        set_module_trainable(self.model.m1, True)
        set_module_trainable(self.model.m2, False)
        set_module_trainable(self.model.sm, False)
        
        total_loss = 0.0
        num_batches = 0
        
        # 获取动态权重
        current_gamma_sm = self._get_dynamic_gamma(round_idx, total_rounds)
        
        # 偶数轮 M2 的权重保持为 1.0，确保不遗忘本地知识
        gamma_m2 = 1.0
        
        for epoch in range(self.local_epochs):
            for data, target in self.train_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                if data.size(0) > 1:
                    with torch.no_grad():
                        inputs_aug = self.transform_train(data)
                else:
                    inputs_aug = data

                if target.dim() > 1: target = target.squeeze().long()
                else: target = target.long()
                
                self.optimizer_m1.zero_grad()

                feat_strong = self.model.m1(inputs_aug)
                with torch.no_grad():
                    feat_weak = self.model.m1(data)
                loss_cons = self.consistency_criterion(feat_strong, feat_weak)

                inputs_mix, targets_a, targets_b, lam = self.apply_data_augmentation(inputs_aug, target)
                feat_mix = self.model.m1(inputs_mix)
                output_sm = self.model.sm(feat_mix)
                output_m2 = self.model.m2(feat_mix)
                
                loss_sm = mixup_criterion(self.criterion, output_sm, targets_a, targets_b, lam)
                loss_m2 = mixup_criterion(self.criterion, output_m2, targets_a, targets_b, lam)

                # 在热身初期，current_gamma_sm 较小，此时这一步主要靠 loss_m2 拉动
                # 这相当于增加了额外的本地训练轮次，非常有利于 Non-IID
                loss = current_gamma_sm * loss_sm + gamma_m2 * loss_m2 + self.consistency_beta * loss_cons
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.m1.parameters(), max_norm=1.0)
                self.optimizer_m1.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches if num_batches > 0 else 0
    
    # --- 评估函数保持不变 ---
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                if target.dim() > 1: target = target.squeeze().long()
                else: target = target.long()
                output = self.model(data, head='m2')
                loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        accuracy = 100. * correct / total if total > 0 else 0
        avg_loss = loss / len(self.test_loader) if len(self.test_loader) > 0 else 0
        return accuracy, avg_loss

    def evaluate_train(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                if target.dim() > 1: target = target.squeeze().long()
                else: target = target.long()
                output = self.model(data, head='m2')
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        accuracy = 100. * correct / total if total > 0 else 0
        return accuracy

    def evaluate_sm(self):
        self.model.eval()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                if target.dim() > 1: target = target.squeeze().long()
                else: target = target.long()
                output = self.model(data, head='sm')
                loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        accuracy = 100. * correct / total if total > 0 else 0
        avg_loss = loss / len(self.test_loader) if len(self.test_loader) > 0 else 0
        return accuracy, avg_loss

    def evaluate_train_sm(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                if target.dim() > 1: target = target.squeeze().long()
                else: target = target.long()
                output = self.model(data, head='sm')
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        accuracy = 100. * correct / total if total > 0 else 0
        return accuracy