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


def mixup_data(x, y, alpha=1.0, device='cpu'):
    """返回 mixup 后的输入、目标对和 lambda"""
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
    """计算 mixup 损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0, min_ratio=0.1, max_ratio=0.8, prob=1.0, device='cpu'):
    """CutMix 数据增强"""
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


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """计算 CutMix 损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def set_module_trainable(module, trainable):
    """设置模块是否可训练"""
    for param in module.parameters():
        param.requires_grad = trainable

class CrossFreezeClient:
    """CrossFreeze 客户端"""
    
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
        
        # CutMix 数据增强参数
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        self.cutmix_min_ratio = cutmix_min_ratio
        self.cutmix_max_ratio = cutmix_max_ratio
        self.use_cutmix = use_cutmix
        self.mixup_cutmix_ratio = mixup_cutmix_ratio
        
        # 一致性损失参数
        self.consistency_beta = consistency_beta
        self.consistency_criterion = nn.MSELoss()
        
        # 【新增】学习率和损失权重参数
        self.min_lr = min_lr
        self.gamma_sm = gamma_sm  # 保存 gamma_sm 参数
        
        # 损失函数 (标签平滑)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.criterion_none = nn.CrossEntropyLoss(reduction='none')
        
        # 根据数据集选择增强策略
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
                T.RandomGrayscale(p=0.2)
            ])
        else:
            self.transform_train = T.Compose([])

        # 优化器定义
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
        
        # CosineAnnealingLR 调度器 - 使用传入的 min_lr 参数
        # [修改] 将 T_max 设为 total_epochs 的 1.5 倍
        # 这样在训练结束时，学习率不会降到 min_lr，而是保持在中高位，这对 Non-IID 至关重要
        sched_epochs = int(total_epochs * 1.5)
        
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
        """应用 Mixup 或 CutMix"""
        if not self.use_cutmix:
            return mixup_data(data, target, alpha=self.cutmix_alpha, device=self.device)
        
        if np.random.random() < self.mixup_cutmix_ratio:
            return cutmix_data(
                data, target, 
                alpha=self.cutmix_alpha,
                min_ratio=self.cutmix_min_ratio,
                max_ratio=self.cutmix_max_ratio,
                prob=self.cutmix_prob,
                device=self.device
            )
        else:
            return mixup_data(data, target, alpha=self.cutmix_alpha, device=self.device)

    def train(self, round_idx, total_rounds):
        if (round_idx + 1) % 2 == 1:
            # 奇数轮: S1 (M1+M2) -> S2 (Sm) -> Upload
            avg_loss_s1 = self.train_s1(round_idx, total_rounds)
            avg_loss_s2, hard_samples_count = self.train_s2(round_idx, total_rounds)
            sm_params = self.get_sm_parameters()
            
            self.scheduler_m1_m2.step()
            self.scheduler_sm.step()
            return sm_params, avg_loss_s1, avg_loss_s2, hard_samples_count
        else:
            # 偶数轮: Even (M1)
            avg_loss_even = self.train_even(round_idx, total_rounds)
            self.scheduler_m1.step()
            return None, avg_loss_even, 0.0, 0

    def train_s1(self, round_idx=None, total_rounds=None):
        """S1: 性能优化版 - 只计算一次 M1"""
        self.model.train()
        set_module_trainable(self.model.m1, True)
        set_module_trainable(self.model.m2, True)
        set_module_trainable(self.model.sm, False)
        
        total_loss = 0.0
        num_batches = 0
        
        # 动态调整 gamma_sm 权重
        # 【新增】Tiny-ImageNet 专用逻辑
        if 'tiny_imagenet' in self.dataset_name.lower():
            # IID 场景下，Sm 在初期是混乱的，绝对不能信！
            # 只有当训练到后期，特征空间趋于稳定，才能给一点点权重
            if round_idx is not None and round_idx < 20:
                gamma_sm = 0.0  # 前20轮完全不看全局，只做本地训练
            else:
                gamma_sm = 0.1  # 后期也只给极小的权重
        else:
            # 其他数据集使用配置参数中的 gamma_sm
            gamma_sm = self.gamma_sm
        
        for epoch in range(self.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # 几何增强
                if data.size(0) > 1:
                    with torch.no_grad():
                        data = self.transform_train(data)

                if target.dim() > 1: target = target.squeeze().long()
                else: target = target.long()

                # 1. 计算一致性损失 (新增逻辑)
                # 需要对原图做两次前向传播：一次强增强，一次弱增强(原图)
                if self.consistency_beta > 0:
                    # 强增强图 (重新生成，与分类任务的 Mixup/CutMix 分离)
                    if data.size(0) > 1:
                        with torch.no_grad():
                            inputs_aug_cons = self.transform_train(data) # 强增强
                    else:
                        inputs_aug_cons = data
                    
                    feat_strong = self.model.m1(inputs_aug_cons)
                    with torch.no_grad(): # 弱增强/原图不传梯度，作为 Anchor
                        feat_weak = self.model.m1(data)
                    
                    loss_cons = self.consistency_criterion(feat_strong, feat_weak)
                else:
                    loss_cons = 0.0

                # 2. 计算分类损失 (保持不变，使用 Mixup/CutMix 后的 inputs)
                inputs, targets_a, targets_b, lam = self.apply_data_augmentation(data, target)

                self.optimizer_m1_m2.zero_grad()
                
                # 【优化点】: 显式提取特征，避免重复计算 M1
                features = self.model.m1(inputs)
                
                # 分别传入两个头
                output_m2 = self.model.m2(features)
                output_sm = self.model.sm(features)
                
                loss_m2 = mixup_criterion(self.criterion, output_m2, targets_a, targets_b, lam)
                loss_sm = mixup_criterion(self.criterion, output_sm, targets_a, targets_b, lam)

                # 3. 统一目标函数 (Unified Objective)
                # Total Loss = L_task(M2) + gamma * L_anchor(Sm) + beta * L_cons
                loss = loss_m2 + gamma_sm * loss_sm + self.consistency_beta * loss_cons
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.m1.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.model.m2.parameters(), max_norm=1.0)
                self.optimizer_m1_m2.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches if num_batches > 0 else 0

    def train_s2(self, round_idx, total_rounds):
        """S2: 全量训练 Sm"""
        # (保持不变，省略 identify_hard_samples 的无用调用，直接全量训练更稳)
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
                if data.size(0) == 1: continue 
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
        """Even: 性能优化版 - 只计算一次 M1 用于分类"""
        self.model.train()
        set_module_trainable(self.model.m1, True)
        set_module_trainable(self.model.m2, False)
        set_module_trainable(self.model.sm, False)
        
        total_loss = 0.0
        num_batches = 0
        
        # 【修改】确保权重与 train_s1 一致
        # 确保这里使用的权重与 S1 完全一致
        # S1 中 loss_m2 的系数是 1.0
        gamma_m2 = 1.0  # 确保与 S1 一致
        
        # 【新增】动态调整权重 - 添加 Tiny-ImageNet 专用逻辑
        if 'tiny_imagenet' in self.dataset_name.lower():
            # 偶数轮本意是用 Global Sm 纠正 M1
            # 但在 Tiny-ImageNet IID 上，Global Sm 质量很差
            # 我们主要依靠 M2 (本地头) 来训练 M1，Sm 仅作为极弱的辅助
            gamma_sm = 0.1
            # gamma_m2 = 1.0  # 已在上面设置，保持一致
        elif 'cifar' in self.dataset_name.lower():
            # CIFAR策略: 强对齐全局(Sm)，弱化本地偏见(M2)
            gamma_sm = 1.0
            # 注意：这里不再修改 gamma_m2，保持为 1.0 与 S1 一致
        else:
            # PathMNIST 等差异大的数据集保留原有逻辑
            gamma_sm = 1.0
            # gamma_m2 = 0.5  # 改为 1.0 保持一致 
        
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

                # 1. 一致性 Loss (必须计算两次 M1，这是逻辑需要的，无法省略)
                feat_strong = self.model.m1(inputs_aug)
                with torch.no_grad():
                    feat_weak = self.model.m1(data)
                loss_cons = self.consistency_criterion(feat_strong, feat_weak)

                # 2. 分类 Loss
                inputs_mix, targets_a, targets_b, lam = self.apply_data_augmentation(inputs_aug, target)
                
                # 【优化点】: 对 Mixup 数据只计算一次 M1
                feat_mix = self.model.m1(inputs_mix)
                
                # 分别传入两个头
                output_sm = self.model.sm(feat_mix)
                output_m2 = self.model.m2(feat_mix)
                
                loss_sm = mixup_criterion(self.criterion, output_sm, targets_a, targets_b, lam)
                loss_m2 = mixup_criterion(self.criterion, output_m2, targets_a, targets_b, lam)

                loss = gamma_sm * loss_sm + gamma_m2 * loss_m2 + self.consistency_beta * loss_cons
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.m1.parameters(), max_norm=1.0)
                self.optimizer_m1.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches if num_batches > 0 else 0
    
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

class ClientManager:
    """客户端管理器"""
    def __init__(self, dataset_name, num_clients, model, client_loaders, test_loader,
                 lr=0.01, local_epochs=5, device='cpu',
                 momentum=0.9, weight_decay=1e-4,
                 lr_decay_step=20, lr_decay_gamma=0.5, mu=0.01,
                 total_epochs=50, cutmix_alpha=1.0, cutmix_prob=0.8,
                 cutmix_min_ratio=0.1, cutmix_max_ratio=0.8, use_cutmix=True,
                 mixup_cutmix_ratio=0.5, consistency_beta=5.0, min_lr=1e-5, gamma_sm=1.0):
        self.num_clients = num_clients
        self.clients = []
        for i in range(num_clients):
            client = CrossFreezeClient(
                client_id=i,
                dataset_name=dataset_name,
                model=model,
                train_loader=client_loaders[i],
                test_loader=test_loader,
                lr=lr,
                local_epochs=local_epochs,
                device=device,
                momentum=momentum,
                weight_decay=weight_decay,
                lr_decay_step=lr_decay_step,
                lr_decay_gamma=lr_decay_gamma,
                mu=mu,
                total_epochs=total_epochs,
                cutmix_alpha=cutmix_alpha,
                cutmix_prob=cutmix_prob,
                cutmix_min_ratio=cutmix_min_ratio,
                cutmix_max_ratio=cutmix_max_ratio,
                use_cutmix=use_cutmix,
                mixup_cutmix_ratio=mixup_cutmix_ratio,
                consistency_beta=consistency_beta,
                min_lr=min_lr,  # 【新增】传递最小学习率
                gamma_sm=gamma_sm  # 【新增】传递S1阶段损失权重
            )
            self.clients.append(client)

    def get_client(self, client_id):
        return self.clients[client_id]

    def get_all_clients(self):
        return self.clients