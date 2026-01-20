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

def set_module_trainable(module, trainable):
    """设置模块是否可训练"""
    for param in module.parameters():
        param.requires_grad = trainable

class CrossFreezeClient:
    """CrossFreeze 客户端 - 最优版 (原型学习 + GPU加速 + 归一化 + 偶数轮保护)"""
    
    def __init__(self, client_id, model, train_loader, test_loader, 
                 dataset_name, lr=0.01, local_epochs=5, device='cpu', 
                 momentum=0.9, weight_decay=1e-4, gamma_sm=1.0, ld=0.1): # 建议默认 ld=0.1
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        
        # 核心超参数
        self.gamma_sm = gamma_sm  # Sm 蒸馏权重
        self.ld = ld              # 原型损失权重 (建议 0.1 或 0.01)
        
        # 存储全局原型
        self.global_prototypes = None 
        self.global_protos_tensor = None  # 张量化GPU版本
        self.known_labels = None          # 已知类别标签索引

        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器定义
        self.optimizer_m1_m2 = torch.optim.SGD(
            list(self.model.m1.parameters()) + list(self.model.m2.parameters()),
            lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.optimizer_sm = torch.optim.SGD(
            self.model.sm.parameters(),
            lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.optimizer_m1 = torch.optim.SGD(
            self.model.m1.parameters(),
            lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    def adjust_learning_rate(self, new_lr):
        """
        手动更新所有优化器的学习率
        """
        # 更新 M1+M2 联合优化器
        for param_group in self.optimizer_m1_m2.param_groups:
            param_group['lr'] = new_lr
            
        # 更新 Sm 优化器
        for param_group in self.optimizer_sm.param_groups:
            param_group['lr'] = new_lr
            
        # 更新 M1 独立优化器
        for param_group in self.optimizer_m1.param_groups:
            param_group['lr'] = new_lr

    def set_sm_parameters(self, sm_state_dict):
        """接收并加载全局 Sm 参数"""
        self.model.sm.load_state_dict(sm_state_dict)

    def set_global_prototypes(self, global_prototypes):
        """接收并处理全局原型"""
        if not global_prototypes:
            self.global_protos_tensor = None
            self.known_labels = None
            return

        # 1. 提取标签和原型
        sorted_labels = sorted(global_prototypes.keys())
        # 【修正点】确保 label 是 Tensor 且在正确的 device 上
        self.known_labels = torch.tensor(sorted_labels, device=self.device, dtype=torch.long)
        
        protos_list = [global_prototypes[l] for l in sorted_labels]
        
        # 2. 堆叠并归一化
        if protos_list:
            self.global_protos_tensor = torch.stack(protos_list).to(self.device)
            self.global_protos_tensor = F.normalize(self.global_protos_tensor, p=2, dim=1)
        else:
            self.global_protos_tensor = None

    def calculate_proto_loss(self, features, targets):
        """
        【最终修正版】基于原型的交叉熵损失 (Prototype CrossEntropy / InfoNCE)
        作用：强力拉近同类原型，推开异类原型。
        """
        # 保险检查：如果不包含 global_protos_tensor 或者 known_labels 为空
        if self.global_protos_tensor is None or self.known_labels is None:
            return torch.tensor(0.0, device=self.device)
        
        # 1. 过滤有效样本 (只计算出现在 global_protos 中的类别)
        # self.known_labels 在 set_global_prototypes 中维护
        if self.known_labels is None:
            return torch.tensor(0.0, device=self.device)
            
        mask = torch.isin(targets, self.known_labels)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        valid_features = features[mask]
        valid_targets = targets[mask]

        # 2. 特征归一化 (B, Feature_Dim)
        valid_features_norm = F.normalize(valid_features, p=2, dim=1)
        
        # 3. 获取全局原型矩阵 (Num_Classes, Feature_Dim)
        # 假设 set_global_prototypes 已经做过 F.normalize
        all_protos_norm = self.global_protos_tensor 

        # 4. 计算相似度 logits (B, Num_Classes)
        # 温度系数 tau=0.1，越小，对不匹配的惩罚越重
        tau = 0.1 
        logits = torch.matmul(valid_features_norm, all_protos_norm.T) / tau
        
        # 5. 计算交叉熵损失
        # valid_targets 必须是全局标签索引 (0~9)，直接作为 target
        loss = F.cross_entropy(logits, valid_targets)
        
        return loss

    def train(self, round_idx):
        """主训练入口"""
        if (round_idx + 1) % 2 == 1:
            # 奇数轮
            loss_s1 = self.train_s1()
            
            # 融合了原型计算的 train_s2
            loss_s2, local_protos, counts = self.train_s2()
            
            sm_params = self.model.sm.state_dict()
            
            return sm_params, local_protos, counts, (loss_s1, loss_s2)
            
        else:
            # 偶数轮
            loss_even = self.train_even()
            return None, None, None, loss_even

    def train_s1(self):
        """奇数轮 S1: M1 + M2 (开启原型约束)"""
        self.model.train()
        set_module_trainable(self.model.m1, True) # M1与M2更新
        set_module_trainable(self.model.m2, True)
        set_module_trainable(self.model.sm, False) # Sm冻结
        
        epoch_loss = 0
        for epoch in range(self.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                if target.dim() > 1: target = target.squeeze().long()
                
                self.optimizer_m1_m2.zero_grad(set_to_none=True)
                
                features = self.model.m1(data)
                out_m2 = self.model.m2(features)
                out_sm = self.model.sm(features)
                
                loss_cls = self.criterion(out_m2, target) # 分类损失
                loss_distill = self.criterion(out_sm, target) # 蒸馏损失
                
                loss_proto = self.calculate_proto_loss(features, target) # 原型损失
                
                # 总 Loss
                loss = loss_cls + self.gamma_sm * loss_distill + self.ld * loss_proto
                
                loss.backward()
                self.optimizer_m1_m2.step()
                epoch_loss += loss.item()
                
        return epoch_loss / len(self.train_loader)

    def train_s2(self):
        """
        奇数轮 S2: 训练 Sm + 计算本地原型
        1. 将 M1/M2 设为 eval 模式，提供稳定的蒸馏目标。
        2. 在最后一个 epoch 顺便累加特征，省去单独的一次全量前向传播。
        """
        # M1/M2 设为 eval (冻结+稳定)，只有 Sm 设为 train
        self.model.m1.eval() 
        self.model.m2.eval()
        self.model.sm.train()
        
        set_module_trainable(self.model.m1, False) # M1冻结
        set_module_trainable(self.model.m2, False) # M2冻结
        set_module_trainable(self.model.sm, True) # Sm更新
        
        # 初始化原型累加器
        prototypes = {}
        class_counts = {}
        
        epoch_loss = 0
        T = 3.0 #蒸馏的温度系数
        
        for epoch in range(self.local_epochs):
            # 标记是否是最后一个epoch,用于收集原型
            is_last_epoch = (epoch == self.local_epochs - 1)
            
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                if target.dim() > 1: target = target.squeeze().long() # squeeze()移除张量中维度为1的轴
                
                self.optimizer_sm.zero_grad(set_to_none=True) # set_to_none 比 zero 更快
                
                # M1 处于 eval 模式，no_grad 环境下计算特征
                with torch.no_grad():
                    features = self.model.m1(data)
                    out_m2 = self.model.m2(features)
                
                # 在这里收集原型,不需额外前向传播
                if is_last_epoch:
                    # 这里的features在eval()模型下直接用
                    unique_labels = torch.unique(target)

                    features_norm = F.normalize(features, p=2, dim=1)

                    for label in unique_labels:
                        label_item = label.item()
                        idxs = (target == label).nonzero(as_tuple=True)[0] # nonzero()返回张量中所有非零元素的索引
                        #  as_tuple=True表示返回的结果是一个元组，每个维度的索引作为元组中的一个元素，比如(tensor([1, 3]),)
                        
                        # 累加,在GPU上进行
                        sum_features = features_norm[idxs].sum(dim=0)
                        count = idxs.size(0)
                        
                        if label_item not in prototypes:
                            prototypes[label_item] = sum_features
                            class_counts[label_item] = count # 初始化计数
                        else:
                            prototypes[label_item] += sum_features
                            class_counts[label_item] += count # 累加计数
                
                # Sm 的训练继续
                # features虽然是no_grad出来的，但Sm接收它作为输入，Sm的参数可以更新
                out_sm = self.model.sm(features)

                # 1. 硬损失 (Hard Loss): Sm 也要做对分类任务
                loss_cls = self.criterion(out_sm, target)
                
                # 2. 软损失 (Distillation Loss) 标准 KL 散度, 老师Softmax + Temperature
                teacher_probs = F.softmax(out_m2 / T, dim=1)
                student_log_probs = F.log_softmax(out_sm / T, dim=1) # 学生LogSoftmax + Temperature
                
                # 计算 KL 散度并乘以 T^2 以保持梯度量级,KLDiv 要求输入是 log_prob
                # reduction='batchmean' 是PyTorch推荐的数学上正确的平均方式
                loss_distill = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
                
                # 总 Loss
                loss = loss_cls + self.gamma_sm * loss_distill
                
                loss.backward()
                self.optimizer_sm.step()
                epoch_loss += loss.item()
        
        # 循环结束，处理原型均值并转回CPU
        final_prototypes = {}
        for label, sum_feat in prototypes.items():
            final_prototypes[label] = (sum_feat / class_counts[label]).cpu()
            
        return epoch_loss / len(self.train_loader), final_prototypes, class_counts

    def train_even(self):
        """偶数轮: M1 微调 (必须开启原型约束！已修正)"""
        self.model.train()
        set_module_trainable(self.model.m1, True) 
        set_module_trainable(self.model.m2, False) 
        set_module_trainable(self.model.sm, False) 
        
        epoch_loss = 0
        # 注意：外部参数 local_epochs 若设为 2，这里循环次数变少
        for epoch in range(self.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                if target.dim() > 1: target = target.squeeze().long()
                
                self.optimizer_m1.zero_grad(set_to_none=True)
                
                features = self.model.m1(data)
                out_m2 = self.model.m2(features)
                out_sm = self.model.sm(features)
                
                loss_m2 = self.criterion(out_m2, target)
                loss_sm = self.criterion(out_sm, target)
                
                # 【关键修改】加回原型损失
                loss_proto = self.calculate_proto_loss(features, target)
                
                # 权重组合：
                # gamma_sm 限制输出逻辑，ld 限制特征空间
                loss = loss_m2 + self.gamma_sm * loss_sm + self.ld * loss_proto
                
                loss.backward()
                self.optimizer_m1.step()
                epoch_loss += loss.item()
                
        return epoch_loss / len(self.train_loader)

    def get_num_samples(self):
        """获取训练集样本数"""
        return len(self.train_loader.dataset)

    def evaluate(self):
        """M1+M2 评估"""
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                if target.dim() > 1: target = target.squeeze().long()
                
                out = self.model(data) # M1+M2
                loss += self.criterion(out, target).item()
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss /= len(self.test_loader)
        acc = 100. * correct / len(self.test_loader.dataset)
        return acc, loss

    def evaluate_sm(self):
        """M1+Sm 评估"""
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                if target.dim() > 1: target = target.squeeze().long()
                
                features = self.model.m1(data)
                out = self.model.sm(features)
                loss += self.criterion(out, target).item()
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        loss /= len(self.test_loader)
        acc = 100. * correct / len(self.test_loader.dataset)
        return acc, loss

    def evaluate_train(self):
        """M1+M2 训练集准确率"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # 为节省时间，只评估部分训练集（例如前 20 个 batch）
            # 如果需要全量评估，去掉 enumerate 限制
            for i, (data, target) in enumerate(self.train_loader):
                if i > 20: break 
                data, target = data.to(self.device), target.to(self.device)
                if target.dim() > 1: target = target.squeeze().long()
                
                out = self.model(data)
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return 100. * correct / total if total > 0 else 0

    def evaluate_train_sm(self):
        """M1+Sm 训练集准确率"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # 为节省时间，只评估部分训练集（例如前 20 个 batch）
            # 如果需要全量评估，去掉 enumerate 限制
            for i, (data, target) in enumerate(self.train_loader):
                if i > 20: break 
                data, target = data.to(self.device), target.to(self.device)
                if target.dim() > 1: target = target.squeeze().long()
                
                features = self.model.m1(data)
                out = self.model.sm(features)
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return 100. * correct / total if total > 0 else 0

    def evaluate_per_class_accuracy(self, num_classes, use_sm=False):
        """
        评估每个类别的准确率
        
        Args:
            num_classes: 总类别数
            use_sm: 是否使用 Sm 头 (True: M1+Sm, False: M1+M2)
            
        Returns:
            per_class_acc: 每个类别的准确率列表
        """
        self.model.eval()
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                if target.dim() > 1: 
                    target = target.squeeze().long()
                
                # 根据 use_sm 选择不同的头
                if use_sm:
                    # 使用 M1+Sm
                    features = self.model.m1(data)
                    output = self.model.sm(features)
                else:
                    # 使用 M1+M2 (默认的个性化模型)
                    output = self.model(data)
                
                pred = output.argmax(dim=1)
                
                # 统计每个类别的正确数和总数
                for i in range(target.size(0)):
                    label = target[i].item()
                    class_total[label] += 1
                    if pred[i] == label:
                        class_correct[label] += 1
        
        # 计算每个类别的准确率
        per_class_acc = []
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
            else:
                acc = 0.0
            per_class_acc.append(acc)
        
        return per_class_acc


class ClientManager:
    """客户端管理器 - 兼容 main.py 接口"""
    def __init__(self, dataset_name, num_clients, model, client_loaders, test_loader,
                 lr=0.01, local_epochs=5, device='cpu',
                 momentum=0.9, weight_decay=1e-4, gamma_sm=1.0, ld=0.1): # 默认参数对齐
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
                gamma_sm=gamma_sm,
                ld=ld
            )
            self.clients.append(client)

    def get_client(self, client_id):
        return self.clients[client_id]

    def get_all_clients(self):
        return self.clients