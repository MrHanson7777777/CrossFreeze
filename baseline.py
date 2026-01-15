"""
Standard FedAvg Baseline Implementation (Based on Original FedAvg Code)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import torchvision.transforms as T


class BaselineClient:
    """Basic Client for Standard FedAvg - Based on Original FedAvg Code"""
    
    def __init__(self, client_id, dataset_name, model, train_loader, test_loader, 
                 lr=0.01, local_epochs=5, device='cpu', momentum=0.9, weight_decay=1e-4,
                 lr_decay_step=20, lr_decay_gamma=0.5):
        """
        Args:
            client_id: Client ID
            dataset_name: 数据集名称，用于选择合适的数据增强策略
            model: Model
            train_loader: Training data loader
            test_loader: Test data loader
            lr: Learning rate
            local_epochs: Local training epochs
            device: Device
            momentum: SGD momentum
            weight_decay: Weight decay
            lr_decay_step: Learning rate decay step
            lr_decay_gamma: Learning rate decay gamma
        """
        self.client_id = client_id
        self.dataset_name = dataset_name  # 保存数据集名称
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = device
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer - train entire model
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma
        )
        
    def set_parameters(self, global_params):
        """Set global model parameters"""
        self.model.load_state_dict(global_params)
    
    def get_parameters(self):
        """Get local model parameters"""
        return copy.deepcopy(self.model.state_dict())

    def train(self, round_idx=None):
        """Local training - based on original fedavg code"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # --- 修复：根据数据集选择增强策略 ---
        if 'pathmnist' in self.dataset_name.lower():
            transform_train = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5), # 病理图像上下翻转也合理
                T.RandomRotation(15),        # 随机旋转
                T.RandomCrop(28, padding=4), # 随机裁剪
            ])
        elif 'cifar' in self.dataset_name.lower(): # CIFAR-10 / CIFAR-100
            transform_train = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(32, padding=4), # 保持 32x32
            ])
        else: # MNIST 等
            transform_train = None  # 不需要几何增强
        
        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # 【性能优化】先传输到 GPU，再做几何增强
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # 【性能优化】在 GPU 上进行几何增强，避免阻塞 GPU
                if data.size(0) > 1 and transform_train is not None: 
                    with torch.no_grad():  # 增强不需要梯度
                        data = transform_train(data)
                
                # 【修正】处理 MedMNIST 的标签维度 [B, 1] -> [B] 并转为 long 类型
                if target.dim() > 1:
                    target = target.squeeze().long()
                else:
                    target = target.long()

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        # Update learning rate after local training
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Return both parameters and loss for fedavg compatibility
        return self.get_parameters(), avg_loss

    def evaluate(self):
        """Evaluate model - based on original fedavg code"""
        self.model.eval()
        correct = 0
        total = 0
        loss = 0.0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                # 处理标签维度（与训练方法保持一致）
                if target.dim() > 1:
                    target = target.squeeze().long()
                else:
                    target = target.long()

                output = self.model(data)
                loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total if total > 0 else 0
        avg_loss = loss / len(self.test_loader) if len(self.test_loader) > 0 else 0
        
        return accuracy, avg_loss

    def evaluate_train(self):
        """Evaluate training set performance"""
        self.model.eval()
        correct = 0
        total = 0
        loss = 0.0
        
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # 处理标签维度（与训练方法保持一致）
                if target.dim() > 1:
                    target = target.squeeze().long()
                else:
                    target = target.long()
                
                output = self.model(data)
                loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total if total > 0 else 0
        avg_loss = loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        
        return accuracy  # Return only accuracy to match main.py expectations

    def get_num_samples(self):
        """Return number of training samples for weighted aggregation"""
        try:
            return len(self.train_loader.dataset)
        except Exception:
            return 0


class BaselineServer:
    """Basic Server - based on original fedavg code"""
    
    def __init__(self, model, test_loader=None, device='cpu'):
        self.global_model = copy.deepcopy(model).to(device)
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def get_global_model(self):
        return self.global_model
    
    def get_global_parameters(self):
        """Get global model parameters as dict"""
        return self.global_model.state_dict()
    
    def aggregate(self, client_models, client_weights):
        """Aggregate client models - based on original fedavg code"""
        total_weight = sum(client_weights)
        
        # Initialize aggregated parameters to zero
        aggregated_params = {}
        for name, param in self.global_model.named_parameters():
            aggregated_params[name] = torch.zeros_like(param)
        
        # Weighted aggregation
        for client_model, weight in zip(client_models, client_weights):
            client_weight = weight / total_weight
            for name, param in client_model.items():
                aggregated_params[name] += client_weight * param
        
        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.copy_(aggregated_params[name])
    
    def evaluate(self):
        """Evaluate global model"""
        if self.test_loader is None:
            return 0.0, 0.0
            
        self.global_model.eval()
        correct = 0
        total = 0
        loss = 0.0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                output = self.global_model(data)
                loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total if total > 0 else 0
        avg_loss = loss / len(self.test_loader) if len(self.test_loader) > 0 else 0
        
        return accuracy, avg_loss


class BaselineClientManager:
    """Baseline Client Manager - based on original fedavg code"""

    def __init__(self, dataset_name, num_clients, model, client_loaders, test_loader,
                 lr=0.01, local_epochs=5, device='cpu',
                 momentum=0.9, weight_decay=1e-4,
                 lr_decay_step=20, lr_decay_gamma=0.5):
        self.num_clients = num_clients
        self.clients = []
        for i in range(num_clients):
            client = BaselineClient(
                client_id=i,
                dataset_name=dataset_name,  # 【新增】传递数据集名称
                model=model,
                train_loader=client_loaders[i],
                test_loader=test_loader,
                lr=lr,
                local_epochs=local_epochs,
                device=device,
                momentum=momentum,
                weight_decay=weight_decay,
                lr_decay_step=lr_decay_step,
                lr_decay_gamma=lr_decay_gamma
            )
            self.clients.append(client)

    def get_client(self, client_id):
        return self.clients[client_id]

    def get_all_clients(self):
        return self.clients


def select_clients(num_clients, fraction):
    """Select participating clients"""
    num_selected = max(1, int(num_clients * fraction))
    return np.random.choice(num_clients, num_selected, replace=False)


def evaluate_baseline_clients(client_manager, test_loader, device):
    """Evaluate all clients' test performance (simplified version)"""
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    for client in client_manager.get_all_clients():
        accuracy, loss = client.evaluate()
        num_samples = client.get_num_samples()
        
        total_correct += (accuracy / 100.0) * num_samples
        total_samples += num_samples
        total_loss += loss * num_samples
    
    avg_accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    return avg_accuracy, avg_loss


def evaluate_baseline_clients_detailed(client_manager, test_loader, device):
    """Detailed evaluation of all clients' training and test performance"""
    test_accuracies = []
    train_accuracies = []
    client_weights = []
    
    for client in client_manager.get_all_clients():
        test_acc, _ = client.evaluate()
        test_accuracies.append(test_acc)
        
        train_acc = client.evaluate_train()
        train_accuracies.append(train_acc)
        
        num_samples = client.get_num_samples()
        client_weights.append(num_samples)
    
    total_weight = sum(client_weights)
    if total_weight > 0:
        weighted_avg_test = sum(acc * w for acc, w in zip(test_accuracies, client_weights)) / total_weight
        weighted_avg_train = sum(acc * w for acc, w in zip(train_accuracies, client_weights)) / total_weight
    else:
        weighted_avg_test = weighted_avg_train = 0.0
    
    return weighted_avg_test, weighted_avg_train, test_accuracies, train_accuracies, client_weights