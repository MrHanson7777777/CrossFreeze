"""
数据加载和分割 (位于 data/ 目录)
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset as TorchSubset
from torchvision import datasets, transforms
import os
import sys

# 添加 PathMNIST 支持
try:
    import medmnist
    from medmnist import INFO
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False
    print("Warning: medmnist 未安装，PathMNIST 数据集不可用")

# 动态添加根目录到 sys.path
# 假设 data_loader.py 在 /path/to/project/data/
# 根目录是 /path/to/project/
# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(root_dir)
# print(f"Data_loader.py: sys.path 已添加 {root_dir}")

# 尝试导入配置 (如果需要)
# try:
#     from config import DATASET_CONFIG
# except ImportError:
#     print("data_loader.py: 无法导入 config.py。将使用本地定义。")
#     # (如果需要，在此处复制DATASET_CONFIG)
#     pass


class CustomDataset(Dataset):
    """自定义数据集"""
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


def load_dataset(dataset_name, data_dir='./data'):
    """
    加载数据集
    Args:
        data_dir: 应该是 'main.py' 传入的路径, e.g., './data'
                  torchvision 将在此路径下查找 e.g., './data/mnist'
    """
    print(f"Data_loader: 正在从 {data_dir} 加载 {dataset_name}...")
    
    if dataset_name.lower() == 'mnist':
        # 训练集数据增强
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # 测试集不做增强
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform_test)
        
    elif dataset_name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        
    elif dataset_name.lower() == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test)
    
    elif dataset_name.lower() == 'pathmnist':
        if not MEDMNIST_AVAILABLE:
            raise ImportError("medmnist 包未安装。请运行: pip install medmnist")
        
        # PathMNIST 配置
        info = INFO['pathmnist']
        DataClass = getattr(medmnist, info['python_class'])
        
        # 训练集数据增强
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # 测试集不做增强
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # 加载数据集
        train_dataset = DataClass(split='train', transform=transform_train, download=True, root=data_dir)
        test_dataset = DataClass(split='test', transform=transform_test, download=True, root=data_dir)
        
        print(f"PathMNIST 加载完成: 训练集 {len(train_dataset)} 张，测试集 {len(test_dataset)} 张")
        
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return train_dataset, test_dataset


def split_iid(dataset, num_clients):
    """IID数据分割"""
    num_items = len(dataset) // num_clients
    client_data_indices = {}
    all_indices = list(range(len(dataset)))
    np.random.shuffle(all_indices)
    
    for i in range(num_clients):
        client_data_indices[i] = all_indices[i * num_items:(i + 1) * num_items]
    
    return client_data_indices


def split_noniid(dataset, num_clients, alpha=0.5):
    """Non-IID数据分割 (使用Dirichlet分布)"""
    try:
        # torchvision dataset (e.g., MNIST, CIFAR)
        labels = np.array(dataset.targets)
    except AttributeError:
        try:
            # MedMNIST dataset (PathMNIST)
            labels = dataset.labels.squeeze()
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
        except AttributeError:
            # Subset 或其他格式
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # 确保 labels 是 1D 数组
    labels = np.array(labels).flatten()
        
    num_classes = len(np.unique(labels))
    
    # 为每个类别的数据创建索引
    label_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}
    
    client_data_indices = {i: [] for i in range(num_clients)}
    
    # 使用Dirichlet分布分配数据
    for c_idx in range(num_classes):
        class_indices = label_indices[c_idx]
        np.random.shuffle(class_indices)
        
        # Dirichlet分布产生每个客户端获得该类别数据的比例
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # 确保数据分配（处理不平衡的数据集）
        total_class_size = len(class_indices)
        client_sizes = (proportions * total_class_size).astype(int)
        
        # 处理可能的舍入误差
        remainder = total_class_size - client_sizes.sum()
        for i in range(remainder):
            client_sizes[i % num_clients] += 1
            
        # 分配
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + client_sizes[client_id]
            client_data_indices[client_id].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx
    
    # 打乱每个客户端的数据
    for i in range(num_clients):
        np.random.shuffle(client_data_indices[i])
    
    return client_data_indices


def get_client_dataloaders(dataset_name, num_clients, batch_size, iid=True, alpha=0.5, data_dir='./data'):
    """获取所有客户端的数据加载器"""
    train_dataset, test_dataset = load_dataset(dataset_name, data_dir)
    
    # 分割训练数据
    if iid:
        print("使用IID数据分割")
        client_indices = split_iid(train_dataset, num_clients)
    else:
        print(f"使用Non-IID数据分割 (alpha={alpha})")
        client_indices = split_noniid(train_dataset, num_clients, alpha)
    
    # 创建客户端数据加载器
    client_loaders = {}
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        if not indices:
            print(f"警告: 客户端 {client_id} 没有分配到数据。")
            # 创建一个空的 DataLoader
            client_loaders[client_id] = DataLoader(
                TorchSubset(train_dataset, []), 
                batch_size=batch_size, shuffle=False
            )
            continue

        subset = TorchSubset(train_dataset, indices)
        loader = DataLoader(
            subset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True, 
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        client_loaders[client_id] = loader
        print(f"客户端 {client_id}: {len(indices)} 个样本")
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return client_loaders, test_loader