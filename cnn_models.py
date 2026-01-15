"""
CNN模型定义 (已修正：确保 Baseline 和 CrossFreeze 使用完全一致的 PathMNIST 架构)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# ==========================================
# 1. PathMNIST 专用组件 (统一架构核心)
# ==========================================

class PathMNISTFeatures(nn.Module):
    """
    M1: PathMNIST 特征提取器
    结构: 3层 Conv + GroupNorm + MaxPool + GAP
    输出: [B, 256]
    """
    def __init__(self, input_channels=3):
        super(PathMNISTFeatures, self).__init__()
        # Conv 1
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 64)
        
        # Conv 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(16, 128)
        
        # Conv 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(32, 256)
        
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Layer 1
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        # Layer 2
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        # Layer 3
        x = self.pool(F.relu(self.gn3(self.conv3(x))))
        
        # Global Average Pooling (GAP)
        # [B, 256, H, W] -> [B, 256, 1, 1]
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1) # Flatten -> [B, 256]
        
        return x

class PathMNISTHead(nn.Module):
    """
    Sm / M2: PathMNIST 分类头
    结构: Dropout + Linear
    """
    def __init__(self, input_dim=256, num_classes=9):
        super(PathMNISTHead, self).__init__()
        # 既然使用了 GAP，Dropout 可以设小一点或保留 0.2
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

class PathMNISTCNN(nn.Module):
    """
    Baseline 专用: 完整的 PathMNIST 模型
    为了公平对比，它必须在数学上等同于 PathMNISTFeatures + PathMNISTHead
    """
    def __init__(self, num_classes=9, input_channels=3):
        super(PathMNISTCNN, self).__init__()
        self.features = PathMNISTFeatures(input_channels)
        self.classifier = PathMNISTHead(input_dim=256, num_classes=num_classes)
        
    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out


# ==========================================
# 2. 其他数据集组件 (保持不变)
# ==========================================

class SimpleCNNFeatures(nn.Module):
    """M1: MNIST"""
    def __init__(self):
        super(SimpleCNNFeatures, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(8, 32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.gn2 = nn.GroupNorm(8, 64)
        self.feature_dim = 64 * 7 * 7 
        self.fc1 = nn.Linear(self.feature_dim, 512)
        self.ln3 = nn.LayerNorm(512)
        self.dropout_features = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = x.view(-1, self.feature_dim)
        x = F.relu(self.ln3(self.fc1(x)))
        x = self.dropout_features(x)
        return x

class SimpleCNNHead(nn.Module):
    """M2/Sm: MNIST"""
    def __init__(self, num_classes=10):
        super(SimpleCNNHead, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.dropout2(self.fc2(x))
        return x

# --- CIFAR (ResNet-9) ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn = nn.GroupNorm(8, out_channels)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        
    def forward(self, x):
        out = self.pool(F.relu(self.gn(self.conv(x))))
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, in_channels)
        
    def forward(self, x):
        res = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = F.relu(self.gn2(self.conv2(out)))
        return out + res

class CIFARCNNFeatures(nn.Module):
    """M1: ResNet-9"""
    def __init__(self, input_channels=3):
        super(CIFARCNNFeatures, self).__init__()
        self.prep = ConvBlock(input_channels, 64)
        self.layer1_conv = ConvBlock(64, 128, pool=True)
        self.layer1_res = ResBlock(128)
        self.layer2_conv = ConvBlock(128, 256, pool=True)
        self.layer3_conv = ConvBlock(256, 512, pool=True)
        self.layer3_res = ResBlock(512)
        self.ln_final = nn.LayerNorm(512)
        self.feature_dim = 512

    def forward(self, x):
        out = self.prep(x)
        out = self.layer1_res(self.layer1_conv(out))
        out = self.layer2_conv(out)
        out = self.layer3_res(self.layer3_conv(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.ln_final(out)
        return out

class CIFARCNNHead(nn.Module):
    """Sm: Linear Head"""
    def __init__(self, num_classes=10):
        super(CIFARCNNHead, self).__init__()
        self.fc = nn.Linear(512, num_classes, bias=False) 

    def forward(self, x):
        return self.fc(x)

class CIFARCNN(nn.Module):
    """Baseline: ResNet-9 整体版"""
    def __init__(self, num_classes=10, input_channels=3):
        super(CIFARCNN, self).__init__()
        self.features = CIFARCNNFeatures(input_channels)
        self.head = CIFARCNNHead(num_classes)
        
    def forward(self, x):
        return self.head(self.features(x))

class SimpleCNN(nn.Module):
    """Baseline: MNIST 整体版"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = SimpleCNNFeatures()
        self.head = SimpleCNNHead(num_classes)
    def forward(self, x):
        return self.head(self.features(x))


# ==========================================
# 3. CrossFreeze 组合模型
# ==========================================

class CrossFreezeModel(nn.Module):
    def __init__(self, features_module, head_module):
        super(CrossFreezeModel, self).__init__()
        self.m1 = features_module
        self.m2 = head_module
        self.sm = copy.deepcopy(head_module)

    def forward(self, x, head='m2'):
        features = self.m1(x)
        if head == 'm2':
            return self.m2(features)
        elif head == 'sm':
            return self.sm(features)
        elif head == 'all':
            return self.m2(features), self.sm(features)
        else:
            raise ValueError(f"未知的 head: {head}")

# ==========================================
# 4. 工厂函数 (这是修复的关键)
# ==========================================
def get_model(dataset, num_classes):
    """获取 CrossFreeze 模型 (分离式)"""
    dataset_lower = dataset.lower()
    
    if dataset_lower == 'mnist':
        features = SimpleCNNFeatures()
        head = SimpleCNNHead(num_classes)
        
    elif dataset_lower == 'pathmnist':
        features = PathMNISTFeatures(input_channels=3)
        head = PathMNISTHead(input_dim=256, num_classes=num_classes)
        
    elif dataset_lower in ['cifar', 'cifar10', 'cifar100']:
        features = CIFARCNNFeatures(input_channels=3)
        head = CIFARCNNHead(num_classes)
        
    else:
        raise ValueError(f"不支持的数据集: {dataset}")
        
    return CrossFreezeModel(features, head)

def get_baseline_model(dataset, num_classes):
    """获取 Baseline 模型 (整体式)"""
    dataset_lower = dataset.lower()
    
    if dataset_lower == 'mnist':
        return SimpleCNN(num_classes)
        
    elif dataset_lower == 'pathmnist':
        # 【修正】Baseline 使用与 CrossFreeze 完全一致的架构
        return PathMNISTCNN(num_classes=num_classes, input_channels=3)
        
    elif dataset_lower in ['cifar', 'cifar10', 'cifar100']:
        return CIFARCNN(num_classes=num_classes, input_channels=3)
        
    else:
        raise ValueError(f"不支持的数据集: {dataset}")

# 工具函数保持不变
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_sm_parameters(model):
    if hasattr(model, 'sm'):
        return sum(p.numel() for p in model.sm.parameters() if p.requires_grad)
    return 0