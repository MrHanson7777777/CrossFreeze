"""
服务器实现 (CrossFreeze + 原型学习)
"""
import torch
import copy
import numpy as np


class CrossFreezeServer:
    """CrossFreeze 服务器 - 支持原型聚合"""
    
    def __init__(self, model, test_loader, device='cpu'):
        """
        Args:
            model: 全局模型 (CrossFreezeModel 实例)
            test_loader: 测试数据加载器 (用于客户端评估)
            device: 设备
        """
        # 服务器持有的 global_model 仅用于存储 Sm_global
        # M1 和 M2 部分不被使用
        self.global_model = copy.deepcopy(model).to(device)
        self.test_loader = test_loader  # 传递给客户端
        self.device = device
        
        # 存储全局原型
        self.global_prototypes = {}
        
    def get_global_sm_state_dict(self):
        """获取全局 Sm 模块的状态字典"""
        return self.global_model.sm.state_dict()
    
    def get_global_prototypes(self):
        """获取全局原型"""
        return self.global_prototypes
    
    def aggregate_prototypes(self, client_protos_list, client_counts_list):
        """
        聚合客户端上传的原型
        Args:
            client_protos_list: List[Dict{label: tensor}]
            client_counts_list: List[Dict{label: int}]
        Returns:
            global_prototypes: Dict{label: tensor}
        """
        temp_protos = {}
        total_counts = {}
        
        # 1. 累加
        for client_idx, protos in enumerate(client_protos_list):
            counts = client_counts_list[client_idx]
            
            for label, proto in protos.items():
                if label not in temp_protos:
                    temp_protos[label] = torch.zeros_like(proto).to(self.device)
                    total_counts[label] = 0
                
                # 确保在同一设备
                proto = proto.to(self.device)
                count = counts[label]
                
                # 加权累加 (Feature * Count)
                temp_protos[label] += proto * count
                total_counts[label] += count
        
        # 2. 平均
        for label in temp_protos:
            if total_counts[label] > 0:
                temp_protos[label] /= total_counts[label]
        
        self.global_prototypes = temp_protos
        return self.global_prototypes
    
    def aggregate(self, client_sm_state_dicts, client_weights):
        """
        仅聚合 Sm 模块的参数
        
        Args:
            client_sm_state_dicts: 客户端 Sm 状态字典列表 [state_dict_1, state_dict_2, ...]
            client_weights: 客户端权重列表 (通常基于数据量)
        """
        if not client_sm_state_dicts:
            return

        # 归一化权重
        total_weight = sum(client_weights)
        weights = [w / total_weight for w in client_weights]
        
        # 聚合
        global_sm_dict = self.global_model.sm.state_dict()
        new_state = {}
        
        # 只遍历 Sm 的键
        for key in global_sm_dict.keys():
            # 如果是浮点类型，按常规方式加权求和
            if global_sm_dict[key].is_floating_point():
                acc = torch.zeros_like(global_sm_dict[key], device=self.device)
                for client_dict, weight in zip(client_sm_state_dicts, weights):
                    tensor = client_dict[key]
                    # 确保tensor在正确设备上，避免重复传输
                    if tensor.device != self.device:
                        tensor = tensor.to(self.device, non_blocking=True)
                    if not tensor.is_floating_point():
                        tensor = tensor.float()
                    acc.add_(tensor, alpha=weight)  # 使用原地操作
                new_state[key] = acc.type_as(global_sm_dict[key])
            else:
                # 对于非浮点（例如 BatchNorm 的 num_batches_tracked）
                acc = None
                for client_dict, weight in zip(client_sm_state_dicts, weights):
                    tensor = client_dict[key]
                    if tensor.device != self.device:
                        tensor = tensor.to(self.device, non_blocking=True)
                    tensor = tensor.float()
                    scaled = tensor * weight
                    if acc is None:
                        acc = scaled
                    else:
                        acc.add_(scaled)  # 使用原地操作
                
                if acc is None:
                    new_state[key] = global_sm_dict[key]
                else:
                    rounded = acc.round()
                    new_state[key] = rounded.to(dtype=global_sm_dict[key].dtype)

        # 加载新的 state_dict 到 Sm 模块
        self.global_model.sm.load_state_dict(new_state)

def select_clients(num_clients, frac):
    """
    选择参与训练的客户端
    
    Args:
        num_clients: 总客户端数
        frac: 参与比例
        
    Returns:
        selected_indices: 选中的客户端索引列表
    """
    num_selected = max(1, int(num_clients * frac))
    selected_indices = np.random.choice(num_clients, num_selected, replace=False)
    return selected_indices.tolist()