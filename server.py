"""
服务器实现 (CrossFreeze)
"""
import torch
import copy
import numpy as np
from model_utils import (
    model_to_params_dict, params_dict_to_model,
    add_params_dict, subtract_params_dict, zero_params_dict, scale_params_dict
)


class CrossFreezeServer:
    """CrossFreeze 服务器 (仅聚合Sm)"""
    
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
        self.test_loader = test_loader # 传递给客户端
        self.device = device
        
    def get_global_sm_state_dict(self):
        """获取全局 Sm 模块的状态字典"""
        return self.global_model.sm.state_dict()
    
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