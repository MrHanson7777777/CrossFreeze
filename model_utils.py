"""
模型参数字典操作的辅助函数
"""
import torch
import torch.nn as nn
import numpy as np


def model_to_params_dict(model):
    """将模型转换为参数字典"""
    params_dict = {}
    for name, param in model.named_parameters():
        params_dict[name] = param.data.clone()
    return params_dict

def params_dict_to_model(params_dict, model):
    """将参数字典加载到模型（保持设备/数据类型一致）"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in params_dict:
                src = params_dict[name]
                if not torch.is_floating_point(src):
                    src = src.float()
                param.data.copy_(src.to(param.data.device).type_as(param.data))

def add_params_dict(dict1, dict2, weight=1.0):
    """参数字典加法"""
    result = {}
    for name in dict1.keys():
        if name in dict2:
            a = dict1[name]
            b = dict2[name]
            if not torch.is_floating_point(a):
                a = a.float()
            if not torch.is_floating_point(b):
                b = b.float()
            result[name] = a + (weight * b.to(a.device).type_as(a))
        else:
            result[name] = dict1[name].clone()
    return result


def subtract_params_dict(dict1, dict2):
    """参数字典减法"""
    result = {}
    for name in dict1.keys():
        if name in dict2:
            a = dict1[name]
            b = dict2[name]
            if not torch.is_floating_point(a):
                a = a.float()
            if not torch.is_floating_point(b):
                b = b.float()
            result[name] = a - b.to(a.device).type_as(a)
        else:
            result[name] = dict1[name].clone()
    return result

def scale_params_dict(params_dict, scale):
    """参数字典缩放"""
    result = {}
    for name, param in params_dict.items():
        p = param.float() if not torch.is_floating_point(param) else param
        result[name] = p * float(scale)
    return result

def zero_params_dict(params_dict):
    """创建全零参数字典"""
    result = {}
    for name, param in params_dict.items():
        result[name] = torch.zeros_like(param)
    return result