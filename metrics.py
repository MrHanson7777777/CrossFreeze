"""
评估指标 (分离存储版本)
"""
import numpy as np
import json
import os


class MetricsRecorder:
    """指标记录器 - 支持 CrossFreeze 和 Baseline 分离存储"""
    
    def __init__(self):
        # 通用指标
        self.metrics = {
            'round': [],
            'communication_cost': [],
            'test_loss': [], # 全局测试损失
            'train_loss': [], # 全局训练损失 (新增)
        }
        
        # --- CrossFreeze 专用 ---
        self.cf_metrics = {
            'loss_s1': [],
            'loss_s2': [],
            'loss_even': [],
            'hard_samples': [],
            'test_acc_m2': [], # 个性化
            'train_acc_m2': [],
            'test_acc_sm': [], # 全局
            'train_acc_sm': [],
            # 详细数据保留用于散点图等
            'clients_test_m2': [], 
            'clients_train_m2': [],
            'clients_test_sm': [],
            'clients_train_sm': [],
            # 新增：per-class 准确率
            'per_class_acc_m2': [],  # M2头的各类别准确率
            'per_class_acc_sm': [],  # Sm头的各类别准确率
        }
        
        # --- 偶数轮 Even Loss 单独记录 ---
        self.even_metrics = {
            'round': [],  # 偶数轮轮次
            'loss_even': []  # 偶数轮 loss
        }
        
        # --- Baseline (FedAvg) 专用 ---
        self.bl_metrics = {
            'test_acc': [],
            'train_acc': [],
            'clients_test_acc': [], # 记录每个客户端的精度，用于画方差带
            'clients_train_acc': [],
            # 新增：per-class 准确率
            'per_class_acc': [],  # Baseline的各类别准确率
        }
        
        # --- 早停相关 ---
        self.best_accuracy = 0.0  # 历史最佳准确率
        self.best_round = -1      # 最佳准确率对应的轮次
        self.patience_counter = 0 # 耐心计数器

    def add_crossfreeze_record(self, round_idx, comm_cost, 
                             test_loss, loss_s1, loss_s2, hard_samples,
                             w_test_m2, w_train_m2, w_test_sm, w_train_sm,
                             c_test_m2, c_train_m2, c_test_sm, c_train_sm,
                             per_class_m2=None, per_class_sm=None):
        """CrossFreeze 专用记录接口 (奇数轮)"""
        self.metrics['round'].append(round_idx)
        self.metrics['communication_cost'].append(comm_cost)
        self.metrics['test_loss'].append(test_loss)
        
        self.cf_metrics['loss_s1'].append(loss_s1)
        self.cf_metrics['loss_s2'].append(loss_s2)
        self.cf_metrics['hard_samples'].append(hard_samples)
        
        self.cf_metrics['test_acc_m2'].append(w_test_m2)
        self.cf_metrics['train_acc_m2'].append(w_train_m2)
        self.cf_metrics['test_acc_sm'].append(w_test_sm)
        self.cf_metrics['train_acc_sm'].append(w_train_sm)
        
        self.cf_metrics['clients_test_m2'].append(c_test_m2)
        self.cf_metrics['clients_train_m2'].append(c_train_m2)
        self.cf_metrics['clients_test_sm'].append(c_test_sm)
        self.cf_metrics['clients_train_sm'].append(c_train_sm)
        
        # 新增：per-class 准确率记录
        self.cf_metrics['per_class_acc_m2'].append(per_class_m2)
        self.cf_metrics['per_class_acc_sm'].append(per_class_sm)

    def add_even_loss_record(self, round_idx, loss_even):
        """记录偶数轮的 Even Loss"""
        self.even_metrics['round'].append(round_idx)
        self.even_metrics['loss_even'].append(loss_even)

    def add_baseline_record(self, round_idx, comm_cost, 
                          test_loss, train_loss,
                          w_test_acc, w_train_acc,
                          c_test_acc, c_train_acc,
                          per_class_acc=None):
        """Baseline 专用记录接口"""
        self.metrics['round'].append(round_idx)
        self.metrics['communication_cost'].append(comm_cost)
        self.metrics['test_loss'].append(test_loss)
        self.metrics['train_loss'].append(train_loss)
        
        self.bl_metrics['test_acc'].append(w_test_acc)
        self.bl_metrics['train_acc'].append(w_train_acc)
        self.bl_metrics['clients_test_acc'].append(c_test_acc)
        self.bl_metrics['clients_train_acc'].append(c_train_acc)
        
        # 新增：per-class 准确率记录
        self.bl_metrics['per_class_acc'].append(per_class_acc)
    
    def get_best_accuracy(self):
        """获取最佳准确率"""
        if self.bl_metrics['test_acc']:
            return max(self.bl_metrics['test_acc'])
        elif self.cf_metrics['test_acc_m2'] and self.cf_metrics['test_acc_sm']:
            return max(max(self.cf_metrics['test_acc_m2']), max(self.cf_metrics['test_acc_sm']))
        return 0
    
    def get_current_accuracy(self):
        """获取当前准确率 (最新一轮的准确率)"""
        if self.bl_metrics['test_acc']:
            return self.bl_metrics['test_acc'][-1]
        elif self.cf_metrics['test_acc_m2'] and self.cf_metrics['test_acc_sm']:
            # 对于 CrossFreeze，返回较好的模型的准确率
            latest_m2 = self.cf_metrics['test_acc_m2'][-1] if self.cf_metrics['test_acc_m2'] else 0
            latest_sm = self.cf_metrics['test_acc_sm'][-1] if self.cf_metrics['test_acc_sm'] else 0
            return max(latest_m2, latest_sm)
        return 0
    
    def should_early_stop(self, patience, min_delta=0.01):
        """
        判断是否应该早停
        
        Args:
            patience: 耐心轮数
            min_delta: 最小改善阈值 (百分点)
            
        Returns:
            tuple: (should_stop, improved, current_acc, best_acc)
        """
        current_acc = self.get_current_accuracy()
        
        # 检查是否有改善
        improved = current_acc - self.best_accuracy > min_delta
        
        if improved:
            # 有改善，更新最佳记录并重置计数器
            self.best_accuracy = current_acc
            self.best_round = len(self.metrics['round']) - 1 if self.metrics['round'] else 0
            self.patience_counter = 0
            return False, True, current_acc, self.best_accuracy
        else:
            # 无改善，增加计数器
            self.patience_counter += 1
            should_stop = self.patience_counter >= patience
            return should_stop, False, current_acc, self.best_accuracy
    
    def reset_early_stopping(self):
        """重置早停状态"""
        self.best_accuracy = 0.0
        self.best_round = -1
        self.patience_counter = 0
    
    def get_final_accuracy(self):
        """获取最终准确率"""
        if self.bl_metrics['test_acc']:
            return self.bl_metrics['test_acc'][-1]
        elif self.cf_metrics['test_acc_m2'] and self.cf_metrics['test_acc_sm']:
            # 返回最佳模型的最终准确率
            final_m2 = self.cf_metrics['test_acc_m2'][-1] if self.cf_metrics['test_acc_m2'] else 0
            final_sm = self.cf_metrics['test_acc_sm'][-1] if self.cf_metrics['test_acc_sm'] else 0
            best_m2 = max(self.cf_metrics['test_acc_m2']) if self.cf_metrics['test_acc_m2'] else 0
            best_sm = max(self.cf_metrics['test_acc_sm']) if self.cf_metrics['test_acc_sm'] else 0
            return final_m2 if best_m2 >= best_sm else final_sm
        return 0
    
    def save_to_file(self, filepath):
        """保存逻辑更新"""
        import os, json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'common': self.metrics,
            'crossfreeze': self.cf_metrics,
            'baseline': self.bl_metrics,
            'even': self.even_metrics
        }
        
        # 简单的序列化辅助函数
        def default(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=default)
        print(f"指标已保存到: {filepath}")
        
    def print_summary(self):
        """打印总结"""
        if self.bl_metrics['test_acc']:  # 如果有baseline数据
            print("=== Baseline (FedAvg) Summary ===")
            print(f"最终测试准确率: {self.bl_metrics['test_acc'][-1]:.2f}%")
            print(f"最佳测试准确率: {max(self.bl_metrics['test_acc']):.2f}%")
        elif self.cf_metrics['test_acc_sm']:  # 如果有CrossFreeze数据
            print("=== CrossFreeze Summary ===")
            print(f"M1+M2 最终准确率: {self.cf_metrics['test_acc_m2'][-1]:.2f}%")
            print(f"M1+Sm 最终准确率: {self.cf_metrics['test_acc_sm'][-1]:.2f}%")
            print(f"M1+M2 最佳准确率: {max(self.cf_metrics['test_acc_m2']):.2f}%")
            print(f"M1+Sm 最佳准确率: {max(self.cf_metrics['test_acc_sm']):.2f}%")


def calculate_communication_cost(params_dict):
    """计算参数字典的通信成本(参数数量)"""
    total = 0
    if params_dict is None:
        return 0
    for param in params_dict.values():
        total += param.numel()
    return total
