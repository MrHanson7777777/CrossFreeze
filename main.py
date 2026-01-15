import torch
import numpy as np
import random
import os
from datetime import datetime
from tqdm import tqdm

from config import get_args, get_dataset_config
# 路径假定: main.py 在根目录, data_loader.py 在 ./data/
from data.data_loader import get_client_dataloaders
from cnn_models import get_model, count_parameters, count_sm_parameters, get_baseline_model
from client import ClientManager
from server import CrossFreezeServer, select_clients
from baseline import BaselineClientManager, BaselineServer, evaluate_baseline_clients, evaluate_baseline_clients_detailed
from metrics import MetricsRecorder, calculate_communication_cost
from visualization import plot_all_metrics, plot_accuracy_curves
import matplotlib.pyplot as plt

def plot_crossfreeze_metrics(metrics_recorder, save_dir, experiment_name):
    """CrossFreeze 专用绘图函数"""
    print("生成 CrossFreeze 可视化图表...")
    
    # 使用英文标题避免字体问题
    rounds = metrics_recorder.metrics['round']
    
    # 1. CrossFreeze 准确率曲线 (训练+测试)
    plt.figure(figsize=(12, 8))
    
    # 绘制测试准确率（实线）
    plt.plot(rounds, metrics_recorder.cf_metrics['test_acc_m2'], 
             label='M1+M2 Test (Personalized)', marker='o', linewidth=2, color='blue', linestyle='-')
    plt.plot(rounds, metrics_recorder.cf_metrics['test_acc_sm'], 
             label='M1+Sm Test (Global)', marker='s', linewidth=2, color='red', linestyle='-')
    
    # 绘制训练准确率（虚线）
    plt.plot(rounds, metrics_recorder.cf_metrics['train_acc_m2'], 
             label='M1+M2 Train (Personalized)', marker='o', linewidth=2, color='blue', linestyle='--', alpha=0.7)
    plt.plot(rounds, metrics_recorder.cf_metrics['train_acc_sm'], 
             label='M1+Sm Train (Global)', marker='s', linewidth=2, color='red', linestyle='--', alpha=0.7)
    
    # 添加客户端方差带（如果有详细数据）
    if metrics_recorder.cf_metrics['clients_test_m2']:
        m2_std = [np.std(clients) for clients in metrics_recorder.cf_metrics['clients_test_m2']]
        m2_mean = metrics_recorder.cf_metrics['test_acc_m2']
        plt.fill_between(rounds, 
                        [m - s for m, s in zip(m2_mean, m2_std)],
                        [m + s for m, s in zip(m2_mean, m2_std)],
                        alpha=0.15, color='blue')
    
    if metrics_recorder.cf_metrics['clients_test_sm']:
        sm_std = [np.std(clients) for clients in metrics_recorder.cf_metrics['clients_test_sm']]
        sm_mean = metrics_recorder.cf_metrics['test_acc_sm']
        plt.fill_between(rounds, 
                        [m - s for m, s in zip(sm_mean, sm_std)],
                        [m + s for m, s in zip(sm_mean, sm_std)],
                        alpha=0.15, color='red')
    
    plt.title(f'{experiment_name} - CrossFreeze Train & Test Accuracy')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'crossfreeze_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 训练损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, metrics_recorder.cf_metrics['loss_s1'], 
             label='S1 Loss (M1+M2)', marker='o', linewidth=2)
    plt.plot(rounds, metrics_recorder.cf_metrics['loss_s2'], 
             label='S2 Loss (Sm)', marker='s', linewidth=2)
    
    # Even loss 单独绘制在偶数轮位置
    if metrics_recorder.even_metrics['round'] and metrics_recorder.even_metrics['loss_even']:
        even_rounds = metrics_recorder.even_metrics['round']
        even_losses = metrics_recorder.even_metrics['loss_even']
        plt.plot(even_rounds, even_losses, 
                 label='Even Loss (Consistency)', marker='^', linewidth=2, linestyle='--')
    
    plt.title(f'{experiment_name} - CrossFreeze Training Loss')
    plt.xlabel('Communication Round')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'crossfreeze_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 困难样本数量
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, metrics_recorder.cf_metrics['hard_samples'], 
             label='Hard Samples Count', marker='o', linewidth=2, color='orange')
    plt.title(f'{experiment_name} - CrossFreeze Hard Samples Trend')
    plt.xlabel('Communication Round')
    plt.ylabel('Hard Samples Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'crossfreeze_hard_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"CrossFreeze 可视化图表已保存到: {save_dir}")

def plot_baseline_metrics(metrics_recorder, save_dir, experiment_name):
    """Baseline (FedAvg) 专用绘图函数"""
    print("生成 Baseline 可视化图表...")
    
    # 使用英文标题避免字体问题
    rounds = metrics_recorder.metrics['round']
    
    # 1. FedAvg 准确率曲线
    plt.figure(figsize=(12, 8))
    
    # 绘制测试和训练准确率
    plt.plot(rounds, metrics_recorder.bl_metrics['test_acc'], 
             label='Test Accuracy', marker='o', linewidth=2, color='green')
    plt.plot(rounds, metrics_recorder.bl_metrics['train_acc'], 
             label='Train Accuracy', marker='s', linewidth=2, color='purple')
    
    # 添加客户端方差带（如果有详细数据）
    if metrics_recorder.bl_metrics['clients_test_acc']:
        test_std = [np.std(clients) for clients in metrics_recorder.bl_metrics['clients_test_acc']]
        test_mean = metrics_recorder.bl_metrics['test_acc']
        plt.fill_between(rounds, 
                        [m - s for m, s in zip(test_mean, test_std)],
                        [m + s for m, s in zip(test_mean, test_std)],
                        alpha=0.2, color='green')
    
    if metrics_recorder.bl_metrics['clients_train_acc']:
        train_std = [np.std(clients) for clients in metrics_recorder.bl_metrics['clients_train_acc']]
        train_mean = metrics_recorder.bl_metrics['train_acc']
        plt.fill_between(rounds, 
                        [m - s for m, s in zip(train_mean, train_std)],
                        [m + s for m, s in zip(train_mean, train_std)],
                        alpha=0.2, color='purple')
    
    plt.title(f'{experiment_name} - FedAvg Accuracy')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'baseline_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 测试和训练损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, metrics_recorder.metrics['test_loss'], 
             label='Test Loss', marker='o', linewidth=2, color='red')
    plt.plot(rounds, metrics_recorder.metrics['train_loss'], 
             label='Train Loss', marker='s', linewidth=2, color='blue')
    plt.title(f'{experiment_name} - FedAvg Loss')
    plt.xlabel('Communication Round')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'baseline_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Baseline 可视化图表已保存到: {save_dir}")

def get_timestamp():
    """生成当前时间的时间戳字符串,格式:YYYY-MM-DD_HH-MM-SS"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 启用cuDNN优化

def optimize_gpu_settings():
    """优化GPU设置以提高性能"""
    if torch.cuda.is_available():
        # 启用GPU优化
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # 设置内存分配策略
        torch.cuda.empty_cache()
        print(f"GPU优化已启用，可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def evaluate_clients(client_manager, test_loader, device):
    """
    评估所有客户端的个性化模型 (M1+M2)
    
    Args:
        client_manager: 客户端管理器
        test_loader: 全局测试集
        device: 设备
        
    Returns:
        (avg_accuracy, avg_loss)
    """
    accuracies = []
    losses = []
    
    # 遍历所有客户端
    all_clients = client_manager.get_all_clients()
    if not all_clients:
        return 0.0, 0.0
        
    for client in all_clients:
        # 每个客户端使用自己的 M1+M2 在 全局测试集上评估
        acc, loss = client.evaluate()
        accuracies.append(acc)
        losses.append(loss)
    
    avg_accuracy = np.mean(accuracies)
    avg_loss = np.mean(losses)
    
    return avg_accuracy, avg_loss

def evaluate_clients_detailed(client_manager, test_loader, device):
    """
    详细评估所有客户端的个性化模型和全局模型，返回加权平均的准确率
    
    Args:
        client_manager: 客户端管理器
        test_loader: 全局测试集
        device: 设备
        
    Returns:
        (weighted_avg_test_m2, weighted_avg_train_m2, weighted_avg_test_sm, weighted_avg_train_sm,
         test_accuracies_m2, train_accuracies_m2, test_accuracies_sm, train_accuracies_sm, client_weights)
    """
    test_accuracies_m2 = []  # M1+M2测试准确率
    train_accuracies_m2 = []  # M1+M2训练准确率
    test_accuracies_sm = []   # M1+Sm测试准确率
    train_accuracies_sm = []  # M1+Sm训练准确率
    client_weights = []       # 客户端权重(数据量)
    
    # 遍历所有客户端
    all_clients = client_manager.get_all_clients()
    if not all_clients:
        return 0.0, 0.0, 0.0, 0.0, [], [], [], [], []
        
    for client in all_clients:
        # M1+M2 评估
        test_acc_m2, _ = client.evaluate()  # 个性化模型测试准确率
        train_acc_m2 = client.evaluate_train()  # 个性化模型训练准确率
        
        # M1+Sm 评估
        test_acc_sm, _ = client.evaluate_sm()  # 全局模型测试准确率
        train_acc_sm = client.evaluate_train_sm()  # 全局模型训练准确率
        
        # 客户端数据量权重
        weight = client.get_num_samples()
        
        test_accuracies_m2.append(test_acc_m2)
        train_accuracies_m2.append(train_acc_m2)
        test_accuracies_sm.append(test_acc_sm)
        train_accuracies_sm.append(train_acc_sm)
        client_weights.append(weight)
    
    # 计算加权平均
    total_weight = sum(client_weights)
    if total_weight > 0:
        weighted_avg_test_m2 = sum(acc * w for acc, w in zip(test_accuracies_m2, client_weights)) / total_weight
        weighted_avg_train_m2 = sum(acc * w for acc, w in zip(train_accuracies_m2, client_weights)) / total_weight
        weighted_avg_test_sm = sum(acc * w for acc, w in zip(test_accuracies_sm, client_weights)) / total_weight
        weighted_avg_train_sm = sum(acc * w for acc, w in zip(train_accuracies_sm, client_weights)) / total_weight
    else:
        weighted_avg_test_m2 = weighted_avg_train_m2 = weighted_avg_test_sm = weighted_avg_train_sm = 0.0
    
    return (weighted_avg_test_m2, weighted_avg_train_m2, weighted_avg_test_sm, weighted_avg_train_sm,
            test_accuracies_m2, train_accuracies_m2, test_accuracies_sm, train_accuracies_sm, client_weights)

def run_crossfreeze(args):
    """运行CrossFreeze实验"""
    print("\n" + "="*60)
    print("运行 CrossFreeze 实验")
    print("="*60)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 优化GPU设置
    optimize_gpu_settings()
    
    # 获取数据集配置
    dataset_config = get_dataset_config(args.dataset)
    
    # 创建模型(CrossFreezeModel)
    model = get_model(args.dataset, dataset_config['num_classes'])
    total_params = count_parameters(model)
    sm_params_count = count_sm_parameters(model)
    sm_ratio = (sm_params_count / total_params) * 100
    
    print(f"模型总参数量: {total_params:,}")
    print(f"Sm (通信) 参数量: {sm_params_count:,}")
    print(f"Sm 参数占总参数量比例: {sm_ratio:.2f}%")
    
    # 加载数据
    # data_dir 是相对于 main.py 的路径, e.g., './data'
    client_loaders, test_loader = get_client_dataloaders(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        iid=(args.iid == 1),
        alpha=args.alpha,
        data_dir=args.data_dir 
    )
    
    # 创建服务器 (CrossFreezeServer)
    server = CrossFreezeServer(model, test_loader, device=args.device)
    
    # 创建客户端管理器 (CrossFreezeClient)
    client_manager = ClientManager(
        dataset_name=args.dataset,  # 【新增】传递数据集名称
        num_clients=args.num_clients,
        model=model, # 原型模型
        client_loaders=client_loaders,
        test_loader=test_loader,
        lr=args.lr,
        local_epochs=args.local_epochs,
        device=args.device,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_decay_step=args.lr_decay_step, # (此参数 CosineAnnealingLR 不再使用)
        lr_decay_gamma=args.lr_decay_gamma, # (此参数 CosineAnnealingLR 不再使用)
        mu=args.mu,
        total_epochs=args.epochs,  # --- 新增：传入总轮数 ---
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        cutmix_min_ratio=args.cutmix_min_ratio,
        cutmix_max_ratio=args.cutmix_max_ratio,
        use_cutmix=(args.use_cutmix == 1),
        mixup_cutmix_ratio=args.mixup_cutmix_ratio,
        consistency_beta=args.consistency_beta,  # 【新增】传递参数
        min_lr=args.min_lr,  # 【新增】传递最小学习率参数
        gamma_sm=args.gamma_sm  # 【新增】传递S1损失权重参数
    )
    
    # 指标记录
    metrics = MetricsRecorder()
    
    # 训练
    print(f"\n开始训练 {args.epochs} 轮...")
    for round_idx in tqdm(range(args.epochs), desc="训练进度"):
        
        # 选择客户端
        selected_ids = select_clients(args.num_clients, args.frac)
        
        train_losses_s1 = []
        train_losses_s2 = []
        train_losses_even = []
        hard_samples_counts = []
        comm_cost = 0.0

        # 获取最新的全局Sm
        global_sm_state_dict = server.get_global_sm_state_dict()

        if (round_idx + 1) % 2 == 1:
            # --- 奇数轮: S1, S2, S3(聚合) ---
            client_sm_params = []
            client_weights = []
            
            for client_id in selected_ids:
                client = client_manager.get_client(client_id)
                
                if round_idx == 0:
                   # 奇数轮S2阶段训练需要Sm参数,这相当于理论上前面奇数轮的S4阶段
                   client.set_sm_parameters(global_sm_state_dict)
                
                # 客户端执行S1和S2
                sm_params, loss_s1, loss_s2, n_hard = client.train(round_idx, args.epochs)
                
                train_losses_s1.append(loss_s1)
                train_losses_s2.append(loss_s2)
                hard_samples_counts.append(n_hard)
                
                # 收集Sm参数
                client_sm_params.append(sm_params)
                client_weights.append(client.get_num_samples())
            
            # S3阶段服务器聚合
            server.aggregate(client_sm_params, client_weights)
            
            # 通信成本: N个客户端上传Sm + N个客户端下载Sm
            comm_cost = sm_params_count * len(selected_ids) * 2
            
            avg_loss_s1 = np.mean(train_losses_s1)
            avg_loss_s2 = np.mean(train_losses_s2)
            avg_hard_samples = np.mean(hard_samples_counts)

        else:
            # --- 偶数轮: 优化 M1 ---
            for client_id in selected_ids:
                client = client_manager.get_client(client_id)
                
                # 客户端下载最新的Sm_global,相当于理论上前面奇数轮的S4阶段
                client.set_sm_parameters(global_sm_state_dict)
                
                # 客户端执行Even 轮(只训练M1)
                _, loss_even, _, _ = client.train(round_idx, args.epochs)
                train_losses_even.append(loss_even)

            comm_cost = 0.0 
            
            avg_loss_s1 = np.nan
            avg_loss_s2 = np.nan
            avg_loss_even = np.mean(train_losses_even)
            avg_hard_samples = 0
            
            # 单独记录偶数轮的 Even Loss
            metrics.add_even_loss_record(round_idx, avg_loss_even)
        
        # 评估 - 只在奇数轮进行 (因为偶数轮没有聚合，Sm 还是旧的)
        # 这样可以避免 M1+Sm 准确率曲线的锯齿状波动
        evaluate_this_round = False
        
        if (round_idx + 1) % 2 == 1:  # 奇数轮
            if round_idx % args.log_interval == 0 or round_idx == args.epochs - 1:
                evaluate_this_round = True
        
        if evaluate_this_round:
            # 详细评估所有客户端的模型性能
            (weighted_avg_test_m2, weighted_avg_train_m2, weighted_avg_test_sm, weighted_avg_train_sm,
             test_accuracies_m2, train_accuracies_m2, test_accuracies_sm, train_accuracies_sm, 
             client_weights) = evaluate_clients_detailed(client_manager, test_loader, args.device)
            
            test_acc, test_loss = evaluate_clients(client_manager, test_loader, args.device)
            
            # 使用新的 CrossFreeze 记录接口
            metrics.add_crossfreeze_record(
                round_idx=round_idx,
                comm_cost=comm_cost,
                test_loss=test_loss,
                loss_s1=avg_loss_s1,
                loss_s2=avg_loss_s2,
                hard_samples=avg_hard_samples,
                w_test_m2=weighted_avg_test_m2,
                w_train_m2=weighted_avg_train_m2,
                w_test_sm=weighted_avg_test_sm,
                w_train_sm=weighted_avg_train_sm,
                c_test_m2=test_accuracies_m2,
                c_train_m2=train_accuracies_m2,
                c_test_sm=test_accuracies_sm,
                c_train_sm=train_accuracies_sm
            )
            
            # 早停检查
            if args.early_stopping == 1:
                should_stop, improved, current_acc, best_acc = metrics.should_early_stop(
                    patience=args.patience, 
                    min_delta=args.min_delta
                )
                
                if should_stop:
                    print(f"\n🛑 早停触发!")
                    print(f"  当前准确率: {current_acc:.2f}%")
                    print(f"  最佳准确率: {best_acc:.2f}% (第 {metrics.best_round+1} 轮)")
                    print(f"  已连续 {args.patience} 轮无改善 (阈值: {args.min_delta:.2f}%)")
                    print(f"  提前结束训练 (计划 {args.epochs} 轮, 实际 {round_idx+1} 轮)")
                    break
                elif improved:
                    print(f"\n✅ 准确率提升: {current_acc:.2f}% (最佳: {best_acc:.2f}%)")
                else:
                    print(f"\n⏳ 无改善 ({metrics.patience_counter}/{args.patience}): 当前 {current_acc:.2f}%, 最佳 {best_acc:.2f}%")
            
            print(f"\n{'='*80}")
            actual_round = round_idx + 1
            round_type = "奇数轮(S1+S2)" if (round_idx + 1) % 2 == 1 else "偶数轮(Even)"
            print(f"第 {actual_round} 轮 ({round_type}):")
            print(f"  Sm参数占比: {sm_ratio:.2f}% ({sm_params_count:,}/{total_params:,})")
            print(f"\n  【加权平均准确率 (基于客户端数据量)】")
            print(f"  M1+M2 (个性化): 测试={weighted_avg_test_m2:.2f}%, 训练={weighted_avg_train_m2:.2f}%")
            print(f"  M1+Sm (全局):   测试={weighted_avg_test_sm:.2f}%, 训练={weighted_avg_train_sm:.2f}%")
            print(f"  测试损失: {test_loss:.4f}")
            
            if not np.isnan(avg_loss_s1):
                print(f"  S1(M1+M2) Loss: {avg_loss_s1:.4f}")
            if not np.isnan(avg_loss_s2):
                print(f"  S2(Sm) Loss: {avg_loss_s2:.4f} (Avg Hard: {avg_hard_samples:.1f})")
            # Even loss 将在单独的 even_metrics 中记录
            
            # 打印每个客户端的详细准确率
            print(f"\n  【各客户端详细准确率】")
            print(f"  {'ID':<4} {'数据量':<8} {'M1+M2测试':<12} {'M1+M2训练':<12} {'M1+Sm测试':<12} {'M1+Sm训练':<12}")
            print(f"  {'-'*4} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
            for i, (test_m2, train_m2, test_sm, train_sm, weight) in enumerate(
                zip(test_accuracies_m2, train_accuracies_m2, test_accuracies_sm, train_accuracies_sm, client_weights)):
                print(f"  {i:<4} {weight:<8} {test_m2:>10.2f}% {train_m2:>10.2f}% {test_sm:>10.2f}% {train_sm:>10.2f}%")
            print(f"{'='*80}")
        else:
            # 偶数轮只显示简单的训练损失信息
            print(f"\n第 {round_idx + 1} 轮 (偶数轮-Even): M1对齐训练")
            if not np.isnan(avg_loss_even):
                print(f"  Even(M1) Loss: {avg_loss_even:.4f}")
            print(f"  (跳过评估，避免 M1+Sm 曲线锯齿)")

    
    # 保存结果
    iid_str = "iid" if args.iid == 1 else "noniid"
    timestamp = get_timestamp()
    save_name = f"crossfreeze_{args.dataset}_{iid_str}_{timestamp}"
    
    save_dir = os.path.join(args.save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    
    metrics.save_to_file(os.path.join(save_dir, 'metrics.json'))
    metrics.print_summary()
    
    if args.plot:
        # 使用 CrossFreeze 专用绘图函数
        plot_crossfreeze_metrics(metrics, save_dir, save_name)
        
        # 【新增】绘制 M1+M2 vs M1+Sm 的准确率对比图
        plot_accuracy_curves(metrics, 
                           os.path.join(save_dir, 'accuracy_comparison.png'),
                           title=f"{save_name} - M1+M2 vs M1+Sm Accuracy Comparison")
    else:
        print("可视化图表生成已跳过 (使用 --plot 1 启用)")
    
    if args.save_model:
        # 仅保存全局Sm
        torch.save(server.get_global_sm_state_dict(), 
                  os.path.join(save_dir, 'final_global_sm.pt'))
    
    return metrics

def run_baseline_fedavg(args):
    """运行标准FedAvg基线实验"""
    print("\n" + "="*60)
    print("运行 标准FedAvg 基线实验")
    print("="*60)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 优化GPU设置
    optimize_gpu_settings()
    
    # 获取数据集配置
    dataset_config = get_dataset_config(args.dataset)
    
    # 创建模型 - 使用标准整体模型而不是CrossFreeze模型
    model = get_baseline_model(args.dataset, dataset_config['num_classes'])
    total_params = count_parameters(model)
    # FedAvg传输和使用全部模型参数
    comm_params = total_params
    useful_params = total_params
    comm_ratio = (comm_params / useful_params) * 100  # 100%
    
    print(f"模型总参数量: {total_params:,}")
    print(f"FedAvg使用参数量: {useful_params:,}")
    print(f"FedAvg通信参数量: {comm_params:,}")
    print(f"通信参数占使用参数量比例: {comm_ratio:.2f}%")  # 应该是100%
    
    # 加载数据
    client_loaders, test_loader = get_client_dataloaders(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        iid=(args.iid == 1),
        alpha=args.alpha,
        data_dir=args.data_dir 
    )
    
    # 创建服务器
    server = BaselineServer(model, device=args.device)
    
    # 创建客户端管理器
    client_manager = BaselineClientManager(
        dataset_name=args.dataset,  # 【新增】传递数据集名称
        num_clients=args.num_clients,
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        lr=args.lr,
        local_epochs=args.local_epochs,
        device=args.device,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        # --- 传入 lr_decay 参数 (这些参数来自 config.py) ---
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma
    )
    
    # 指标记录
    metrics = MetricsRecorder()
    
    # 训练
    print(f"\n开始训练 {args.epochs} 轮...")
    for round_idx in tqdm(range(args.epochs), desc="训练进度"):
        
        # 选择客户端
        selected_ids = select_clients(args.num_clients, args.frac)
        
        train_losses = []
        client_params_list = []
        client_weights = []

        # 获取全局参数
        global_params = server.get_global_parameters()
        
        # 客户端训练
        for client_id in selected_ids:
            client = client_manager.get_client(client_id)
            
            # 下载全局参数
            client.set_parameters(global_params)
            
            # 本地训练
            local_params, train_loss = client.train(round_idx)
            
            train_losses.append(train_loss)
            client_params_list.append(local_params)
            client_weights.append(client.get_num_samples())
        
        # 服务器聚合
        server.aggregate(client_params_list, client_weights)
        
        # 通信成本: N个客户端上传(M1+M2) + N个客户端下载(M1+M2)
        comm_cost = comm_params * len(selected_ids) * 2
        
        avg_train_loss = np.mean(train_losses)
        
        # 评估
        if round_idx % args.log_interval == 0 or round_idx == args.epochs - 1:
            
            # 评估所有客户端 
            (weighted_avg_test, weighted_avg_train, test_accuracies, train_accuracies, 
             client_weights_eval) = evaluate_baseline_clients_detailed(client_manager, test_loader, args.device)
            
            test_acc, test_loss = evaluate_baseline_clients(client_manager, test_loader, args.device)
            
            # 使用新的 Baseline 记录接口
            metrics.add_baseline_record(
                round_idx=round_idx,
                comm_cost=comm_cost,
                test_loss=test_loss,
                train_loss=avg_train_loss,
                w_test_acc=weighted_avg_test,
                w_train_acc=weighted_avg_train,
                c_test_acc=test_accuracies,
                c_train_acc=train_accuracies
            )
            
            # 早停检查
            if args.early_stopping == 1:
                should_stop, improved, current_acc, best_acc = metrics.should_early_stop(
                    patience=args.patience, 
                    min_delta=args.min_delta
                )
                
                if should_stop:
                    print(f"\n🛑 早停触发!")
                    print(f"  当前准确率: {current_acc:.2f}%")
                    print(f"  最佳准确率: {best_acc:.2f}% (第 {metrics.best_round+1} 轮)")
                    print(f"  已连续 {args.patience} 轮无改善 (阈值: {args.min_delta:.2f}%)")
                    print(f"  提前结束训练 (计划 {args.epochs} 轮, 实际 {round_idx+1} 轮)")
                    break
                elif improved:
                    print(f"\n✅ 准确率提升: {current_acc:.2f}% (最佳: {best_acc:.2f}%)")
                else:
                    print(f"\n⏳ 无改善 ({metrics.patience_counter}/{args.patience}): 当前 {current_acc:.2f}%, 最佳 {best_acc:.2f}%")
            
            print(f"\n{'='*80}")
            actual_round = round_idx + 1
            print(f"第 {actual_round} 轮 (FedAvg):")
            print(f"  通信参数占比: {comm_ratio:.2f}% ({comm_params:,}/{useful_params:,})")
            print(f"\n  【加权平均准确率 (基于客户端数据量)】")
            print(f"  FedAvg 标准模型: 测试={weighted_avg_test:.2f}%, 训练={weighted_avg_train:.2f}%")
            print(f"  测试损失: {test_loss:.4f}")
            print(f"  训练损失: {avg_train_loss:.4f}")
            
            # 打印每个客户端的详细准确率
            print(f"\n  【各客户端详细准确率】")
            print(f"  {'ID':<4} {'数据量':<8} {'测试准确率':<12} {'训练准确率':<12}")
            print(f"  {'-'*4} {'-'*8} {'-'*12} {'-'*12}")
            for i, (test_acc_i, train_acc_i, weight) in enumerate(
                zip(test_accuracies, train_accuracies, client_weights_eval)):
                print(f"  {i:<4} {weight:<8} {test_acc_i:>10.2f}% {train_acc_i:>10.2f}%")
            print(f"{'='*80}")
    
    # 保存结果
    iid_str = "iid" if args.iid == 1 else "noniid"
    timestamp = get_timestamp()
    save_name = f"baseline_fedavg_{args.dataset}_{iid_str}_{timestamp}"
    
    save_dir = os.path.join(args.save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    
    metrics.save_to_file(os.path.join(save_dir, 'metrics.json'))
    metrics.print_summary()
    
    if args.plot:
        # 使用 Baseline 专用绘图函数
        plot_baseline_metrics(metrics, save_dir, save_name)
    else:
        print("可视化图表生成已跳过 (使用 --plot 1 启用)")
    
    if args.save_model:
        # 保存全局模型
        torch.save(server.get_global_parameters(), 
                  os.path.join(save_dir, 'final_global_model.pt'))
    
    return metrics

def main():
    """主函数"""
    args = get_args()
    
    print("\n" + "="*60)
    print("CrossFreeze 联邦学习")
    print("="*60)
    print(f"实验类型: {args.experiment}")
    print(f"数据集: {args.dataset}")
    print(f"IID: {'是' if args.iid == 1 else '否'}")
    print(f"客户端数: {args.num_clients}")
    print(f"参与比例: {args.frac}")
    print(f"全局轮数: {args.epochs}")
    print(f"本地轮数: {args.local_epochs}")
    print(f"设备: {args.device}")
    print("="*60 + "\n")
    
    # 根据实验类型运行
    if args.experiment == 'crossfreeze':
        run_crossfreeze(args)
    elif args.experiment == 'baseline':
        run_baseline_fedavg(args)
    
    print("\n实验完成!")


if __name__ == '__main__':
    main()