#!/usr/bin/env python3
"""
可视化训练好的模型预测结果
绘制预测 vs 真实值对比图
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
import sys
from pathlib import Path

# 设置中文字体 (macOS系统字体)
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK', 'STHeiti', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stgnn_enhanced import create_enhanced_model


def generate_test_data(n_samples=20, seq_len=24, n_nodes=5, n_features=6):
    """生成测试数据"""
    
    data = []
    targets = []
    
    # 创建不同洪水场景列表
    scenario_types = ['low', 'medium', 'high', 'extreme']
    
    for i in range(n_samples):
        scenario = scenario_types[i % 4]
        
        if scenario == 'low':
            base_level = 1000 + np.random.rand() * 50
            base_discharge = 1200 + np.random.exponential(200)
        elif scenario == 'medium':
            base_level = 1050 + np.random.rand() * 80
            base_discharge = 2000 + np.random.exponential(400)
        elif scenario == 'high':
            base_level = 1100 + np.random.rand() * 100
            base_discharge = 3500 + np.random.exponential(600)
        else:  # extreme
            base_level = 1150 + np.random.rand() * 100
            base_discharge = 4500 + np.random.exponential(800)
        
        base_discharge = min(base_discharge, 5500)
        base_sediment = 5 + 0.004 * base_discharge + np.random.randn() * 3
        
        # 序列数据
        seq_data = []
        for t in range(seq_len):
            trend = np.sin(2 * np.pi * t / 24) * 30
            for node in range(n_nodes):
                features = [
                    base_level + trend + np.random.randn() * 8,
                    base_discharge + trend * 2 + np.random.randn() * 80,
                    max(0, base_sediment + np.random.randn() * 2),
                    20 + np.random.randn() * 4,
                    max(0, np.random.exponential(3) if scenario in ['high', 'extreme'] else np.random.exponential(1)),
                    np.random.rand()
                ]
                seq_data.append(features)
        
        data.append(seq_data)
        
        # 目标
        target = []
        for t in range(6):
            for node in range(n_nodes):
                target.append([
                    base_level + np.random.randn() * 15,
                    base_discharge + np.random.randn() * 120,
                    max(0, base_sediment + np.random.randn() * 3),
                    20 + np.random.randn() * 4,
                    max(0, np.random.exponential(2)),
                    np.random.rand()
                ])
        targets.append(target)
    
    data = np.array(data).reshape(n_samples, seq_len, n_nodes, n_features)
    targets = np.array(targets).reshape(n_samples, 6, n_nodes, n_features)
    
    # 生成每个样本的场景标签
    sample_scenarios = [scenario_types[i % 4] for i in range(n_samples)]
    
    return torch.from_numpy(data).float(), torch.from_numpy(targets).float(), sample_scenarios


def main():
    print("="*70)
    print("📊 生成预测 vs 真实值可视化")
    print("="*70)
    
    device = torch.device('cpu')
    
    # 加载模型
    print("\n[1] 加载训练好的模型...")
    checkpoint_path = Path('experiments/models/pinn_best.pth')
    if not checkpoint_path.exists():
        checkpoint_path = Path('experiments/models/best_model.pth')
    if not checkpoint_path.exists():
        raise FileNotFoundError("未找到 PINN 检查点，请先运行训练脚本")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get('protocol_version', 1) < 2:
        raise RuntimeError("检查点使用旧评估协议，请先重新训练模型")
    model_config = checkpoint.get('config', checkpoint.get('model_config'))
    if model_config is None:
        raise KeyError(f"检查点 {checkpoint_path} 缺少模型配置")
    
    model = create_enhanced_model(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    mean = checkpoint.get('mean', checkpoint.get('data_mean'))
    std = checkpoint.get('std', checkpoint.get('data_std'))
    if mean is None or std is None:
        raise KeyError(f"检查点 {checkpoint_path} 缺少训练集 mean/std")
    mean = mean.squeeze().view(1, 1, 1, -1)
    std = std.squeeze().view(1, 1, 1, -1)
    
    print("    模型加载成功!")
    
    # 生成测试数据
    print("\n[2] 生成测试数据...")
    test_data, test_targets, scenarios = generate_test_data(n_samples=12)
    
    # 归一化
    test_data_norm = (test_data - mean) / std
    
    # 预测
    print("\n[3] 进行预测...")
    with torch.no_grad():
        predictions_norm = model(test_data_norm)
    
    # 反归一化
    predictions = predictions_norm * std + mean
    
    print(f"    预测形状: {predictions.shape}")
    
    # 转换为 numpy
    pred_np = predictions.cpu().numpy()
    target_np = test_targets.cpu().numpy()
    
    # 站点名称
    stations = ['兰州', '河口镇', '龙门', '三门峡', '花园口']
    features = ['水位(m)', '流量(m³/s)', '泥沙(kg/m³)']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # 计算误差指标
    mse = np.mean((pred_np - target_np) ** 2)
    mae = np.mean(np.abs(pred_np - target_np))
    rmse = np.sqrt(mse)
    
    print(f"\n[4] 预测性能:")
    print(f"    MSE:  {mse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    
    # 创建可视化
    print("\n[5] 生成可视化图表...")
    
    # 图1: 水位预测对比 (多站点)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('黄河防汛 PINN v3.0 - 水位预测 vs 真实值', fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(stations):
            station = stations[idx]
            
            # 取前6个样本的前3个时间步
            pred_water = pred_np[:6, :3, idx, 0].flatten()
            true_water = target_np[:6, :3, idx, 0].flatten()
            
            x = np.arange(len(pred_water))
            
            ax.plot(x, true_water, 'o-', label='真实值', color='#e74c3c', linewidth=2, markersize=6)
            ax.plot(x, pred_water, 's--', label='预测值', color='#3498db', linewidth=2, markersize=6)
            
            ax.set_title(f'{station}', fontsize=12, fontweight='bold')
            ax.set_xlabel('时间步', fontsize=10)
            ax.set_ylabel('水位 (m)', fontsize=10)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/visualizations/water_level_comparison.png', dpi=300, bbox_inches='tight')
    print("    保存: water_level_comparison.png")
    plt.close()
    
    # 图2: 流量预测对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('黄河防汛 PINN v3.0 - 流量预测 vs 真实值', fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(stations):
            station = stations[idx]
            
            pred_discharge = pred_np[:6, :3, idx, 1].flatten()
            true_discharge = target_np[:6, :3, idx, 1].flatten()
            
            x = np.arange(len(pred_discharge))
            
            ax.plot(x, true_discharge, 'o-', label='真实值', color='#e74c3c', linewidth=2, markersize=6)
            ax.plot(x, pred_discharge, 's--', label='预测值', color='#2ecc71', linewidth=2, markersize=6)
            
            ax.set_title(f'{station}', fontsize=12, fontweight='bold')
            ax.set_xlabel('时间步', fontsize=10)
            ax.set_ylabel('流量 (m³/s)', fontsize=10)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/visualizations/discharge_comparison.png', dpi=300, bbox_inches='tight')
    print("    保存: discharge_comparison.png")
    plt.close()
    
    # 图3: 泥沙浓度预测
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('黄河防汛 PINN v3.0 - 泥沙浓度预测 vs 真实值', fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(stations):
            station = stations[idx]
            
            pred_sediment = pred_np[:6, :3, idx, 2].flatten()
            true_sediment = target_np[:6, :3, idx, 2].flatten()
            
            x = np.arange(len(pred_sediment))
            
            ax.plot(x, true_sediment, 'o-', label='真实值', color='#e74c3c', linewidth=2, markersize=6)
            ax.plot(x, pred_sediment, 's--', label='预测值', color='#f39c12', linewidth=2, markersize=6)
            
            ax.set_title(f'{station}', fontsize=12, fontweight='bold')
            ax.set_xlabel('时间步', fontsize=10)
            ax.set_ylabel('泥沙浓度 (kg/m³)', fontsize=10)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/visualizations/sediment_comparison.png', dpi=300, bbox_inches='tight')
    print("    保存: sediment_comparison.png")
    plt.close()
    
    # 图4: 散点图 - 预测 vs 真实 (所有站点合并)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('黄河防汛 PINN v3.0 - 预测精度散点图', fontsize=16, fontweight='bold')
    
    for f_idx, (feature_name, ax) in enumerate(zip(features, axes)):
        pred_flat = pred_np[:, :, :, f_idx].flatten()
        true_flat = target_np[:, :, :, f_idx].flatten()
        
        # 绘制散点
        ax.scatter(true_flat, pred_flat, alpha=0.5, s=20, color=colors[f_idx])
        
        # 理想线 y=x
        min_val = min(true_flat.min(), pred_flat.min())
        max_val = max(true_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线 (y=x)')
        
        # 计算 R²
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        ax.set_xlabel(f'真实值 - {feature_name}', fontsize=11)
        ax.set_ylabel(f'预测值 - {feature_name}', fontsize=11)
        ax.set_title(f'{feature_name}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/visualizations/scatter_comparison.png', dpi=300, bbox_inches='tight')
    print("    保存: scatter_comparison.png")
    plt.close()
    
    # 图5: 误差分布
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('黄河防汛 PINN v3.0 - 预测误差分布', fontsize=16, fontweight='bold')
    
    for f_idx, (feature_name, ax) in enumerate(zip(features, axes)):
        errors = (pred_np[:, :, :, f_idx] - target_np[:, :, :, f_idx]).flatten()
        
        ax.hist(errors, bins=50, color=colors[f_idx], alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='零误差')
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        ax.set_xlabel('误差', fontsize=11)
        ax.set_ylabel('频次', fontsize=11)
        ax.set_title(f'{feature_name}\nMean={mean_error:.2f}, Std={std_error:.2f}', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/visualizations/error_distribution.png', dpi=300, bbox_inches='tight')
    print("    保存: error_distribution.png")
    plt.close()
    
    # 图6: 时间序列 - 单个样本详细对比
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('黄河防汛 PINN v3.0 - 单个样本详细预测 (兰州站)', fontsize=16, fontweight='bold')
    
    sample_idx = 0
    station_idx = 0
    time_steps = np.arange(6)
    
    for f_idx, (feature_name, ax) in enumerate(zip(features, axes)):
        pred_vals = pred_np[sample_idx, :, station_idx, f_idx]
        true_vals = target_np[sample_idx, :, station_idx, f_idx]
        
        ax.plot(time_steps, true_vals, 'o-', label='真实值', color='#e74c3c', linewidth=2, markersize=8)
        ax.plot(time_steps, pred_vals, 's--', label='预测值', color='#3498db', linewidth=2, markersize=8)
        
        # 填充误差区域
        ax.fill_between(time_steps, true_vals, pred_vals, alpha=0.2, color='gray', label='误差')
        
        ax.set_ylabel(feature_name, fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(time_steps)
        ax.set_xticklabels([f't+{i+1}h' for i in time_steps])
    
    axes[-1].set_xlabel('预测时间步', fontsize=12)
    plt.tight_layout()
    plt.savefig('experiments/visualizations/time_series_detail.png', dpi=300, bbox_inches='tight')
    print("    保存: time_series_detail.png")
    plt.close()
    
    # 保存数据到 CSV
    print("\n[6] 保存预测数据...")
    
    # 创建详细数据表
    results = []
    for sample in range(12):
        for station_idx, station in enumerate(stations):
            for t in range(6):
                results.append({
                    'sample_id': sample,
                    'station': station,
                    'scenario': scenarios[sample],
                    'time_step': f't+{t+1}h',
                    'pred_water': round(float(pred_np[sample, t, station_idx, 0]), 2),
                    'true_water': round(float(target_np[sample, t, station_idx, 0]), 2),
                    'pred_discharge': round(float(pred_np[sample, t, station_idx, 1]), 2),
                    'true_discharge': round(float(target_np[sample, t, station_idx, 1]), 2),
                    'pred_sediment': round(float(pred_np[sample, t, station_idx, 2]), 3),
                    'true_sediment': round(float(target_np[sample, t, station_idx, 2]), 3),
                })
    
    df = pd.DataFrame(results)
    df.to_csv('experiments/visualizations/prediction_results.csv', index=False, encoding='utf-8-sig')
    print("    保存: prediction_results.csv")
    
    # 保存指标
    metrics = {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
    }
    
    # 计算每个特征的 R²
    for f_idx, feature_name in enumerate(features):
        pred_flat = pred_np[:, :, :, f_idx].flatten()
        true_flat = target_np[:, :, :, f_idx].flatten()
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics[f'R2_{feature_name}'] = float(r2)
    
    with open('experiments/visualizations/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("    保存: metrics.json")
    
    print("\n" + "="*70)
    print("✅ 可视化生成完成!")
    print("-" * 70)
    print("📊 生成的文件:")
    print("    1. water_level_comparison.png    - 水位预测对比")
    print("    2. discharge_comparison.png      - 流量预测对比")
    print("    3. sediment_comparison.png         - 泥沙浓度预测对比")
    print("    4. scatter_comparison.png        - 预测精度散点图")
    print("    5. error_distribution.png        - 误差分布图")
    print("    6. time_series_detail.png        - 时间序列详细图")
    print("    7. prediction_results.csv        - 预测数据表")
    print("    8. metrics.json                  - 性能指标")
    print("-" * 70)
    print("📁 输出目录: experiments/visualizations/")
    print("="*70)


if __name__ == "__main__":
    main()
