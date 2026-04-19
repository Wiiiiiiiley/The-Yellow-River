#!/usr/bin/env python3
"""
PINN vs Baseline 模型对比可视化
比较增强版 STGNN 与纯数据驱动 Baseline LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK', 'STHeiti', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stgnn_enhanced import create_enhanced_model
from src.models.baseline_models import BaselineLSTM


def generate_test_data(n_samples=100, seq_len=24, n_nodes=5, n_features=6):
    """生成测试数据 - 模拟黄河洪水场景"""
    
    data = []
    targets = []
    timestamps = []
    base_time = datetime(2024, 8, 15, 0, 0)
    
    for i in range(n_samples):
        flood_scenario = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        
        if flood_scenario == 'low':
            base_level = 1000 + np.random.rand() * 50
            base_discharge = 1200 + np.random.exponential(200)
        elif flood_scenario == 'medium':
            base_level = 1050 + np.random.rand() * 80
            base_discharge = 2000 + np.random.exponential(400)
        else:
            base_level = 1100 + np.random.rand() * 100
            base_discharge = 3500 + np.random.exponential(600)
        
        base_discharge = min(base_discharge, 5500)
        base_sediment = 5 + 0.004 * base_discharge + np.random.randn() * 3
        
        seq_data = []
        for t in range(seq_len):
            trend = np.sin(2 * np.pi * t / 24) * 30
            
            for node in range(n_nodes):
                delay = node * 2
                attenuation = 1 - node * 0.08
                
                effective_t = min(max(t - delay, 0), seq_len - 1)
                effective_trend = np.sin(2 * np.pi * effective_t / 24) * 30 * attenuation
                
                features = [
                    base_level + effective_trend + np.random.randn() * 8,
                    base_discharge + effective_trend * 2 + np.random.randn() * 80,
                    max(0, base_sediment + np.random.randn() * 2),
                    20 + np.random.randn() * 4,
                    max(0, np.random.exponential(3) if flood_scenario == 'high' else np.random.exponential(1)),
                    np.random.rand()
                ]
                seq_data.append(features)
        
        data.append(seq_data)
        
        target = []
        for t in range(6):
            future_t = seq_len + t
            for node in range(n_nodes):
                delay = node * 2
                attenuation = 1 - node * 0.08
                
                effective_t = min(max(future_t - delay, 0), seq_len - 1)
                effective_trend = np.sin(2 * np.pi * effective_t / 24) * 30 * attenuation
                
                target.append([
                    base_level + effective_trend * 1.3 + np.random.randn() * 12,
                    base_discharge + effective_trend * 2.5 + np.random.randn() * 120,
                    max(0, base_sediment * 1.15 + np.random.randn() * 3),
                    20 + np.random.randn() * 4,
                    max(0, np.random.exponential(2)),
                    np.random.rand()
                ])
        
        targets.append(target)
        timestamps.append(base_time + timedelta(hours=i))
    
    data = np.array(data).reshape(n_samples, seq_len, n_nodes, n_features)
    targets = np.array(targets).reshape(n_samples, 6, n_nodes, n_features)
    
    return torch.from_numpy(data).float(), torch.from_numpy(targets).float(), timestamps


def compute_metrics(pred, target):
    """计算评估指标"""
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = np.sqrt(mse)
    
    # R²
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    ss_res = torch.sum((target_flat - pred_flat) ** 2).item()
    ss_tot = torch.sum((target_flat - target_flat.mean()) ** 2).item()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2}


def plot_comparison(pinn_pred, baseline_pred, targets, timestamps, stations, save_path):
    """绘制 PINN vs Baseline 对比图"""
    
    features = ['水位(m)', '流量(m³/s)', '泥沙(kg/m³)']
    colors = {'targets': '#e74c3c', 'pinn': '#3498db', 'baseline': '#2ecc71'}
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('PINN vs Baseline 模型预测对比', fontsize=16, fontweight='bold')
    
    for f_idx, (feature_name, ax_row) in enumerate(zip(features, axes)):
        # 左列: 所有站点平均
        ax = ax_row[0]
        
        # 取前20个样本，所有站点平均
        n_show = min(20, len(pinn_pred))
        pinn_vals = pinn_pred[:n_show, 0, :, f_idx].mean(dim=1).numpy()
        baseline_vals = baseline_pred[:n_show, 0, :, f_idx].mean(dim=1).numpy()
        target_vals = targets[:n_show, 0, :, f_idx].mean(dim=1).numpy()
        
        x = np.arange(n_show)
        
        ax.plot(x, target_vals, 'o-', label='真实值', color=colors['targets'], 
                linewidth=2, markersize=5, alpha=0.8)
        ax.plot(x, pinn_vals, 's--', label='PINN (物理约束)', color=colors['pinn'], 
                linewidth=2, markersize=5, alpha=0.8)
        ax.plot(x, baseline_vals, '^:', label='Baseline (纯数据)', color=colors['baseline'], 
                linewidth=2, markersize=5, alpha=0.8)
        
        ax.set_title(f'{feature_name} - 多站点平均', fontsize=12, fontweight='bold')
        ax.set_xlabel('样本', fontsize=10)
        ax.set_ylabel(feature_name, fontsize=10)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 右列: 兰州站详细
        ax = ax_row[1]
        
        pinn_vals = pinn_pred[:n_show, :, 0, f_idx].mean(dim=1).numpy()  # 兰州站，所有时间步平均
        baseline_vals = baseline_pred[:n_show, :, 0, f_idx].mean(dim=1).numpy()
        target_vals = targets[:n_show, :, 0, f_idx].mean(dim=1).numpy()
        
        ax.plot(x, target_vals, 'o-', label='真实值', color=colors['targets'], 
                linewidth=2, markersize=5, alpha=0.8)
        ax.plot(x, pinn_vals, 's--', label='PINN', color=colors['pinn'], 
                linewidth=2, markersize=5, alpha=0.8)
        ax.plot(x, baseline_vals, '^:', label='Baseline', color=colors['baseline'], 
                linewidth=2, markersize=5, alpha=0.8)
        
        ax.set_title(f'{feature_name} - 兰州站', fontsize=12, fontweight='bold')
        ax.set_xlabel('样本', fontsize=10)
        ax.set_ylabel(feature_name, fontsize=10)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存: {save_path}")
    plt.close()


def plot_error_distribution(pinn_pred, baseline_pred, targets, save_path):
    """绘制误差分布对比"""
    
    features = ['水位(m)', '流量(m³/s)', '泥沙(kg/m³)']
    colors = {'pinn': '#3498db', 'baseline': '#2ecc71'}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('PINN vs Baseline 预测误差分布对比', fontsize=16, fontweight='bold')
    
    for f_idx, (feature_name, ax) in enumerate(zip(features, axes)):
        pinn_errors = (pinn_pred[:, :, :, f_idx] - targets[:, :, :, f_idx]).flatten().numpy()
        baseline_errors = (baseline_pred[:, :, :, f_idx] - targets[:, :, :, f_idx]).flatten().numpy()
        
        ax.hist(pinn_errors, bins=50, color=colors['pinn'], alpha=0.6, 
                label='PINN', edgecolor='black', linewidth=0.5)
        ax.hist(baseline_errors, bins=50, color=colors['baseline'], alpha=0.6, 
                label='Baseline', edgecolor='black', linewidth=0.5)
        
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='零误差')
        
        pinn_mae = np.mean(np.abs(pinn_errors))
        baseline_mae = np.mean(np.abs(baseline_errors))
        
        ax.set_xlabel('误差', fontsize=11)
        ax.set_ylabel('频次', fontsize=11)
        ax.set_title(f'{feature_name}\nPINN MAE={pinn_mae:.2f}, Baseline MAE={baseline_mae:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"误差分布图已保存: {save_path}")
    plt.close()


def generate_comparison_table(pinn_pred, baseline_pred, targets, save_path):
    """生成对比表格"""
    
    stations = ['兰州', '河口镇', '龙门', '三门峡', '花园口']
    features = ['水位(m)', '流量(m³/s)', '泥沙(kg/m³)']
    
    results = []
    
    for s_idx, station in enumerate(stations):
        for f_idx, feature in enumerate(features):
            pinn_mae = torch.mean(torch.abs(pinn_pred[:, :, s_idx, f_idx] - targets[:, :, s_idx, f_idx])).item()
            baseline_mae = torch.mean(torch.abs(baseline_pred[:, :, s_idx, f_idx] - targets[:, :, s_idx, f_idx])).item()
            
            improvement = (baseline_mae - pinn_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
            
            results.append({
                '站点': station,
                '特征': feature,
                'PINN_MAE': f'{pinn_mae:.3f}',
                'Baseline_MAE': f'{baseline_mae:.3f}',
                '提升(%)': f'{improvement:.1f}%',
                '优势': 'PINN' if improvement > 0 else 'Baseline'
            })
    
    df = pd.DataFrame(results)
    csv_path = save_path.replace('.png', '_metrics.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"对比表已保存: {csv_path}")
    
    # 打印摘要
    print("\n" + "="*70)
    print("PINN vs Baseline 性能对比摘要")
    print("="*70)
    
    summary = df.groupby('特征').apply(
        lambda x: pd.Series({
            'PINN平均MAE': f"{x['PINN_MAE'].astype(float).mean():.3f}",
            'Baseline平均MAE': f"{x['Baseline_MAE'].astype(float).mean():.3f}",
            '平均提升': f"{(x['Baseline_MAE'].astype(float) - x['PINN_MAE'].astype(float)).mean() / x['Baseline_MAE'].astype(float).mean() * 100:.1f}%"
        })
    )
    
    print(summary)
    print("="*70)


def main():
    print("="*70)
    print("🌊 PINN vs Baseline 模型对比")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 生成测试数据
    print("\n[1] 生成测试数据...")
    test_data, test_targets, timestamps = generate_test_data(n_samples=100)
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)
    print(f"    数据形状: {test_data.shape}")
    
    # 创建模型
    print("\n[2] 创建模型...")
    
    # PINN 模型配置
    pinn_config = {
        'node_features': 6,
        'n_nodes': 5,
        'gnn_hidden': 64,
        'gnn_layers': 3,
        'transformer_heads': 8,
        'transformer_layers': 4,
        'sequence_length': 24,
        'prediction_horizon': 6,
        'dropout': 0.1,
        'use_physics_attention': True,
        'use_adaptive_gnn': True,
        'use_multiscale_temporal': True,
        'use_physics_postprocess': True,
        'use_bayesian': False
    }
    
    pinn_model = create_enhanced_model(pinn_config).to(device)
    
    # Baseline LSTM
    baseline_model = BaselineLSTM(
        input_size=6,
        hidden_size=128,
        num_layers=2,
        output_size=6,
        prediction_horizon=6,
        dropout=0.1
    ).to(device)
    
    pinn_params = sum(p.numel() for p in pinn_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    
    print(f"    PINN (STGNN+物理约束): {pinn_params:,} 参数")
    print(f"    Baseline (LSTM): {baseline_params:,} 参数")
    
    # 加载训练好的权重（如果有）
    checkpoint_path = Path('experiments/models/best_model.pth')
    if checkpoint_path.exists():
        print(f"\n[3] 加载训练好的 PINN 模型...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        pinn_model.load_state_dict(checkpoint['model_state_dict'])
        print("    PINN 模型加载成功")
    else:
        print("\n[3] 未找到训练好的模型，使用随机权重进行对比演示")
    
    pinn_model.eval()
    baseline_model.eval()
    
    # 生成预测
    print("\n[4] 生成预测...")
    with torch.no_grad():
        pinn_pred = pinn_model(test_data)
        baseline_pred = baseline_model(test_data)
    
    print(f"    PINN预测: {pinn_pred.shape}")
    print(f"    Baseline预测: {baseline_pred.shape}")
    
    # 计算指标
    print("\n[5] 计算评估指标...")
    pinn_metrics = compute_metrics(pinn_pred, test_targets)
    baseline_metrics = compute_metrics(baseline_pred, test_targets)
    
    print(f"\n    📊 PINN (增强版 STGNN + 物理约束):")
    print(f"      MSE:  {pinn_metrics['mse']:.4f}")
    print(f"      MAE:  {pinn_metrics['mae']:.4f}")
    print(f"      RMSE: {pinn_metrics['rmse']:.4f}")
    print(f"      R²:   {pinn_metrics['r2']:.4f}")
    
    print(f"\n    📊 Baseline (纯数据驱动 LSTM):")
    print(f"      MSE:  {baseline_metrics['mse']:.4f}")
    print(f"      MAE:  {baseline_metrics['mae']:.4f}")
    print(f"      RMSE: {baseline_metrics['rmse']:.4f}")
    print(f"      R²:   {baseline_metrics['r2']:.4f}")
    
    improvement = (baseline_metrics['mae'] - pinn_metrics['mae']) / baseline_metrics['mae'] * 100
    print(f"\n    🎯 PINN 相对于 Baseline 提升: {improvement:.1f}%")
    
    # 生成可视化
    print("\n[6] 生成可视化图表...")
    
    output_dir = Path('experiments/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stations = ['兰州', '河口镇', '龙门', '三门峡', '花园口']
    
    # 对比图
    plot_comparison(
        pinn_pred.cpu(), baseline_pred.cpu(), test_targets.cpu(),
        timestamps, stations, str(output_dir / 'pinn_vs_baseline_comparison.png')
    )
    
    # 误差分布
    plot_error_distribution(
        pinn_pred.cpu(), baseline_pred.cpu(), test_targets.cpu(),
        str(output_dir / 'pinn_vs_baseline_errors.png')
    )
    
    # 对比表格
    generate_comparison_table(
        pinn_pred.cpu(), baseline_pred.cpu(), test_targets.cpu(),
        str(output_dir / 'pinn_vs_baseline_comparison.png')
    )
    
    # 保存指标
    metrics = {
        'pinn': pinn_metrics,
        'baseline': baseline_metrics,
        'improvement_percent': improvement
    }
    
    with open(output_dir / 'pinn_vs_baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[7] 指标已保存: {output_dir / 'pinn_vs_baseline_metrics.json'}")
    
    print("\n" + "="*70)
    print("✅ PINN vs Baseline 对比完成!")
    print("-"*70)
    print("📊 生成的文件:")
    print("    1. pinn_vs_baseline_comparison.png - 预测对比图")
    print("    2. pinn_vs_baseline_errors.png - 误差分布对比")
    print("    3. pinn_vs_baseline_metrics.csv - 详细指标表")
    print("    4. pinn_vs_baseline_metrics.json - 指标汇总")
    print("-"*70)
    print(f"📁 输出目录: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
