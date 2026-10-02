#!/usr/bin/env python3
"""
同时训练 PINN 和 Baseline 模型
确保公平对比：相同数据、相同训练轮数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stgnn_enhanced import create_enhanced_model
from src.models.baseline_models import BaselineLSTM
from src.models.physics_constraints_enhanced import EnhancedPhysicsConstraintLoss


def generate_training_data(n_samples=500, seq_len=24, n_nodes=5, n_features=6):
    """生成训练数据"""
    data = []
    targets = []
    
    for _ in range(n_samples):
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
    
    data = np.array(data).reshape(n_samples, seq_len, n_nodes, n_features)
    targets = np.array(targets).reshape(n_samples, 6, n_nodes, n_features)
    
    return torch.from_numpy(data).float(), torch.from_numpy(targets).float()


def _feature_stats_view(stats, data):
    """将特征统计量转换成可与时空张量广播的形状。"""
    stats = torch.as_tensor(stats, dtype=data.dtype, device=data.device)
    if stats.ndim == 1:
        stats = stats.view(*([1] * (data.ndim - 1)), -1)
    return stats


def normalize_data(data, targets, mean=None, std=None):
    """Z-score 归一化；验证集和测试集必须复用训练集统计量。"""
    if mean is None or std is None:
        mean = data.mean(dim=(0, 1, 2), keepdim=True)
        std = data.std(dim=(0, 1, 2), keepdim=True) + 1e-8
    else:
        mean = _feature_stats_view(mean, data)
        std = _feature_stats_view(std, data)
    
    data_norm = (data - mean) / std
    targets_norm = (targets - mean) / std
    
    return data_norm, targets_norm, mean.squeeze(), std.squeeze()


def denormalize_data(data, mean, std):
    """恢复物理量纲，供物理方程损失使用。"""
    mean = _feature_stats_view(mean, data)
    std = _feature_stats_view(std, data)
    return data * std + mean


def train_pinn(model, physics_loss, train_data, train_targets, optimizer, device, epoch,
               feature_mean, feature_std):
    """训练PINN一个epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    batch_size = 16
    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i+batch_size].to(device)
        batch_targets = train_targets[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        
        predictions = model(batch_data)
        
        # 数据损失
        data_loss = F.mse_loss(predictions, batch_targets)
        
        # 物理方程必须在米、m³/s、kg/m³ 等真实量纲下计算。
        physical_predictions = denormalize_data(predictions, feature_mean, feature_std)
        physical_targets = denormalize_data(batch_targets, feature_mean, feature_std)

        physics_loss_fn = physics_loss.to(device)
        physics_loss_fn.set_epoch(epoch)
        
        with torch.enable_grad():
            total_loss_batch, loss_dict = physics_loss_fn(
                physical_predictions, physical_targets, data_loss, model,
                dt=3600.0, dx=1000.0
            )
        
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        n_batches += 1
    
    return total_loss / n_batches


def train_baseline(model, train_data, train_targets, optimizer, device):
    """训练Baseline一个epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    batch_size = 16
    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i+batch_size].to(device)
        batch_targets = train_targets[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        
        predictions = model(batch_data)
        # Baseline 也直接拟合每个站点，与 PINN 使用完全相同的目标。
        loss = F.mse_loss(predictions, batch_targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, val_data, val_targets, device, is_baseline=False):
    """验证模型"""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(val_data), 16):
            batch_data = val_data[i:i+16].to(device)
            batch_targets = val_targets[i:i+16].to(device)
            
            predictions = model(batch_data)
            
            loss = F.mse_loss(predictions, batch_targets)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else float('inf')


def main(epochs=50):
    print("="*70)
    print("🌊 双模型训练: PINN vs Baseline")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 生成数据
    print("\n[1] 生成训练数据...")
    train_data, train_targets = generate_training_data(n_samples=500)
    val_data, val_targets = generate_training_data(n_samples=100)
    
    # 归一化
    train_data, train_targets, data_mean, data_std = normalize_data(train_data, train_targets)
    val_data, val_targets, _, _ = normalize_data(
        val_data, val_targets, data_mean, data_std
    )
    
    print(f"    训练集: {train_data.shape}")
    print(f"    验证集: {val_data.shape}")
    
    # 创建PINN模型
    print("\n[2] 创建 PINN 模型...")
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
    pinn_params = sum(p.numel() for p in pinn_model.parameters())
    print(f"    参数量: {pinn_params:,}")
    
    # 物理损失 - 完整物理约束（真实数据推荐）
    physics_loss = EnhancedPhysicsConstraintLoss(
        manning_coefficient=0.03,
        bed_slope=0.0001,
        gravity=9.81,
        river_width=500.0,
        use_ntk_adaptive=True,       # ✅ NTK自适应权重
        use_curriculum=True,         # ✅ 课程学习
        use_high_order=True,
        use_normalization=True,
        use_sediment=True,           # ✅ Exner泥沙输运方程
        total_epochs=50
    )
    
    # 创建Baseline模型
    print("\n[3] 创建 Baseline LSTM 模型...")
    baseline_model = BaselineLSTM(
        input_size=6,
        hidden_size=128,
        num_layers=2,
        output_size=6,
        prediction_horizon=6,
        dropout=0.1
    ).to(device)
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"    参数量: {baseline_params:,}")
    
    # 优化器
    pinn_optimizer = torch.optim.Adam(pinn_model.parameters(), lr=0.001)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
    
    # 训练
    print("\n[4] 开始训练...")
    print("-"*70)
    
    best_pinn_loss = float('inf')
    best_baseline_loss = float('inf')
    
    pinn_history = {'train': [], 'val': []}
    baseline_history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        # 训练PINN
        pinn_train_loss = train_pinn(
            pinn_model, physics_loss, train_data, train_targets, pinn_optimizer,
            device, epoch, data_mean, data_std
        )
        pinn_val_loss = validate(pinn_model, val_data, val_targets, device, is_baseline=False)
        
        # 训练Baseline
        baseline_train_loss = train_baseline(baseline_model, train_data, train_targets, baseline_optimizer, device)
        baseline_val_loss = validate(baseline_model, val_data, val_targets, device, is_baseline=True)
        
        pinn_history['train'].append(pinn_train_loss)
        pinn_history['val'].append(pinn_val_loss)
        baseline_history['train'].append(baseline_train_loss)
        baseline_history['val'].append(baseline_val_loss)
        
        # 保存最佳模型
        if pinn_val_loss < best_pinn_loss:
            best_pinn_loss = pinn_val_loss
            torch.save({
                'protocol_version': 2,
                'epoch': epoch,
                'model_state_dict': pinn_model.state_dict(),
                'optimizer_state_dict': pinn_optimizer.state_dict(),
                'val_loss': best_pinn_loss,
                'config': pinn_config,
                'mean': data_mean,
                'std': data_std
            }, 'experiments/models/pinn_best.pth')
        
        if baseline_val_loss < best_baseline_loss:
            best_baseline_loss = baseline_val_loss
            torch.save({
                'protocol_version': 2,
                'epoch': epoch,
                'model_state_dict': baseline_model.state_dict(),
                'optimizer_state_dict': baseline_optimizer.state_dict(),
                'val_loss': best_baseline_loss,
                'mean': data_mean,
                'std': data_std
            }, 'experiments/models/baseline_best.pth')
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:2d} | PINN Train: {pinn_train_loss:.4f} Val: {pinn_val_loss:.4f} | "
                  f"Baseline Train: {baseline_train_loss:.4f} Val: {baseline_val_loss:.4f}")
    
    print("-"*70)
    
    # 最终评估
    print("\n[5] 最终评估...")
    
    # 加载最佳权重
    pinn_checkpoint = torch.load('experiments/models/pinn_best.pth', map_location=device)
    baseline_checkpoint = torch.load('experiments/models/baseline_best.pth', map_location=device)
    
    pinn_model.load_state_dict(pinn_checkpoint['model_state_dict'])
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    
    # 测试数据
    test_data, test_targets = generate_training_data(n_samples=50)
    test_data, test_targets, _, _ = normalize_data(
        test_data, test_targets, data_mean, data_std
    )
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)
    
    with torch.no_grad():
        pinn_pred = pinn_model(test_data)
        baseline_pred = baseline_model(test_data)
        
        pinn_mse = F.mse_loss(pinn_pred, test_targets).item()
        baseline_mse = F.mse_loss(baseline_pred, test_targets).item()
        
        pinn_mae = torch.mean(torch.abs(pinn_pred - test_targets)).item()
        baseline_mae = torch.mean(torch.abs(baseline_pred - test_targets)).item()
    
    print(f"\n    📊 PINN (STGNN+物理约束):")
    print(f"      MSE: {pinn_mse:.4f}")
    print(f"      MAE: {pinn_mae:.4f}")
    
    print(f"\n    📊 Baseline (LSTM):")
    print(f"      MSE: {baseline_mse:.4f}")
    print(f"      MAE: {baseline_mae:.4f}")
    
    improvement = (baseline_mae - pinn_mae) / baseline_mae * 100
    print(f"\n    🎯 PINN 相对于 Baseline 提升: {improvement:.1f}%")
    
    # 保存训练历史
    history = {
        'protocol_version': 2,
        'pinn': pinn_history,
        'baseline': baseline_history,
        'final_metrics': {
            'pinn_mse': pinn_mse,
            'pinn_mae': pinn_mae,
            'baseline_mse': baseline_mse,
            'baseline_mae': baseline_mae,
            'improvement_percent': improvement
        }
    }
    
    with open('experiments/models/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n[6] 保存结果...")
    print(f"    PINN最佳模型: experiments/models/pinn_best.pth")
    print(f"    Baseline最佳模型: experiments/models/baseline_best.pth")
    print(f"    训练历史: experiments/models/training_history.json")
    
    print("\n" + "="*70)
    print("✅ 双模型训练完成!")
    print("="*70)


if __name__ == '__main__':
    main()
