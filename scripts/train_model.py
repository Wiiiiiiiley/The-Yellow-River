#!/usr/bin/env python3
"""
训练增强版 PINN v3.0 模型
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

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stgnn_enhanced import create_enhanced_model
from src.models.physics_constraints_enhanced import (
    EnhancedPhysicsConstraintLoss, 
    BoundaryConditionEnforcer
)


def generate_training_data(n_samples=500, seq_len=24, n_nodes=5, n_features=6):
    """生成训练数据 - 模拟黄河洪水场景"""
    
    data = []
    targets = []
    
    for _ in range(n_samples):
        # 随机选择洪水场景
        flood_scenario = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        
        if flood_scenario == 'low':
            base_level = 1000 + np.random.rand() * 50
            base_discharge = 1200 + np.random.exponential(200)
        elif flood_scenario == 'medium':
            base_level = 1050 + np.random.rand() * 80
            base_discharge = 2000 + np.random.exponential(400)
        else:  # high
            base_level = 1100 + np.random.rand() * 100
            base_discharge = 3500 + np.random.exponential(600)
        
        base_discharge = min(base_discharge, 5500)
        base_sediment = 5 + 0.004 * base_discharge + np.random.randn() * 3
        
        # 生成序列数据
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
        
        # 生成目标 (未来6小时)
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


def normalize_data(data, targets, data_mean=None, data_std=None):
    """数据归一化"""
    if data_mean is None or data_std is None:
        data_mean = data.mean(dim=(0, 1, 2), keepdim=True)
        data_std = data.std(dim=(0, 1, 2), keepdim=True) + 1e-8
    else:
        data_mean = torch.as_tensor(data_mean, dtype=data.dtype, device=data.device)
        data_std = torch.as_tensor(data_std, dtype=data.dtype, device=data.device)
        if data_mean.ndim == 1:
            data_mean = data_mean.view(1, 1, 1, -1)
            data_std = data_std.view(1, 1, 1, -1)
    
    # 归一化
    data_norm = (data - data_mean) / data_std
    targets_norm = (targets - data_mean) / data_std
    
    return data_norm, targets_norm, data_mean, data_std


def train_epoch(model, physics_loss, train_data, train_targets, optimizer, device, epoch,
                data_mean, data_std):
    """训练一个 epoch"""
    
    model.train()
    total_loss = 0
    n_batches = 0
    
    # 随机批次
    batch_size = 16
    n_samples = len(train_data)
    indices = torch.randperm(n_samples)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_data = train_data[batch_indices].to(device)
        batch_targets = train_targets[batch_indices].to(device)

        optimizer.zero_grad()
        
        # 前向传播
        predictions = model(batch_data)
        
        # 数据损失
        data_loss = F.mse_loss(predictions, batch_targets)
        
        mean = data_mean.to(device)
        std = data_std.to(device)
        physical_predictions = predictions * std + mean
        physical_targets = batch_targets * std + mean
        physics_loss.set_epoch(epoch)
        loss, loss_dict = physics_loss(
            physical_predictions,
            physical_targets,
            data_loss,
            model=model,
            dt=3600.0,
            dx=1000.0,
        )
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, val_data, val_targets, device):
    """验证模型"""
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for i in range(0, len(val_data), 16):
            batch_data = val_data[i:i+16].to(device)
            batch_targets = val_targets[i:i+16].to(device)
            
            predictions = model(batch_data)
            loss = F.mse_loss(predictions, batch_targets)
            total_loss += loss.item()
    
    return total_loss / (len(val_data) // 16 + 1)


def main():
    print("="*70)
    print("🌊 黄河防汛 PINN v3.0 模型训练")
    print("="*70)
    
    # 设备
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
    
    # 创建模型
    print("\n[2] 创建模型...")
    model_config = {
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
    
    model = create_enhanced_model(model_config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    参数量: {total_params:,}")
    
    # 物理损失
    physics_loss = EnhancedPhysicsConstraintLoss(
        manning_coefficient=0.03,
        bed_slope=0.0001,
        gravity=9.81,
        river_width=500.0,
        use_ntk_adaptive=True,
        use_curriculum=True,
        use_high_order=True,
        use_normalization=True,
        use_sediment=True,
        total_epochs=100
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # 训练
    print("\n[3] 开始训练...")
    print("-" * 70)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(100):
        # 训练
        train_loss = train_epoch(
            model, physics_loss, train_data, train_targets, optimizer, device,
            epoch, data_mean, data_std
        )
        
        # 验证
        val_loss = validate(model, val_data, val_targets, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # 打印进度
        if epoch % 10 == 0 or val_loss < best_val_loss:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # 保存模型
            save_dir = Path("experiments/models")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'protocol_version': 2,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': model_config,
                'mean': data_mean.squeeze(),
                'std': data_std.squeeze(),
            }, save_dir / "best_model.pth")
    
    print("-" * 70)
    print(f"\n✅ 训练完成!")
    print(f"    最佳验证损失: {best_val_loss:.6f}")
    print(f"    模型保存于: experiments/models/best_model.pth")
    
    # 保存训练历史
    Path("experiments/logs").mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv("experiments/logs/training_history.csv", index=False)
    
    # 绘制训练曲线
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History - PINN v3.0')
        plt.legend()
        plt.grid(True)
        plt.savefig("experiments/visualizations/training_curve.png", dpi=300, bbox_inches='tight')
        print(f"    训练曲线: experiments/visualizations/training_curve.png")
    except:
        pass
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
