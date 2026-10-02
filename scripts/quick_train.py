#!/usr/bin/env python3
"""快速训练脚本 (20 epochs)"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stgnn_enhanced import create_enhanced_model

print("="*60)
print("🌊 快速训练 PINN v3.0 (20 epochs)")
print("="*60)

device = torch.device('cpu')
print(f"\n设备: {device}")

# 生成小数据集
print("\n[1] 生成训练数据 (100 samples)...")
def generate_data(n_samples=100):
    data = []
    targets = []
    for _ in range(n_samples):
        base_level = 1000 + np.random.rand() * 200
        base_discharge = 1500 + np.random.exponential(500)
        base_sediment = 5 + 0.003 * base_discharge
        
        seq_data = []
        for t in range(24):
            for node in range(5):
                seq_data.append([
                    base_level + np.random.randn() * 10,
                    base_discharge + np.random.randn() * 100,
                    max(0, base_sediment + np.random.randn() * 1),
                    15 + np.random.randn() * 5,
                    max(0, np.random.exponential(2)),
                    np.random.rand()
                ])
        data.append(seq_data)
        
        target = []
        for t in range(6):
            for node in range(5):
                target.append([
                    base_level + np.random.randn() * 15,
                    base_discharge + np.random.randn() * 120,
                    max(0, base_sediment + np.random.randn() * 3),
                    15 + np.random.randn() * 5,
                    max(0, np.random.exponential(2)),
                    np.random.rand()
                ])
        targets.append(target)
    
    data = np.array(data).reshape(n_samples, 24, 5, 6)
    targets = np.array(targets).reshape(n_samples, 6, 5, 6)
    return torch.from_numpy(data).float(), torch.from_numpy(targets).float()

train_data, train_targets = generate_data(100)
val_data, val_targets = generate_data(20)

# 归一化
mean = train_data.mean(dim=(0,1,2), keepdim=True)
std = train_data.std(dim=(0,1,2), keepdim=True) + 1e-8
train_data = (train_data - mean) / std
train_targets = (train_targets - mean) / std
val_data = (val_data - mean) / std
val_targets = (val_targets - mean) / std

print(f"    训练集: {train_data.shape}")
print(f"    验证集: {val_data.shape}")

# 创建模型
print("\n[2] 创建模型...")
model_config = {
    'node_features': 6, 'n_nodes': 5, 'gnn_hidden': 64,
    'gnn_layers': 3, 'transformer_heads': 8, 'transformer_layers': 4,
    'sequence_length': 24, 'prediction_horizon': 6, 'dropout': 0.1,
    'use_physics_attention': True, 'use_adaptive_gnn': True,
    'use_multiscale_temporal': True, 'use_physics_postprocess': True,
    'use_bayesian': False
}

model = create_enhanced_model(model_config).to(device)
print(f"    参数量: {sum(p.numel() for p in model.parameters()):,}")

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
print("\n[3] 开始训练 (20 epochs)...")
print("-" * 60)

best_loss = float('inf')
for epoch in range(20):
    # 训练
    model.train()
    total_loss = 0
    for i in range(0, len(train_data), 8):
        batch_data = train_data[i:i+8].to(device)
        batch_targets = train_targets[i:i+8].to(device)
        
        predictions = model(batch_data)
        loss = nn.functional.mse_loss(predictions, batch_targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / (len(train_data) // 8 + 1)
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_pred = model(val_data.to(device))
        val_loss = nn.functional.mse_loss(val_pred, val_targets.to(device)).item()
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            'protocol_version': 2,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'config': model_config,
            'mean': mean, 'std': std
        }, 'experiments/models/best_model.pth')
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}")

print("-" * 60)
print(f"\n✅ 训练完成!")
print(f"    最佳验证损失: {best_loss:.6f}")
print(f"    模型保存于: experiments/models/best_model.pth")
print("\n" + "="*60)
