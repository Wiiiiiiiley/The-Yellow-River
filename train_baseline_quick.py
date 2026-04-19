#!/usr/bin/env python3
"""快速训练 Baseline 7D 模型"""

import torch
import numpy as np
import pandas as pd
import os

print("="*60)
print("快速训练 Baseline 7D (30 epochs)")
print("="*60)

# 加载数据
df = pd.read_csv('data/processed/train_normalized.csv')
X = df[['time_sin', 'time_cos', 'time_linear', 'location', 'flow_rate', 'rainfall', 'temperature']].values
y = df[['water_level', 'sediment_content']].values

# 分割数据 (80/20)
n_samples = len(X)
n_train = int(0.8 * n_samples)

x_train = torch.FloatTensor(X[:n_train])
y_train = torch.FloatTensor(y[:n_train])
x_val = torch.FloatTensor(X[n_train:])
y_val = torch.FloatTensor(y[n_train:])

print(f"数据: train={n_train}, val={len(x_val)}")

# Baseline 模型 (7D, 256 hidden)
class BaselineModel(torch.nn.Module):
    def __init__(self, input_dim=7, hidden_dim=256, output_dim=2):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

model = BaselineModel(input_dim=7, hidden_dim=256, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
best_val_loss = float('inf')
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    
    y_pred = model(x_train)
    loss = torch.nn.functional.mse_loss(y_pred, y_train)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # 验证
    model.eval()
    with torch.no_grad():
        y_val_pred = model(x_val)
        val_loss = torch.nn.functional.mse_loss(y_val_pred, y_val).item()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}: Train Loss={loss.item():.4f}, Val Loss={val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 保存模型
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'epoch': epoch
        }, 'models/model_baseline_7d_quick.pth')

print(f"\n✓ Baseline 训练完成！最佳验证损失: {best_val_loss:.4f}")
print(f"✓ 模型保存到: models/model_baseline_7d_quick.pth")

# 快速测试
model.eval()
with torch.no_grad():
    y_test = model(torch.FloatTensor(X[:365])).numpy()

print(f"\n预测水位: [{y_test[:,0].min():.3f}, {y_test[:,0].max():.3f}], 波动={y_test[:,0].max()-y_test[:,0].min():.3f}")
print(f"真实水位: [{y[:365,0].min():.3f}, {y[:365,0].max():.3f}], 波动={y[:365,0].max()-y[:365,0].min():.3f}")

if y_test[:,0].max() - y_test[:,0].min() > 0.5:
    print("\n✅ Baseline 学到了周期性！")
else:
    print("\n⚠️ Baseline 预测波动较小")

print("="*60)
