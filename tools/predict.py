#!/usr/bin/env python3
"""加载模型并进行预测"""

import torch
import numpy as np
import pandas as pd
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pinn_model_v2 import YellowRiverPINNv2

print("="*60)
print("PINN v2.0 预测")
print("="*60)

# 加载测试数据
df = pd.read_csv('data/processed/train_normalized.csv')
X = df[['time_sin', 'time_cos', 'time_linear', 'location', 'flow_rate', 'rainfall', 'temperature']].values
y = df[['water_level', 'sediment_content']].values

# 取365天测试
X_test = X[:365]
y_true = y[:365]

print(f"\n测试数据: {len(X_test)} 天")
print(f"真实水位: [{y_true[:,0].min():.3f}, {y_true[:,0].max():.3f}], 波动={y_true[:,0].max()-y_true[:,0].min():.3f}")

# 查找 PINN 模型
pinn_path = 'models/pinn/pinn_v2_quick_final.pt'

if not os.path.exists(pinn_path):
    # 回退到 baseline 或 pinn 版本
    pinn_dirs = glob.glob('models/pinn/*/best_model.pt')
    if pinn_dirs:
        pinn_path = sorted(pinn_dirs)[-1]
    else:
        print("❌ 未找到 PINN 模型!")
        exit(1)
print(f"\n加载模型: {pinn_path}")

checkpoint = torch.load(pinn_path, map_location='cpu', weights_only=False)

# 创建模型
model = YellowRiverPINNv2(input_dim=7, hidden_dim=256, output_dim=2, num_res_blocks=4, use_attention=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 预测
X_tensor = torch.FloatTensor(X_test)
with torch.no_grad():
    y_pred = model(X_tensor).numpy()

print(f"\n预测水位: [{y_pred[:,0].min():.3f}, {y_pred[:,0].max():.3f}], 波动={y_pred[:,0].max()-y_pred[:,0].min():.3f}")
print(f"预测含沙量: [{y_pred[:,1].min():.3f}, {y_pred[:,1].max():.3f}], 波动={y_pred[:,1].max()-y_pred[:,1].min():.3f}")

# 计算 RMSE
rmse_water = np.sqrt(np.mean((y_pred[:,0] - y_true[:,0])**2))
rmse_sediment = np.sqrt(np.mean((y_pred[:,1] - y_true[:,1])**2))
print(f"\nRMSE: 水位={rmse_water:.4f}, 含沙量={rmse_sediment:.4f}")

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

days = range(len(y_true))

# 水位
axes[0].plot(days, y_true[:,0], 'b-', label='True', linewidth=2)
axes[0].plot(days, y_pred[:,0], 'r--', label='Predicted', linewidth=2)
axes[0].set_ylabel('Water Level (normalized)')
axes[0].set_title(f'PINN v2.0 Prediction - Water Level (RMSE: {rmse_water:.4f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 含沙量
axes[1].plot(days, y_true[:,1], 'b-', label='True', linewidth=2)
axes[1].plot(days, y_pred[:,1], 'r--', label='Predicted', linewidth=2)
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Sediment (normalized)')
axes[1].set_title(f'PINN v2.0 Prediction - Sediment (RMSE: {rmse_sediment:.4f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/pinn_v2_prediction.png', dpi=150)
print(f"\n✓ 预测图已保存: results/pinn_v2_prediction.png")

# 评估
water_var = y_pred[:,0].max() - y_pred[:,0].min()
if water_var < 0.3:
    print("\n⚠️  预测波动较小，可能需要检查模型")
else:
    print(f"\n✓ 预测波动正常: {water_var:.3f}")

print("="*60)
