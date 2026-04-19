#!/usr/bin/env python3
"""对比 Baseline vs PINN v2.0 模型"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pinn_model import YellowRiverPINNv2

print("="*60)
print("对比 Baseline 7D vs PINN v2.0")
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

# ========== Baseline 模型 ==========
print("\n加载 Baseline 模型...")
checkpoint_base = torch.load('models/model_baseline_7d_quick.pth', map_location='cpu', weights_only=False)

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
    def forward(self, x): return self.network(x)

model_base = BaselineModel(input_dim=7, hidden_dim=256, output_dim=2)
model_base.load_state_dict(checkpoint_base['model_state_dict'])
model_base.eval()

with torch.no_grad():
    y_pred_base = model_base(torch.FloatTensor(X_test)).numpy()

rmse_base_water = np.sqrt(np.mean((y_pred_base[:,0] - y_true[:,0])**2))
rmse_base_sed = np.sqrt(np.mean((y_pred_base[:,1] - y_true[:,1])**2))

print(f"Baseline - 水位: [{y_pred_base[:,0].min():.3f}, {y_pred_base[:,0].max():.3f}], 波动={y_pred_base[:,0].max()-y_pred_base[:,0].min():.3f}")
print(f"Baseline - RMSE: 水位={rmse_base_water:.4f}, 含沙量={rmse_base_sed:.4f}")

# ========== PINN 模型 ==========
print("\n加载 PINN v2.0 模型...")
checkpoint_pinn = torch.load('models/pinn_v2_quick/20260419_final/best_model.pt', map_location='cpu', weights_only=False)
model_pinn = YellowRiverPINNv2(input_dim=7, hidden_dim=256, output_dim=2, num_res_blocks=4, use_attention=True)
model_pinn.load_state_dict(checkpoint_pinn['model_state_dict'])
model_pinn.eval()

with torch.no_grad():
    y_pred_pinn = model_pinn(torch.FloatTensor(X_test)).numpy()

rmse_pinn_water = np.sqrt(np.mean((y_pred_pinn[:,0] - y_true[:,0])**2))
rmse_pinn_sed = np.sqrt(np.mean((y_pred_pinn[:,1] - y_true[:,1])**2))

print(f"PINN - 水位: [{y_pred_pinn[:,0].min():.3f}, {y_pred_pinn[:,0].max():.3f}], 波动={y_pred_pinn[:,0].max()-y_pred_pinn[:,0].min():.3f}")
print(f"PINN - RMSE: 水位={rmse_pinn_water:.4f}, 含沙量={rmse_pinn_sed:.4f}")

# ========== 生成对比图 ==========
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

days = range(len(y_true))

# 水位对比
axes[0,0].plot(days, y_true[:,0], 'k-', label='True', linewidth=2)
axes[0,0].plot(days, y_pred_base[:,0], 'b--', label=f'Baseline (RMSE:{rmse_base_water:.3f})', linewidth=2)
axes[0,0].plot(days, y_pred_pinn[:,0], 'r--', label=f'PINN (RMSE:{rmse_pinn_water:.3f})', linewidth=2)
axes[0,0].set_ylabel('Water Level (normalized)', fontsize=12)
axes[0,0].set_title('Water Level Prediction - 7D Models', fontsize=14)
axes[0,0].legend(fontsize=10)
axes[0,0].grid(True, alpha=0.3)

# 含沙量对比
axes[0,1].plot(days, y_true[:,1], 'k-', label='True', linewidth=2)
axes[0,1].plot(days, y_pred_base[:,1], 'b--', label=f'Baseline (RMSE:{rmse_base_sed:.3f})', linewidth=2)
axes[0,1].plot(days, y_pred_pinn[:,1], 'r--', label=f'PINN (RMSE:{rmse_pinn_sed:.3f})', linewidth=2)
axes[0,1].set_ylabel('Sediment (normalized)', fontsize=12)
axes[0,1].set_title('Sediment Prediction - 7D Models', fontsize=14)
axes[0,1].legend(fontsize=10)
axes[0,1].grid(True, alpha=0.3)

# 误差分布
axes[1,0].hist(y_pred_base[:,0] - y_true[:,0], bins=30, alpha=0.5, label='Baseline', color='blue')
axes[1,0].hist(y_pred_pinn[:,0] - y_true[:,0], bins=30, alpha=0.5, label='PINN', color='red')
axes[1,0].set_xlabel('Prediction Error', fontsize=12)
axes[1,0].set_ylabel('Frequency', fontsize=12)
axes[1,0].set_title('Water Level Error Distribution', fontsize=14)
axes[1,0].legend(fontsize=10)
axes[1,0].grid(True, alpha=0.3)

# 散点图对比
axes[1,1].scatter(y_true[:,0], y_pred_base[:,0], alpha=0.5, label='Baseline', color='blue', s=20)
axes[1,1].scatter(y_true[:,0], y_pred_pinn[:,0], alpha=0.5, label='PINN', color='red', s=20)
min_val, max_val = y_true[:,0].min(), y_true[:,0].max()
axes[1,1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect', linewidth=2)
axes[1,1].set_xlabel('True Water Level', fontsize=12)
axes[1,1].set_ylabel('Predicted Water Level', fontsize=12)
axes[1,1].set_title('Prediction vs True', fontsize=14)
axes[1,1].legend(fontsize=10)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/baseline_vs_pinn_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 对比图已保存: results/baseline_vs_pinn_comparison.png")

# 总结
print("\n" + "="*60)
print("对比总结")
print("="*60)
print(f"真实数据: 波动={y_true[:,0].max()-y_true[:,0].min():.3f}")
print(f"Baseline: 波动={y_pred_base[:,0].max()-y_pred_base[:,0].min():.3f}, RMSE={rmse_base_water:.4f}")
print(f"PINN:     波动={y_pred_pinn[:,0].max()-y_pred_pinn[:,0].min():.3f}, RMSE={rmse_pinn_water:.4f}")

if rmse_pinn_water < rmse_base_water:
    print(f"\n✅ PINN v2.0 表现更好 (RMSE 低 {((rmse_base_water-rmse_pinn_water)/rmse_base_water*100):.1f}%)")
else:
    print(f"\n✅ Baseline 表现更好 (RMSE 低 {((rmse_pinn_water-rmse_base_water)/rmse_pinn_water*100):.1f}%)")

print("="*60)
