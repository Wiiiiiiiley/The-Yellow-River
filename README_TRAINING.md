# 黄河防汛 PINN v3.0 - 独立训练指南

## 快速开始

### 1. 训练单个模型（PINN）

```bash
cd /Users/winniemei/CascadeProjects/The-Yellow-River_副本

# 快速训练（20 epochs，约3-5分钟）
python3 scripts/quick_train.py

# 完整训练（50 epochs，约10-15分钟）
python3 scripts/train_model.py
```

### 2. 同时训练两个模型（PINN vs Baseline）

```bash
# 50 epochs，约15-20分钟
python3 scripts/train_both_models.py
```

### 3. 生成预测

```bash
# 使用训练好的模型生成预测
python3 scripts/generate_predictions.py
```

### 4. 可视化结果

```bash
# 生成预测图表（中文已修复）
python3 scripts/visualize_results.py

# PINN vs Baseline 对比
python3 scripts/compare_baseline.py
```

---

## 训练参数调整

编辑脚本文件修改训练参数：

### `scripts/train_model.py` 第246行：
```python
for epoch in range(100):  # 修改这里的数字改变训练轮数
```

### `scripts/train_both_models.py` 第189行：
```python
epochs = 50  # 修改训练轮数
```

### 学习率调整（第168-169行）：
```python
pinn_optimizer = torch.optim.Adam(pinn_model.parameters(), lr=0.001)  # 改这里
baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
```

---

## 输出文件位置

| 文件 | 路径 |
|------|------|
| PINN模型 | `experiments/models/pinn_best.pth` 或 `best_model.pth` |
| Baseline模型 | `experiments/models/baseline_best.pth` |
| 训练历史 | `experiments/models/training_history.json` |
| 预测结果 | `experiments/predictions/predictions_all.csv` |
| 可视化图表 | `experiments/visualizations/*.png` |

---

## 检查训练状态

### 查看模型是否训练过：
```bash
# 检查权重是否接近0（未训练）或有值（已训练）
python3 -c "
import torch
ckpt = torch.load('experiments/models/best_model.pth', map_location='cpu')
state = ckpt['model_state_dict']
w = list(state.values())[0]
print(f'Epoch: {ckpt.get(\"epoch\", \"N/A\")}')
print(f'权重 std: {w.std():.4f} (>0.05表示已训练)')
"
```

### 查看训练指标：
```bash
# 查看JSON格式的训练历史
cat experiments/models/training_history.json
```

---

## 常见问题

### Q: 训练很慢？
A: PINN 物理约束计算复杂，CPU 训练 50 epochs 约需 15-20 分钟。减少 `epochs` 或样本数可加速。

### Q: 中文显示方格？
A: 已修复。`scripts/visualize_results.py` 已设置 macOS 中文字体。

### Q: Baseline 比 PINN 好？
A: 因为使用的是合成数据，没有真实物理规律。真实黄河数据才能体现 PINN 物理约束优势。

### Q: 如何加载训练好的模型？

```python
import torch
from src.models.stgnn_enhanced import create_enhanced_model

# 加载配置
checkpoint = torch.load('experiments/models/best_model.pth', map_location='cpu')
config = checkpoint['config']

# 创建模型并加载权重
model = create_enhanced_model(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 命令速查表

| 任务 | 命令 |
|------|------|
| 快速训练 PINN | `python3 scripts/quick_train.py` |
| 完整训练 PINN | `python3 scripts/train_model.py` |
| 双模型训练 | `python3 scripts/train_both_models.py` |
| 生成预测 | `python3 scripts/generate_predictions.py` |
| 可视化 | `python3 scripts/visualize_results.py` |
| 对比实验 | `python3 scripts/compare_baseline.py` |

---

## 文件结构

```
scripts/
├── quick_train.py           # 快速训练（20 epochs）
├── train_model.py           # 完整训练（100 epochs）
├── train_both_models.py     # 双模型对比训练
├── generate_predictions.py  # 生成预测
├── visualize_results.py     # 结果可视化
└── compare_baseline.py      # PINN vs Baseline 对比
```

---

## 技术细节

- **PINN 模型**: STGNN + Transformer + 物理约束（Exner泥沙方程、连续性方程、动量方程）
- **Baseline**: LSTM + Attention，纯数据驱动
- **输入**: (batch, 24时间步, 5站点, 6特征) - 水位、流量、泥沙、温度、降雨、其他
- **输出**: (batch, 6预测步, 5站点, 6特征)
