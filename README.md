# 智慧黄河防汛 PINN v3.0

**Physics-Informed Neural Network for Yellow River Flood Prediction** —— 基于物理信息神经网络的黄河洪水预测系统，对比验证物理约束对预测精度的提升。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🌊 核心功能

**PINN vs Baseline 双模型对比系统**

```
┌─────────────────────────────────────────────────────────────────┐
│                  黄河防汛 PINN v3.0                              │
├─────────────────────────────────────────────────────────────────┤
│  📊 双模型对比训练                                               │
│  ├── PINN (STGNN + 物理约束)        68.5万参数                   │
│  │   ├── 圣维南方程（连续性）                                     │
│  │   ├── 动量方程（曼宁摩擦）                                     │
│  │   ├── Exner泥沙输运方程                                       │
│  │   └── NTK自适应权重 + 课程学习                                │
│  │                                                               │
│  └── Baseline (纯LSTM)              33.6万参数                   │
│      └── 纯数据驱动，无物理约束                                   │
├─────────────────────────────────────────────────────────────────┤
│  📈 评估对比维度                                                 │
│  ├── 水位预测精度 (R²)                                          │
│  ├── 流量预测精度 (MSE/MAE)                                     │
│  ├── 泥沙浓度预测                                               │
│  └── 物理一致性验证                                             │
├─────────────────────────────────────────────────────────────────┤
│  🎯 研究目标                                                   │
│  └── 验证物理约束在极端洪水场景下的稳定性优势                      │
└─────────────────────────────────────────────────────────────────┘
```

### 当前状态
- ✅ **架构修复**: Baseline LSTM 与 PINN 公平对比（均处理5节点）
- ✅ **中文支持**: 可视化图表正确显示中文
- ✅ **归一化修复**: 自动输入归一化/输出反归一化
- ⚠️ **数据限制**: 当前使用合成数据（需真实黄河数据体现PINN优势）

---

## 📁 项目结构

```
The-Yellow-River/
├── scripts/                    # 可执行脚本 ✅ 已整理
│   ├── train_model.py          # PINN完整训练
│   ├── train_both_models.py    # PINN + Baseline 双模型训练 ✅ 新增
│   ├── quick_train.py          # 快速训练（20 epochs）
│   ├── generate_predictions.py # 生成预测（含归一化修复）✅ 已修复
│   ├── visualize_results.py    # 可视化（中文支持）✅ 已修复
│   └── compare_baseline.py   # 模型对比
├── src/                        # 源代码
│   ├── models/
│   │   ├── stgnn_enhanced.py          # STGNN v3.0 模型
│   │   ├── physics_constraints_enhanced.py  # PINN 物理约束
│   │   └── baseline_models.py         # Baseline LSTM ✅ 已修复多节点
│   ├── data/
│   │   └── data_adapter.py            # 数据接入层
│   ├── training/
│   │   └── BASELINE_QUICKSTART.py     # Baseline快速入门
│   └── utils/
├── configs/                    # 配置文件
├── data/                       # 数据目录（待添加真实数据）
├── experiments/                # 实验输出
│   ├── models/                 # 训练好的模型
│   │   ├── pinn_best.pth
│   │   ├── baseline_best.pth
│   │   └── training_history.json
│   ├── predictions/            # 预测结果
│   └── visualizations/         # 可视化图表
├── README_TRAINING.md          # 独立训练指南 ✅ 新增
├── README_REAL_DATA.md         # 真实数据训练指南 ✅ 新增
└── README.md                   # 本文件
```

---

## 🚀 快速开始

### 安装依赖

```bash
# 进入项目目录
cd /Users/winniemei/CascadeProjects/黄河

# 安装依赖
pip install torch torch-geometric gym numpy pandas pyyaml streamlit plotly

# 生产环境额外依赖
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# AutoPilot 依赖
pip install apscheduler optuna aiohttp
```

### 模型训练（当前可用）

```bash
# 1. 快速训练 PINN（20 epochs，约3-5分钟）
python3 scripts/quick_train.py

# 2. 完整训练 PINN（100 epochs，约15-20分钟）
python3 scripts/train_model.py

# 3. 双模型对比训练（PINN + Baseline，50 epochs）
python3 scripts/train_both_models.py
```

### 生成预测与可视化

```bash
# 4. 生成预测（自动归一化/反归一化）
python3 scripts/generate_predictions.py

# 5. 可视化结果（支持中文显示）
python3 scripts/visualize_results.py

# 6. PINN vs Baseline 对比
python3 scripts/compare_baseline.py
```

### 查看训练指南

```bash
# 详细训练说明
cat README_TRAINING.md

# 真实数据训练说明
cat README_REAL_DATA.md
```

---

## 🔧 项目改进记录

### 近期修复 ✅

| 问题 | 解决方案 | 文件 |
|------|---------|------|
| **Baseline架构不公平** | 修复为真正处理5节点输入/输出 | `src/models/baseline_models.py` |
| **归一化缺失** | 添加输入归一化/输出反归一化 | `scripts/generate_predictions.py` |
| **中文显示方格** | 添加matplotlib中文字体配置 | `scripts/visualize_results.py` |
| **文件结构混乱** | 整理脚本到`scripts/`目录 | - |
| **物理约束权重** | 恢复完整PINN配置(NTK/课程学习/泥沙) | `scripts/train_both_models.py` |

### 已知问题 ⚠️

- **合成数据限制**: 当前使用合成数据，无真实物理规律
- **PINN表现**: 在合成数据上可能不如Baseline
- **解决方案**: 使用真实黄河水文数据（见`README_REAL_DATA.md`）

### 下一步建议

1. **短期**: 使用当前合成数据验证代码流程
2. **中期**: 获取真实黄河水文数据（水位/流量/泥沙）
3. **长期**: 用真实数据重新训练，体现PINN物理约束优势

---

## 🔬 研究价值

### 核心研究贡献

1. **物理信息深度学习用于水文学**
   > "我将圣维南方程（连续性方程、动量方程）作为软约束整合到 STGNN-Transformer 架构中，通过 NTK 自适应权重平衡数据拟合与物理一致性，确保预测在极端外推条件下仍满足质量/动量守恒。"

2. **PINN vs Baseline 公平对比框架**
   > "我设计了严格的对照实验：相同数据、相同架构复杂度下，对比物理约束模型（PINN）与纯数据驱动模型（Baseline LSTM）。关键在于修复 Baseline 的多节点处理能力，确保两者在公平条件下竞争，验证物理约束的真实价值。"

3. **泥沙输运物理建模**
   > "我实现了 Exner 泥沙连续性方程，将泥沙浓度预测与水动力学耦合。黄河独特的"大水大沙"特性要求模型不仅拟合统计关系，更要理解泥沙侵蚀-沉积的物理机制。"

4. **数据归一化与物理一致性**
   > "我解决了 PINN 训练中的关键工程问题：自动归一化输入/反归一化输出，确保物理方程在无量纲空间计算正确，同时预测结果恢复物理量纲（米、m³/s、kg/m³）。"

### 关键发现（合成数据验证）

```
在合成洪水场景下（水位1000-1100m，流量1000-5000m³/s）：
- 水位预测：PINN R² = 0.72，Baseline R² ≈ 0.68（接近）
- 流量/泥沙预测：PINN 受物理约束限制，暂未达到预期优势

结论：合成数据缺乏真实物理规律，无法体现 PINN 在极端洪水下的稳定性优势。
真实黄河数据（含"大水大沙"特性）是验证 PINN 价值的关键。
```

---

## 🎯 当前性能基准（合成数据）

| 指标 | PINN | Baseline | 备注 |
|------|------|----------|------|
| 参数量 | 68.5万 | 33.6万 | PINN 多一倍 |
| 水位 MAE | ~300 | ~270 | 接近 |
| 水位 R² | 0.72 | 0.68 | 略优 |
| 流量 R² | -0.94 | -1.10 | 均较差 |
| 训练时间 | 15min | 8min | 50 epochs |

**待验证**：真实数据下 PINN 物理约束应在极端场景显著优于 Baseline。

---

## 🔧 技术规格

### 输入输出
- **输入**: (batch, 24时间步, 5站点, 6特征)
  - 特征：水位(m)、流量(m³/s)、泥沙(kg/m³)、温度(°C)、降雨(mm)、其他
- **输出**: (batch, 6预测步, 5站点, 6特征)

### PINN 架构
- **STGNN**: 3层 GAT + 自适应图结构
- **Transformer**: 4层，8头注意力
- **物理约束**: 连续性方程 + 动量方程 + Exner泥沙方程
- **自适应**: NTK权重平衡 + 课程学习

### Baseline 架构
- **LSTM**: 2层，128 hidden size
- **注意力**: Multi-head 自注意力
- **输出**: 真正多节点（非复制），与PINN公平对比

### 黄河个性学习 ⭐ 新增
- **季节编码**: 汛期/枯水期周期的 Sin/Cos 编码（年度+季度）
- **特征交叉**: Q×S 显式交互实现"大水大沙"
- **泥沙梯度监控**: 检测洪峰期间的突发模式
- **Exner 动力学**: 带孔隙参数的侵蚀/沉积建模
- **NTK 自适应加权**: 训练期间自动物理权重调整
- **两阶段训练**: Adam（1000 轮）→ L-BFGS（物理残差 < 0.1）

### 验证
- **双模型对比**: PINN vs Baseline LSTM
- **指标**: MSE、MAE、RMSE、R²
- **可视化**: 预测对比图、散点图、误差分布

### 数据支持
- **当前**: 合成洪水数据（用于验证代码流程）
- **待添加**: 真实黄河水文数据（SQLite/CSV/API）
- **站点**: 兰州 → 河口镇 → 龙门 → 三门峡 → 花园口

---

## 📊 可视化输出

运行 `python3 scripts/visualize_results.py` 生成：

1. **water_level_comparison.png** - 水位预测对比（5站点×6时间步）
2. **discharge_comparison.png** - 流量预测对比
3. **sediment_comparison.png** - 泥沙浓度预测对比
4. **scatter_comparison.png** - 预测精度散点图（预测 vs 真实）
5. **error_distribution.png** - 误差分布直方图
6. **time_series_detail.png** - 时间序列详细图

**中文显示已修复**（macOS系统字体配置）

---

## 🛠️ 开发指南

### 添加新功能

**自定义物理约束**
```python
# 在 src/models/physics_constraints_enhanced.py
# 修改 EnhancedPhysicsConstraintLoss 类
```

**调整模型架构**
```python
# 在 src/models/stgnn_enhanced.py
# 修改 GNN 层数、Transformer 头数等
```

**修改训练参数**
```python
# 在 scripts/train_both_models.py
# 调整 epochs、batch_size、学习率等
```

### 测试模型

```bash
# 快速测试模型导入
python3 -c "
from src.models.stgnn_enhanced import create_enhanced_model
from src.models.baseline_models import BaselineLSTM
from src.models.physics_constraints_enhanced import EnhancedPhysicsConstraintLoss
print('✅ 所有模型导入成功')
"
```

---

## 📚 引用

如在你的研究中使用本系统，请引用：

```bibtex
@software{smart_yellow_river_flood_system,
  author = {Yellow River Digital Twin Project},
  title = {Smart Flood Management System with Physics-Informed Deep Learning},
  year = {2024},
  url = {https://github.com/Wiiiiiiiley/The-Yellow-River}
}
```

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🤝 致谢

- 黄河水利委员会提供的领域专业知识
- PyTorch Geometric 团队的 GNN 框架
- Informer 作者的高效注意力机制

---

**联系方式**: 如有技术问题或合作咨询，请提交 issue 或联系开发团队。

**项目状态**: v3.0 可用 - PINN vs Baseline 双模型对比系统
- ✅ 物理信息神经网络 (PINN) 洪水预测
- ✅ Baseline LSTM 对比基线（公平多节点架构）
- ✅ 自动归一化/反归一化数据处理
- ✅ 中文可视化支持
- ⚠️ 待添加：真实黄河水文数据（当前使用合成数据）
