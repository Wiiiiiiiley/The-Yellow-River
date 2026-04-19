# 快速开始指南

## 环境设置

### 1. 克隆项目
```bash
git clone https://github.com/Wiiiiiiiley/The-Yellow-River.git
cd The-Yellow-River
```

### 2. 创建虚拟环境
```bash
python -m venv .venv

# 激活虚拟环境
# macOS/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt

# 或使用 setup.py
pip install -e .
```

## 快速启动

### 方式 1: 使用启动脚本（推荐）

```bash
chmod +x run.sh  # macOS/Linux
./run.sh
```

然后按照提示选择运行模式。

### 方式 2: 直接运行

#### 启动 Web 仪表板 v2（推荐）
```bash
streamlit run smart_flood_system/app/dashboard_v2.py --server.port 8501
```

打开浏览器访问：http://localhost:8501

#### 启动 CLI 监控器
```bash
python smart_flood_system/monitor.py
```

#### 启动完整系统
```bash
python smart_flood_system/main_system.py --mode full
```

#### 启动 AutoPilot 自主系统
```bash
python smart_flood_system/auto_pilot.py --mode full
```

#### 训练模式
```bash
python smart_flood_system/main_system.py --mode train
```

## 主要特性

### 1. 🌊 GNN-Transformer 时空预测
- 图神经网络建模河流拓扑
- Transformer 层捕捉长程时间依赖
- 24 小时输入窗口，6 小时预测范围

### 2. 🔬 PINN 物理约束
- Saint-Venant 方程约束
- 自适应加权物理损失
- 物理残差 < 0.1 的目标

### 3. 🤖 强化学习控制
- PPO 算法学习最优泄洪策略
- 19 维状态空间，5 维动作空间
- 实时决策建议

### 4. 🤖 AutoPilot 自主系统
- 每日凌晨 2 点自动下载数据
- 每 7 天自动增量训练
- 自动物理参数校准
- 每月回溯验证防止遗忘

### 5. 📊 多源数据适配
- SQLite 实时数据
- CSV 历史数据
- API 远程数据
- 模拟数据回退

### 6. 💻 Dashboard v2
- 实时水文预测
- 险情识别和多模态分析
- RL 决策建议
- 进化轨迹可视化

## 项目结构

```
smart_flood_system/
├── main_system.py           # 核心系统
├── auto_pilot.py            # 自主系统
├── data_adapter.py          # 数据适配器
├── monitor.py               # CLI 监控器
├── models/
│   ├── graph_transformer/   # GNN-Transformer 模型
│   ├── pinn_loss/          # PINN 物理约束
│   └── rl_agent/           # RL PPO 控制器
├── app/
│   └── dashboard_v2.py     # Web 仪表板
└── configs/
    ├── system_config.yaml  # 系统配置
    ├── evolution_config.yaml
    └── auto_pilot.yaml     # AutoPilot 配置
```

## 配置文件

### system_config.yaml
系统总配置，包括：
- 数据配置（5 个站点信息）
- 模型配置（GNN、PINN、RL 参数）
- 训练配置
- 预测配置
- 仪表板配置

### evolution_config.yaml
进化引擎配置：
- 错误监控
- 困难样本挖掘
- 物理校准参数
- 增量学习配置
- A/B 测试配置

### auto_pilot.yaml
AutoPilot 配置：
- 调度时间表
- 自动下载器配置
- 自动训练器配置
- 自我进化配置
- 回溯验证器配置

## 常见问题

### Q: 首次运行时没有数据怎么办？
A: 系统会自动使用模拟数据。模拟数据遵循黄河"大水大沙"特征，包括：
- 泥沙浓度与流量的正相关关系
- 泥沙滞后于流量峰值
- 季节变化特征

### Q: 如何使用真实数据？
A: 
1. 将 CSV 数据放在 `data/archive/` 目录
2. 或在 `data_adapter.py` 中配置 API 端点
3. SQLite 数据会自动保存在 `data/autopilot.db`

### Q: 如何手动触发进化？
A: 运行以下命令：
```bash
python smart_flood_system/auto_pilot.py --mode trigger-evolution
```

### Q: DriveTrainError 或其他依赖错误？
A: 可能是依赖版本冲突。尝试：
```bash
pip install --upgrade torch torch-geometric
```

### Q: 如何查看日志？
A: 
```bash
# CLI 监控器中选择"显示日志"
python smart_flood_system/monitor.py

# 或直接查看文件
tail -f logs/autopilot.log
```

## 数据维度说明

### 输入特征（6 维）
- [0] 水位 (m)
- [1] 流量 (m³/s)
- [2] 含沙量 (kg/m³)
- [3] 温度 (°C)
- [4] 降雨 (mm)
- [5] 时间特征

### 预测范围
- 输入窗口：24 小时
- 输出窗口：6 小时
- 预测范围：未来 6 小时水文过程

## 社群和支持

- GitHub Issues: 报告 bug 和功能请求
- Discussions: 讨论和交流
- Wiki: 详细文档和使用案例

## 许可证

MIT License - 详见 LICENSE 文件

## 致谢

感谢黄河水利委员会的技术支持和域名专业知识！

---

**祝你使用愉快！** 🚀
