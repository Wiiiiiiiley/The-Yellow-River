# 使用真实黄河水文数据训练

## 当前问题

合成数据的问题：
- ❌ 随机噪声主导，无真实物理规律
- ❌ PINN 物理约束成为负担而非帮助
- ❌ 无法体现物理约束在极端洪水场景的优势

## 真实数据优势

真实黄河水文数据的特点：
- ✅ 水位-流量关系遵循曼宁公式
- ✅ 上下游站点有传播延迟（2-4小时）
- ✅ 洪水过程有物理规律（上涨-峰值-消退）
- ✅ 泥沙输运符合 Exner 方程
- ✅ PINN 物理约束能有效抑制过拟合

---

## 数据准备

### 1. 数据格式要求

CSV 文件格式（每小时一条记录）：

```csv
timestamp,station_id,water_level,discharge,sediment,temperature,rainfall
2024-07-15 00:00,兰州,1050.5,1500.2,12.3,22.5,0.0
2024-07-15 00:00,河口镇,1048.3,1480.1,11.8,22.3,0.0
...
```

### 2. 数据存放位置

```
data/
├── yellow_river_hydrology.csv    # 主数据文件
├── stations.csv                   # 站点信息
└── README.md
```

### 3. 数据加载示例

编辑 `scripts/train_both_models.py`，替换 `generate_training_data`：

```python
def load_real_data(csv_path: str, seq_len: int = 24, n_nodes: int = 5):
    """加载真实黄河水文数据"""
    import pandas as pd
    
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    
    # 站点映射
    stations = ['兰州', '河口镇', '龙门', '三门峡', '花园口']
    station_map = {s: i for i, s in enumerate(stations)}
    
    # 构建时空序列数据
    data = []
    targets = []
    
    # 按时间窗口切分
    for i in range(0, len(df) - seq_len - 6, 6):  # 步长6小时
        window = df.iloc[i:i+seq_len]
        target_window = df.iloc[i+seq_len:i+seq_len+6]
        
        # 构建 (time, nodes, features) 数组
        seq_data = np.zeros((seq_len, n_nodes, 6))
        for t, (_, row) in enumerate(window.iterrows()):
            node_idx = station_map[row['station_id']]
            seq_data[t, node_idx] = [
                row['water_level'],
                row['discharge'],
                row['sediment'],
                row['temperature'],
                row['rainfall'],
                row.get('other', 0)
            ]
        
        # 目标同理
        target_data = np.zeros((6, n_nodes, 6))
        for t, (_, row) in enumerate(target_window.iterrows()):
            node_idx = station_map[row['station_id']]
            target_data[t, node_idx] = [
                row['water_level'],
                row['discharge'],
                row['sediment'],
                row['temperature'],
                row['rainfall'],
                row.get('other', 0)
            ]
        
        data.append(seq_data)
        targets.append(target_data)
    
    return torch.FloatTensor(data), torch.FloatTensor(targets)
```

---

## 真实数据训练步骤

### 1. 启用完整物理约束

```python
physics_loss = EnhancedPhysicsConstraintLoss(
    manning_coefficient=0.03,
    bed_slope=0.0001,
    gravity=9.81,
    river_width=500.0,
    use_ntk_adaptive=True,      # 开启NTK自适应
    use_curriculum=True,        # 开启课程学习
    use_high_order=True,
    use_normalization=True,
    use_sediment=True,          # 开启泥沙输运（真实数据有效）
    total_epochs=100
)
```

### 2. 训练命令

```bash
cd /Users/winniemei/CascadeProjects/The-Yellow-River_副本

# 准备数据
cp your_data.csv data/yellow_river_hydrology.csv

# 修改 train_both_models.py 中的数据加载函数
# 然后运行
python3 scripts/train_both_models.py
```

### 3. 预期结果

真实数据上 PINN 应表现更好：

| 场景 | PINN 优势 |
|------|-----------|
| 常规洪水 | 物理约束防止过拟合 |
| 极端洪水 | 连续性/动量方程保证稳定性 |
| 超长期预测 | 物理规律外推能力强 |
| 数据缺失 | 物理约束填补空白 |

---

## 无真实数据时的替代方案

如果暂时没有真实数据，可以：

### A. 改进合成数据（半物理仿真）

编辑 `generate_training_data()`，添加物理规律：

```python
def generate_physics_based_data(n_samples=500):
    """基于圣维南方程的合成数据"""
    
    for _ in range(n_samples):
        # 1. 上游边界条件（流量过程线）
        Q_upstream = generate_flood_hydrograph()
        
        # 2. 使用运动波方程传播
        for node in range(5):
            delay = node * 2  # 2小时传播延迟
            attenuation = 0.9 ** node  # 衰减
            
            # 水位从流量计算（曼宁公式）
            h_node = calculate_from_manning(Q_upstream, delay, attenuation)
            
            # 泥沙从流量计算（MPM公式）
            sediment_node = calculate_from_mpm(Q_upstream, h_node)
        
        # 这样生成的数据有物理一致性
```

### B. 使用项目自带数据适配器

项目支持从 SQLite/CSV/API 加载：

```python
from src.data.data_adapter import SQLiteSource, CSVSource

# SQLite 数据源
db = SQLiteSource('data/hydrology.db')
df = db.get_readings('兰州', hours=168)  # 获取7天数据

# CSV 数据源
csv = CSVSource('data/yellow_river.csv')
df = csv.get_readings('花园口', hours=168)
```

---

## 总结建议

| 情况 | 建议 |
|------|------|
| 有真实黄河数据 | 使用方案3，启用完整物理约束 |
| 无真实数据 | 先用方案1+2（已修改），或创建半物理合成数据 |
| 快速验证 | 当前合成数据 + 关闭物理约束 |
| 生产部署 | 必须用真实数据训练 |

当前已修改：
- ✅ Baseline 架构公平（真正多节点）
- ✅ PINN 物理约束降级（合成数据适配）
- ⏳ 真实数据训练（需要数据文件）
