# Baseline 模型集成说明

## 📌 概述

本文档说明黄河智慧防汛系统 v2.0 中新增的 **Baseline 模型验证框架**。

## 🎯 目标

通过将 PINN（物理约束神经网络）与 Baseline（纯数据驱动模型）进行对比验证，
实现：
- ✅ 模型性能的客观评估
- ✅ 物理约束贡献的量化
- ✅ 极端情况预测的可靠性保证
- ✅ [自适应置信度评分

## 📦 核心组件

### 1. Baseline 模型集合 (`baseline_models.py`)

三种 Baseline 实现：

| 模型 | 架构 | 特点 | 用途 |
|------|------|------|------|
| **LSTM Baseline** | 2层LSTM + 多头注意力 + GRU解码器 | 擅长长期依赖 | 序列预测 |
| **Transformer Baseline** | Transformer编码器-解码器 | 并行计算高效 | 并行处理 |
| **Ensemble Baseline** | LSTM + Transformer（学习权重） | 结合两者优势 | 综合预测 |

**关键特性**：
- 支持 4D 输入 `(batch, time, nodes, features)`
- 自动形状转换匹配 PINN 输出
- 可学习的融合权重

### 2. 多模型验证框架 (`ensemble_verifier.py`)

#### MultiModelValidator
对比 PINN vs Baseline 预测：

```
指标计算:
├─ MSE Loss → 均方误差对比
├─ MAE Loss → 平均绝对误差
├─ Cosine Similarity → 向量相似度
├─ Pearson Correlation → 统计相关性
└─ Divergence Detection → 发散检测 (>15%)
```

#### EnsembleValidator
多模型投票与不确定性量化：

```
投票流程:
├─ 收集多个模型预测
├─ 加权平均融合
├─ 计算预测不确定性
└─ 置信度评分
```

## 🔧 使用方法

### 方法 1: 集成预测 (推荐)

```python
from smart_flood_system.main_system import SmartFloodSystem

# 初始化系统
system = SmartFloodSystem()

# 执行预测（自动启用 Baseline 对比）
predictions = system.predict(
    station_id='HYD_001',
    sequence_length=24,
    with_baseline_validation=True  # 启用对比
)

# 检查对比结果
if 'model_comparison' in predictions:
    comp = predictions['model_comparison']
    
    # 检查协议度
    agreement = predictions['model_agreement_score']
    if agreement > 0.85:
        use_pinn = True  # 使用 PINN 预测
    else:
        print("⚠️ 警告: 模型协议度低！")
        
    # 检查发散
    if comp.get('has_divergence'):
        print("⚠️ 警告: 检测到模型发散！")
```

### 方法 2: 直接对比

```python
from smart_flood_system.utils.ensemble_verifier import MultiModelValidator
from smart_flood_system.models.baseline.baseline_models import create_baseline_model
import torch

# 创建模型
pinn_model = create_baseline_model('transformer')
baseline_model = create_baseline_model('lstm')

# 创建验证器
validator = MultiModelValidator(
    pinn_model=pinn_model,
    baseline_model=baseline_model
)

# 获取预测并对比
pinn_pred = pinn_model(inputs)
baseline_pred = baseline_model(inputs)

# 对比分析
comparison = validator.compare_predictions(inputs)
divergence = validator.detect_divergence()
```

### 方法 3: 双模型训练

```python
# 同时训练 PINN 和 Baseline
system.train(
    data_path='data/training.csv',
    epochs=100,
    batch_size=32,
    train_baseline=True  # 启用 Baseline 训练
)

# 保存两个模型
system.save_models('models/latest/')
```

## 📊 输出格式

### 预测输出 (with_baseline_validation=True)

```python
predictions = {
    'model_type': 'PINN+GNN-Transformer',
    'pinn_prediction': {
        'water_level': [1001.5, 1002.3, ...],  # 6小时预测
        'discharge': [1523.4, 1534.2, ...],
        ...
    },
    'baseline_prediction': {
        'water_level': [1000.8, 1001.9, ...],
        'discharge': [1520.1, 1530.5, ...],
        ...
    },
    'model_comparison': {
        'mean_abs_difference': 1.234,
        'max_abs_difference': 5.678,
        'cosine_similarity': 0.945,
        'correlation': 0.932,
        'has_divergence': False,
        'divergence_info': {...}
    },
    'confidence_level': 'high',          # 置信度等级
    'model_agreement_score': 0.945,     # 协议度 0-1
    'timestamp': '2024-XX-XX HXX:XX:XX',
    'station_id': 'HYD_001'
}
```

### 置信度等级映射

```
协议度 (model_agreement_score)
├─ > 0.95  → 'very_high'  (🟢 非常高)
├─ > 0.85  → 'high'       (🟢 高)
├─ > 0.70  → 'medium'     (🟡 中等)
└─ ≤ 0.70  → 'low'        (🔴 低)
```

## 📈 Dashboard 集成

新增"模型对比"标签页实时显示：

1. **对比指标**
   - 模型协议度
   - 置信度等级
   - 平均/最大差异
   - 发散性检测

2. **预测图表**
   - 水位对比曲线
   - 流量对比柱状图

3. **性能指标表**
   - 两个模型的对比表格
   - 各自的优缺点

4. **使用建议**
   - 何时使用哪个模型
   - 如何处理发散情况

## 🎓 原理说明

### 为什么需要 Baseline 对比？

```
┌─────────────────────────────────┐
│ PINN 模型的双重属性             │
├─────────────────────────────────┤
│ 优势:                           │
│  • 物理一致性 (Saint-Venant)    │
│  • 极端情况稳定性                │
│  • 可解释性强                    │
├─────────────────────────────────┤
│ 风险:                           │
│  • 物理参数可能不准(本地变异)   │
│  • 极端条件可能缺少约束         │
│  • 计算成本较高                  │
└─────────────────────────────────┘

Baseline 提供：
✓ 纯数据驱动的"实证对标"
✓ 模型约束贡献的量化
✓ 预测可靠性的客观评估
```

### 模型发散的常见原因

| 原因 | 特征 | 处理方案 |
|------|------|---------|
| 数据质量差 | NSE 极低，噪声大 | 数据清洗 → 重训 |
| 极端水文 | PINN > Baseline | 优先 PINN |
| 季节变化 | 春夏协议度差 | 按季节分模型 |
| 模型过拟合 | 内部验证好，预测差 | 增加正则化 |

## 🔍 诊断与调试

### 检查点清单

```python
# 1. 验证模型初始化
system = SmartFloodSystem()
assert system.gnn_model is not None, "PINN 初始化失败"
assert system.baseline_model is not None, "Baseline 初始化失败"

# 2. 验证预测形状
pred = system.predict('HYD_001', with_baseline_validation=True)
assert 'pinn_prediction' in pred
assert 'baseline_prediction' in pred
assert 'model_comparison' in pred

# 3. 检查对比指标
comp = pred['model_comparison']
assert 0 <= pred['model_agreement_score'] <= 1
assert comp['mean_abs_difference'] >= 0
assert comp['cosine_similarity'] <= 1

# 4. 发散警告处理
if comp['has_divergence']:
    print("⚠️ 模型发散！原因:", comp['divergence_info']['reason'])
    
# 5. 置信度判断
if pred['confidence_level'] in ['very_high', 'high']:
    use_prediction = True
else:
    check_data_quality()
```

## 📋 配置参数

### models.rl (影响 RL 控制层)
```yaml
models:
  rl:
    algorithm: "PPO"
    state_dim: 19
    action_dim: 5
    learning_rate: 0.0003  # 自动类型转换为 float
```

### training (影响双模型训练)
```yaml
training:
  epochs: 1000
  batch_size: 32
  learning_rate: 0.001  # Baseline & PINN 均使用此 LR
```

## 🚀 最佳实践

### DO ✅
- ✅ 在所有生产预测中启用 Baseline 对比
- ✅ 监控 `model_agreement_score` 趋势
- ✅ 定期重训（周期 7 天）来更新 Baseline
- ✅ 极端事件时优先参考 PINN

### DON'T ❌
- ❌ 盲目相信任何单一模型
- ❌ 忽视 `has_divergence` 警告
- ❌ 在发散时切换到 Baseline
- ❌ 假设两个模型总是一致的

## 📞 常见问题

**Q: Baseline 模型训练需要多久？**  
A: 与 PINN 相当，约 100 epoch 需 5-10 分钟

**Q: 为什么协议度随时间变化？**  
A: 数据分布变化、季节变化、极端事件

**Q: 应该使用哪个模型的预测？**  
A: 协议度 > 0.85 时组合使用，否则优先 PINN

**Q: 如何处理一直发散的情况？**  
A: 检查数据源、增加训练数据、调整物理参数

## 📚 参考资源

- 主文档: [README.md](README.md)
- 快速入门: [BASELINE_QUICKSTART.py](BASELINE_QUICKSTART.py)
- 完整报告: [BASELINE_INTEGRATION_REPORT.md](BASELINE_INTEGRATION_REPORT.md)
- 测试代码: [test_baseline_integration.py](test_baseline_integration.py)

## 📝 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v2.1.0 | 2024 | 初始 Baseline 集成 |
| v2.0.0 | 2024 | PINN + GNN-Transformer |

---

**更新时间**: 2024  
**维护者**: 黄河智慧防汛项目组
