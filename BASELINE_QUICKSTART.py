#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
黄河智慧防汛系统 v2.0 - Baseline 模型对比快速入门

演示如何使用新的 PINN vs Baseline 多模型对比验证功能
"""

import sys
from pathlib import Path
import numpy as np
import torch

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from smart_flood_system.main_system import SmartFloodSystem


def example_1_basic_prediction_with_comparison():
    """示例 1: 基本预测包含 Baseline 对比"""
    print("\n" + "="*60)
    print("示例 1: 基本预测与 Baseline 对比")
    print("="*60)
    
    # 初始化系统
    system = SmartFloodSystem()
    print("✅ 系统初始化完成")
    
    # 执行预测（默认启用 Baseline 对比）
    try:
        predictions = system.predict(
            station_id='HYD_001',
            sequence_length=24,
            with_baseline_validation=True
        )
        
        print("\n📊 预测结果摘要:")
        print("-" * 60)
        
        # PINN 预测
        pinn_pred = predictions['pinn_prediction']
        print(f"PINN 模型预测水位: {pinn_pred['water_level'][:3]} ... (前 3 小时)")
        print(f"PINN 模型预测流量: {pinn_pred['discharge'][:3]} ... (前 3 小时)")
        
        # Baseline 预测
        baseline_pred = predictions.get('baseline_prediction', {})
        if baseline_pred:
            print(f"\nBaseline 模型预测水位: {baseline_pred['water_level'][:3]} ... (前 3 小时)")
            print(f"Baseline 模型预测流量: {baseline_pred['discharge'][:3]} ... (前 3 小时)")
        
        # 模型对比
        if 'model_comparison' in predictions:
            comp = predictions['model_comparison']
            print(f"\n对比指标:")
            print(f"  • 模型协议度: {predictions.get('model_agreement_score', 0):.4f}")
            print(f"  • 置信度等级: {predictions.get('confidence_level', '未知')}")
            print(f"  • 平均差异: {comp.get('mean_abs_difference', 0):.6f}")
            print(f"  • 模型发散: {'是 ⚠️' if comp.get('has_divergence') else '否 ✅'}")
        
    except Exception as e:
        print(f"预测失败: {e}")
        import traceback
        traceback.print_exc()


def example_2_direct_model_comparison():
    """示例 2: 直接使用验证框架进行模型对比"""
    print("\n" + "="*60)
    print("示例 2: 直接模型对比")
    print("="*60)
    
    from smart_flood_system.models.baseline.baseline_models import create_baseline_model
    from smart_flood_system.utils.ensemble_verifier import MultiModelValidator
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建两个模型
    print("\n创建模型...")
    pinn_model = create_baseline_model(
        model_type='transformer',
        input_size=6,
        hidden_size=64,
        output_size=6,
        prediction_horizon=6
    ).to(device)
    
    baseline_model = create_baseline_model(
        model_type='lstm',
        input_size=6,
        hidden_size=64,
        output_size=6,
        prediction_horizon=6
    ).to(device)
    
    # 创建验证器
    validator = MultiModelValidator(
        pinn_model=pinn_model,
        baseline_model=baseline_model,
        device=device
    )
    print("✅ 模型和验证器创建完成")
    
    # 生成测试数据
    print("\n生成测试数据...")
    test_input = torch.randn(4, 24, 5, 6).to(device)  # (batch, time, nodes, features)
    ground_truth = torch.randn(4, 6, 5, 6).to(device)
    
    # 执行对比
    print("执行模型对比...")
    with torch.no_grad():
        pinn_pred = pinn_model(test_input)
        baseline_pred = baseline_model(test_input)
    
    # 计算指标
    pinn_mse = torch.nn.functional.mse_loss(pinn_pred, ground_truth).item()
    baseline_mse = torch.nn.functional.mse_loss(baseline_pred, ground_truth).item()
    
    improvement = (1 - pinn_mse / (baseline_mse + 1e-6)) * 100
    
    print(f"\n对比结果:")
    print(f"  • PINN MSE: {pinn_mse:.6f}")
    print(f"  • Baseline MSE: {baseline_mse:.6f}")
    print(f"  • 性能改进: {improvement:.2f}%")
    
    # 发散检测
    divergence = validator.detect_divergence()
    print(f"  • 模型发散: {divergence.get('has_divergence', False)}")


def example_3_model_ensemble_voting():
    """示例 3: 集成模型的投票预测"""
    print("\n" + "="*60)
    print("示例 3: 集成模型投票预测")
    print("="*60)
    
    from smart_flood_system.models.baseline.baseline_models import create_baseline_model
    from smart_flood_system.utils.ensemble_verifier import EnsembleValidator
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建三个不同的模型
    print("创建 3 个 Baseline 模型用于投票...")
    model1 = create_baseline_model('lstm', prediction_horizon=6).to(device)
    model2 = create_baseline_model('transformer', prediction_horizon=6).to(device)
    model3 = create_baseline_model('ensemble', prediction_horizon=6).to(device)
    
    models = [model1, model2, model3]
    
    # 创建 Ensemble 验证器
    ensemble_validator = EnsembleValidator()
    print("✅ Ensemble 验证器创建完成")
    
    # 生成测试数据
    test_input = torch.randn(2, 24, 5, 6).to(device)
    
    # 获取所有模型的预测
    print("获取三个模型的预测...")
    predictions = []
    with torch.no_grad():
        for i, model in enumerate(models):
            pred = model(test_input)
            predictions.append(pred)
            print(f"  Model {i+1} 输出形状: {pred.shape}")
    
    # 集成投票
    predictions_tensor = torch.stack(predictions)  # (num_models, batch, time, nodes, features)
    
    # 计算不确定性
    uncertainty = torch.std(predictions_tensor, dim=0)
    ensemble_pred = torch.mean(predictions_tensor, dim=0)
    
    print(f"\n投票结果:")
    print(f"  • Ensemble 预测形状: {ensemble_pred.shape}")
    print(f"  • 不确定性形状: {uncertainty.shape}")
    print(f"  • 平均不确定性: {uncertainty.mean().item():.6f}")
    print(f"  • 最大不确定性: {uncertainty.max().item():.6f}")
    print(f"  • 最小不确定性: {uncertainty.min().item():.6f}")


def example_4_confidence_scoring():
    """示例 4: 置信度评分与模型选择"""
    print("\n" + "="*60)
    print("示例 4: 置信度评分与模型选择")
    print("="*60)
    
    print("""
模型协议度评分指南:

┌─────────────────┬────────────────┬──────────────────────────────┐
│ 协议度范围      │ 置信度等级     │ 建议操作                     │
├─────────────────┼────────────────┼──────────────────────────────┤
│ > 0.95          │ 🟢 非常高      │ 两个模型都可信，优先用 PINN  │
│ 0.85 - 0.95     │ 🟢 高          │ 模型协议度高，可以组合使用   │
│ 0.70 - 0.85     │ 🟡 中等        │ 差异较大，需特别关注         │
│ < 0.70          │ 🔴 低          │ 模型发散，检查数据并重新训练 │
└─────────────────┴────────────────┴──────────────────────────────┘

模型选择策略:
    
1️⃣  常规预测 (协议度 > 0.85)
   → 使用 Baseline (LSTM/Transformer)
   → 理由: 计算效率高，精度足够
   
2️⃣  极端洪水 (任何情况)
   → 优先使用 PINN
   → 理由: 物理约束保证稳定性
   
3️⃣  模型发散 (协议度 < 0.70)
   → 检查输入数据质量
   → 确认是否有极端水文事件
   → 优先参考 PINN 预测
   
4️⃣  实时运营决策
   → 同时参考两个模型
   → 融合预测 (Ensemble)
   → 置信度低时触发告警
    """)


def example_5_training_both_models():
    """示例 5: 同时训练 PINN 和 Baseline"""
    print("\n" + "="*60)
    print("示例 5: 双模型训练")
    print("="*60)
    
    print("""
train() 方法的新参数:
    • epochs: 训练轮数
    • batch_size: 批大小
    • train_baseline: 是否同时训练 Baseline 模型 (Default: True)
    
双模型训练流程:
    1. 训练 PINN 主模型 (含物理约束)
    2. 训练 Baseline 模型 (纯数据驱动)
    3. 计算模型对比指标
    4. 显示性能改进百分比

示例代码:
    
    system = SmartFloodSystem()
    system.train(
        data_path='data/training.csv',
        epochs=100,
        batch_size=32,
        train_baseline=True  # 启用 Baseline 训练
    )
    
输出示例:
    2024-XX-XX 14:30:00 开始训练: 100 轮, 批大小 32
    2024-XX-XX 14:30:10 PINN Epoch 10/100
    ...
    2024-XX-XX 14:35:00 PINN 主模型训练完成
    2024-XX-XX 14:35:01 开始训练 Baseline 对比模型...
    2024-XX-XX 14:40:00 Baseline Epoch 10/100
    ...
    2024-XX-XX 14:45:00 Baseline 模型训练完成
    2024-XX-XX 14:45:01 进行多模型对比验证...
    2024-XX-XX 14:45:05 PINN MSE: 0.123456
    2024-XX-XX 14:45:05 Baseline MSE: 0.189012
    2024-XX-XX 14:45:05 PINN 改进: 34.67%
    """)


def main():
    """主程序"""
    print("\n" + "="*60)
    print("🌊 黄河智慧防汛系统 v2.0")
    print("   Baseline 模型对比快速入门")
    print("="*60)
    
    # 运行示例
    examples = [
        ("基本预测与对比", example_1_basic_prediction_with_comparison),
        ("直接模型对比", example_2_direct_model_comparison),
        ("集成模型投票", example_3_model_ensemble_voting),
        ("置信度评分", example_4_confidence_scoring),
        ("双模型训练", example_5_training_both_models),
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n运行所有示例演示...")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ 示例执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ 快速入门演示完成!")
    print("="*60)
    print("""
下一步:
  1. 查看 BASELINE_INTEGRATION_REPORT.md 了解完整文档
  2. 在 Dashboard v2 的"模型对比"标签页查看实时预测
  3. 针对你的特定场景自定义模型配置
  4. 在生产环境中部署和监控模型性能
    """)


if __name__ == '__main__':
    main()
