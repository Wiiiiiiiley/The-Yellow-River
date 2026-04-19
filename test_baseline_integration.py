#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline 模型集成测试脚本

测试内容:
1. 验证 Baseline 模型初始化
2. 测试多模型预测
3. 验证模型对比指标
4. 测试双模型训练
"""

import torch
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from smart_flood_system.main_system import SmartFloodSystem
from smart_flood_system.models.baseline.baseline_models import create_baseline_model
from smart_flood_system.utils.ensemble_verifier import MultiModelValidator, EnsembleValidator


def test_baseline_initialization():
    """测试 Baseline 模型初始化"""
    print("\n" + "="*60)
    print("测试 1: Baseline 模型初始化")
    print("="*60)
    
    try:
        # 创建 Baseline 模型
        baseline_lstm = create_baseline_model(
            model_type='lstm',
            input_size=6,
            hidden_size=128,
            output_size=6,
            prediction_horizon=6
        )
        print("✅ LSTM Baseline 创建成功")
        
        baseline_transformer = create_baseline_model(
            model_type='transformer',
            input_size=6,
            hidden_size=128,
            output_size=6,
            prediction_horizon=6
        )
        print("✅ Transformer Baseline 创建成功")
        
        baseline_ensemble = create_baseline_model(
            model_type='ensemble',
            input_size=6,
            hidden_size=128,
            output_size=6,
            prediction_horizon=6
        )
        print("✅ Ensemble Baseline 创建成功")
        
        # 测试前向传播
        test_input = torch.randn(2, 24, 5, 6)  # (batch, time, nodes, features)
        
        with torch.no_grad():
            lstm_out = baseline_lstm(test_input)
            print(f"   LSTM 输出形状: {lstm_out.shape}")
            
            transformer_out = baseline_transformer(test_input)
            print(f"   Transformer 输出形状: {transformer_out.shape}")
            
            ensemble_out = baseline_ensemble(test_input)
            print(f"   Ensemble 输出形状: {ensemble_out.shape}")
        
        print("\n✅ Baseline 模型初始化测试通过")
        return True
    
    except Exception as e:
        print(f"\n❌ Baseline 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_model_prediction():
    """测试多模型预测"""
    print("\n" + "="*60)
    print("测试 2: 多模型预测")
    print("="*60)
    
    try:
        # 初始化系统
        print("初始化 SmartFloodSystem...")
        system = SmartFloodSystem()
        print("✅ 系统初始化成功")
        
        # 进行预测（带 Baseline 对比）
        print("\n执行预测（带 Baseline 对比）...")
        predictions = system.predict(
            station_id='HYD_001',
            sequence_length=24,
            with_baseline_validation=True
        )
        
        print(f"✅ 预测成功")
        print(f"   PINN 预测键值: {list(predictions.get('pinn_prediction', {}).keys())}")
        print(f"   Baseline 预测键值: {list(predictions.get('baseline_prediction', {}).keys())}")
        
        # 打印模型对比信息
        if 'model_comparison' in predictions:
            comp = predictions['model_comparison']
            print(f"\n   模型协议度: {predictions.get('model_agreement_score', 0):.4f}")
            print(f"   置信度: {predictions.get('confidence_level', '未知')}")
            print(f"   平均差异: {comp.get('mean_abs_difference', 0):.6f}")
            print(f"   最大差异: {comp.get('max_abs_difference', 0):.6f}")
            print(f"   余弦相似度: {comp.get('cosine_similarity', 0):.4f}")
            print(f"   相关系数: {comp.get('correlation', 0):.4f}")
            print(f"   模型发散: {comp.get('has_divergence', False)}")
        
        print("\n✅ 多模型预测测试通过")
        return True
    
    except Exception as e:
        print(f"\n❌ 多模型预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_comparison():
    """测试模型对比指标"""
    print("\n" + "="*60)
    print("测试 3: 模型对比指标计算")
    print("="*60)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建 PINN 和 Baseline 模型
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
        print("✅ 多模型验证器创建成功")
        
        # 生成测试数据
        test_input = torch.randn(4, 24, 5, 6).to(device)
        
        # 获取比较结果
        with torch.no_grad():
            pinn_out = pinn_model(test_input)
            baseline_out = baseline_model(test_input)
        
        # 计算对比指标
        comparison = {
            'pinn_pred': pinn_out,
            'baseline_pred': baseline_out,
            'ground_truth': torch.randn_like(pinn_out)
        }
        
        print("\n执行对比分析...")
        
        # 比较预测
        pinn_mse = torch.nn.functional.mse_loss(pinn_out, comparison['ground_truth']).item()
        baseline_mse = torch.nn.functional.mse_loss(baseline_out, comparison['ground_truth']).item()
        
        improvement = (1 - pinn_mse / (baseline_mse + 1e-6)) * 100
        
        print(f"   PINN MSE: {pinn_mse:.6f}")
        print(f"   Baseline MSE: {baseline_mse:.6f}")
        print(f"   性能改进: {improvement:.2f}%")
        
        # 检测发散
        divergence_info = validator.detect_divergence()
        print(f"   模型发散检测: {divergence_info.get('has_divergence', False)}")
        if divergence_info.get('has_divergence'):
            print(f"   发散原因: {divergence_info.get('reason', '未知')}")
        
        print("\n✅ 模型对比指标测试通过")
        return True
    
    except Exception as e:
        print(f"\n❌ 模型对比指标测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_model_training():
    """测试双模型训练"""
    print("\n" + "="*60)
    print("测试 4: 双模型训练")
    print("="*60)
    
    try:
        # 初始化系统
        print("初始化系统...")
        system = SmartFloodSystem()
        print("✅ 系统初始化成功")
        
        # 创建模拟数据文件（仅用于测试）
        import tempfile
        import pandas as pd
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # 创建简单的 CSV 数据
            df = pd.DataFrame({
                'station': ['HYD_001'] * 100,
                'water_level': np.random.uniform(1000, 1050, 100),
                'discharge': np.random.uniform(1000, 2000, 100),
                'sediment': np.random.uniform(20, 50, 100),
            })
            df.to_csv(f.name, index=False)
            data_file = f.name
        
        print(f"创建测试数据文件: {data_file}")
        
        # 执行训练（简化版本，仅 5 个 epoch）
        print("\n执行双模型训练（5 epochs）...")
        system.train(
            data_path=data_file,
            epochs=5,
            batch_size=16,
            train_baseline=True
        )
        print("✅ 双模型训练完成")
        
        # 清理测试文件
        import os
        os.unlink(data_file)
        
        print("\n✅ 双模型训练测试通过")
        return True
    
    except Exception as e:
        print(f"\n❌ 双模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_report():
    """测试验证报告生成"""
    print("\n" + "="*60)
    print("测试 5: 验证报告生成")
    print("="*60)
    
    try:
        # 初始化系统
        system = SmartFloodSystem()
        
        # 生成测试预测和真实值
        test_predictions = np.random.randn(32, 6, 5, 6)
        ground_truth = np.random.randn(32, 6, 5, 6)
        
        # 获取验证报告
        print("生成验证报告...")
        report = system.get_validation_report(
            test_predictions=test_predictions,
            ground_truth=ground_truth
        )
        
        print("✅ 验证报告生成成功")
        print(f"   报告键值: {list(report.keys())}")
        
        # 打印报告摘要
        if 'model_summary' in report:
            print(f"\n   模型摘要:")
            for key, value in report['model_summary'].items():
                print(f"     {key}: {value}")
        
        print("\n✅ 验证报告生成测试通过")
        return True
    
    except Exception as e:
        print(f"\n❌ 验证报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("🧪 Baseline 模型集成测试")
    print("="*60)
    
    results = {}
    
    # 运行所有测试
    results['初始化'] = test_baseline_initialization()
    results['预测'] = test_multi_model_prediction()
    results['对比'] = test_model_comparison()
    results['报告'] = test_validation_report()
    # results['训练'] = test_dual_model_training()  # 可选：耗时较长
    
    # 打印总结
    print("\n" + "="*60)
    print("📊 测试结果总结")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总体: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！Baseline 模型集成完全就绪")
        return 0
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败，请检查错误信息")
        return 1


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
