#!/usr/bin/env python3
"""
生成模型预测结果
使用增强版 STGNN v3.0 进行预测
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stgnn_enhanced import create_enhanced_model


def generate_test_data(n_samples=50, seq_len=24, n_nodes=5, n_features=6):
    """生成测试数据 - 模拟黄河洪水场景"""
    
    data = []
    targets = []
    timestamps = []
    base_time = datetime(2024, 8, 15, 0, 0)  # 汛期时间
    
    for i in range(n_samples):
        # 模拟不同洪水强度
        flood_scenario = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        
        if flood_scenario == 'low':
            base_level = 1000 + np.random.rand() * 50
            base_discharge = 1200 + np.random.exponential(200)
        elif flood_scenario == 'medium':
            base_level = 1050 + np.random.rand() * 80
            base_discharge = 2000 + np.random.exponential(400)
        else:  # high
            base_level = 1100 + np.random.rand() * 100
            base_discharge = 3500 + np.random.exponential(600)
        
        base_discharge = min(base_discharge, 5500)
        base_sediment = 5 + 0.004 * base_discharge + np.random.randn() * 3
        
        # 生成序列数据
        seq_data = []
        for t in range(seq_len):
            trend = np.sin(2 * np.pi * t / 24) * 30
            
            for node in range(n_nodes):
                delay = node * 2
                attenuation = 1 - node * 0.08
                
                effective_t = min(max(t - delay, 0), seq_len - 1)
                effective_trend = np.sin(2 * np.pi * effective_t / 24) * 30 * attenuation
                
                features = [
                    base_level + effective_trend + np.random.randn() * 8,
                    base_discharge + effective_trend * 2 + np.random.randn() * 80,
                    max(0, base_sediment + np.random.randn() * 2),
                    20 + np.random.randn() * 4,
                    max(0, np.random.exponential(3) if flood_scenario == 'high' else np.random.exponential(1)),
                    np.random.rand()
                ]
                seq_data.append(features)
        
        data.append(seq_data)
        
        # 生成真实目标 (未来6小时)
        target = []
        for t in range(6):
            future_t = seq_len + t
            for node in range(n_nodes):
                delay = node * 2
                attenuation = 1 - node * 0.08
                
                effective_t = min(max(future_t - delay, 0), seq_len - 1)
                effective_trend = np.sin(2 * np.pi * effective_t / 24) * 30 * attenuation
                
                target.append([
                    base_level + effective_trend * 1.3 + np.random.randn() * 12,
                    base_discharge + effective_trend * 2.5 + np.random.randn() * 120,
                    max(0, base_sediment * 1.15 + np.random.randn() * 3),
                    20 + np.random.randn() * 4,
                    max(0, np.random.exponential(2)),
                    np.random.rand()
                ])
        
        targets.append(target)
        timestamps.append(base_time + timedelta(hours=i))
    
    data = np.array(data).reshape(n_samples, seq_len, n_nodes, n_features)
    targets = np.array(targets).reshape(n_samples, 6, n_nodes, n_features)
    
    return (
        torch.from_numpy(data).float(),
        torch.from_numpy(targets).float(),
        timestamps
    )


def main():
    print("="*70)
    print("🌊 黄河防汛 PINN v3.0 预测生成")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 创建模型
    print("\n[1] 创建增强版 STGNN v3.0 模型...")
    model_config = {
        'node_features': 6,
        'n_nodes': 5,
        'gnn_hidden': 64,
        'gnn_layers': 3,
        'transformer_heads': 8,
        'transformer_layers': 4,
        'sequence_length': 24,
        'prediction_horizon': 6,
        'dropout': 0.1,
        'use_physics_attention': True,
        'use_adaptive_gnn': True,
        'use_multiscale_temporal': True,
        'use_physics_postprocess': True,
        'use_bayesian': False
    }
    
    model = create_enhanced_model(model_config).to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    模型参数量: {total_params:,}")
    
    # 加载归一化参数（用于反归一化）
    print("\n[1.5] 加载归一化参数...")
    checkpoint_path = Path("experiments/models/pinn_best.pth")
    if not checkpoint_path.exists():
        checkpoint_path = Path("experiments/models/best_model.pth")
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get('protocol_version', 1) < 2:
            raise RuntimeError("检查点使用旧评估协议，请先重新训练模型")
        model.load_state_dict(checkpoint['model_state_dict'])
        data_mean = checkpoint.get('mean', checkpoint.get('data_mean'))
        data_std = checkpoint.get('std', checkpoint.get('data_std'))
        if data_mean is not None and data_std is not None:
            data_mean = data_mean.squeeze()
            data_std = data_std.squeeze()
            print(f"    已加载模型权重和 mean/std")
        else:
            raise KeyError(f"检查点 {checkpoint_path} 缺少训练集 mean/std")
    else:
        raise FileNotFoundError("未找到 PINN 检查点，请先运行训练脚本")
    
    # 生成测试数据
    print("\n[2] 生成测试数据 (50个样本, 模拟洪水场景)...")
    test_data, test_targets, timestamps = generate_test_data(n_samples=50)
    
    # 归一化输入数据（与训练时一致）
    print("    对输入数据进行归一化...")
    mean_reshaped = data_mean.view(1, 1, 1, -1)
    std_reshaped = data_std.view(1, 1, 1, -1)
    test_data_norm = (test_data - mean_reshaped) / std_reshaped
    test_targets_norm = (test_targets - mean_reshaped) / std_reshaped
    
    test_data_norm = test_data_norm.to(device)
    test_targets_norm = test_targets_norm.to(device)
    
    print(f"    输入形状: {test_data_norm.shape}")
    print(f"    目标形状: {test_targets_norm.shape}")
    
    # 生成预测
    print("\n[3] 生成预测...")
    with torch.no_grad():
        predictions = model(test_data_norm)
    
    print(f"    预测形状: {predictions.shape}")
    
    # 反归一化（如果知道 mean/std）
    print("\n[3.5] 反归一化预测值...")
    # 扩展到预测维度 (batch, time, nodes, features)
    mean_expanded = data_mean.view(1, 1, 1, -1).to(device)
    std_expanded = data_std.view(1, 1, 1, -1).to(device)
    
    predictions_orig = predictions * std_expanded + mean_expanded
    test_targets_orig = test_targets_norm * std_expanded + mean_expanded
    
    print(f"    预测值反归一化后范围: [{predictions_orig.min():.2f}, {predictions_orig.max():.2f}]")
    print(f"    真实值范围: [{test_targets_orig.min():.2f}, {test_targets_orig.max():.2f}]")
    
    # 计算误差（使用反归一化后的值）
    mse = torch.mean((predictions_orig - test_targets_orig) ** 2).item()
    mae = torch.mean(torch.abs(predictions_orig - test_targets_orig)).item()
    rmse = np.sqrt(mse)
    
    print(f"\n[4] 预测性能（原始尺度）:")
    print(f"    MSE:  {mse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    
    # 站点名称
    stations = ['兰州', '河口镇', '龙门', '三门峡', '花园口']
    features = ['水位(m)', '流量(m³/s)', '泥沙(kg/m³)', '温度(°C)', '降雨(mm/h)', '其他']
    
    # 保存预测结果
    print("\n[5] 保存预测结果...")
    
    output_dir = Path("experiments/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换为 numpy（使用反归一化后的值）
    pred_np = predictions_orig.cpu().numpy()
    target_np = test_targets_orig.cpu().numpy()
    
    # 保存为 JSON 格式 (前5个样本)
    results = []
    for i in range(min(5, len(timestamps))):
        sample_result = {
            "sample_id": i,
            "timestamp": timestamps[i].isoformat(),
            "stations": {}
        }
        
        for s_idx, station in enumerate(stations):
            station_data = {
                "predictions": {},
                "ground_truth": {}
            }
            
            for f_idx, feature in enumerate(features):
                # 取第一个预测时间步
                pred_val = float(pred_np[i, 0, s_idx, f_idx])
                true_val = float(target_np[i, 0, s_idx, f_idx])
                
                station_data["predictions"][feature] = round(pred_val, 3)
                station_data["ground_truth"][feature] = round(true_val, 3)
            
            sample_result["stations"][station] = station_data
        
        results.append(sample_result)
    
    json_path = output_dir / "predictions_sample.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"    JSON样本: {json_path}")
    
    # 保存为 CSV 格式 (所有样本, 所有站点, 水位和流量)
    csv_data = []
    for i in range(len(timestamps)):
        for s_idx, station in enumerate(stations):
            for t in range(6):  # 6个预测时间步
                row = {
                    'sample_id': i,
                    'timestamp': timestamps[i] + timedelta(hours=t+1),
                    'station': station,
                    'pred_step': t + 1,
                    'pred_water_level': round(float(pred_np[i, t, s_idx, 0]), 2),
                    'true_water_level': round(float(target_np[i, t, s_idx, 0]), 2),
                    'pred_discharge': round(float(pred_np[i, t, s_idx, 1]), 2),
                    'true_discharge': round(float(target_np[i, t, s_idx, 1]), 2),
                    'pred_sediment': round(float(pred_np[i, t, s_idx, 2]), 3),
                    'true_sediment': round(float(target_np[i, t, s_idx, 2]), 3),
                }
                csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_path = output_dir / "predictions_all.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"    CSV完整数据: {csv_path}")
    print(f"    总记录数: {len(df)}")
    
    # 保存为 numpy 格式 (用于后续分析)
    np_path = output_dir / "predictions.npz"
    np.savez(np_path, 
             predictions=pred_np, 
             ground_truth=target_np,
             timestamps=[t.isoformat() for t in timestamps])
    
    print(f"    NumPy数组: {np_path}")
    
    # 打印样本预览
    print("\n[6] 预测样本预览 (样本 #0, 兰州站):")
    print("-" * 70)
    print(f"{'时间步':<8} {'特征':<12} {'预测值':<12} {'真实值':<12} {'误差':<10}")
    print("-" * 70)
    
    for t in range(6):
        for f_idx, feature in enumerate(features[:4]):  # 前4个特征
            pred_val = pred_np[0, t, 0, f_idx]
            true_val = target_np[0, t, 0, f_idx]
            error = abs(pred_val - true_val)
            
            print(f"t+{t+1:<5}  {feature:<10} {pred_val:>10.2f}  {true_val:>10.2f}  {error:>8.2f}")
    
    print("-" * 70)
    
    # 保存模型配置
    config_path = output_dir / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"\n[7] 模型配置: {config_path}")
    
    print("\n" + "="*70)
    print("✅ 预测生成完成!")
    print(f"📁 输出目录: {output_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()
