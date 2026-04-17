"""Baseline vs PINN 模型对比可视化脚本。

对比内容：
1. 纯数据驱动 (λ_phys=0.0) vs 物理约束 (λ_phys=0.1)
2. 在同一图上展示两个模型的预测曲线和真实观测点

Usage:
    python scripts/compare.py
    python scripts/compare.py --test-data data/processed/test.csv
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.pinn_model import YellowRiverPINN


def load_model_and_norm_params(model_path: str, norm_params_path: str):
    """加载模型和归一化参数。"""
    # 加载归一化参数
    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)
    
    x_mean = np.array(norm_params['x_mean'])
    x_std = np.array(norm_params['x_std'])
    y_mean = np.array(norm_params['y_mean'])
    y_std = np.array(norm_params['y_std'])
    
    # 初始化模型
    model = YellowRiverPINN(input_dim=5, output_dim=2, activation='swish')
    
    # 加载权重 (weights_only=False for compatibility with PyTorch 2.6+)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, (x_mean, x_std, y_mean, y_std)


def inverse_normalize(y_normalized: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray):
    """反归一化。"""
    return y_normalized * y_std + y_mean


def generate_test_data(n_samples: int = 200, seed: int = 123):
    """生成测试数据（真实观测点）。"""
    np.random.seed(seed)
    
    # 时间序列：模拟连续的时间段
    time = np.linspace(0, 10, n_samples)
    
    # 空间位置：固定河段位置
    location = np.ones(n_samples) * 0.5
    
    # 流量：正弦波动（模拟季节性变化）
    flow_rate = 1000 + 300 * np.sin(time * 0.5) + np.random.normal(0, 50, n_samples)
    
    # 降雨量：随机降雨事件
    rainfall = np.maximum(0, np.random.exponential(10, n_samples) - 5)
    
    # 温度：季节性变化
    temperature = 15 + 10 * np.sin(time * 0.3) + np.random.normal(0, 2, n_samples)
    
    # 构建物理关系：水位和含沙量的真实值
    # 水位与流量正相关，与温度负相关（蒸发）
    water_level = 50 + 0.02 * flow_rate + 0.5 * rainfall - 0.3 * temperature + np.random.normal(0, 2, n_samples)
    
    # 含沙量与流量正相关（冲刷作用），与降雨量正相关（侵蚀）
    sediment_content = 5 + 0.005 * flow_rate + 0.1 * rainfall + np.random.normal(0, 0.5, n_samples)
    
    df = pd.DataFrame({
        'time': time,
        'location': location,
        'flow_rate': flow_rate,
        'rainfall': rainfall,
        'temperature': temperature,
        'water_level': water_level,
        'sediment_content': sediment_content
    })
    
    return df


def compare_models(
    baseline_model: YellowRiverPINN,
    pinn_model: YellowRiverPINN,
    norm_params: tuple,
    test_df: pd.DataFrame,
    save_path: str
):
    """对比两个模型的预测效果。"""
    x_mean, x_std, y_mean, y_std = norm_params
    
    # 提取输入特征
    input_features = ['time', 'location', 'flow_rate', 'rainfall', 'temperature']
    X_test = test_df[input_features].values
    
    # 归一化
    X_normalized = (X_test - x_mean) / x_std
    X_tensor = torch.FloatTensor(X_normalized)
    
    # 获取预测
    with torch.no_grad():
        y_baseline_pred = baseline_model(X_tensor).numpy()
        y_pinn_pred = pinn_model(X_tensor).numpy()
    
    # 反归一化
    y_baseline_pred = inverse_normalize(y_baseline_pred, y_mean, y_std)
    y_pinn_pred = inverse_normalize(y_pinn_pred, y_mean, y_std)
    
    # 真实值
    y_true = test_df[['water_level', 'sediment_content']].values
    
    # 计算误差
    baseline_mse = np.mean((y_baseline_pred - y_true) ** 2, axis=0)
    pinn_mse = np.mean((y_pinn_pred - y_true) ** 2, axis=0)
    
    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    time = test_df['time'].values
    
    # 1. 水位对比
    ax1 = axes[0]
    ax1.scatter(time, y_true[:, 0], c='black', s=30, alpha=0.6, 
                label='真实观测点 (True)', zorder=5)
    ax1.plot(time, y_baseline_pred[:, 0], 'b-', linewidth=2, 
             label=f'基线模型 λ=0.0 (MSE={baseline_mse[0]:.2f})', alpha=0.8)
    ax1.plot(time, y_pinn_pred[:, 0], 'r-', linewidth=2, 
             label=f'PINN模型 λ=0.1 (MSE={pinn_mse[0]:.2f})', alpha=0.8)
    ax1.set_xlabel('时间 (Time)', fontsize=12)
    ax1.set_ylabel('水位 (Water Level)', fontsize=12)
    ax1.set_title('黄河水位预测对比：Baseline vs PINN', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. 含沙量对比
    ax2 = axes[1]
    ax2.scatter(time, y_true[:, 1], c='black', s=30, alpha=0.6, 
                label='真实观测点 (True)', zorder=5)
    ax2.plot(time, y_baseline_pred[:, 1], 'b-', linewidth=2, 
             label=f'基线模型 λ=0.0 (MSE={baseline_mse[1]:.3f})', alpha=0.8)
    ax2.plot(time, y_pinn_pred[:, 1], 'r-', linewidth=2, 
             label=f'PINN模型 λ=0.1 (MSE={pinn_mse[1]:.3f})', alpha=0.8)
    ax2.set_xlabel('时间 (Time)', fontsize=12)
    ax2.set_ylabel('含沙量 (Sediment Content)', fontsize=12)
    ax2.set_title('黄河含沙量预测对比：Baseline vs PINN', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[Plot] 对比图保存至: {save_path}")
    
    # 打印误差对比
    print("\n" + "=" * 60)
    print("模型对比结果")
    print("=" * 60)
    print(f"{'指标':<20} {'基线模型 (λ=0.0)':<20} {'PINN模型 (λ=0.1)':<20}")
    print("-" * 60)
    print(f"{'水位 MSE':<20} {baseline_mse[0]:<20.4f} {pinn_mse[0]:<20.4f}")
    print(f"{'含沙量 MSE':<20} {baseline_mse[1]:<20.4f} {pinn_mse[1]:<20.4f}")
    print("=" * 60)
    
    # 判断哪个更好
    water_improvement = (baseline_mse[0] - pinn_mse[0]) / baseline_mse[0] * 100
    sediment_improvement = (baseline_mse[1] - pinn_mse[1]) / baseline_mse[1] * 100
    
    print(f"\nPINN 相比 Baseline:")
    print(f"  - 水位预测改进: {water_improvement:+.1f}%")
    print(f"  - 含沙量预测改进: {sediment_improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Compare Baseline vs PINN Models')
    parser.add_argument('--baseline-model', type=str, default='models/model_baseline.pth',
                        help='基线模型路径')
    parser.add_argument('--pinn-model', type=str, default='models/model_pinn.pth',
                        help='PINN模型路径')
    parser.add_argument('--norm-params', type=str, default='data/processed/norm_params.json',
                        help='归一化参数路径')
    parser.add_argument('--test-data', type=str, default=None,
                        help='测试数据CSV路径 (默认生成模拟数据)')
    parser.add_argument('--save-path', type=str, default='results/comparison.png',
                        help='对比图保存路径')
    parser.add_argument('--n-samples', type=int, default=200,
                        help='测试样本数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Baseline vs PINN 模型对比")
    print("=" * 60)
    
    # 加载模型
    print("\n[Loading] 加载模型...")
    
    # 尝试从训练目录找归一化参数
    baseline_norm = 'models/baseline/20260417_101109/norm_params.json'
    if os.path.exists(baseline_norm):
        args.norm_params = baseline_norm
    
    baseline_model, norm_params = load_model_and_norm_params(
        args.baseline_model, args.norm_params
    )
    pinn_model, _ = load_model_and_norm_params(args.pinn_model, args.norm_params)
    
    print(f"  ✓ Baseline: {args.baseline_model}")
    print(f"  ✓ PINN: {args.pinn_model}")
    
    # 加载或生成测试数据
    if args.test_data and os.path.exists(args.test_data):
        print(f"\n[Data] 加载测试数据: {args.test_data}")
        test_df = pd.read_csv(args.test_data)
    else:
        print(f"\n[Data] 生成 {args.n_samples} 个测试样本...")
        test_df = generate_test_data(args.n_samples)
    
    # 创建结果目录
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 对比并绘图
    compare_models(baseline_model, pinn_model, norm_params, test_df, args.save_path)
    
    print("\n" + "=" * 60)
    print("对比完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
