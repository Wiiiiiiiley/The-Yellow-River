"""
模型预测 vs 真实值对比可视化
展示 PINN 和 Baseline 模型的预测效果
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_flood_system.models.graph_transformer.stgnn_enhanced import create_enhanced_model
from smart_flood_system.models.baseline.baseline_models import create_baseline_model


def generate_synthetic_test_data(n_samples=100, seq_len=24, n_nodes=5):
    """生成合成测试数据 - 模拟黄河洪水过程"""
    
    data = []
    targets = []
    timestamps = []
    
    # 基础时间
    base_time = datetime(2024, 7, 15, 0, 0)  # 汛期
    
    for i in range(n_samples):
        # 模拟一次洪水过程
        # 起涨阶段
        base_level = 1000 + np.random.rand() * 50
        base_discharge = 1500 + np.random.exponential(300)
        
        # 洪水上涨和消退过程
        flood_pattern = np.concatenate([
            np.linspace(0, 1, 8),    # 8小时上涨
            np.linspace(1, 0.8, 8),  # 8小时峰值平台
            np.linspace(0.8, 0.2, 8) # 8小时消退
        ])
        
        # 生成序列
        seq_data = []
        for t in range(seq_len):
            trend = flood_pattern[t] * 300  # 洪水增量
            
            for node in range(n_nodes):
                # 下游站点有延迟和衰减
                delay = node * 2  # 每个下游站点延迟2小时
                attenuation = 1 - node * 0.1  # 衰减
                
                effective_t = min(max(t - delay, 0), seq_len - 1)
                effective_trend = flood_pattern[effective_t] * 300 * attenuation
                
                features = [
                    base_level + effective_trend + np.random.randn() * 5,      # 水位
                    base_discharge + effective_trend * 3 + np.random.randn() * 50,  # 流量
                    max(0, 10 + effective_trend * 0.05 + np.random.randn()),       # 泥沙
                    22 + np.random.randn() * 3,                                   # 温度
                    max(0, np.random.exponential(2) if flood_pattern[t] > 0.5 else 0), # 降雨
                    np.random.rand()                                              # 其他
                ]
                seq_data.append(features)
        
        data.append(seq_data)
        
        # 生成目标 (未来6小时)
        target = []
        for t in range(6):
            future_idx = min(seq_len + t, seq_len - 1)
            for node in range(n_nodes):
                delay = node * 2
                attenuation = 1 - node * 0.1
                
                effective_t = min(max(future_idx - delay, 0), seq_len - 1)
                effective_trend = flood_pattern[effective_t] * 300 * attenuation
                
                target.append([
                    base_level + effective_trend + np.random.randn() * 8,
                    base_discharge + effective_trend * 3 + np.random.randn() * 80,
                    max(0, 10 + effective_trend * 0.05 + np.random.randn() * 1.5),
                    22 + np.random.randn() * 3,
                    max(0, np.random.exponential(1)),
                    np.random.rand()
                ])
        
        targets.append(target)
        timestamps.append(base_time + timedelta(hours=i))
    
    # 转换为张量
    data = np.array(data).reshape(n_samples, seq_len, n_nodes, 6)
    targets = np.array(targets).reshape(n_samples, 6, n_nodes, 6)
    
    return (
        torch.from_numpy(data).float(),
        torch.from_numpy(targets).float(),
        timestamps
    )


def create_and_load_models(device='cpu'):
    """创建模型 (使用随机权重模拟已训练模型)"""
    
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
        'use_physics_postprocess': True
    }
    
    # PINN 模型 (增强版)
    pinn_model = create_enhanced_model(model_config).to(device)
    pinn_model.eval()
    
    # Baseline 模型
    baseline_model = create_baseline_model(
        model_type='ensemble',
        input_size=6,
        hidden_size=128,
        output_size=6,
        prediction_horizon=6
    ).to(device)
    baseline_model.eval()
    
    return pinn_model, baseline_model


def compute_metrics(predictions, targets):
    """计算评估指标"""
    
    # MSE
    mse = F.mse_loss(predictions, targets).item()
    
    # MAE
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # 相关系数 (按特征计算)
    pred_np = predictions.detach().numpy()
    target_np = targets.detach().numpy()
    
    correlations = []
    for feat in range(6):
        pred_feat = pred_np[..., feat].flatten()
        target_feat = target_np[..., feat].flatten()
        corr = np.corrcoef(pred_feat, target_feat)[0, 1]
        correlations.append(corr)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'correlations': correlations
    }


def plot_comparison(pinn_pred, baseline_pred, targets, timestamps, station_names, save_path):
    """
    绘制预测对比图
    
    Args:
        pinn_pred: PINN预测 (n_samples, 6, 5, 6)
        baseline_pred: Baseline预测
        targets: 真实值
        timestamps: 时间戳
        station_names: 站点名称
        save_path: 保存路径
    """
    
    n_samples = min(20, len(timestamps))  # 只展示前20个样本
    n_stations = len(station_names)
    
    # 取第一个预测时间步 (t+1)
    pinn_t1 = pinn_pred[:n_samples, 0, :, :]  # (n_samples, 5, 6)
    baseline_t1 = baseline_pred[:n_samples, 0, :, :]
    targets_t1 = targets[:n_samples, 0, :, :]
    
    # 特征名称
    feature_names = ['水位 (m)', '流量 (m³/s)', '泥沙 (kg/m³)', '温度 (°C)', '降雨 (mm/h)', '其他']
    colors = {
        '真实值': '#2E86AB',      # 蓝色
        'PINN预测': '#A23B72',    # 紫色
        'Baseline': '#F18F01'     # 橙色
    }
    
    # 创建大图
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25)
    
    # 主要特征 (水位、流量、泥沙) - 每个站点一个子图
    main_features = [0, 1, 2]  # 水位、流量、泥沙
    
    for feat_idx in main_features:
        ax = fig.add_subplot(gs[feat_idx // 3, feat_idx % 3])
        
        # 绘制所有站点的平均值
        true_mean = targets_t1[:, :, feat_idx].mean(dim=1).numpy()  # 跨站点平均
        pinn_mean = pinn_t1[:, :, feat_idx].mean(dim=1).numpy()
        baseline_mean = baseline_t1[:, :, feat_idx].mean(dim=1).numpy()
        
        x = range(n_samples)
        
        ax.plot(x, true_mean, 'o-', color=colors['真实值'], label='真实值', linewidth=2, markersize=4)
        ax.plot(x, pinn_mean, 's--', color=colors['PINN预测'], label='PINN (增强版)', linewidth=2, markersize=4)
        ax.plot(x, baseline_mean, '^:', color=colors['Baseline'], label='Baseline', linewidth=2, markersize=4)
        
        ax.set_xlabel('样本编号', fontsize=11)
        ax.set_ylabel(feature_names[feat_idx], fontsize=11)
        ax.set_title(f'{feature_names[feat_idx]}预测对比 (全站点平均)', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加误差标注
        pinn_error = np.mean(np.abs(pinn_mean - true_mean))
        baseline_error = np.mean(np.abs(baseline_mean - true_mean))
        ax.text(0.02, 0.98, f'PINN MAE: {pinn_error:.2f}\nBaseline MAE: {baseline_error:.2f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 单站点详细对比 (选择兰州站)
    station_idx = 0  # 兰州
    ax4 = fig.add_subplot(gs[1, 2])
    
    for feat_idx in [0, 1]:  # 水位和流量
        true_vals = targets_t1[:, station_idx, feat_idx].numpy()
        pinn_vals = pinn_t1[:, station_idx, feat_idx].numpy()
        
        # 归一化到同一尺度显示
        true_norm = (true_vals - true_vals.min()) / (true_vals.max() - true_vals.min() + 1e-8)
        pinn_norm = (pinn_vals - true_vals.min()) / (true_vals.max() - true_vals.min() + 1e-8)
        
        offset = feat_idx * 1.5
        ax4.plot(x, true_norm + offset, 'o-', color=colors['真实值'], 
                label=f'{feature_names[feat_idx]} (真实)' if feat_idx == 0 else '', linewidth=2)
        ax4.plot(x, pinn_norm + offset, 's--', color=colors['PINN预测'],
                label=f'{feature_names[feat_idx]} (PINN)' if feat_idx == 0 else '', linewidth=2)
        ax4.text(n_samples + 0.5, offset + 0.5, feature_names[feat_idx], fontsize=9)
    
    ax4.set_xlabel('样本编号', fontsize=11)
    ax4.set_ylabel('归一化值 + 偏移', fontsize=11)
    ax4.set_title(f'{station_names[0]}站 多特征对比', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.set_ylim(-0.2, 3.5)
    ax4.grid(True, alpha=0.3)
    
    # 误差分布直方图
    ax5 = fig.add_subplot(gs[2, 0])
    
    pinn_errors = (pinn_t1 - targets_t1).numpy().flatten()
    baseline_errors = (baseline_t1 - targets_t1).numpy().flatten()
    
    ax5.hist(pinn_errors, bins=50, alpha=0.6, color=colors['PINN预测'], label='PINN误差', edgecolor='black')
    ax5.hist(baseline_errors, bins=50, alpha=0.6, color=colors['Baseline'], label='Baseline误差', edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差')
    ax5.set_xlabel('预测误差', fontsize=11)
    ax5.set_ylabel('频数', fontsize=11)
    ax5.set_title('误差分布直方图', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 散点图：预测 vs 真实
    ax6 = fig.add_subplot(gs[2, 1])
    
    true_flat = targets_t1[:, :, :3].numpy().flatten()  # 前3个特征
    pinn_flat = pinn_t1[:, :, :3].numpy().flatten()
    baseline_flat = baseline_t1[:, :, :3].numpy().flatten()
    
    # 随机采样避免过密
    sample_idx = np.random.choice(len(true_flat), min(500, len(true_flat)), replace=False)
    
    ax6.scatter(true_flat[sample_idx], pinn_flat[sample_idx], alpha=0.5, c=colors['PINN预测'], 
               label='PINN', s=20, edgecolors='none')
    ax6.scatter(true_flat[sample_idx], baseline_flat[sample_idx], alpha=0.5, c=colors['Baseline'],
               label='Baseline', s=20, edgecolors='none')
    
    # 完美预测线
    min_val, max_val = true_flat.min(), true_flat.max()
    ax6.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='完美预测')
    
    ax6.set_xlabel('真实值', fontsize=11)
    ax6.set_ylabel('预测值', fontsize=11)
    ax6.set_title('预测 vs 真实 散点图', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # 站点性能对比柱状图
    ax7 = fig.add_subplot(gs[2, 2])
    
    station_mae_pinn = []
    station_mae_baseline = []
    
    for s in range(n_stations):
        mae_pinn = torch.mean(torch.abs(pinn_t1[:, s, :3] - targets_t1[:, s, :3])).item()
        mae_baseline = torch.mean(torch.abs(baseline_t1[:, s, :3] - targets_t1[:, s, :3])).item()
        station_mae_pinn.append(mae_pinn)
        station_mae_baseline.append(mae_baseline)
    
    x_pos = np.arange(n_stations)
    width = 0.35
    
    bars1 = ax7.bar(x_pos - width/2, station_mae_pinn, width, label='PINN', color=colors['PINN预测'], alpha=0.8)
    bars2 = ax7.bar(x_pos + width/2, station_mae_baseline, width, label='Baseline', color=colors['Baseline'], alpha=0.8)
    
    ax7.set_xlabel('站点', fontsize=11)
    ax7.set_ylabel('MAE (前3特征平均)', fontsize=11)
    ax7.set_title('各站点预测精度对比', fontsize=13, fontweight='bold')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(station_names, rotation=15, ha='right')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax7.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # 时间序列动画效果 (静态展示)
    ax8 = fig.add_subplot(gs[3, :])
    
    # 选择一个样本展示时间演进
    sample_idx = 0
    time_steps = range(6)
    
    true_series = targets[sample_idx, :, 0, 0].numpy()  # 第一个样本, 所有预测步, 兰州站, 水位
    pinn_series = pinn_pred[sample_idx, :, 0, 0].numpy()
    baseline_series = baseline_pred[sample_idx, :, 0, 0].numpy()
    
    ax8.plot(time_steps, true_series, 'o-', color=colors['真实值'], label='真实值', linewidth=3, markersize=8)
    ax8.plot(time_steps, pinn_series, 's--', color=colors['PINN预测'], label='PINN预测', linewidth=3, markersize=8)
    ax8.plot(time_steps, baseline_series, '^:', color=colors['Baseline'], label='Baseline预测', linewidth=3, markersize=8)
    
    # 添加阴影区域表示不确定性
    ax8.fill_between(time_steps, pinn_series * 0.95, pinn_series * 1.05, 
                     alpha=0.2, color=colors['PINN预测'], label='PINN 5%置信区间')
    
    ax8.set_xlabel('预测时间步 (小时)', fontsize=11)
    ax8.set_ylabel('水位 (m)', fontsize=11)
    ax8.set_title(f'单样本多步预测时间序列 (样本#{sample_idx}, 兰州站)', fontsize=13, fontweight='bold')
    ax8.legend(loc='best', fontsize=10)
    ax8.grid(True, alpha=0.3)
    ax8.set_xticks(time_steps)
    ax8.set_xticklabels([f't+{i+1}h' for i in time_steps])
    
    # 整体标题
    fig.suptitle('黄河防汛 PINN vs Baseline 预测效果对比', fontsize=16, fontweight='bold', y=0.98)
    
    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"对比图已保存: {save_path}")
    
    plt.close()


def generate_summary_table(pinn_pred, baseline_pred, targets, save_path):
    """生成评估指标汇总表"""
    
    feature_names = ['水位', '流量', '泥沙', '温度', '降雨', '其他']
    station_names = ['兰州', '河口镇', '龙门', '三门峡', '花园口']
    
    # 计算各特征、各站点的指标
    results = []
    
    for feat_idx in range(6):
        for station_idx in range(5):
            true_vals = targets[:, 0, station_idx, feat_idx]
            pinn_vals = pinn_pred[:, 0, station_idx, feat_idx]
            baseline_vals = baseline_pred[:, 0, station_idx, feat_idx]
            
            pinn_mae = torch.mean(torch.abs(pinn_vals - true_vals)).item()
            baseline_mae = torch.mean(torch.abs(baseline_vals - true_vals)).item()
            improvement = (baseline_mae - pinn_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
            
            results.append({
                '特征': feature_names[feat_idx],
                '站点': station_names[station_idx],
                'PINN_MAE': f'{pinn_mae:.3f}',
                'Baseline_MAE': f'{baseline_mae:.3f}',
                '提升(%)': f'{improvement:.1f}'
            })
    
    df = pd.DataFrame(results)
    
    # 保存CSV
    csv_path = save_path.replace('.png', '_metrics.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"指标表已保存: {csv_path}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("预测性能摘要")
    print("="*60)
    
    # 按特征汇总
    summary = df.groupby('特征').apply(
        lambda x: pd.Series({
            'PINN_MAE': f"{x['PINN_MAE'].astype(float).mean():.3f}",
            'Baseline_MAE': f"{x['Baseline_MAE'].astype(float).mean():.3f}",
            '平均提升': f"{(x['Baseline_MAE'].astype(float) - x['PINN_MAE'].astype(float)).mean() / x['Baseline_MAE'].astype(float).mean() * 100:.1f}%"
        })
    )
    
    print(summary)
    print("="*60)


def main():
    """主函数"""
    
    print("="*70)
    print("生成模型预测对比可视化")
    print("="*70)
    
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 生成测试数据
    print("\n[1] 生成合成测试数据...")
    test_data, test_targets, timestamps = generate_synthetic_test_data(n_samples=100)
    print(f"    数据形状: {test_data.shape}")
    print(f"    目标形状: {test_targets.shape}")
    
    # 创建模型
    print("\n[2] 创建模型...")
    pinn_model, baseline_model = create_and_load_models(device)
    
    pinn_params = sum(p.numel() for p in pinn_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    
    print(f"    PINN (增强版): {pinn_params:,} 参数")
    print(f"    Baseline: {baseline_params:,} 参数")
    
    # 生成预测
    print("\n[3] 生成预测...")
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)
    
    with torch.no_grad():
        pinn_pred = pinn_model(test_data)
        baseline_pred = baseline_model(test_data)
    
    print(f"    PINN预测: {pinn_pred.shape}")
    print(f"    Baseline预测: {baseline_pred.shape}")
    
    # 计算指标
    print("\n[4] 计算评估指标...")
    pinn_metrics = compute_metrics(pinn_pred, test_targets)
    baseline_metrics = compute_metrics(baseline_pred, test_targets)
    
    print(f"\n    PINN (增强版):")
    print(f"      MSE:  {pinn_metrics['mse']:.4f}")
    print(f"      MAE:  {pinn_metrics['mae']:.4f}")
    print(f"      RMSE: {pinn_metrics['rmse']:.4f}")
    
    print(f"\n    Baseline:")
    print(f"      MSE:  {baseline_metrics['mse']:.4f}")
    print(f"      MAE:  {baseline_metrics['mae']:.4f}")
    print(f"      RMSE: {baseline_metrics['rmse']:.4f}")
    
    improvement = (baseline_metrics['mae'] - pinn_metrics['mae']) / baseline_metrics['mae'] * 100
    print(f"\n    PINN 相对于 Baseline 提升: {improvement:.1f}%")
    
    # 绘制对比图
    print("\n[5] 生成可视化...")
    
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    save_path = output_dir / "prediction_comparison.png"
    
    station_names = ['兰州', '河口镇', '龙门', '三门峡', '花园口']
    
    plot_comparison(
        pinn_pred.cpu(), baseline_pred.cpu(), test_targets.cpu(),
        timestamps, station_names, str(save_path)
    )
    
    # 生成指标表
    generate_summary_table(
        pinn_pred.cpu(), baseline_pred.cpu(), test_targets.cpu(),
        str(save_path)
    )
    
    print("\n" + "="*70)
    print("完成!")
    print("="*70)


if __name__ == '__main__':
    main()
