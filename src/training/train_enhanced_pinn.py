"""
增强版 PINN 训练脚本 v2.0
演示所有改进特性的使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_flood_system.models.graph_transformer.stgnn import create_model
from smart_flood_system.models.pinn_loss.physics_constraints_enhanced import (
    EnhancedPhysicsConstraintLoss,
    BoundaryConditionEnforcer,
    MultiFidelityPINN,
    SedimentTransportLoss
)
from smart_flood_system.models.baseline.baseline_models import create_baseline_model


def create_synthetic_data(n_samples=1000, seq_len=24, n_nodes=5, n_features=6):
    """创建合成训练数据 - 模拟黄河水文特征"""
    
    data = []
    targets = []
    
    for _ in range(n_samples):
        # 生成符合黄河特征的序列
        # 水位: 1000-1200m, 汛期偏高
        base_level = 1000 + np.random.rand() * 200
        
        # 流量: 1000-4000 m³/s, 洪水期可达 5000+
        base_discharge = 1500 + np.random.exponential(500)
        base_discharge = min(base_discharge, 5000)
        
        # 泥沙浓度: 与流量正相关 ("大水大沙")
        base_sediment = 5 + 0.003 * base_discharge + np.random.randn() * 2
        
        # 时间序列 (添加趋势和噪声)
        seq_data = []
        for t in range(seq_len):
            # 添加日变化趋势
            trend = np.sin(2 * np.pi * t / 24) * 50
            
            for node in range(n_nodes):
                features = [
                    base_level + trend + np.random.randn() * 10,      # 水位
                    base_discharge + trend * 2 + np.random.randn() * 100,  # 流量
                    max(0, base_sediment + np.random.randn() * 1),      # 泥沙
                    15 + np.random.randn() * 5,                        # 温度
                    max(0, np.random.exponential(2)),                  # 降雨
                    np.random.rand()                                   # 其他
                ]
                seq_data.append(features)
        
        data.append(seq_data)
        
        # 预测目标 (未来6小时)
        target = []
        for t in range(6):
            for node in range(n_nodes):
                target.append([
                    base_level + trend * 1.5 + np.random.randn() * 15,
                    base_discharge + trend * 3 + np.random.randn() * 150,
                    max(0, base_sediment * 1.2 + np.random.randn() * 2),
                    15 + np.random.randn() * 5,
                    max(0, np.random.exponential(2)),
                    np.random.rand()
                ])
        
        targets.append(target)
    
    # 转换为 numpy 并 reshape
    data = np.array(data).reshape(n_samples, seq_len, n_nodes, n_features)
    targets = np.array(targets).reshape(n_samples, 6, n_nodes, n_features)
    
    return torch.from_numpy(data).float(), torch.from_numpy(targets).float()


def train_enhanced_pinn():
    """训练增强版 PINN 模型"""
    
    print("=" * 70)
    print("增强版 PINN 训练演示 v2.0")
    print("=" * 70)
    
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    batch_size = 8
    seq_len = 24
    n_nodes = 5
    n_features = 6
    epochs = 100
    
    # 创建模型
    print("\n[1] 创建 GNN-Transformer 模型...")
    model_config = {
        'node_features': n_features,
        'n_nodes': n_nodes,
        'gnn_hidden': 64,
        'gnn_layers': 3,
        'transformer_heads': 8,
        'transformer_layers': 4,
        'sequence_length': seq_len,
        'prediction_horizon': 6,
        'dropout': 0.1
    }
    
    gnn_model = create_model(model_config).to(device)
    param_count = sum(p.numel() for p in gnn_model.parameters())
    print(f"    模型参数数量: {param_count:,}")
    
    # 创建增强版物理约束
    print("\n[2] 初始化增强版物理约束...")
    physics_loss = EnhancedPhysicsConstraintLoss(
        manning_coefficient=0.03,
        bed_slope=0.0001,
        gravity=9.81,
        river_width=500.0,
        use_ntk_adaptive=True,
        use_curriculum=True,
        use_high_order=True,
        use_normalization=True,
        use_sediment=True,
        total_epochs=epochs
    )
    print("    ✓ NTK自适应权重")
    print("    ✓ 课程学习")
    print("    ✓ 高阶数值差分 (4阶)")
    print("    ✓ 特征尺度归一化")
    print("    ✓ 泥沙输运物理 (Exner方程)")
    
    # 创建边界条件强制器
    print("\n[3] 初始化边界条件强制器...")
    boundary_enforcer = BoundaryConditionEnforcer(
        upstream_station_idx=0,    # 兰州
        downstream_station_idx=4   # 花园口
    )
    print("    上游边界: 兰州 (流量边界)")
    print("    下游边界: 花园口 (水位边界)")
    
    # 创建优化器
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # 生成训练数据
    print("\n[4] 生成合成训练数据...")
    train_data, train_targets = create_synthetic_data(n_samples=200)
    train_data = train_data.to(device)
    train_targets = train_targets.to(device)
    print(f"    训练样本: {len(train_data)}")
    
    # 训练循环
    print("\n[5] 开始训练...")
    print("-" * 70)
    
    gnn_model.train()
    
    for epoch in range(epochs):
        # 设置课程学习轮数
        physics_loss.set_epoch(epoch)
        
        # 随机批次
        batch_indices = np.random.choice(len(train_data), batch_size)
        batch_data = train_data[batch_indices]
        batch_targets = train_targets[batch_indices]
        
        # 前向传播
        predictions = gnn_model(batch_data)
        
        # 应用边界条件 (每5轮)
        if epoch % 5 == 0:
            upstream_Q = torch.tensor(1500.0 + np.random.rand() * 500).to(device)
            predictions = boundary_enforcer.apply_upstream_flow(predictions, upstream_Q)
            downstream_h = torch.tensor(1000.0 + np.random.rand() * 100).to(device)
            predictions = boundary_enforcer.apply_downstream_level(predictions, downstream_h)
        
        # 数据损失
        data_loss = F.mse_loss(predictions, batch_targets)
        
        # 物理约束损失
        total_loss, loss_dict = physics_loss(
            predictions,
            batch_targets,
            data_loss,
            model=gnn_model
        )
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # 打印进度
        if (epoch + 1) % 20 == 0:
            progress = (epoch + 1) / epochs
            physics_weight = physics_loss.get_curriculum_weight()
            
            print(f"\nEpoch [{epoch+1}/{epochs}] - 进度: {progress*100:.1f}%")
            print(f"  课程权重: {physics_weight:.3f}")
            print(f"  总损失: {loss_dict['total_loss']:.6f}")
            print(f"  数据损失: {loss_dict['data_loss']:.6f}")
            print(f"  物理损失: {loss_dict['physics_loss']:.6f}")
            print(f"  ├─ 连续性: {loss_dict['continuity_loss']:.6f}")
            print(f"  ├─ 动量: {loss_dict['momentum_loss']:.6f}")
            if 'exner_loss' in loss_dict:
                print(f"  ├─ 泥沙(Exner): {loss_dict['exner_loss']:.6f}")
            print(f"  权重: 数据={loss_dict['data_weight']:.3f}, 物理={loss_dict['physics_weight']:.3f}")
    
    print("\n" + "-" * 70)
    print("训练完成!")
    
    # 验证
    print("\n[6] 模型验证...")
    gnn_model.eval()
    with torch.no_grad():
        val_data, val_targets = create_synthetic_data(n_samples=20)
        val_data = val_data.to(device)
        val_targets = val_targets.to(device)
        
        val_predictions = gnn_model(val_data)
        val_loss = F.mse_loss(val_predictions, val_targets)
        
        # 计算物理残差
        h = val_predictions[:, :, :, 0]
        Q = val_predictions[:, :, :, 1]
        u = Q / (torch.clamp(h, min=0.1) * 500.0)
        
        continuity_check = physics_loss.compute_continuity_loss_enhanced(h, u, Q)
        momentum_check = physics_loss.compute_momentum_loss_enhanced(u, h)
        
        print(f"    验证 MSE: {val_loss.item():.6f}")
        print(f"    连续性残差: {continuity_check.item():.6f}")
        print(f"    动量残差: {momentum_check.item():.6f}")
    
    # 保存模型
    print("\n[7] 保存模型...")
    save_path = Path(__file__).parent.parent / "models" / "enhanced_pinn_v2.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': gnn_model.state_dict(),
        'config': model_config,
        'epoch': epochs,
        'val_loss': val_loss.item()
    }, save_path)
    print(f"    模型已保存: {save_path}")
    
    print("\n" + "=" * 70)
    print("所有任务完成!")
    print("=" * 70)
    
    return gnn_model, physics_loss


def compare_with_baseline():
    """对比增强版 PINN 与 Baseline 模型"""
    
    print("\n" + "=" * 70)
    print("PINN vs Baseline 对比实验")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    test_data, test_targets = create_synthetic_data(n_samples=50)
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)
    
    # 创建模型
    model_config = {
        'node_features': 6,
        'n_nodes': 5,
        'gnn_hidden': 64,
        'gnn_layers': 3,
        'transformer_heads': 8,
        'transformer_layers': 4,
        'sequence_length': 24,
        'prediction_horizon': 6,
        'dropout': 0.1
    }
    
    # PINN 模型
    pinn_model = create_model(model_config).to(device)
    
    # Baseline 模型
    baseline_model = create_baseline_model(
        model_type='ensemble',
        input_size=6,
        hidden_size=128,
        output_size=6,
        prediction_horizon=6
    ).to(device)
    
    # 快速训练
    print("\n快速训练对比模型...")
    optimizer_pinn = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)
    optimizer_base = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    
    physics_loss = EnhancedPhysicsConstraintLoss(use_sediment=False)
    
    for epoch in range(50):
        # PINN 训练
        pinn_model.train()
        pred_pinn = pinn_model(test_data[:20])
        loss_data = F.mse_loss(pred_pinn, test_targets[:20])
        loss_total, _ = physics_loss(pred_pinn, test_targets[:20], loss_data, model=pinn_model)
        
        optimizer_pinn.zero_grad()
        loss_total.backward()
        optimizer_pinn.step()
        
        # Baseline 训练
        baseline_model.train()
        pred_base = baseline_model(test_data[:20])
        loss_base = F.mse_loss(pred_base, test_targets[:20])
        
        optimizer_base.zero_grad()
        loss_base.backward()
        optimizer_base.step()
    
    # 验证对比
    print("\n验证对比结果:")
    print("-" * 70)
    
    with torch.no_grad():
        pinn_model.eval()
        baseline_model.eval()
        
        pred_pinn = pinn_model(test_data[20:])
        pred_base = baseline_model(test_data[20:])
        
        # MSE 对比
        mse_pinn = F.mse_loss(pred_pinn, test_targets[20:]).item()
        mse_base = F.mse_loss(pred_base, test_targets[20:]).item()
        
        print(f"\n均方误差 (MSE):")
        print(f"  PINN:     {mse_pinn:.6f}")
        print(f"  Baseline: {mse_base:.6f}")
        print(f"  提升:     {(mse_base - mse_pinn) / mse_base * 100:.2f}%")
        
        # 泥沙预测对比
        sediment_pinn = pred_pinn[:, :, :, 2].mean().item()
        sediment_base = pred_base[:, :, :, 2].mean().item()
        sediment_true = test_targets[20:][:, :, :, 2].mean().item()
        
        print(f"\n泥沙浓度预测 (kg/m³):")
        print(f"  真实值:   {sediment_true:.2f}")
        print(f"  PINN:     {sediment_pinn:.2f} (误差: {abs(sediment_pinn - sediment_true):.2f})")
        print(f"  Baseline: {sediment_base:.2f} (误差: {abs(sediment_base - sediment_true):.2f})")
        
        # 物理残差对比
        h_pinn = pred_pinn[:, :, :, 0]
        Q_pinn = pred_pinn[:, :, :, 1]
        u_pinn = Q_pinn / (torch.clamp(h_pinn, min=0.1) * 500.0)
        
        h_base = pred_base[:, :, :, 0]
        Q_base = pred_base[:, :, :, 1]
        u_base = Q_base / (torch.clamp(h_base, min=0.1) * 500.0)
        
        cont_pinn = physics_loss.compute_continuity_loss_enhanced(h_pinn, u_pinn, Q_pinn).item()
        cont_base = physics_loss.compute_continuity_loss_enhanced(h_base, u_base, Q_base).item()
        
        print(f"\n物理连续性残差:")
        print(f"  PINN:     {cont_pinn:.6f}")
        print(f"  Baseline: {cont_base:.6f}")
        print(f"  PINN 物理一致性: {'✓' if cont_pinn < cont_base else '✗'}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    # 运行增强版 PINN 训练
    model, physics = train_enhanced_pinn()
    
    # 对比实验
    compare_with_baseline()
