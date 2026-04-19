"""PINN 模型训练脚本 for Yellow River Digital Twin.

训练流程：
1. 加载预处理后的数据
2. 划分训练/验证集
3. 初始化 PINN 模型和优化器
4. 交替优化 Data Loss 和 Physics Loss
5. 保存最佳模型和训练日志

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --epochs 200 --lambda-phys 0.5 --lr 0.001
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.pinn_model import YellowRiverPINN, compute_total_loss


def set_seed(seed: int = 42):
    """设置随机种子保证可复现性。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(
    data_path: str,
    input_features: list = None,
    target_features: list = None,
    test_ratio: float = 0.2
) -> Tuple[torch.Tensor, ...]:
    """加载并划分数据集。
    
    Args:
        data_path: 数据文件路径 (CSV)
        input_features: 输入特征列名列表
        target_features: 目标特征列名列表 (water_level, sediment)
        test_ratio: 验证集比例
    
    Returns:
        x_train, y_train, x_val, y_val: 划分后的张量
    """
    # 默认特征配置
    if input_features is None:
        input_features = ['time', 'location', 'flow_rate', 'rainfall', 'temperature']
    if target_features is None:
        target_features = ['water_level', 'sediment_content']
    
    # 加载数据
    df = pd.read_csv(data_path)
    print(f"[Data] 加载数据: {data_path}, 样本数: {len(df)}")
    
    # 提取特征
    X = df[input_features].values
    y = df[target_features].values
    
    # 数据标准化 (可选，这里使用简单的归一化)
    x_mean, x_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    
    # 避免除零
    x_std = np.where(x_std == 0, 1, x_std)
    y_std = np.where(y_std == 0, 1, y_std)
    
    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std
    
    # 保存归一化参数
    norm_params = {
        'x_mean': x_mean.tolist(),
        'x_std': x_std.tolist(),
        'y_mean': y_mean.tolist(),
        'y_std': y_std.tolist(),
        'input_features': input_features,
        'target_features': target_features
    }
    
    # 划分训练/验证集 (时间序列数据应该按时间划分，这里简单随机划分)
    n_samples = len(X)
    n_val = int(n_samples * test_ratio)
    
    # 随机打乱
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    # 转换为张量
    x_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    x_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    print(f"[Data] 训练集: {len(x_train)}, 验证集: {len(x_val)}")
    
    return x_train, y_train, x_val, y_val, norm_params


def create_dataloaders(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """创建 DataLoader。"""
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader


def train_epoch(
    model: YellowRiverPINN,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_phys: float,
    alpha: float,
    device: torch.device
) -> Dict[str, float]:
    """训练一个 epoch。
    
    Returns:
        包含 avg_total_loss, avg_data_loss, avg_physics_loss 的字典
    """
    model.train()
    
    total_losses = []
    data_losses = []
    physics_losses = []
    
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        # 计算总损失 (Data Loss + Physics Loss)
        total_loss, logs = compute_total_loss(
            model, x_batch, y_batch,
            x_physics=x_batch,  # 使用相同 batch 计算物理损失
            lambda_phys=lambda_phys,
            alpha=alpha
        )
        
        total_loss.backward()
        optimizer.step()
        
        total_losses.append(logs['total_loss'].item())
        data_losses.append(logs['data_loss'].item())
        physics_losses.append(logs['physics_loss'].item())
    
    return {
        'avg_total_loss': np.mean(total_losses),
        'avg_data_loss': np.mean(data_losses),
        'avg_physics_loss': np.mean(physics_losses)
    }


def validate(
    model: YellowRiverPINN,
    val_loader: DataLoader,
    lambda_phys: float,
    alpha: float,
    device: torch.device
) -> Dict[str, float]:
    """验证模型。
    
    注意：验证时仍需要计算物理损失（需要梯度），因此不使用 torch.no_grad()。
    物理约束是PINN的核心，验证时也需要监控物理一致性。
    """
    model.eval()
    
    total_losses = []
    data_losses = []
    physics_losses = []
    
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # 验证阶段仍计算物理损失以监控物理一致性
        # 但不进行反向传播
        total_loss, logs = compute_total_loss(
            model, x_batch, y_batch,
            x_physics=x_batch,
            lambda_phys=lambda_phys,
            alpha=alpha
        )
        
        total_losses.append(logs['total_loss'].item())
        data_losses.append(logs['data_loss'].item())
        physics_losses.append(logs['physics_loss'].item())
    
    return {
        'val_total_loss': np.mean(total_losses),
        'val_data_loss': np.mean(data_losses),
        'val_physics_loss': np.mean(physics_losses)
    }


def train(
    model: YellowRiverPINN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epochs: int,
    lambda_phys: float,
    alpha: float,
    device: torch.device,
    save_dir: str,
    early_stop_patience: int = 20
) -> Dict[str, list]:
    """完整训练流程。
    
    Returns:
        history: 训练历史记录
    """
    history = {
        'epoch': [],
        'train_total_loss': [],
        'train_data_loss': [],
        'train_physics_loss': [],
        'val_total_loss': [],
        'val_data_loss': [],
        'val_physics_loss': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n[Training] 开始训练: {epochs} epochs")
    print(f"[Training] Device: {device}")
    print(f"[Training] Lambda_phys: {lambda_phys}, Alpha: {alpha}")
    print("=" * 60)
    
    for epoch in range(epochs):
        # 训练
        train_logs = train_epoch(
            model, train_loader, optimizer,
            lambda_phys, alpha, device
        )
        
        # 验证
        val_logs = validate(
            model, val_loader,
            lambda_phys, alpha, device
        )
        
        # 学习率调整
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step(val_logs['val_total_loss'])
        
        # 记录历史
        history['epoch'].append(epoch + 1)
        history['train_total_loss'].append(train_logs['avg_total_loss'])
        history['train_data_loss'].append(train_logs['avg_data_loss'])
        history['train_physics_loss'].append(train_logs['avg_physics_loss'])
        history['val_total_loss'].append(val_logs['val_total_loss'])
        history['val_data_loss'].append(val_logs['val_data_loss'])
        history['val_physics_loss'].append(val_logs['val_physics_loss'])
        history['lr'].append(current_lr)
        
        # 每 10 个 epoch 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | "
                  f"Train Loss: {train_logs['avg_total_loss']:.6f} | "
                  f"Val Loss: {val_logs['val_total_loss']:.6f} | "
                  f"LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_logs['val_total_loss'] < best_val_loss:
            best_val_loss = val_logs['val_total_loss']
            patience_counter = 0
            
            # 保存模型
            model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'lambda_phys': lambda_phys,
                'alpha': alpha
            }, model_path)
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= early_stop_patience:
            print(f"\n[Early Stop] 验证损失 {early_stop_patience} 轮未改善，提前停止")
            break
    
    print(f"\n[Training] 训练完成. 最佳验证损失: {best_val_loss:.6f}")
    
    return history


def plot_training_history(history: Dict[str, list], save_path: str):
    """绘制训练曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = history['epoch']
    
    # 总损失
    axes[0, 0].plot(epochs, history['train_total_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_total_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss (Data + Physics)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 数据损失
    axes[0, 1].plot(epochs, history['train_data_loss'], label='Train')
    axes[0, 1].plot(epochs, history['val_data_loss'], label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Data Loss (MSE)')
    axes[0, 1].set_title('Data Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 物理损失
    axes[1, 0].plot(epochs, history['train_physics_loss'], label='Train')
    axes[1, 0].plot(epochs, history['val_physics_loss'], label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Physics Loss (PDE Residual)')
    axes[1, 0].set_title('Physics Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 学习率
    axes[1, 1].plot(epochs, history['lr'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] 训练曲线保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Yellow River PINN Model')
    parser.add_argument('--data-path', type=str, default='data/processed/train.csv',
                        help='训练数据路径')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--lambda-phys', type=float, default=1.0,
                        help='物理损失权重系数')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='PDE方程比例系数')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='隐藏层维度')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='隐藏层数量')
    parser.add_argument('--activation', type=str, default='swish',
                        choices=['swish', 'tanh'],
                        help='激活函数类型')
    parser.add_argument('--save-dir', type=str, default='models/checkpoints',
                        help='模型保存目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--early-stop', type=int, default=20,
                        help='早停耐心值')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Yellow River PINN - Training")
    print("=" * 60)
    print(f"Config: {vars(args)}")
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"\n[Warning] 数据文件不存在: {args.data_path}")
        print("[Info] 生成示例数据进行演示训练...")
        
        # 生成示例数据
        from models.pinn_model import YellowRiverPINN as _YP
        x_demo = torch.randn(1000, 5)
        model_demo = _YP(input_dim=5, output_dim=2)
        with torch.no_grad():
            y_demo = model_demo(x_demo) + 0.1 * torch.randn(1000, 2)
        
        # 保存示例数据
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        demo_df = pd.DataFrame(
            torch.cat([x_demo, y_demo], dim=1).numpy(),
            columns=['time', 'location', 'flow_rate', 'rainfall', 'temperature',
                     'water_level', 'sediment_content']
        )
        demo_df.to_csv(args.data_path, index=False)
        print(f"[Info] 示例数据已保存: {args.data_path}")
    
    # 加载数据
    x_train, y_train, x_val, y_val, norm_params = load_data(
        args.data_path,
        test_ratio=args.test_ratio
    )
    
    # 保存归一化参数
    with open(os.path.join(save_dir, 'norm_params.json'), 'w') as f:
        json.dump(norm_params, f, indent=2)
    
    # 创建 DataLoader
    train_loader, val_loader = create_dataloaders(
        x_train, y_train, x_val, y_val,
        batch_size=args.batch_size
    )
    
    # 初始化模型
    model = YellowRiverPINN(
        input_dim=5,
        hidden_dim=args.hidden_dim,
        output_dim=2,
        num_hidden_layers=args.num_layers,
        activation=args.activation
    ).to(device)
    
    print(f"\n[Model] PINN 参数总量: {sum(p.numel() for p in model.parameters())}")
    
    # 优化器和学习率调度
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 训练
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        lambda_phys=args.lambda_phys,
        alpha=args.alpha,
        device=device,
        save_dir=save_dir,
        early_stop_patience=args.early_stop
    )
    
    # 保存训练历史
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plot_training_history(history, os.path.join(save_dir, 'training_curve.png'))
    
    # 保存配置
    config_dict = vars(args)
    config_dict['timestamp'] = timestamp
    config_dict['device'] = str(device)
    config_dict['total_params'] = sum(p.numel() for p in model.parameters())
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Training Complete! Model saved to: {save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
