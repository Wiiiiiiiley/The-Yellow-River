"""PINN v2.0 Training Script - Research Grade.

Features:
1. Two-stage training: Adam (fast descent) + L-BFGS (fine-tuning)
2. Adaptive loss weighting with learnable uncertainty
3. Gradient clipping for stable momentum equation training
4. Comprehensive logging and checkpointing

Usage:
    python scripts/train_v2.py --epochs 200 --lbfgs-epochs 50
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.pinn_model_v2 import YellowRiverPINNv2, AdaptiveLossWeights, compute_total_loss_v2


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip gradients by norm.
    
    Returns:
        Total gradient norm before clipping
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    
    return total_norm


def load_data(
    data_path: str,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[torch.Tensor, ...]:
    """Load and split dataset.
    
    Returns:
        x_train, y_train, x_val, y_val, norm_params
    """
    df = pd.read_csv(data_path)
    print(f"[Data] Loaded: {data_path}, samples: {len(df)}")
    
    # Feature configuration
    input_features = ['time', 'location', 'flow_rate', 'rainfall', 'temperature']
    target_features = ['water_level', 'sediment_content']
    
    # Check available columns
    available_inputs = [f for f in input_features if f in df.columns]
    available_targets = [f for f in target_features if f in df.columns]
    
    if len(available_inputs) < 3:
        raise ValueError(f"Need at least 3 input features, found: {available_inputs}")
    
    X = df[available_inputs].values
    y = df[available_targets].values if available_targets else np.zeros((len(df), 2))
    
    # Normalization
    x_mean, x_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    x_std = np.where(x_std == 0, 1, x_std)
    y_std = np.where(y_std == 0, 1, y_std)
    
    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std
    
    norm_params = {
        'x_mean': x_mean.tolist(),
        'x_std': x_std.tolist(),
        'y_mean': y_mean.tolist(),
        'y_std': y_std.tolist(),
        'input_features': available_inputs,
        'target_features': available_targets
    }
    
    # Train/val split
    np.random.seed(seed)
    n_samples = len(X)
    n_val = int(n_samples * test_ratio)
    
    indices = np.random.permutation(n_samples)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    
    x_train = torch.FloatTensor(X[train_idx])
    y_train = torch.FloatTensor(y[train_idx])
    x_val = torch.FloatTensor(X[val_idx])
    y_val = torch.FloatTensor(y[val_idx])
    
    print(f"[Data] Train: {len(x_train)}, Val: {len(x_val)}")
    print(f"[Data] Input dim: {x_train.shape[1]}, Output dim: {y_train.shape[1]}")
    
    return x_train, y_train, x_val, y_val, norm_params


def train_epoch_adam(
    model: YellowRiverPINNv2,
    adaptive_weights: AdaptiveLossWeights,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
    g: float, n: float, S0: float, B: float,
    device: torch.device
) -> Dict[str, float]:
    """One epoch of Adam training with gradient clipping.
    
    Returns:
        Dictionary with averaged metrics (3-loss structure)
    """
    model.train()
    adaptive_weights.train()
    
    metrics = {
        'total_loss': [], 'data_loss': [], 'continuity_loss': [], 'momentum_loss': [],
        'lambda_data': [], 'lambda_continuity': [], 'lambda_momentum': [],
        'log_var_data': [], 'log_var_continuity': [], 'log_var_momentum': [],
        'grad_norm': []
    }
    
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        total_loss, logs = compute_total_loss_v2(
            model, adaptive_weights, x_batch, y_batch,
            x_physics=x_batch, g=g, n=n, S0=S0, B=B,
            max_grad_norm=max_grad_norm
        )
        
        total_loss.backward()
        
        # Gradient clipping with torch.nn.utils.clip_grad_norm_
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(adaptive_weights.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Record metrics
        metrics['total_loss'].append(logs['total_loss'].item())
        metrics['data_loss'].append(logs['data_loss'].item())
        metrics['continuity_loss'].append(logs['continuity_loss'].item())
        metrics['momentum_loss'].append(logs['momentum_loss'].item())
        metrics['lambda_data'].append(logs['lambda_data'].item())
        metrics['lambda_continuity'].append(logs['lambda_continuity'].item())
        metrics['lambda_momentum'].append(logs['lambda_momentum'].item())
        metrics['log_var_data'].append(logs['log_var_data'].item())
        metrics['log_var_continuity'].append(logs['log_var_continuity'].item())
        metrics['log_var_momentum'].append(logs['log_var_momentum'].item())
        metrics['grad_norm'].append(grad_norm.item())
    
    return {k: np.mean(v) for k, v in metrics.items()}


def train_epoch_lbfgs(
    model: YellowRiverPINNv2,
    adaptive_weights: AdaptiveLossWeights,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    optimizer: torch.optim.LBFGS,
    max_grad_norm: float,
    g: float, n: float, S0: float, B: float
) -> Dict[str, float]:
    """One epoch of L-BFGS full-batch training.
    
    L-BFGS operates on full batch and requires a closure function.
    """
    model.train()
    adaptive_weights.train()
    
    def closure():
        optimizer.zero_grad()
        
        total_loss, logs = compute_total_loss_v2(
            model, adaptive_weights, x_train, y_train,
            x_physics=x_train, g=g, n=n, S0=S0, B=B,
            max_grad_norm=max_grad_norm
        )
        
        total_loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(adaptive_weights.parameters(), max_grad_norm)
        
        return total_loss
    
    # L-BFGS step
    loss = optimizer.step(closure)
    
    # Evaluate after step
    with torch.no_grad():
        _, logs = compute_total_loss_v2(
            model, adaptive_weights, x_train, y_train,
            x_physics=x_train, g=g, n=n, S0=S0, B=B,
            max_grad_norm=None
        )
        
        metrics = {
            'total_loss': logs['total_loss'].item(),
            'data_loss': logs['data_loss'].item(),
            'continuity_loss': logs['continuity_loss'].item(),
            'momentum_loss': logs['momentum_loss'].item(),
            'lambda_data': logs['lambda_data'].item(),
            'lambda_continuity': logs['lambda_continuity'].item(),
            'lambda_momentum': logs['lambda_momentum'].item(),
            'log_var_data': logs['log_var_data'].item(),
            'log_var_continuity': logs['log_var_continuity'].item(),
            'log_var_momentum': logs['log_var_momentum'].item(),
            'grad_norm': 0.0  # Not tracked for L-BFGS
        }
    
    return metrics


def validate(
    model: YellowRiverPINNv2,
    adaptive_weights: AdaptiveLossWeights,
    val_loader: DataLoader,
    g: float, n: float, S0: float, B: float,
    device: torch.device
) -> Dict[str, float]:
    """Validation (physics loss requires gradients for monitoring)."""
    model.eval()
    adaptive_weights.eval()
    
    metrics = {
        'total_loss': [], 'data_loss': [], 'continuity_loss': [], 'momentum_loss': []
    }
    
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        _, logs = compute_total_loss_v2(
            model, adaptive_weights, x_batch, y_batch,
            x_physics=x_batch, g=g, n=n, S0=S0, B=B,
            max_grad_norm=None
        )
        
        metrics['total_loss'].append(logs['total_loss'].item())
        metrics['data_loss'].append(logs['data_loss'].item())
        metrics['continuity_loss'].append(logs['continuity_loss'].item())
        metrics['momentum_loss'].append(logs['momentum_loss'].item())
    
    return {k: np.mean(v) for k, v in metrics.items()}


def plot_sota_dashboard(history: Dict[str, list], save_path: str = 'results/sota_dashboard.png'):
    """Generate SOTA-grade 6-panel research dashboard.
    
    Layout (2x3):
        1. Total Loss (log scale)
        2. Data vs Physics
        3. PDE Residuals (Continuity + Momentum)
        4. Adaptive Weights (λ_data, λ_cont, λ_mom)
        5. Uncertainty (log-variances)
        6. Gradient Norm
    """
    # Ensure results directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Yellow River PINN v2.0 - SOTA Research Dashboard', fontsize=16, fontweight='bold')
    
    epochs = history.get('epoch', list(range(1, len(history['train_total_loss']) + 1)))
    stages = history.get('stage', ['Adam'] * len(epochs))
    
    # 1. Total Loss (log scale)
    ax = axes[0, 0]
    ax.semilogy(epochs, history['train_total_loss'], 'b-', label='Train', linewidth=2)
    ax.semilogy(epochs, history['val_total_loss'], 'r--', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.set_title('1. Total Loss (Adaptive Weighted)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Mark L-BFGS stage
    lbfgs_start = next((i for i, s in enumerate(stages) if s == 'L-BFGS'), len(stages))
    if lbfgs_start < len(stages):
        ax.axvline(x=epochs[lbfgs_start], color='g', linestyle=':', label='L-BFGS start')
    
    # 2. Data vs Physics Loss
    ax = axes[0, 1]
    ax.semilogy(epochs, history['train_data_loss'], 'b-', label='Data', linewidth=2)
    physics_total = [d + c + m for d, c, m in zip(
        history['train_data_loss'], 
        history['train_continuity_loss'], 
        history['train_momentum_loss']
    )]
    ax.semilogy(epochs, physics_total, 'r--', label='Total Physics', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.set_title('2. Data Loss vs Physics Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. PDE Residuals (Continuity + Momentum)
    ax = axes[0, 2]
    ax.semilogy(epochs, history['train_continuity_loss'], 'g-', label='Continuity', linewidth=2)
    ax.semilogy(epochs, history['train_momentum_loss'], 'm-', label='Momentum', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Residual (log)')
    ax.set_title('3. PDE Residuals (Saint-Venant)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Adaptive Weights (λ)
    ax = axes[1, 0]
    ax.plot(epochs, history['lambda_data'], 'b-', label='λ_data', linewidth=2)
    ax.plot(epochs, history['lambda_continuity'], 'g-', label='λ_continuity', linewidth=2)
    ax.plot(epochs, history['lambda_momentum'], 'm-', label='λ_momentum', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight λ')
    ax.set_title('4. Adaptive Weights (Auto-balanced)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Uncertainty (log-variances)
    ax = axes[1, 1]
    ax.plot(epochs, history['log_var_data'], 'b-', label='log(σ²_data)', linewidth=2)
    ax.plot(epochs, history['log_var_continuity'], 'g-', label='log(σ²_cont)', linewidth=2)
    ax.plot(epochs, history['log_var_momentum'], 'm-', label='log(σ²_mom)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log Variance')
    ax.set_title('5. Learned Uncertainties')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Gradient Norm
    ax = axes[1, 2]
    ax.semilogy(epochs, history['grad_norm'], 'k-', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Clip Threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm (log)')
    ax.set_title('6. Gradient Norm (Numerical Stability)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[Dashboard] SOTA dashboard saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train PINN v2.0 (Research Grade)')
    parser.add_argument('--data-path', type=str, default='data/processed/train.csv')
    parser.add_argument('--epochs', type=int, default=200, help='Adam training epochs')
    parser.add_argument('--lbfgs-epochs', type=int, default=50, help='L-BFGS fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001, help='Adam learning rate')
    parser.add_argument('--lbfgs-lr', type=float, default=0.1, help='L-BFGS learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-res-blocks', type=int, default=4)
    parser.add_argument('--use-attention', type=bool, default=True)
    parser.add_argument('--g', type=float, default=9.8, help='Gravity (m/s²)')
    parser.add_argument('--n', type=float, default=0.03, help='Manning roughness')
    parser.add_argument('--S0', type=float, default=0.001, help='Riverbed slope')
    parser.add_argument('--B', type=float, default=50.0, help='Channel width (m)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Gradient clipping')
    
    # Sensitivity analysis arguments
    parser.add_argument('--run-sensitivity', action='store_true',
                        help='Run sensitivity analysis after training')
    parser.add_argument('--S0-range', type=str, default='0.0005,0.002,20',
                        help='S0 sensitivity range: min,max,steps')
    parser.add_argument('--B-range', type=str, default='30,70,20',
                        help='B sensitivity range: min,max,steps')
    parser.add_argument('--save-dir', type=str, default='models/pinn_v2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--early-stop', type=int, default=30)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("Yellow River PINN v2.0 - Research Grade Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Config: {vars(args)}")
    
    # Load data
    if not os.path.exists(args.data_path):
        print(f"\n[Warning] Data not found, generating demo data...")
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        from scripts.pinn_model_v2 import YellowRiverPINNv2 as YP2
        x_demo = torch.randn(1000, 5)
        m_demo = YP2(5, 2)
        with torch.no_grad():
            y_demo = m_demo(x_demo) + 0.1 * torch.randn(1000, 2)
        demo_df = pd.DataFrame(
            torch.cat([x_demo, y_demo], dim=1).numpy(),
            columns=['time', 'location', 'flow_rate', 'rainfall', 'temperature',
                     'water_level', 'sediment_content']
        )
        demo_df.to_csv(args.data_path, index=False)
    
    x_train, y_train, x_val, y_val, norm_params = load_data(
        args.data_path, args.test_ratio, args.seed
    )
    
    with open(os.path.join(save_dir, 'norm_params.json'), 'w') as f:
        json.dump(norm_params, f, indent=2)
    
    # DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model and adaptive weights
    model = YellowRiverPINNv2(
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=y_train.shape[1],
        num_res_blocks=args.num_res_blocks,
        use_attention=args.use_attention
    ).to(device)
    
    adaptive_weights = AdaptiveLossWeights(num_losses=3).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in adaptive_weights.parameters())
    print(f"\n[Model] Total parameters: {total_params:,}")
    print(f"[Physics] g={args.g}, n={args.n}, S0={args.S0}, B={args.B}")
    
    # Move data to device
    x_train_device = x_train.to(device)
    y_train_device = y_train.to(device)
    
    # Training history (3-loss structure)
    history = {
        'epoch': [], 'stage': [],
        'train_total_loss': [], 'train_data_loss': [],
        'train_continuity_loss': [], 'train_momentum_loss': [],
        'val_total_loss': [], 'val_data_loss': [],
        'val_continuity_loss': [], 'val_momentum_loss': [],
        'lambda_data': [], 'lambda_continuity': [], 'lambda_momentum': [],
        'log_var_data': [], 'log_var_continuity': [], 'log_var_momentum': [],
        'grad_norm': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # ========== Stage 1: Adam Optimization ==========
    print("\n" + "=" * 70)
    print(f"Stage 1: Adam Optimization ({args.epochs} epochs)")
    print("=" * 70)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(adaptive_weights.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch_adam(
            model, adaptive_weights, train_loader, optimizer,
            args.max_grad_norm, args.g, args.n, args.S0, args.B, device
        )
        val_metrics = validate(model, adaptive_weights, val_loader, args.g, args.n, args.S0, args.B, device)
        
        scheduler.step(val_metrics['total_loss'])
        
        # Record 3-loss history
        history['epoch'].append(epoch + 1)
        history['stage'].append('Adam')
        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_data_loss'].append(train_metrics['data_loss'])
        history['train_continuity_loss'].append(train_metrics['continuity_loss'])
        history['train_momentum_loss'].append(train_metrics['momentum_loss'])
        history['val_total_loss'].append(val_metrics['total_loss'])
        history['val_data_loss'].append(val_metrics['data_loss'])
        history['val_continuity_loss'].append(val_metrics['continuity_loss'])
        history['val_momentum_loss'].append(val_metrics['momentum_loss'])
        history['lambda_data'].append(train_metrics['lambda_data'])
        history['lambda_continuity'].append(train_metrics['lambda_continuity'])
        history['lambda_momentum'].append(train_metrics['lambda_momentum'])
        history['log_var_data'].append(train_metrics['log_var_data'])
        history['log_var_continuity'].append(train_metrics['log_var_continuity'])
        history['log_var_momentum'].append(train_metrics['log_var_momentum'])
        history['grad_norm'].append(train_metrics['grad_norm'])
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | "
                  f"Loss: {train_metrics['total_loss']:.6f} | "
                  f"Val: {val_metrics['total_loss']:.6f} | "
                  f"λ: [{train_metrics['lambda_data']:.3f}, {train_metrics['lambda_continuity']:.3f}, {train_metrics['lambda_momentum']:.3f}]")
        
        # Save best
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'adaptive_weights_state_dict': adaptive_weights.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': vars(args)
            }, os.path.join(save_dir, 'best_model_adam.pt'))
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"\n[Early Stop] No improvement for {args.early_stop} epochs")
                break
    
    # ========== Stage 2: L-BFGS Fine-tuning ==========
    if args.lbfgs_epochs > 0:
        print("\n" + "=" * 70)
        print(f"Stage 2: L-BFGS Fine-tuning ({args.lbfgs_epochs} epochs)")
        print("=" * 70)
        
        lbfgs_optimizer = torch.optim.LBFGS(
            list(model.parameters()) + list(adaptive_weights.parameters()),
            lr=args.lbfgs_lr,
            max_iter=20,
            history_size=50,
            line_search_fn='strong_wolfe'
        )
        
        for epoch in range(args.lbfgs_epochs):
            train_metrics = train_epoch_lbfgs(
                model, adaptive_weights, x_train_device, y_train_device,
                lbfgs_optimizer, args.max_grad_norm, args.g, args.n, args.S0, args.B
            )
            val_metrics = validate(model, adaptive_weights, val_loader, args.g, args.n, args.S0, args.B, device)
            
            # Record 3-loss history
            history['epoch'].append(len(history['epoch']) + 1)
            history['stage'].append('L-BFGS')
            history['train_total_loss'].append(train_metrics['total_loss'])
            history['train_data_loss'].append(train_metrics['data_loss'])
            history['train_continuity_loss'].append(train_metrics['continuity_loss'])
            history['train_momentum_loss'].append(train_metrics['momentum_loss'])
            history['val_total_loss'].append(val_metrics['total_loss'])
            history['val_data_loss'].append(val_metrics['data_loss'])
            history['val_continuity_loss'].append(val_metrics['continuity_loss'])
            history['val_momentum_loss'].append(val_metrics['momentum_loss'])
            history['lambda_data'].append(train_metrics['lambda_data'])
            history['lambda_continuity'].append(train_metrics['lambda_continuity'])
            history['lambda_momentum'].append(train_metrics['lambda_momentum'])
            history['log_var_data'].append(train_metrics['log_var_data'])
            history['log_var_continuity'].append(train_metrics['log_var_continuity'])
            history['log_var_momentum'].append(train_metrics['log_var_momentum'])
            history['grad_norm'].append(0.0)
            
            if (epoch + 1) % 10 == 0:
                print(f"L-BFGS Epoch {epoch+1:03d} | "
                      f"Loss: {train_metrics['total_loss']:.6f} | "
                      f"Val: {val_metrics['total_loss']:.6f}")
            
            # Save best L-BFGS model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                torch.save({
                    'epoch': len(history['epoch']),
                    'model_state_dict': model.state_dict(),
                    'adaptive_weights_state_dict': adaptive_weights.state_dict(),
                    'val_loss': best_val_loss,
                    'config': vars(args)
                }, os.path.join(save_dir, 'best_model_lbfgs.pt'))
    
    print(f"\n[Training] Best validation loss: {best_val_loss:.6f}")
    
    # Save final artifacts
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate SOTA dashboard in results/
    dashboard_path = os.path.join('results', 'sota_dashboard.png')
    plot_sota_dashboard(history, dashboard_path)
    
    # Run sensitivity analysis if requested
    if args.run_sensitivity:
        print("\n" + "=" * 70)
        print("Running Physical Parameter Sensitivity Analysis")
        print("=" * 70)
        
        # Parse perturbation ranges
        S0_parts = [float(x) for x in args.S0_range.split(',')]
        B_parts = [float(x) for x in args.B_range.split(',')]
        
        perturbation_range = {}
        if len(S0_parts) == 3:
            perturbation_range['S0'] = (S0_parts[0], S0_parts[1], int(S0_parts[2]))
        if len(B_parts) == 3:
            perturbation_range['B'] = (B_parts[0], B_parts[1], int(B_parts[2]))
        
        base_params = {
            'g': args.g,
            'n': args.n,
            'S0': args.S0,
            'B': args.B
        }
        
        # Prepare test data for sensitivity analysis
        X_test = x_val[:100].to(device)  # Use subset of validation data
        
        sensitivity_results = run_sensitivity_analysis(
            model=model,
            adaptive_weights=adaptive_weights,
            X_test=X_test,
            base_params=base_params,
            perturbation_range=perturbation_range,
            save_path=os.path.join(save_dir, 'sensitivity_analysis.png')
        )
        
        # Save sensitivity results
        with open(os.path.join(save_dir, 'sensitivity_results.json'), 'w') as f:
            json.dump({
                'S0_sensitivity_index': sensitivity_results.get('S0_sensitivity_index', None),
                'B_sensitivity_index': sensitivity_results.get('B_sensitivity_index', None)
            }, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Training Complete! Saved to: {save_dir}")
    print(f"Dashboard: {dashboard_path}")
    if args.run_sensitivity:
        print(f"Sensitivity: {os.path.join(save_dir, 'sensitivity_analysis.png')}")
    print("=" * 70)


def run_sensitivity_analysis(
    model: YellowRiverPINNv2,
    adaptive_weights: AdaptiveLossWeights,
    X_test: torch.Tensor,
    base_params: Dict[str, float],
    perturbation_range: Dict[str, Tuple[float, float, int]],
    save_path: str = 'results/sensitivity_analysis.png'
) -> Dict:
    """Run sensitivity analysis on physical parameters (S0 and B).
    
    This module analyzes how small perturbations in physical parameters
    (riverbed slope S0 and channel width B) affect model predictions.
    High sensitivity indicates regions where physical constraints are
    particularly influential - a key insight for reliability assessment.
    
    Args:
        model: Trained PINN model
        adaptive_weights: Adaptive loss weights module
        X_test: Test input data
        base_params: Base physical parameters {g, n, S0, B}
        perturbation_range: Ranges for perturbation
            e.g., {'S0': (0.0005, 0.002, 20), 'B': (30, 70, 20)}
        save_path: Path to save sensitivity visualization
    
    Returns:
        Dictionary with sensitivity metrics
    
    Research Statement Value:
        "I developed a sensitivity analysis framework to quantify how
        riverbed geometry uncertainties propagate through the physics-
        informed model. This identifies operational regimes where the
        digital twin requires enhanced calibration."
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    adaptive_weights.eval()
    
    results = {}
    
    # Analyze S0 sensitivity
    if 'S0' in perturbation_range:
        S0_min, S0_max, n_steps = perturbation_range['S0']
        S0_values = np.linspace(S0_min, S0_max, n_steps)
        
        S0_predictions = []
        with torch.no_grad():
            for S0_val in S0_values:
                _, logs = compute_total_loss_v2(
                    model, adaptive_weights, X_test, torch.zeros_like(X_test[:, :2]),
                    x_physics=X_test,
                    g=base_params['g'],
                    n=base_params['n'],
                    S0=S0_val,
                    B=base_params['B'],
                    max_grad_norm=None
                )
                # Extract physics residuals
                S0_predictions.append({
                    'S0': S0_val,
                    'continuity_loss': logs['continuity_loss'].item(),
                    'momentum_loss': logs['momentum_loss'].item()
                })
        
        results['S0_sensitivity'] = S0_predictions
        
        # Calculate sensitivity index
        cont_losses = [p['continuity_loss'] for p in S0_predictions]
        mom_losses = [p['momentum_loss'] for p in S0_predictions]
        
        S0_sensitivity_index = np.std(cont_losses) + np.std(mom_losses)
        results['S0_sensitivity_index'] = S0_sensitivity_index
        
        # Warning if highly sensitive
        if S0_sensitivity_index > 1.0:
            print(f"⚠️  [Sensitivity] HIGH S0 sensitivity detected: {S0_sensitivity_index:.4f}")
            print("     Physical constraints in this regime are highly sensitive")
            print("     to riverbed slope variations.")
    
    # Analyze B sensitivity
    if 'B' in perturbation_range:
        B_min, B_max, n_steps = perturbation_range['B']
        B_values = np.linspace(B_min, B_max, n_steps)
        
        B_predictions = []
        with torch.no_grad():
            for B_val in B_values:
                _, logs = compute_total_loss_v2(
                    model, adaptive_weights, X_test, torch.zeros_like(X_test[:, :2]),
                    x_physics=X_test,
                    g=base_params['g'],
                    n=base_params['n'],
                    S0=base_params['S0'],
                    B=B_val,
                    max_grad_norm=None
                )
                B_predictions.append({
                    'B': B_val,
                    'continuity_loss': logs['continuity_loss'].item(),
                    'momentum_loss': logs['momentum_loss'].item()
                })
        
        results['B_sensitivity'] = B_predictions
        
        cont_losses = [p['continuity_loss'] for p in B_predictions]
        mom_losses = [p['momentum_loss'] for p in B_predictions]
        
        B_sensitivity_index = np.std(cont_losses) + np.std(mom_losses)
        results['B_sensitivity_index'] = B_sensitivity_index
        
        if B_sensitivity_index > 1.0:
            print(f"⚠️  [Sensitivity] HIGH B sensitivity detected: {B_sensitivity_index:.4f}")
            print("     Physical constraints highly sensitive to channel width.")
    
    # Generate sensitivity visualization
    if 'S0_sensitivity' in results or 'B_sensitivity' in results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Physical Parameter Sensitivity Analysis\n'
                     'Impact of Riverbed Geometry on Model Constraints',
                     fontsize=14, fontweight='bold')
        
        if 'S0_sensitivity' in results:
            ax = axes[0]
            data = results['S0_sensitivity']
            ax.plot([d['S0'] for d in data], [d['continuity_loss'] for d in data],
                   'b-o', label='Continuity Loss', linewidth=2)
            ax.plot([d['S0'] for d in data], [d['momentum_loss'] for d in data],
                   'r-s', label='Momentum Loss', linewidth=2)
            ax.set_xlabel('Riverbed Slope S0')
            ax.set_ylabel('Physics Loss')
            ax.set_title(f'S0 Sensitivity\n(Index: {results["S0_sensitivity_index"]:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        if 'B_sensitivity' in results:
            ax = axes[1]
            data = results['B_sensitivity']
            ax.plot([d['B'] for d in data], [d['continuity_loss'] for d in data],
                   'b-o', label='Continuity Loss', linewidth=2)
            ax.plot([d['B'] for d in data], [d['momentum_loss'] for d in data],
                   'r-s', label='Momentum Loss', linewidth=2)
            ax.set_xlabel('Channel Width B (m)')
            ax.set_ylabel('Physics Loss')
            ax.set_title(f'B Sensitivity\n(Index: {results["B_sensitivity_index"]:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Sensitivity] Analysis plot saved: {save_path}")
    
    return results


if __name__ == '__main__':
    main()
