"""
Yellow River PINN Personality Training
======================================

Train PINN to capture the intrinsic hydrological character of Yellow River:
1. Seasonal Sin/Cos encoding for flood/dry season cycles
2. Non-linear sediment-flow feature crossing
3. Exner erosion equation + Saint-Venant physics
4. NTK-based adaptive weighting
5. Two-stage optimization (Adam + L-BFGS)

The Yellow River's personality:
- "Big water brings big sand" (大水大沙)
- Seasonal hysteresis: sediment peaks lag behind flow peaks
- Exner dynamics: erosion during floods, deposition during dry periods
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SeasonalEncoder:
    """
    Sin/Cos cyclic encoding for seasonal patterns.
    
    Captures:
    - Annual cycle: Flood season (Jul-Aug) vs Dry season (Dec-Feb)
    - Quarterly patterns: Pre-flood, flood, post-flood, dry
    """
    
    def __init__(self, base_year: int = 2014):
        self.base_year = base_year
        
    def encode(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Convert timestamps to cyclical features.
        
        Returns:
            (N, 4) array: [sin_year, cos_year, sin_season, cos_season]
        """
        # Day of year (0-365)
        day_of_year = timestamps.dayofyear.values
        
        # Annual cycle (period = 365 days)
        sin_year = np.sin(2 * np.pi * day_of_year / 365.25)
        cos_year = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Seasonal cycle (period = 90 days, quarterly)
        sin_season = np.sin(2 * np.pi * day_of_year / 90.0)
        cos_season = np.cos(2 * np.pi * day_of_year / 90.0)
        
        return np.column_stack([sin_year, cos_year, sin_season, cos_season])
    
    def encode_flood_season(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Binary encoding: 1 if in flood season (Jul-Aug), 0 otherwise.
        For use as attention weight or explicit feature.
        """
        months = timestamps.month.values
        is_flood = ((months >= 7) & (months <= 8)).astype(float)
        return is_flood.reshape(-1, 1)


class FeatureCrossing(nn.Module):
    """
    Explicit feature crossing for sediment-flow interactions.
    
    Captures the non-linear "big water brings big sand" relationship.
    """
    
    def __init__(self, feature_dim: int, num_crosses: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_crosses = num_crosses
        
        # Crossed features: Q×S, Q², S²
        self.cross_weights = nn.Parameter(torch.randn(num_crosses, 2))
        self.cross_proj = nn.Linear(num_crosses, feature_dim)
        
    def forward(self, Q: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Q: Flow rate (batch, 1)
            S: Sediment content (batch, 1)
        
        Returns:
            Crossed features (batch, feature_dim)
        """
        # Element-wise products
        Q_S = Q * S  # Key interaction: flow × sediment
        Q2 = Q ** 2  # Non-linear flow effect
        S2 = S ** 2  # Non-linear sediment effect
        
        crossed = torch.cat([Q_S, Q2, S2], dim=-1)  # (batch, 3)
        
        return self.cross_proj(crossed)


class SedimentGradientMonitor(nn.Module):
    """
    Monitor sediment gradient to capture burst patterns during flood waves.
    
    The Yellow River shows characteristic "sediment burst" where 
    sediment concentration spikes rapidly with rising flow.
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, sediment_seq: torch.Tensor) -> torch.Tensor:
        """
        Compute sediment gradient and detect burst patterns.
        
        Args:
            sediment_seq: (batch, seq_len) sediment time series
        
        Returns:
            (batch,) burst intensity score
        """
        # Compute gradient along time dimension
        grad = torch.diff(sediment_seq, dim=1)  # (batch, seq_len-1)
        
        # Detect positive gradients (rising sediment)
        rising_mask = (grad > self.threshold).float()
        
        # Burst intensity: sum of significant rises
        burst_intensity = torch.sum(grad * rising_mask, dim=1)
        
        return burst_intensity


class YellowRiverPersonalityPINN(nn.Module):
    """
    PINN with Yellow River personality.
    
    Captures:
    - Seasonal patterns via sin/cos encoding
    - Non-linear sediment-flow via feature crossing
    - Exner erosion dynamics
    """
    
    def __init__(
        self,
        input_dim: int = 9,  # [flow, rain, temp, sin_year, cos_year, sin_season, cos_season, is_flood, time_of_day]
        hidden_dim: int = 128,
        output_dim: int = 2,  # [water_level, sediment]
        num_layers: int = 4,
        use_feature_crossing: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_feature_crossing = use_feature_crossing
        
        # Feature crossing module
        if use_feature_crossing:
            self.feature_crosser = FeatureCrossing(hidden_dim)
            self.cross_proj = nn.Linear(input_dim, hidden_dim)
        
        # Main network with residual connections
        layers = []
        prev_dim = input_dim if not use_feature_crossing else hidden_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),  # Swish activation for smooth derivatives
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Output heads
        self.water_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.sediment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Sediment gradient monitor
        self.sediment_monitor = SedimentGradientMonitor()
        
    def forward(self, x: torch.Tensor, Q: torch.Tensor = None, S_history: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional feature crossing.
        
        Args:
            x: Input features (batch, input_dim)
            Q: Flow rate for crossing (batch, 1)
            S_history: Historical sediment for gradient monitoring (batch, seq_len)
        
        Returns:
            (batch, 2): [water_level, sediment]
        """
        features = x
        
        # Feature crossing if enabled
        if self.use_feature_crossing and Q is not None:
            # Current sediment prediction (initial guess from features)
            temp_out = self.network(self.cross_proj(x))
            S_init = self.sediment_head(temp_out)
            
            # Cross features
            crossed = self.feature_crosser(Q, S_init)
            features = features + self.cross_proj(x) + crossed
        
        # Main forward
        hidden = self.network(features)
        
        water_level = self.water_head(hidden)
        sediment = self.sediment_head(hidden)
        
        output = torch.cat([water_level, sediment], dim=-1)
        
        return output


class ExnerErosionLoss(nn.Module):
    """
    Exner equation for sediment continuity:
    ∂η/∂t + (1/(1-p)) * ∂Qs/∂x = 0
    
    where:
    - η: bed elevation
    - p: porosity
    - Qs: sediment flux
    
    Captures erosion during floods and deposition during dry periods.
    """
    
    def __init__(self, porosity: float = 0.4):
        super().__init__()
        self.porosity = porosity
        self.eps = 1e-6
        
    def forward(
        self,
        sediment_pred: torch.Tensor,
        flow_rate: torch.Tensor,
        dt: torch.Tensor,
        dx: torch.Tensor,
        create_graph: bool = True
    ) -> torch.Tensor:
        """
        Compute Exner equation residual.
        
        Args:
            sediment_pred: Predicted sediment (batch, 1)
            flow_rate: Flow rate Q (batch, 1)
            dt: Time step
            dx: Spatial step
        
        Returns:
            Exner residual loss
        """
        # Compute gradients
        if create_graph:
            dS_dt = torch.autograd.grad(
                sediment_pred, dt,
                grad_outputs=torch.ones_like(sediment_pred),
                create_graph=True, retain_graph=True
            )[0] if dt.requires_grad else torch.zeros_like(sediment_pred)
            
            # Simplified: Assume Qs ∝ Q * S (sediment flux proportional to flow × concentration)
            Qs = flow_rate * sediment_pred
            dQs_dx = torch.autograd.grad(
                Qs, dx,
                grad_outputs=torch.ones_like(Qs),
                create_graph=True, retain_graph=True
            )[0] if dx.requires_grad else torch.zeros_like(Qs)
        else:
            dS_dt = torch.zeros_like(sediment_pred)
            dQs_dx = torch.zeros_like(sediment_pred)
        
        # Exner equation residual
        # ∂η/∂t ≈ - (1/(1-p)) * ∂Qs/∂x
        residual = dS_dt + (1.0 / (1.0 - self.porosity)) * dQs_dx
        
        return torch.mean(residual ** 2)


class SaintVenantWithExnerLoss(nn.Module):
    """
    Combined physics loss: Saint-Venant + Exner erosion.
    """
    
    def __init__(
        self,
        g: float = 9.8,
        n: float = 0.03,
        S0: float = 0.001,
        B: float = 50.0,
        porosity: float = 0.4,
        alpha_sv: float = 1.0,
        alpha_exner: float = 1.0
    ):
        super().__init__()
        self.g = g
        self.n = n
        self.S0 = S0
        self.B = B
        self.porosity = porosity
        self.alpha_sv = alpha_sv
        self.alpha_exner = alpha_exner
        
    def compute_saint_venant_residuals(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Saint-Venant residuals (continuity and momentum).
        """
        # Continuity: ∂h/∂t + ∂(hu)/∂x = 0
        dh_dt = torch.autograd.grad(h, t, torch.ones_like(h), create_graph=True, retain_graph=True)[0]
        d_hu_dx = torch.autograd.grad(h * u, x, torch.ones_like(h), create_graph=True, retain_graph=True)[0]
        residual_cont = dh_dt + d_hu_dx
        
        # Momentum: ∂u/∂t + u·∂u/∂x = g·∂h/∂x - friction + g·S0
        du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        dh_dx = torch.autograd.grad(h, x, torch.ones_like(h), create_graph=True, retain_graph=True)[0]
        
        # Manning friction
        h_safe = torch.clamp(h, min=0.1)
        friction = self.g * (self.n ** 2) * u * torch.abs(u) / (h_safe ** (4/3))
        
        # Bed slope
        bed_slope = self.g * self.S0
        
        # Pressure gradient
        pressure_grad = self.g * dh_dx
        
        residual_mom = du_dt + u * du_dx - pressure_grad + friction - bed_slope
        
        return residual_cont, residual_mom
    
    def forward(
        self,
        predictions: torch.Tensor,
        flow_rate: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute combined physics loss.
        
        Args:
            predictions: (batch, 2) [water_level, sediment]
            flow_rate: (batch, 1)
            t, x: Time and space coordinates
        
        Returns:
            Combined physics loss
        """
        h = predictions[:, 0:1]
        S = predictions[:, 1:2]
        
        # Compute velocity from flow rate
        A = self.B * torch.clamp(h, min=0.1)
        u = flow_rate / (A + 1e-6)
        
        # Saint-Venant residuals
        res_cont, res_mom = self.compute_saint_venant_residuals(h, u, t, x)
        loss_sv = torch.mean(res_cont ** 2 + res_mom ** 2)
        
        # Exner residual (simplified)
        # ∂S/∂t ∝ ∂(Q·S)/∂x
        dS_dt = torch.autograd.grad(S, t, torch.ones_like(S), create_graph=True, retain_graph=True)[0] if t.requires_grad else torch.zeros_like(S)
        QS = flow_rate * S
        dQS_dx = torch.autograd.grad(QS, x, torch.ones_like(QS), create_graph=True, retain_graph=True)[0] if x.requires_grad else torch.zeros_like(QS)
        res_exner = dS_dt + dQS_dx / (1.0 - self.porosity)
        loss_exner = torch.mean(res_exner ** 2)
        
        total_loss = self.alpha_sv * loss_sv + self.alpha_exner * loss_exner
        
        if return_components:
            return total_loss, {
                'saint_venant': loss_sv.item(),
                'exner': loss_exner.item(),
                'residual_cont': torch.mean(torch.abs(res_cont)).item(),
                'residual_mom': torch.mean(torch.abs(res_mom)).item()
            }
        
        return total_loss


class NTKAdaptiveWeighting:
    """
    Neural Tangent Kernel (NTK) based adaptive weighting.
    
    Automatically adjusts loss weights based on training dynamics.
    Increases physics weight when model starts fitting data too well
    but violating physics.
    """
    
    def __init__(
        self,
        initial_data_weight: float = 1.0,
        initial_physics_weight: float = 0.1,
        target_physics_residual: float = 0.1,
        warmup_epochs: int = 100
    ):
        self.data_weight = initial_data_weight
        self.physics_weight = initial_physics_weight
        self.target_residual = target_physics_residual
        self.warmup_epochs = warmup_epochs
        
        self.epoch = 0
        self.physics_residual_history = []
        
    def step(self, physics_residual: float):
        """
        Update weights based on current physics residual.
        
        Strategy:
        - Early training (warmup): Keep physics weight low
        - Late training: Increase physics weight if residual > target
        """
        self.epoch += 1
        self.physics_residual_history.append(physics_residual)
        
        if self.epoch < self.warmup_epochs:
            # Linear warmup
            progress = self.epoch / self.warmup_epochs
            self.physics_weight = 0.1 + 0.9 * progress
        else:
            # Adaptive: increase physics weight if residual too high
            recent_residual = np.mean(self.physics_residual_history[-10:])
            
            if recent_residual > self.target_physics_residual:
                # Increase physics weight
                self.physics_weight = min(2.0, self.physics_weight * 1.05)
                logger.info(f"NTK: Increasing physics weight to {self.physics_weight:.3f} "
                          f"(residual={recent_residual:.3f})")
            else:
                # Decay slowly
                self.physics_weight = max(0.5, self.physics_weight * 0.99)
        
        return self.data_weight, self.physics_weight


def load_yellow_river_data(
    data_path: str,
    years: int = 10
) -> Tuple[pd.DataFrame, SeasonalEncoder]:
    """
    Load and preprocess 10-year Yellow River data.
    
    Expected columns:
    - timestamp
    - water_level
    - flow_rate
    - sediment_content
    - rainfall
    - temperature
    - station_id
    """
    logger.info(f"Loading Yellow River data from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Initialize seasonal encoder
    seasonal_encoder = SeasonalEncoder(base_year=df['timestamp'].dt.year.min())
    
    # Add seasonal features
    seasonal_features = seasonal_encoder.encode(df['timestamp'])
    df['sin_year'] = seasonal_features[:, 0]
    df['cos_year'] = seasonal_features[:, 1]
    df['sin_season'] = seasonal_features[:, 2]
    df['cos_season'] = seasonal_features[:, 3]
    df['is_flood_season'] = seasonal_encoder.encode_flood_season(df['timestamp'])
    
    # Add time of day (for diurnal patterns)
    df['hour'] = df['timestamp'].dt.hour
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    logger.info(f"Loaded {len(df)} records spanning {years} years")
    logger.info(f"Flood season samples: {df['is_flood_season'].sum()}")
    
    return df, seasonal_encoder


def train_epoch_adam(
    model: YellowRiverPersonalityPINN,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    physics_loss_fn: SaintVenantWithExnerLoss,
    adaptive_weighting: NTKAdaptiveWeighting,
    epoch: int
) -> Dict[str, float]:
    """Single epoch of Adam training."""
    model.train()
    
    total_data_loss = 0
    total_physics_loss = 0
    total_combined_loss = 0
    
    for batch_idx, (X, y, Q, t, x) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(X, Q)
        
        # Data loss (MSE)
        data_loss = nn.MSELoss()(pred, y)
        
        # Physics loss
        physics_loss, phys_components = physics_loss_fn(
            pred, Q, t, x, return_components=True
        )
        
        # Get adaptive weights
        data_weight, phys_weight = adaptive_weighting.step(phys_components['saint_venant'])
        
        # Combined loss
        combined_loss = data_weight * data_loss + phys_weight * physics_loss
        
        # Backward
        combined_loss.backward()
        optimizer.step()
        
        total_data_loss += data_loss.item()
        total_physics_loss += physics_loss.item()
        total_combined_loss += combined_loss.item()
    
    n_batches = len(train_loader)
    return {
        'data_loss': total_data_loss / n_batches,
        'physics_loss': total_physics_loss / n_batches,
        'combined_loss': total_combined_loss / n_batches,
        'data_weight': data_weight,
        'physics_weight': phys_weight,
        'epoch': epoch
    }


def train_lbfgs_finetune(
    model: YellowRiverPersonalityPINN,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    train_Q: torch.Tensor,
    train_t: torch.Tensor,
    train_x: torch.Tensor,
    physics_loss_fn: SaintVenantWithExnerLoss,
    max_iter: int = 200,
    target_residual: float = 0.1
) -> Dict[str, float]:
    """
    L-BFGS fine-tuning for physics residual minimization.
    """
    logger.info("Starting L-BFGS fine-tuning...")
    
    # Flatten parameters for L-BFGS
    def closure():
        optimizer.zero_grad()
        
        pred = model(train_data, train_Q)
        
        data_loss = nn.MSELoss()(pred, train_labels)
        physics_loss, phys_comp = physics_loss_fn(pred, train_Q, train_t, train_x, return_components=True)
        
        # Heavy weight on physics in L-BFGS stage
        loss = 0.1 * data_loss + 10.0 * physics_loss
        
        loss.backward()
        return loss
    
    # L-BFGS optimizer
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=max_iter,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-7,
        tolerance_change=1e-9
    )
    
    # Optimization loop
    prev_loss = float('inf')
    for iteration in range(max_iter):
        loss = optimizer.step(closure)
        
        if iteration % 20 == 0:
            logger.info(f"L-BFGS Iter {iteration}: Loss={loss.item():.6f}")
        
        # Check convergence
        if abs(prev_loss - loss.item()) < 1e-7:
            logger.info(f"L-BFGS converged at iteration {iteration}")
            break
        
        prev_loss = loss.item()
    
    # Final evaluation
    with torch.no_grad():
        final_pred = model(train_data, train_Q)
        final_physics, phys_comp = physics_loss_fn(
            final_pred, train_Q, train_t, train_x, return_components=True
        )
    
    logger.info(f"L-BFGS complete: Physics residual = {phys_comp['saint_venant']:.6f}")
    
    return {
        'final_loss': loss.item(),
        'physics_residual': phys_comp['saint_venant'],
        'exner_residual': phys_comp['exner'],
        'target_achieved': phys_comp['saint_venant'] < target_residual
    }


def main():
    parser = argparse.ArgumentParser(description='Train Yellow River Personality PINN')
    parser.add_argument('--data', type=str, required=True, help='Path to 10-year dataset')
    parser.add_argument('--epochs', type=int, default=1000, help='Adam training epochs')
    parser.add_argument('--lbfgs-iter', type=int, default=200, help='L-BFGS iterations')
    parser.add_argument('--lr', type=float, default=0.001, help='Adam learning rate')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--save-dir', type=str, default='models/pinn_personality')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    df, seasonal_encoder = load_yellow_river_data(args.data)
    
    # Prepare tensors
    feature_cols = [
        'flow_rate', 'rainfall', 'temperature',
        'sin_year', 'cos_year', 'sin_season', 'cos_season',
        'is_flood_season', 'sin_hour', 'cos_hour'
    ]
    target_cols = ['water_level', 'sediment_content']
    
    X = torch.FloatTensor(df[feature_cols].values)
    y = torch.FloatTensor(df[target_cols].values)
    Q = torch.FloatTensor(df['flow_rate'].values).unsqueeze(1)
    
    # Create synthetic t, x for physics (would be actual coordinates in production)
    t = torch.linspace(0, 1, len(df)).unsqueeze(1)
    x = torch.linspace(0, 100, len(df)).unsqueeze(1)
    
    # Split train/val
    train_size = int(0.8 * len(df))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    Q_train, Q_val = Q[:train_size], Q[train_size:]
    t_train, t_val = t[:train_size], t[train_size:]
    x_train, x_val = x[:train_size], x[train_size:]
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train, Q_train, t_train, x_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    model = YellowRiverPersonalityPINN(
        input_dim=X.shape[1],
        hidden_dim=128,
        output_dim=2,
        num_layers=4,
        use_feature_crossing=True
    )
    
    # Initialize physics loss
    physics_loss_fn = SaintVenantWithExnerLoss(
        g=9.8, n=0.03, S0=0.001, B=50.0,
        alpha_sv=1.0, alpha_exner=1.0
    )
    
    # Initialize adaptive weighting
    adaptive_weighting = NTKAdaptiveWeighting(
        initial_data_weight=1.0,
        initial_physics_weight=0.1,
        target_physics_residual=0.1,
        warmup_epochs=100
    )
    
    # Stage 1: Adam pre-training
    logger.info("=" * 80)
    logger.info("Stage 1: Adam Pre-training")
    logger.info("=" * 80)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    adam_history = []
    for epoch in range(args.epochs):
        metrics = train_epoch_adam(
            model, train_loader, optimizer,
            physics_loss_fn, adaptive_weighting, epoch
        )
        adam_history.append(metrics)
        
        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}: "
                       f"Data Loss={metrics['data_loss']:.4f}, "
                       f"Physics Loss={metrics['physics_loss']:.4f}, "
                       f"λ_phys={metrics['physics_weight']:.3f}")
    
    # Save Adam checkpoint
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'adam_history': adam_history
    }, os.path.join(save_dir, 'adam_checkpoint.pt'))
    
    # Stage 2: L-BFGS fine-tuning
    logger.info("=" * 80)
    logger.info("Stage 2: L-BFGS Fine-tuning")
    logger.info("=" * 80)
    
    lbfgs_results = train_lbfgs_finetune(
        model, X_train, y_train, Q_train, t_train, x_train,
        physics_loss_fn, max_iter=args.lbfgs_iter
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'seasonal_encoder': seasonal_encoder,
        'adam_history': adam_history,
        'lbfgs_results': lbfgs_results,
        'config': vars(args)
    }, os.path.join(save_dir, 'yellow_river_personality.pt'))
    
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Final physics residual: {lbfgs_results['physics_residual']:.6f}")
    logger.info(f"Target achieved (<0.1): {lbfgs_results['target_achieved']}")
    logger.info(f"Model saved to: {save_dir}")
    logger.info("=" * 80)
    
    # Generate evolution report
    generate_training_report(adam_history, lbfgs_results, save_dir)


def generate_training_report(adam_history, lbfgs_results, save_dir):
    """Generate training visualization report."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Yellow River PINN Personality Training', fontsize=14, fontweight='bold')
    
    epochs = [m['epoch'] for m in adam_history]
    
    # 1. Data loss
    ax = axes[0, 0]
    ax.semilogy(epochs, [m['data_loss'] for m in adam_history], 'b-', label='Data Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.set_title('1. Data Loss (Adam Stage)')
    ax.grid(True, alpha=0.3)
    
    # 2. Physics loss
    ax = axes[0, 1]
    ax.semilogy(epochs, [m['physics_loss'] for m in adam_history], 'r-', label='Physics Loss')
    ax.axhline(y=0.1, color='g', linestyle='--', label='Target')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.set_title('2. Physics Loss Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Adaptive weights
    ax = axes[1, 0]
    ax.plot(epochs, [m['data_weight'] for m in adam_history], 'b-', label='Data Weight')
    ax.plot(epochs, [m['physics_weight'] for m in adam_history], 'r-', label='Physics Weight')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight')
    ax.set_title('3. NTK Adaptive Weighting')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
    Training Summary
    ================
    
    Adam Epochs: {len(adam_history)}
    L-BFGS: Complete
    
    Final Metrics:
      Physics Residual: {lbfgs_results['physics_residual']:.6f}
      Exner Residual: {lbfgs_results['exner_residual']:.6f}
      Target (<0.1): {'✓ Achieved' if lbfgs_results['target_achieved'] else '✗ Not Achieved'}
    
    Model Personality:
      - Sin/Cos seasonal encoding
      - Feature crossing (Q×S)
      - Exner erosion dynamics
      - Saint-Venant physics
    """
    ax.text(0.1, 0.5, summary, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_report.png'), dpi=150, bbox_inches='tight')
    logger.info(f"Training report saved to {save_dir}/training_report.png")


if __name__ == '__main__':
    main()
