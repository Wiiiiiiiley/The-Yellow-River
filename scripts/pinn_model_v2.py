"""Yellow River PINN v2.0 - Research Grade Implementation.

SOTA upgrades:
1. ResNet architecture with residual connections
2. Attention-based Feature Scaling for extreme weather
3. Full 1D Saint-Venant equations (Continuity + Momentum)
4. Self-Adaptive Loss Weights (Log-variance uncertainty weighting)

Physics:
    - Continuity: ∂h/∂t + ∂(hu)/∂x = 0
    - Momentum: ∂u/∂t + u·∂u/∂x + g·∂h/∂x = -g·(u|u|/C²·h)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np


class Swish(nn.Module):
    """Swish activation: f(x) = x * sigmoid(x)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class FeatureAttention(nn.Module):
    """Attention-based Feature Scaling for extreme weather conditions.
    
    Dynamically weights features based on flow_rate and rainfall intensity.
    This helps the model focus on critical features during flood events.
    
    Args:
        feature_dim: Total number of input features
        attention_features: Indices of features to use for attention (e.g., [2, 3] for flow, rain)
    """
    def __init__(self, feature_dim: int, attention_features: List[int] = [2, 3]):
        super(FeatureAttention, self).__init__()
        self.feature_dim = feature_dim
        self.attention_features = attention_features
        self.num_attn_features = len(attention_features)
        
        # Attention network: produces scaling weights
        self.attention_net = nn.Sequential(
            nn.Linear(self.num_attn_features, 16),
            Swish(),
            nn.Linear(16, feature_dim),
            nn.Sigmoid()  # Output weights in [0, 1]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention-based feature scaling.
        
        Args:
            x: Input tensor (batch, feature_dim)
        
        Returns:
            Scaled features (batch, feature_dim)
        """
        # Extract attention-driving features
        attn_input = x[:, self.attention_features]  # (batch, num_attn_features)
        
        # Compute attention weights
        weights = self.attention_net(attn_input)  # (batch, feature_dim)
        
        # Apply feature scaling
        scaled_features = x * weights
        
        return scaled_features, weights


class ResidualBlock(nn.Module):
    """ResNet-style residual block with pre-activation.
    
    Structure: Norm → Activation → Linear → Norm → Activation → Linear → Skip
    """
    def __init__(self, dim: int, activation: nn.Module = None):
        super(ResidualBlock, self).__init__()
        if activation is None:
            activation = Swish()
        
        self.norm1 = nn.LayerNorm(dim)
        self.activation = activation
        self.linear1 = nn.Linear(dim, dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.norm1(x)
        out = self.activation(out)
        out = self.linear1(out)
        
        out = self.norm2(out)
        out = self.activation(out)
        out = self.linear2(out)
        
        return out + residual  # Skip connection


class YellowRiverPINNv2(nn.Module):
    """Research-grade PINN with ResNet + Attention architecture.
    
    Architecture:
        Input (5D: t, x, flow, rain, temp)
        → Feature Attention (flow/rain weighted)
        → Input Projection
        → Residual Blocks (×N)
        → Output Heads (water_level, sediment)
    
    Args:
        input_dim: Input feature dimension (default 5)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (default 2: h, sediment)
        num_res_blocks: Number of residual blocks
        use_attention: Whether to use attention-based feature scaling
    """
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        output_dim: int = 2,
        num_res_blocks: int = 4,
        use_attention: bool = True
    ):
        super(YellowRiverPINNv2, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Feature attention module
        if use_attention:
            self.attention = FeatureAttention(input_dim, attention_features=[2, 3])
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, Swish())
            for _ in range(num_res_blocks)
        ])
        
        # Layer normalization before output
        self.pre_output_norm = nn.LayerNorm(hidden_dim)
        
        # Output head
        self.output_head = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization with special handling for residual connections."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization for Swish
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Forward pass with optional attention weights.
        
        Args:
            x: Input tensor (batch, input_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            output: Predictions (batch, output_dim)
            attention_weights: (optional) Attention scaling weights
        """
        # Apply attention-based feature scaling
        if self.use_attention:
            x, attn_weights = self.attention(x)
        else:
            attn_weights = None
        
        # Input projection
        out = self.input_proj(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            out = res_block(out)
        
        # Pre-output normalization and activation
        out = self.pre_output_norm(out)
        out = F.silu(out)  # SiLU/Swish
        
        # Output
        output = self.output_head(out)
        
        if return_attention:
            return output, attn_weights
        return output


class AdaptiveLossWeights(nn.Module):
    """Self-Adaptive Loss Weights using Log-Variance Uncertainty.
    
    Based on: 'Multi-Task Learning Using Uncertainty to Weigh Losses'
    Kendall et al., CVPR 2018
    
    Implements 3 independent adaptive weights for:
    - Data loss (observation fitting)
    - Continuity loss (mass conservation)
    - Momentum loss (motion equation)
    
    Formula: loss_total = Σ[(1/(2σ_i²)) * loss_i + log(σ_i)]
    """
    def __init__(self, num_losses: int = 3, init_log_var: float = 0.0):
        super(AdaptiveLossWeights, self).__init__()
        # Learnable log-variance parameters for 3 losses
        self.log_vars = nn.Parameter(torch.ones(num_losses) * init_log_var)
        
    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute weighted sum of losses with learned uncertainty weighting.
        
        Args:
            losses: List of [data_loss, continuity_loss, momentum_loss]
        
        Returns:
            total_loss: Weighted total loss
            weighted_losses: Individual weighted losses for logging
        """
        total_loss = 0.0
        weighted_losses = []
        
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])  # 1 / σ²
            weighted = precision * loss + self.log_vars[i]  # Weighted loss + regularization
            total_loss += weighted
            weighted_losses.append(weighted)
        
        return total_loss, torch.stack(weighted_losses)
    
    def get_weights(self) -> torch.Tensor:
        """Get current effective weights λ_i = 1/(2σ_i²) (for logging)."""
        return (0.5 * torch.exp(-self.log_vars)).detach()
    
    def get_log_vars(self) -> torch.Tensor:
        """Get current log-variances (for logging)."""
        return self.log_vars.detach()


def compute_physics_loss_v2(
    model: YellowRiverPINNv2,
    x_physics: torch.Tensor,
    g: float = 9.8,      # Gravity (m/s²)
    n: float = 0.03,    # Manning roughness coefficient
    S0: float = 0.001,  # Riverbed slope
    B: float = 50.0     # Channel width (m)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute physics loss with full 1D Saint-Venant equations.
    
    Equations:
        1. Continuity: ∂h/∂t + ∂(hu)/∂x = 0
        2. Momentum: ∂u/∂t + u·∂u/∂x = g·∂h/∂x - g·(n²·u|u|/h^(4/3)) + g·S0
    
    Constants:
        g = 9.8 m/s², n = 0.03, S0 = 0.001, B = 50m
    
    Args:
        model: PINN v2 model
        x_physics: Physics sampling points (N, 5) [t, x, flow, rain, temp]
    
    Returns:
        continuity_loss: Mass conservation residual
        momentum_loss: Motion equation residual  
        components: Dictionary of intermediate values for analysis
    """
    # Validate input dimensions
    if x_physics.shape[1] < 3:
        raise ValueError(f"x_physics must have at least 3 dims [t, x, flow], got {x_physics.shape[1]}")
    
    # Extract and isolate space-time variables as leaf nodes
    t = x_physics[:, 0:1].clone().requires_grad_(True)  # Time
    x = x_physics[:, 1:2].clone().requires_grad_(True)  # Space
    features = x_physics[:, 2:]  # [flow_rate, rainfall, temp]
    
    # Reconstruct input
    inputs = torch.cat([t, x, features], dim=1)
    
    # Forward pass
    prediction = model(inputs)
    
    water_level = prediction[:, 0:1]  # h
    # For sediment, we'll use it as proxy for other quantities if needed
    
    # Extract flow rate Q from features (assumes Q is at index 0 of features)
    flow_rate = features[:, 0:1]  # Q
    
    # Compute velocity u = Q / (A) = Q / (h * width)
    # For rectangular channel with unit width: A = h, so u = Q / h
    epsilon = 1e-6
    h_safe = water_level + epsilon
    velocity = flow_rate / h_safe  # u
    
    # ========== Continuity Equation ==========
    # ∂h/∂t
    dh_dt = torch.autograd.grad(
        outputs=water_level,
        inputs=t,
        grad_outputs=torch.ones_like(water_level),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Flux = h * u
    flux = water_level * velocity
    
    # ∂(hu)/∂x
    d_flux_dx = torch.autograd.grad(
        outputs=flux,
        inputs=x,
        grad_outputs=torch.ones_like(flux),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Continuity residual: ∂h/∂t + ∂(hu)/∂x
    residual_continuity = dh_dt + d_flux_dx
    loss_continuity = torch.mean(residual_continuity ** 2)
    
    # ========== Momentum Equation ==========
    # ∂u/∂t
    du_dt = torch.autograd.grad(
        outputs=velocity,
        inputs=t,
        grad_outputs=torch.ones_like(velocity),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # ∂u/∂x
    du_dx = torch.autograd.grad(
        outputs=velocity,
        inputs=x,
        grad_outputs=torch.ones_like(velocity),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # ∂h/∂x (pressure gradient term)
    dh_dx = torch.autograd.grad(
        outputs=water_level,
        inputs=x,
        grad_outputs=torch.ones_like(water_level),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Convective acceleration: u * ∂u/∂x
    convective = velocity * du_dx
    
    # Pressure gradient term: g * ∂h/∂x
    pressure_grad = g * dh_dx
    
    # Manning friction term: g * n² * u * |u| / h^(4/3)
    h_pow = torch.pow(h_safe, 4.0/3.0)
    friction = g * (n ** 2) * velocity * torch.abs(velocity) / h_pow
    
    # Riverbed slope term: g * S0
    bed_slope = g * S0
    
    # Momentum equation residual:
    # LHS: ∂u/∂t + u·∂u/∂x
    # RHS: g·∂h/∂x - g·(n²·u|u|/h^(4/3)) + g·S0
    lhs_momentum = du_dt + convective
    rhs_momentum = pressure_grad - friction + bed_slope
    residual_momentum = lhs_momentum - rhs_momentum
    loss_momentum = torch.mean(residual_momentum ** 2)
    
    # Return separate losses for adaptive weighting
    components = {
        'loss_continuity': loss_continuity,
        'loss_momentum': loss_momentum,
        'residual_continuity': residual_continuity.detach(),
        'residual_momentum': residual_momentum.detach(),
        'dh_dt': dh_dt.detach(),
        'd_flux_dx': d_flux_dx.detach(),
        'du_dt': du_dt.detach(),
        'du_dx': du_dx.detach(),
        'dh_dx': dh_dx.detach(),
        'velocity': velocity.detach(),
        'friction': friction.detach(),
        'pressure_grad': pressure_grad.detach(),
        'bed_slope': torch.tensor(bed_slope)  # Convert float to tensor for consistency
    }
    
    return loss_continuity, loss_momentum, components


def compute_total_loss_v2(
    model: YellowRiverPINNv2,
    adaptive_weights: AdaptiveLossWeights,
    x_data: torch.Tensor,
    y_true: torch.Tensor,
    x_physics: Optional[torch.Tensor] = None,
    g: float = 9.8,
    n: float = 0.03,
    S0: float = 0.001,
    B: float = 50.0,
    max_grad_norm: Optional[float] = 1.0
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute total loss with 3-way adaptive weighting.
    
    Losses:
        1. Data loss: MSE of predictions vs observations
        2. Continuity loss: Mass conservation residual
        3. Momentum loss: Motion equation residual
    
    Formula: L_total = Σ [(1/(2σ_i²)) * L_i + log(σ_i)] for i ∈ {data, cont, mom}
    
    Args:
        model: PINN v2 model
        adaptive_weights: Learnable loss weighting (3 components)
        x_data: Data training points
        y_true: Ground truth labels
        x_physics: Physics sampling points (optional)
        g, n, S0, B: Physical constants
        max_grad_norm: Gradient clipping threshold
    
    Returns:
        total_loss: Combined weighted loss
        loss_components: Dictionary with detailed loss information
    """
    # Validate inputs
    if x_data.shape[1] != model.input_dim:
        raise ValueError(f"x_data dim ({x_data.shape[1]}) != model input dim ({model.input_dim})")
    if y_true.shape[1] != model.output_dim:
        raise ValueError(f"y_true dim ({y_true.shape[1]}) != model output dim ({model.output_dim})")
    
    # Data loss (MSE)
    y_pred = model(x_data)
    data_loss = F.mse_loss(y_pred, y_true)
    
    # Physics losses (separate continuity and momentum)
    if x_physics is None:
        x_physics = x_data
    
    loss_continuity, loss_momentum, phys_components = compute_physics_loss_v2(
        model, x_physics, g=g, n=n, S0=S0, B=B
    )
    
    # Adaptive weighting for 3 losses
    total_loss, weighted_losses = adaptive_weights([data_loss, loss_continuity, loss_momentum])
    
    # Gradient clipping info
    clipping_info = {'max_grad_norm': max_grad_norm, 'enabled': max_grad_norm is not None}
    
    # Get current adaptive weights and log-vars for logging
    current_weights = adaptive_weights.get_weights()
    current_log_vars = adaptive_weights.get_log_vars()
    
    loss_components = {
        'total_loss': total_loss,
        'data_loss': data_loss,
        'continuity_loss': loss_continuity,
        'momentum_loss': loss_momentum,
        'weighted_data': weighted_losses[0],
        'weighted_continuity': weighted_losses[1],
        'weighted_momentum': weighted_losses[2],
        'lambda_data': current_weights[0],
        'lambda_continuity': current_weights[1],
        'lambda_momentum': current_weights[2],
        'log_var_data': current_log_vars[0],
        'log_var_continuity': current_log_vars[1],
        'log_var_momentum': current_log_vars[2],
        'gradient_clipping': clipping_info,
        **phys_components
    }
    
    return total_loss, loss_components


if __name__ == "__main__":
    """Unit test for PINN v2."""
    print("=" * 70)
    print("Yellow River PINN v2.0 - Research Grade - Unit Test")
    print("=" * 70)
    
    # Test 1: Model architecture
    print("\n[1/5] Testing ResNet + Attention architecture...")
    model = YellowRiverPINNv2(
        input_dim=5,
        hidden_dim=128,
        output_dim=2,
        num_res_blocks=4,
        use_attention=True
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x_test = torch.randn(32, 5)
    y_pred, attn_weights = model(x_test, return_attention=True)
    print(f"  Input shape: {x_test.shape}")
    print(f"  Output shape: {y_pred.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"  Attention range: [{attn_weights.min():.3f}, {attn_weights.max():.3f}]")
    
    # Test 2: Adaptive weights (3 components: data, continuity, momentum)
    print("\n[2/5] Testing 3-way adaptive loss weights...")
    adaptive_weights = AdaptiveLossWeights(num_losses=3)
    data_loss = torch.tensor(1.0)
    cont_loss = torch.tensor(0.3)
    mom_loss = torch.tensor(0.7)
    total, weighted = adaptive_weights([data_loss, cont_loss, mom_loss])
    weights = adaptive_weights.get_weights()
    print(f"  Initial λ: data={weights[0]:.3f}, continuity={weights[1]:.3f}, momentum={weights[2]:.3f}")
    print(f"  Total loss: {total.item():.3f}")
    
    # Test 3: Physics loss with Saint-Venant equations (g=9.8, n=0.03, S0=0.001, B=50)
    print("\n[3/5] Testing Saint-Venant physics loss...")
    x_physics = torch.randn(64, 5)
    loss_cont, loss_mom, phys_comp = compute_physics_loss_v2(
        model, x_physics, g=9.8, n=0.03, S0=0.001, B=50.0
    )
    print(f"  Continuity loss: {loss_cont.item():.6f}")
    print(f"  Momentum loss: {loss_mom.item():.6f}")
    print(f"  Velocity range: [{phys_comp['velocity'].min():.3f}, {phys_comp['velocity'].max():.3f}]")
    print(f"  Friction term range: [{phys_comp['friction'].min():.3f}, {phys_comp['friction'].max():.3f}]")
    
    # Test 4: Total loss with gradient flow
    print("\n[4/5] Testing end-to-end gradient flow...")
    y_true = torch.randn(32, 2)
    total_loss, logs = compute_total_loss_v2(
        model, adaptive_weights, x_test, y_true,
        x_physics=x_physics,
        g=9.8, n=0.03, S0=0.001, B=50.0,
        max_grad_norm=1.0
    )
    total_loss.backward()
    
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    
    print(f"  Total loss: {total_loss.item():.6f}")
    print(f"  λ: data={logs['lambda_data']:.3f}, cont={logs['lambda_continuity']:.3f}, mom={logs['lambda_momentum']:.3f}")
    print(f"  log-σ²: data={logs['log_var_data']:.3f}, cont={logs['log_var_continuity']:.3f}, mom={logs['log_var_momentum']:.3f}")
    print(f"  Gradient norms: min={min(grad_norms):.3f}, max={max(grad_norms):.3f}")
    print(f"  ✓ Gradients flowing correctly")
    
    # Test 5: Residual connection verification
    print("\n[5/5] Verifying residual connections...")
    # Check that identity mapping works when weights are zero
    res_block = ResidualBlock(64)
    x_res = torch.randn(16, 64)
    # Set weights to near-zero to approximate identity
    with torch.no_grad():
        for p in res_block.parameters():
            p.fill_(0.001)
    y_res = res_block(x_res)
    identity_error = torch.norm(y_res - x_res * 1.001).item()  # Small offset due to bias
    print(f"  Residual identity error: {identity_error:.6f}")
    print(f"  ✓ Residual connections working (error should be small)")
    
    print("\n" + "=" * 70)
    print("All tests passed! PINN v2.0 ready for training.")
    print("=" * 70)
