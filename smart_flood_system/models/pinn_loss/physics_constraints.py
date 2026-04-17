"""
Physics-Informed Loss Functions for Saint-Venant Equations
===========================================================

Implements 1D Saint-Venant (shallow water) equations as soft constraints:
1. Continuity Equation: ∂h/∂t + ∂(hu)/∂x = 0
2. Momentum Equation: ∂u/∂t + u·∂u/∂x = g·∂h/∂x - g·(n²·u|u|/h^(4/3)) + g·S0

These equations constrain the neural network outputs to satisfy physical laws,
ensuring predictions remain physically plausible even in extrapolation regimes.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class SaintVenantLoss(nn.Module):
    """
    Physics loss based on 1D Saint-Venant (shallow water) equations.
    
    The loss penalizes violations of mass and momentum conservation,
    effectively regularizing the model to learn physically consistent dynamics.
    
    Args:
        g: Gravitational acceleration (m/s²), default 9.8
        n: Manning roughness coefficient, default 0.03
        S0: Riverbed slope, default 0.001
        B: Channel width (m), default 50.0
        alpha_cont: Weight for continuity equation loss
        alpha_mom: Weight for momentum equation loss
    """
    
    def __init__(
        self,
        g: float = 9.8,
        n: float = 0.03,
        S0: float = 0.001,
        B: float = 50.0,
        alpha_cont: float = 1.0,
        alpha_mom: float = 1.0
    ):
        super().__init__()
        
        self.g = g
        self.n = n
        self.S0 = S0
        self.B = B
        self.alpha_cont = alpha_cont
        self.alpha_mom = alpha_mom
        
        # Small epsilon for numerical stability
        self.eps = 1e-6
        
    def compute_derivatives(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute spatial and temporal derivatives using automatic differentiation.
        
        Args:
            h: Water depth (batch, 1)
            u: Flow velocity (batch, 1)
            x: Spatial coordinate (batch, 1), requires_grad=True
            t: Temporal coordinate (batch, 1), requires_grad=True
        
        Returns:
            Dictionary of partial derivatives
        """
        derivatives = {}
        
        # ∂h/∂t
        dh_dt = torch.autograd.grad(
            outputs=h,
            inputs=t,
            grad_outputs=torch.ones_like(h),
            create_graph=True,
            retain_graph=True
        )[0]
        derivatives['dh_dt'] = dh_dt
        
        # ∂h/∂x
        dh_dx = torch.autograd.grad(
            outputs=h,
            inputs=x,
            grad_outputs=torch.ones_like(h),
            create_graph=True,
            retain_graph=True
        )[0]
        derivatives['dh_dx'] = dh_dx
        
        # ∂u/∂t
        du_dt = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        derivatives['du_dt'] = du_dt
        
        # ∂u/∂x
        du_dx = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        derivatives['du_dx'] = du_dx
        
        # ∂(hu)/∂x = u·∂h/∂x + h·∂u/∂x (product rule)
        derivatives['d_hu_dx'] = u * dh_dx + h * du_dx
        
        return derivatives
    
    def forward(
        self,
        predictions: torch.Tensor,
        data: 'torch_geometric.data.Data',
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute physics-informed loss.
        
        Args:
            predictions: Model predictions (N, 2) [water_level, sediment]
            data: PyG Data object with:
                - x: Input features including [t, x_coord, flow_rate, ...]
                - edge_index: Graph connectivity
            return_components: If True, return individual loss components
        
        Returns:
            Total physics loss (weighted sum of continuity and momentum losses)
        """
        # Extract water level from predictions
        h = predictions[:, 0:1]  # (N, 1)
        
        # Extract flow rate Q from input features (assuming Q is at index 2)
        Q = data.x[:, 2:3]  # (N, 1)
        
        # Compute velocity: u = Q / (B·h) = Q / A
        h_safe = torch.clamp(h, min=self.eps)
        A = self.B * h_safe  # Cross-sectional area
        u = Q / A  # Velocity
        
        # Extract coordinates and enable gradient tracking
        t = data.x[:, 0:1].clone().requires_grad_(True)
        x_coord = data.x[:, 1:2].clone().requires_grad_(True)
        
        # Compute all required derivatives
        derivatives = self.compute_derivatives(h_safe, u, x_coord, t)
        
        # === Continuity Equation: ∂h/∂t + ∂(hu)/∂x = 0 ===
        residual_cont = derivatives['dh_dt'] + derivatives['d_hu_dx']
        loss_continuity = torch.mean(residual_cont ** 2)
        
        # === Momentum Equation ===
        # LHS: ∂u/∂t + u·∂u/∂x (acceleration terms)
        lhs_momentum = derivatives['du_dt'] + u * derivatives['du_dx']
        
        # RHS: g·∂h/∂x - g·(n²·u|u|/h^(4/3)) + g·S0
        # Pressure gradient term
        pressure_grad = self.g * derivatives['dh_dx']
        
        # Friction term (Manning equation)
        h_pow = torch.pow(h_safe, 4.0/3.0)
        friction = self.g * (self.n ** 2) * u * torch.abs(u) / h_pow
        
        # Bed slope term
        bed_slope = self.g * self.S0
        
        rhs_momentum = pressure_grad - friction + bed_slope
        
        # Momentum residual
        residual_momentum = lhs_momentum - rhs_momentum
        loss_momentum = torch.mean(residual_momentum ** 2)
        
        # Total physics loss
        total_loss = self.alpha_cont * loss_continuity + self.alpha_mom * loss_momentum
        
        if return_components:
            return total_loss, {
                'continuity_loss': loss_continuity.item(),
                'momentum_loss': loss_momentum.item(),
                'residual_cont': residual_cont.detach(),
                'residual_mom': residual_momentum.detach(),
                'velocity': u.detach(),
                'friction': friction.detach()
            }
        
        return total_loss
    
    def compute_residuals(
        self,
        predictions: torch.Tensor,
        data: 'torch_geometric.data.Data'
    ) -> Dict[str, float]:
        """
        Compute physics residuals for monitoring (no gradients).
        
        Returns:
            Dictionary of residual magnitudes
        """
        with torch.no_grad():
            _, components = self.forward(predictions, data, return_components=True)
        
        return {
            'continuity': components['continuity_loss'],
            'momentum': components['momentum_loss'],
            'velocity_mean': components['velocity'].mean().item(),
            'velocity_std': components['velocity'].std().item()
        }


class AdaptivePhysicsLoss(nn.Module):
    """
    Adaptive weighting for physics loss components based on training progress.
    
    Gradually increases physics constraint weight as data fitting improves,
    following curriculum learning principles.
    """
    
    def __init__(
        self,
        base_physics_loss: SaintVenantLoss,
        initial_weight: float = 0.1,
        final_weight: float = 1.0,
        warmup_epochs: int = 50
    ):
        super().__init__()
        self.base_loss = base_physics_loss
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch: int):
        """Update current epoch for adaptive weighting."""
        self.current_epoch = epoch
        
    def get_current_weight(self) -> float:
        """Compute adaptive weight based on training progress."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.current_epoch / self.warmup_epochs
            return self.initial_weight + alpha * (self.final_weight - self.initial_weight)
        else:
            return self.final_weight
    
    def forward(self, predictions, data, return_components=False):
        weight = self.get_current_weight()
        
        if return_components:
            loss, components = self.base_loss(predictions, data, return_components=True)
            weighted_loss = weight * loss
            components['adaptive_weight'] = weight
            return weighted_loss, components
        
        return weight * self.base_loss(predictions, data)


class BoundaryConditionLoss(nn.Module):
    """
    Enforces boundary conditions at upstream/downstream boundaries.
    """
    
    def __init__(self, upstream_idx: int, downstream_idx: int):
        super().__init__()
        self.upstream_idx = upstream_idx
        self.downstream_idx = downstream_idx
        
    def forward(
        self,
        predictions: torch.Tensor,
        upstream_target: Optional[torch.Tensor] = None,
        downstream_target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss = 0.0
        
        if upstream_target is not None:
            loss += torch.mean(
                (predictions[self.upstream_idx] - upstream_target) ** 2
            )
        
        if downstream_target is not None:
            loss += torch.mean(
                (predictions[self.downstream_idx] - downstream_target) ** 2
            )
        
        return loss


if __name__ == "__main__":
    """Unit test for physics constraints."""
    print("Testing SaintVenantLoss...")
    
    # Create dummy data
    from torch_geometric.data import Data
    
    num_nodes = 10
    x = torch.randn(num_nodes, 5)  # [t, x, flow, rain, temp]
    x[:, 0] = torch.linspace(0, 1, num_nodes)  # time
    x[:, 1] = torch.linspace(0, 10, num_nodes)  # space
    x[:, 2] = 2000 + 500 * torch.randn(num_nodes)  # flow
    
    edge_index = torch.randint(0, num_nodes, (2, 20))
    data = Data(x=x, edge_index=edge_index)
    
    # Create predictions (water_level, sediment)
    predictions = torch.randn(num_nodes, 2, requires_grad=True)
    predictions[:, 0] = torch.abs(predictions[:, 0]) + 10  # Ensure positive depth
    
    # Initialize physics loss
    physics_loss = SaintVenantLoss(g=9.8, n=0.03, S0=0.001, B=50.0)
    
    # Compute loss
    loss, components = physics_loss(predictions, data, return_components=True)
    
    print(f"Total Physics Loss: {loss.item():.6f}")
    print(f"Continuity Loss: {components['continuity_loss']:.6f}")
    print(f"Momentum Loss: {components['momentum_loss']:.6f}")
    print(f"Mean Velocity: {components['velocity'].mean().item():.2f} m/s")
    
    # Test gradients flow
    loss.backward()
    assert predictions.grad is not None
    print(f"✓ Gradients computed: mean={predictions.grad.mean().item():.6f}")
    
    print("✓ All physics loss tests passed!")
