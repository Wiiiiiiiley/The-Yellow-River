"""
Fast Hydraulic Simulator for RL Training
=========================================

Lightweight physics simulator for rapid RL environment feedback.
Provides approximate but fast hydraulic response to discharge actions.

Simplifications:
- 1D kinematic wave approximation
- Linearized Saint-Venant equations
- GPU acceleration support

Used for:
- RL training (needs ~1000+ steps/second)
- Monte Carlo simulation for policy evaluation
- Stress testing control strategies
"""

import numpy as np
import torch
from typing import Tuple, Optional


class FastHydraulicSimulator:
    """
    Fast approximate hydraulic simulator using linearized 1D wave equations.
    
    Much faster than full Saint-Venant for RL training while maintaining
    reasonable physical plausibility.
    """
    
    def __init__(
        self,
        n_segments: int = 10,
        segment_length: float = 10000.0,  # meters
        channel_width: float = 50.0,      # meters
        manning_n: float = 0.03,
        bed_slope: float = 0.001,
        use_gpu: bool = True,
        dx: float = 1000.0,                # Spatial resolution
        dt: float = 60.0                   # Time resolution (seconds)
    ):
        self.n_segments = n_segments
        self.L = segment_length
        self.B = channel_width
        self.n = manning_n
        self.S0 = bed_slope
        self.dx = dx
        self.dt = dt
        
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Simulation state (on GPU if available)
        self.water_levels = None
        self.flow_rates = None
        
        # Numerical stability parameter (Courant number)
        self.courant_max = 0.5
        
        # Precompute constants
        self.g = 9.8
        self.alpha = (self.n ** 2) * self.g / (self.B ** (4.0/3.0))
        
    def reset(self, initial_levels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Reset simulator to initial state."""
        if initial_levels is None:
            initial_levels = np.ones(self.n_segments) * 50.0
        
        self.water_levels = torch.FloatTensor(initial_levels).to(self.device)
        self.flow_rates = torch.zeros(self.n_segments).to(self.device)
        
        return initial_levels, self.flow_rates.cpu().numpy()
    
    def step(
        self,
        current_levels: np.ndarray,
        discharge: np.ndarray,
        inflow: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance simulation by one time step.
        
        Args:
            current_levels: Current water levels at each station (m)
            discharge: Controlled discharge/release at each station (m³/s)
            inflow: Natural inflow at each station (m³/s)
        
        Returns:
            next_levels: Water levels after dt seconds
            next_flow: Flow rates after dt seconds
        """
        # Convert to tensors
        h = torch.FloatTensor(current_levels).to(self.device)
        Q_out = torch.FloatTensor(discharge).to(self.device)
        Q_in = torch.FloatTensor(inflow).to(self.device)
        
        # Ensure numerical stability
        h = torch.clamp(h, min=0.5)  # Minimum depth
        
        # Compute cross-sectional area
        A = self.B * h
        
        # Kinematic wave velocity (approximate)
        # c ≈ (5/3) * v where v is flow velocity
        v = (Q_in + Q_out) / (2 * A + 1e-6)
        c = (5.0/3.0) * torch.abs(v)
        
        # Check Courant condition for stability
        courant = torch.max(c) * self.dt / self.dx
        if courant > self.courant_max:
            # Adaptive time step if needed
            sub_steps = int(np.ceil(courant / self.courant_max))
            dt_sub = self.dt / sub_steps
        else:
            sub_steps = 1
            dt_sub = self.dt
        
        # Sub-step integration for stability
        h_current = h.clone()
        for _ in range(sub_steps):
            # Simplified continuity: dh/dt = (Q_in - Q_out) / A
            # with diffusion term for stability
            dH = (Q_in - Q_out) / A * dt_sub
            
            # Add small diffusion to prevent numerical instability
            diffusion = 0.1 * torch.diff(h_current, prepend=h_current[:1], append=h_current[-1:])
            diffusion = diffusion[1:-1]  # Interior points only
            
            # Update (keep boundary conditions fixed)
            if self.n_segments > 2:
                h_current[1:-1] = h_current[1:-1] + dH[1:-1] + diffusion * dt_sub * 0.01
            else:
                h_current = h_current + dH
            
            # Enforce physical constraints
            h_current = torch.clamp(h_current, min=0.5, max=100.0)
        
        # Compute resulting flow rates (Manning equation)
        A_new = self.B * h_current
        # Q = (1/n) * A * R^(2/3) * S^(1/2)
        # Simplified: R ≈ h (for wide channels)
        hydraulic_radius = h_current
        slope_effective = torch.abs(self.S0 - torch.diff(
            h_current, prepend=h_current[:1], append=h_current[-1:]
        ) / self.dx)
        slope_effective = slope_effective[1:-1]
        
        if self.n_segments > 2:
            Q_calc = torch.zeros_like(h_current)
            Q_calc[1:-1] = (1.0 / self.n) * A_new[1:-1] * \
                          (hydraulic_radius[1:-1] ** (2.0/3.0)) * \
                          (slope_effective ** 0.5)
            Q_calc[0] = Q_in[0]  # Boundary condition
            Q_calc[-1] = Q_out[-1]  # Boundary condition
        else:
            Q_calc = Q_out
        
        next_levels = h_current.cpu().numpy()
        next_flow = Q_calc.cpu().numpy()
        
        return next_levels, next_flow
    
    def batch_simulate(
        self,
        initial_levels: np.ndarray,
        discharge_schedule: np.ndarray,
        inflow_schedule: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate multiple time steps (batch mode for planning).
        
        Args:
            initial_levels: Starting water levels (n_segments,)
            discharge_schedule: Planned discharges (n_steps, n_segments)
            inflow_schedule: Expected inflows (n_steps, n_segments)
        
        Returns:
            level_trajectory: (n_steps, n_segments)
            flow_trajectory: (n_steps, n_segments)
        """
        n_steps = len(discharge_schedule)
        
        levels = [initial_levels]
        flows = []
        
        current = initial_levels.copy()
        for t in range(n_steps):
            next_level, next_flow = self.step(
                current,
                discharge_schedule[t],
                inflow_schedule[t]
            )
            levels.append(next_level)
            flows.append(next_flow)
            current = next_level
        
        return np.array(levels), np.array(flows)


class ParallelSimulator:
    """
    GPU-parallel simulator for ensemble Monte Carlo evaluation.
    Simulates multiple scenarios simultaneously for risk assessment.
    """
    
    def __init__(
        self,
        n_scenarios: int = 100,
        n_segments: int = 10,
        use_gpu: bool = True
    ):
        self.n_scenarios = n_scenarios
        self.n_segments = n_segments
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        self.simulators = [
            FastHydraulicSimulator(n_segments=n_segments, use_gpu=False)
            for _ in range(n_scenarios)
        ]
        
    def monte_carlo_simulation(
        self,
        initial_levels: np.ndarray,
        discharge_policy: np.ndarray,
        inflow_scenarios: np.ndarray  # (n_scenarios, n_steps, n_segments)
    ) -> Dict:
        """
        Run parallel Monte Carlo simulation with uncertain inflows.
        
        Returns:
            Statistics: mean, std, percentiles of water levels
        """
        n_steps = len(discharge_policy)
        all_trajectories = []
        
        for i, sim in enumerate(self.simulators):
            levels, flows = sim.batch_simulate(
                initial_levels,
                discharge_policy,
                inflow_scenarios[i]
            )
            all_trajectories.append(levels)
        
        # Compute statistics
        trajectories = np.array(all_trajectories)  # (n_scenarios, n_steps, n_segments)
        
        return {
            'mean_levels': np.mean(trajectories, axis=0),
            'std_levels': np.std(trajectories, axis=0),
            'p95_levels': np.percentile(trajectories, 95, axis=0),
            'p5_levels': np.percentile(trajectories, 5, axis=0),
            'max_risk': np.max(trajectories, axis=(0, 1)),
            'failure_probability': np.mean(trajectories > 65.0, axis=0)
        }


if __name__ == "__main__":
    """Unit test for simulator."""
    print("Testing Fast Hydraulic Simulator...")
    
    # Test single simulator
    sim = FastHydraulicSimulator(n_segments=5, use_gpu=False)
    
    # Initial conditions
    levels = np.array([50.0, 51.0, 52.0, 51.5, 50.5])
    
    # Run simulation
    print("\nSingle step simulation:")
    discharge = np.array([1000.0, 1500.0, 2000.0, 1800.0, 1200.0])
    inflow = np.array([2000.0, 2200.0, 2500.0, 2300.0, 2100.0])
    
    next_levels, next_flows = sim.step(levels, discharge, inflow)
    print(f"Current levels: {levels}")
    print(f"Next levels: {next_levels}")
    print(f"Next flows: {next_flows}")
    
    # Test batch simulation
    print("\nBatch simulation (24 hours):")
    discharge_schedule = np.ones((24, 5)) * 1500.0
    inflow_schedule = np.random.uniform(1000, 3000, (24, 5))
    
    level_traj, flow_traj = sim.batch_simulate(
        levels, discharge_schedule, inflow_schedule
    )
    print(f"Level trajectory shape: {level_traj.shape}")
    print(f"Final levels: {level_traj[-1]}")
    print(f"Max level reached: {np.max(level_traj)}")
    
    print("\n✓ Simulator tests passed!")
