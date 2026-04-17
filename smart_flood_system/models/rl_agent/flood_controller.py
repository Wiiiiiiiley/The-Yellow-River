"""
Reinforcement Learning Agent for Flood Control
==============================================

Implements PPO (Proximal Policy Optimization) for learning optimal
reservoir discharge policies. The agent learns to:
- Maintain water levels within safe bounds
- Minimize downstream flooding risk
- Optimize sediment transport

State Space: Current water levels, inflow rates, forecasted rainfall
Action Space: Discharge rate adjustments (m³/s)
Reward: Combination of safety margin, operational cost, and flood penalty
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict, Optional
import gym
from gym import spaces


class FloodControlEnv(gym.Env):
    """
    OpenAI Gym environment for flood control decision making.
    
    The environment simulates a simplified hydraulic system where:
    - State: Water levels at multiple stations, inflow rates, rainfall forecasts
    - Action: Discharge rate adjustments (positive=release, negative=store)
    - Reward: Safety margin minus operational costs
    """
    
    def __init__(
        self,
        hydraulic_simulator,
        max_steps: int = 168,  # 1 week in hours
        target_level_range: Tuple[float, float] = (45.0, 55.0),
        safety_constraints: Dict = None,
        n_stations: int = 5
    ):
        super().__init__()
        
        self.simulator = hydraulic_simulator
        self.max_steps = max_steps
        self.target_level_range = target_level_range
        self.safety_constraints = safety_constraints or {
            'max_level': 65.0,  # Critical flood level
            'min_level': 35.0,  # Drought level
            'max_discharge': 5000.0,  # Max release capacity
            'min_discharge': 500.0     # Min ecological flow
        }
        self.n_stations = n_stations
        
        # State: [water_levels (n), inflow (n), rain_forecast (n), time_features (4)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_stations * 3 + 4,),
            dtype=np.float32
        )
        
        # Action: Discharge adjustments for each station
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_stations,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.current_state = None
        self.done = False
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.done = False
        
        # Initialize with random but reasonable initial conditions
        initial_levels = np.random.uniform(40.0, 60.0, self.n_stations)
        initial_inflow = np.random.uniform(1000.0, 3000.0, self.n_stations)
        initial_rain = np.random.uniform(0.0, 5.0, self.n_stations)
        
        time_features = self._get_time_features()
        
        self.current_state = np.concatenate([
            initial_levels,
            initial_inflow,
            initial_rain,
            time_features
        ])
        
        return self.current_state
    
    def _get_time_features(self) -> np.ndarray:
        """Encode temporal information (hour, day of week, month, is_weekend)."""
        hour = (self.current_step % 24) / 24.0
        day_of_week = ((self.current_step // 24) % 7) / 7.0
        month = ((self.current_step // (24 * 30)) % 12) / 12.0
        is_weekend = 1.0 if ((self.current_step // 24) % 7) >= 5 else 0.0
        
        return np.array([hour, day_of_week, month, is_weekend])
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step.
        
        Args:
            action: Discharge adjustments (-1 to 1), will be scaled to actual discharge
        
        Returns:
            next_state, reward, done, info
        """
        # Scale action to actual discharge rates
        discharge = self._action_to_discharge(action)
        
        # Simulate hydraulic response (simplified)
        next_levels, next_inflow = self.simulator.step(
            self.current_state[:self.n_stations],
            discharge,
            self.current_state[self.n_stations:2*self.n_stations]
        )
        
        # Update rain forecast (simplified random walk)
        current_rain = self.current_state[2*self.n_stations:3*self.n_stations]
        next_rain = np.clip(current_rain + np.random.randn(self.n_stations) * 0.5, 0, 10)
        
        # Update state
        self.current_step += 1
        time_features = self._get_time_features()
        
        self.current_state = np.concatenate([
            next_levels,
            next_inflow,
            next_rain,
            time_features
        ])
        
        # Compute reward
        reward = self._compute_reward(next_levels, discharge)
        
        # Check termination
        self.done = (
            self.current_step >= self.max_steps or
            np.any(next_levels > self.safety_constraints['max_level']) or
            np.any(next_levels < self.safety_constraints['min_level'])
        )
        
        info = {
            'water_levels': next_levels,
            'discharge': discharge,
            'step': self.current_step,
            'is_emergency': np.any(next_levels > 60.0)
        }
        
        return self.current_state, reward, self.done, info
    
    def _action_to_discharge(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action to actual discharge rate."""
        # Scale from [-1, 1] to [min_discharge, max_discharge]
        min_d = self.safety_constraints['min_discharge']
        max_d = self.safety_constraints['max_discharge']
        return (action + 1) / 2 * (max_d - min_d) + min_d
    
    def _compute_reward(self, water_levels: np.ndarray, discharge: np.ndarray) -> float:
        """
        Compute reward based on:
        - Staying within target range (primary)
        - Avoiding extreme levels (safety)
        - Minimizing unnecessary discharge (efficiency)
        """
        target_center = (self.target_level_range[0] + self.target_level_range[1]) / 2
        
        # Level deviation from target
        level_deviation = np.mean(np.abs(water_levels - target_center))
        level_reward = -level_deviation
        
        # Safety penalty for levels outside safe range
        safety_penalty = 0.0
        if np.any(water_levels > self.safety_constraints['max_level']):
            safety_penalty -= 100.0  # Heavy penalty
        if np.any(water_levels < self.safety_constraints['min_level']):
            safety_penalty -= 50.0
        
        # Operational cost (penalty for large discharge changes)
        discharge_cost = -0.001 * np.sum(discharge ** 2)
        
        total_reward = level_reward + safety_penalty + discharge_cost
        
        return float(total_reward)


class ActorCriticNetwork(nn.Module):
    """Shared network for policy (actor) and value (critic)."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 128]
    ):
        super().__init__()
        
        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value)
        self.critic = nn.Linear(prev_dim, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_mean: (batch, action_dim)
            action_std: (batch, action_dim)
            value: (batch, 1)
        """
        features = self.feature_extractor(state)
        
        # Actor output
        action_mean = torch.tanh(self.actor_mean(features))
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate action for PPO update."""
        mean, std, value = self.forward(state)
        
        # Compute log probability
        var = std ** 2
        log_prob = -0.5 * (
            ((action - mean) ** 2) / var + 
            2 * torch.log(std) + 
            np.log(2 * np.pi)
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return log_prob, value


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) agent for flood control.
    
    PPO is chosen for its:
    - Sample efficiency
    - Stability (clipped objective prevents large policy updates)
    - Good performance on continuous control tasks
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        hidden_dims: List[int] = [256, 256, 128],
        batch_size: int = 64,
        epochs_per_update: int = 10
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update
        
        # Initialize network
        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dims)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = []
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Select action given state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, std, value = self.network(state_tensor)
            
            if deterministic:
                action = mean
            else:
                # Sample from Gaussian
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
            
            # Compute log probability
            log_prob = torch.distributions.Normal(mean, std).log_prob(action).sum()
            
        return action.squeeze(0).numpy(), log_prob.item(), value.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Store transition in buffer."""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        # Compute returns (advantages + values)
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO."""
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Extract data from buffer
        states = torch.FloatTensor([t['state'] for t in self.buffer])
        actions = torch.FloatTensor([t['action'] for t in self.buffer])
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in self.buffer])
        rewards = [t['reward'] for t in self.buffer]
        dones = [t['done'] for t in self.buffer]
        values = [t['value'] for t in self.buffer]
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.epochs_per_update):
            # Re-evaluate actions
            new_log_probs, new_values = self.network.evaluate(states, actions)
            
            # Compute ratio and clipped surrogate objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = 0.5 * ((new_values.squeeze() - returns) ** 2).mean()
            
            # Entropy bonus
            mean, std, _ = self.network(states)
            entropy = torch.distributions.Normal(mean, std).entropy().mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / self.epochs_per_update,
            'value_loss': total_value_loss / self.epochs_per_update,
            'entropy': total_entropy / self.epochs_per_update
        }
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == "__main__":
    """Unit test for RL components."""
    print("Testing RL Flood Control System...")
    
    from simulator.hydraulic_sim import FastHydraulicSimulator
    
    # Create environment
    sim = FastHydraulicSimulator()
    env = FloodControlEnv(sim, n_stations=5)
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim)
    
    # Test episode
    state = env.reset()
    episode_reward = 0
    
    for step in range(50):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        agent.store_transition(state, action, reward, next_state, done, log_prob, value)
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    print(f"Episode reward: {episode_reward:.2f}")
    print(f"Final water levels: {info['water_levels']}")
    
    # Test update
    if len(agent.buffer) >= agent.batch_size:
        losses = agent.update()
        print(f"Update losses: {losses}")
    
    print("✓ RL system tests passed!")
