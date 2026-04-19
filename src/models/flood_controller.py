"""
RL 强化学习控制器 - PPO 算法
用于学习最优泄洪策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ActorNetwork(nn.Module):
    """策略网络 - 输出泄洪动作"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dim: int = 256, dropout: float = 0.1):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出均值和标准差
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self.mean.weight.data.mul_(0.1)
        self.mean.bias.data.mul_(0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch_size, state_dim)
        
        Returns:
            mean: (batch_size, action_dim)
            std: (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        mean = self.mean(x)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        
        return mean, std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """从高斯分布采样动作"""
        mean, std = self.forward(state)
        
        if deterministic:
            return mean, torch.log(std)
        
        # 从 N(mean, std^2) 采样
        action = torch.normal(mean, std)
        
        # 计算 log_prob
        log_prob = self._compute_log_prob(action, mean, std)
        
        return action, log_prob
    
    def _compute_log_prob(self, action: torch.Tensor, mean: torch.Tensor, 
                         std: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率"""
        var = std ** 2
        log_prob = -0.5 * ((action - mean) ** 2 / var + 2 * torch.log(std))
        log_prob = log_prob.sum(-1)
        return log_prob


class CriticNetwork(nn.Module):
    """价值网络 - 估计状态价值"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch_size, state_dim)
        
        Returns:
            value: (batch_size, 1)
        """
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        value = self.value(x)
        
        return value


class PPOAgent:
    """PPO (Proximal Policy Optimization) 智能体"""
    
    def __init__(self,
                 state_dim: int = 19,
                 action_dim: int = 5,
                 hidden_dim: int = 256,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 entropy_coeff: float = 0.01,
                 device: str = 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        
        # 网络
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # 经验缓冲区
        self.states = deque(maxlen=2000)
        self.actions = deque(maxlen=2000)
        self.rewards = deque(maxlen=2000)
        self.values = deque(maxlen=2000)
        self.log_probs = deque(maxlen=2000)
        self.dones = deque(maxlen=2000)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """选择动作"""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor, deterministic=deterministic)
            value = self.critic(state_tensor)
        
        action = action.squeeze(0).detach().cpu().numpy()
        log_prob = log_prob.squeeze(0).detach().cpu().numpy()
        value = value.squeeze(0).item()
        
        # 夹住动作在合理范围
        action = np.clip(action, -1.0, 1.0)
        
        return action, log_prob, value
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, value: float, log_prob: float, done: bool):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """计算广义优势估计 (GAE)"""
        
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # 添加下一个状态的价值
        values = np.append(values, next_value)
        
        # 计算 TD 残差
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]
        
        # 计算 GAE
        advantages = np.zeros_like(deltas)
        advantage = 0
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
            advantages[t] = advantage
        
        # 计算回报
        returns = advantages + values[:-1]
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, batch_size: int = 64, epochs: int = 3):
        """更新策略和价值网络"""
        
        if len(self.states) < batch_size:
            return
        
        # 转换为张量
        states = torch.from_numpy(np.array(list(self.states))).float().to(self.device)
        actions = torch.from_numpy(np.array(list(self.actions))).float().to(self.device)
        old_log_probs = torch.from_numpy(np.array(list(self.log_probs))).float().to(self.device)
        
        # 计算回报和优势
        returns, advantages = self.compute_gae()
        returns = torch.from_numpy(returns).float().to(self.device)
        advantages = torch.from_numpy(advantages).float().to(self.device)
        
        # 训练前的旧值（用于计算价值损失）
        with torch.no_grad():
            old_values = self.critic(states).squeeze(-1)
        
        # 多个 epoch 更新
        indices = np.arange(len(states))
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            
            for i in range(0, len(states), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # 计算新的 log_probs 和熵
                mean, std = self.actor.fc1 = nn.Linear(self.state_dim, 256)
                # 实际获取新概率
                _, new_log_probs = self.actor.get_action(batch_states)
                new_log_probs = new_log_probs.squeeze(-1)
                
                # 计算策略比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 裁剪的策略目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值网络损失
                new_values = self.critic(batch_states).squeeze(-1)
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # 总损失
                total_loss = policy_loss + 0.5 * value_loss
                
                # 更新策略网络
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # 更新价值网络
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
        
        # 清空缓冲区
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)
        logger.info(f"模型已保存到 {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        logger.info(f"模型已从 {path} 加载")


class FloodControlEnvironment:
    """泄洪控制环境模拟器"""
    
    def __init__(self, n_gates: int = 5):
        self.n_gates = n_gates
        self.current_water_level = np.array([1000.0] * n_gates)
        self.current_discharge = np.array([1500.0] * n_gates)
        self.current_step = 0
        self.max_steps = 168  # 一周
        
        # 安全约束
        self.safe_level = 1100.0
        self.critical_level = 1200.0
        self.min_discharge = 500.0
        self.max_discharge = 5000.0
    
    def reset(self):
        """重置环境"""
        self.current_water_level = np.random.uniform(950, 1050, self.n_gates)
        self.current_discharge = np.random.uniform(1000, 2000, self.n_gates)
        self.current_step = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """获取状态向量"""
        state = np.concatenate([
            self.current_water_level,  # 5
            self.current_discharge,     # 5
            np.random.randn(5),         # 预报降雨 5
            np.array([self.current_step / self.max_steps] * 4)  # 时间特征 4
        ])  # 总计 19 维
        
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """执行动作，返回新状态、奖励、是否结束"""
        
        # 动作：调整各闸门的泄洪
        discharge_adjustment = action * 500  # 最多调整 ±500
        new_discharge = self.current_discharge + discharge_adjustment
        new_discharge = np.clip(new_discharge, self.min_discharge, self.max_discharge)
        
        # 简单水温动态
        inflow = np.random.uniform(1000, 2000, self.n_gates)
        self.current_discharge = 0.7 * self.current_discharge + 0.3 * new_discharge
        
        # 水位变化
        water_change = (inflow - self.current_discharge) / 100  # 简化模型
        self.current_water_level += water_change
        self.current_water_level = np.clip(self.current_water_level, 800, 1500)
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 检查是否完成
        done = self.current_step >= self.max_steps
        self.current_step += 1
        
        return self._get_state(), reward, done
    
    def _compute_reward(self) -> float:
        """计算结论"""
        
        reward = 0.0
        
        # 1. 防洪安全奖励
        for level in self.current_water_level:
            if level < self.safe_level:
                reward += 1.0  # 安全
            elif level < self.critical_level:
                reward -= 2.0  # 警告
            else:
                reward -= 10.0  # 危险
        
        # 2. 运营成本（泄洪）
        discharge_cost = np.mean(self.current_discharge) / 1000
        reward -= discharge_cost * 0.1
        
        # 3. 流量平衡度
        discharge_variance = np.std(self.current_discharge)
        if discharge_variance < 500:
            reward += 0.5
        
        return reward


if __name__ == '__main__':
    # 测试 PPO 智能体
    logging.basicConfig(level=logging.INFO)
    
    agent = PPOAgent(
        state_dim=19,
        action_dim=5,
        hidden_dim=128,
        learning_rate=3e-4
    )
    
    env = FloodControlEnvironment(n_gates=5)
    
    # 训练
    n_episodes = 10
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        
        for step in range(100):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.store_transition(state, action, reward, value, log_prob, done)
            episode_reward += reward
            
            if done:
                break
            
            state = next_state
        
        # 更新策略
        agent.update()
        
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")
