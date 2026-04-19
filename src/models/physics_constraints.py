"""
PINN 物理约束损失函数
基于 Saint-Venant 方程的物理信息神经网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PhysicsConstraintLoss(nn.Module):
    """
    Saint-Venant 方程物理约束
    
    连续性方程: ∂h/∂t + ∂(hu)/∂x = 0
    动量方程: ∂u/∂t + u·∂u/∂x = -g·∂h/∂x - f·u + g·S0
    
    其中:
        h: 水深
        u: 流速
        g: 重力加速度（9.81 m/s²）
        f: 摩擦系数
        S0: 河床坡度
    """
    
    def __init__(self, 
                 manning_coefficient: float = 0.03,
                 bed_slope: float = 0.0001,
                 gravity: float = 9.81,
                 river_width: float = 500.0,
                 adaptive_weight: bool = True):
        super(PhysicsConstraintLoss, self).__init__()
        
        self.manning = manning_coefficient
        self.S0 = bed_slope
        self.g = gravity
        self.B = river_width
        self.adaptive_weight = adaptive_weight
        
        # 权重调度器
        self.register_buffer('physics_weight', torch.tensor(0.5))
    
    def compute_continuity_loss(self, h: torch.Tensor, u: torch.Tensor,
                               dt: float = 1.0, dx: float = 1000.0) -> torch.Tensor:
        """
        计算连续性方程损失
        ∂h/∂t + ∂(hu)/∂x = 0
        
        Args:
            h: 水位 (batch_size, seq_len, n_nodes)
            u: 流速 (batch_size, seq_len, n_nodes)
            dt: 时间步长
            dx: 空间步长
        """
        # 数值差分
        dh_dt = (h[:, 1:, :] - h[:, :-1, :]) / dt  # ∂h/∂t
        
        hu = h * u  # 流量
        dhu_dx = (hu[:, :, 1:] - hu[:, :, :-1]) / dx  # ∂(hu)/∂x
        
        # 连续性方程：应该满足 ∂h/∂t + ∂(hu)/∂x ≈ 0
        continuity_error = dh_dt[:, :, :-1] + dhu_dx[:, 1:, :]
        
        continuity_loss = torch.mean(continuity_error ** 2)
        
        return continuity_loss
    
    def compute_momentum_loss(self, u: torch.Tensor, h: torch.Tensor,
                             dt: float = 1.0, dx: float = 1000.0) -> torch.Tensor:
        """
        计算动量方程损失
        ∂u/∂t + u·∂u/∂x = -g·∂h/∂x - friction + g·S0
        
        Args:
            u: 流速 (batch_size, seq_len, n_nodes)
            h: 水位 (batch_size, seq_len, n_nodes)
            dt: 时间步长
            dx: 空间步长
        """
        # 时间导数
        du_dt = (u[:, 1:, :] - u[:, :-1, :]) / dt  # ∂u/∂t
        
        # 对流项
        du_dx = (u[:, :, 1:] - u[:, :, :-1]) / dx  # ∂u/∂x
        convection = u[:, :, :-1] * du_dx[:, 1:, :]  # u·∂u/∂x
        
        # 压力梯度
        dh_dx = (h[:, :, 1:] - h[:, :, :-1]) / dx  # ∂h/∂x
        pressure_gradient = -self.g * dh_dx[:, 1:, :]
        
        # 摩擦损失（曼宁公式）
        # f·u = (g·n²·u²) / h^(1/3)
        u_mid = (u[:, :, :-1] + u[:, :, 1:]) / 2
        h_mid = (h[:, :, :-1] + h[:, :, 1:]) / 2
        
        # 避免数值问题
        h_mid = torch.clamp(h_mid, min=0.1)
        
        friction = -(self.g * self.manning ** 2 * torch.abs(u_mid) * u_mid[1:, :]) / (h_mid[1:, :] ** (1/3))
        
        # 河床坡度项
        bed_slope_term = self.g * self.S0
        
        # 动量方程应该满足：
        # ∂u/∂t + u·∂u/∂x ≈ -g·∂h/∂x - friction + g·S0
        momentum_rhs = pressure_gradient + friction + bed_slope_term
        momentum_error = du_dt[:, :, :-1] + convection - momentum_rhs
        
        momentum_loss = torch.mean(momentum_error ** 2)
        
        return momentum_loss
    
    def compute_residuals(self, h: torch.Tensor, u: torch.Tensor,
                         dt: float = 1.0, dx: float = 1000.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算物理残差"""
        continuity_residual = self.compute_continuity_loss(h, u, dt, dx)
        momentum_residual = self.compute_momentum_loss(u, h, dt, dx)
        
        return continuity_residual, momentum_residual
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
               data_loss: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        整体 PINN 损失函数
        
        Args:
            predictions: 模型预测 (batch_size, pred_horizon, n_nodes, features)
                        features: [water_level, discharge, sediment, temp, rainfall]
            targets: 真实值 (batch_size, pred_horizon, n_nodes, features)
            data_loss: 数据驱动损失 (MSE)
        
        Returns:
            total_loss: 总损失
            loss_dict: 损失分量字典
        """
        # 提取水文变量
        # 假设: features 0=water_level, 1=discharge (Q), 2=sediment
        h = predictions[:, :, :, 0]  # 水位
        Q = predictions[:, :, :, 1]  # 流量
        
        # 流速 u = Q / (h * B)
        h_clamp = torch.clamp(h, min=0.1)
        u = Q / (h_clamp * self.B)
        
        # 计算物理损失
        continuity_loss = self.compute_continuity_loss(h, u)
        momentum_loss = self.compute_momentum_loss(u, h)
        
        # 平均物理损失
        physics_loss = (continuity_loss + momentum_loss) / 2
        
        # 加权组合
        if self.adaptive_weight:
            # 自适应加权：如果数据误差大，增加物理权重
            physics_weight = self.physics_weight.item()
            if data_loss.item() > 0.5:
                physics_weight = min(physics_weight * 1.1, 1.0)
            else:
                physics_weight = max(physics_weight * 0.9, 0.1)
            self.physics_weight.data.fill_(physics_weight)
        else:
            physics_weight = 0.5
        
        data_weight = 1.0 - physics_weight
        
        total_loss = data_weight * data_loss + physics_weight * physics_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'continuity_loss': continuity_loss.item(),
            'momentum_loss': momentum_loss.item(),
            'physics_weight': physics_weight,
            'data_weight': data_weight,
        }
        
        return total_loss, loss_dict
    
    def validate_physics(self, h: torch.Tensor, u: torch.Tensor,
                        tolerance: float = 0.1) -> dict:
        """
        验证预测是否满足物理约束
        
        Returns:
            validation_result: 包含满足度指标的字典
        """
        continuity_residual, momentum_residual = self.compute_residuals(h, u)
        
        # 标准化
        continuity_residual_norm = continuity_residual.item()
        momentum_residual_norm = momentum_residual.item()
        
        # 判断是否满足
        continuity_satisfied = continuity_residual_norm < tolerance
        momentum_satisfied = momentum_residual_norm < tolerance
        
        return {
            'continuity_residual': continuity_residual_norm,
            'momentum_residual': momentum_residual_norm,
            'continuity_satisfied': continuity_satisfied,
            'momentum_satisfied': momentum_satisfied,
            'overall_satisfied': continuity_satisfied and momentum_satisfied,
            'physics_quality': 1.0 - min((continuity_residual_norm + momentum_residual_norm) / 2, 1.0)
        }


class CoarseningPhysicsLoss(PhysicsConstraintLoss):
    """
    多尺度物理约束损失
    在不同粗糙度级别验证物理约束
    """
    
    def __init__(self, *args, n_scales: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_scales = n_scales
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
               data_loss: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """在多个尺度上计算物理损失"""
        
        total_physics_loss = torch.tensor(0.0, device=predictions.device)
        loss_scales = {}
        
        for scale in range(1, self.n_scales + 1):
            # 粗糙化预测
            if scale == 1:
                pred_coarse = predictions
            else:
                # 时间上的粗糙化
                pred_coarse = predictions[:, ::scale, :, :]
            
            h = pred_coarse[:, :, :, 0]
            Q = pred_coarse[:, :, :, 1]
            h_clamp = torch.clamp(h, min=0.1)
            u = Q / (h_clamp * self.B)
            
            # 物理损失在此尺度
            continuity = self.compute_continuity_loss(h, u, dt=scale)
            momentum = self.compute_momentum_loss(u, h, dt=scale)
            
            scale_loss = (continuity + momentum) / 2
            total_physics_loss = total_physics_loss + scale_loss
            
            loss_scales[f'scale_{scale}'] = scale_loss.item()
        
        # 平均物理损失
        physics_loss = total_physics_loss / self.n_scales
        
        # 加权组合
        physics_weight = 0.5
        data_weight = 0.5
        
        total_loss = data_weight * data_loss + physics_weight * physics_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            **loss_scales,
            'physics_weight': physics_weight,
        }
        
        return total_loss, loss_dict


class SeasonalPhysicsConstraint(PhysicsConstraintLoss):
    """
    带季节校准的物理约束
    黄河"大水大沙"特征需要调整参数
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 季节参数
        self.register_buffer('seasonal_manning', torch.tensor([0.025, 0.030, 0.035, 0.040]))  # 淡水/汛期
        self.register_buffer('seasonal_slope', torch.tensor([0.0001, 0.00015, 0.0002, 0.00012]))
    
    def get_seasonal_params(self, month: int) -> Tuple[float, float]:
        """根据月份获取季节参数"""
        # 分为 4 个季节
        season_idx = (month - 1) // 3
        season_idx = min(season_idx, 3)
        
        manning = self.seasonal_manning[season_idx].item()
        slope = self.seasonal_slope[season_idx].item()
        
        return manning, slope
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
               data_loss: torch.Tensor, month: int = 7) -> Tuple[torch.Tensor, dict]:
        """计算带季节调整的物理损失"""
        
        # 更新季节参数
        manning, slope = self.get_seasonal_params(month)
        self.manning = manning
        self.S0 = slope
        
        # 调用父类方法
        return super().forward(predictions, targets, data_loss)


if __name__ == '__main__':
    # 测试
    batch_size = 4
    seq_len = 6
    n_nodes = 5
    n_features = 5
    
    physics_loss = PhysicsConstraintLoss(
        manning_coefficient=0.03,
        bed_slope=0.0001,
        river_width=500.0
    )
    
    # 模拟数据
    predictions = torch.randn(batch_size, seq_len, n_nodes, n_features)
    predictions[:, :, :, 0] = 1000 + torch.abs(predictions[:, :, :, 0]) * 100  # 水位 1000~1100
    predictions[:, :, :, 1] = 1500 + torch.abs(predictions[:, :, :, 1]) * 500  # 流量 1500~2000
    
    targets = predictions + torch.randn_like(predictions) * 0.1
    
    data_loss = F.mse_loss(predictions, targets)
    
    # 计算物理约束
    total_loss, loss_dict = physics_loss(predictions, targets, data_loss)
    
    print("损失分量:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    # 验证物理约束
    h = predictions[:, :, :, 0]
    Q = predictions[:, :, :, 1]
    u = Q / h
    
    validation = physics_loss.validate_physics(h, u)
    print("\n物理约束满足情况:")
    for k, v in validation.items():
        print(f"  {k}: {v}")
