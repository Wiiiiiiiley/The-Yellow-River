"""
增强版 PINN 物理约束损失函数 v2.0
包含: Exner泥沙方程, NTK自适应权重, 高阶差分, 课程学习, 边界条件, 多保真度

改进内容:
1. 泥沙输运物理 (Exner方程 + Meyer-Peter Müller公式)
2. NTK神经正切核自适应权重
3. 四阶中心差分提高数值精度
4. 特征尺度归一化
5. 课程学习渐进式约束
6. 边界条件强制
7. 多保真度训练支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List


class SedimentTransportLoss(nn.Module):
    """
    泥沙输运物理约束 - 黄河"大水大沙"特性建模
    
    Exner 方程: ∂z_b/∂t + (1/(1-p)) * ∂q_s/∂x = 0
    
    Meyer-Peter & Müller 输沙公式:
        q_s = 8 * (τ - τ_c)^(3/2) / (ρ_s - ρ_w) * g^(1/2) * d^(-1/2)
    
    其中:
        z_b: 河床高程
        p: 孔隙率 (~0.4)
        q_s: 单宽输沙率 (m²/s)
        τ: 床面剪切应力 = ρ_w * g * h * S
        τ_c: 临界起动应力
        ρ_s: 泥沙密度 (~2650 kg/m³)
        ρ_w: 水密度 (~1000 kg/m³)
        d: 泥沙粒径
    """
    
    def __init__(self,
                 porosity: float = 0.4,
                 sediment_density: float = 2650.0,
                 water_density: float = 1000.0,
                 grain_diameter: float = 0.00025,  # 0.25mm 中值粒径
                 critical_shield_stress: float = 0.047,
                 gravity: float = 9.81):
        super(SedimentTransportLoss, self).__init__()
        
        self.p = porosity
        self.rho_s = sediment_density
        self.rho_w = water_density
        self.d = grain_diameter
        self.tau_c = critical_shield_stress * (self.rho_s - self.rho_w) * gravity * grain_diameter
        self.g = gravity
    
    def compute_bed_shear_stress(self, h: torch.Tensor, u: torch.Tensor, 
                                  manning_n: float = 0.03) -> torch.Tensor:
        """
        计算床面剪切应力 τ = ρ_w * u_*^2
        使用曼宁公式: u = (1/n) * R^(2/3) * S^(1/2)
        剪切流速: u_* = u * n / R^(1/6) * g^(1/2)
        """
        #  Hydraulic radius ≈ h (宽浅河道近似)
        R = torch.clamp(h, min=0.1)
        
        # 剪切流速
        u_star = u * manning_n / (R ** (1/6)) * torch.sqrt(torch.tensor(self.g))
        
        # 床面剪切应力
        tau = self.rho_w * u_star ** 2
        
        return tau
    
    def compute_mpm_transport_rate(self, h: torch.Tensor, u: torch.Tensor,
                                    slope: torch.Tensor) -> torch.Tensor:
        """
        Meyer-Peter & Müller 输沙率公式 (无量纲化)
        q_s* = 8 * (τ* - τ_c*)^(3/2)  (当 τ* > τ_c*)
        q_s* = 0                      (当 τ* <= τ_c*)
        
        返回: 单宽输沙率 q_s (m²/s)
        """
        # 计算剪切应力
        tau = self.compute_bed_shear_stress(h, u)
        
        # 无量纲剪切应力 τ*
        tau_star = tau / ((self.rho_s - self.rho_w) * self.g * self.d)
        
        # 临界无量纲剪切应力 (Shield 参数 ~0.047)
        tau_c_star = self.tau_c / ((self.rho_s - self.rho_w) * self.g * self.d)
        
        # MPM 公式
        excess_stress = torch.clamp(tau_star - tau_c_star, min=0.0)
        q_s_star = 8.0 * (excess_stress ** 1.5)
        
        # 转换为有量纲输沙率
        sqrt_term = torch.sqrt(torch.tensor(
            (self.rho_s - self.rho_w) / self.rho_w * self.g * self.d ** 3,
            device=q_s_star.device, dtype=q_s_star.dtype
        ))
        q_s = q_s_star * sqrt_term
        
        return q_s
    
    def compute_exner_loss(self, h: torch.Tensor, u: torch.Tensor,
                          z_b: torch.Tensor, dt: float = 1.0, 
                          dx: float = 1000.0,
                          slope: torch.Tensor = None) -> torch.Tensor:
        """
        Exner 方程损失: ∂z_b/∂t + (1/(1-p)) * ∂q_s/∂x = 0
        
        Args:
            h: 水深 (batch, seq_len, n_nodes)
            u: 流速
            z_b: 河床高程 (从水位反推: z_b = h - water_depth_ref)
            dt: 时间步长
            dx: 空间步长
            slope: 河床坡度 (可选，默认0.0001)
        """
        if slope is None:
            slope = torch.tensor(0.0001, device=h.device)
        
        # 计算输沙率
        q_s = self.compute_mpm_transport_rate(h, u, slope)
        
        # 时间导数 (河床演变)
        dz_dt = (z_b[:, 1:, :] - z_b[:, :-1, :]) / dt
        
        # 空间导数 (输沙梯度)
        dqs_dx = (q_s[:, :, 1:] - q_s[:, :, :-1]) / dx
        
        # Exner 方程残差
        # 注意: 维度对齐，dz_dt 和 dqs_dx 的时间维度需要匹配
        min_time = min(dz_dt.shape[1], dqs_dx.shape[1])
        exner_residual = dz_dt[:, :min_time, :-1] + dqs_dx[:, :min_time, :] / (1 - self.p)
        
        exner_loss = torch.mean(exner_residual ** 2)
        
        return exner_loss, q_s
    
    def compute_sediment_mass_balance(self, sediment_conc: torch.Tensor,
                                       h: torch.Tensor, u: torch.Tensor,
                                       dt: float = 1.0, dx: float = 1000.0) -> torch.Tensor:
        """
        悬移质泥沙质量守恒: ∂(h*C)/∂t + ∂(h*u*C)/∂x = 侵蚀/沉积率
        简化为对流-扩散形式
        """
        # 泥沙通量
        sediment_flux = h * u * sediment_conc
        
        # 时间导数
        d_hC_dt = (h[:, 1:, :] * sediment_conc[:, 1:, :] - 
                   h[:, :-1, :] * sediment_conc[:, :-1, :]) / dt
        
        # 空间导数
        d_flux_dx = (sediment_flux[:, :, 1:] - sediment_flux[:, :, :-1]) / dx
        
        # 质量守恒残差
        min_time = min(d_hC_dt.shape[1], d_flux_dx.shape[1])
        mass_residual = d_hC_dt[:, :min_time, :-1] + d_flux_dx[:, :min_time, :]
        
        return torch.mean(mass_residual ** 2)
    
    def forward(self, predictions: torch.Tensor, h: torch.Tensor, 
                u: torch.Tensor, dt: float = 1.0, dx: float = 1000.0) -> Tuple[torch.Tensor, Dict]:
        """
        计算泥沙输运损失
        
        Args:
            predictions: (batch, pred_horizon, n_nodes, features)
                        假设 features[2] = sediment_concentration
            h: 水深
            u: 流速
        """
        # 提取泥沙浓度
        sediment_conc = predictions[:, :, :, 2]  # 假设泥沙是第3个特征
        
        # 估算河床高程 (简化: 从水位减去参考水深)
        z_b = h - 10.0  # 假设平均水深 10m
        
        # Exner 方程损失
        exner_loss, q_s = self.compute_exner_loss(h, u, z_b, dt, dx)
        
        # 悬移质质量守恒
        mass_loss = self.compute_sediment_mass_balance(sediment_conc, h, u, dt, dx)
        
        # 总泥沙损失
        total_sediment_loss = exner_loss + 0.5 * mass_loss
        
        loss_dict = {
            'exner_loss': exner_loss.item(),
            'sediment_mass_loss': mass_loss.item(),
            'total_sediment_loss': total_sediment_loss.item(),
            'mean_transport_rate': torch.mean(q_s).item()
        }
        
        return total_sediment_loss, loss_dict


class EnhancedPhysicsConstraintLoss(nn.Module):
    """
    增强版物理约束损失 - 集成所有改进
    
    特性:
    1. NTK自适应权重
    2. 高阶数值差分 (四阶)
    3. 特征尺度归一化
    4. 课程学习
    5. 泥沙输运 (可选)
    """
    
    def __init__(self,
                 manning_coefficient: float = 0.03,
                 bed_slope: float = 0.0001,
                 gravity: float = 9.81,
                 river_width: float = 500.0,
                 use_ntk_adaptive: bool = True,
                 use_curriculum: bool = True,
                 use_high_order: bool = True,
                 use_normalization: bool = True,
                 use_sediment: bool = True,
                 total_epochs: int = 1000):
        super(EnhancedPhysicsConstraintLoss, self).__init__()
        
        self.manning = manning_coefficient
        self.S0 = bed_slope
        self.g = gravity
        self.B = river_width
        
        # 功能开关
        self.use_ntk = use_ntk_adaptive
        self.use_curriculum = use_curriculum
        self.use_high_order = use_high_order
        self.use_normalization = use_normalization
        self.use_sediment = use_sediment
        self.total_epochs = total_epochs
        
        # 特征尺度参考值 (黄河典型值)
        self.register_buffer('h_ref', torch.tensor(10.0))      # 水深 (m)
        self.register_buffer('u_ref', torch.tensor(2.0))       # 流速 (m/s)
        self.register_buffer('Q_ref', torch.tensor(2000.0))  # 流量 (m³/s)
        self.register_buffer('z_ref', torch.tensor(1000.0))  # 水位 (m)
        
        # 课程学习状态
        self.register_buffer('current_epoch', torch.tensor(0))
        
        # 泥沙损失模块
        if self.use_sediment:
            self.sediment_loss = SedimentTransportLoss()
    
    def set_epoch(self, epoch: int):
        """设置当前训练轮数 (用于课程学习)"""
        self.current_epoch.data.fill_(epoch)
    
    def get_curriculum_weight(self) -> float:
        """
        课程学习: 渐进式增加物理约束权重
        
        阶段1 (0-30%): 物理权重 0.1 (数据主导)
        阶段2 (30-70%): 物理权重线性增加到 0.9
        阶段3 (70-100%): 物理权重 0.9 (物理主导)
        """
        if not self.use_curriculum:
            return 0.5
        
        progress = self.current_epoch.item() / self.total_epochs
        
        if progress < 0.3:
            return 0.1
        elif progress < 0.7:
            return 0.1 + 0.8 * (progress - 0.3) / 0.4
        else:
            return 0.9
    
    def compute_high_order_time_derivative(self, field: torch.Tensor, 
                                           dt: float) -> torch.Tensor:
        """
        四阶中心差分计算时间导数 O(Δt^4)
        
        公式: f' ≈ (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / (12Δt)
        """
        if not self.use_high_order or field.shape[1] < 5:
            # 回退到二阶差分
            return (field[:, 2:, :] - field[:, :-2, :]) / (2 * dt)
        
        # 四阶中心差分
        df_dt = (-field[:, 4:, :] + 8*field[:, 3:-1, :] - 
                 8*field[:, 1:-3, :] + field[:, :-4, :]) / (12 * dt)
        
        return df_dt
    
    def compute_high_order_space_derivative(self, field: torch.Tensor,
                                            dx: float) -> torch.Tensor:
        """
        四阶中心差分计算空间导数
        """
        if not self.use_high_order or field.shape[2] < 5:
            # 回退到二阶
            return (field[:, :, 2:] - field[:, :, :-2]) / (2 * dx)
        
        # 四阶中心差分
        df_dx = (-field[:, :, 4:] + 8*field[:, :, 3:-1] - 
                 8*field[:, :, 1:-3] + field[:, :, :-4]) / (12 * dx)
        
        return df_dx
    
    def normalize_variables(self, h: torch.Tensor, u: torch.Tensor,
                            Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        特征尺度归一化
        """
        if not self.use_normalization:
            return h, u, Q
        
        h_norm = h / self.h_ref
        u_norm = u / self.u_ref
        Q_norm = Q / self.Q_ref
        
        return h_norm, u_norm, Q_norm
    
    def compute_ntk_weights(self, model: nn.Module, 
                            data_loss: torch.Tensor,
                            physics_loss: torch.Tensor) -> Tuple[float, float]:
        """
        NTK自适应权重: 平衡数据损失和物理损失的梯度贡献
        
        策略: 计算两种损失对模型参数的梯度范数，
              调整权重使它们的梯度贡献相当
        """
        if not self.use_ntk:
            return 0.5, 0.5
        
        parameters = [param for param in model.parameters() if param.requires_grad]

        def gradient_norm(loss: torch.Tensor) -> float:
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            norm_squared = torch.zeros((), device=loss.device)
            for gradient in gradients:
                if gradient is not None:
                    norm_squared = norm_squared + gradient.detach().pow(2).sum()
            return torch.sqrt(norm_squared).item()

        # optimizer.zero_grad() 会将 param.grad 设为 None，因此必须直接
        # 从两个损失计算梯度，不能依赖预先存在的 param.grad。
        data_grad_norm = gradient_norm(data_loss)
        physics_grad_norm = gradient_norm(physics_loss)

        if physics_grad_norm > 1e-8 and data_grad_norm > 1e-8:
            total_norm = data_grad_norm + physics_grad_norm
            lambda_data = physics_grad_norm / total_norm
            lambda_physics = data_grad_norm / total_norm
            # 防止单一损失在一个 batch 内完全主导训练。
            lambda_physics = min(max(lambda_physics, 0.05), 0.95)
            lambda_data = 1.0 - lambda_physics
            return lambda_data, lambda_physics
        
        return 0.5, 0.5
    
    def compute_continuity_loss_enhanced(self, h: torch.Tensor, 
                                          u: torch.Tensor,
                                          Q: torch.Tensor,
                                          dt: float = 1.0, 
                                          dx: float = 1000.0) -> torch.Tensor:
        """
        增强版连续性方程损失 (使用高阶差分和归一化)
        """
        # 归一化
        h_norm, u_norm, Q_norm = self.normalize_variables(h, u, Q)
        
        # 计算 hu
        hu = h_norm * u_norm
        
        # 高阶时间导数
        dh_dt = self.compute_high_order_time_derivative(h_norm, dt)
        
        # 高阶空间导数
        dhu_dx = self.compute_high_order_space_derivative(hu, dx)
        
        # 维度对齐
        min_time = min(dh_dt.shape[1], dhu_dx.shape[1])
        min_space = min(dh_dt.shape[2], dhu_dx.shape[2])
        
        # 连续性残差
        continuity_residual = dh_dt[:, :min_time, :min_space] + dhu_dx[:, :min_time, :min_space]
        
        return torch.mean(continuity_residual ** 2)
    
    def compute_momentum_loss_enhanced(self, u: torch.Tensor, h: torch.Tensor,
                                        dt: float = 1.0, dx: float = 1000.0) -> torch.Tensor:
        """
        增强版动量方程损失
        """
        # 归一化
        h_norm, u_norm, _ = self.normalize_variables(h, u, torch.zeros_like(h))
        
        # 高阶时间导数
        du_dt = self.compute_high_order_time_derivative(u_norm, dt)
        
        # 高阶空间导数 (对流项)
        du_dx = self.compute_high_order_space_derivative(u_norm, dx)
        
        # 需要维度对齐
        min_time = min(du_dt.shape[1], du_dx.shape[1])
        min_space = min(du_dt.shape[2], du_dx.shape[2])
        
        # 对流项 u * ∂u/∂x
        u_mid = u_norm[:, 1:1+min_time, 1:1+min_space]
        convection = u_mid * du_dx[:, :min_time, :min_space]
        
        # 压力梯度 (归一化动量方程系数: g*h_ref/u_ref^2)
        dh_dx = self.compute_high_order_space_derivative(h_norm, dx)
        pressure_gradient = -self.g * dh_dx[:, :min_time, :min_space] * (self.h_ref / self.u_ref**2)
        
        # 摩擦损失 (曼宁公式)
        h_mid = torch.clamp(h_norm[:, 1:1+min_time, 1:1+min_space], min=0.01)
        friction = -(self.g * self.manning ** 2 * torch.abs(u_mid) * u_mid) / (h_mid ** (1/3))
        
        # 河床坡度项
        bed_slope_term = self.g * self.S0 * self.h_ref / (self.u_ref ** 2)
        
        # 动量方程残差
        momentum_rhs = pressure_gradient + friction + bed_slope_term
        momentum_residual = du_dt[:, :min_time, :min_space] + convection - momentum_rhs
        
        return torch.mean(momentum_residual ** 2)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                data_loss: torch.Tensor, model: Optional[nn.Module] = None,
                dt: float = 1.0, dx: float = 1000.0) -> Tuple[torch.Tensor, Dict]:
        """
        增强版前向传播
        
        Args:
            predictions: 模型预测
            targets: 真实值
            data_loss: 数据驱动损失
            model: 模型实例 (用于NTK权重计算)
            dt: 时间步长
            dx: 空间步长
        """
        # 输入必须是已恢复量纲的物理量。数据集给出的是水位高程，
        # Saint-Venant 方程需要水深，因此用典型水位和典型水深转换。
        water_level = predictions[:, :, :, 0]
        h = torch.clamp(water_level - self.z_ref + self.h_ref, min=0.1)
        Q = predictions[:, :, :, 1]  # 流量
        
        # 计算流速 u = Q / (h * B)
        u = Q / (h * self.B)
        
        # 计算物理损失
        continuity_loss = self.compute_continuity_loss_enhanced(h, u, Q, dt, dx)
        momentum_loss = self.compute_momentum_loss_enhanced(u, h, dt, dx)
        
        # 基础物理损失
        base_physics_loss = (continuity_loss + momentum_loss) / 2
        
        # 泥沙损失 (可选)
        sediment_loss_val = torch.tensor(0.0, device=predictions.device)
        sediment_dict = {}
        if self.use_sediment and hasattr(self, 'sediment_loss'):
            sediment_loss_val, sediment_dict = self.sediment_loss(predictions, h, u, dt, dx)
        
        # 总物理损失
        physics_loss = base_physics_loss + 0.3 * sediment_loss_val
        
        # 计算权重
        if model is not None and self.use_ntk:
            data_weight, physics_weight = self.compute_ntk_weights(model, data_loss, physics_loss)
            if self.use_curriculum:
                # 课程学习限制早期物理项的最大权重。
                physics_weight = min(physics_weight, self.get_curriculum_weight())
                data_weight = 1.0 - physics_weight
        else:
            physics_weight = self.get_curriculum_weight()
            data_weight = 1.0 - physics_weight
        
        # 组合损失
        total_loss = data_weight * data_loss + physics_weight * physics_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'continuity_loss': continuity_loss.item(),
            'momentum_loss': momentum_loss.item(),
            'data_weight': data_weight,
            'physics_weight': physics_weight,
            'current_epoch': self.current_epoch.item(),
            **sediment_dict
        }
        
        return total_loss, loss_dict


class BoundaryConditionEnforcer:
    """
    边界条件强制器 - 硬约束实现
    
    黄河开边界条件:
    - 上游 (兰州): 流量边界 (Q 给定)
    - 下游 (花园口): 水位边界 (h 给定) 或 辐射边界
    """
    
    def __init__(self, upstream_station_idx: int = 0, 
                 downstream_station_idx: int = 4):
        self.upstream_idx = upstream_station_idx
        self.downstream_idx = downstream_station_idx
    
    def apply_upstream_flow(self, predictions: torch.Tensor, 
                           target_Q: torch.Tensor) -> torch.Tensor:
        """
        强制上游流量边界条件
        
        Args:
            predictions: (batch, pred_horizon, n_nodes, features)
            target_Q: 目标流量 (batch, pred_horizon) 或标量
        """
        predictions_enforced = predictions.clone()
        
        # 假设流量是第2个特征 (索引1)
        if isinstance(target_Q, (int, float)):
            predictions_enforced[:, :, self.upstream_idx, 1] = target_Q
        else:
            # target_Q: (batch, pred_horizon)
            predictions_enforced[:, :, self.upstream_idx, 1] = target_Q.unsqueeze(-1)
        
        return predictions_enforced
    
    def apply_downstream_level(self, predictions: torch.Tensor,
                               target_h: torch.Tensor) -> torch.Tensor:
        """
        强制下游水位边界条件
        """
        predictions_enforced = predictions.clone()
        
        # 假设水位是第1个特征 (索引0)
        if isinstance(target_h, (int, float)):
            predictions_enforced[:, :, self.downstream_idx, 0] = target_h
        else:
            predictions_enforced[:, :, self.downstream_idx, 0] = target_h.unsqueeze(-1)
        
        return predictions_enforced
    
    def apply_radiation_boundary(self, predictions: torch.Tensor,
                                  celerity: float = 5.0) -> torch.Tensor:
        """
        辐射边界条件 (特征线法)
        
        ∂Q/∂t + c * ∂Q/∂x = 0 (特征线速度 c = sqrt(g*h))
        
        用于下游边界允许扰动传出而不反射
        """
        # 简化为保持流量梯度平滑
        predictions_enforced = predictions.clone()
        
        # 获取下游最后两个节点的流量
        Q_downstream = predictions[:, :, self.downstream_idx, 1]
        Q_adjacent = predictions[:, :, self.downstream_idx - 1, 1]
        
        # 平滑过渡 (数值粘性)
        Q_smoothed = 0.5 * Q_downstream + 0.5 * Q_adjacent
        predictions_enforced[:, :, self.downstream_idx, 1] = Q_smoothed
        
        return predictions_enforced


class MultiFidelityPINN(nn.Module):
    """
    多保真度 PINN 训练器
    
    结合高保真 (观测数据) 和低保真 (简化模拟) 数据
    """
    
    def __init__(self, 
                 high_fidelity_config: Optional[Dict] = None,
                 low_fidelity_config: Optional[Dict] = None):
        super(MultiFidelityPINN, self).__init__()
        
        # 高保真损失 (精细物理, 小时间步长)
        hf_config = high_fidelity_config or {}
        self.hf_loss = EnhancedPhysicsConstraintLoss(
            use_high_order=True,
            use_sediment=True,
            **hf_config
        )
        
        # 低保真损失 (简化物理, 大时间步长)
        lf_config = low_fidelity_config or {}
        self.lf_loss = EnhancedPhysicsConstraintLoss(
            use_high_order=False,
            use_sediment=False,
            **lf_config
        )
        
        # 保真度权重 (可学习)
        self.hf_weight = nn.Parameter(torch.tensor(0.7))
        self.lf_weight = nn.Parameter(torch.tensor(0.3))
    
    def forward(self,
                pred_hf: torch.Tensor, target_hf: torch.Tensor,
                pred_lf: torch.Tensor, target_lf: torch.Tensor,
                data_loss_hf: torch.Tensor, data_loss_lf: torch.Tensor,
                model: nn.Module) -> Tuple[torch.Tensor, Dict]:
        """
        多保真度损失计算
        
        Args:
            pred_hf: 高保真预测 (观测数据)
            target_hf: 高保真目标
            pred_lf: 低保真预测 (模拟数据)
            target_lf: 低保真目标
        """
        # 高保真损失
        loss_hf, dict_hf = self.hf_loss(pred_hf, target_hf, data_loss_hf, model)
        
        # 低保真损失
        loss_lf, dict_lf = self.lf_loss(pred_lf, target_lf, data_loss_lf, model)
        
        # 可学习权重
        weights = torch.softmax(torch.stack([self.hf_weight, self.lf_weight]), dim=0)
        
        # 加权组合
        total_loss = weights[0] * loss_hf + weights[1] * loss_lf
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'hf_loss': loss_hf.item(),
            'lf_loss': loss_lf.item(),
            'hf_weight': weights[0].item(),
            'lf_weight': weights[1].item(),
            'hf_details': dict_hf,
            'lf_details': dict_lf
        }
        
        return total_loss, loss_dict


# 向后兼容: 导出原始类名
PhysicsConstraintLoss = EnhancedPhysicsConstraintLoss


if __name__ == '__main__':
    # 测试增强版 PINN
    print("=" * 60)
    print("测试增强版 PINN 物理约束损失 v2.0")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 12  # 需要至少 5 个时间步用于高阶差分
    n_nodes = 5
    n_features = 6
    
    # 创建测试数据
    predictions = torch.randn(batch_size, seq_len, n_nodes, n_features)
    predictions[:, :, :, 0] = 1000 + torch.abs(predictions[:, :, :, 0]) * 100  # 水位 1000~1100
    predictions[:, :, :, 1] = 1500 + torch.abs(predictions[:, :, :, 1]) * 500  # 流量 1500~2000
    predictions[:, :, :, 2] = 5 + torch.abs(predictions[:, :, :, 2]) * 15      # 泥沙浓度 5~20
    
    targets = predictions + torch.randn_like(predictions) * 0.1
    data_loss = F.mse_loss(predictions, targets)
    
    print(f"\n测试配置:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Nodes: {n_nodes}")
    print(f"  Features: {n_features}")
    
    # 1. 测试泥沙输运损失
    print("\n" + "-" * 40)
    print("1. 泥沙输运损失 (SedimentTransportLoss)")
    print("-" * 40)
    
    sediment_loss = SedimentTransportLoss()
    h = predictions[:, :, :, 0]
    Q = predictions[:, :, :, 1]
    u = Q / (h * 500.0)
    
    sed_loss, sed_dict = sediment_loss(predictions, h, u)
    print(f"  泥沙损失: {sed_loss.item():.6f}")
    print(f"  平均输沙率: {sed_dict['mean_transport_rate']:.6f} m²/s")
    
    # 2. 测试增强版物理约束
    print("\n" + "-" * 40)
    print("2. 增强版物理约束 (EnhancedPhysicsConstraintLoss)")
    print("-" * 40)
    
    # 测试不同配置
    configs = [
        ("基础版", {'use_ntk_adaptive': False, 'use_curriculum': False, 'use_high_order': False}),
        ("高阶差分", {'use_ntk_adaptive': False, 'use_curriculum': False, 'use_high_order': True}),
        ("课程学习", {'use_ntk_adaptive': False, 'use_curriculum': True, 'use_high_order': True}),
        ("完整版", {'use_ntk_adaptive': True, 'use_curriculum': True, 'use_high_order': True, 'use_sediment': True}),
    ]
    
    for name, config in configs:
        loss_fn = EnhancedPhysicsConstraintLoss(total_epochs=100, **config)
        loss_fn.set_epoch(50)  # 设置中间轮数
        
        total_loss, loss_dict = loss_fn(predictions, targets, data_loss)
        
        print(f"\n  [{name}]")
        print(f"    Total: {loss_dict['total_loss']:.6f}")
        print(f"    Physics: {loss_dict['physics_loss']:.6f}")
        print(f"    Data weight: {loss_dict['data_weight']:.3f}")
        print(f"    Physics weight: {loss_dict['physics_weight']:.3f}")
    
    # 3. 测试边界条件强制
    print("\n" + "-" * 40)
    print("3. 边界条件强制 (BoundaryConditionEnforcer)")
    print("-" * 40)
    
    bc_enforcer = BoundaryConditionEnforcer()
    
    # 强制上游流量 = 2000
    pred_enforced = bc_enforcer.apply_upstream_flow(predictions, 2000.0)
    print(f"  上游流量强制: {pred_enforced[0, 0, 0, 1].item():.1f} (应为 2000)")
    
    # 强制下游水位 = 1050
    pred_enforced = bc_enforcer.apply_downstream_level(pred_enforced, 1050.0)
    print(f"  下游水位强制: {pred_enforced[0, 0, 4, 0].item():.1f} (应为 1050)")
    
    # 4. 测试多保真度
    print("\n" + "-" * 40)
    print("4. 多保真度 PINN (MultiFidelityPINN)")
    print("-" * 40)
    
    mf_pinn = MultiFidelityPINN()
    
    # 模拟高保真和低保真数据
    pred_hf = predictions
    pred_lf = predictions + torch.randn_like(predictions) * 0.5  # 低保真有噪声
    target_hf = targets
    target_lf = targets + torch.randn_like(targets) * 0.3
    
    # 创建简单模型用于测试
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
    
    dummy_model = DummyModel()
    
    mf_loss, mf_dict = mf_pinn(
        pred_hf, target_hf,
        pred_lf, target_lf,
        data_loss, data_loss * 1.2,
        dummy_model
    )
    
    print(f"  多保真度总损失: {mf_dict['total_loss']:.6f}")
    print(f"  高保真权重: {mf_dict['hf_weight']:.3f}")
    print(f"  低保真权重: {mf_dict['lf_weight']:.3f}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
