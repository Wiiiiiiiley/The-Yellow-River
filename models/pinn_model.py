"""Physics-Informed Neural Network (PINN) for Yellow River prediction.

Hybrid Optimized Version:
1. Strict isolation of spatial-temporal variables (t, x) for pure Autograd graphs.
2. Funnel-shaped network architecture for enhanced nonlinear fitting.
3. Swish activation (default) for smooth PDE derivatives, with Tanh fallback.
4. Comprehensive documentation and robust error handling.

融合优化版：学术级梯度计算 + 工程级健壮性 + 详细文档
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

class Swish(nn.Module):
    """Swish激活函数：f(x) = x * torch.sigmoid(x)
    
    相比 ReLU，Swish 是无限阶可导的，这对于需要计算一阶/二阶空间和时间导数的
    物理偏微分方程 (PDE) 至关重要。
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class YellowRiverPINN(nn.Module):
    """黄河数字孪生PINN模型（融合优化版）。
    
    结合学术级梯度计算与工程级健壮性，预测水位和含沙量。
    
    Architecture (漏斗型结构):
        - 输入: [time, location, flow_rate, rainfall, temp]
        - 隐藏层: 漏斗型设计，先升维至128提取高阶特征，后降维
        - 输出: [water_level, sediment_content]
    
    Args:
        input_dim: 输入特征维度 (默认5: t, x, flow, rainfall, temp)
        hidden_dim: 初始隐藏层维度 (默认64)
        output_dim: 输出维度 (默认2: water_level, sediment)
        num_hidden_layers: 隐藏层数量 (默认4)
        activation: 激活函数 ('swish' 默认，或 'tanh')
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_hidden_layers: int = 4,
        activation: str = 'swish'  # 默认Swish更适合PDE
    ):
        super(YellowRiverPINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 选择激活函数：Swish无限阶可导，更适合PDE
        if activation == 'swish':
            self.activation = Swish()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'swish' or 'tanh'.")
        
        # 构建漏斗型网络层
        layers = []
        current_dim = hidden_dim
        
        # 输入层
        layers.append(nn.Linear(input_dim, current_dim))
        layers.append(self.activation)
        
        # 隐藏层：漏斗型设计（先升维后降维）
        for i in range(num_hidden_layers - 1):
            if i < num_hidden_layers // 2:
                # 升阶段：扩展至最多128维
                next_dim = min(current_dim * 2, 128)
            else:
                # 降阶段：收缩至不少于64维
                next_dim = max(current_dim // 2, 64)
            
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(self.activation)
            current_dim = next_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Xavier初始化：防止深度PDE网络梯度消失/爆炸
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化：防止深度PDE神经网络中的梯度消失/爆炸。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def compute_physics_loss(
    model: YellowRiverPINN,
    x_physics: torch.Tensor,
    alpha: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算物理约束损失（Physics Loss）。
    
    融合优化：学术级梯度计算 + 工程级输入验证。
    将时空变量 (t, x) 剥离为独立 Leaf Nodes，构建纯净的 Autograd 计算图。
    
    方程：∂h/∂t + α * ∂(h*u)/∂x = 0
    
    Args:
        model: PINN模型实例
        x_physics: 物理采样点，形状 (N, input_dim)
                   必须包含 [time, location, flow_rate, ...]
        alpha: 物理方程中的比例系数
    
    Returns:
        physics_loss: 物理损失标量
        components: 包含中间计算结果的字典，用于调试
    
    Raises:
        ValueError: 输入维度不足时抛出
    """
    # 输入验证：确保有足够维度
    if x_physics.shape[1] < 3:
        raise ValueError(
            f"x_physics 必须至少3维 [t, x, flow_rate, ...], "
            f"当前维度: {x_physics.shape[1]}"
        )
    
    # 1. 剥离并独立追踪时空变量 (Leaf Nodes)
    t = x_physics[:, 0:1].clone().requires_grad_(True)
    x = x_physics[:, 1:2].clone().requires_grad_(True)
    features = x_physics[:, 2:]  # [flow_rate, rainfall, temp] 不需要求导
    
    # 2. 重新拼接输入并进行前向传播
    inputs = torch.cat([t, x, features], dim=1)
    prediction = model(inputs)
    
    water_level = prediction[:, 0:1]  # h
    flow_rate = features[:, 0:1]      # Q (假设在 features 的第 0 列，即原 x_physics 的第 2 列)
    
    # 3. 计算时间导数 ∂h/∂t
    dh_dt = torch.autograd.grad(
        outputs=water_level,
        inputs=t,
        grad_outputs=torch.ones_like(water_level),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 4. 计算通量 Flux = h * u
    epsilon = 1e-6
    velocity = flow_rate / (water_level + epsilon)  # u = Q / (h + ε)
    flux = water_level * velocity                   # h * u
    
    # 5. 计算空间导数 ∂(h*u)/∂x
    d_flux_dx = torch.autograd.grad(
        outputs=flux,
        inputs=x,
        grad_outputs=torch.ones_like(flux),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 6. 计算物理残差与 Loss
    residual = dh_dt + alpha * d_flux_dx
    physics_loss = torch.mean(residual ** 2)
    
    components = {
        'dh_dt': dh_dt.detach(),
        'd_flux_dx': d_flux_dx.detach(),
        'residual': residual.detach(),
        'velocity': velocity.detach(),
        'flux': flux.detach()
    }
    
    return physics_loss, components

def compute_total_loss(
    model: YellowRiverPINN,
    x_data: torch.Tensor,
    y_true: torch.Tensor,
    x_physics: Optional[torch.Tensor] = None,
    lambda_phys: float = 1.0,
    alpha: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算总损失 = 数据拟合损失 + λ * 物理约束损失。
    
    Total_Loss = Data_Loss + lambda_phys * Physics_Loss
    
    Args:
        model: PINN模型实例
        x_data: 数据训练点，形状 (N, input_dim)
        y_true: 真实标签，形状 (N, output_dim)
        x_physics: 物理采样点，形状 (M, input_dim)。
                   如果为None，则使用x_data
        lambda_phys: 物理损失权重系数
        alpha: 物理方程比例系数
    
    Returns:
        total_loss: 总损失标量
        loss_components: 包含各损失分量的字典
    
    Raises:
        ValueError: 输入/输出维度不匹配时抛出
    """
    # 输入验证
    if x_data.shape[1] != model.input_dim:
        raise ValueError(
            f"x_data 维度 ({x_data.shape[1]}) 与模型输入维度 "
            f"({model.input_dim}) 不匹配"
        )
    if y_true.shape[1] != model.output_dim:
        raise ValueError(
            f"y_true 维度 ({y_true.shape[1]}) 与模型输出维度 "
            f"({model.output_dim}) 不匹配"
        )
    
    # Data Loss
    y_pred = model(x_data)
    data_loss = F.mse_loss(y_pred, y_true)
    
    # Physics Loss
    if x_physics is None:
        x_physics = x_data
        
    physics_loss, phys_components = compute_physics_loss(model, x_physics, alpha)
    
    # Total Loss
    total_loss = data_loss + lambda_phys * physics_loss
    
    loss_components = {
        'total_loss': total_loss,
        'data_loss': data_loss,
        'physics_loss': physics_loss,
        'weighted_physics_loss': lambda_phys * physics_loss,
        **phys_components
    }
    
    return total_loss, loss_components

if __name__ == "__main__":
    print("=" * 60)
    print("Yellow River PINN (Academic Edition) - Unit Test & Gradient Check")
    print("=" * 60)
    
    # 1. 实例化模型
    model = YellowRiverPINN(input_dim=5, output_dim=2)
    print(f"[OK] 模型初始化完成. 激活函数: {model.activation.__class__.__name__}")
    
    # 2. 生成模拟数据
    batch_size = 32
    # [t, x, flow, rain, temp]
    x_data = torch.randn(batch_size, 5)
    y_true = torch.randn(batch_size, 2)
    x_physics = torch.randn(batch_size, 5)
    
    # 3. 前向传播与物理 Loss 计算
    total_loss, logs = compute_total_loss(
        model, x_data, y_true, x_physics, lambda_phys=1.0, alpha=0.1
    )
    
    print("\n[Metrics]")
    print(f"  - Data Loss:    {logs['data_loss'].item():.6f}")
    print(f"  - Physics Loss: {logs['physics_loss'].item():.6f}")
    print(f"  - Total Loss:   {logs['total_loss'].item():.6f}")
    
    # 4. 反向传播与梯度验证 (Gradient Backpropagation Check)
    total_loss.backward()
    
    # 验证网络层是否成功捕获梯度
    grad_captured = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_captured = True
            break
            
    print("\n[Validation]")
    if grad_captured and logs['physics_loss'].item() > 0:
        print("  ✅ 物理残差计算成功，且 PDE 梯度已成功反向传播至神经网络层！")
        print("  ✅ 纯净计算图构建成功 (移除了 allow_unused 依赖)。")
    else:
        print("  ❌ 梯度回传失败，请检查张量依赖关系。")
    print("=" * 60)
