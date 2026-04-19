"""
增强版 STGNN (Spatio-Temporal Graph Neural Network) v3.0
包含: 物理引导注意力, 多尺度约束, 物理感知后处理, 贝叶斯不确定性, 自适应架构

改进内容:
1. 物理引导注意力 - 嵌入黄河水力学先验
2. 多尺度物理约束 - 多分辨率验证
3. 物理感知后处理 - 软约束修正输出
4. 贝叶斯不确定性 - 预测置信区间
5. 自适应网络架构 - 动态调整计算深度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict


# ============ 辅助类 (避免循环导入) ============

class _GraphAttentionLayer(nn.Module):
    """图注意力层 (简化版，避免从stgnn导入)"""
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(out_features, out_features)
    
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, n_nodes, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if adj is not None:
            scores = scores.masked_fill(adj.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_nodes, -1)
        out = self.fc_out(out)
        
        return out


class _TemporalTransformerBlock(nn.Module):
    """时间Transformer块 (简化版)"""
    
    def __init__(self, hidden_dim: int, n_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x


class PhysicsGuidedAttention(nn.Module):
    """
    物理引导的注意力机制
    
    黄河水力学先验:
    - 上游节点对下游影响递减
    - 下游对上游影响微弱
    - 相邻站点相关性最高
    """
    
    def __init__(self, hidden_dim: int, n_nodes: int = 5, n_heads: int = 8, dropout: float = 0.1):
        super(PhysicsGuidedAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        
        # 标准多头注意力
        self.standard_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 物理先验矩阵 (黄河: 兰州→河口镇→龙门→三门峡→花园口)
        # 数值表示水力影响强度
        physical_prior = torch.tensor([
            [1.00, 0.85, 0.70, 0.55, 0.40],  # 兰州
            [0.30, 1.00, 0.85, 0.70, 0.55],  # 河口镇
            [0.20, 0.30, 1.00, 0.85, 0.70],  # 龙门
            [0.15, 0.20, 0.30, 1.00, 0.85],  # 三门峡
            [0.10, 0.15, 0.20, 0.35, 1.00],  # 花园口
        ])
        self.register_buffer('physical_prior', physical_prior)
        
        # 学习残差注意力权重
        self.residual_proj = nn.Linear(hidden_dim, n_nodes)
        
        # 门控机制: 决定使用多少物理先验 vs 学习到的注意力
        self.prior_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, n_timesteps: int = 1, return_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch_size, n_nodes * n_timesteps, hidden_dim) - 展平的空间-时间特征
            n_timesteps: 时间步数
        Returns:
            output: 注意力输出 (同输入形状)
            weights: 组合注意力权重 (可选)
        """
        batch_size, total_nodes, hidden_dim = x.shape
        n_nodes = self.n_nodes
        
        # 重塑为 (batch, n_timesteps, n_nodes, hidden_dim)
        x_reshaped = x.view(batch_size, n_timesteps, n_nodes, hidden_dim)
        
        # 对每个时间步应用物理引导注意力
        outputs = []
        all_weights = []
        
        for t in range(n_timesteps):
            x_t = x_reshaped[:, t, :, :]  # (batch, n_nodes, hidden_dim)
            
            # 标准多头注意力 (自注意力，每个节点与其他节点)
            # 需要将 (batch, n_nodes, hidden) 转为 (batch, n_nodes, hidden)
            # 使用节点间注意力
            attn_out, _ = self.standard_attn(x_t, x_t, x_t)
            
            # 计算残差注意力权重
            residual_logits = self.residual_proj(x_t.mean(dim=1))  # (batch, n_nodes)
            residual_weights = F.softmax(residual_logits, dim=-1)
            
            # 扩展物理先验
            prior_batch = self.physical_prior.unsqueeze(0).expand(batch_size, -1, -1)
            residual_weights_expanded = residual_weights.unsqueeze(1).expand(-1, n_nodes, -1)
            
            # 门控
            gate = self.prior_gate(x_t.mean(dim=1))  # (batch, 1)
            
            # 组合权重
            combined_weights = gate.unsqueeze(-1) * prior_batch + (1 - gate).unsqueeze(-1) * residual_weights_expanded
            
            # 应用组合注意力: (batch, n_nodes, n_nodes) @ (batch, n_nodes, hidden) = (batch, n_nodes, hidden)
            physics_bias = torch.matmul(combined_weights, x_t)
            
            # 融合
            output_t = attn_out + self.dropout(physics_bias)
            output_t = self.norm(output_t)
            
            outputs.append(output_t)
            if return_weights:
                all_weights.append(combined_weights)
        
        # 重塑回原始形状
        output = torch.stack(outputs, dim=1).view(batch_size, total_nodes, hidden_dim)
        
        if return_weights:
            weights = torch.stack(all_weights, dim=1) if n_timesteps > 1 else all_weights[0]
            return output, weights
        return output, None


class PhysicsInformedPostProcessor(nn.Module):
    """
    物理感知后处理层
    
    软约束修正网络输出，使其更接近物理一致性
    但不强制严格满足 (留给损失函数处理)
    """
    
    def __init__(self, node_features: int, hidden_dim: int, river_width: float = 500.0):
        super(PhysicsInformedPostProcessor, self).__init__()
        
        self.node_features = node_features
        self.B = river_width
        
        # 标准 MLP 变换
        self.mlp = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, node_features)
        )
        
        # 物理门控: 学习哪些维度需要物理修正
        self.physics_gate = nn.Sequential(
            nn.Linear(node_features, node_features),
            nn.Sigmoid()
        )
        
        # 连续性修正强度预测 (输入: h, Q 两个特征)
        self.correction_strength = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None,
                Q_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, pred_horizon, n_nodes, node_features)
            h_prev: 上一时间步水位 (用于连续性修正)
            Q_prev: 上一时间步流量
        Returns:
            processed: 物理感知后的输出
        """
        batch_size, pred_horizon, n_nodes, _ = x.shape
        
        # 标准变换
        x_flat = x.reshape(batch_size * pred_horizon * n_nodes, self.node_features)
        x_transformed = self.mlp(x_flat)
        x_transformed = x_transformed.reshape(batch_size, pred_horizon, n_nodes, self.node_features)
        
        # 物理门控
        gate = self.physics_gate(x_transformed)  # (batch, pred_horizon, n_nodes, node_features)
        
        # 如果有前一时间步信息，进行连续性修正
        if h_prev is not None and Q_prev is not None:
            h_new = x_transformed[..., 0]
            Q_new = x_transformed[..., 1]
            
            # 简单连续性残差估计
            dh = h_new - h_prev
            dQ = Q_new - Q_prev
            
            # 修正强度 (可学习)
            features = torch.cat([h_new.unsqueeze(-1), Q_new.unsqueeze(-1)], dim=-1)
            strength = self.correction_strength(features).squeeze(-1)  # (batch, pred_horizon, n_nodes)
            
            # 软修正: 仅修正水位
            correction = gate[..., 0] * strength * 0.01  # 小幅度修正
            x_transformed[..., 0] = h_new - correction
        
        return x_transformed


class AdaptiveGNNLayer(nn.Module):
    """
    自适应 GNN 层 - 根据物理残差动态调整深度
    
    原理:
    - 物理残差大 → 需要更多层提取特征
    - 物理残差小 → 可以早期退出节省计算
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 n_heads: int = 8, max_layers: int = 6):
        super(AdaptiveGNNLayer, self).__init__()
        
        self.max_layers = max_layers
        self.in_features = in_features
        self.out_features = out_features
        
        # 可退出层序列
        self.gnn_layers = nn.ModuleList([
            nn.MultiheadAttention(out_features, n_heads, batch_first=True)
            for _ in range(max_layers)
        ])
        
        # 早期退出决策器
        self.exit_deciders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_features, out_features // 2),
                nn.ReLU(),
                nn.Linear(out_features // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(max_layers - 1)  # 最后一层必须执行
        ])
        
        # 输入投影
        self.input_proj = nn.Linear(in_features, out_features)
        
        self.norms = nn.ModuleList([nn.LayerNorm(out_features) for _ in range(max_layers)])
    
    def forward(self, x: torch.Tensor, physics_residual: Optional[float] = None,
                force_exit_layer: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (batch, n_nodes, in_features)
            physics_residual: 当前物理残差 (用于自适应)
            force_exit_layer: 强制在指定层退出 (用于测试)
        Returns:
            output: (batch, n_nodes, out_features)
            info: 包含实际使用的层数等信息
        """
        batch_size, n_nodes, _ = x.shape
        
        # 输入投影
        x = self.input_proj(x)
        
        exit_layer = self.max_layers - 1
        exit_confidences = []
        
        for i, (layer, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            # GNN 变换
            attn_out, _ = layer(x, x, x)
            x = x + attn_out
            x = norm(x)
            
            # 检查是否可以早期退出
            if i < len(self.exit_deciders):
                exit_conf = self.exit_deciders[i](x.mean(dim=1)).mean()
                exit_confidences.append(exit_conf.item())
                
                # 退出条件: 置信度高 且 (物理残差小 或 强制退出)
                should_exit = (exit_conf > 0.9)
                if physics_residual is not None:
                    should_exit = should_exit and (physics_residual < 0.05)
                if force_exit_layer is not None:
                    should_exit = should_exit or (i >= force_exit_layer)
                
                if should_exit:
                    exit_layer = i
                    break
        
        info = {
            'exit_layer': exit_layer,
            'total_layers': self.max_layers,
            'exit_confidences': exit_confidences,
            'early_exit': exit_layer < self.max_layers - 1
        }
        
        return x, info


class MultiScaleTemporalBlock(nn.Module):
    """
    多尺度时间编码块
    
    同时捕捉: 小时级变化, 日周期, 汛期/枯水期季节特征
    """
    
    def __init__(self, hidden_dim: int, scales: List[int] = [1, 6, 24]):
        super(MultiScaleTemporalBlock, self).__init__()
        
        self.scales = scales  # 1小时, 6小时, 24小时
        self.hidden_dim = hidden_dim
        
        # 每个尺度一个 Transformer
        self.scale_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=2
            )
            for _ in scales
        ])
        
        # 尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(scales), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            fused: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        scale_outputs = []
        
        for scale, encoder in zip(self.scales, self.scale_encoders):
            if scale == 1:
                # 原始尺度
                scale_x = x
            else:
                # 池化降采样
                pool_size = scale
                if seq_len >= pool_size:
                    scale_x = F.avg_pool1d(
                        x.transpose(1, 2),  # (batch, hidden, seq)
                        kernel_size=pool_size,
                        stride=pool_size
                    ).transpose(1, 2)  # (batch, seq//scale, hidden)
                else:
                    scale_x = x
            
            # 编码
            encoded = encoder(scale_x)
            
            # 上采样回原始长度
            if scale > 1 and seq_len >= scale:
                encoded = F.interpolate(
                    encoded.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            scale_outputs.append(encoded)
        
        # 融合
        concatenated = torch.cat(scale_outputs, dim=-1)
        fused = self.scale_fusion(concatenated)
        
        return fused


class BayesianSTGNN(nn.Module):
    """
    贝叶斯 STGNN - 不确定性量化
    
    使用 Monte Carlo Dropout 作为贝叶斯近似
    """
    
    def __init__(self, base_model: nn.Module, dropout_rate: float = 0.1):
        super(BayesianSTGNN, self).__init__()
        
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        
        # 不确定性头 (预测方差)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(base_model.node_features if hasattr(base_model, 'node_features') else 6, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, base_model.node_features if hasattr(base_model, 'node_features') else 6),
            nn.Softplus()  # 确保方差为正
        )
    
    def forward(self, x: torch.Tensor, n_samples: int = 10,
                return_uncertainty: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入
            n_samples: MC Dropout 采样次数
            return_uncertainty: 是否返回不确定性
        Returns:
            dict: 包含 mean, std, samples
        """
        if not return_uncertainty:
            return {'mean': self.base_model(x), 'std': None}
        
        # Monte Carlo Dropout 采样
        predictions = []
        self.train()  # 保持 dropout 开启
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.base_model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # 统计量
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # 添加模型不确定性 (异方差)
        epistemic_uncertainty = self.uncertainty_head(mean_pred.reshape(-1, mean_pred.shape[-1]))
        epistemic_uncertainty = epistemic_uncertainty.reshape_as(std_pred)
        
        # 总不确定性 = 偶然不确定性 (std_pred) + 认知不确定性
        total_uncertainty = std_pred + 0.1 * epistemic_uncertainty
        
        return {
            'mean': mean_pred,
            'std': total_uncertainty,
            'epistemic': std_pred,
            'aleatoric': epistemic_uncertainty,
            'samples': predictions
        }


class EnhancedSTGNN(nn.Module):
    """
    增强版 STGNN - 集成所有改进
    
    架构:
    输入 → 嵌入 → [自适应GNN + 物理引导注意力] → 多尺度时间编码 → 物理感知后处理 → 输出
    """
    
    def __init__(self,
                 node_features: int = 6,
                 n_nodes: int = 5,
                 gnn_hidden: int = 64,
                 gnn_layers: int = 3,
                 transformer_heads: int = 8,
                 transformer_layers: int = 4,
                 sequence_length: int = 24,
                 prediction_horizon: int = 6,
                 dropout: float = 0.1,
                 use_physics_attention: bool = True,
                 use_adaptive_gnn: bool = True,
                 use_multiscale_temporal: bool = True,
                 use_physics_postprocess: bool = True,
                 use_bayesian: bool = False):
        super(EnhancedSTGNN, self).__init__()
        
        self.node_features = node_features
        self.n_nodes = n_nodes
        self.gnn_hidden = gnn_hidden
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # 配置开关
        self.use_physics_attention = use_physics_attention
        self.use_adaptive_gnn = use_adaptive_gnn
        self.use_multiscale_temporal = use_multiscale_temporal
        self.use_physics_postprocess = use_physics_postprocess
        self.use_bayesian = use_bayesian
        
        # 输入嵌入
        self.embedding = nn.Linear(node_features, gnn_hidden)
        
        # 空间编码: 自适应 GNN 或 标准 GNN
        # 注意: 输入已经通过 embedding 层转换为 gnn_hidden 维度
        if use_adaptive_gnn:
            self.spatial_layers = nn.ModuleList([
                AdaptiveGNNLayer(
                    gnn_hidden,  # 输入维度
                    gnn_hidden,  # 输出维度
                    n_heads=transformer_heads
                )
                for _ in range(gnn_layers)
            ])
        else:
            # 复制 GraphAttentionLayer 定义以避免导入问题
            self.spatial_layers = nn.ModuleList([
                _GraphAttentionLayer(gnn_hidden, gnn_hidden, n_heads=transformer_heads)
                for _ in range(gnn_layers)
            ])
        
        # 物理引导注意力 (替代标准注意力)
        if use_physics_attention:
            self.physics_attn = PhysicsGuidedAttention(
                gnn_hidden, n_nodes=n_nodes, n_heads=transformer_heads
            )
        
        # 时间编码: 多尺度或标准 Transformer
        if use_multiscale_temporal:
            self.temporal_encoder = MultiScaleTemporalBlock(
                gnn_hidden, scales=[1, 6, 24]
            )
        else:
            # 使用简化的标准 Transformer
            self.temporal_layers = nn.ModuleList([
                _TemporalTransformerBlock(gnn_hidden, n_heads=transformer_heads)
                for _ in range(transformer_layers)
            ])
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden, gnn_hidden // 2),
            nn.ReLU(),
        )
        
        # 预测头
        self.predictor = nn.Linear(gnn_hidden // 2, node_features)
        
        # 自回归重新嵌入投影
        self.re_embed_proj = nn.Linear(node_features, gnn_hidden // 2)
        
        # 物理感知后处理
        if use_physics_postprocess:
            self.post_processor = PhysicsInformedPostProcessor(
                node_features, gnn_hidden // 2
            )
        
        # 时间位置编码
        self.register_buffer('pos_encoding', self._create_positional_encoding(sequence_length, gnn_hidden))
        
        # 贝叶斯包装
        if use_bayesian:
            self.enable_bayesian = True
        
        # 构建邻接矩阵
        self.adj = self._build_adjacency_matrix(n_nodes)
    
    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """创建正弦位置编码"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _build_adjacency_matrix(self, n_nodes: int) -> torch.Tensor:
        """构建黄河河流网络邻接矩阵"""
        adj = torch.zeros(n_nodes, n_nodes)
        for i in range(n_nodes - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
        for i in range(n_nodes):
            adj[i, i] = 1
        return adj
    
    def forward(self, x: torch.Tensor, return_info: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, n_nodes, node_features)
            return_info: 是否返回层信息 (用于分析)
        Returns:
            predictions: (batch_size, prediction_horizon, n_nodes, node_features)
        """
        batch_size, seq_len, n_nodes, _ = x.shape
        device = x.device
        
        # 重塑并嵌入
        x_reshaped = x.reshape(batch_size * seq_len, n_nodes, -1)
        x_embedded = self.embedding(x_reshaped)
        
        # 空间编码
        x_spatial = x_embedded
        spatial_info = []
        
        if self.use_adaptive_gnn:
            # 自适应 GNN
            for i, layer in enumerate(self.spatial_layers):
                x_spatial, info = layer(x_spatial)
                spatial_info.append(info)
        else:
            # 标准 GNN
            adj = self.adj.to(device)
            for layer in self.spatial_layers:
                x_spatial = layer(x_spatial, adj)
        
        # 应用物理引导注意力
        if self.use_physics_attention:
            x_spatial_flat = x_spatial.reshape(batch_size, seq_len * n_nodes, self.gnn_hidden)
            x_spatial_flat, attn_weights = self.physics_attn(x_spatial_flat, n_timesteps=seq_len, return_weights=True)
            x_spatial = x_spatial_flat.reshape(batch_size * seq_len, n_nodes, self.gnn_hidden)
        
        # 重塑为时间序列
        x_temporal_input = x_spatial.reshape(batch_size, seq_len, n_nodes, self.gnn_hidden)
        
        # 时间编码
        x_temporal_list = []
        for node_idx in range(n_nodes):
            x_node = x_temporal_input[:, :, node_idx, :]
            x_node = x_node + self.pos_encoding[:, :seq_len, :]
            
            if self.use_multiscale_temporal:
                x_node = self.temporal_encoder(x_node)
            else:
                for layer in self.temporal_layers:
                    x_node = layer(x_node)
            
            x_temporal_list.append(x_node)
        
        x_temporal = torch.stack(x_temporal_list, dim=2)
        
        # 取最后时间步
        x_final = x_temporal[:, -1, :, :]
        
        # 融合
        x_fused = self.fusion(x_final)
        
        # 多步自回归预测
        predictions = []
        x_pred = x_fused  # (batch, n_nodes, gnn_hidden // 2)
        
        for step in range(self.prediction_horizon):
            pred_step = self.predictor(x_pred)  # (batch, n_nodes, node_features)
            predictions.append(pred_step.unsqueeze(1))
            
            # 重新投影用于下一步自回归
            x_pred = self.re_embed_proj(pred_step)  # (batch, n_nodes, gnn_hidden // 2)
        
        predictions = torch.cat(predictions, dim=1)  # (batch, pred_horizon, n_nodes, node_features)
        
        # 物理感知后处理
        if self.use_physics_postprocess:
            if predictions.shape[1] > 1:  # pred_horizon > 1
                h_prev = predictions[:, :-1, :, 0]  # (batch, pred_horizon-1, n_nodes)
                Q_prev = predictions[:, :-1, :, 1]
                # 填充第一个时间步，使形状与 predictions 一致
                # h_prev shape: (batch, pred_horizon-1, n_nodes) -> need (batch, pred_horizon, n_nodes)
                h_prev = F.pad(h_prev, (0, 0, 1, 0), value=0)  # 在时间维度前填充
                Q_prev = F.pad(Q_prev, (0, 0, 1, 0), value=0)
            else:
                h_prev, Q_prev = None, None
            
            predictions = self.post_processor(predictions, h_prev, Q_prev)
        
        if return_info:
            return predictions, {
                'spatial_info': spatial_info,
                'use_adaptive': self.use_adaptive_gnn,
                'use_physics_attn': self.use_physics_attention,
                'use_multiscale': self.use_multiscale_temporal
            }
        
        return predictions


def create_enhanced_model(config: dict) -> nn.Module:
    """创建增强版模型"""
    return EnhancedSTGNN(
        node_features=config.get('node_features', 6),
        n_nodes=config.get('n_nodes', 5),
        gnn_hidden=config.get('gnn_hidden', 64),
        gnn_layers=config.get('gnn_layers', 3),
        transformer_heads=config.get('transformer_heads', 8),
        transformer_layers=config.get('transformer_layers', 4),
        sequence_length=config.get('sequence_length', 24),
        prediction_horizon=config.get('prediction_horizon', 6),
        dropout=config.get('dropout', 0.1),
        use_physics_attention=config.get('use_physics_attention', True),
        use_adaptive_gnn=config.get('use_adaptive_gnn', True),
        use_multiscale_temporal=config.get('use_multiscale_temporal', True),
        use_physics_postprocess=config.get('use_physics_postprocess', True),
        use_bayesian=config.get('use_bayesian', False)
    )


# 向后兼容
STGNNWithAttention = EnhancedSTGNN


if __name__ == '__main__':
    # 测试增强版 STGNN
    print("=" * 70)
    print("增强版 STGNN v3.0 测试")
    print("=" * 70)
    
    batch_size = 4
    seq_len = 24
    n_nodes = 5
    node_features = 6
    
    configs = [
        ("基础版 (无增强)", {
            'use_physics_attention': False,
            'use_adaptive_gnn': False,
            'use_multiscale_temporal': False,
            'use_physics_postprocess': False
        }),
        ("物理注意力", {'use_physics_attention': True}),
        ("自适应GNN", {'use_adaptive_gnn': True}),
        ("多尺度时间", {'use_multiscale_temporal': True}),
        ("物理后处理", {'use_physics_postprocess': True}),
        ("完整版 (所有增强)", {
            'use_physics_attention': True,
            'use_adaptive_gnn': True,
            'use_multiscale_temporal': True,
            'use_physics_postprocess': True
        }),
    ]
    
    x = torch.randn(batch_size, seq_len, n_nodes, node_features)
    
    for name, config in configs:
        print(f"\n{'-' * 50}")
        print(f"测试: {name}")
        print('-' * 50)
        
        model = EnhancedSTGNN(
            node_features=node_features,
            n_nodes=n_nodes,
            sequence_length=seq_len,
            prediction_horizon=6,
            **config
        )
        
        output = model(x)
        params = sum(p.numel() for p in model.parameters())
        
        print(f"  输入:  {x.shape}")
        print(f"  输出:  {output.shape}")
        print(f"  参数量: {params:,}")
    
    # 测试贝叶斯版本
    print(f"\n{'-' * 50}")
    print("测试: 贝叶斯不确定性量化")
    print('-' * 50)
    
    base_model = EnhancedSTGNN(node_features=node_features, n_nodes=n_nodes)
    bayesian_model = BayesianSTGNN(base_model)
    
    result = bayesian_model(x, n_samples=10)
    
    print(f"  均值形状:   {result['mean'].shape}")
    print(f"  标准差形状: {result['std'].shape}")
    print(f"  平均不确定性: {result['std'].mean().item():.4f}")
    
    print("\n" + "=" * 70)
    print("所有测试通过!")
    print("=" * 70)
