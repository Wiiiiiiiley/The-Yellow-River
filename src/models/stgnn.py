"""
GNN-Transformer 时空预测模型
结合 GAT（图注意力）+ Transformer 用于黄河水文预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class GraphAttentionLayer(nn.Module):
    """图注意力层 - 建模站点间空间依赖"""
    
    def __init__(self, in_features: int, out_features: int, 
                 n_heads: int = 8, dropout: float = 0.1):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        
        assert out_features % n_heads == 0, "out_features 必须被 n_heads 整除"
        
        # 多头注意力
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(out_features, out_features)
        
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_nodes, in_features)
            adj: (n_nodes, n_nodes) - 邻接矩阵
        Returns:
            output: (batch_size, n_nodes, out_features)
        """
        batch_size, n_nodes, _ = x.shape
        
        # 线性变换
        Q = self.W_q(x)  # (batch_size, n_nodes, out_features)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 重塑用于多头
        Q = Q.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)  
        K = K.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch_size, n_heads, n_nodes, n_nodes)
        
        # 应用邻接矩阵掩码
        if adj is not None:
            scores = scores.masked_fill(adj.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        # 注意力权重
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力到值
        out = torch.matmul(attention, V)  # (batch_size, n_heads, n_nodes, head_dim)
        
        # 合并多头
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, n_nodes, -1)
        
        # 最终线性层
        out = self.fc_out(out)
        
        return out


class SpatialTransformerBlock(nn.Module):
    """空间 Transformer 块"""
    
    def __init__(self, hidden_dim: int, n_heads: int = 8, 
                 ff_dim: int = 2048, dropout: float = 0.1):
        super(SpatialTransformerBlock, self).__init__()
        
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
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            attn_mask: 注意力掩码
        """
        # 自注意力
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # 前馈网络
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x


class TemporalTransformerBlock(nn.Module):
    """时间 Transformer 块 - 处理时间依赖"""
    
    def __init__(self, hidden_dim: int, n_heads: int = 8,
                 ff_dim: int = 2048, dropout: float = 0.1):
        super(TemporalTransformerBlock, self).__init__()
        
        # ProbSparse 注意力（仅关键时间点）
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
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
        """
        # 自注意力
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # 前馈网络
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x


class GNNTransformer(nn.Module):
    """
    GNN-Transformer 时空预测模型
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
                 dropout: float = 0.1):
        super(GNNTransformer, self).__init__()
        
        self.node_features = node_features
        self.n_nodes = n_nodes
        self.gnn_hidden = gnn_hidden
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # 输入嵌入
        self.embedding = nn.Linear(node_features, gnn_hidden)
        
        # 空间编码：图注意力层
        self.spatial_layers = nn.ModuleList([
            GraphAttentionLayer(gnn_hidden, gnn_hidden, n_heads=transformer_heads, dropout=dropout)
            for _ in range(gnn_layers)
        ])
        
        # 时间编码：Transformer 层
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerBlock(gnn_hidden, n_heads=transformer_heads, 
                                    ff_dim=gnn_hidden * 4, dropout=dropout)
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
        
        # 时间位置编码
        self.register_buffer('pos_encoding', self._create_positional_encoding(sequence_length, gnn_hidden))
    
    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """创建正弦位置编码"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _build_adjacency_matrix(self, n_nodes: int, device: torch.device) -> torch.Tensor:
        """构建黄河河流网络邻接矩阵"""
        # 五个站点的拓扑：兰州 → 河口镇 → 龙门 → 三门峡 → 花园口
        # 邻接矩阵表示相邻关系
        adj = torch.zeros(n_nodes, n_nodes, device=device)
        
        # 相邻节点之间有边
        for i in range(n_nodes - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1  # 无向
        
        # 自环
        for i in range(n_nodes):
            adj[i, i] = 1
        
        return adj
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, n_nodes, node_features)
            
        Returns:
            predictions: (batch_size, prediction_horizon, n_nodes, node_features)
        """
        batch_size, seq_len, n_nodes, _ = x.shape
        device = x.device
        
        # 重塑为 (batch_size * seq_len, n_nodes, node_features)
        x_reshaped = x.reshape(batch_size * seq_len, n_nodes, -1)
        
        # 嵌入
        x_embedded = self.embedding(x_reshaped)  # (batch_size * seq_len, n_nodes, gnn_hidden)
        
        # 空间编码 - 图注意力
        x_spatial = x_embedded
        adj = self._build_adjacency_matrix(n_nodes, device)
        
        for spatial_layer in self.spatial_layers:
            x_spatial = spatial_layer(x_spatial, adj)
        
        # 重塑回时间序列形式
        x_spatial = x_spatial.reshape(batch_size, seq_len, n_nodes, self.gnn_hidden)
        
        # 对每个节点进行时间处理
        x_temporal = torch.zeros_like(x_spatial)
        
        for node_idx in range(n_nodes):
            x_node = x_spatial[:, :, node_idx, :]  # (batch_size, seq_len, gnn_hidden)
            
            # 添加位置编码
            x_node = x_node + self.pos_encoding[:, :seq_len, :]
            
            # 时间 Transformer
            for temporal_layer in self.temporal_layers:
                x_node = temporal_layer(x_node)
            
            x_temporal[:, :, node_idx, :] = x_node
        
        # 取序列的最后一步表示
        x_final = x_temporal[:, -1, :, :]  # (batch_size, n_nodes, gnn_hidden)
        
        # 融合
        x_fused = self.fusion(x_final)  # (batch_size, n_nodes, gnn_hidden // 2)
        
        # 多步预测 - 自回归
        predictions = []
        x_pred = x_fused
        
        for step in range(self.prediction_horizon):
            # 预测下一时间步
            x_pred = self.predictor(x_pred)  # (batch_size, n_nodes, node_features)
            predictions.append(x_pred.unsqueeze(1))
            
            # 重新嵌入用于下一步
            x_pred = self.embedding(x_pred)
        
        predictions = torch.cat(predictions, dim=1)  # (batch_size, prediction_horizon, n_nodes, node_features)
        
        return predictions


class STGNNWithAttention(nn.Module):
    """
    改进的 STGNN - 带空间-时间交叉注意
    """
    
    def __init__(self,
                 node_features: int = 6,
                 n_nodes: int = 5,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 n_heads: int = 8,
                 sequence_length: int = 24,
                 prediction_horizon: int = 6,
                 dropout: float = 0.1):
        super(STGNNWithAttention, self).__init__()
        
        self.gnn_base = GNNTransformer(
            node_features=node_features,
            n_nodes=n_nodes,
            gnn_hidden=hidden_dim,
            gnn_layers=n_layers,
            transformer_heads=n_heads,
            transformer_layers=n_layers,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            dropout=dropout
        )
        
        # 结果后处理
        self.post_processing = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, n_nodes, node_features)
        """
        predictions = self.gnn_base(x)
        
        # 后处理
        batch_size, pred_horizon, n_nodes, n_features = predictions.shape
        predictions_flat = predictions.reshape(batch_size * pred_horizon * n_nodes, n_features)
        predictions_processed = self.post_processing(predictions_flat)
        predictions = predictions_processed.reshape(batch_size, pred_horizon, n_nodes, n_features)
        
        return predictions


def create_model(config: dict) -> nn.Module:
    """创建模型"""
    return STGNNWithAttention(
        node_features=config.get('node_features', 6),
        n_nodes=config.get('n_nodes', 5),
        hidden_dim=config.get('gnn_hidden', 64),
        n_layers=config.get('gnn_layers', 3),
        n_heads=config.get('transformer_heads', 8),
        sequence_length=config.get('sequence_length', 24),
        prediction_horizon=config.get('prediction_horizon', 6),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == '__main__':
    # 测试
    batch_size = 4
    seq_len = 24
    n_nodes = 5
    node_features = 6
    
    model = STGNNWithAttention(
        node_features=node_features,
        n_nodes=n_nodes,
        hidden_dim=64,
        n_layers=3,
        n_heads=8,
        sequence_length=seq_len,
        prediction_horizon=6
    )
    
    # 输入数据
    x = torch.randn(batch_size, seq_len, n_nodes, node_features)
    
    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
