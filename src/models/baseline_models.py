"""
Baseline 模型 - 纯数据驱动预测
不包含物理约束，用于对比验证
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class BaselineLSTM(nn.Module):
    """
    Baseline 模型：LSTM + 全连接层
    纯数据驱动，不含物理约束
    用于与 PINN 模型进行对比 - 公平的多节点版本
    """
    
    def __init__(self,
                 input_size: int = 6,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 6,
                 prediction_horizon: int = 6,
                 n_nodes: int = 5,
                 dropout: float = 0.1):
        super(BaselineLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.prediction_horizon = prediction_horizon
        self.n_nodes = n_nodes
        
        # LSTM 编码器 - 接受展平的多节点输入
        self.lstm = nn.LSTM(
            input_size=input_size * n_nodes,  # 5节点 * 6特征 = 30
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 多步预测解码器 - 输入是output_size，输出是hidden_size
        self.decoder = nn.GRUCell(output_size, hidden_size)
        
        # 节点投影层：将隐状态投影到 (nodes, features)
        self.node_proj = nn.Linear(hidden_size, n_nodes * output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, time_steps, num_nodes, input_size)
        
        Returns:
            predictions: (batch_size, prediction_horizon, num_nodes, output_size)
        """
        # 处理 4D 输入 (batch, time, nodes, features)
        if x.dim() == 4:
            batch_size, seq_len, num_nodes, input_size = x.shape
            # 将节点维度展平到特征维度: (batch, time, nodes*features)
            x = x.reshape(batch_size, seq_len, num_nodes * input_size)
            output_nodes = num_nodes
        else:
            batch_size, seq_len = x.shape[:2]
            x = x
            output_nodes = 1
        
        # LSTM 编码
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # 注意力
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch_size, seq_len, hidden_size)
        
        # 融合
        context = lstm_out + attn_out  # (batch_size, seq_len, hidden_size)
        
        # 取最后时间步作为初始状态
        final_state = context[:, -1, :]  # (batch_size, hidden_size)
        
        # 自回归预测 - 输出所有节点
        predictions = []
        hidden = final_state
        
        for t in range(self.prediction_horizon):
            # 投影到 (nodes, features)
            node_output = self.node_proj(hidden)  # (batch_size, n_nodes * output_size)
            node_output = node_output.reshape(batch_size, self.n_nodes, self.output_size)
            predictions.append(node_output.unsqueeze(1))  # (batch, 1, nodes, features)
            
            # 使用平均特征更新隐状态
            pooled = node_output.mean(dim=1)  # (batch, output_size)
            hidden = self.decoder(pooled, hidden)
        
        # 合并预测: (batch, prediction_horizon, n_nodes, output_size)
        predictions = torch.cat(predictions, dim=1)
        
        return predictions


def create_baseline_model(model_type: str = 'lstm',
                         input_size: int = 6,
                         hidden_size: int = 128,
                         output_size: int = 6,
                         prediction_horizon: int = 6,
                         n_nodes: int = 5) -> nn.Module:
    """创建 Baseline LSTM 模型
    
    注意：已移除 transformer 和 ensemble 类型（架构过时未修复）
    """
    
    if model_type != 'lstm':
        raise ValueError(f"只支持 'lstm' 类型，'{model_type}' 已被移除（架构过时）")
    
    return BaselineLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=2,
        output_size=output_size,
        prediction_horizon=prediction_horizon,
        n_nodes=n_nodes
    )
