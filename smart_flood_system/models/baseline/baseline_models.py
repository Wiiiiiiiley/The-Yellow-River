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
    用于与 PINN 模型进行对比
    """
    
    def __init__(self,
                 input_size: int = 6,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 6,
                 prediction_horizon: int = 6,
                 dropout: float = 0.1):
        super(BaselineLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.prediction_horizon = prediction_horizon
        
        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=input_size,
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
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # 多步预测解码器
        self.decoder = nn.GRUCell(output_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, time_steps, num_nodes, input_size) 或 (batch_size, seq_len, input_size)
        
        Returns:
            predictions: (batch_size, prediction_horizon, num_nodes, output_size) 或 (batch_size, prediction_horizon, output_size)
        """
        # 处理 4D 输入 (batch, time, nodes, features)
        if x.dim() == 4:
            batch_size, seq_len, num_nodes, input_size = x.shape
            # 提取第一个节点（兰州站）的数据用于预测
            x = x[:, :, 0, :]  # (batch_size, seq_len, input_size)
            output_nodes = num_nodes
        else:
            batch_size, seq_len = x.shape[:2]
            output_nodes = 1
        
        # LSTM 编码
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # 注意力
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch_size, seq_len, hidden_size)
        
        # 融合
        context = lstm_out + attn_out  # (batch_size, seq_len, hidden_size)
        
        # 取最后时间步作为初始状态
        final_state = context[:, -1, :]  # (batch_size, hidden_size)
        
        # 自回归预测
        predictions = []
        hidden = final_state
        
        for t in range(self.prediction_horizon):
            # 预测当前时间步
            output = self.output_proj(hidden)  # (batch_size, output_size)
            predictions.append(output.unsqueeze(1))
            
            # 更新隐状态用于下一时间步
            hidden = self.decoder(output, hidden)
        
        # 合并预测
        predictions = torch.cat(predictions, dim=1)  # (batch_size, prediction_horizon, output_size)
        
        # 如果输入是 4D，添加节点维度 (将单个节点扩展为所有节点)
        if output_nodes > 1:
            predictions = predictions.unsqueeze(2)  # (batch_size, prediction_horizon, 1, output_size)
            # 复制并扩展到所有节点
            predictions = predictions.expand(-1, -1, output_nodes, -1)
        
        return predictions


class BaselineTransformer(nn.Module):
    """
    Baseline 模型：纯 Transformer
    用于时间序列预测
    """
    
    def __init__(self,
                 input_size: int = 6,
                 d_model: int = 64,
                 nhead: int = 8,
                 num_layers: int = 3,
                 output_size: int = 6,
                 prediction_horizon: int = 6,
                 dropout: float = 0.1):
        super(BaselineTransformer, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        self.prediction_horizon = prediction_horizon
        
        # 输入投影
        self.input_proj = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.register_buffer('pos_encoding', self._create_positional_encoding(100, d_model))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size)
        )
        
        # 解码器用于多步预测
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 解码器输出
        self.decoder_output = nn.Linear(d_model, output_size)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, time_steps, num_nodes, input_size) 或 (batch_size, seq_len, input_size)
        
        Returns:
            predictions: (batch_size, prediction_horizon, num_nodes, output_size) 或 (batch_size, prediction_horizon, output_size)
        """
        # 处理 4D 输入 (batch, time, nodes, features)
        if x.dim() == 4:
            batch_size, seq_len, num_nodes, input_size = x.shape
            # 提取第一个节点（兰州站）的数据用于预测
            x = x[:, :, 0, :]  # (batch_size, seq_len, input_size)
            output_nodes = num_nodes
        else:
            batch_size, seq_len = x.shape[:2]
            output_nodes = 1
        
        # 输入投影
        x_proj = self.input_proj(x)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x_proj = x_proj + self.pos_encoding[:, :seq_len, :]
        
        # Transformer 编码
        encoded = self.transformer_encoder(x_proj)  # (batch_size, seq_len, d_model)
        
        # 使用最后时间步作为初始解码器输入
        decoder_input = encoded[:, -1:, :].repeat(1, self.prediction_horizon, 1)
        
        # 解码预测
        decoded = self.transformer_decoder(decoder_input, encoded)
        
        # 输出投影
        predictions = self.decoder_output(decoded)  # (batch_size, prediction_horizon, output_size)
        
        # 如果输入是 4D，添加节点维度
        if output_nodes > 1:
            predictions = predictions.unsqueeze(2)  # (batch_size, prediction_horizon, 1, output_size)
            # 复制并扩展到所有节点
            predictions = predictions.expand(-1, -1, output_nodes, -1)
        
        return predictions


class EnsembleBaseline(nn.Module):
    """
    集成 Baseline 模型
    结合 LSTM 和 Transformer
    """
    
    def __init__(self,
                 input_size: int = 6,
                 hidden_size: int = 128,
                 output_size: int = 6,
                 prediction_horizon: int = 6,
                 dropout: float = 0.1):
        super(EnsembleBaseline, self).__init__()
        
        # LSTM 分支
        self.lstm_model = BaselineLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            output_size=output_size,
            prediction_horizon=prediction_horizon,
            dropout=dropout
        )
        
        # Transformer 分支
        self.transformer_model = BaselineTransformer(
            input_size=input_size,
            d_model=hidden_size,
            nhead=8,
            num_layers=2,
            output_size=output_size,
            prediction_horizon=prediction_horizon,
            dropout=dropout
        )
        
        # 权重
        self.lstm_weight = nn.Parameter(torch.tensor(0.5))
        self.transformer_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        融合两个模型的预测
        """
        lstm_pred = self.lstm_model(x)
        transformer_pred = self.transformer_model(x)
        
        # 标准化权重
        weights = torch.softmax(torch.stack([self.lstm_weight, self.transformer_weight]), dim=0)
        
        # 加权融合
        predictions = weights[0] * lstm_pred + weights[1] * transformer_pred
        
        return predictions


def create_baseline_model(model_type: str = 'lstm',
                         input_size: int = 6,
                         hidden_size: int = 128,
                         output_size: int = 6,
                         prediction_horizon: int = 6) -> nn.Module:
    """创建 baseline 模型"""
    
    if model_type == 'lstm':
        return BaselineLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            output_size=output_size,
            prediction_horizon=prediction_horizon
        )
    elif model_type == 'transformer':
        return BaselineTransformer(
            input_size=input_size,
            d_model=hidden_size,
            nhead=8,
            num_layers=2,
            output_size=output_size,
            prediction_horizon=prediction_horizon
        )
    elif model_type == 'ensemble':
        return EnsembleBaseline(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            prediction_horizon=prediction_horizon
        )
    else:
        raise ValueError(f"Unknown baseline model type: {model_type}")


if __name__ == '__main__':
    # 测试
    batch_size = 4
    seq_len = 24
    input_size = 6
    output_size = 6
    horizon = 6
    
    # 测试 LSTM baseline
    lstm_model = BaselineLSTM(
        input_size=input_size,
        hidden_size=128,
        output_size=output_size,
        prediction_horizon=horizon
    )
    
    x = torch.randn(batch_size, seq_len, input_size)
    output = lstm_model(x)
    print(f"LSTM 输入形状: {x.shape}")
    print(f"LSTM 输出形状: {output.shape}")
    print(f"LSTM 参数数量: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    # 测试 Transformer baseline
    transformer_model = BaselineTransformer(
        input_size=input_size,
        d_model=128,
        output_size=output_size,
        prediction_horizon=horizon
    )
    
    output = transformer_model(x)
    print(f"\nTransformer 输出形状: {output.shape}")
    print(f"Transformer 参数数量: {sum(p.numel() for p in transformer_model.parameters()):,}")
    
    # 测试集成 baseline
    ensemble_model = EnsembleBaseline(
        input_size=input_size,
        output_size=output_size,
        prediction_horizon=horizon
    )
    
    output = ensemble_model(x)
    print(f"\nEnsemble 输出形状: {output.shape}")
    print(f"Ensemble 参数数量: {sum(p.numel() for p in ensemble_model.parameters()):,}")
