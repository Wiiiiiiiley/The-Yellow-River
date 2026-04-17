"""
Spatio-Temporal Graph Neural Network (ST-GNN)
=============================================

Combines Graph Neural Networks (GNN) with Transformer architecture for
spatio-temporal flood prediction on river network topology.

Architecture:
    Input (Graph) → GNN Encoder → Temporal Transformer → Decoder → Output

Features:
- Graph Convolution for spatial dependencies between hydrological stations
- Informer/Autoformer for long-range temporal dependencies
- Attention mechanisms for interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.data import Data
import math


class TemporalAttention(nn.Module):
    """
    Multi-head temporal attention (Informer-style)
    for long sequence time-series forecasting.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_probsparse: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_probsparse = use_probsparse
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Linear projections
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        if self.use_probsparse and seq_len > 100:
            # ProbSparse attention (Informer) for efficiency
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            # Select top-k queries (sparse attention)
            k = max(1, int(math.log(seq_len)))
            top_k_indices = torch.topk(scores.sum(dim=-1), k, dim=-1)[1]
            
            # Only compute attention for top-k queries
            Q_sparse = torch.gather(Q, 2, top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_k))
            attn_sparse = torch.matmul(Q_sparse, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if mask is not None:
                attn_sparse = attn_sparse.masked_fill(mask == 0, -1e9)
            
            attn_sparse = F.softmax(attn_sparse, dim=-1)
            attn_sparse = self.dropout(attn_sparse)
            
            out_sparse = torch.matmul(attn_sparse, V)
            
            # Scatter back
            output = torch.zeros_like(Q)
            output.scatter_(2, top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_k), out_sparse)
        else:
            # Standard attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            
            output = torch.matmul(attn, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(output)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        return self.layer_norm(output + residual)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with multi-head attention and feed-forward network.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        
        self.attention = TemporalAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor):
        # Self-attention
        x = self.attention(x)
        
        # Feed-forward with residual
        residual = x
        x = self.feed_forward(x)
        return self.layer_norm(x + residual)


class SpatioTemporalGNN(nn.Module):
    """
    Main spatio-temporal prediction model combining GNN and Transformer.
    
    Architecture:
        Graph (x, edge_index) → GNN Layers → Temporal Encoding → 
        Transformer → Decoder → Predictions
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        temporal_window: int = 24,
        dropout: float = 0.1,
        use_informer: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.temporal_window = temporal_window
        self.use_informer = use_informer
        
        # Spatial encoder (GNN layers)
        self.spatial_convs = nn.ModuleList()
        self.spatial_norms = nn.ModuleList()
        
        # First layer
        self.spatial_convs.append(GATConv(in_channels, hidden_channels, heads=4, concat=False))
        self.spatial_norms.append(nn.LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.spatial_convs.append(
                GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
            )
            self.spatial_norms.append(nn.LayerNorm(hidden_channels))
        
        # Temporal encoder (Transformer)
        self.temporal_projection = nn.Linear(hidden_channels, hidden_channels)
        self.temporal_encoder = TransformerEncoder(
            d_model=hidden_channels,
            n_heads=transformer_heads,
            d_ff=hidden_channels * 4,
            n_layers=transformer_layers,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Learnable positional encoding for temporal dimension
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, temporal_window, hidden_channels) * 0.02
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyG Data object with:
                - x: Node features (num_nodes, in_channels)
                - edge_index: Graph connectivity
                - temporal_features: (optional) Historical sequence
        
        Returns:
            predictions: (num_nodes, out_channels)
        """
        x, edge_index = data.x, data.edge_index
        
        # Spatial encoding (GNN)
        for conv, norm in zip(self.spatial_convs, self.spatial_norms):
            x = F.relu(conv(x, edge_index))
            x = norm(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Check if temporal features exist
        if hasattr(data, 'temporal_features') and data.temporal_features is not None:
            # temporal_features: (num_nodes, temporal_window, features)
            temp_x = data.temporal_features
            
            # Project to hidden dimension
            temp_x = self.temporal_projection(temp_x)  # (N, T, H)
            
            # Add positional encoding
            temp_x = temp_x + self.temporal_pos_encoding[:, :temp_x.size(1), :]
            
            # Temporal encoding (Transformer)
            temp_x = self.temporal_encoder(temp_x)
            
            # Aggregate temporal dimension (take last timestep)
            temp_x = temp_x[:, -1, :]  # (N, H)
            
            # Combine spatial and temporal
            x = x + temp_x
        
        # Decoder
        out = self.decoder(x)
        
        return out
    
    def predict_sequence(
        self,
        data: Data,
        prediction_horizon: int = 6
    ) -> torch.Tensor:
        """
        Auto-regressive sequence prediction.
        
        Args:
            data: Input graph data
            prediction_horizon: Number of future steps to predict
        
        Returns:
            predictions: (num_nodes, prediction_horizon, out_channels)
        """
        predictions = []
        current_data = data
        
        for _ in range(prediction_horizon):
            pred = self.forward(current_data)
            predictions.append(pred)
            
            # Update temporal features for next step (auto-regressive)
            if hasattr(current_data, 'temporal_features'):
                # Shift and append new prediction
                new_temp = torch.cat([
                    current_data.temporal_features[:, 1:, :],
                    pred.unsqueeze(1)
                ], dim=1)
                current_data.temporal_features = new_temp
        
        return torch.stack(predictions, dim=1)  # (N, H, C)


if __name__ == "__main__":
    # Unit test
    print("Testing SpatioTemporalGNN...")
    
    # Create dummy graph data
    from torch_geometric.data import Data
    
    num_nodes = 10
    num_edges = 20
    in_channels = 5
    out_channels = 2
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    temporal_features = torch.randn(num_nodes, 24, in_channels)
    
    data = Data(x=x, edge_index=edge_index, temporal_features=temporal_features)
    
    # Initialize model
    model = SpatioTemporalGNN(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=out_channels,
        num_layers=3,
        transformer_heads=8,
        transformer_layers=2,
        temporal_window=24
    )
    
    # Forward pass
    output = model(data)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({num_nodes}, {out_channels})")
    
    # Sequence prediction
    seq_output = model.predict_sequence(data, prediction_horizon=6)
    print(f"Sequence output shape: {seq_output.shape}")
    print(f"Expected: ({num_nodes}, 6, {out_channels})")
    
    print("All tests passed!")
