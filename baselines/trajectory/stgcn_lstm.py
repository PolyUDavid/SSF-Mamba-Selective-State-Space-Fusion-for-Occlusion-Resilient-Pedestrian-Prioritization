"""
STGCN-LSTM Baseline (Standard LSTM Encoder)

STGCN with standard LSTM instead of Mamba-2 encoder.
This ablation tests the contribution of the Mamba-2 architecture.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)
"""

import torch
import torch.nn as nn
import math


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer
    
    Computes attention-weighted aggregation over spatial neighbors.
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # Linear transformations
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        self.W_out = nn.Linear(out_dim, out_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, in_dim)
            adj_matrix: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            output: Updated node features (num_nodes, out_dim)
        """
        num_nodes = x.shape[0]
        
        # Linear projections
        Q = self.W_q(x).view(num_nodes, self.num_heads, self.head_dim)  # (N, H, D)
        K = self.W_k(x).view(num_nodes, self.num_heads, self.head_dim)  # (N, H, D)
        V = self.W_v(x).view(num_nodes, self.num_heads, self.head_dim)  # (N, H, D)
        
        # Compute attention scores
        scores = torch.einsum('nhd,mhd->nmh', Q, K) / math.sqrt(self.head_dim)  # (N, N, H)
        
        # Mask with adjacency
        mask = adj_matrix.unsqueeze(-1)  # (N, N, 1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=1)  # (N, N, H)
        
        # Aggregate
        output = torch.einsum('nmh,mhd->nhd', attn_weights, V)  # (N, H, D)
        output = output.reshape(num_nodes, self.out_dim)  # (N, out_dim)
        
        # Output projection
        output = self.W_out(output)
        output = self.norm(output + self.W_q(x))  # Residual connection
        
        return output


class STGCN_LSTM(nn.Module):
    """
    Spatio-Temporal GCN with LSTM encoder
    
    Uses standard LSTM for temporal encoding instead of Mamba-2.
    Graph convolution remains for spatial interaction modeling.
    """
    def __init__(self,
                 input_dim: int = 4,
                 hidden_dim: int = 128,
                 num_gcn_layers: int = 3,
                 num_heads: int = 4,
                 lstm_layers: int = 2,
                 pred_len: int = 30):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_gcn_layers: Number of GCN layers
            num_heads: Number of attention heads
            lstm_layers: Number of LSTM layers
            pred_len: Prediction horizon
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.pred_len = pred_len
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM encoder (replaces Mamba-2)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Graph attention layers
        self.gcn_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads)
            for _ in range(num_gcn_layers)
        ])
        
        # Global pooling
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2 * pred_len)
        )
        
    def forward(self, 
                x: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input trajectories (num_peds, obs_len, input_dim)
            adj_matrix: Adjacency matrix (num_peds, num_peds)
            
        Returns:
            predictions: Predicted trajectories (num_peds, pred_len, 2)
        """
        num_peds, obs_len, _ = x.shape
        
        # Input embedding
        h = self.input_proj(x)  # (num_peds, obs_len, hidden_dim)
        
        # LSTM temporal encoding
        h, _ = self.lstm(h)  # (num_peds, obs_len, hidden_dim)
        
        # Use last timestep
        h = h[:, -1, :]  # (num_peds, hidden_dim)
        
        # Graph convolution for spatial aggregation
        for gcn_layer in self.gcn_layers:
            h = gcn_layer(h, adj_matrix)  # (num_peds, hidden_dim)
        
        # Global pooling
        h_global = torch.relu(self.pool_proj(h))  # (num_peds, hidden_dim)
        
        # Predict future trajectory
        pred_flat = self.pred_head(h_global)  # (num_peds, 2 * pred_len)
        
        # Reshape
        predictions = pred_flat.view(num_peds, self.pred_len, 2)
        
        return predictions
    
    def build_adjacency_matrix(self, positions: torch.Tensor, threshold: float = 10.0) -> torch.Tensor:
        """
        Build adjacency matrix based on distance threshold
        
        Args:
            positions: Current positions (num_peds, 2)
            threshold: Distance threshold (meters)
            
        Returns:
            adj_matrix: Adjacency matrix (num_peds, num_peds)
        """
        num_peds = positions.shape[0]
        device = positions.device
        
        # Compute pairwise distances
        pos_expanded_i = positions.unsqueeze(1)  # (num_peds, 1, 2)
        pos_expanded_j = positions.unsqueeze(0)  # (1, num_peds, 2)
        distances = torch.norm(pos_expanded_i - pos_expanded_j, dim=2)  # (num_peds, num_peds)
        
        # Create adjacency matrix
        adj_matrix = (distances < threshold).float()
        
        # Add self-loops
        adj_matrix = adj_matrix + torch.eye(num_peds, device=device)
        
        return adj_matrix


def create_stgcn_lstm(hidden_dim: int = 128,
                      num_gcn_layers: int = 3,
                      pred_len: int = 30) -> STGCN_LSTM:
    """
    Factory function to create STGCN-LSTM baseline
    
    Args:
        hidden_dim: Hidden dimension
        num_gcn_layers: Number of GCN layers
        pred_len: Prediction horizon
        
    Returns:
        model: Initialized STGCN-LSTM model
    """
    model = STGCN_LSTM(
        input_dim=4,
        hidden_dim=hidden_dim,
        num_gcn_layers=num_gcn_layers,
        num_heads=4,
        lstm_layers=2,
        pred_len=pred_len
    )
    return model

