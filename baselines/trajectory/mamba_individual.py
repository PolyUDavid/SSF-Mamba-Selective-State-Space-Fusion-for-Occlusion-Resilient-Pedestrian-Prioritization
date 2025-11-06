"""
Mamba-2 Individual Baseline (No Graph Convolution)

Mamba-2 encoder for trajectory prediction without social interaction modeling.
This baseline tests whether the graph structure contributes to performance.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)
"""

import torch
import torch.nn as nn


class SimplifiedMambaBlock(nn.Module):
    """
    Simplified Mamba-2 block for sequence modeling
    
    Uses gated mechanism and temporal dependency without full SSM complexity.
    """
    def __init__(self, dim: int = 128, expansion_factor: int = 2):
        super().__init__()
        self.dim = dim
        self.expanded_dim = dim * expansion_factor
        
        # Input projection
        self.in_proj = nn.Linear(dim, self.expanded_dim * 2)
        
        # Temporal mixing (simplified)
        self.conv1d = nn.Conv1d(
            self.expanded_dim, 
            self.expanded_dim, 
            kernel_size=3, 
            padding=1, 
            groups=self.expanded_dim
        )
        
        # Output projection
        self.out_proj = nn.Linear(self.expanded_dim, dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, dim)
            
        Returns:
            output: Processed sequence (batch, seq_len, dim)
        """
        residual = x
        x = self.norm(x)
        
        # Split into gate and ssm paths
        x_proj = self.in_proj(x)  # (batch, seq_len, expanded_dim * 2)
        x_gate, x_ssm = x_proj.chunk(2, dim=-1)  # Each: (batch, seq_len, expanded_dim)
        
        # Temporal convolution
        x_ssm = x_ssm.transpose(1, 2)  # (batch, expanded_dim, seq_len)
        x_ssm = self.conv1d(x_ssm)
        x_ssm = x_ssm.transpose(1, 2)  # (batch, seq_len, expanded_dim)
        
        # Gated activation
        x_gate = torch.nn.functional.silu(x_gate)
        x = x_gate * x_ssm
        
        # Output projection
        x = self.out_proj(x)
        
        return x + residual


class MambaIndividual(nn.Module):
    """
    Individual trajectory predictor using Mamba-2 encoder
    
    Processes each pedestrian independently without graph structure.
    This ablation tests the contribution of social interaction modeling.
    """
    def __init__(self,
                 input_dim: int = 4,  # [x, y, vx, vy]
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 pred_len: int = 30):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of Mamba blocks
            pred_len: Prediction horizon
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Mamba-2 encoder
        self.mamba_layers = nn.ModuleList([
            SimplifiedMambaBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2 * pred_len)  # Predict all future positions
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input trajectory (num_peds, obs_len, input_dim)
            
        Returns:
            predictions: Predicted trajectory (num_peds, pred_len, 2)
        """
        # Input embedding
        h = self.input_proj(x)  # (num_peds, obs_len, hidden_dim)
        
        # Mamba encoding
        for layer in self.mamba_layers:
            h = layer(h)  # (num_peds, obs_len, hidden_dim)
        
        # Use last timestep for prediction
        last_state = h[:, -1, :]  # (num_peds, hidden_dim)
        
        # Predict future trajectory
        pred_flat = self.pred_head(last_state)  # (num_peds, 2 * pred_len)
        
        # Reshape to trajectory
        predictions = pred_flat.view(-1, self.pred_len, 2)  # (num_peds, pred_len, 2)
        
        return predictions


def create_mamba_individual(hidden_dim: int = 128,
                           num_layers: int = 4,
                           pred_len: int = 30) -> MambaIndividual:
    """
    Factory function to create Mamba Individual baseline
    
    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of Mamba blocks
        pred_len: Prediction horizon
        
    Returns:
        model: Initialized Mamba Individual model
    """
    model = MambaIndividual(
        input_dim=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pred_len=pred_len
    )
    return model

