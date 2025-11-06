"""
Mamba-2 Cross-Modal Fusion Implementation

Integrates wearable (BLE) and visual modalities using selective state space
mechanisms with cross-attention for robust pedestrian state representation.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)

Key Features:
    - Cross-modal attention for modality interaction
    - Selective State Space Model (SSM) for temporal memory
    - Dynamic modality weighting for occlusion robustness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FusionConfig:
    """Configuration for Mamba-2 fusion module"""
    # Input dimensions
    ble_feature_dim: int = 128          # STGCN-BLE output dimension
    visual_feature_dim: int = 256       # YOLOv8-ViT output dimension
    
    # Fusion architecture
    hidden_dim: int = 256               # Common hidden dimension
    num_mamba_blocks: int = 6           # Number of Mamba-2 blocks
    num_heads: int = 8                  # Attention heads
    
    # Sequence length
    target_length: int = 32             # Aligned sequence length
    
    # Training
    dropout: float = 0.1


class ModalityProjection(nn.Module):
    """
    Projects different modalities to common hidden dimension
    and adds learnable modality embeddings.
    """
    def __init__(self, input_dim: int, hidden_dim: int, modality_name: str):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.modality_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.norm = nn.LayerNorm(hidden_dim)
        self.modality_name = modality_name
        
    def forward(self, x):
        """
        Args:
            x: Input features (batch, seq_len, input_dim)
            
        Returns:
            Projected features (batch, seq_len, hidden_dim)
        """
        x = self.projection(x)
        x = x + self.modality_embed
        x = self.norm(x)
        return x


class CrossModalAttention(nn.Module):
    """
    Cross-attention layer for inter-modality interaction
    
    One modality queries the other to learn complementary information.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, query_input, key_value_input):
        """
        Args:
            query_input: Query modality (batch, seq_len_q, hidden_dim)
            key_value_input: Key/Value modality (batch, seq_len_kv, hidden_dim)
            
        Returns:
            Attended features (batch, seq_len_q, hidden_dim)
        """
        batch_size = query_input.shape[0]
        residual = query_input
        
        # Project to Q, K, V
        Q = self.query(query_input)
        K = self.key(key_value_input)
        V = self.value(key_value_input)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # Residual and norm
        out = self.norm(out + residual)
        
        return out, attn_weights


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (core of Mamba-2)
    
    Maintains temporal state with selective gating for efficient long-range
    dependency modeling.
    """
    def __init__(self, dim: int, state_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        
        # Input projection
        self.in_proj = nn.Linear(dim, dim * 2)  # Split into gate and input
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.02)
        self.B = nn.Parameter(torch.randn(state_dim, dim) * 0.02)
        self.C = nn.Parameter(torch.randn(dim, state_dim) * 0.02)
        
        # Selection mechanism
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        """
        Args:
            x: Input sequence (batch, seq_len, dim)
            
        Returns:
            Output sequence (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape
        
        # Project input
        x_proj = self.in_proj(x)
        x_gate, x_input = x_proj.chunk(2, dim=-1)
        
        # Apply selection gate
        gate = self.gate(x)
        x_input = x_input * gate
        
        # Initialize state
        h = torch.zeros(batch_size, self.state_dim, device=x.device)
        
        # Recurrent processing
        outputs = []
        for t in range(seq_len):
            # Update state
            h = torch.matmul(h, self.A.t()) + torch.matmul(x_input[:, t, :], self.B.t())
            
            # Compute output
            y = torch.matmul(h, self.C.t())
            outputs.append(y)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        
        # Output projection
        outputs = self.out_proj(outputs)
        
        return outputs


class MambaBlock(nn.Module):
    """
    Complete Mamba-2 block with SSM and feedforward network
    """
    def __init__(self, dim: int, state_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ssm = SelectiveSSM(dim, state_dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input (batch, seq_len, dim)
            
        Returns:
            Output (batch, seq_len, dim)
        """
        # SSM with residual
        x = x + self.ssm(self.norm1(x))
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x


class Mamba2Fusion(nn.Module):
    """
    Complete Mamba-2 Cross-Modal Fusion Module
    
    Fuses BLE trajectory features with visual perception features through:
        1. Modality projection and embedding
        2. Sequence alignment (interpolation)
        3. Cross-modal attention (bidirectional)
        4. Selective state space modeling
        5. Global pooling for unified representation
    """
    def __init__(self, config: Optional[FusionConfig] = None):
        super().__init__()
        
        if config is None:
            config = FusionConfig()
        self.config = config
        
        # Modality projections
        self.ble_proj = ModalityProjection(config.ble_feature_dim, config.hidden_dim, 'BLE')
        self.visual_proj = ModalityProjection(config.visual_feature_dim, config.hidden_dim, 'Visual')
        
        # Cross-modal attention layers
        self.cross_attn_ble_to_vis = CrossModalAttention(config.hidden_dim, config.num_heads, config.dropout)
        self.cross_attn_vis_to_ble = CrossModalAttention(config.hidden_dim, config.num_heads, config.dropout)
        
        # Mamba-2 blocks for temporal modeling
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(config.hidden_dim, state_dim=16, dropout=config.dropout)
            for _ in range(config.num_mamba_blocks)
        ])
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
    def align_sequences(self, x, target_length):
        """
        Align sequence length through interpolation
        
        Args:
            x: Input sequence (batch, seq_len, dim)
            target_length: Target sequence length
            
        Returns:
            Aligned sequence (batch, target_length, dim)
        """
        if x.shape[1] == target_length:
            return x
        
        # Transpose for interpolation (batch, dim, seq_len)
        x = x.transpose(1, 2)
        
        # Interpolate
        x_aligned = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        
        # Transpose back
        x_aligned = x_aligned.transpose(1, 2)
        
        return x_aligned
        
    def forward(self, ble_features, visual_features):
        """
        Fuse BLE and visual modalities
        
        Args:
            ble_features: BLE trajectory features (batch, seq_len_ble, ble_dim)
            visual_features: Visual features (batch, seq_len_vis, vis_dim)
            
        Returns:
            fused_features: Unified representation (batch, hidden_dim)
            modality_weights: Attention weights for analysis
        """
        # Project to common space
        ble_proj = self.ble_proj(ble_features)
        vis_proj = self.visual_proj(visual_features)
        
        # Align sequence lengths
        ble_aligned = self.align_sequences(ble_proj, self.config.target_length)
        vis_aligned = self.align_sequences(vis_proj, self.config.target_length)
        
        # Cross-modal attention (bidirectional)
        ble_attended, attn_ble_to_vis = self.cross_attn_ble_to_vis(ble_aligned, vis_aligned)
        vis_attended, attn_vis_to_ble = self.cross_attn_vis_to_ble(vis_aligned, ble_aligned)
        
        # Concatenate attended features
        fused = ble_attended + vis_attended  # Element-wise sum for fusion
        
        # Temporal modeling with Mamba-2
        for mamba_block in self.mamba_blocks:
            fused = mamba_block(fused)
        
        # Global pooling (mean over time)
        fused_global = torch.mean(fused, dim=1)  # (batch, hidden_dim)
        
        # Final projection
        fused_global = self.final_proj(fused_global)
        
        # Compute modality weights for analysis
        # Average attention weights over heads and sequence
        weight_ble = attn_ble_to_vis.mean(dim=(1, 2, 3))  # (batch,)
        weight_vis = attn_vis_to_ble.mean(dim=(1, 2, 3))  # (batch,)
        
        # Normalize
        total = weight_ble + weight_vis + 1e-6
        modality_weights = {
            'ble': weight_ble / total,
            'visual': weight_vis / total
        }
        
        return fused_global, modality_weights
    
    def get_modality_importance(self, ble_features, visual_features):
        """
        Analyze modality importance (for visualization)
        
        Returns dictionary with importance scores.
        """
        with torch.no_grad():
            _, weights = self.forward(ble_features, visual_features)
        return weights


def create_mamba2_fusion(config: Optional[FusionConfig] = None) -> Mamba2Fusion:
    """
    Factory function to create Mamba-2 fusion model
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    model = Mamba2Fusion(config)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    return model

