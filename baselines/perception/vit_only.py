"""
ViT-Only Baseline

Vision Transformer processing full frames without YOLOv8 detection stage.
Tests the necessity of the efficient two-stage approach.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them
    """
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding via convolution
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (batch, channels, height, width)
            
        Returns:
            patches: Embedded patches (batch, num_patches, embed_dim)
        """
        x = self.proj(x)  # (batch, embed_dim, H', W')
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer block
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, embed_dim)
            
        Returns:
            output: Transformed output (batch, seq_len, embed_dim)
        """
        # Self-attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class ViTOnlyBaseline(nn.Module):
    """
    Vision Transformer baseline without detection stage
    
    Processes the full image with ViT, which is computationally expensive
    and lacks the focused attention that detection boxes provide.
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 num_age_classes: int = 3):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads
            num_age_classes: Number of age categories
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Task heads
        self.detection_head = nn.Linear(embed_dim, 5)  # [x, y, w, h, conf]
        self.age_head = nn.Linear(embed_dim, num_age_classes)
        self.intent_head = nn.Linear(embed_dim, 2)
        
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass
        
        Args:
            x: Input image (batch, channels, height, width)
            
        Returns:
            outputs: Dictionary with predictions
        """
        batch = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(batch, -1, -1)  # (batch, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)  # (batch, num_patches + 1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        # Extract class token
        cls_output = x[:, 0]  # (batch, embed_dim)
        
        # Task predictions
        detection = self.detection_head(cls_output)
        age_logits = self.age_head(cls_output)
        intent_logits = self.intent_head(cls_output)
        
        return {
            'detection': detection,
            'age': age_logits,
            'intent': intent_logits,
            'features': cls_output
        }
    
    def forward_sequence(self, video: torch.Tensor) -> dict:
        """
        Process video sequence
        
        Args:
            video: Video frames (batch, seq_len, channels, height, width)
            
        Returns:
            outputs: Dictionary with averaged predictions
        """
        batch, seq_len, c, h, w = video.shape
        
        # Process each frame
        all_outputs = []
        for t in range(seq_len):
            frame_output = self.forward(video[:, t, :, :, :])
            all_outputs.append(frame_output)
        
        # Average predictions
        detection = torch.stack([o['detection'] for o in all_outputs]).mean(dim=0)
        age_logits = torch.stack([o['age'] for o in all_outputs]).mean(dim=0)
        intent_logits = torch.stack([o['intent'] for o in all_outputs]).mean(dim=0)
        features = torch.stack([o['features'] for o in all_outputs]).mean(dim=0)
        
        return {
            'detection': detection,
            'age': age_logits,
            'intent': intent_logits,
            'features': features
        }


def create_vit_only_baseline(img_size: int = 224,
                             num_age_classes: int = 3) -> ViTOnlyBaseline:
    """
    Factory function to create ViT-Only baseline
    
    Args:
        img_size: Input image size
        num_age_classes: Number of age categories
        
    Returns:
        model: Initialized ViT-Only model
    """
    model = ViTOnlyBaseline(
        img_size=img_size,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_age_classes=num_age_classes
    )
    return model

