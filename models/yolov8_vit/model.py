"""
YOLOv8-ViT Model Implementation

Hierarchical perception pipeline for pedestrian detection, age classification,
and crossing intent prediction.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)

Architecture:
    Stage 1 (YOLOv8-Nano): Fast detection + age estimation
    Stage 2 (Temporal ViT): Fine-grained intent prediction from video tubes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class PerceptionConfig:
    """Configuration for YOLOv8-ViT perception module"""
    # Input
    image_size: int = 640                    # Input image resolution
    num_classes: int = 1                     # Number of object classes (pedestrian)
    
    # YOLOv8 architecture
    yolo_depth_multiple: float = 0.33        # Depth scaling (Nano version)
    yolo_width_multiple: float = 0.25        # Width scaling (Nano version)
    
    # ViT architecture
    patch_size: int = 16                     # Patch size for ViT
    vit_hidden_dim: int = 384                # ViT hidden dimension
    vit_layers: int = 6                      # Number of transformer layers
    vit_heads: int = 6                       # Number of attention heads
    sequence_length: int = 16                # Temporal window (frames)
    
    # Task heads
    num_age_classes: int = 3                 # child, adult, elderly
    
    # Training
    dropout: float = 0.1                     # Dropout rate


class CSPLayer(nn.Module):
    """
    Cross Stage Partial Layer (used in YOLOv8 backbone)
    
    Splits feature maps into two paths for efficient gradient flow.
    """
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1):
        super().__init__()
        hidden_channels = out_channels // 2
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.conv3 = nn.Conv2d(2 * hidden_channels, out_channels, 1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Bottleneck blocks
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True)
            ) for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        x1 = self.bn1(self.conv1(x))
        x1 = self.blocks(x1)
        
        x2 = self.bn2(self.conv2(x))
        
        x = torch.cat([x1, x2], dim=1)
        x = self.bn3(self.conv3(x))
        x = F.silu(x, inplace=True)
        
        return x


class YOLOv8Nano(nn.Module):
    """
    YOLOv8-Nano backbone for real-time pedestrian detection
    
    Lightweight architecture optimized for edge deployment while maintaining
    high detection accuracy.
    """
    def __init__(self, config: PerceptionConfig):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True)
        )
        
        # Backbone stages
        self.stage1 = CSPLayer(16, 32, num_blocks=1)
        self.downsample1 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        
        self.stage2 = CSPLayer(32, 64, num_blocks=2)
        self.downsample2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        
        self.stage3 = CSPLayer(64, 128, num_blocks=2)
        self.downsample3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        
        self.stage4 = CSPLayer(128, 256, num_blocks=1)
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # (x, y, w, h, objectness)
        )
        
        # Age classification head
        self.age_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_age_classes)
        )
        
        # Global feature for ViT
        self.feature_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        Args:
            x: Input image (batch, 3, H, W)
            
        Returns:
            detection: Bounding box predictions (batch, 5)
            age_logits: Age classification logits (batch, 3)
            features: Global features for ViT (batch, 256)
        """
        # Backbone
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample1(x)
        x = self.stage2(x)
        x = self.downsample2(x)
        x = self.stage3(x)
        x = self.downsample3(x)
        x = self.stage4(x)
        
        # Multi-task outputs
        detection = self.detection_head(x)
        age_logits = self.age_head(x)
        
        # Extract global features
        features = self.feature_pool(x).squeeze(-1).squeeze(-1)
        
        return detection, age_logits, features


class TemporalViT(nn.Module):
    """
    Temporal Vision Transformer for crossing intent prediction
    
    Processes video tubes (sequence of crops) to predict pedestrian crossing intent
    using self-attention over spatial and temporal dimensions.
    """
    def __init__(self, config: PerceptionConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.vit_hidden_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, self.hidden_dim, 
                                     kernel_size=config.patch_size, 
                                     stride=config.patch_size)
        
        # Position embeddings
        # For 64x64 crop with patch_size=16: (64/16)^2 = 16 patches per frame
        num_spatial_patches = (64 // config.patch_size) ** 2
        num_total_tokens = config.sequence_length * num_spatial_patches + 1  # +1 for CLS
        
        self.pos_embed = nn.Parameter(torch.randn(1, num_total_tokens, self.hidden_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        
        # Age embedding (optional conditioning)
        self.age_embed = nn.Embedding(config.num_age_classes, self.hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.vit_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.vit_layers)
        
        # Intent prediction head
        self.intent_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim // 2, 2)  # Binary: cross / not cross
        )
        
    def forward(self, video_tube, age_labels=None):
        """
        Args:
            video_tube: Sequence of pedestrian crops (batch, seq_len, 3, 64, 64)
            age_labels: Optional age category indices (batch,)
            
        Returns:
            intent_logits: Intent prediction logits (batch, 2)
        """
        batch_size, seq_len, C, H, W = video_tube.shape
        
        # Flatten temporal dimension for patching
        video_flat = video_tube.view(batch_size * seq_len, C, H, W)
        
        # Extract patches
        patches = self.patch_embed(video_flat)  # (batch*seq_len, hidden_dim, H_p, W_p)
        patches = patches.flatten(2).transpose(1, 2)  # (batch*seq_len, num_patches, hidden_dim)
        
        # Reshape back to temporal dimension
        num_patches = patches.shape[1]
        patches = patches.view(batch_size, seq_len * num_patches, self.hidden_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        
        # Add positional embeddings
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        
        # Optionally add age conditioning
        if age_labels is not None:
            age_emb = self.age_embed(age_labels).unsqueeze(1)  # (batch, 1, hidden_dim)
            # Add age embedding to CLS token
            tokens[:, 0:1, :] = tokens[:, 0:1, :] + age_emb
        
        # Transformer encoding
        tokens = self.transformer(tokens)
        
        # Use CLS token for prediction
        cls_output = tokens[:, 0, :]
        
        # Predict intent
        intent_logits = self.intent_head(cls_output)
        
        return intent_logits


class YOLOv8_ViT(nn.Module):
    """
    Complete hierarchical perception pipeline
    
    Stage 1: YOLOv8-Nano for detection and coarse attributes
    Stage 2: Temporal ViT for fine-grained intent prediction
    """
    def __init__(self, config: Optional[PerceptionConfig] = None):
        super().__init__()
        
        if config is None:
            config = PerceptionConfig()
        self.config = config
        
        # Stage 1: YOLOv8-Nano
        self.yolo = YOLOv8Nano(config)
        
        # Stage 2: Temporal ViT
        self.vit = TemporalViT(config)
        
        # Feature fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(256 + config.vit_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128)
        )
        
    def forward_stage1(self, image):
        """
        Stage 1: Fast detection and age estimation
        
        Args:
            image: Input image (batch, 3, H, W)
            
        Returns:
            detection: Bounding boxes (batch, 5)
            age_logits: Age predictions (batch, 3)
            yolo_features: Global features (batch, 256)
        """
        return self.yolo(image)
    
    def forward_stage2(self, video_tube, age_labels=None):
        """
        Stage 2: Intent prediction from video sequence
        
        Args:
            video_tube: Cropped pedestrian sequence (batch, seq_len, 3, 64, 64)
            age_labels: Optional age conditioning (batch,)
            
        Returns:
            intent_logits: Intent predictions (batch, 2)
        """
        return self.vit(video_tube, age_labels)
    
    def forward(self, image, video_tube=None, age_labels=None):
        """
        Full forward pass through both stages
        
        Args:
            image: Current frame (batch, 3, H, W)
            video_tube: Optional temporal sequence (batch, seq_len, 3, 64, 64)
            age_labels: Optional age labels (batch,)
            
        Returns:
            Dictionary with all predictions
        """
        # Stage 1
        detection, age_logits, yolo_feat = self.forward_stage1(image)
        
        outputs = {
            'detection': detection,
            'age_logits': age_logits,
            'yolo_features': yolo_feat
        }
        
        # Stage 2 (if video tube provided)
        if video_tube is not None:
            intent_logits = self.forward_stage2(video_tube, age_labels)
            outputs['intent_logits'] = intent_logits
            
            # Fuse features
            # Average ViT features over time
            vit_feat = self.vit.transformer(
                self.vit.patch_embed(video_tube[:, -1]).flatten(2).transpose(1, 2)
            )[:, 0, :]
            
            fused_feat = self.fusion_mlp(torch.cat([yolo_feat, vit_feat], dim=-1))
            outputs['fused_features'] = fused_feat
        
        return outputs


def create_yolov8_vit(config: Optional[PerceptionConfig] = None) -> YOLOv8_ViT:
    """
    Factory function to create YOLOv8-ViT model
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    model = YOLOv8_ViT(config)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    return model

