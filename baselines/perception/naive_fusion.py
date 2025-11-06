"""
Naive Fusion Baseline

YOLOv8 + ViT with late concatenation fusion (no internal conditioning).
Tests the benefit of our internal fusion strategy (age injection into ViT).

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)
"""

import torch
import torch.nn as nn


class NaiveFusionBaseline(nn.Module):
    """
    Naive fusion baseline
    
    Runs YOLOv8 and ViT independently, then concatenates their features
    at the end. This tests whether our internal conditioning strategy
    (injecting age as a token into ViT) provides benefits.
    """
    def __init__(self,
                 yolo_feature_dim: int = 256,
                 vit_feature_dim: int = 768,
                 num_age_classes: int = 3):
        """
        Args:
            yolo_feature_dim: YOLOv8 feature dimension
            vit_feature_dim: ViT feature dimension
            num_age_classes: Number of age categories
        """
        super().__init__()
        
        # Simplified YOLOv8 (detection + age)
        self.yolo_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, yolo_feature_dim, 3, 2, 1),
            nn.BatchNorm2d(yolo_feature_dim),
            nn.SiLU(),
        )
        self.yolo_pool = nn.AdaptiveAvgPool2d(1)
        
        # Detection head
        self.det_head = nn.Sequential(
            nn.Linear(yolo_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        # Age head
        self.age_head = nn.Sequential(
            nn.Linear(yolo_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_age_classes)
        )
        
        # Simplified ViT (intent)
        self.vit_patch_embed = nn.Conv2d(3, vit_feature_dim, 16, 16)
        self.vit_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(vit_feature_dim, 8, batch_first=True),
            num_layers=6
        )
        self.vit_cls_token = nn.Parameter(torch.zeros(1, 1, vit_feature_dim))
        
        # Late fusion: concatenate YOLOv8 and ViT features
        # NOTE: In our full model, age is injected INTO the ViT as a token
        # Here, we just concatenate at the end (naive)
        fusion_dim = yolo_feature_dim + vit_feature_dim
        
        self.intent_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass
        
        Args:
            x: Input image (batch, channels, height, width)
            
        Returns:
            outputs: Dictionary with predictions
        """
        batch = x.shape[0]
        
        # ===== YOLOv8 Branch =====
        yolo_features = self.yolo_backbone(x)  # (batch, yolo_dim, H', W')
        yolo_features = self.yolo_pool(yolo_features).view(batch, -1)  # (batch, yolo_dim)
        
        detection = self.det_head(yolo_features)
        age_logits = self.age_head(yolo_features)
        
        # ===== ViT Branch =====
        vit_patches = self.vit_patch_embed(x)  # (batch, vit_dim, H'', W'')
        vit_patches = vit_patches.flatten(2).transpose(1, 2)  # (batch, num_patches, vit_dim)
        
        # Add class token
        cls_token = self.vit_cls_token.expand(batch, -1, -1)
        vit_patches = torch.cat([cls_token, vit_patches], dim=1)
        
        # Transformer
        vit_features = self.vit_transformer(vit_patches)
        vit_cls = vit_features[:, 0, :]  # (batch, vit_dim)
        
        # ===== Late Fusion =====
        # Naive concatenation (vs. our internal conditioning)
        fused_features = torch.cat([yolo_features, vit_cls], dim=1)  # (batch, yolo_dim + vit_dim)
        
        intent_logits = self.intent_head(fused_features)
        
        return {
            'detection': detection,
            'age': age_logits,
            'intent': intent_logits,
            'features': fused_features
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
        
        all_outputs = []
        for t in range(seq_len):
            frame_output = self.forward(video[:, t, :, :, :])
            all_outputs.append(frame_output)
        
        # Average
        detection = torch.stack([o['detection'] for o in all_outputs]).mean(dim=0)
        age_logits = torch.stack([o['age'] for o in all_outputs]).mean(dim=0)
        intent_logits = torch.stack([o['intent'] for o in all_outputs]).mean(dim=0)
        
        return {
            'detection': detection,
            'age': age_logits,
            'intent': intent_logits
        }


def create_naive_fusion_baseline(num_age_classes: int = 3) -> NaiveFusionBaseline:
    """
    Factory function to create Naive Fusion baseline
    
    Args:
        num_age_classes: Number of age categories
        
    Returns:
        model: Initialized Naive Fusion model
    """
    model = NaiveFusionBaseline(
        yolo_feature_dim=256,
        vit_feature_dim=768,
        num_age_classes=num_age_classes
    )
    return model

