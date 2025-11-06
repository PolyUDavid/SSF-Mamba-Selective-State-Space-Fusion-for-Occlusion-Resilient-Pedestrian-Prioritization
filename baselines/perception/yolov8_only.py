"""
YOLOv8-Only Baseline

Single-stage detector that attempts all tasks (detection, age, intent) 
without the hierarchical ViT refinement stage.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)
"""

import torch
import torch.nn as nn


class YOLOv8OnlyBaseline(nn.Module):
    """
    YOLOv8-Only baseline for multi-task pedestrian perception
    
    Attempts to perform detection, age classification, and intent prediction
    using only the YOLOv8 detector without ViT refinement.
    
    This tests whether the two-stage hierarchical approach is necessary.
    """
    def __init__(self,
                 input_channels: int = 3,
                 num_age_classes: int = 3,
                 feature_dim: int = 256):
        """
        Args:
            input_channels: Number of input channels (RGB=3)
            num_age_classes: Number of age categories
            feature_dim: Feature dimension
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # Simplified YOLOv8 backbone (CSPDarknet-like)
        self.backbone = nn.Sequential(
            # Stem
            nn.Conv2d(input_channels, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            # Stage 1
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            # Stage 2
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            
            # Stage 3
            nn.Conv2d(256, feature_dim, 3, 2, 1),
            nn.BatchNorm2d(feature_dim),
            nn.SiLU(),
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Detection head (simplified)
        self.det_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # [x, y, w, h, confidence]
        )
        
        # Age classification head
        self.age_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_age_classes)
        )
        
        # Intent prediction head (naive - without temporal context)
        self.intent_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary: cross / not-cross
        )
        
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass
        
        Args:
            x: Input image (batch, channels, height, width)
            
        Returns:
            outputs: Dictionary containing:
                - 'detection': Bounding box predictions (batch, 5)
                - 'age': Age class logits (batch, num_age_classes)
                - 'intent': Intent logits (batch, 2)
                - 'features': Extracted features (batch, feature_dim)
        """
        # Backbone feature extraction
        features = self.backbone(x)  # (batch, feature_dim, H', W')
        
        # Global pooling
        pooled = self.pool(features)  # (batch, feature_dim, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch, feature_dim)
        
        # Multi-task predictions
        detection = self.det_head(pooled)  # (batch, 5)
        age_logits = self.age_head(pooled)  # (batch, num_age_classes)
        intent_logits = self.intent_head(pooled)  # (batch, 2)
        
        return {
            'detection': detection,
            'age': age_logits,
            'intent': intent_logits,
            'features': pooled
        }
    
    def forward_sequence(self, video: torch.Tensor) -> dict:
        """
        Process video sequence
        
        Args:
            video: Video frames (batch, seq_len, channels, height, width)
            
        Returns:
            outputs: Dictionary with predictions
        """
        batch, seq_len, c, h, w = video.shape
        
        # Process each frame independently
        all_detections = []
        all_age_logits = []
        all_intent_logits = []
        all_features = []
        
        for t in range(seq_len):
            frame_output = self.forward(video[:, t, :, :, :])
            all_detections.append(frame_output['detection'])
            all_age_logits.append(frame_output['age'])
            all_intent_logits.append(frame_output['intent'])
            all_features.append(frame_output['features'])
        
        # Average over sequence (naive temporal aggregation)
        detection = torch.stack(all_detections).mean(dim=0)  # (batch, 5)
        age_logits = torch.stack(all_age_logits).mean(dim=0)  # (batch, num_age_classes)
        intent_logits = torch.stack(all_intent_logits).mean(dim=0)  # (batch, 2)
        features = torch.stack(all_features).mean(dim=0)  # (batch, feature_dim)
        
        return {
            'detection': detection,
            'age': age_logits,
            'intent': intent_logits,
            'features': features
        }


def create_yolov8_only_baseline(num_age_classes: int = 3) -> YOLOv8OnlyBaseline:
    """
    Factory function to create YOLOv8-Only baseline
    
    Args:
        num_age_classes: Number of age categories
        
    Returns:
        model: Initialized YOLOv8-Only model
    """
    model = YOLOv8OnlyBaseline(
        input_channels=3,
        num_age_classes=num_age_classes,
        feature_dim=256
    )
    return model

