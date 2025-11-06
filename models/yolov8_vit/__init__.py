"""
YOLOv8-ViT: Hierarchical Visual Perception Module

Combines YOLOv8-Nano for real-time detection with Vision Transformer
for fine-grained intent prediction.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)
"""

from .model import YOLOv8_ViT, PerceptionConfig
from .yolov8_backbone import YOLOv8Nano
from .vit_head import TemporalViT

__all__ = ['YOLOv8_ViT', 'PerceptionConfig', 'YOLOv8Nano', 'TemporalViT']

__version__ = '1.0.0'

