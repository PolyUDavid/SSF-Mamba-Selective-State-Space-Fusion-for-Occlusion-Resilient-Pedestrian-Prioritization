"""
Baseline Models for Visual Perception

Comparison baselines for the YOLOv8-ViT perception module.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)
"""

from .yolov8_only import YOLOv8OnlyBaseline
from .vit_only import ViTOnlyBaseline
from .naive_fusion import NaiveFusionBaseline

__all__ = ['YOLOv8OnlyBaseline', 'ViTOnlyBaseline', 'NaiveFusionBaseline']

