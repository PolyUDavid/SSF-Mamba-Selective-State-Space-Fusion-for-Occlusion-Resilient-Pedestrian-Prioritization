"""
Mamba-2 Cross-Modal Fusion Module

Fuses BLE-based trajectory features with visual perception features
using selective state space mechanisms for robust multi-modal integration.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)
"""

from .model import Mamba2Fusion, FusionConfig
from .cross_attention import CrossModalAttention
from .state_space import SelectiveSSM

__all__ = ['Mamba2Fusion', 'FusionConfig', 'CrossModalAttention', 'SelectiveSSM']

__version__ = '1.0.0'

