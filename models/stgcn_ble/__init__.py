"""
STGCN-BLE: Spatio-Temporal Graph Convolutional Network for BLE-based Trajectory Prediction

This module implements the enhanced STGCN with Mamba-2 encoder for collective pedestrian
trajectory modeling using physics-informed BLE RSSI data.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)
"""

from .model import STGCN_BLE, STGCNConfig
from .rssi_processor import RSSIProcessor, PathLossModel
from .graph_builder import DynamicGraphBuilder

__all__ = ['STGCN_BLE', 'STGCNConfig', 'RSSIProcessor', 'PathLossModel', 'DynamicGraphBuilder']

__version__ = '1.0.0'

