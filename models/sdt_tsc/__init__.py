"""
SDT-TSC: Sequence Decision Transformer for Traffic Signal Control

Model-based reinforcement learning agent that learns optimal traffic signal
control policy from unified pedestrian state representation.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)
"""

from .model import SDT_TSC, DecisionConfig
from .action_tokenizer import ActionTokenizer
from .reward_model import RewardPredictor

__all__ = ['SDT_TSC', 'DecisionConfig', 'ActionTokenizer', 'RewardPredictor']

__version__ = '1.0.0'

