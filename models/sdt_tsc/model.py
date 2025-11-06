"""
SDT-TSC Model Implementation

Sequence Decision Transformer for adaptive traffic signal control.
Uses transformer decoder to generate action sequences conditioned on
pedestrian state and desired rewards.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)

Architecture:
    Input: (state, action, reward) trajectories
    Output: Next action predictions
    
    Based on Decision Transformer paradigm with custom action vocabulary
    for traffic signal control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class DecisionConfig:
    """Configuration for SDT-TSC model"""
    # State representation
    state_dim: int = 256                    # Fused pedestrian state dimension
    vehicle_state_dim: int = 32             # Vehicle state features
    
    # Action space
    vocab_size: int = 215                   # Number of action tokens
    max_sequence_length: int = 10           # Maximum action sequence length
    
    # Model architecture
    hidden_dim: int = 512                   # Transformer hidden dimension
    num_layers: int = 6                     # Number of transformer layers
    num_heads: int = 8                      # Attention heads
    
    # Context
    context_length: int = 20                # Historical trajectory length
    
    # Training
    dropout: float = 0.1


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
        
    def forward(self, x):
        """
        Args:
            x: Input (batch, seq_len, d_model)
            
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class RewardPredictor(nn.Module):
    """
    Predicts future rewards from state-action pairs
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state, action):
        """
        Args:
            state: State features (batch, state_dim)
            action: Action embeddings (batch, action_dim)
            
        Returns:
            Predicted reward (batch, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class ActionTokenizer:
    """
    Converts between action tokens and traffic signal commands
    
    Vocabulary includes:
        - IDLE: Maintain current phase
        - EXTEND_PED: Extend pedestrian green
        - EXTEND_VEH: Extend vehicle green
        - TRANSITION: Switch phase
        - PRIORITY_CHILD: Prioritize child pedestrians
        - PRIORITY_ELDERLY: Prioritize elderly pedestrians
        - BATCH_SERVICE: Clear entire pedestrian queue
        - ... (total 215 tokens)
    """
    def __init__(self, vocab_size: int = 215):
        self.vocab_size = vocab_size
        
        # Define core action tokens
        self.IDLE = 0
        self.EXTEND_PED = 1
        self.EXTEND_VEH = 2
        self.TRANSITION = 3
        self.PRIORITY_CHILD = 4
        self.PRIORITY_ELDERLY = 5
        self.BATCH_SERVICE = 6
        self.END = 7
        
        # Action names for interpretation
        self.action_names = {
            0: 'IDLE',
            1: 'EXTEND_PED',
            2: 'EXTEND_VEH',
            3: 'TRANSITION',
            4: 'PRIORITY_CHILD',
            5: 'PRIORITY_ELDERLY',
            6: 'BATCH_SERVICE',
            7: 'END'
        }
        
    def encode(self, action_sequence):
        """
        Encode action names to token IDs
        
        Args:
            action_sequence: List of action names
            
        Returns:
            List of token IDs
        """
        # Simple implementation - can be extended
        name_to_id = {v: k for k, v in self.action_names.items()}
        return [name_to_id.get(action, self.IDLE) for action in action_sequence]
    
    def decode(self, token_ids):
        """
        Decode token IDs to action names
        
        Args:
            token_ids: List or tensor of token IDs
            
        Returns:
            List of action names
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        
        return [self.action_names.get(int(token_id), 'UNKNOWN') for token_id in token_ids]


class SDT_TSC(nn.Module):
    """
    Sequence Decision Transformer for Traffic Signal Control
    
    Implements a transformer decoder that:
        1. Encodes state, action, reward trajectories
        2. Generates action sequences autoregressively
        3. Optimizes for cumulative reward (return-to-go)
    
    Key innovation: Uses token-based action representation for
    interpretable and composable control policies.
    """
    def __init__(self, config: Optional[DecisionConfig] = None):
        super().__init__()
        
        if config is None:
            config = DecisionConfig()
        self.config = config
        
        # Input embeddings
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim + config.vehicle_state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim)
        )
        
        self.action_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.reward_encoder = nn.Linear(1, config.hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_dim, 
                                              max_len=config.context_length * 3)  # *3 for (s, a, r)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.num_layers)
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.vocab_size)
        )
        
        # Reward predictor (for planning)
        self.reward_predictor = RewardPredictor(config.hidden_dim, config.hidden_dim)
        
        # Action tokenizer
        self.tokenizer = ActionTokenizer(config.vocab_size)
        
    def forward(self, 
                states, 
                actions, 
                rewards, 
                returns_to_go,
                mask=None):
        """
        Forward pass through decision transformer
        
        Args:
            states: State sequence (batch, seq_len, state_dim)
            actions: Action sequence (batch, seq_len) [token IDs]
            rewards: Reward sequence (batch, seq_len, 1)
            returns_to_go: Cumulative future rewards (batch, seq_len, 1)
            mask: Optional attention mask
            
        Returns:
            action_logits: Predicted action distributions (batch, seq_len, vocab_size)
            predicted_rewards: Predicted rewards (batch, seq_len, 1)
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Encode inputs
        state_emb = self.state_encoder(states)  # (batch, seq_len, hidden_dim)
        action_emb = self.action_embed(actions)  # (batch, seq_len, hidden_dim)
        reward_emb = self.reward_encoder(rewards)  # (batch, seq_len, hidden_dim)
        rtg_emb = self.reward_encoder(returns_to_go)  # (batch, seq_len, hidden_dim)
        
        # Interleave: (rtg_1, s_1, a_1, rtg_2, s_2, a_2, ...)
        # Create sequence of shape (batch, seq_len*3, hidden_dim)
        sequence = torch.stack([rtg_emb, state_emb, action_emb], dim=2)  # (batch, seq_len, 3, hidden_dim)
        sequence = sequence.reshape(batch_size, seq_len * 3, self.config.hidden_dim)
        
        # Add positional encoding
        sequence = self.pos_encoding(sequence)
        
        # Create causal mask
        if mask is None:
            mask = self.generate_square_subsequent_mask(seq_len * 3).to(sequence.device)
        
        # Transformer decoder
        # For decoder, we need memory (encoder output) - use sequence as both
        decoded = self.transformer_decoder(sequence, sequence, tgt_mask=mask)
        
        # Extract action positions (every third position, offset by 2)
        action_positions = decoded[:, 2::3, :]  # (batch, seq_len, hidden_dim)
        
        # Predict actions
        action_logits = self.action_head(action_positions)
        
        # Predict rewards
        predicted_rewards = self.reward_predictor(state_emb, action_emb)
        
        return action_logits, predicted_rewards
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask for autoregressive generation
        
        Args:
            sz: Sequence length
            
        Returns:
            Mask tensor (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def generate_actions(self, 
                        state, 
                        vehicle_state,
                        target_return,
                        max_length=10,
                        temperature=1.0):
        """
        Generate action sequence autoregressively
        
        Args:
            state: Current pedestrian state (batch, state_dim)
            vehicle_state: Current vehicle state (batch, vehicle_state_dim)
            target_return: Desired cumulative reward (scalar or batch)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            
        Returns:
            action_sequence: Generated action tokens (batch, seq_len)
            action_names: Decoded action names
        """
        self.eval()
        batch_size = state.shape[0]
        device = state.device
        
        # Concatenate states
        full_state = torch.cat([state, vehicle_state], dim=-1).unsqueeze(1)  # (batch, 1, state_dim)
        
        # Initialize sequence
        actions = torch.zeros(batch_size, 1, dtype=torch.long, device=device)  # Start with IDLE
        rewards = torch.zeros(batch_size, 1, 1, device=device)
        
        if isinstance(target_return, (int, float)):
            returns_to_go = torch.full((batch_size, 1, 1), target_return, device=device)
        else:
            returns_to_go = target_return.view(batch_size, 1, 1)
        
        # Autoregressive generation
        for _ in range(max_length):
            # Forward pass
            action_logits, _ = self.forward(full_state, actions, rewards, returns_to_go)
            
            # Get logits for last position
            next_logits = action_logits[:, -1, :] / temperature
            
            # Sample next action
            probs = F.softmax(next_logits, dim=-1)
            next_action = torch.multinomial(probs, 1)  # (batch, 1)
            
            # Check for END token
            if (next_action == self.tokenizer.END).all():
                break
            
            # Append to sequence
            actions = torch.cat([actions, next_action], dim=1)
            
            # Update returns-to-go (placeholder - should use reward predictor)
            next_reward = torch.zeros(batch_size, 1, 1, device=device)
            rewards = torch.cat([rewards, next_reward], dim=1)
            
            next_rtg = returns_to_go[:, -1:, :] - next_reward
            returns_to_go = torch.cat([returns_to_go, next_rtg], dim=1)
        
        # Decode actions
        action_names = []
        for b in range(batch_size):
            names = self.tokenizer.decode(actions[b])
            action_names.append(names)
        
        return actions, action_names
    
    def get_model_size(self):
        """Calculate total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_sdt_tsc(config: Optional[DecisionConfig] = None) -> SDT_TSC:
    """
    Factory function to create SDT-TSC model
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    model = SDT_TSC(config)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(m.weight if hasattr(m, 'weight') else m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    return model

