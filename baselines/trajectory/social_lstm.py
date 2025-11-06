"""
Social-LSTM Baseline

Implementation of Social LSTM for pedestrian trajectory prediction.
Reference: Alahi et al., "Social LSTM: Human Trajectory Prediction in Crowded Spaces", CVPR 2016

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)
"""

import torch
import torch.nn as nn
from typing import Optional


class SocialPooling(nn.Module):
    """
    Social pooling layer that aggregates information from neighboring pedestrians
    """
    def __init__(self, 
                 hidden_dim: int = 128,
                 grid_size: int = 8,
                 neighborhood_size: float = 32.0):
        """
        Args:
            hidden_dim: Hidden state dimension
            grid_size: Number of grid cells per dimension
            neighborhood_size: Physical size of neighborhood (meters)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        self.cell_size = neighborhood_size / grid_size
        
        # Embedding for pooled neighborhood
        self.pool_embed = nn.Linear(grid_size * grid_size * hidden_dim, hidden_dim)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Hidden states (num_peds, hidden_dim)
            positions: Current positions (num_peds, 2)
            
        Returns:
            pooled_features: Socially pooled features (num_peds, hidden_dim)
        """
        num_peds = hidden_states.shape[0]
        device = hidden_states.device
        
        # Initialize pooled tensors for each pedestrian
        pooled_features = []
        
        for i in range(num_peds):
            # Create grid centered at pedestrian i
            center = positions[i]  # (2,)
            
            # Initialize grid
            grid = torch.zeros(self.grid_size, self.grid_size, self.hidden_dim, device=device)
            
            # Pool neighbors into grid
            for j in range(num_peds):
                if i == j:
                    continue
                
                # Relative position
                rel_pos = positions[j] - center  # (2,)
                
                # Check if in neighborhood
                if torch.abs(rel_pos[0]) > self.neighborhood_size/2 or \
                   torch.abs(rel_pos[1]) > self.neighborhood_size/2:
                    continue
                
                # Convert to grid coordinates
                grid_x = int((rel_pos[0] + self.neighborhood_size/2) / self.cell_size)
                grid_y = int((rel_pos[1] + self.neighborhood_size/2) / self.cell_size)
                
                # Clamp to grid
                grid_x = max(0, min(grid_x, self.grid_size - 1))
                grid_y = max(0, min(grid_y, self.grid_size - 1))
                
                # Add to grid (max pooling)
                grid[grid_y, grid_x] = torch.maximum(grid[grid_y, grid_x], hidden_states[j])
            
            # Flatten and embed grid
            flat_grid = grid.view(-1)  # (grid_size^2 * hidden_dim,)
            pooled = self.pool_embed(flat_grid)  # (hidden_dim,)
            pooled_features.append(pooled)
        
        return torch.stack(pooled_features)  # (num_peds, hidden_dim)


class SocialLSTM(nn.Module):
    """
    Social LSTM for trajectory prediction
    
    Architecture:
    - Encoder LSTM processes observation sequence
    - Social pooling at each step
    - Decoder LSTM generates predictions
    """
    def __init__(self,
                 input_dim: int = 2,
                 embedding_dim: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 grid_size: int = 8,
                 neighborhood_size: float = 32.0):
        """
        Args:
            input_dim: Input feature dimension (usually 2 for x,y)
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            grid_size: Social pooling grid size
            neighborhood_size: Neighborhood size for social pooling (meters)
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input embedding
        self.input_embed = nn.Linear(input_dim, embedding_dim)
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTMCell(embedding_dim + hidden_dim, hidden_dim)
        
        # Social pooling
        self.social_pool = SocialPooling(hidden_dim, grid_size, neighborhood_size)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTMCell(embedding_dim + hidden_dim, hidden_dim)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self,
                obs_traj: torch.Tensor,
                pred_len: int = 30,
                obs_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            obs_traj: Observed trajectory (num_peds, obs_len, input_dim)
            pred_len: Number of steps to predict
            obs_positions: Positions for social pooling (num_peds, obs_len, 2)
                          If None, uses obs_traj directly
            
        Returns:
            pred_traj: Predicted trajectory (num_peds, pred_len, input_dim)
        """
        num_peds, obs_len, _ = obs_traj.shape
        device = obs_traj.device
        
        if obs_positions is None:
            obs_positions = obs_traj  # Assume trajectory is position
        
        # Initialize hidden states
        h = torch.zeros(num_peds, self.hidden_dim, device=device)
        c = torch.zeros(num_peds, self.hidden_dim, device=device)
        
        # Encode observation sequence
        for t in range(obs_len):
            # Embed input
            embedded = torch.relu(self.input_embed(obs_traj[:, t, :]))  # (num_peds, embedding_dim)
            
            # Social pooling
            social_context = self.social_pool(h, obs_positions[:, t, :])  # (num_peds, hidden_dim)
            
            # LSTM step
            lstm_input = torch.cat([embedded, social_context], dim=1)  # (num_peds, embedding_dim + hidden_dim)
            h, c = self.encoder_lstm(lstm_input, (h, c))
        
        # Decode future trajectory
        predictions = []
        current_pos = obs_traj[:, -1, :]  # (num_peds, input_dim)
        
        for t in range(pred_len):
            # Embed current position
            embedded = torch.relu(self.input_embed(current_pos))  # (num_peds, embedding_dim)
            
            # Social pooling (use current predicted positions)
            social_context = self.social_pool(h, current_pos[:, :2])  # (num_peds, hidden_dim)
            
            # LSTM step
            lstm_input = torch.cat([embedded, social_context], dim=1)
            h, c = self.decoder_lstm(lstm_input, (h, c))
            
            # Predict next position
            output = self.output_layer(h)  # (num_peds, input_dim)
            predictions.append(output)
            
            # Update current position
            current_pos = output
        
        pred_traj = torch.stack(predictions, dim=1)  # (num_peds, pred_len, input_dim)
        return pred_traj


def create_social_lstm(hidden_dim: int = 128,
                      grid_size: int = 8) -> SocialLSTM:
    """
    Factory function to create Social-LSTM model
    
    Args:
        hidden_dim: Hidden dimension
        grid_size: Social pooling grid size
        
    Returns:
        model: Initialized Social-LSTM model
    """
    model = SocialLSTM(
        input_dim=2,
        embedding_dim=64,
        hidden_dim=hidden_dim,
        num_layers=1,
        grid_size=grid_size,
        neighborhood_size=32.0
    )
    return model

