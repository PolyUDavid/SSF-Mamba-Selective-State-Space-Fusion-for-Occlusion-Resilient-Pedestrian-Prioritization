"""
BLE RSSI Signal Processing Module

Physics-informed processing of BLE RSSI signals for pedestrian localization.
Implements enhanced path loss model with learnable parameters for environmental adaptation.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class PathLossModel(nn.Module):
    """
    Enhanced path loss model for BLE RSSI signal propagation
    
    Implements the log-distance path loss model with additional terms for:
        - Environmental shadowing (Gaussian noise)
        - Body shadowing (exponential decay with distance)
        - Learnable parameters for adaptation
    
    Model:
        RSSI = P_tx - (20*log10(d) + n*10*log10(d)) - X_shadow - L_body
    
    where:
        P_tx: Transmission power (learnable)
        d: Distance to scanner
        n: Path loss exponent (learnable)
        X_shadow: Shadowing noise ~ N(0, sigma^2)
        L_body: Body shadowing attenuation
    """
    def __init__(self, 
                 init_tx_power: float = -4.0,
                 init_path_loss_exp: float = 2.5,
                 init_sigma_shadow: float = 4.0):
        super().__init__()
        
        # Learnable parameters
        self.tx_power = nn.Parameter(torch.tensor(init_tx_power))
        self.path_loss_exp = nn.Parameter(torch.tensor(init_path_loss_exp))
        self.sigma_shadow = nn.Parameter(torch.tensor(init_sigma_shadow))
        
        # Fixed parameter for body shadowing
        self.body_decay_rate = 0.5
        
    def forward(self, distances: torch.Tensor, 
                add_noise: bool = False) -> torch.Tensor:
        """
        Compute expected RSSI given distances
        
        Args:
            distances: Distance to scanners (batch, num_scanners)
            add_noise: Whether to add stochastic shadowing noise
            
        Returns:
            rssi: Predicted RSSI values (batch, num_scanners)
        """
        # Avoid log(0) by adding small epsilon
        eps = 1e-6
        d_safe = distances + eps
        
        # Log-distance path loss
        path_loss = 20 * torch.log10(d_safe) + self.path_loss_exp * 10 * torch.log10(d_safe)
        
        # Body shadowing (exponential decay)
        body_loss = 10 * torch.exp(-self.body_decay_rate * distances)
        
        # Expected RSSI
        rssi = self.tx_power - path_loss - body_loss
        
        # Add shadowing noise if requested
        if add_noise and self.training:
            noise = torch.randn_like(rssi) * self.sigma_shadow
            rssi = rssi + noise
        
        return rssi
    
    def rssi_to_distance(self, rssi: torch.Tensor, 
                         method: str = 'iterative') -> torch.Tensor:
        """
        Estimate distance from RSSI measurement (inverse problem)
        
        Args:
            rssi: Measured RSSI values (batch, num_scanners)
            method: Estimation method ('iterative' or 'approximate')
            
        Returns:
            distances: Estimated distances (batch, num_scanners)
        """
        if method == 'approximate':
            # Simplified approximation (ignoring body loss)
            numerator = self.tx_power - rssi
            denominator = 20 + self.path_loss_exp * 10
            log_d = numerator / denominator
            distances = 10 ** log_d
            
        elif method == 'iterative':
            # Iterative Newton-Raphson method for more accuracy
            # Initialize with approximate solution
            d = self.rssi_to_distance(rssi, method='approximate')
            
            # Refine with 3 iterations
            for _ in range(3):
                # Compute predicted RSSI
                rssi_pred = self.forward(d, add_noise=False)
                
                # Compute derivative
                eps = 1e-6
                d_safe = d + eps
                grad = -((20 + self.path_loss_exp * 10) / (d_safe * np.log(10)) + 
                        self.body_decay_rate * 10 * torch.exp(-self.body_decay_rate * d))
                
                # Newton update
                residual = rssi_pred - rssi
                d = d - residual / (grad + eps)
                
                # Clamp to reasonable range
                d = torch.clamp(d, min=0.1, max=100.0)
                
            distances = d
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return distances


class RSSIProcessor(nn.Module):
    """
    Complete RSSI processing pipeline for pedestrian localization
    
    Converts raw BLE RSSI measurements to estimated 2D positions using:
        1. Path loss model for distance estimation
        2. Multilateration for position triangulation
        3. Multi-layer perceptron for refinement
    """
    def __init__(self, num_scanners: int = 4, 
                 scanner_positions: Optional[np.ndarray] = None,
                 hidden_dim: int = 64):
        super().__init__()
        
        self.num_scanners = num_scanners
        
        # Scanner positions (if not provided, use default square layout)
        if scanner_positions is None:
            # Default: scanners at corners of 10m x 10m area
            scanner_positions = np.array([
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0]
            ])
        
        self.register_buffer('scanner_positions', 
                           torch.tensor(scanner_positions, dtype=torch.float32))
        
        # Path loss model
        self.path_loss_model = PathLossModel()
        
        # Position refinement network
        self.position_refiner = nn.Sequential(
            nn.Linear(num_scanners + 2, hidden_dim),  # +2 for temporal encoding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # Output: (x, y)
        )
        
    def multilateration(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Estimate position using multilateration
        
        Uses weighted least squares to triangulate position from distances
        to multiple scanners.
        
        Args:
            distances: Distances to scanners (batch, num_scanners)
            
        Returns:
            positions: Estimated 2D positions (batch, 2)
        """
        batch_size = distances.shape[0]
        
        # Use first scanner as reference
        ref_pos = self.scanner_positions[0]  # (2,)
        
        # Compute relative positions
        A_list = []
        b_list = []
        
        for i in range(1, self.num_scanners):
            # Vector from reference to scanner i
            delta_pos = self.scanner_positions[i] - ref_pos  # (2,)
            
            # Distance difference
            d_ref = distances[:, 0]  # (batch,)
            d_i = distances[:, i]    # (batch,)
            
            # Build linear system: 2*delta_pos^T * (p - ref_pos) = d_ref^2 - d_i^2 - ||delta_pos||^2
            A_list.append(2 * delta_pos.unsqueeze(0).expand(batch_size, -1))  # (batch, 2)
            b_entry = d_ref**2 - d_i**2 - torch.sum(delta_pos**2)
            b_list.append(b_entry.unsqueeze(-1))  # (batch, 1)
        
        # Stack into matrices
        A = torch.stack(A_list, dim=1)  # (batch, num_scanners-1, 2)
        b = torch.cat(b_list, dim=-1)    # (batch, num_scanners-1)
        
        # Solve least squares: A^T A x = A^T b
        ATA = torch.bmm(A.transpose(1, 2), A)  # (batch, 2, 2)
        ATb = torch.bmm(A.transpose(1, 2), b.unsqueeze(-1))  # (batch, 2, 1)
        
        # Add regularization for numerical stability
        ATA = ATA + 1e-3 * torch.eye(2, device=ATA.device).unsqueeze(0)
        
        # Solve using Cholesky decomposition
        try:
            pos_rel = torch.linalg.solve(ATA, ATb).squeeze(-1)  # (batch, 2)
        except:
            # Fallback to pseudo-inverse if Cholesky fails
            pos_rel = torch.bmm(torch.linalg.pinv(ATA), ATb).squeeze(-1)
        
        # Convert to absolute position
        positions = pos_rel + ref_pos.unsqueeze(0)
        
        return positions
        
    def forward(self, rssi: torch.Tensor, 
                temporal_encoding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process RSSI to position estimate
        
        Args:
            rssi: RSSI measurements (batch, num_scanners)
            temporal_encoding: Optional temporal features (batch, 2)
            
        Returns:
            position: Estimated position (batch, 2)
            distances: Estimated distances to scanners (batch, num_scanners)
        """
        # Estimate distances from RSSI
        distances = self.path_loss_model.rssi_to_distance(rssi, method='iterative')
        
        # Initial position from multilateration
        pos_init = self.multilateration(distances)
        
        # Refine with neural network
        if temporal_encoding is not None:
            refiner_input = torch.cat([rssi, temporal_encoding], dim=-1)
        else:
            # Use zero temporal encoding if not provided
            temporal_encoding = torch.zeros(rssi.shape[0], 2, device=rssi.device)
            refiner_input = torch.cat([rssi, temporal_encoding], dim=-1)
        
        pos_offset = self.position_refiner(refiner_input)
        position = pos_init + pos_offset
        
        return position, distances

