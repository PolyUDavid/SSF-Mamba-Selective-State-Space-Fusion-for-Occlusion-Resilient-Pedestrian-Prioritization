"""
Dynamic Graph Construction for Pedestrian Social Interactions

Builds spatial graphs based on pedestrian proximity and social relationship features.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk  
Date: November 5, 2025 (Revision)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class DynamicGraphBuilder:
    """
    Constructs dynamic spatial graphs for pedestrian interactions
    
    Edges are created between pedestrians within a spatial threshold,
    with edge features encoding their social relationship:
        - Velocity alignment (cosine similarity)
        - Proximity (inverse distance)
        - Group membership score
    """
    def __init__(self, spatial_threshold: float = 5.0):
        """
        Args:
            spatial_threshold: Maximum distance for edge creation (meters)
        """
        self.spatial_threshold = spatial_threshold
        
    def build_graph(self, 
                    positions: torch.Tensor,
                    velocities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build spatial graph for current timestep
        
        Args:
            positions: Pedestrian positions (num_peds, 2)
            velocities: Pedestrian velocities (num_peds, 2)
            
        Returns:
            edge_index: Edge connectivity (2, num_edges)
            edge_features: Edge relationship features (num_edges, 3)
        """
        num_peds = positions.shape[0]
        
        # Compute pairwise distances
        pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (num_peds, num_peds, 2)
        distances = torch.norm(pos_diff, dim=-1)  # (num_peds, num_peds)
        
        # Create edges for pedestrians within threshold
        edge_mask = (distances < self.spatial_threshold) & (distances > 0)  # Exclude self-loops
        edge_index = edge_mask.nonzero(as_tuple=False).t()  # (2, num_edges)
        
        if edge_index.shape[1] == 0:
            # No edges, return empty
            return edge_index, torch.zeros(0, 3, device=positions.device)
        
        # Extract edge pairs
        src_idx = edge_index[0]  # Source nodes
        dst_idx = edge_index[1]  # Destination nodes
        
        # Compute edge features
        edge_features = self._compute_edge_features(
            positions, velocities, distances, src_idx, dst_idx
        )
        
        return edge_index, edge_features
    
    def _compute_edge_features(self,
                               positions: torch.Tensor,
                               velocities: torch.Tensor,
                               distances: torch.Tensor,
                               src_idx: torch.Tensor,
                               dst_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute social relationship features for edges
        
        Args:
            positions: All pedestrian positions (num_peds, 2)
            velocities: All pedestrian velocities (num_peds, 2)
            distances: Pairwise distance matrix (num_peds, num_peds)
            src_idx: Source node indices for edges (num_edges,)
            dst_idx: Destination node indices for edges (num_edges,)
            
        Returns:
            edge_features: (num_edges, 3) containing [vel_align, proximity, group_score]
        """
        num_edges = src_idx.shape[0]
        
        # 1. Velocity alignment (cosine similarity)
        vel_src = velocities[src_idx]  # (num_edges, 2)
        vel_dst = velocities[dst_idx]  # (num_edges, 2)
        
        # Normalize velocities
        vel_src_norm = vel_src / (torch.norm(vel_src, dim=-1, keepdim=True) + 1e-6)
        vel_dst_norm = vel_dst / (torch.norm(vel_dst, dim=-1, keepdim=True) + 1e-6)
        
        # Cosine similarity
        vel_alignment = torch.sum(vel_src_norm * vel_dst_norm, dim=-1)  # (num_edges,)
        vel_alignment = (vel_alignment + 1) / 2  # Normalize to [0, 1]
        
        # 2. Proximity (inverse distance, normalized)
        edge_distances = distances[src_idx, dst_idx]  # (num_edges,)
        proximity = 1.0 / (edge_distances + 1.0)  # Inverse distance with offset
        proximity = proximity / (1.0 / 1.0)  # Normalize so proximity=1 at distance=0
        
        # 3. Group membership score (based on both position and velocity)
        # High score if pedestrians are close AND moving in similar direction
        pos_similarity = torch.exp(-edge_distances / self.spatial_threshold)  # (num_edges,)
        vel_similarity = vel_alignment  # Already computed
        
        group_score = pos_similarity * vel_similarity  # Combined score
        
        # Stack features
        edge_features = torch.stack([vel_alignment, proximity, group_score], dim=-1)
        
        return edge_features
    
    def build_batch_graph(self,
                         positions: torch.Tensor,
                         velocities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build graphs for a batch of timesteps
        
        Args:
            positions: (batch_size, num_peds, 2)
            velocities: (batch_size, num_peds, 2)
            
        Returns:
            edge_index_list: List of edge indices for each sample
            edge_features_list: List of edge features for each sample
            batch_idx: Batch indices for each edge
        """
        batch_size = positions.shape[0]
        
        edge_index_list = []
        edge_features_list = []
        batch_idx_list = []
        
        for b in range(batch_size):
            edge_index, edge_features = self.build_graph(
                positions[b], velocities[b]
            )
            
            if edge_index.shape[1] > 0:
                edge_index_list.append(edge_index)
                edge_features_list.append(edge_features)
                batch_idx_list.append(torch.full((edge_index.shape[1],), b, 
                                                 dtype=torch.long, device=positions.device))
        
        if len(edge_index_list) == 0:
            # No edges in entire batch
            return (torch.zeros(2, 0, dtype=torch.long, device=positions.device),
                   torch.zeros(0, 3, device=positions.device),
                   torch.zeros(0, dtype=torch.long, device=positions.device))
        
        # Concatenate all edges
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_features = torch.cat(edge_features_list, dim=0)
        batch_idx = torch.cat(batch_idx_list, dim=0)
        
        return edge_index, edge_features, batch_idx
    
    def visualize_graph(self, positions: np.ndarray, edge_index: np.ndarray) -> None:
        """
        Visualize spatial graph (for debugging)
        
        Args:
            positions: Pedestrian positions (num_peds, 2)
            edge_index: Edge connectivity (2, num_edges)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for visualization")
            return
        
        plt.figure(figsize=(8, 8))
        
        # Plot nodes
        plt.scatter(positions[:, 0], positions[:, 1], c='blue', s=100, zorder=2)
        
        # Plot edges
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            plt.plot([positions[src, 0], positions[dst, 0]],
                    [positions[src, 1], positions[dst, 1]],
                    'k-', alpha=0.3, zorder=1)
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Pedestrian Social Graph')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.show()

