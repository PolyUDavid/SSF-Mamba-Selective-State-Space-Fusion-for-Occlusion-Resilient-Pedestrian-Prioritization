"""
Kalman Filter Baseline for Trajectory Prediction

Classical physics-based approach using constant velocity model.
Serves as a non-learning baseline for comparison.

Author: Nok KO
Contact: Nok-david.ko@connect.polyu.hk
Date: November 5, 2025 (Revision)
"""

import numpy as np
from typing import Tuple


class KalmanFilterPredictor:
    """
    Kalman Filter for pedestrian trajectory prediction
    
    Uses a constant velocity motion model with Gaussian process noise.
    This serves as a physics-based baseline without any learning component.
    
    State vector: [x, y, vx, vy]
    Measurement: [x, y]
    """
    def __init__(self, 
                 process_noise: float = 0.1,
                 measurement_noise: float = 0.5,
                 dt: float = 0.1):
        """
        Args:
            process_noise: Process noise standard deviation
            measurement_noise: Measurement noise standard deviation
            dt: Time step (seconds)
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise**2
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise**2
        
    def initialize_state(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize state from first few observations
        
        Args:
            observations: Position observations (T, 2)
            
        Returns:
            state: Initial state estimate (4,)
            covariance: Initial covariance (4, 4)
        """
        # Use first two observations to estimate velocity
        pos = observations[0]  # (2,)
        if len(observations) > 1:
            vel = (observations[1] - observations[0]) / self.dt
        else:
            vel = np.zeros(2)
        
        state = np.concatenate([pos, vel])  # (4,)
        covariance = np.eye(4) * 1.0
        
        return state, covariance
    
    def predict(self, state: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step
        
        Args:
            state: Current state (4,)
            covariance: Current covariance (4, 4)
            
        Returns:
            predicted_state: Predicted state (4,)
            predicted_covariance: Predicted covariance (4, 4)
        """
        predicted_state = self.F @ state
        predicted_covariance = self.F @ covariance @ self.F.T + self.Q
        
        return predicted_state, predicted_covariance
    
    def update(self, 
               predicted_state: np.ndarray,
               predicted_covariance: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step (correction)
        
        Args:
            predicted_state: Predicted state (4,)
            predicted_covariance: Predicted covariance (4, 4)
            measurement: New measurement (2,)
            
        Returns:
            updated_state: Updated state (4,)
            updated_covariance: Updated covariance (4, 4)
        """
        # Innovation (measurement residual)
        innovation = measurement - self.H @ predicted_state
        
        # Innovation covariance
        S = self.H @ predicted_covariance @ self.H.T + self.R
        
        # Kalman gain
        K = predicted_covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        updated_state = predicted_state + K @ innovation
        
        # Update covariance
        updated_covariance = (np.eye(4) - K @ self.H) @ predicted_covariance
        
        return updated_state, updated_covariance
    
    def predict_trajectory(self, 
                          observations: np.ndarray,
                          pred_len: int) -> np.ndarray:
        """
        Predict future trajectory from observations
        
        Args:
            observations: Historical observations (obs_len, 2)
            pred_len: Number of future steps to predict
            
        Returns:
            predictions: Predicted positions (pred_len, 2)
        """
        # Initialize from observations
        state, covariance = self.initialize_state(observations)
        
        # Filter through all observations
        for i in range(len(observations)):
            # Predict
            state, covariance = self.predict(state, covariance)
            
            # Update with measurement
            state, covariance = self.update(state, covariance, observations[i])
        
        # Predict future trajectory
        predictions = []
        for _ in range(pred_len):
            # Predict next state
            state, covariance = self.predict(state, covariance)
            
            # Extract position
            position = state[:2]
            predictions.append(position)
        
        return np.array(predictions)  # (pred_len, 2)
    
    def predict_batch(self,
                     observations_batch: np.ndarray,
                     pred_len: int) -> np.ndarray:
        """
        Batch prediction for multiple pedestrians
        
        Args:
            observations_batch: Historical observations (num_peds, obs_len, 2)
            pred_len: Number of future steps
            
        Returns:
            predictions_batch: Predicted positions (num_peds, pred_len, 2)
        """
        num_peds = observations_batch.shape[0]
        predictions_batch = []
        
        for i in range(num_peds):
            pred = self.predict_trajectory(observations_batch[i], pred_len)
            predictions_batch.append(pred)
        
        return np.array(predictions_batch)

