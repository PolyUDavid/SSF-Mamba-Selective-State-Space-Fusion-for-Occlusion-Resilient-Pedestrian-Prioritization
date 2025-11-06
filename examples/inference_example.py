"""
Inference Example for P-SAFE Framework

Demonstrates how to use the trained models for inference.
This is a simplified example to understand the workflow.

Author: David KO
Date: November 2025
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import models
from models.stgcn_ble import create_stgcn_ble_model
from models.yolov8_vit import create_yolov8_vit_model
from models.mamba2_fusion import create_mamba2_fusion_model
from models.sdt_tsc import create_sdt_tsc_model


def inference_example():
    """
    Complete inference pipeline example
    
    This demonstrates the data flow through all four modules:
    1. STGCN-BLE: Process BLE signals to predict trajectories
    2. YOLOv8-ViT: Process camera feeds for perception
    3. Mamba-2 Fusion: Fuse BLE and vision features
    4. SDT-TSC: Generate traffic signal control actions
    """
    print("="*80)
    print("P-SAFE Inference Example")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # ========== Step 1: Initialize Models ==========
    print("\n[Step 1] Initializing models...")
    
    # Trajectory prediction model
    stgcn_model = create_stgcn_ble_model(
        num_scanners=4,
        hidden_dim=128,
        pred_len=30
    ).to(device)
    stgcn_model.eval()
    print("  ✓ STGCN-BLE model loaded")
    
    # Visual perception model
    yolov8_vit_model = create_yolov8_vit_model(
        num_age_classes=3
    ).to(device)
    yolov8_vit_model.eval()
    print("  ✓ YOLOv8-ViT model loaded")
    
    # Cross-modal fusion model
    fusion_model = create_mamba2_fusion_model(
        ble_dim=128,
        vis_dim=256,
        hidden_dim=256
    ).to(device)
    fusion_model.eval()
    print("  ✓ Mamba-2 Fusion model loaded")
    
    # Decision-making planner
    planner_model = create_sdt_tsc_model(
        state_dim=512,
        action_vocab_size=215
    ).to(device)
    planner_model.eval()
    print("  ✓ SDT-TSC Planner loaded")
    
    # ========== Step 2: Prepare Mock Input Data ==========
    print("\n[Step 2] Preparing input data...")
    
    # Mock BLE RSSI data (3 pedestrians, 20 timesteps, 4 scanners)
    num_pedestrians = 3
    obs_len = 20
    num_scanners = 4
    
    rssi_data = torch.randn(num_pedestrians, obs_len, num_scanners).to(device)
    print(f"  ✓ BLE RSSI data: {rssi_data.shape}")
    
    # Mock video data (1 pedestrian detected, 16 frames, 3 channels, 224x224)
    video_tube = torch.randn(1, 16, 3, 224, 224).to(device)
    print(f"  ✓ Video data: {video_tube.shape}")
    
    # Mock vehicle state (queue length, occupancy, wait time)
    vehicle_state = torch.randn(1, 128).to(device)
    print(f"  ✓ Vehicle state: {vehicle_state.shape}")
    
    # ========== Step 3: STGCN-BLE Trajectory Prediction ==========
    print("\n[Step 3] Running STGCN-BLE for trajectory prediction...")
    
    with torch.no_grad():
        # Process BLE signals
        trajectory_features = stgcn_model(rssi_data)
        # Output: (num_pedestrians, feature_dim)
        
    print(f"  ✓ Trajectory features: {trajectory_features.shape}")
    print(f"    Feature dim: {trajectory_features.shape[-1]}")
    
    # ========== Step 4: YOLOv8-ViT Visual Perception ==========
    print("\n[Step 4] Running YOLOv8-ViT for visual perception...")
    
    with torch.no_grad():
        # Process video tube
        perception_output = yolov8_vit_model(video_tube)
        # Output: dict with 'detection', 'age', 'intent', 'features'
        
    visual_features = perception_output['features']  # (batch, feature_dim)
    age_pred = torch.argmax(perception_output['age'], dim=1)
    intent_prob = torch.softmax(perception_output['intent'], dim=1)[:, 1]  # P(cross)
    
    print(f"  ✓ Visual features: {visual_features.shape}")
    print(f"  ✓ Age prediction: {age_pred.item()} (0=child, 1=adult, 2=elderly)")
    print(f"  ✓ Crossing intent probability: {intent_prob.item():.3f}")
    
    # ========== Step 5: Mamba-2 Cross-Modal Fusion ==========
    print("\n[Step 5] Running Mamba-2 fusion...")
    
    # For this example, use the first pedestrian's trajectory features
    ble_features = trajectory_features[0:1]  # (1, 128)
    
    # Expand to sequence for fusion model
    ble_sequence = ble_features.unsqueeze(1).repeat(1, 32, 1)  # (1, 32, 128)
    vis_sequence = visual_features.unsqueeze(1).repeat(1, 32, 1)  # (1, 32, 256)
    
    with torch.no_grad():
        fused_features = fusion_model(ble_sequence, vis_sequence)
        # Output: (1, 256)
        
    print(f"  ✓ Fused features: {fused_features.shape}")
    
    # ========== Step 6: SDT-TSC Decision Making ==========
    print("\n[Step 6] Running SDT-TSC planner for signal control...")
    
    # Combine all state information
    full_state = torch.cat([
        fused_features,  # Pedestrian state (256)
        vehicle_state    # Vehicle state (128)
    ], dim=1)  # (1, 384)
    
    # Pad to expected state_dim (512)
    padding = torch.zeros(1, 512 - full_state.shape[1]).to(device)
    full_state = torch.cat([full_state, padding], dim=1)
    
    with torch.no_grad():
        # Generate action tokens
        action_tokens = planner_model.generate_action(
            full_state,
            max_length=5
        )
        
    print(f"  ✓ Generated action tokens: {action_tokens.tolist()}")
    
    # Decode action tokens (mock decoder)
    action_names = ['maintain_green', 'extend_pedestrian', 'switch_phase', 'idle']
    print(f"  ✓ Decoded actions: {[action_names[min(t, 3)] for t in action_tokens[0].tolist()]}")
    
    # ========== Summary ==========
    print("\n" + "="*80)
    print("Inference Summary")
    print("="*80)
    print(f"Input: {num_pedestrians} pedestrians with BLE + 1 detected in camera")
    print(f"Trajectory prediction: {trajectory_features.shape[0]} feature vectors")
    print(f"Visual perception: Age={age_pred.item()}, Intent={intent_prob.item():.3f}")
    print(f"Fusion: Robust {fused_features.shape[1]}-dim representation")
    print(f"Decision: {len(action_tokens[0])} action tokens generated")
    print("="*80)
    
    return {
        'trajectory_features': trajectory_features,
        'visual_features': visual_features,
        'fused_features': fused_features,
        'action_tokens': action_tokens,
        'age_prediction': age_pred,
        'intent_probability': intent_prob
    }


def load_pretrained_weights_example():
    """
    Example of how to load pre-trained weights (if available)
    
    Note: Pre-trained weights are not included in this submission
    to protect research assets. This example shows the expected format.
    """
    print("\n" + "="*80)
    print("Loading Pre-trained Weights (Example)")
    print("="*80)
    
    # Example paths (these files don't exist in this submission)
    checkpoint_paths = {
        'stgcn_ble': 'checkpoints/stgcn_ble_best.pth',
        'yolov8_vit': 'checkpoints/yolov8_vit_best.pth',
        'fusion': 'checkpoints/mamba2_fusion_best.pth',
        'planner': 'checkpoints/sdt_tsc_best.pth'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    models = {
        'stgcn_ble': create_stgcn_ble_model(),
        'yolov8_vit': create_yolov8_vit_model(),
        'fusion': create_mamba2_fusion_model(),
        'planner': create_sdt_tsc_model()
    }
    
    # Load weights
    for model_name, model in models.items():
        checkpoint_path = checkpoint_paths[model_name]
        print(f"\n  Loading {model_name}...")
        print(f"  Expected path: {checkpoint_path}")
        print(f"  Status: ⚠️  Checkpoint not included in public submission")
        
        # If checkpoint exists (for private testing):
        # checkpoint = torch.load(checkpoint_path, map_location=device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # print(f"  ✓ Loaded (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
    
    print("\n  Note: Pre-trained weights excluded from public submission")
    print("  Models are initialized with random weights for architecture review")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("P-SAFE Inference Example")
    print("Author: David KO")
    print("Date: November 2025")
    print("="*80)
    
    # Run inference example with random initialization
    print("\n[INFO] Running inference with randomly initialized weights")
    print("[INFO] This demonstrates the architecture and data flow\n")
    
    results = inference_example()
    
    # Show weight loading example
    load_pretrained_weights_example()
    
    print("\n✅ Inference example completed successfully!")
    print("\nFor training, please refer to the training scripts (not included in submission)")
    print("For questions, please open a GitHub issue\n")

