"""
Training script for RNN model
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error
import random
import os
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.rnn import RNNModel
from src.config import TRAINING_CONFIG, MODEL_CONFIG


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_sequence_data():
    """Prepare sequence-to-sequence data maintaining trajectory structure"""
    df = pd.read_csv('data/features/features_selected.csv')
    feature_cols = [col for col in df.columns 
                   if col not in ['X', 'Y', 'trajectory_id', 'step_id']]
    
    
    
    # Prepare training trajectories (0-15)
    X_train_sequences = []
    Y_train_sequences = []
    
    for traj_id in range(16):
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            # Input: feature sequence for this trajectory
            X_train_sequences.append(traj_data[feature_cols].values)
            # Output: position sequence for this trajectory
            Y_train_sequences.append(traj_data[['X', 'Y']].values)
    
    # Prepare validation trajectories (16-19)
    X_val_sequences = []
    Y_val_sequences = []
    
    for traj_id in range(16, 20):
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            # Input: feature sequence for this trajectory
            X_val_sequences.append(traj_data[feature_cols].values)
            # Output: position sequence for this trajectory
            Y_val_sequences.append(traj_data[['X', 'Y']].values)
    
    X_train = np.array(X_train_sequences)  # (16, 10, features)
    Y_train = np.array(Y_train_sequences)  # (16, 10, 2)
    X_val = np.array(X_val_sequences)      # (4, 10, features)
    Y_val = np.array(Y_val_sequences)      # (4, 10, 2)
    
    print(f"Training shape: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Validation shape: X={X_val.shape}, Y={Y_val.shape}")
    print("Task: Feature sequence → Position sequence (maintaining trajectory structure)")
    
    return X_train, Y_train, X_val, Y_val


def train_model():
    """Train and evaluate RNN model"""
    # Set random seed
    set_seed(TRAINING_CONFIG['random_seed'])
    
    # Prepare sequence data
    X_train, Y_train, X_val, Y_val = prepare_sequence_data()
    
    # Flatten for the current RNN model implementation
    # Note: The model will reshape internally, but we need to maintain trajectory order
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    Y_train_flat = Y_train.reshape(-1, 2)
    
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    Y_val_flat = Y_val.reshape(-1, 2)
    
    # Train model
    print(f"\nTraining RNN model with hidden_dim={MODEL_CONFIG['hidden_dim']}, num_layers={MODEL_CONFIG['num_layers']}, dropout={MODEL_CONFIG['dropout']}...")
    print("Objective: Minimize average error across all 16 training trajectories")
    
    model = RNNModel(
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout'],
        learning_rate=TRAINING_CONFIG['learning_rate'],
        epochs=TRAINING_CONFIG['epochs']
    )
    
    # The model's fit method will handle the reshaping correctly
    model.fit(X_train_flat, Y_train_flat)
    
    print("Training complete!")
    
    # Save model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'rnn_model.pkl'
    
    # Save the entire model object
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'model_config': {
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'dropout': model.dropout,
            'input_size': X_train.shape[-1]
        },
        'scaler_features': model.scaler_features,
        'scaler_targets': model.scaler_targets,
        'training_config': TRAINING_CONFIG,
        'model_config': MODEL_CONFIG
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Evaluate on validation trajectories
    print("\nValidating on 4 trajectories (sequence-to-sequence)...")
    
    # Predict on flattened validation data
    y_pred_flat = model.predict(X_val_flat)
    
    # Reshape predictions back to trajectory format
    y_pred = y_pred_flat.reshape(X_val.shape[0], X_val.shape[1], 2)
    
    # Calculate trajectory-level metrics
    print("\nTrajectory-Level Validation Results:")
    print("="*60)
    
    total_rmse_x = 0
    total_rmse_y = 0
    
    for i in range(4):
        traj_id = 16 + i
        
        # RMSE for this trajectory
        rmse_x = np.sqrt(mean_squared_error(Y_val[i, :, 0], y_pred[i, :, 0]))
        rmse_y = np.sqrt(mean_squared_error(Y_val[i, :, 1], y_pred[i, :, 1]))
        rmse_combined = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
        
        total_rmse_x += rmse_x
        total_rmse_y += rmse_y
        
        print(f"Trajectory {traj_id}: X-RMSE: {rmse_x:.2f}, Y-RMSE: {rmse_y:.2f}, Combined: {rmse_combined:.2f}")
    
    # Overall metrics
    avg_rmse_x = total_rmse_x / 4
    avg_rmse_y = total_rmse_y / 4
    avg_rmse_combined = np.sqrt((avg_rmse_x**2 + avg_rmse_y**2) / 2)
    
    print(f"\nOverall Average:")
    print(f"X-coordinate RMSE: {avg_rmse_x:.2f}")
    print(f"Y-coordinate RMSE: {avg_rmse_y:.2f}")
    print(f"Combined RMSE: {avg_rmse_combined:.2f}")


if __name__ == "__main__":
    train_model()