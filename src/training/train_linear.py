"""
Training script for Linear Regression baseline model
"""
from sklearn.metrics import mean_squared_error
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import random
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.linear import LinearBaselineModel
from src.config import TRAINING_CONFIG


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_trajectory_data():
    """
    Load and prepare trajectory sequence data for linear model
    """
    # Load selected features
    df = pd.read_csv('data/features/features_selected.csv')
    
    # Get feature columns
    feature_cols = [col for col in df.columns 
                   if col not in ['X', 'Y', 'trajectory_id', 'step_id']]
    
    print(f"Using features: {feature_cols}")
    
    # Prepare training trajectories (0-15)
    X_train_trajectories = []
    Y_train_trajectories = []
    
    for traj_id in range(16):
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            # Input: feature sequence for this trajectory
            X_train_trajectories.append(traj_data[feature_cols].values)
            # Output: position sequence for this trajectory  
            Y_train_trajectories.append(traj_data[['X', 'Y']].values)
    
    # Prepare validation trajectories (16-19)
    X_val_trajectories = []
    Y_val_trajectories = []
    
    for traj_id in range(16, 20):
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            # Input: feature sequence for this trajectory
            X_val_trajectories.append(traj_data[feature_cols].values)
            # Output: position sequence for this trajectory
            Y_val_trajectories.append(traj_data[['X', 'Y']].values)
    
    X_train_trajectories = np.array(X_train_trajectories)  # (16, 10, features)
    Y_train_trajectories = np.array(Y_train_trajectories)  # (16, 10, 2)
    X_val_trajectories = np.array(X_val_trajectories)      # (4, 10, features)  
    Y_val_trajectories = np.array(Y_val_trajectories)      # (4, 10, 2)
    
    print(f"Training trajectories: {X_train_trajectories.shape}")
    print(f"Training positions: {Y_train_trajectories.shape}")
    print(f"Validation trajectories: {X_val_trajectories.shape}")
    print(f"Validation positions: {Y_val_trajectories.shape}")
    print("Task: Fit parameters to minimize average error across all 16 training trajectories")
    
    return (X_train_trajectories, Y_train_trajectories, X_val_trajectories, Y_val_trajectories)


def evaluate_trajectory_predictions(true_trajectories, pred_trajectories):
    """
    Evaluate trajectory-level predictions
    """
    trajectory_errors = []
    
    for traj_idx in range(len(true_trajectories)):
        true_traj = true_trajectories[traj_idx]
        pred_traj = pred_trajectories[traj_idx]
        
        # Calculate RMSE for this trajectory
        rmse_x = np.sqrt(mean_squared_error(true_traj[:, 0], pred_traj[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(true_traj[:, 1], pred_traj[:, 1]))
        rmse_combined = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
        
        trajectory_errors.append({
            'trajectory_id': 16 + traj_idx,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y, 
            'rmse_combined': rmse_combined
        })
    
    return trajectory_errors


def train_model():
    """Train trajectory-level linear baseline model"""
    # Set random seed for reproducibility
    set_seed(TRAINING_CONFIG['random_seed'])
    
    # Prepare trajectory data
    (X_train_trajectories, Y_train_trajectories, X_val_trajectories, Y_val_trajectories) = prepare_trajectory_data()
    
    # Initialize model
    model = LinearBaselineModel()
    
    print("\nTraining Trajectory-Level Linear Regression model...")
    print("Objective: Minimize average error across all 16 training trajectories")
    
    # Fit the model on trajectory sequences
    model.fit(X_train_trajectories, Y_train_trajectories)
    
    print("Training complete!")
    
    # Save the model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'linear_baseline_model.pkl'
    
    # Save using joblib (better for scikit-learn models)
    joblib.dump({
        'model': model,
        'feature_count': X_train_trajectories.shape[-1],
        'training_config': TRAINING_CONFIG
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Evaluate on validation trajectories
    print("\nValidating on 4 trajectories...")
    val_predictions = model.predict_trajectories(X_val_trajectories)
    
    # Calculate trajectory-level metrics
    trajectory_errors = evaluate_trajectory_predictions(Y_val_trajectories, val_predictions)
    
    print("\nTrajectory-Level Validation Results:")
    print("="*60)
    
    total_rmse_x = 0
    total_rmse_y = 0
    
    for error_info in trajectory_errors:
        traj_id = error_info['trajectory_id']
        rmse_x = error_info['rmse_x']
        rmse_y = error_info['rmse_y']
        rmse_combined = error_info['rmse_combined']
        
        total_rmse_x += rmse_x
        total_rmse_y += rmse_y
        
        print(f"Trajectory {traj_id}: X-RMSE: {rmse_x:.2f}, Y-RMSE: {rmse_y:.2f}, Combined: {rmse_combined:.2f}")
        
        # Show sample predictions for first trajectory
        if traj_id == 16:
            print(f"  Sample predictions (Trajectory {traj_id}):")
            for step in range(min(5, 10)):
                true_pos = Y_val_trajectories[0][step]
                pred_pos = val_predictions[0][step]
                print(f"    Step {step+1}: True=({true_pos[0]}, {true_pos[1]}) "
                      f"Pred=({pred_pos[0]}, {pred_pos[1]})")
    
    avg_rmse_x = total_rmse_x / len(trajectory_errors)
    avg_rmse_y = total_rmse_y / len(trajectory_errors)
    avg_rmse_combined = np.sqrt((avg_rmse_x**2 + avg_rmse_y**2) / 2)
    
    print(f"\nOverall Average:")
    print(f"X-coordinate RMSE: {avg_rmse_x:.2f}")
    print(f"Y-coordinate RMSE: {avg_rmse_y:.2f}")
    print(f"Combined RMSE: {avg_rmse_combined:.2f}")


def load_and_evaluate():
    """Load saved model and evaluate"""
    model_path = Path('results/models/linear_baseline_model.pkl')
    
    if not model_path.exists():
        print(f"Model not found at {model_path}. Train the model first.")
        return
    
    # Load model
    checkpoint = joblib.load(model_path)
    model = checkpoint['model']
    
    print(f"Model loaded from: {model_path}")
    
    # Load test data
    df = pd.read_csv('data/features/features_selected.csv')
    feature_cols = [col for col in df.columns 
                   if col not in ['X', 'Y', 'trajectory_id', 'step_id']]
        
        


if __name__ == "__main__":
    train_model()
    print("\nEvaluating saved model...")
    load_and_evaluate()
