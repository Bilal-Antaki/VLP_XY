"""
Training script for Linear Regression baseline model
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error
import random
import os
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.linear import LinearBaselineModel
from src.config import TRAINING_CONFIG


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_data():
    """
    Load and prepare data for linear model
    """
    # Load selected features
    df = pd.read_csv('data/features/features_selected.csv')
    
    # Get feature columns
    feature_cols = [col for col in df.columns 
                   if col not in ['X', 'Y', 'trajectory_id', 'step_id']]
    
    print(f"Using features: {feature_cols}")
    
    # Split by trajectory IDs
    train_traj_ids = list(range(16))
    val_traj_ids = list(range(16, 20))
    
    # Prepare training data
    train_df = df[df['trajectory_id'].isin(train_traj_ids)]
    X_train = train_df[feature_cols].values
    Y_train = train_df[['X', 'Y']].values
    
    # Prepare validation data
    val_df = df[df['trajectory_id'].isin(val_traj_ids)]
    X_val = val_df[feature_cols].values
    Y_val = val_df[['X', 'Y']].values
    
    # Get trajectory structure for validation
    val_trajectories = []
    for traj_id in val_traj_ids:
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            val_trajectories.append({
                'X': traj_data[feature_cols].values,
                'Y': traj_data[['X', 'Y']].values,
                'id': traj_id
            })
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    return (X_train, Y_train, X_val, Y_val), val_trajectories


def evaluate_trajectories(model, trajectories):
    """
    Evaluate model on trajectory level (not just individual points)
    """
    rmse_x_list = []
    rmse_y_list = []
    
    for traj in trajectories:
        # Predict
        predictions = model.predict(traj['X'])
        
        # Calculate RMSE for this trajectory
        rmse_x = np.sqrt(mean_squared_error(traj['Y'][:, 0], predictions[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(traj['Y'][:, 1], predictions[:, 1]))
        
        rmse_x_list.append(rmse_x)
        rmse_y_list.append(rmse_y)
    
    return np.array(rmse_x_list), np.array(rmse_y_list)


def train_model():
    """Train linear baseline model"""
    # Set random seed for reproducibility
    set_seed(TRAINING_CONFIG['random_seed'])
    
    # Prepare data
    (X_train, Y_train, X_val, Y_val), val_trajectories = prepare_data()
    
    # Initialize model
    model = LinearBaselineModel()
    
    print("\nTraining Linear Regression baseline model...")
    
    # Fit the model
    model.fit(X_train, Y_train)
    
    print("Training complete!")
    
    # Save the model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'linear_baseline_model.pkl'
    
    # Save using joblib (better for scikit-learn models)
    joblib.dump({
        'model': model,
        'feature_count': X_train.shape[1],
        'training_config': TRAINING_CONFIG
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Evaluate on validation set (point-wise)
    val_pred = model.predict(X_val)
    
    # Point-wise metrics
    rmse_x_points = np.sqrt(mean_squared_error(Y_val[:, 0], val_pred[:, 0]))
    rmse_y_points = np.sqrt(mean_squared_error(Y_val[:, 1], val_pred[:, 1]))
    
    print(f"\nPoint-wise Validation Metrics:")
    print(f"RMSE X: {rmse_x_points:.2f}")
    print(f"RMSE Y: {rmse_y_points:.2f}")
    
    # Trajectory-level evaluation
    rmse_x_trajs, rmse_y_trajs = evaluate_trajectories(model, val_trajectories)
    
    print(f"\nTrajectory-level Validation Metrics:")
    print(f"X-coordinate:")
    print(f"  RMSE: {rmse_x_trajs.mean():.2f}")
    print(f"  Std: {rmse_x_trajs.std():.2f}")
    
    print(f"Y-coordinate:")
    print(f"  RMSE: {rmse_y_trajs.mean():.2f}")
    print(f"  Std: {rmse_y_trajs.std():.2f}")
    
    # Combined metric
    rmse_combined = np.sqrt((rmse_x_trajs**2 + rmse_y_trajs**2) / 2)
    print(f"\nCombined RMSE: {rmse_combined.mean():.2f} ± {rmse_combined.std():.2f}")
    
    # Print sample predictions
    if val_trajectories:
        sample_traj = val_trajectories[0]
        sample_pred = model.predict(sample_traj['X'])
        
        print(f"\nSample predictions for trajectory {sample_traj['id']}:")
        print("Step | True X | Pred X | True Y | Pred Y")
        print("-" * 45)
        for i in range(min(5, len(sample_pred))):  # Show first 5 steps
            print(f"{i+1:4d} | {sample_traj['Y'][i, 0]:6.0f} | {sample_pred[i, 0]:6.0f} | "
                  f"{sample_traj['Y'][i, 1]:6.0f} | {sample_pred[i, 1]:6.0f}")


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
    
    # Test on a specific trajectory
    test_traj_id = 19
    test_data = df[df['trajectory_id'] == test_traj_id].sort_values('step_id')
    
    if len(test_data) == 10:
        X_test = test_data[feature_cols].values
        Y_test = test_data[['X', 'Y']].values
        
        # Predict
        predictions = model.predict(X_test)
        
        print(f"\nPredictions for trajectory {test_traj_id}:")
        print("Step | True X | Pred X | True Y | Pred Y | Error X | Error Y")
        print("-" * 70)
        for i in range(len(predictions)):
            error_x = abs(Y_test[i, 0] - predictions[i, 0])
            error_y = abs(Y_test[i, 1] - predictions[i, 1])
            print(f"{i+1:4d} | {Y_test[i, 0]:6.0f} | {predictions[i, 0]:6.0f} | "
                  f"{Y_test[i, 1]:6.0f} | {predictions[i, 1]:6.0f} | "
                  f"{error_x:7.1f} | {error_y:7.1f}")
        
        # Overall metrics
        mae_x = np.mean(np.abs(Y_test[:, 0] - predictions[:, 0]))
        mae_y = np.mean(np.abs(Y_test[:, 1] - predictions[:, 1]))
        rmse_x = np.sqrt(mean_squared_error(Y_test[:, 0], predictions[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(Y_test[:, 1], predictions[:, 1]))
        
        print(f"\nMetrics for trajectory {test_traj_id}:")
        print(f"MAE  - X: {mae_x:.2f}, Y: {mae_y:.2f}")
        print(f"RMSE - X: {rmse_x:.2f}, Y: {rmse_y:.2f}")


if __name__ == "__main__":
    train_model()
