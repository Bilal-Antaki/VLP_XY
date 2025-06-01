"""
Training script for Multi-Layer Perceptron model
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error
import random
import os
import torch
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.mlp import MLPModel
from src.config import TRAINING_CONFIG, MLP_CONFIG


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model():
    """Train and evaluate MLP model"""
    # Set random seed
    set_seed(TRAINING_CONFIG['random_seed'])
    
    # Load data
    df = pd.read_csv('data/features/features_selected.csv')
    
    # Get feature columns
    feature_cols = [col for col in df.columns 
                   if col not in ['X', 'Y', 'trajectory_id', 'step_id']]
    
    # Split data
    train_df = df[df['trajectory_id'] < 16]
    val_df = df[df['trajectory_id'] >= 16]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[['X', 'Y']].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df[['X', 'Y']].values
    
    # Train model
    print(f"Training MLP model with architecture {MLP_CONFIG['hidden_sizes']}...")
    model = MLPModel(
        hidden_sizes=MLP_CONFIG['hidden_sizes'],
        dropout=MLP_CONFIG['dropout'],
        learning_rate=MLP_CONFIG['learning_rate'],
        epochs=MLP_CONFIG['epochs']
    )
    model.fit(X_train, y_train)
    
    print("Training complete!")
    
    # Save model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'mlp_model.pkl'
    
    # Save the entire model object
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'model_config': {
            'hidden_sizes': model.hidden_sizes,
            'dropout': model.dropout,
            'input_size': X_train.shape[1]
        },
        'scaler_features': model.scaler_features,
        'scaler_targets': model.scaler_targets,
        'training_config': TRAINING_CONFIG,
        'mlp_config': MLP_CONFIG
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Predict
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    rmse_x = np.sqrt(mean_squared_error(y_val[:, 0], y_pred[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(y_val[:, 1], y_pred[:, 1]))
    
    errors_x = np.abs(y_val[:, 0] - y_pred[:, 0])
    errors_y = np.abs(y_val[:, 1] - y_pred[:, 1])
    
    # Print results
    print("\nMLP Results:")
    print("-" * 40)
    print(f"X coordinate:")
    print(f"  RMSE: {rmse_x:.2f}")
    print(f"  Mean error: {errors_x.mean():.2f}")
    print(f"  Std error: {errors_x.std():.2f}")
    print(f"\nY coordinate:")
    print(f"  RMSE: {rmse_y:.2f}")
    print(f"  Mean error: {errors_y.mean():.2f}")
    print(f"  Std error: {errors_y.std():.2f}")
    
    # Combined metric
    rmse_combined = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
    print(f"\nCombined RMSE: {rmse_combined:.2f}")


if __name__ == "__main__":
    train_model() 