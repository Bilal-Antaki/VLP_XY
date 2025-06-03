"""
Training script for GRU model
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
from src.models.gru import GRUModel
from src.config import TRAINING_CONFIG, MODEL_CONFIG


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model():
    """Train and evaluate GRU model"""
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
    print(f"Training GRU model with hidden_dim={MODEL_CONFIG['hidden_dim']}, num_layers={MODEL_CONFIG['num_layers']}, dropout={MODEL_CONFIG['dropout']}...")
    model = GRUModel(
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout'],
        learning_rate=TRAINING_CONFIG['learning_rate'],
        epochs=TRAINING_CONFIG['epochs']
    )
    model.fit(X_train, y_train)
    
    print("Training complete!")
    
    # Save model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'gru_model.pkl'
    
    # Save the entire model object
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'model_config': {
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'dropout': model.dropout,
            'input_size': X_train.shape[1]
        },
        'scaler_features': model.scaler_features,
        'scaler_targets': model.scaler_targets,
        'training_config': TRAINING_CONFIG,
        'model_config': MODEL_CONFIG
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
    print("\nGRU Results:")
    print("-" * 40)
    print(f"X coordinate:")
    print(f"  RMSE: {rmse_x:.2f}")
    print(f"  Std: {errors_x.std():.2f}")
    print(f"  Mean: {np.mean(y_val[:, 0] - y_pred[:, 0]):.2f}")

    print(f"\nY coordinate:")
    print(f"  RMSE: {rmse_y:.2f}")
    print(f"  Std: {errors_y.std():.2f}")
    print(f"  Mean: {np.mean(y_val[:, 1] - y_pred[:, 1]):.2f}")
    
    # Combined metric
    rmse_combined = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
    print(f"\nCombined RMSE: {rmse_combined:.2f}")


if __name__ == "__main__":
    train_model() 