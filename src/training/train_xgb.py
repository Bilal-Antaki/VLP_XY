"""
Training script for XGBoost model
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
from src.models.xgb import XGBoostModel
from src.config import TRAINING_CONFIG, XGB_CONFIG


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model():
    """Train and evaluate XGBoost model"""
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
    print(f"Training XGBoost model with n_estimators={XGB_CONFIG['n_estimators']}, max_depth={XGB_CONFIG['max_depth']}, learning_rate={XGB_CONFIG['learning_rate']}...")
    model = XGBoostModel(
        n_estimators=XGB_CONFIG['n_estimators'],
        max_depth=XGB_CONFIG['max_depth'],
        learning_rate=XGB_CONFIG['learning_rate'],
        random_state=XGB_CONFIG['random_state']
    )
    model.fit(X_train, y_train)
    
    print("Training complete!")
    
    # Save model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'xgb_model.pkl'
    
    joblib.dump({
        'model': model,
        'feature_count': X_train.shape[1],
        'training_config': TRAINING_CONFIG,
        'xgb_config': XGB_CONFIG
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
    print("\nXGBoost Results:")
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