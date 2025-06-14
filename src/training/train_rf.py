"""
Training script for Random Forest model
"""

from src.config import TRAINING_CONFIG, RF_CONFIG
from sklearn.metrics import mean_squared_error
from src.models.rf import RandomForestModel
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import random
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model():
    """Train and evaluate Random Forest model"""
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
    print(f"Training Random Forest model with n_estimators={RF_CONFIG['n_estimators']}, max_depth={RF_CONFIG['max_depth']}...")
    model = RandomForestModel(
        n_estimators=RF_CONFIG['n_estimators'],
        max_depth=RF_CONFIG['max_depth'],
        random_state=RF_CONFIG['random_state']
    )
    model.fit(X_train, y_train)
    
    print("Training complete!")
    
    # Save model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'rf_model.pkl'
    
    joblib.dump({
        'model': model,
        'feature_count': X_train.shape[1],
        'training_config': TRAINING_CONFIG,
        'rf_config': RF_CONFIG
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Predict
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    rmse_x = np.sqrt(mean_squared_error(y_val[:, 0], y_pred[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(y_val[:, 1], y_pred[:, 1]))
    
    errors_x = np.abs(y_val[:, 0] - y_pred[:, 0])
    errors_y = np.abs(y_val[:, 1] - y_pred[:, 1])
