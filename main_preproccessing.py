"""
Main script to run the complete feature engineering pipeline
This script should be placed in the project root directory
"""

import sys
from pathlib import Path
import os
import pandas as pd

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import the feature selection module which includes feature engineering
from src.data.feature_engineering import main as run_feature_engineering
from src.data.preprocessing import prepare_training_data
from src.data.feature_selection import main as run_feature_selection

# Create a wrapper to run the complete pipeline
def run_complete_pipeline(selection_method='lasso'):
    """
    Run the complete data preprocessing pipeline
    
    Parameters
    ----------
    selection_method : str
        Feature selection method - 'lasso'
        
    Returns
    -------
    tuple
        (features_df, selected_features, (X_train, Y_train, X_val, Y_val))
    """
    print("\nRunning complete preprocessing pipeline...")
    print(f"Method: {selection_method.upper()}")
    
    # Run feature selection
    selected_features = run_feature_selection(method=selection_method)
    
    # Load selected features
    features_df = pd.read_csv('data/features/features_selected.csv')
    
    # Prepare data for training
    X_train, Y_train, X_val, Y_val = prepare_training_data(features_df, selected_features)
    
    print("\nPreprocessing complete!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"\nSelected features include mandatory PL and RMS plus top 5 from {selection_method}")
    
    return features_df, selected_features, (X_train, Y_train, X_val, Y_val)

if __name__ == "__main__":
    run_complete_pipeline()