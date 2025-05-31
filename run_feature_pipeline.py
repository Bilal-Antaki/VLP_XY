"""
Main script to run the complete feature engineering pipeline
This script should be placed in the project root directory
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import the feature selection module which includes feature engineering
from src.data.feature_engineering import main as run_feature_engineering

# Create a wrapper to also run feature selection
def run_complete_pipeline(selection_method='lasso'):
    """
    Run the complete feature engineering and selection pipeline
    
    Parameters:
    -----------
    selection_method : str
        Feature selection method - 'lasso' or 'random_forest'
    """
    print("="*60)
    print("TRAJECTORY PREDICTION - FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Step 1: Run feature engineering
    print("\nStep 1: Engineering features...")
    print("-"*40)
    features_df, feature_names = run_feature_engineering()
    
    # Step 2: Run feature selection
    print("\n\nStep 2: Selecting best features...")
    print(f"Method: {selection_method.upper()}")
    print("-"*40)
    
    # Import and run feature selection with the specified method
    from src.data.feature_selection import main as run_feature_selection
    selected_features = run_feature_selection(method=selection_method)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n✓ All features saved to: data/features/features_all.csv")
    print(f"✓ Selected features (7 total) saved to: data/features/features_selected.csv")
    print(f"✓ Feature importance plot saved to: data/features/feature_importance.png")
    print(f"\nSelected features include mandatory PL and RMS plus top 5 from {selection_method}")
    
    return features_df, selected_features


if __name__ == "__main__":

    METHOD = 'random_forest'  # 'lasso' or 'random_forest'
    
    # Run the complete pipeline
    features_df, selected_features = run_complete_pipeline(selection_method=METHOD)