"""
Main entry point for the Position Estimation project
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.training.train_lstm import train_model as train_lstm
from src.training.train_linear import train_model as train_linear
from src.training.train_svr import train_model as train_svr
from src.training.train_rf import train_model as train_rf
from src.training.train_xgb import train_model as train_xgb
from src.training.train_mlp import train_model as train_mlp


def main():
    """Run model training based on command line arguments"""
    parser = argparse.ArgumentParser(description='Train models for position estimation')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['lstm', 'linear', 'svr', 'rf', 'xgb', 'mlp', 'all'],
                        help='Which model to train (default: all)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare results of all models')
    
    args = parser.parse_args()
    
    if args.model in ['lstm', 'all']:
        print("=" * 60)
        print("Training LSTM Model")
        print("=" * 60)
        train_lstm()
        print("\n")
    
    if args.model in ['linear', 'all']:
        print("=" * 60)
        print("Training Linear Baseline Model")
        print("=" * 60)
        train_linear()
        print("\n")
    
    if args.model in ['svr', 'all']:
        print("=" * 60)
        print("Training SVR Model")
        print("=" * 60)
        train_svr()
        print("\n")
    
    if args.model in ['rf', 'all']:
        print("=" * 60)
        print("Training Random Forest Model")
        print("=" * 60)
        train_rf()
        print("\n")
    
    if args.model in ['xgb', 'all']:
        print("=" * 60)
        print("Training XGBoost Model")
        print("=" * 60)
        train_xgb()
        print("\n")
    
    if args.model in ['mlp', 'all']:
        print("=" * 60)
        print("Training MLP (Multi-Layer Perceptron) Model")
        print("=" * 60)
        train_mlp()
        print("\n")
    
    if args.compare and args.model == 'all':
        print("=" * 60)
        print("Model Comparison Summary")
        print("=" * 60)
        compare_models()


def compare_models():
    """Compare the performance of all models"""
    import joblib
    import torch
    from pathlib import Path
    
    # Check if all models exist
    lstm_path = Path('results/models/lstm_best_model.pth')
    linear_path = Path('results/models/linear_baseline_model.pkl')
    svr_path = Path('results/models/svr_model.pkl')
    rf_path = Path('results/models/rf_model.pkl')
    xgb_path = Path('results/models/xgb_model.pkl')
    mlp_path = Path('results/models/mlp_model.pkl')
    
    if not all([lstm_path.exists(), linear_path.exists(), svr_path.exists(), 
                rf_path.exists(), xgb_path.exists(), mlp_path.exists()]):
        print("All models need to be trained first for comparison.")
        return
    
    print("\nModel comparison functionality can be extended here.")
    print("All models have been trained and saved successfully.")
    
    # You can add more detailed comparison logic here
    # For example, loading all models and comparing their validation metrics


if __name__ == "__main__":
    # If no arguments provided, show menu
    if len(sys.argv) == 1:
        print("\nPosition Estimation Model Training")
        print("-" * 35)
        print("1. Train LSTM model")
        print("2. Train Linear baseline model")
        print("3. Train SVR model")
        print("4. Train Random Forest model")
        print("5. Train XGBoost model")
        print("6. Train MLP model")
        print("7. Train all models")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ")
        
        if choice == '1':
            sys.argv.extend(['--model', 'lstm'])
        elif choice == '2':
            sys.argv.extend(['--model', 'linear'])
        elif choice == '3':
            sys.argv.extend(['--model', 'svr'])
        elif choice == '4':
            sys.argv.extend(['--model', 'rf'])
        elif choice == '5':
            sys.argv.extend(['--model', 'xgb'])
        elif choice == '6':
            sys.argv.extend(['--model', 'mlp'])
        elif choice == '7':
            sys.argv.extend(['--model', 'all'])
        elif choice == '8':
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Exiting...")
            sys.exit(1)
    
    main()