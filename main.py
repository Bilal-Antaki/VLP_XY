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


def main():
    """Run model training based on command line arguments"""
    parser = argparse.ArgumentParser(description='Train models for position estimation')
    parser.add_argument('--model', type=str, default='both', 
                        choices=['lstm', 'linear', 'both'],
                        help='Which model to train (default: both)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare results of both models')
    
    args = parser.parse_args()
    
    if args.model in ['lstm', 'both']:
        print("=" * 60)
        print("Training LSTM Model")
        print("=" * 60)
        train_lstm()
        print("\n")
    
    if args.model in ['linear', 'both']:
        print("=" * 60)
        print("Training Linear Baseline Model")
        print("=" * 60)
        train_linear()
        print("\n")
    
    if args.compare and args.model == 'both':
        print("=" * 60)
        print("Model Comparison Summary")
        print("=" * 60)
        compare_models()


def compare_models():
    """Compare the performance of LSTM and Linear models"""
    import joblib
    import torch
    from pathlib import Path
    
    # Check if both models exist
    lstm_path = Path('models/saved/lstm_best_model.pth')
    linear_path = Path('models/saved/linear_baseline_model.pkl')
    
    if not lstm_path.exists() or not linear_path.exists():
        print("Both models need to be trained first for comparison.")
        return
    
    print("\nModel comparison functionality can be extended here.")
    print("Both models have been trained and saved successfully.")
    
    # You can add more detailed comparison logic here
    # For example, loading both models and comparing their validation metrics


if __name__ == "__main__":
    # If no arguments provided, show menu
    if len(sys.argv) == 1:
        print("\nPosition Estimation Model Training")
        print("-" * 35)
        print("1. Train LSTM model")
        print("2. Train Linear baseline model")
        print("3. Train both models")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ")
        
        if choice == '1':
            sys.argv.extend(['--model', 'lstm'])
        elif choice == '2':
            sys.argv.extend(['--model', 'linear'])
        elif choice == '3':
            sys.argv.extend(['--model', 'both'])
        elif choice == '4':
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Exiting...")
            sys.exit(1)
    
    main()