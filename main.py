"""
Main entry point for the Position Estimation project
"""
from src.training.train_linear import train_model as train_linear
from src.training.train_lstm import train_model as train_lstm
from src.training.train_xgb import train_model as train_xgb
from src.training.train_svr import train_model as train_svr
from src.training.train_mlp import train_model as train_mlp
from src.training.train_rf import train_model as train_rf
from main_preproccessing import run_complete_pipeline
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))


def main():
    # Train all models
    print("=" * 60)
    print("Training LSTM Model")
    print("=" * 60)
    train_lstm()
    print("\n")
    
    print("=" * 60)
    print("Training Linear Baseline Model")
    print("=" * 60)
    train_linear()
    print("\n")
    
    print("=" * 60)
    print("Training SVR Model")
    print("=" * 60)
    train_svr()
    print("\n")
    
    print("=" * 60)
    print("Training Random Forest Model")
    print("=" * 60)
    train_rf()
    print("\n")
    
    print("=" * 60)
    print("Training XGBoost Model")
    print("=" * 60)
    train_xgb()
    print("\n")
    
    print("=" * 60)
    print("Training MLP (Multi-Layer Perceptron) Model")
    print("=" * 60)
    train_mlp()
    print("\n")
    
    print("=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    compare_models()


def compare_models():
    """Compare the performance of all models"""
    
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


if __name__ == "__main__":
    METHOD = 'random_forest'  # 'lasso' or 'random_forest'
    
    # Run the complete pipeline
    features_df, selected_features, (X_train, Y_train, X_val, Y_val) = run_complete_pipeline(selection_method=METHOD)
    print(X_train.shape)
    print(Y_train.shape)

    main()