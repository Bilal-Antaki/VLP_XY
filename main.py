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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def load_model_predictions():
    """Load all models and make predictions on validation data"""
    # Load data
    df = pd.read_csv('data/features/features_selected.csv')
    feature_cols = [col for col in df.columns 
                   if col not in ['X', 'Y', 'trajectory_id', 'step_id']]
    
    # Validation data (trajectories 16-19)
    val_df = df[df['trajectory_id'] >= 16]
    X_val = val_df[feature_cols].values
    Y_val = val_df[['X', 'Y']].values
    
    # Get validation trajectories
    val_trajectories = []
    for traj_id in range(16, 20):
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            val_trajectories.append({
                'X': traj_data[feature_cols].values,
                'Y': traj_data[['X', 'Y']].values,
                'id': traj_id
            })
    
    models = {}
    predictions = {}
    rmse_scores = {}
    
    # Load Linear model
    try:
        linear_data = joblib.load('results/models/linear_baseline_model.pkl')
        models['Linear'] = linear_data['model']
        pred = models['Linear'].predict(X_val)
        predictions['Linear'] = pred
        rmse_x = np.sqrt(mean_squared_error(Y_val[:, 0], pred[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(Y_val[:, 1], pred[:, 1]))
        rmse_scores['Linear'] = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
        print(f"Linear model loaded - RMSE: {rmse_scores['Linear']:.2f}")
    except Exception as e:
        print(f"Failed to load Linear model: {e}")
    
    # Load SVR model
    try:
        svr_data = joblib.load('results/models/svr_model.pkl')
        models['SVR'] = svr_data['model']
        pred = models['SVR'].predict(X_val)
        predictions['SVR'] = pred
        rmse_x = np.sqrt(mean_squared_error(Y_val[:, 0], pred[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(Y_val[:, 1], pred[:, 1]))
        rmse_scores['SVR'] = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
        print(f"SVR model loaded - RMSE: {rmse_scores['SVR']:.2f}")
    except Exception as e:
        print(f"Failed to load SVR model: {e}")
    
    # Load Random Forest model
    try:
        rf_data = joblib.load('results/models/rf_model.pkl')
        models['Random Forest'] = rf_data['model']
        pred = models['Random Forest'].predict(X_val)
        predictions['Random Forest'] = pred
        rmse_x = np.sqrt(mean_squared_error(Y_val[:, 0], pred[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(Y_val[:, 1], pred[:, 1]))
        rmse_scores['Random Forest'] = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
        print(f"Random Forest model loaded - RMSE: {rmse_scores['Random Forest']:.2f}")
    except Exception as e:
        print(f"Failed to load Random Forest model: {e}")
    
    # Load XGBoost model
    try:
        xgb_data = joblib.load('results/models/xgb_model.pkl')
        models['XGBoost'] = xgb_data['model']
        pred = models['XGBoost'].predict(X_val)
        predictions['XGBoost'] = pred
        rmse_x = np.sqrt(mean_squared_error(Y_val[:, 0], pred[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(Y_val[:, 1], pred[:, 1]))
        rmse_scores['XGBoost'] = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
        print(f"XGBoost model loaded - RMSE: {rmse_scores['XGBoost']:.2f}")
    except Exception as e:
        print(f"Failed to load XGBoost model: {e}")
    
    # Load MLP model
    try:
        mlp_data = torch.load('results/models/mlp_model.pkl', weights_only=False)
        from src.models.mlp import MLPModel
        model_config = mlp_data['model_config']
        mlp_model = MLPModel(
            hidden_sizes=model_config['hidden_sizes'],
            dropout=model_config['dropout']
        )
        mlp_model.model.load_state_dict(mlp_data['model_state_dict'])
        mlp_model.scaler_features = mlp_data['scaler_features']
        mlp_model.scaler_targets = mlp_data['scaler_targets']
        models['MLP'] = mlp_model
        pred = models['MLP'].predict(X_val)
        predictions['MLP'] = pred
        rmse_x = np.sqrt(mean_squared_error(Y_val[:, 0], pred[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(Y_val[:, 1], pred[:, 1]))
        rmse_scores['MLP'] = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
        print(f"MLP model loaded - RMSE: {rmse_scores['MLP']:.2f}")
    except Exception as e:
        print(f"Failed to load MLP model: {e}")
    
    # Load LSTM model
    try:
        lstm_data = torch.load('results/models/lstm_best_model.pth', weights_only=False)
        from src.models.lstm import TrajectoryLSTM
        model_config = lstm_data['model_config']
        lstm_model = TrajectoryLSTM(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout']
        )
        lstm_model.load_state_dict(lstm_data['model_state_dict'])
        lstm_model.eval()
        
        # Prepare LSTM data (sequence-to-sequence)
        scaler_X = lstm_data['scaler_X']
        scaler_Y = lstm_data['scaler_Y']
        
        # Get validation trajectories as sequences
        lstm_predictions = []
        for traj_id in range(16, 20):
            traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
            if len(traj_data) == 10:
                # Get feature sequence for this trajectory
                X_seq = traj_data[feature_cols].values.reshape(1, 10, -1)  # (1, 10, features)
                
                # Scale the sequence
                X_seq_flat = X_seq.reshape(-1, X_seq.shape[-1])
                X_seq_scaled = scaler_X.transform(X_seq_flat).reshape(X_seq.shape)
                X_seq_tensor = torch.FloatTensor(X_seq_scaled)
                
                # Predict sequence
                with torch.no_grad():
                    pred_seq_scaled = lstm_model(X_seq_tensor)  # (1, 10, 2)
                    pred_seq_flat = pred_seq_scaled.view(-1, 2).numpy()
                    pred_seq = scaler_Y.inverse_transform(pred_seq_flat).reshape(1, 10, 2)
                    
                # Add to predictions
                lstm_predictions.extend(pred_seq[0])  # Flatten to (10, 2) and extend
        
        predictions['LSTM'] = np.array(lstm_predictions)  # (40, 2)
        
        rmse_x = np.sqrt(mean_squared_error(Y_val[:, 0], predictions['LSTM'][:, 0]))
        rmse_y = np.sqrt(mean_squared_error(Y_val[:, 1], predictions['LSTM'][:, 1]))
        rmse_scores['LSTM'] = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
        print(f"LSTM model loaded - RMSE: {rmse_scores['LSTM']:.2f}")
    except Exception as e:
        print(f"Failed to load LSTM model: {e}")
    
    return predictions, rmse_scores, Y_val, val_trajectories, df


def create_model_comparison_plot(rmse_scores):
    """Create model RMSE comparison plot"""
    plt.figure(figsize=(10, 6))
    
    models = list(rmse_scores.keys())
    scores = list(rmse_scores.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    bars = plt.bar(models, scores, color=colors[:len(models)])
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Model Performance Comparison - Combined RMSE', fontsize=14, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('results/plots/model_rmse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Model comparison plot saved to: results/plots/model_rmse_comparison.png")


def create_individual_model_plots(predictions, Y_val, val_trajectories, df):
    """Create 4-figure plots for each model: X pred vs actual, Y pred vs actual, first traj, second traj"""
    
    for model_name, pred in predictions.items():
        print(f"Creating plots for {model_name}...")
        
        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Figure 1: X coordinate predicted vs actual
        ax1.scatter(Y_val[:, 0], pred[:, 0], alpha=0.6, s=30, color='blue')
        
        # Perfect prediction line
        min_x = min(Y_val[:, 0].min(), pred[:, 0].min())
        max_x = max(Y_val[:, 0].max(), pred[:, 0].max())
        ax1.plot([min_x, max_x], [min_x, max_x], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        ax1.set_title(f'{model_name} - X Coordinate: Predicted vs Actual', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Actual X', fontsize=11)
        ax1.set_ylabel('Predicted X', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Figure 2: Y coordinate predicted vs actual
        ax2.scatter(Y_val[:, 1], pred[:, 1], alpha=0.6, s=30, color='red')
        
        # Perfect prediction line
        min_y = min(Y_val[:, 1].min(), pred[:, 1].min())
        max_y = max(Y_val[:, 1].max(), pred[:, 1].max())
        ax2.plot([min_y, max_y], [min_y, max_y], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        ax2.set_title(f'{model_name} - Y Coordinate: Predicted vs Actual', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Actual Y', fontsize=11)
        ax2.set_ylabel('Predicted Y', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Figure 3: First validation trajectory (trajectory 16)
        if len(val_trajectories) >= 1:
            traj = val_trajectories[0]
            true_path = traj['Y']
            start_idx = 0 * 10
            end_idx = start_idx + 10
            pred_path = pred[start_idx:end_idx]
            
            # Plot true trajectory
            ax3.plot(true_path[:, 0], true_path[:, 1], 'o-', 
                    label=f'True Traj {traj["id"]}', color='blue', alpha=0.8, linewidth=2, markersize=6)
            # Plot predicted trajectory
            ax3.plot(pred_path[:, 0], pred_path[:, 1], 's--', 
                    label=f'Pred Traj {traj["id"]}', color='orange', alpha=0.8, linewidth=2, markersize=6)
            
            # Mark starting points with large red stars
            ax3.scatter(true_path[0, 0], true_path[0, 1], 
                       s=150, c='red', marker='*', zorder=5, label='Start Point', edgecolors='black')
            ax3.scatter(pred_path[0, 0], pred_path[0, 1], 
                       s=150, c='darkred', marker='*', zorder=5, edgecolors='black')
        
        ax3.set_title(f'{model_name} - First Validation Trajectory', fontweight='bold', fontsize=12)
        ax3.set_xlabel('X Position', fontsize=11)
        ax3.set_ylabel('Y Position', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Figure 4: Second validation trajectory (trajectory 17)
        if len(val_trajectories) >= 2:
            traj = val_trajectories[1]
            true_path = traj['Y']
            start_idx = 1 * 10
            end_idx = start_idx + 10
            pred_path = pred[start_idx:end_idx]
            
            # Plot true trajectory
            ax4.plot(true_path[:, 0], true_path[:, 1], 'o-', 
                    label=f'True Traj {traj["id"]}', color='green', alpha=0.8, linewidth=2, markersize=6)
            # Plot predicted trajectory
            ax4.plot(pred_path[:, 0], pred_path[:, 1], 's--', 
                    label=f'Pred Traj {traj["id"]}', color='purple', alpha=0.8, linewidth=2, markersize=6)
            
            # Mark starting points with large red stars
            ax4.scatter(true_path[0, 0], true_path[0, 1], 
                       s=150, c='red', marker='*', zorder=5, label='Start Point', edgecolors='black')
            ax4.scatter(pred_path[0, 0], pred_path[0, 1], 
                       s=150, c='darkred', marker='*', zorder=5, edgecolors='black')
        
        ax4.set_title(f'{model_name} - Second Validation Trajectory', fontweight='bold', fontsize=12)
        ax4.set_xlabel('X Position', fontsize=11)
        ax4.set_ylabel('Y Position', fontsize=11)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f'results/plots/{model_name.lower().replace(" ", "_")}_performance.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{model_name} performance plot saved to: {filename}")


def visualize_all_models():
    """Generate all visualization plots for trained models"""
    print("Loading models and generating predictions...")
    
    # Ensure results/plots directory exists
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    
    # Load all models and predictions
    predictions, rmse_scores, Y_val, val_trajectories, df = load_model_predictions()
    
    if not predictions:
        print("No models found to visualize!")
        return
    
    # Create model comparison plot
    create_model_comparison_plot(rmse_scores)
    
    # Create individual model plots
    create_individual_model_plots(predictions, Y_val, val_trajectories, df)
    
    print(f"\nAll visualization plots have been saved to: results/plots/")
    print(f"Generated plots for {len(predictions)} models: {list(predictions.keys())}")


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
    
    print("\nGenerating visualization plots for all models...")
    visualize_all_models()
    print("All models have been trained and visualized successfully.")


if __name__ == "__main__":
    METHOD = 'random_forest'  # 'lasso' or 'random_forest'
    
    # Run the complete pipeline
    features_df, selected_features, (X_train, Y_train, X_val, Y_val) = run_complete_pipeline(selection_method=METHOD)
    print(X_train.shape)
    print(Y_train.shape)

    main()
