from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch.optim as optim
from pathlib import Path
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import torch
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.lstm import TrajectoryLSTM
from src.config import MODEL_CONFIG, TRAINING_CONFIG


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_sequence_data():
    """Prepare sequence-to-sequence data"""
    df = pd.read_csv('data/features/features_selected.csv')
    feature_cols = [col for col in df.columns 
                   if col not in ['X', 'Y', 'trajectory_id', 'step_id']]
    
    print(f"Using features: {feature_cols}")
    
    # Prepare training trajectories (0-15)
    X_train = []
    Y_train = []
    
    for traj_id in range(16):
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            # Input: feature sequence for this trajectory
            X_train.append(traj_data[feature_cols].values)
            # Output: position sequence for this trajectory
            Y_train.append(traj_data[['X', 'Y']].values)
    
    # Prepare validation trajectories (16-19)
    X_val = []
    Y_val = []
    
    for traj_id in range(16, 20):
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            # Input: feature sequence for this trajectory
            X_val.append(traj_data[feature_cols].values)
            # Output: position sequence for this trajectory
            Y_val.append(traj_data[['X', 'Y']].values)
    
    X_train = np.array(X_train)  # (16, 10, features)
    Y_train = np.array(Y_train)  # (16, 10, 2)
    X_val = np.array(X_val)      # (4, 10, features)
    Y_val = np.array(Y_val)      # (4, 10, 2)
    
    print(f"Training shape: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Validation shape: X={X_val.shape}, Y={Y_val.shape}")
    print("Task: Feature sequence → Position sequence (for same trajectory)")
    
    # Scale data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    # Flatten for scaling
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    Y_train_flat = Y_train.reshape(-1, Y_train.shape[-1])
    
    X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train.shape)
    Y_train_scaled = scaler_Y.fit_transform(Y_train_flat).reshape(Y_train.shape)
    
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    Y_val_flat = Y_val.reshape(-1, Y_val.shape[-1])
    
    X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape)
    Y_val_scaled = scaler_Y.transform(Y_val_flat).reshape(Y_val.shape)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    Y_train_tensor = torch.FloatTensor(Y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    Y_val_tensor = torch.FloatTensor(Y_val_scaled)
    
    return (X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor), (scaler_X, scaler_Y)


def train_model():
    """Train sequence-to-sequence LSTM model"""
    set_seed(TRAINING_CONFIG['random_seed'])
    
    # Prepare data
    (X_train, Y_train, X_val, Y_val), (scaler_X, scaler_Y) = prepare_sequence_data()
    
    # Initialize model
    model = TrajectoryLSTM(
        input_size=X_train.shape[-1], 
        hidden_size=128, 
        num_layers=2, 
        dropout=0.3
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=TRAINING_CONFIG['learning_rate'], 
                           weight_decay=TRAINING_CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # Training
    epochs = TRAINING_CONFIG['epochs']
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting Sequence-to-Sequence LSTM training...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        train_loss = criterion(outputs, Y_train)
        
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, Y_val)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
        if patience_counter >= 50:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model and save
    model.load_state_dict(best_model_state)
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': X_train.shape[-1],
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3
        },
        'scaler_X': scaler_X,
        'scaler_Y': scaler_Y,
        'best_val_loss': best_val_loss.item()
    }, model_dir / 'lstm_best_model.pth')
    
    print(f"\nModel saved to: {model_dir / 'lstm_best_model.pth'}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred_scaled = model(X_val)
        
        # Unscale predictions
        val_pred_flat = val_pred_scaled.view(-1, 2).numpy()
        Y_val_flat = Y_val.view(-1, 2).numpy()
        
        val_pred = scaler_Y.inverse_transform(val_pred_flat).reshape(val_pred_scaled.shape).astype(int)
        Y_val_true = scaler_Y.inverse_transform(Y_val_flat).reshape(Y_val.shape).astype(int)
        
        # Calculate RMSE for each validation trajectory
        print("\nSequence-to-Sequence LSTM Results:")
        print("="*60)
        
        total_rmse_x = 0
        total_rmse_y = 0
        
        for i in range(4):
            traj_id = 16 + i
            
            # RMSE for this trajectory
            rmse_x = np.sqrt(mean_squared_error(Y_val_true[i, :, 0], val_pred[i, :, 0]))
            rmse_y = np.sqrt(mean_squared_error(Y_val_true[i, :, 1], val_pred[i, :, 1]))
            rmse_combined = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
            
            total_rmse_x += rmse_x
            total_rmse_y += rmse_y
            
            print(f"Trajectory {traj_id}: X-RMSE: {rmse_x:.2f}, Y-RMSE: {rmse_y:.2f}, Combined: {rmse_combined:.2f}")
            
            # Show sample predictions for first trajectory
            if i == 0:
                print(f"  Sample predictions (Trajectory {traj_id}):")
                for step in range(min(5, 10)):
                    print(f"    Step {step+1}: True=({Y_val_true[i, step, 0]}, {Y_val_true[i, step, 1]}) "
                          f"Pred=({val_pred[i, step, 0]}, {val_pred[i, step, 1]})")
        
        avg_rmse_x = total_rmse_x / 4
        avg_rmse_y = total_rmse_y / 4
        avg_rmse_combined = np.sqrt((avg_rmse_x**2 + avg_rmse_y**2) / 2)
        
        print(f"\nOverall Average:")
        print(f"X-coordinate RMSE: {avg_rmse_x:.2f}")
        print(f"Y-coordinate RMSE: {avg_rmse_y:.2f}")
        print(f"Combined RMSE: {avg_rmse_combined:.2f}")


if __name__ == "__main__":
    train_model()
