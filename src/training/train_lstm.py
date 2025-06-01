import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import random
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


def prepare_data():
    """
    Load and prepare data with better scaling
    """
    # Load selected features
    df = pd.read_csv('data/features/features_selected.csv')
    
    # Get feature columns
    feature_cols = [col for col in df.columns 
                   if col not in ['X', 'Y', 'trajectory_id', 'step_id']]
    
    print(f"Using features: {feature_cols}")
    
    # Split trajectories
    train_traj_ids = list(range(16))
    val_traj_ids = list(range(16, 20))
    
    # Prepare training data
    X_train, Y_train = [], []
    for traj_id in train_traj_ids:
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            X_train.append(traj_data[feature_cols].values)
            Y_train.append(traj_data[['X', 'Y']].values)
    
    # Prepare validation data
    X_val, Y_val = [], []
    for traj_id in val_traj_ids:
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            X_val.append(traj_data[feature_cols].values)
            Y_val.append(traj_data[['X', 'Y']].values)
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    
    # Use RobustScaler for better handling of outliers
    scaler_X = RobustScaler()
    scaler_Y = RobustScaler()
    
    # Fit scalers on training data
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    Y_train_flat = Y_train.reshape(-1, Y_train.shape[-1])
    
    X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train.shape)
    Y_train_scaled = scaler_Y.fit_transform(Y_train_flat).reshape(Y_train.shape)
    
    # Transform validation data
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    Y_val_flat = Y_val.reshape(-1, Y_val.shape[-1])
    
    X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape)
    Y_val_scaled = scaler_Y.transform(Y_val_flat).reshape(Y_val.shape)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    Y_train_tensor = torch.FloatTensor(Y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    Y_val_tensor = torch.FloatTensor(Y_val_scaled)
    
    print(f"Training shape: X={X_train_tensor.shape}, Y={Y_train_tensor.shape}")
    print(f"Validation shape: X={X_val_tensor.shape}, Y={Y_val_tensor.shape}")
    
    return (X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor), (scaler_X, scaler_Y)


def train_model():
    # Set random seed for reproducibility
    set_seed(TRAINING_CONFIG['random_seed'])
    
    # Prepare data
    (X_train, Y_train, X_val, Y_val), (scaler_X, scaler_Y) = prepare_data()
    
    # Model parameters from config
    input_size = X_train.shape[-1]
    hidden_size = MODEL_CONFIG['hidden_dim']
    num_layers = MODEL_CONFIG['num_layers']
    dropout = MODEL_CONFIG['dropout']
    output_size = 2
    
    # Initialize model
    model = TrajectoryLSTM(input_size, hidden_size, num_layers, output_size, dropout=dropout)
    
    # Loss and optimizer with config values
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'], 
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # Training with early stopping
    epochs = TRAINING_CONFIG['epochs']
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        train_loss = criterion(outputs, Y_train)
        
        # Backward pass
        train_loss.backward()
        
        # Gradient clipping
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
            
        if patience_counter >= patience:
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save the model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'lstm_best_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size,
            'dropout': dropout
        },
        'scaler_X': scaler_X,
        'scaler_Y': scaler_Y,
        'best_val_loss': best_val_loss.item()
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Validation predictions
        val_pred_scaled = model(X_val)
        
        # Convert back to numpy and unscale
        val_pred_scaled_np = val_pred_scaled.numpy()
        Y_val_np = Y_val.numpy()
        
        # Reshape for inverse transform
        val_pred_flat = val_pred_scaled_np.reshape(-1, 2)
        Y_val_flat = Y_val_np.reshape(-1, 2)
        
        # Inverse transform
        val_pred = scaler_Y.inverse_transform(val_pred_flat).reshape(val_pred_scaled_np.shape).astype(int)
        Y_val_true = scaler_Y.inverse_transform(Y_val_flat).reshape(Y_val_np.shape).astype(int)
        
        # Calculate RMSE for each trajectory, separately for X and Y
        rmse_x_per_traj = []
        rmse_y_per_traj = []
        
        for i in range(len(val_pred)):
            # X coordinate RMSE
            mse_x = mean_squared_error(Y_val_true[i, :, 0], val_pred[i, :, 0])
            rmse_x = np.sqrt(mse_x)
            rmse_x_per_traj.append(rmse_x)
            
            # Y coordinate RMSE
            mse_y = mean_squared_error(Y_val_true[i, :, 1], val_pred[i, :, 1])
            rmse_y = np.sqrt(mse_y)
            rmse_y_per_traj.append(rmse_y)
        
        rmse_x_per_traj = np.array(rmse_x_per_traj)
        rmse_y_per_traj = np.array(rmse_y_per_traj)
        
        # Print detailed results
        print("\nX-coordinate Metrics:")
        print(f"RMSE: {rmse_x_per_traj.mean():.2f}")
        print(f"Std: {rmse_x_per_traj.std():.2f}")
        print(f"Mean: {np.mean(np.abs(Y_val_true[:, :, 0] - val_pred[:, :, 0])):.2f}")
        
        print("\nY-coordinate Metrics:")
        print(f"RMSE: {rmse_y_per_traj.mean():.2f}")
        print(f"Std: {rmse_y_per_traj.std():.2f}")
        print(f"Mean: {np.mean(np.abs(Y_val_true[:, :, 1] - val_pred[:, :, 1])):.2f}")
        
        print("\nOverall Metrics:")
        rmse_total = np.sqrt((rmse_x_per_traj**2 + rmse_y_per_traj**2)/2)
        print(f"Combined RMSE: {rmse_total.mean():.2f}")
        print(f"Combined Std: {rmse_total.std():.2f}")
        
        # Print sample predictions if needed
        print(f"\nSample true and predicted values (first trajectory):")
        print(f"X: {Y_val_true[0, :, 0]}")
        print(f"X_pred: {val_pred[0, :, 0]}")
        print(f"Y: {Y_val_true[0, :, 1]}")
        print(f"Y_pred: {val_pred[0, :, 1]}")


if __name__ == "__main__":
    train_model()