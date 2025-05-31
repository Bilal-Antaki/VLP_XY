"""
Training script for the RNN trajectory prediction model
"""

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn
import numpy as np
import torch
import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


# Import using absolute paths
from src.models.rnn import TrajectoryRNN
from src.data.preprocessing import load_and_preprocess_data


def normalize_data(X_train, Y_train, X_val, Y_val):
    """
    Normalize features and targets using StandardScaler
    """
    # Initialize scalers
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Reshape data for scaling
    X_train_2d = X_train.reshape(-1, X_train.shape[2])
    Y_train_2d = Y_train.reshape(-1, Y_train.shape[2])
    X_val_2d = X_val.reshape(-1, X_val.shape[2])
    Y_val_2d = Y_val.reshape(-1, Y_val.shape[2])
    
    # Fit and transform training data
    X_train_scaled = feature_scaler.fit_transform(X_train_2d)
    Y_train_scaled = target_scaler.fit_transform(Y_train_2d)
    
    # Transform validation data
    X_val_scaled = feature_scaler.transform(X_val_2d)
    Y_val_scaled = target_scaler.transform(Y_val_2d)
    
    # Reshape back to 3D
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    Y_train_scaled = Y_train_scaled.reshape(Y_train.shape)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    Y_val_scaled = Y_val_scaled.reshape(Y_val.shape)
    
    return (X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled), (feature_scaler, target_scaler)


def train_model(model, train_loader, val_loader, feature_scaler, target_scaler, 
                num_epochs=300, learning_rate=0.001, device='cpu'):
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15  # Increased patience for early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    
    print("\nTraining started...")
    print("="*50)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.float().to(device)
            batch_y = batch_y.float().to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.float().to(device)
                batch_y = batch_y.float().to(device)
                
                predictions, _ = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_rnn_model.pth')
            np.save('models/feature_scaler.npy', {
                'mean_': feature_scaler.mean_,
                'scale_': feature_scaler.scale_,
                'var_': feature_scaler.var_
            })
            np.save('models/target_scaler.npy', {
                'mean_': target_scaler.mean_,
                'scale_': target_scaler.scale_,
                'var_': target_scaler.var_
            })
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print("-"*30)
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    return train_losses, val_losses


def load_scalers():
    """Load and reconstruct the scalers"""
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Load scaler parameters
    feature_params = np.load('models/feature_scaler.npy', allow_pickle=True).item()
    target_params = np.load('models/target_scaler.npy', allow_pickle=True).item()
    
    # Reconstruct scalers
    feature_scaler.mean_ = feature_params['mean_']
    feature_scaler.scale_ = feature_params['scale_']
    feature_scaler.var_ = feature_params['var_']
    
    target_scaler.mean_ = target_params['mean_']
    target_scaler.scale_ = target_params['scale_']
    target_scaler.var_ = target_params['var_']
    
    return feature_scaler, target_scaler


def evaluate_model(model, val_loader, target_scaler, device='cpu'):
    """
    Evaluate the model and compute statistics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.float().to(device)
            predictions, _ = model(batch_X)
            
            # Move to CPU and convert to numpy
            predictions = predictions.cpu().numpy()
            batch_y = batch_y.numpy()
            
            # Inverse transform predictions and targets
            predictions_2d = predictions.reshape(-1, predictions.shape[2])
            targets_2d = batch_y.reshape(-1, batch_y.shape[2])
            
            predictions_original = target_scaler.inverse_transform(predictions_2d)
            targets_original = target_scaler.inverse_transform(targets_2d)
            
            # Reshape back to 3D
            predictions_original = predictions_original.reshape(predictions.shape)
            targets_original = targets_original.reshape(batch_y.shape)
            
            all_predictions.append(predictions_original)
            all_targets.append(targets_original)
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate statistics
    rmse_x = np.sqrt(mean_squared_error(targets[:,:,0], predictions[:,:,0]))
    rmse_y = np.sqrt(mean_squared_error(targets[:,:,1], predictions[:,:,1]))
    
    mean_error_x = np.mean(np.abs(targets[:,:,0] - predictions[:,:,0]))
    mean_error_y = np.mean(np.abs(targets[:,:,1] - predictions[:,:,1]))
    
    std_error_x = np.std(targets[:,:,0] - predictions[:,:,0])
    std_error_y = np.std(targets[:,:,1] - predictions[:,:,1])
    
    print("\nModel Performance Metrics:")
    print("="*50)
    print(f"RMSE X: {rmse_x:.2f}")
    print(f"RMSE Y: {rmse_y:.2f}")
    print(f"Mean Error X: {mean_error_x:.2f}")
    print(f"Mean Error Y: {mean_error_y:.2f}")
    print(f"Std Error X: {std_error_x:.2f}")
    print(f"Std Error Y: {std_error_y:.2f}")
    
    return predictions, targets


def plot_results(predictions, targets, save_dir='plots'):
    """
    Plot predicted vs actual trajectories
    """
    # Create plots directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Flatten the sequence dimension for plotting
    pred_x = predictions[:,:,0].flatten()
    pred_y = predictions[:,:,1].flatten()
    true_x = targets[:,:,0].flatten()
    true_y = targets[:,:,1].flatten()
    
    # Calculate R² scores
    r2_x = np.corrcoef(true_x, pred_x)[0,1]**2
    r2_y = np.corrcoef(true_y, pred_y)[0,1]**2
    
    # Create figure with adjusted spacing
    plt.figure(figsize=(15, 7))
    
    # Plot X coordinates
    ax1 = plt.subplot(121)  # 1x2 grid, first plot
    plt.scatter(true_x, pred_x, alpha=0.5, c='blue', label='Predictions')
    
    # Plot perfect prediction line
    min_x, max_x = min(true_x.min(), pred_x.min()), max(true_x.max(), pred_x.max())
    plt.plot([min_x, max_x], [min_x, max_x], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual X Position')
    plt.ylabel('Predicted X Position')
    plt.title(f'X Coordinate Predictions\nR² = {r2_x:.4f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add error distribution as an inset
    x_errors = pred_x - true_x
    ax_inset1 = plt.axes([0.2, 0.65, 0.15, 0.15])  # Adjusted inset position
    ax_inset1.hist(x_errors, bins=30, color='blue', alpha=0.5)
    ax_inset1.set_title('Error Distribution')
    ax_inset1.set_xlabel('Error')
    
    # Plot Y coordinates
    ax2 = plt.subplot(122)  # 1x2 grid, second plot
    plt.scatter(true_y, pred_y, alpha=0.5, c='green', label='Predictions')
    
    # Plot perfect prediction line
    min_y, max_y = min(true_y.min(), pred_y.min()), max(true_y.max(), pred_y.max())
    plt.plot([min_y, max_y], [min_y, max_y], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Y Position')
    plt.ylabel('Predicted Y Position')
    plt.title(f'Y Coordinate Predictions\nR² = {r2_y:.4f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add error distribution as an inset
    y_errors = pred_y - true_y
    ax_inset2 = plt.axes([0.65, 0.65, 0.15, 0.15])  # Adjusted inset position
    ax_inset2.hist(y_errors, bins=30, color='green', alpha=0.5)
    ax_inset2.set_title('Error Distribution')
    ax_inset2.set_xlabel('Error')
    
    # Add main title with adjusted spacing
    plt.suptitle('Predicted vs Actual Position Coordinates', fontsize=14, y=1.02)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(top=0.85, wspace=0.3)
    
    # Save plot with fixed filename
    plot_path = f"{save_dir}/rnn_predictions.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"\nPrediction plots saved to: {plot_path}")
    plt.close()


def main():
    # Load and preprocess data
    feature_path = "data/features/features_selected.csv"
    X_train, Y_train, X_val, Y_val = load_and_preprocess_data(feature_path)
    
    # Normalize data
    (X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled), (feature_scaler, target_scaler) = normalize_data(X_train, Y_train, X_val, Y_val)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train_scaled)
    Y_train = torch.FloatTensor(Y_train_scaled)
    X_val = torch.FloatTensor(X_val_scaled)
    Y_val = torch.FloatTensor(Y_val_scaled)
    
    # Create data loaders with larger batch size
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    input_size = X_train.shape[2]  # Number of features
    model = TrajectoryRNN(
        input_size=input_size,
        hidden_size=256,
        num_layers=1,
        dropout=0.2
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Train model
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        feature_scaler,
        target_scaler,
        num_epochs=300,  # Increased epochs
        learning_rate=0.001,
        device=device
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/training_curves.png')
    plt.close()
    
    # Load best model and scalers
    model.load_state_dict(torch.load('models/best_rnn_model.pth'))
    _, target_scaler = load_scalers()
    
    # Evaluate
    predictions, targets = evaluate_model(model, val_loader, target_scaler, device=device)
    
    # Plot results
    plot_results(predictions, targets)


if __name__ == "__main__":
    main()
