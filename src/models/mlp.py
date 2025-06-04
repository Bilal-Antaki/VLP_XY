"""
Multi-Layer Perceptron (MLP) model for trajectory prediction
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class MLPNetwork(nn.Module):
    """
    Multi-Layer Perceptron with batch normalization and dropout
    """
    
    def __init__(self, input_size, hidden_sizes, output_size=2, dropout=0.2):
        super(MLPNetwork, self).__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size, momentum=0.1),  # Reduced momentum for better stability
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class MLPModel:
    """
    MLP wrapper for trajectory prediction
    Fits a neural network for X and Y coordinates
    """
    
    def __init__(self, hidden_sizes=[128, 64, 32], dropout=0.2, learning_rate=0.001, epochs=100,
                 weight_decay=1e-5, patience=20, min_delta=1e-4):
        """
        Initialize the MLP model
        
        Parameters:
        -----------
        hidden_sizes : list, default=[128, 64, 32]
            Sizes of hidden layers
        dropout : float, default=0.2
            Dropout rate
        learning_rate : float, default=0.001
            Learning rate for optimization
        epochs : int, default=100
            Number of training epochs
        weight_decay : float, default=1e-5
            L2 regularization strength
        patience : int, default=20
            Number of epochs to wait for improvement before early stopping
        min_delta : float, default=1e-4
            Minimum change in validation loss to be considered as improvement
        """
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()
        self.model = None
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_model_state = None
        self.best_val_loss = float('inf')
        
    def fit(self, X, y, validation_split=0.2):
        """Fit the MLP model"""
        # Scale features and targets
        X_scaled = self.scaler_features.fit_transform(X)
        y_scaled = self.scaler_targets.fit_transform(y)
        
        # Split into train and validation sets
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        val_size = int(n_samples * validation_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train = X_scaled[train_indices]
        y_train = y_scaled[train_indices]
        X_val = X_scaled[val_indices]
        y_val = y_scaled[val_indices]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Initialize model
        input_size = X.shape[1]
        self.model = MLPNetwork(input_size, self.hidden_sizes, output_size=2, dropout=self.dropout)
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Training
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            train_loss = criterion(outputs, y_train_tensor)
            train_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}]')
                print(f'Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
                print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            if patience_counter >= self.patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self.is_fitted = True
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler_features.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform predictions
        predictions = self.scaler_targets.inverse_transform(predictions_scaled)
        
        return predictions.astype(int) 