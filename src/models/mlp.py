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
                nn.BatchNorm1d(hidden_size),
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
    
    def __init__(self, hidden_sizes=[128, 64, 32], dropout=0.2, learning_rate=0.001, epochs=100):
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
        """
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()
        self.model = None
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        """Fit the MLP model"""
        # Scale features and targets
        X_scaled = self.scaler_features.fit_transform(X)
        y_scaled = self.scaler_targets.fit_transform(y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        
        # Initialize model
        input_size = X.shape[1]
        self.model = MLPNetwork(input_size, self.hidden_sizes, output_size=2, dropout=self.dropout)
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
        
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