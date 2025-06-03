"""
GRU model for trajectory prediction
"""

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class GRUNetwork(nn.Module):
    """
    GRU with batch normalization and dropout
    """
    
    def __init__(self, input_size, hidden_dim, num_layers=1, dropout=0.2):
        super(GRUNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        
        return out


class GRUModel:
    """
    GRU wrapper for trajectory prediction
    Fits a neural network for X and Y coordinates
    """
    
    def __init__(self, hidden_dim=128, num_layers=1, dropout=0.2, learning_rate=0.001, epochs=100):
        """
        Initialize the GRU model
        
        Parameters:
        -----------
        hidden_dim : int, default=128
            Number of features in the hidden state
        num_layers : int, default=1
            Number of recurrent layers
        dropout : float, default=0.2
            Dropout rate
        learning_rate : float, default=0.001
            Learning rate for optimization
        epochs : int, default=100
            Number of training epochs
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()
        self.model = None
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        """Fit the GRU model"""
        # Scale features and targets
        X_scaled = self.scaler_features.fit_transform(X)
        y_scaled = self.scaler_targets.fit_transform(y)
        
        # Reshape for sequence data (batch_size, seq_len, features)
        X_reshaped = X_scaled.reshape(-1, 10, X_scaled.shape[1])
        y_reshaped = y_scaled.reshape(-1, 10, 2)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
        y_tensor = torch.FloatTensor(y_reshaped).to(self.device)
        
        # Initialize model
        input_size = X.shape[1]
        self.model = GRUNetwork(input_size, self.hidden_dim, self.num_layers, self.dropout)
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
        
        # Reshape for sequence data
        X_reshaped = X_scaled.reshape(-1, 10, X_scaled.shape[1])
        X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        # Reshape back to original format
        predictions_scaled = predictions_scaled.reshape(-1, 2)
        
        # Inverse transform predictions
        predictions = self.scaler_targets.inverse_transform(predictions_scaled)
        
        return predictions.astype(int) 