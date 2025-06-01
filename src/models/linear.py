"""
Linear regression baseline model for trajectory prediction
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class LinearBaselineModel:
    """
    Simple linear regression baseline for trajectory prediction
    Fits separate models for X and Y coordinates
    """
    
    def __init__(self):
        """Initialize the linear models and scalers"""
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        self.scaler_features = StandardScaler()
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit the linear models
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples, 2)
            Target values [X, Y coordinates]
        """
        # Scale features
        X_scaled = self.scaler_features.fit_transform(X)
        
        # Scale targets separately
        y_x = y[:, 0].reshape(-1, 1)
        y_y = y[:, 1].reshape(-1, 1)
        
        y_x_scaled = self.scaler_x.fit_transform(y_x).ravel()
        y_y_scaled = self.scaler_y.fit_transform(y_y).ravel()
        
        # Fit models
        self.model_x.fit(X_scaled, y_x_scaled)
        self.model_y.fit(X_scaled, y_y_scaled)
        
        self.is_fitted = True
        
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features to predict on
            
        Returns:
        --------
        array-like of shape (n_samples, 2) : Predictions [X, Y]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler_features.transform(X)
        
        # Predict
        pred_x_scaled = self.model_x.predict(X_scaled)
        pred_y_scaled = self.model_y.predict(X_scaled)
        
        # Inverse transform predictions
        pred_x = self.scaler_x.inverse_transform(pred_x_scaled.reshape(-1, 1)).ravel()
        pred_y = self.scaler_y.inverse_transform(pred_y_scaled.reshape(-1, 1)).ravel()
        
        # Combine predictions
        predictions = np.column_stack([pred_x, pred_y])
        
        return predictions.astype(int)
