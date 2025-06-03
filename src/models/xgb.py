"""
XGBoost model for trajectory prediction
"""

import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler


class XGBoostModel:
    """
    XGBoost Regressor for trajectory prediction
    Fits separate models for X and Y coordinates
    """
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """Initialize the XGBoost models and scalers"""
        self.model_x = XGBRegressor(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   learning_rate=learning_rate,
                                   random_state=random_state)
        self.model_y = XGBRegressor(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   learning_rate=learning_rate,
                                   random_state=random_state)
        self.scaler_features = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit the XGBoost models"""
        # Scale features
        X_scaled = self.scaler_features.fit_transform(X)
        
        # Fit models
        self.model_x.fit(X_scaled, y[:, 0])
        self.model_y.fit(X_scaled, y[:, 1])
        
        self.is_fitted = True
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler_features.transform(X)
        
        # Predict
        pred_x = self.model_x.predict(X_scaled)
        pred_y = self.model_y.predict(X_scaled)
        
        # Combine predictions
        predictions = np.column_stack([pred_x, pred_y])
        
        return predictions.astype(int) 