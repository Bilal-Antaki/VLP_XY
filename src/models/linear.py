"""
Linear regression baseline model for trajectory prediction
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class LinearBaselineModel:
    """
    Trajectory-level linear regression for trajectory prediction
    Fits parameters that minimize average error across all trajectories
    """
    
    def __init__(self, alpha=1.0):
        """Initialize the ridge models and scalers"""
        self.model_x = Ridge(alpha=alpha)
        self.model_y = Ridge(alpha=alpha)
        self.scaler_features = StandardScaler()
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_trajectories, y_trajectories):
        """
        Fit the linear models on trajectory sequences
        
        Parameters:
        -----------
        X_trajectories : array-like of shape (n_trajectories, seq_len, n_features)
            Training feature sequences for each trajectory
        y_trajectories : array-like of shape (n_trajectories, seq_len, 2)
            Target position sequences [X, Y coordinates] for each trajectory
        """
        print(f"Training on {len(X_trajectories)} trajectories")
        
        # Flatten all trajectory data for fitting
        # This allows the model to learn patterns across all trajectories
        X_all = []
        y_all = []
        
        for traj_idx in range(len(X_trajectories)):
            X_traj = X_trajectories[traj_idx]  # (seq_len, features)
            y_traj = y_trajectories[traj_idx]  # (seq_len, 2)
            
            # Add trajectory data to the combined dataset
            X_all.append(X_traj)
            y_all.append(y_traj)
        
        # Combine all trajectory data
        X_combined = np.vstack(X_all)  # (n_trajectories * seq_len, features)
        y_combined = np.vstack(y_all)  # (n_trajectories * seq_len, 2)
        
        print(f"Combined training data shape: X={X_combined.shape}, y={y_combined.shape}")
        
        # Scale features
        X_scaled = self.scaler_features.fit_transform(X_combined)
        
        # Scale targets separately
        y_x = y_combined[:, 0].reshape(-1, 1)
        y_y = y_combined[:, 1].reshape(-1, 1)
        
        y_x_scaled = self.scaler_x.fit_transform(y_x).ravel()
        y_y_scaled = self.scaler_y.fit_transform(y_y).ravel()
        
        # Fit models to minimize error across all trajectories
        self.model_x.fit(X_scaled, y_x_scaled)
        self.model_y.fit(X_scaled, y_y_scaled)
        
        self.is_fitted = True
        
        # Calculate training error across all trajectories
        train_pred = self.predict_trajectories(X_trajectories)
        train_errors = []
        
        for traj_idx in range(len(X_trajectories)):
            true_traj = y_trajectories[traj_idx]
            pred_traj = train_pred[traj_idx]
            
            # Calculate RMSE for this trajectory
            rmse_x = np.sqrt(np.mean((true_traj[:, 0] - pred_traj[:, 0])**2))
            rmse_y = np.sqrt(np.mean((true_traj[:, 1] - pred_traj[:, 1])**2))
            rmse_combined = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
            train_errors.append(rmse_combined)
        
        avg_train_error = np.mean(train_errors)
        print(f"Average training error across all trajectories: {avg_train_error:.2f}")
        
    def predict_trajectories(self, X_trajectories):
        """
        Predict complete trajectories
        
        Parameters:
        -----------
        X_trajectories : array-like of shape (n_trajectories, seq_len, n_features)
            Feature sequences for each trajectory
            
        Returns:
        --------
        list of arrays : Predicted trajectories, each of shape (seq_len, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        
        for traj_idx in range(len(X_trajectories)):
            X_traj = X_trajectories[traj_idx]  # (seq_len, features)
            
            # Scale features
            X_scaled = self.scaler_features.transform(X_traj)
            
            # Predict
            pred_x_scaled = self.model_x.predict(X_scaled)
            pred_y_scaled = self.model_y.predict(X_scaled)
            
            # Inverse transform predictions
            pred_x = self.scaler_x.inverse_transform(pred_x_scaled.reshape(-1, 1)).ravel()
            pred_y = self.scaler_y.inverse_transform(pred_y_scaled.reshape(-1, 1)).ravel()
            
            # Combine predictions for this trajectory
            traj_pred = np.column_stack([pred_x, pred_y]).astype(int)
            predictions.append(traj_pred)
        
        return predictions
        
    def predict(self, X):
        """
        Make predictions on individual points (for compatibility with main.py)
        
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
