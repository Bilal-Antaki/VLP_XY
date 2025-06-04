"""
Support Vector Regression model for trajectory prediction
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline


class SVRModel:
    """
    SVR wrapper for trajectory prediction
    Uses MultiOutputRegressor to handle X and Y coordinates
    """
    
    def __init__(self, kernel='rbf', C=10.0, epsilon=0.1, gamma='scale', tol=1e-3, max_iter=10000):
        """
        Initialize the SVR model
        
        Parameters:
        -----------
        kernel : str, default='rbf'
            Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        C : float, default=10.0
            Regularization parameter
        epsilon : float, default=0.1
            Epsilon-tube width
        gamma : str or float, default='scale'
            Kernel coefficient
        tol : float, default=1e-3
            Tolerance for stopping criterion
        max_iter : int, default=10000
            Maximum number of iterations
        """
        # Create base SVR model
        base_svr = SVR(
            kernel=kernel,
            C=C,
            epsilon=epsilon,
            gamma=gamma,
            tol=tol,
            max_iter=max_iter,
            cache_size=2000  # Increased cache size for better performance
        )
        
        # Create pipeline with robust scaling and multi-output regression
        self.model = Pipeline([
            ('scaler', RobustScaler()),  # More robust to outliers
            ('svr', MultiOutputRegressor(base_svr, n_jobs=-1))
        ])
        
        # Separate scaler for targets
        self.target_scaler = RobustScaler()
        
    def fit(self, X, y):
        """
        Fit the SVR model
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target matrix with X and Y coordinates
        """
        print("Training SVR model...")
        
        # Scale targets separately
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Fit model with scaled targets
        self.model.fit(X, y_scaled)
        print("Training complete!")
        
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
            
        Returns:
        --------
        np.array : Predicted X and Y coordinates
        """
        # Get scaled predictions
        predictions_scaled = self.model.predict(X)
        
        # Inverse transform predictions
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        
        return predictions 