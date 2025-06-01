"""
Feature selection for trajectory prediction
Selects the best features using Lasso regularization or Random Forest
Always includes original PL and RMS features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    def __init__(self, target_cols=['X', 'Y'], n_features=7, method='lasso'):
        """
        Initialize the feature selector
        
        Parameters:
        -----------
        target_cols : list
            List of target column names
        n_features : int
            Total number of features to select (including PL and RMS)
        method : str
            Selection method - 'lasso' or 'random_forest'
        """
        self.target_cols = target_cols
        self.n_features = n_features
        self.method = method
        self.selected_features = []
        self.feature_scores = {}
        self.scaler = StandardScaler()
        # Always include these base features
        self.mandatory_features = ['PL', 'RMS']
        
    def load_features(self, features_path='data/features/features_all.csv'):
        """
        Load the engineered features
        
        Parameters:
        -----------
        features_path : str
            Path to the features CSV file
            
        Returns:
        --------
        pd.DataFrame : DataFrame with all features
        """
        df = pd.read_csv(features_path)
        print(f"Loaded features from: {features_path}")
        print(f"Shape: {df.shape}")
        return df
    
    def prepare_data(self, df):
        """
        Prepare data for feature selection
        FIXED: Ensures we work with properly structured trajectory data
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with all features
            
        Returns:
        --------
        tuple : (X, y, feature_names)
        """
        # Verify trajectory structure exists
        if 'trajectory_id' not in df.columns or 'step_id' not in df.columns:
            raise ValueError("Data must contain trajectory_id and step_id columns")
        
        # Identify feature columns (exclude targets and metadata)
        exclude_cols = self.target_cols + ['r', 'trajectory_id', 'step_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any remaining NaN or infinite values
        df_clean = df.copy()
        df_clean[feature_cols] = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values within each trajectory independently
        for traj_id in df_clean['trajectory_id'].unique():
            traj_mask = df_clean['trajectory_id'] == traj_id
            traj_data = df_clean.loc[traj_mask, feature_cols]
            # Fill with trajectory-specific means, then 0 for any remaining NaN
            df_clean.loc[traj_mask, feature_cols] = traj_data.fillna(traj_data.mean()).fillna(0)
        
        # Extract features and targets
        X = df_clean[feature_cols].values
        y = df_clean[self.target_cols].values
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target matrix shape: {y.shape}")
        print(f"Number of trajectories: {df_clean['trajectory_id'].nunique()}")
        
        return X, y, feature_cols
    
    def lasso_selection(self, X, y, feature_names):
        """
        Feature selection using Lasso regularization
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target matrix
        feature_names : list
            List of feature names
            
        Returns:
        --------
        list : Selected feature names
        """
        print(f"\n--- Lasso-based Feature Selection (Top {self.n_features} features) ---")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # For multi-output, we'll use Lasso on each target and combine results
        feature_importance = np.zeros(len(feature_names))
        
        for i, target in enumerate(self.target_cols):
            print(f"\nAnalyzing features for {target}...")
            
            # Lasso with cross-validation
            lasso = LassoCV(cv=5, random_state=42, max_iter=5000, n_alphas=100)
            lasso.fit(X_scaled, y[:, i])
            
            # Get feature importances (absolute coefficients)
            importance = np.abs(lasso.coef_)
            feature_importance += importance
            
            print(f"Lasso alpha for {target}: {lasso.alpha_:.6f}")
            print(f"Number of non-zero coefficients: {np.sum(importance > 0)}")
        
        # Average importance across targets
        feature_importance /= len(self.target_cols)
        
        # Create a dict of feature scores
        feature_score_dict = dict(zip(feature_names, feature_importance))
        
        # Get top features (excluding mandatory ones first)
        non_mandatory_features = [f for f in feature_names if f not in self.mandatory_features]
        non_mandatory_scores = [(f, feature_score_dict[f]) for f in non_mandatory_features]
        non_mandatory_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top features after mandatory ones
        n_additional = self.n_features - len(self.mandatory_features)
        selected_additional = [f for f, score in non_mandatory_scores[:n_additional]]
        
        # Combine mandatory and selected features
        selected_features = self.mandatory_features + selected_additional
        
        # Print all selected features with scores
        print(f"\nSelected {len(selected_features)} features:")
        for feat in selected_features:
            print(f"  {feat}: {feature_score_dict[feat]:.4f}")
        
        # Store scores
        self.feature_scores['lasso'] = feature_score_dict
        
        return selected_features
    
    def random_forest_selection(self, X, y, feature_names):
        """
        Feature selection using Random Forest feature importance
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target matrix
        feature_names : list
            List of feature names
            
        Returns:
        --------
        list : Selected feature names
        """
        print(f"\n--- Random Forest Feature Selection (Top {self.n_features} features) ---")
        
        # Use MultiOutputRegressor for multi-target regression
        rf = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10)
        )
        rf.fit(X, y)
        
        # Get feature importances (average across all trees and targets)
        feature_importance = np.zeros(len(feature_names))
        
        for i, estimator in enumerate(rf.estimators_):
            feature_importance += estimator.feature_importances_
            
        feature_importance /= len(rf.estimators_)
        
        # Create a dict of feature scores
        feature_score_dict = dict(zip(feature_names, feature_importance))
        
        # Print scores for mandatory features
        print(f"\nMandatory feature scores:")
        for feat in self.mandatory_features:
            if feat in feature_score_dict:
                print(f"  {feat}: {feature_score_dict[feat]:.4f}")
        
        # Get top features (excluding mandatory ones first)
        non_mandatory_features = [f for f in feature_names if f not in self.mandatory_features]
        non_mandatory_scores = [(f, feature_score_dict[f]) for f in non_mandatory_features]
        non_mandatory_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top features after mandatory ones
        n_additional = self.n_features - len(self.mandatory_features)
        selected_additional = [f for f, score in non_mandatory_scores[:n_additional]]
        
        # Combine mandatory and selected features
        selected_features = self.mandatory_features + selected_additional
        
        # Print all selected features with scores
        print(f"\nSelected {len(selected_features)} features:")
        for feat in selected_features:
            print(f"  {feat}: {feature_score_dict[feat]:.4f}")
        
        # Store scores
        self.feature_scores['random_forest'] = feature_score_dict
        
        return selected_features
    
    def visualize_feature_importance(self, selected_features):
        """
        Visualize feature importance scores
        
        Parameters:
        -----------
        selected_features : list
            List of selected feature names
        """
        if not self.feature_scores:
            print("No feature scores to visualize")
            return
        
        # Get the scores for the method used
        method_scores = self.feature_scores[self.method]
        
        # Get scores for selected features
        selected_scores = [(feat, method_scores.get(feat, 0)) for feat in selected_features]
        selected_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        features, scores = zip(*selected_scores)
        positions = range(len(features))
        
        # Create bar plot
        bars = plt.bar(positions, scores)
        
        # Color mandatory features differently
        for i, feat in enumerate(features):
            if feat in self.mandatory_features:
                bars[i].set_color('darkred')
            else:
                bars[i].set_color('steelblue')
        
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.title(f'{self.method.title()} Feature Importance - Top {self.n_features} Features', fontsize=14)
        plt.xticks(positions, features, rotation=45, ha='right')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkred', label='Mandatory (PL, RMS)'),
            Patch(facecolor='steelblue', label='Selected')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('data/features/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nFeature importance plot saved to: data/features/feature_importance.png")
    
    def select_features(self):
        """
        Main method to perform feature selection
        
        Returns:
        --------
        list : Selected feature names
        """
        # Load features
        df = self.load_features()
        
        # Prepare data
        X, y, feature_names = self.prepare_data(df)
        
        # Ensure mandatory features exist in the data
        missing_mandatory = [f for f in self.mandatory_features if f not in feature_names]
        if missing_mandatory:
            print(f"Warning: Mandatory features {missing_mandatory} not found in data!")
            self.mandatory_features = [f for f in self.mandatory_features if f in feature_names]
        
        # Apply selected method
        if self.method == 'lasso':
            selected_features = self.lasso_selection(X, y, feature_names)
        elif self.method == 'random_forest':
            selected_features = self.random_forest_selection(X, y, feature_names)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'lasso' or 'random_forest'")
        
        self.selected_features = selected_features
        
        # Visualize feature importance
        self.visualize_feature_importance(selected_features)
        
        # Save selected features
        self.save_selected_features(df, selected_features)
        
        return selected_features
    
    def save_selected_features(self, df, selected_features):
        """
        Save dataset with only selected features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original DataFrame with all features
        selected_features : list
            List of selected feature names
        """
        # Include targets and metadata in the output
        output_cols = self.target_cols + ['trajectory_id', 'step_id'] + selected_features
        
        # Ensure all columns exist
        output_cols = [col for col in output_cols if col in df.columns]
        
        # Create output DataFrame
        output_df = df[output_cols].copy()
        for col in selected_features:
            if col in output_df.columns:
                output_df[col] = output_df[col].round(2)
        
        # Save to CSV
        output_path = Path('data/features/features_selected.csv')
        output_df.to_csv(output_path, index=False)
        
        print(f"\n--- Feature Selection Complete ---")
        print(f"Selected {len(selected_features)} features")
        print(f"Saved to: {output_path}")
        print(f"Output shape: {output_df.shape}")
        
        # Print selected features
        print(f"\nSelected features:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i:2d}. {feature}")


def main(method='random_forest'):
    """
    Main function to perform feature selection
    
    Parameters:
    -----------
    method : str
        Feature selection method - 'lasso' or 'random_forest'
    """
    # Initialize selector with 7 features total (including PL and RMS)
    selector = FeatureSelector(target_cols=['X', 'Y'], n_features=7, method=method)
    
    # Perform feature selection
    selected_features = selector.select_features()
    
    return selected_features