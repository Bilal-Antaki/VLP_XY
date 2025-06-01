"""
Feature engineering for trajectory prediction
Generates feature interactions from PL and RMS values
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import loader
sys.path.append(str(Path(__file__).parent.parent))
from data.loader import TrajectoryDataLoader


class FeatureEngineer:
    def __init__(self):
        """Initialize the feature engineer"""
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_basic_features(self, df):
        """
        Create basic feature transformations
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with PL and RMS columns
            
        Returns:
        --------
        pd.DataFrame : DataFrame with additional basic features
        """
        features_df = df.copy()
        
        # Basic transformations
        features_df['PL_squared'] = df['PL'] ** 2
        features_df['RMS_squared'] = df['RMS'] ** 2
        features_df['PL_cubed'] = df['PL'] ** 3
        features_df['RMS_cubed'] = df['RMS'] ** 3
        
        # Logarithmic transformations (add small constant to avoid log(0))
        features_df['PL_log'] = np.log(df['PL'] + 1e-10)
        features_df['RMS_log'] = np.log(df['RMS'] + 1e-10)
        
        # Square root transformations
        features_df['PL_sqrt'] = np.sqrt(np.abs(df['PL']))
        features_df['RMS_sqrt'] = np.sqrt(np.abs(df['RMS']))
        
        # Reciprocal transformations (avoid division by zero)
        features_df['PL_reciprocal'] = 1 / (df['PL'] + 1e-10)
        features_df['RMS_reciprocal'] = 1 / (df['RMS'] + 1e-10)
        
        return features_df
    
    def create_interaction_features(self, df):
        """
        Create interaction features between PL and RMS
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with basic features
            
        Returns:
        --------
        pd.DataFrame : DataFrame with interaction features
        """
        features_df = df.copy()
        
        # Multiplicative interactions
        features_df['PL_RMS'] = df['PL'] * df['RMS']
        features_df['PL_squared_RMS'] = df['PL'] ** 2 * df['RMS']
        features_df['PL_RMS_squared'] = df['PL'] * df['RMS'] ** 2
        features_df['PL_squared_RMS_squared'] = df['PL'] ** 2 * df['RMS'] ** 2
        
        # Ratio features
        features_df['PL_RMS_ratio'] = df['PL'] / (df['RMS'] + 1e-10)
        features_df['RMS_PL_ratio'] = df['RMS'] / (df['PL'] + 1e-10)
        
        # Difference features
        features_df['PL_minus_RMS'] = df['PL'] - df['RMS']
        features_df['PL_plus_RMS'] = df['PL'] + df['RMS']
        features_df['abs_PL_minus_RMS'] = np.abs(df['PL'] - df['RMS'])
        
        # Complex interactions
        features_df['PL_RMS_harmonic_mean'] = 2 * df['PL'] * df['RMS'] / (df['PL'] + df['RMS'] + 1e-10)
        features_df['PL_RMS_geometric_mean'] = np.sqrt(np.abs(df['PL'] * df['RMS']))
        
        return features_df
    
    def create_temporal_features(self, df):
        """
        Create temporal features based on trajectory sequences
        FIXED: Ensures temporal features are computed within each trajectory independently
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with trajectory_id and step_id
            
        Returns:
        --------
        pd.DataFrame : DataFrame with temporal features
        """
        features_df = df.copy()
        
        # Sort by trajectory and step to ensure correct order
        features_df = features_df.sort_values(['trajectory_id', 'step_id']).reset_index(drop=True)
        
        # Process each trajectory independently to avoid data leakage
        processed_trajectories = []
        
        for traj_id in features_df['trajectory_id'].unique():
            traj_data = features_df[features_df['trajectory_id'] == traj_id].copy()
            traj_data = traj_data.sort_values('step_id').reset_index(drop=True)
            
            # Lag features (previous step) - within trajectory only
            for col in ['PL', 'RMS']:
                traj_data[f'{col}_lag1'] = traj_data[col].shift(1)
                traj_data[f'{col}_lag2'] = traj_data[col].shift(2)
                
                # Lead features (next step) - within trajectory only
                traj_data[f'{col}_lead1'] = traj_data[col].shift(-1)
                
                # Rolling statistics - within trajectory only
                traj_data[f'{col}_rolling_mean_3'] = traj_data[col].rolling(3, center=True, min_periods=1).mean()
                traj_data[f'{col}_rolling_std_3'] = traj_data[col].rolling(3, center=True, min_periods=1).std()
                
                # Differences - within trajectory only
                traj_data[f'{col}_diff'] = traj_data[col].diff()
                traj_data[f'{col}_diff2'] = traj_data[f'{col}_diff'].diff()
            
            # Fill NaN values with trajectory-specific methods
            # For lag features: forward fill within trajectory
            lag_cols = [col for col in traj_data.columns if 'lag' in col or 'lead' in col]
            for col in lag_cols:
                traj_data[col] = traj_data[col].ffill().bfill()
            
            # For rolling and diff features: fill with trajectory mean
            other_temporal_cols = [col for col in traj_data.columns 
                                 if any(x in col for x in ['rolling', 'diff']) and col not in lag_cols]
            for col in other_temporal_cols:
                traj_data[col] = traj_data[col].fillna(traj_data[col].mean()).fillna(0)
            
            processed_trajectories.append(traj_data)
        
        # Combine all processed trajectories
        features_df = pd.concat(processed_trajectories, ignore_index=True)
        features_df = features_df.sort_values(['trajectory_id', 'step_id']).reset_index(drop=True)
        
        return features_df
    
    def create_polynomial_features(self, df, degree=3):
        """
        Create polynomial features up to specified degree
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with PL and RMS
        degree : int
            Maximum polynomial degree
            
        Returns:
        --------
        pd.DataFrame : DataFrame with polynomial features
        """
        # Select only PL and RMS for polynomial features
        base_features = df[['PL', 'RMS']].values
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(base_features)
        
        # Get feature names
        feature_names = poly.get_feature_names_out(['PL', 'RMS'])
        
        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # Merge with original DataFrame
        result_df = pd.concat([df, poly_df], axis=1)
        
        # Remove duplicate columns
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        
        return result_df
    
    def engineer_all_features(self, df):
        """
        Create all engineered features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original DataFrame with X, Y, PL, RMS, trajectory_id, step_id
            
        Returns:
        --------
        pd.DataFrame : DataFrame with all engineered features
        """
        # Keep original columns
        features_df = df.copy()
        
        # Create basic features
        features_df = self.create_basic_features(features_df)
        
        # Create interaction features
        features_df = self.create_interaction_features(features_df)
        
        # Create temporal features
        #features_df = self.create_temporal_features(features_df)
        
        # Create polynomial features
        features_df = self.create_polynomial_features(features_df, degree=3)
        
        # Store feature names (excluding target and metadata columns)
        exclude_cols = ['X', 'Y', 'r', 'trajectory_id', 'step_id']
        self.feature_names = [col for col in features_df.columns if col not in exclude_cols]
        
        print(f"\nTotal features created: {len(self.feature_names)}")
        print(f"Feature names: {self.feature_names[:10]}... (showing first 10)")
        
        return features_df


def main():
    # Load data
    loader = TrajectoryDataLoader()
    df = loader.load_data()
    
    # Split into trajectories
    train_df, val_df, _, _ = loader.split_trajectories(df)
    
    # Combine train and val for feature engineering (will split later)
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.engineer_all_features(full_df)
    
    # Create output directory if it doesn't exist
    output_dir = Path('data/features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all features
    output_path = output_dir / 'features_all.csv'
    features_df.to_csv(output_path, index=False)
    print(f"\nSaved all features to: {output_path}")
    print(f"Shape: {features_df.shape}")
    
    return features_df, engineer.feature_names


if __name__ == "__main__":
    features_df, feature_names = main()