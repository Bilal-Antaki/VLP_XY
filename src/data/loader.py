"""
Data loader for trajectory prediction project
Loads the FCPR-D1_CIR.csv file containing x, y, PL, and RMS values
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


class TrajectoryDataLoader:
    def __init__(self, data_path='data/processed/FCPR-D1_CIR.csv'):
        """
        Initialize the data loader
        
        Parameters:
        -----------
        data_path : str
            Relative path to the CSV file
        """
        # Convert to relative path and handle different OS path separators
        self.data_path = Path(data_path)
        
    def load_data(self):
        """
        Load the trajectory data from CSV file
        
        Returns:
        --------
        pd.DataFrame : DataFrame containing the loaded data
        """
        try:
            # Read the CSV file
            df = pd.read_csv(self.data_path)
            
            # Verify required columns exist
            required_columns = ['X', 'Y', 'PL', 'RMS']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"Successfully loaded data from {self.data_path}")
            print(f"Data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def split_trajectories(self, df, trajectories_per_set=10, train_trajectories=16):
        """
        Split data into trajectories and then into train/validation sets
        
        Parameters:
        -----------
        df : pd.DataFrame
            The loaded data
        trajectories_per_set : int
            Number of points per trajectory (default: 10)
        train_trajectories : int
            Number of trajectories for training (default: 16)
            
        Returns:
        --------
        tuple : (train_df, val_df, train_indices, val_indices)
        """
        total_points = len(df)
        total_trajectories = total_points // trajectories_per_set
        
        if total_points % trajectories_per_set != 0:
            print(f"Warning: Total points ({total_points}) not divisible by trajectory length ({trajectories_per_set})")
        
        # Create trajectory indices
        df['trajectory_id'] = np.repeat(range(total_trajectories), trajectories_per_set)[:total_points]
        df['step_id'] = np.tile(range(trajectories_per_set), total_trajectories)[:total_points]
        
        # Split into train and validation trajectories
        train_traj_ids = list(range(train_trajectories))
        val_traj_ids = list(range(train_trajectories, total_trajectories))
        
        train_df = df[df['trajectory_id'].isin(train_traj_ids)].copy()
        val_df = df[df['trajectory_id'].isin(val_traj_ids)].copy()
        
        print(f"\nTrajectory split:")
        print(f"Total trajectories: {total_trajectories}")
        print(f"Training trajectories: {len(train_traj_ids)} ({len(train_df)} points)")
        print(f"Validation trajectories: {len(val_traj_ids)} ({len(val_df)} points)")
        
        return train_df, val_df, train_traj_ids, val_traj_ids


def main():
    """
    Example usage of the TrajectoryDataLoader
    """
    # Initialize loader
    loader = TrajectoryDataLoader()
    
    # Load data
    df = loader.load_data()
    
    # Display basic statistics
    print("\nData statistics:")
    print(df[['X', 'Y', 'PL', 'RMS']].describe())
    
    # Split into trajectories
    train_df, val_df, train_ids, val_ids = loader.split_trajectories(df)
    
    return df, train_df, val_df


if __name__ == "__main__":
    df, train_df, val_df = main()