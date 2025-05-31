import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Get the absolute path of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (2 levels up from script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def load_and_preprocess_data(feature_file_path, train_split=0.8, sequence_length=10):
    """
    Load and preprocess trajectory data
    
    Parameters:
    -----------
    feature_file_path : str
        Path to the CSV file containing features
    train_split : float
        Fraction of data to use for training (default: 0.8)
    sequence_length : int
        Length of sequences to generate (default: 10)
        
    Returns:
    --------
    tuple
        (X_train, Y_train, X_val, Y_val) arrays
    """
    print(f"\nLoading data from: {feature_file_path}")
    
    # Load feature data
    try:
        df = pd.read_csv(feature_file_path)
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")
    
    print(f"Total records: {len(df)}")
    
    # Sort by trajectory and step
    df = df.sort_values(by=["trajectory_id", "step_id"]).reset_index(drop=True)
    
    # Feature columns (exclude x, y, trajectory_id, step_id)
    feature_cols = [col for col in df.columns if col not in ["X", "Y", "trajectory_id", "step_id"]]
    print(f"Number of features: {len(feature_cols)}")
    
    # Group by trajectory
    X_sequences = []
    Y_sequences = []
    trajectory_ids = sorted(df["trajectory_id"].unique())
    print(f"Number of trajectories: {len(trajectory_ids)}")
    
    for traj_id in trajectory_ids:
        traj_data = df[df["trajectory_id"] == traj_id]
        
        # Skip if trajectory is too short
        if len(traj_data) < sequence_length:
            continue
            
        # Create sequences from trajectory
        for i in range(len(traj_data) - sequence_length + 1):
            X_seq = traj_data[feature_cols].iloc[i:i+sequence_length].values
            Y_seq = traj_data[["X", "Y"]].iloc[i:i+sequence_length].values
            
            X_sequences.append(X_seq)
            Y_sequences.append(Y_seq)
    
    if not X_sequences:
        raise Exception("No valid sequences could be created from the data")
    
    X_sequences = np.array(X_sequences)
    Y_sequences = np.array(Y_sequences)
    
    print(f"Generated sequences shape: X={X_sequences.shape}, Y={Y_sequences.shape}")
    
    # Shuffle and split into train and validation sets
    indices = np.arange(len(X_sequences))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train = X_sequences[train_indices]
    Y_train = Y_sequences[train_indices]
    X_val = X_sequences[val_indices]
    Y_val = Y_sequences[val_indices]
    
    print(f"\nTraining set: {len(X_train)} sequences")
    print(f"Validation set: {len(X_val)} sequences")
    
    return X_train, Y_train, X_val, Y_val