"""
Improved preprocessing for trajectory prediction
Handles sequences properly without data leakage
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

# Get the absolute path of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (2 levels up from script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def load_and_preprocess_data(feature_file_path, train_trajectories=16, approach='full_trajectory'):
    """
    Load and preprocess trajectory data with multiple approach options
    
    Parameters:
    -----------
    feature_file_path : str
        Path to the CSV file containing features
    train_trajectories : int
        Number of trajectories to use for training (default: 16 out of 20)
    approach : str
        'full_trajectory': Use complete 10-step trajectories
        'next_step': Predict next step given history
        'sliding_window': Create sequences with sliding window (non-overlapping between train/val)
        
    Returns:
    --------
    dict containing different data formats based on approach
    """
    print(f"\nLoading data from: {feature_file_path}")
    print(f"Preprocessing approach: {approach}")
    
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
    print(f"Features used: {feature_cols}")
    
    # Get unique trajectories
    trajectory_ids = sorted(df["trajectory_id"].unique())
    n_trajectories = len(trajectory_ids)
    print(f"Number of trajectories: {n_trajectories}")
    
    # Split trajectories into train and validation
    train_traj_ids = trajectory_ids[:train_trajectories]
    val_traj_ids = trajectory_ids[train_trajectories:]
    
    print(f"Training trajectories: {len(train_traj_ids)}")
    print(f"Validation trajectories: {len(val_traj_ids)}")
    
    if approach == 'full_trajectory':
        return preprocess_full_trajectory(df, feature_cols, train_traj_ids, val_traj_ids)
    elif approach == 'next_step':
        return preprocess_next_step(df, feature_cols, train_traj_ids, val_traj_ids)
    elif approach == 'sliding_window':
        return preprocess_sliding_window(df, feature_cols, train_traj_ids, val_traj_ids, window_size=5)
    else:
        raise ValueError(f"Unknown approach: {approach}")


def preprocess_full_trajectory(df, feature_cols, train_traj_ids, val_traj_ids):
    """
    Use complete trajectories as sequences
    """
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    
    # Process training trajectories
    for traj_id in train_traj_ids:
        traj_data = df[df["trajectory_id"] == traj_id].sort_values("step_id")
        if len(traj_data) == 10:  # Ensure complete trajectory
            X_train.append(traj_data[feature_cols].values)
            Y_train.append(traj_data[["X", "Y"]].values)
    
    # Process validation trajectories
    for traj_id in val_traj_ids:
        traj_data = df[df["trajectory_id"] == traj_id].sort_values("step_id")
        if len(traj_data) == 10:  # Ensure complete trajectory
            X_val.append(traj_data[feature_cols].values)
            Y_val.append(traj_data[["X", "Y"]].values)
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    
    print(f"\nFull trajectory approach:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    
    return X_train, Y_train, X_val, Y_val


def preprocess_next_step(df, feature_cols, train_traj_ids, val_traj_ids, history_len=3):
    """
    Predict next position given history of previous positions
    """
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    
    # Process training trajectories
    for traj_id in train_traj_ids:
        traj_data = df[df["trajectory_id"] == traj_id].sort_values("step_id")
        
        # Create sequences for next-step prediction
        for i in range(history_len, len(traj_data)):
            # Use history_len previous steps to predict current step
            X_seq = traj_data[feature_cols].iloc[i-history_len:i].values
            Y_target = traj_data[["X", "Y"]].iloc[i].values
            
            X_train.append(X_seq)
            Y_train.append(Y_target)
    
    # Process validation trajectories
    for traj_id in val_traj_ids:
        traj_data = df[df["trajectory_id"] == traj_id].sort_values("step_id")
        
        for i in range(history_len, len(traj_data)):
            X_seq = traj_data[feature_cols].iloc[i-history_len:i].values
            Y_target = traj_data[["X", "Y"]].iloc[i].values
            
            X_val.append(X_seq)
            Y_val.append(Y_target)
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    
    print(f"\nNext-step prediction approach:")
    print(f"Training: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Validation: X={X_val.shape}, Y={Y_val.shape}")
    
    return X_train, Y_train, X_val, Y_val


def preprocess_sliding_window(df, feature_cols, train_traj_ids, val_traj_ids, window_size=5):
    """
    Create non-overlapping sequences within trajectories
    """
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    
    # Process training trajectories
    for traj_id in train_traj_ids:
        traj_data = df[df["trajectory_id"] == traj_id].sort_values("step_id")
        
        # Create non-overlapping windows
        for i in range(0, len(traj_data) - window_size + 1, window_size//2):
            X_seq = traj_data[feature_cols].iloc[i:i+window_size].values
            Y_seq = traj_data[["X", "Y"]].iloc[i:i+window_size].values
            
            if len(X_seq) == window_size:  # Ensure complete window
                X_train.append(X_seq)
                Y_train.append(Y_seq)
    
    # Process validation trajectories
    for traj_id in val_traj_ids:
        traj_data = df[df["trajectory_id"] == traj_id].sort_values("step_id")
        
        for i in range(0, len(traj_data) - window_size + 1, window_size//2):
            X_seq = traj_data[feature_cols].iloc[i:i+window_size].values
            Y_seq = traj_data[["X", "Y"]].iloc[i:i+window_size].values
            
            if len(X_seq) == window_size:
                X_val.append(X_seq)
                Y_val.append(Y_seq)
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    
    print(f"\nSliding window approach (window_size={window_size}):")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    
    return X_train, Y_train, X_val, Y_val


def create_lagged_features(df, feature_cols, n_lags=2):
    """
    Create lagged features for better temporal modeling
    """
    df_lagged = df.copy()
    
    # Create lagged features
    for lag in range(1, n_lags + 1):
        for col in ['X', 'Y']:
            df_lagged[f'{col}_lag{lag}'] = df_lagged.groupby('trajectory_id')[col].shift(lag)
    
    # Drop rows with NaN values from lagging
    df_lagged = df_lagged.groupby('trajectory_id').apply(
        lambda x: x.iloc[n_lags:] if len(x) > n_lags else x.iloc[0:0]
    ).reset_index(drop=True)
    
    return df_lagged


def normalize_by_trajectory(X_train, Y_train, X_val, Y_val):
    """
    Normalize features with trajectory-aware scaling
    """
    # Flatten for normalization but keep track of shapes
    train_shape = X_train.shape
    val_shape = X_val.shape
    
    # Normalize features
    feature_scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    
    X_train_scaled = feature_scaler.fit_transform(X_train_flat).reshape(train_shape)
    X_val_scaled = feature_scaler.transform(X_val_flat).reshape(val_shape)
    
    # Normalize targets
    target_scaler = StandardScaler()
    if Y_train.ndim == 3:  # Full sequences
        Y_train_flat = Y_train.reshape(-1, Y_train.shape[-1])
        Y_val_flat = Y_val.reshape(-1, Y_val.shape[-1])
        Y_train_scaled = target_scaler.fit_transform(Y_train_flat).reshape(Y_train.shape)
        Y_val_scaled = target_scaler.transform(Y_val_flat).reshape(Y_val.shape)
    else:  # Single targets
        Y_train_scaled = target_scaler.fit_transform(Y_train)
        Y_val_scaled = target_scaler.transform(Y_val)
    
    return (X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled), (feature_scaler, target_scaler)


def prepare_training_data(features_df, selected_features):
    """
    Prepare data for training using selected features
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing all features
    selected_features : list
        List of selected feature names
        
    Returns
    -------
    tuple
        (X_train, Y_train, X_val, Y_val)
    """
    # Get feature columns (excluding target and ID columns)
    feature_cols = [col for col in features_df.columns 
                   if col not in ['X', 'Y', 'trajectory_id', 'step_id']]
    
    # Ensure all selected features exist in the data
    missing_features = [f for f in selected_features if f not in feature_cols]
    if missing_features:
        raise ValueError(f"Selected features not found in data: {missing_features}")
    
    # Prepare data
    X = features_df[selected_features].values
    Y = features_df[['X', 'Y']].values
    
    # Split into training and validation sets
    # Use trajectory_id for splitting to keep trajectories together
    train_trajectories = features_df['trajectory_id'].unique()[:16]  # First 16 trajectories for training
    val_trajectories = features_df['trajectory_id'].unique()[16:]    # Last 4 trajectories for validation
    
    # Split data
    train_mask = features_df['trajectory_id'].isin(train_trajectories)
    val_mask = features_df['trajectory_id'].isin(val_trajectories)
    
    X_train = X[train_mask]
    Y_train = Y[train_mask]
    X_val = X[val_mask]
    Y_val = Y[val_mask]
    
    print(f"\nData split:")
    print(f"Training set: {X_train.shape[0]} samples from {len(train_trajectories)} trajectories")
    print(f"Validation set: {X_val.shape[0]} samples from {len(val_trajectories)} trajectories")
    
    return X_train, Y_train, X_val, Y_val