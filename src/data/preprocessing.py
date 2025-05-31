import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Get the absolute path of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (2 levels up from script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def load_and_preprocess_data(feature_file_path, train_trajectories=16, sequence_length=10):
    # Load feature data
    df = pd.read_csv(feature_file_path)

    # Sort by trajectory and step
    df = df.sort_values(by=["trajectory_id", "step_id"]).reset_index(drop=True)

    # Feature columns (exclude x, y, trajectory_id, step_id)
    feature_cols = [col for col in df.columns if col not in ["X", "Y", "trajectory_id", "step_id"]]

    # Group by trajectory
    X_sequences = []
    Y_sequences = []
    trajectory_ids = sorted(df["trajectory_id"].unique())

    for traj_id in trajectory_ids:
        traj_data = df[df["trajectory_id"] == traj_id]
        X_seq = traj_data[feature_cols].values
        Y_seq = traj_data[["X", "Y"]].values

        if len(X_seq) == sequence_length:
            X_sequences.append(X_seq)
            Y_sequences.append(Y_seq)

    X_sequences = np.array(X_sequences)  # Shape: (20, 10, n_features)
    Y_sequences = np.array(Y_sequences)  # Shape: (20, 10, 2)

    # Shuffle and split into train and validation sets
    indices = np.arange(len(X_sequences))
    np.random.shuffle(indices)

    X_sequences = X_sequences[indices]
    Y_sequences = Y_sequences[indices]

    X_train = X_sequences[:train_trajectories]
    Y_train = Y_sequences[:train_trajectories]
    X_val = X_sequences[train_trajectories:]
    Y_val = Y_sequences[train_trajectories:]

    return X_train, Y_train, X_val, Y_val


if __name__ == "__main__":
    feature_path = os.path.join(PROJECT_ROOT, "data", "features", "features_selected.csv")
    X_train, Y_train, X_val, Y_val = load_and_preprocess_data(feature_path)