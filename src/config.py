"""
Configuration settings for the Position Estimation project
"""

# LSTM Model Configuration
MODEL_CONFIG = {
    'hidden_dim': 100,
    'num_layers': 3,
    'dropout': 0.3
}

# SVR Model Configuration
SVR_CONFIG = {
    'kernel': 'sigmoid',  # 'linear', 'poly', 'rbf', 'sigmoid'
    'C': 1.5,         # Regularization parameter
    'epsilon': 0.05,   # Epsilon-tube width
    'gamma': 'scale'  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
}

# Random Forest Configuration
RF_CONFIG = {
    'n_estimators': 150,
    'max_depth': None,
    'random_state': 42
}

# MLP Configuration
MLP_CONFIG = {
    'hidden_sizes': [128, 64, 32],  # Architecture: input -> 128 -> 64 -> 32 -> output
    'dropout': 0.3,
    'learning_rate': 0.01,
    'epochs': 300
}

# Training Configuration
TRAINING_CONFIG = {
    'learning_rate': 0.005,
    'batch_size': 10,
    'epochs': 300,
    'train_simulations': 19,
    'weight_decay': 1e-5,
    'validation_split': 0.2,
    'random_seed': 42,
    'test_size': 0.2,
    'validation_size': 0.2
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'correlation_threshold': 0.1,
    'simulation_length': 10,
    'base_features': ['PL', 'RMS']
}

# Data Processing Configuration
DATA_CONFIG = {
    'input_file': 'data/processed/FCPR-D1_CIR.csv',
    'target_column': 'r',
    'processed_dir': 'data/processed'
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'feature_selection': {
        'correlation_threshold': 0.3,
        'excluded_features': [
            'r', 'X', 'Y', 'source_file', 'radius', 'angle',
            'manhattan_dist', 'quadrant', 'X_Y_ratio', 'Y_X_ratio',
            'X_Y_product', 'X_normalized', 'Y_normalized'
        ]
    },
    'visualization': {
        'figure_sizes': {
            'data_exploration': (12, 5),
            'model_comparison': (17, 6)
        },
        'height_ratios': [1, 1],
        'scatter_alpha': 0.6,
        'scatter_size': 20,
        'grid_alpha': 0.3
    },
    'output': {
        'results_dir': 'results',
        'report_file': 'analysis_report.txt'
    }
}



