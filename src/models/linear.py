import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('data/features/features_selected.csv')

# Features and targets
feature_cols = ['PL', 'RMS', 'PL_minus_RMS', 'abs_PL_minus_RMS',
                'RMS_rolling_mean_3', 'PL_rolling_mean_3', 'RMS_lag2']
target_cols = ['X', 'Y']

X_features = df[feature_cols]
y_targets = df[target_cols]

# Train-validation split by trajectory_id
train_ids = df['trajectory_id'].unique()[:16]
val_ids = df['trajectory_id'].unique()[16:]

train_df = df[df['trajectory_id'].isin(train_ids)]
val_df = df[df['trajectory_id'].isin(val_ids)]

X_train = train_df[feature_cols]
y_train = train_df[target_cols]

X_val = val_df[feature_cols]
y_val = val_df[target_cols]

# Fit separate models for X and Y
model_X = LinearRegression()
model_Y = LinearRegression()

model_X.fit(X_train, y_train['X'])
model_Y.fit(X_train, y_train['Y'])

# Predict on validation set
pred_X = model_X.predict(X_val)
pred_Y = model_Y.predict(X_val)

# Evaluate using RMSE
rmse_X = np.sqrt(mean_squared_error(y_val['X'], pred_X))
rmse_Y = np.sqrt(mean_squared_error(y_val['Y'], pred_Y))

print(f"Validation RMSE - X: {rmse_X:.2f}, Y: {rmse_Y:.2f}")
