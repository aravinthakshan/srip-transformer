import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dataloader import T1Dataset, T2Dataset, T3Dataset
from lstm import T1LSTMModel, T2LSTMModel, T3LSTMModel

# --- CONFIGURATION ---
csv_path = "/home/aravinthakshan/Projects/main-srip/hierarchial/new_Handia_with_hierarchical_quantiles.csv"  # Path to your station CSV
features =  ['rainfall', 'tmin', 'tmax', 'waterlevel_final', 'streamflow_final',
            'Rain_cumulative_7d', 'Rain_cumulative_3d', 'hierarchical_feature_quantile_0.5',
            'waterlevel_upstream', 'streamflow_upstream']
 # <-- Fill with your feature column names
target = "streamflow_final"  # <-- Fill with your target column name
lookback = 7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_samples_to_show = 3

# --- LOAD DATA ---
df = pd.read_csv(csv_path, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# --- T1 DATASET INSPECTION ---
t1_ds = T1Dataset(df, features, target, lookback)
print("\n--- T1Dataset Inspection ---")
for i in range(num_samples_to_show):
    x, y = t1_ds[i]
    print(f"Sample {i}:")
    print("  Input (scaled):", x.numpy())
    print("  Target (scaled):", y.item())
    # Inverse scale input and target
    dummy = np.zeros((lookback, len(features)))
    dummy[:, :] = x.numpy()
    inv_x = scaler.inverse_transform(dummy)
    print("  Input (inverse):", inv_x)
    dummy_y = np.zeros((1, len(features)))
    dummy_y[0, features.index(target)] = y.item()
    inv_y = scaler.inverse_transform(dummy_y)[0, features.index(target)]
    print("  Target (inverse):", inv_y)

# --- T1 MODEL PREDICTION INSPECTION ---
t1_model = T1LSTMModel(input_size=len(features)).to(device)
t1_model.eval()
with torch.no_grad():
    for i in range(num_samples_to_show):
        x, y = t1_ds[i]
        x_in = x.unsqueeze(0).to(device)
        pred = t1_model(x_in).cpu().numpy()[0]
        print(f"\nT1 Sample {i} Prediction:")
        print("  Predicted (scaled):", pred)
        dummy_pred = np.zeros((1, len(features)))
        dummy_pred[0, features.index(target)] = pred
        inv_pred = scaler.inverse_transform(dummy_pred)[0, features.index(target)]
        print("  Predicted (inverse):", inv_pred)

# --- T2 DATASET INSPECTION (requires T1 predictions) ---
t1_preds = np.array([t1_model(x.unsqueeze(0).to(device)).cpu().numpy()[0] for x, _ in t1_ds])
t2_ds = T2Dataset(df, features, target, t1_preds, rating_curve_preds=None, lookback=lookback)
print("\n--- T2Dataset Inspection ---")
for i in range(num_samples_to_show):
    x, t1_pred, rating_pred, y = t2_ds[i]
    print(f"Sample {i}:")
    print("  Input (scaled):", x.numpy())
    print("  T1 pred (scaled):", t1_pred.item())
    print("  Target (scaled):", y.item())
    # Inverse scale input and target
    dummy = np.zeros((lookback, len(features)))
    dummy[:, :] = x.numpy()
    inv_x = scaler.inverse_transform(dummy)
    print("  Input (inverse):", inv_x)
    dummy_y = np.zeros((1, len(features)))
    dummy_y[0, features.index(target)] = y.item()
    inv_y = scaler.inverse_transform(dummy_y)[0, features.index(target)]
    print("  Target (inverse):", inv_y)
    dummy_t1 = np.zeros((1, len(features)))
    dummy_t1[0, features.index(target)] = t1_pred.item()
    inv_t1 = scaler.inverse_transform(dummy_t1)[0, features.index(target)]
    print("  T1 pred (inverse):", inv_t1)

# --- T3 DATASET INSPECTION (requires T1 and T2 predictions) ---
t2_preds = np.array([0.0 for _ in range(len(t2_ds))])  # Replace with actual T2 model predictions if available
t3_ds = T3Dataset(df, features, target, t1_preds, t2_preds, rating_curve_preds_t1=None, rating_curve_preds_t2=None, lookback=lookback)
print("\n--- T3Dataset Inspection ---")
for i in range(num_samples_to_show):
    x, t1_pred, t2_pred, rating_pred_t1, rating_pred_t2, y = t3_ds[i]
    print(f"Sample {i}:")
    print("  Input (scaled):", x.numpy())
    print("  T1 pred (scaled):", t1_pred.item())
    print("  T2 pred (scaled):", t2_pred.item())
    print("  Target (scaled):", y.item())
    # Inverse scale input and target
    dummy = np.zeros((lookback, len(features)))
    dummy[:, :] = x.numpy()
    inv_x = scaler.inverse_transform(dummy)
    print("  Input (inverse):", inv_x)
    dummy_y = np.zeros((1, len(features)))
    dummy_y[0, features.index(target)] = y.item()
    inv_y = scaler.inverse_transform(dummy_y)[0, features.index(target)]
    print("  Target (inverse):", inv_y)
    dummy_t1 = np.zeros((1, len(features)))
    dummy_t1[0, features.index(target)] = t1_pred.item()
    inv_t1 = scaler.inverse_transform(dummy_t1)[0, features.index(target)]
    print("  T1 pred (inverse):", inv_t1)
    dummy_t2 = np.zeros((1, len(features)))
    dummy_t2[0, features.index(target)] = t2_pred.item()
    inv_t2 = scaler.inverse_transform(dummy_t2)[0, features.index(target)]
    print("  T2 pred (inverse):", inv_t2) 