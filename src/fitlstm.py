import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import date, datetime
import argparse
import os
from tqdm import tqdm
import json
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from misc import quadratic, exponential_offset, shifted_quadratic, shifted_cubic
from metrics import nse, pbias, kge, evaluate
from dataloader import T1Dataset, T2Dataset, T3Dataset
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lstm import T1LSTMModel, T2LSTMModel, T3LSTMModel
from misc import create_run_directory, save_run_info

        
### ------------------------ evaluation and training scripts ------------------------ ###

def get_predictions(model, data_loader, device='cpu', use_rating_curve=False):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            try:
                if len(batch) == 2:  # T+1
                    x, _ = batch
                    pred = model(x.to(device))
                elif len(batch) == 4:  # T+2
                    x, t1_pred, rating_pred, _ = batch
                    if use_rating_curve:
                        pred = model(x.to(device), t1_pred.to(device), rating_pred.to(device))
                    else:
                        pred = model(x.to(device), t1_pred.to(device))
                elif len(batch) == 6:  # T+3
                    x, t1_pred, t2_pred, rating_t1, rating_t2, _ = batch
                    if use_rating_curve:
                        pred = model(x.to(device), t1_pred.to(device), t2_pred.to(device), 
                                    rating_t1.to(device), rating_t2.to(device))
                    else:
                        pred = model(x.to(device), t1_pred.to(device), t2_pred.to(device))
                else:
                    raise ValueError(f"Unexpected batch format. Got length {len(batch)}")
                
                predictions.append(pred.cpu().numpy())
                
            except Exception as e:
                print(f"Error in get_predictions: {e}")
                raise
    return np.concatenate(predictions)

def train_sequential_models(train_dfs, target, features, num_epochs=20, lr=1e-3, run_dir=None, 
                          lookback=7, rating_curve_fitter=None, waterlevel_col=None, scaler=None, station_name='unknown'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_streamflow = 'streamflow' in target.lower()
    use_rating_curve = is_streamflow
    if is_streamflow and (rating_curve_fitter is None or waterlevel_col is None):
        print("Error: Rating curve and waterlevel column are required for streamflow prediction.")
        return {}, {}
    models = {}
    train_metrics = {}
    train_losses = {'T+1': [], 'T+2': [], 'T+3': []}
    print("=== Training T+1 Model ===")
    t1_model = T1LSTMModel(input_size=len(features)).to(device)
    t1_criterion = nn.MSELoss()
    t1_optimizer = torch.optim.Adam(t1_model.parameters(), lr=lr)
    t1_train_ds = T1Dataset(train_dfs['train'], features, target, lookback)
    if len(t1_train_ds) == 0:
        print("T+1 training dataset is empty. Skipping training for T+1 and subsequent models.")
        return {}, {}
    t1_train_loader = DataLoader(t1_train_ds, batch_size=256, shuffle=False)
    best_t1_nse = -np.inf
    best_t1_state = None
    for epoch in tqdm(range(num_epochs), desc="Training T+1"):
        t1_model.train()
        epoch_loss = 0
        epoch_y_true, epoch_y_pred = [], []
        for xb, yb in t1_train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = t1_model(xb)
            loss = t1_criterion(pred, yb)
            t1_optimizer.zero_grad()
            loss.backward()
            t1_optimizer.step()
            epoch_loss += loss.item()
            epoch_y_true.append(yb.detach().cpu().numpy())
            epoch_y_pred.append(pred.detach().cpu().numpy())
        avg_loss = epoch_loss / len(t1_train_loader)
        train_losses['T+1'].append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            if epoch_y_true and epoch_y_pred:
                y_true = np.concatenate(epoch_y_true)
                y_pred = np.concatenate(epoch_y_pred)
                train_nse = nse(y_true, y_pred)
                if train_nse > best_t1_nse:
                    best_t1_nse = train_nse
                    best_t1_state = t1_model.state_dict().copy()
                print(f"T+1 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, NSE: {train_nse:.4f}")
    if best_t1_state is not None:
        t1_model.load_state_dict(best_t1_state)
    if epoch_y_true and epoch_y_pred:
        y_true = np.concatenate(epoch_y_true)
        y_pred = np.concatenate(epoch_y_pred)
        train_metrics['T+1'] = evaluate(y_true, y_pred)
    else:
        train_metrics['T+1'] = {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan}
    models['T+1'] = t1_model
    t1_train_preds = get_predictions(t1_model, t1_train_loader, device)
    if len(t1_train_preds) == 0:
        print("T+1 predictions for training T+2 are empty. Skipping T+2 and T+3 training.")
        return models, train_metrics
    rating_curve_preds_t1 = None
    if use_rating_curve:
        print("Generating rating curve predictions for T+2 training...")
        try:
            waterlevel_data = train_dfs['train'][waterlevel_col].iloc[lookback:lookback+len(t1_train_preds)].values
            rating_curve_preds_t1 = rating_curve_fitter.predict(waterlevel_data)
        except Exception as e:
            print(f"Warning: Could not generate rating curve predictions: {e}")
            use_rating_curve = False
    print("\n=== Training T+2 Model ===")
    t2_model = T2LSTMModel(input_size=len(features), use_rating_curve=use_rating_curve).to(device)
    t2_criterion = nn.MSELoss()
    t2_optimizer = torch.optim.Adam(t2_model.parameters(), lr=lr)
    t2_train_ds = T2Dataset(train_dfs['train'], features, target, t1_train_preds, 
                       rating_curve_preds_t1, lookback)
    if len(t2_train_ds) == 0:
        print("T+2 training dataset is empty. Skipping training for T+2 and T+3 models.")
        return models, train_metrics
    t2_train_loader = DataLoader(t2_train_ds, batch_size=256, shuffle=False)
    best_t2_nse = -np.inf
    best_t2_state = None
    for epoch in tqdm(range(num_epochs), desc="Training T+2"):
        t2_model.train()
        epoch_loss = 0
        epoch_y_true, epoch_y_pred = [], []
        for batch in t2_train_loader:
            if use_rating_curve:
                xb, t1_pred, rating_pred, yb = batch
                xb, t1_pred, rating_pred, yb = xb.to(device), t1_pred.to(device), rating_pred.to(device), yb.to(device)
                pred = t2_model(xb, t1_pred, rating_pred)
            else:
                xb, t1_pred, _, yb = batch  # Ignore the dummy rating prediction
                xb, t1_pred, yb = xb.to(device), t1_pred.to(device), yb.to(device)
                pred = t2_model(xb, t1_pred)
            loss = t2_criterion(pred, yb)
            t2_optimizer.zero_grad()
            loss.backward()
            t2_optimizer.step()
            epoch_loss += loss.item()
            epoch_y_true.append(yb.detach().cpu().numpy())
            epoch_y_pred.append(pred.detach().cpu().numpy())
        avg_loss = epoch_loss / len(t2_train_loader)
        train_losses['T+2'].append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            if epoch_y_true and epoch_y_pred:
                y_true = np.concatenate(epoch_y_true)
                y_pred = np.concatenate(epoch_y_pred)
                train_nse = nse(y_true, y_pred)
                if train_nse > best_t2_nse:
                    best_t2_nse = train_nse
                    best_t2_state = t2_model.state_dict().copy()
                print(f"T+2 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, NSE: {train_nse:.4f}")
    if best_t2_state is not None:
        t2_model.load_state_dict(best_t2_state)
    if epoch_y_true and epoch_y_pred:
        y_true = np.concatenate(epoch_y_true)
        y_pred = np.concatenate(epoch_y_pred)
        train_metrics['T+2'] = evaluate(y_true, y_pred)
    else:
        train_metrics['T+2'] = {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan}
    models['T+2'] = t2_model
    t2_train_preds = get_predictions(t2_model, t2_train_loader, device, use_rating_curve)
    if len(t2_train_preds) == 0:
        print("T+2 predictions for training T+3 are empty. Skipping T+3 training.")
        return models, train_metrics
    rating_curve_preds_t2 = None
    if use_rating_curve:
        print("Generating rating curve predictions for T+3 training...")
        try:
            waterlevel_data_t2 = train_dfs['train'][waterlevel_col].iloc[lookback+1:lookback+1+len(t2_train_preds)].values
            rating_curve_preds_t2 = rating_curve_fitter.predict(waterlevel_data_t2)
        except Exception as e:
            print(f"Warning: Could not generate T+2 rating curve predictions: {e}")
            rating_curve_preds_t2 = None
    print("\n=== Training T+3 Model ===")
    t3_model = T3LSTMModel(input_size=len(features), use_rating_curve=use_rating_curve).to(device)
    t3_criterion = nn.MSELoss()
    t3_optimizer = torch.optim.Adam(t3_model.parameters(), lr=lr)
    t3_train_ds = T3Dataset(train_dfs['train'], features, target, t1_train_preds, t2_train_preds,
                        rating_curve_preds_t1, rating_curve_preds_t2, lookback)
    if len(t3_train_ds) == 0:
        print("T+3 training dataset is empty. Skipping T+3 training.")
        return models, train_metrics
    t3_train_loader = DataLoader(t3_train_ds, batch_size=256, shuffle=False)
    best_t3_nse = -np.inf
    best_t3_state = None
    for epoch in tqdm(range(num_epochs), desc="Training T+3"):
        t3_model.train()
        epoch_loss = 0
        epoch_y_true, epoch_y_pred = [], []
        for batch in t3_train_loader:
            if use_rating_curve:
                xb, t1_pred, t2_pred, rating_t1, rating_t2, yb = batch
                xb = xb.to(device)
                t1_pred, t2_pred = t1_pred.to(device), t2_pred.to(device)
                rating_t1, rating_t2 = rating_t1.to(device), rating_t2.to(device)
                yb = yb.to(device)
                pred = t3_model(xb, t1_pred, t2_pred, rating_t1, rating_t2)
            else:
                xb, t1_pred, t2_pred, _, _, yb = batch  # Ignore dummy rating predictions
                xb = xb.to(device)
                t1_pred, t2_pred = t1_pred.to(device), t2_pred.to(device)
                yb = yb.to(device)
                pred = t3_model(xb, t1_pred, t2_pred)
            loss = t3_criterion(pred, yb)
            t3_optimizer.zero_grad()
            loss.backward()
            t3_optimizer.step()
            epoch_loss += loss.item()
            epoch_y_true.append(yb.detach().cpu().numpy())
            epoch_y_pred.append(pred.detach().cpu().numpy())
        avg_loss = epoch_loss / len(t3_train_loader)
        train_losses['T+3'].append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            if epoch_y_true and epoch_y_pred:
                y_true = np.concatenate(epoch_y_true)
                y_pred = np.concatenate(epoch_y_pred)
                train_nse = nse(y_true, y_pred)
                if train_nse > best_t3_nse:
                    best_t3_nse = train_nse
                    best_t3_state = t3_model.state_dict().copy()
                print(f"T+3 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, NSE: {train_nse:.4f}")
    if best_t3_state is not None:
        t3_model.load_state_dict(best_t3_state)
    if epoch_y_true and epoch_y_pred:
        y_true = np.concatenate(epoch_y_true)
        y_pred = np.concatenate(epoch_y_pred)
        train_metrics['T+3'] = evaluate(y_true, y_pred)
    else:
        train_metrics['T+3'] = {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan}
    models['T+3'] = t3_model
    if run_dir:
        for horizon, model in models.items():
            torch.save(model.state_dict(), os.path.join(run_dir, f"{horizon.replace('+', '')}_model.pth"))
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(train_losses['T+1'])
        plt.title('Training Loss - T+1 LSTM')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.subplot(1, 3, 2)
        plt.plot(train_losses['T+2'])
        plt.title('Training Loss - T+2 LSTM')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.subplot(1, 3, 3)
        plt.plot(train_losses['T+3'])
        plt.title('Training Loss - T+3 LSTM')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'training_losses_sequential.png'), dpi=300, bbox_inches='tight')
        plt.close()
    print("Training completed!")
    # Save train predictions to CSV (date, predicted, ground_truth, horizon)
    if scaler is not None and run_dir is not None:
        output_csv_dir = "output_csv"
        os.makedirs(output_csv_dir, exist_ok=True)
        train_df = train_dfs['train']
        all_predictions = []
        all_ground_truth = []
        all_horizons = []
        all_dates = []
        try:
            target_idx = features.index(target)
        except ValueError:
            target_idx = None
        # T+1
        t1_train_ds = T1Dataset(train_df, features, target, lookback)
        if len(t1_train_ds) > 0 and target_idx is not None:
            t1_train_loader = DataLoader(t1_train_ds, batch_size=256, shuffle=False)
            t1_train_preds_raw = get_predictions(models['T+1'], t1_train_loader, device)
            y_trues_raw_t1 = train_df[target].iloc[lookback:lookback+len(t1_train_preds_raw)].values
            dummy_true_t1 = np.zeros((len(y_trues_raw_t1), len(features)))
            dummy_pred_t1 = np.zeros((len(t1_train_preds_raw), len(features)))
            dummy_true_t1[:, target_idx] = y_trues_raw_t1
            dummy_pred_t1[:, target_idx] = t1_train_preds_raw
            y_trues_inv_t1 = scaler.inverse_transform(dummy_true_t1)[:, target_idx]
            y_preds_inv_t1 = scaler.inverse_transform(dummy_pred_t1)[:, target_idx]
            all_predictions.extend(y_preds_inv_t1)
            all_ground_truth.extend(y_trues_inv_t1)
            all_horizons.extend(['T+1'] * len(y_preds_inv_t1))
            dates_t1 = train_df['date'].iloc[lookback:lookback+len(t1_train_preds_raw)].values
            all_dates.extend(dates_t1)
        # T+2
        if 'T+2' in models and target_idx is not None:
            t2_train_ds = T2Dataset(train_df, features, target, t1_train_preds, rating_curve_preds_t1, lookback)
            if len(t2_train_ds) > 0:
                t2_train_loader = DataLoader(t2_train_ds, batch_size=256, shuffle=False)
                t2_train_preds_raw = get_predictions(models['T+2'], t2_train_loader, device, use_rating_curve)
                y_trues_raw_t2 = train_df[target].iloc[lookback+1:lookback+1+len(t2_train_preds_raw)].values
                dummy_true_t2 = np.zeros((len(y_trues_raw_t2), len(features)))
                dummy_pred_t2 = np.zeros((len(t2_train_preds_raw), len(features)))
                dummy_true_t2[:, target_idx] = y_trues_raw_t2
                dummy_pred_t2[:, target_idx] = t2_train_preds_raw
                y_trues_inv_t2 = scaler.inverse_transform(dummy_true_t2)[:, target_idx]
                y_preds_inv_t2 = scaler.inverse_transform(dummy_pred_t2)[:, target_idx]
                all_predictions.extend(y_preds_inv_t2)
                all_ground_truth.extend(y_trues_inv_t2)
                all_horizons.extend(['T+2'] * len(y_preds_inv_t2))
                dates_t2 = train_df['date'].iloc[lookback+1:lookback+1+len(t2_train_preds_raw)].values
                all_dates.extend(dates_t2)
        # T+3
        if 'T+3' in models and target_idx is not None:
            t3_train_ds = T3Dataset(train_df, features, target, t1_train_preds, t2_train_preds,
                                rating_curve_preds_t1, rating_curve_preds_t2, lookback)
            if len(t3_train_ds) > 0:
                t3_train_loader = DataLoader(t3_train_ds, batch_size=256, shuffle=False)
                t3_train_preds_raw = get_predictions(models['T+3'], t3_train_loader, device, use_rating_curve)
                y_trues_raw_t3 = train_df[target].iloc[lookback+2:lookback+2+len(t3_train_preds_raw)].values
                dummy_true_t3 = np.zeros((len(y_trues_raw_t3), len(features)))
                dummy_pred_t3 = np.zeros((len(t3_train_preds_raw), len(features)))
                dummy_true_t3[:, target_idx] = y_trues_raw_t3
                dummy_pred_t3[:, target_idx] = t3_train_preds_raw
                y_trues_inv_t3 = scaler.inverse_transform(dummy_true_t3)[:, target_idx]
                y_preds_inv_t3 = scaler.inverse_transform(dummy_pred_t3)[:, target_idx]
                all_predictions.extend(y_preds_inv_t3)
                all_ground_truth.extend(y_trues_inv_t3)
                all_horizons.extend(['T+3'] * len(y_preds_inv_t3))
                dates_t3 = train_df['date'].iloc[lookback+2:lookback+2+len(t3_train_preds_raw)].values
                all_dates.extend(dates_t3)
        if len(all_predictions) > 0:
            predictions_df = pd.DataFrame({
                'date': all_dates,
                'horizon': all_horizons,
                'predicted': all_predictions,
                'ground_truth': all_ground_truth,
                'dataset': ['train'] * len(all_predictions)
            })
            csv_path = os.path.join(output_csv_dir, f"predictions_train_{station_name}_{target}.csv")
            predictions_df.to_csv(csv_path, index=False)
            print(f"Train predictions saved to: {csv_path}")
    return models, train_metrics


import os
import pandas as pd

def evaluate_sequential_models(models, test_df, lookback, scaler, target, features, run_dir=None, 
                             dataset_name="", type=None, rating_curve_fitter=None, waterlevel_col=None, station_name='unknown'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_streamflow = 'streamflow' in target.lower()
    use_rating_curve = is_streamflow  # Add this line here
    
    # Create output_csv directory if it doesn't exist
    output_csv_dir = "output_csv"
    os.makedirs(output_csv_dir, exist_ok=True)
    
    # Generate unique CSV filename
    csv_counter = 1
    csv_filename = f"predictions_{csv_counter}_{station_name}_{target}.csv"
    while os.path.exists(os.path.join(output_csv_dir, csv_filename)):
        csv_counter += 1
        csv_filename = f"predictions_{csv_counter}_{station_name}_{target}.csv"
    
    if is_streamflow and (rating_curve_fitter is None or waterlevel_col is None):
        print(f"Error: Rating curve and waterlevel column are required for streamflow evaluation in {dataset_name}.")
        return horizon_metrics, top10_nse, horizon_nse
    for model in models.values():
        model.to(device)
        model.eval()
    horizon_metrics = {f'T+{i+1}': {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan} for i in range(3)}
    horizon_nse = [np.nan, np.nan, np.nan]
    top10_nse = [np.nan, np.nan, np.nan]
    y_trues_inv = [np.array([]), np.array([]), np.array([])]
    y_preds_inv = [np.array([]), np.array([]), np.array([])]
    dates = np.array([])
    
    # Lists to store all predictions and ground truth for CSV output
    all_predictions = []
    all_ground_truth = []
    all_horizons = []
    all_dates = []
    
    try:
        target_idx = features.index(target)
    except ValueError:
        print(f"Error: Target variable '{target}' not found in the features list used for scaling. Cannot evaluate.")
        return horizon_metrics, top10_nse, horizon_nse
    t1_test_ds = T1Dataset(test_df, features, target, lookback)
    if len(t1_test_ds) > 0:
        t1_test_loader = DataLoader(t1_test_ds, batch_size=256, shuffle=False)
        t1_test_preds_raw = get_predictions(models['T+1'], t1_test_loader, device)
        y_trues_raw_t1 = test_df[target].iloc[lookback:lookback+len(t1_test_preds_raw)].values
        if len(y_trues_raw_t1) > 0:
            dummy_true_t1 = np.zeros((len(y_trues_raw_t1), len(features)))
            dummy_pred_t1 = np.zeros((len(t1_test_preds_raw), len(features)))
            dummy_true_t1[:, target_idx] = y_trues_raw_t1
            dummy_pred_t1[:, target_idx] = t1_test_preds_raw
            y_trues_inv[0] = scaler.inverse_transform(dummy_true_t1)[:, target_idx]
            y_preds_inv[0] = scaler.inverse_transform(dummy_pred_t1)[:, target_idx]
            
            # Store for CSV output - T+1
            all_predictions.extend(y_preds_inv[0])
            all_ground_truth.extend(y_trues_inv[0])
            all_horizons.extend(['T+1'] * len(y_preds_inv[0]))
            dates_t1 = test_df['date'].iloc[lookback:lookback+len(t1_test_preds_raw)].values
            all_dates.extend(dates_t1)
            
            horizon_metrics['T+1'] = evaluate(y_trues_inv[0], y_preds_inv[0])
            horizon_nse[0] = horizon_metrics['T+1']['NSE']
            print(f"NSE for T+1 ({dataset_name}): {horizon_nse[0]:.4f}")
            if len(y_trues_inv[0]) > 0:
                threshold_t1 = np.percentile(y_trues_inv[0], 90)
                high_idx_t1 = y_trues_inv[0] >= threshold_t1
                if np.sum(high_idx_t1) > 0:
                    top10_nse[0] = nse(y_trues_inv[0][high_idx_t1], y_preds_inv[0][high_idx_t1])
                    print(f"Top 10% NSE for T+1 ({dataset_name}): {top10_nse[0]:.4f}")
                else:
                    print(f"No values found in top 10% for T+1 ({dataset_name}).")
            dates = test_df['date'].iloc[lookback:lookback+len(t1_test_preds_raw)].values
        else:
            print(f"Not enough data to evaluate T+1 for {dataset_name}.")
    else:
        print(f"T+1 test dataset is empty for {dataset_name}.")
    rating_curve_preds_t1 = None
    if use_rating_curve and len(t1_test_preds_raw) > 0:
        try:
            waterlevel_data = test_df[waterlevel_col].iloc[lookback:lookback+len(t1_test_preds_raw)].values
            rating_curve_preds_t1 = rating_curve_fitter.predict(waterlevel_data)
        except Exception as e:
            print(f"Warning: Could not generate T+1 rating curve predictions: {e}")
            use_rating_curve = False
    if len(t1_test_preds_raw) > 0:
        t2_test_ds = T2Dataset(test_df, features, target, t1_test_preds_raw, rating_curve_preds_t1, lookback)
        if len(t2_test_ds) > 0:
            t2_test_loader = DataLoader(t2_test_ds, batch_size=256, shuffle=False)
            t2_test_preds_raw = get_predictions(models['T+2'], t2_test_loader, device, use_rating_curve)
            y_trues_raw_t2 = test_df[target].iloc[lookback+1:lookback+1+len(t2_test_preds_raw)].values
            if len(y_trues_raw_t2) > 0:
                dummy_true_t2 = np.zeros((len(y_trues_raw_t2), len(features)))
                dummy_pred_t2 = np.zeros((len(t2_test_preds_raw), len(features)))
                dummy_true_t2[:, target_idx] = y_trues_raw_t2
                dummy_pred_t2[:, target_idx] = t2_test_preds_raw
                y_trues_inv[1] = scaler.inverse_transform(dummy_true_t2)[:, target_idx]
                y_preds_inv[1] = scaler.inverse_transform(dummy_pred_t2)[:, target_idx]
                
                # Store for CSV output - T+2
                all_predictions.extend(y_preds_inv[1])
                all_ground_truth.extend(y_trues_inv[1])
                all_horizons.extend(['T+2'] * len(y_preds_inv[1]))
                dates_t2 = test_df['date'].iloc[lookback+1:lookback+1+len(t2_test_preds_raw)].values
                all_dates.extend(dates_t2)
                
                horizon_metrics['T+2'] = evaluate(y_trues_inv[1], y_preds_inv[1])
                horizon_nse[1] = horizon_metrics['T+2']['NSE']
                print(f"NSE for T+2 ({dataset_name}): {horizon_nse[1]:.4f}")
                if len(y_trues_inv[1]) > 0:
                    threshold_t2 = np.percentile(y_trues_inv[1], 90)
                    high_idx_t2 = y_trues_inv[1] >= threshold_t2
                    if np.sum(high_idx_t2) > 0:
                        top10_nse[1] = nse(y_trues_inv[1][high_idx_t2], y_preds_inv[1][high_idx_t2])
                        print(f"Top 10% NSE for T+2 ({dataset_name}): {top10_nse[1]:.4f}")
                    else:
                        print(f"No values found in top 10% for T+2 ({dataset_name}).")
            else:
                print(f"Not enough data to evaluate T+2 for {dataset_name}.")
        else:
            print(f"T+2 test dataset is empty for {dataset_name}.")
    else:
        print(f"T+1 predictions were not available for T+2 evaluation for {dataset_name}.")
    rating_curve_preds_t2 = None
    if use_rating_curve and 't2_test_preds_raw' in locals() and len(t2_test_preds_raw) > 0:
        try:
            waterlevel_data_t2 = test_df[waterlevel_col].iloc[lookback+1:lookback+1+len(t2_test_preds_raw)].values
            rating_curve_preds_t2 = rating_curve_fitter.predict(waterlevel_data_t2)
        except Exception as e:
            print(f"Warning: Could not generate T+2 rating curve predictions: {e}")
            rating_curve_preds_t2 = None
    if 't2_test_preds_raw' in locals() and len(t2_test_preds_raw) > 0:
        t3_test_ds = T3Dataset(test_df, features, target, t1_test_preds_raw, t2_test_preds_raw, 
                             rating_curve_preds_t1, rating_curve_preds_t2, lookback)
        if len(t3_test_ds) > 0:
            t3_test_loader = DataLoader(t3_test_ds, batch_size=256, shuffle=False)
            t3_test_preds_raw = get_predictions(models['T+3'], t3_test_loader, device, use_rating_curve)
            y_trues_raw_t3 = test_df[target].iloc[lookback+2:lookback+2+len(t3_test_preds_raw)].values
            if len(y_trues_raw_t3) > 0:
                dummy_true_t3 = np.zeros((len(y_trues_raw_t3), len(features)))
                dummy_pred_t3 = np.zeros((len(t3_test_preds_raw), len(features)))
                dummy_true_t3[:, target_idx] = y_trues_raw_t3
                dummy_pred_t3[:, target_idx] = t3_test_preds_raw
                y_trues_inv[2] = scaler.inverse_transform(dummy_true_t3)[:, target_idx]
                y_preds_inv[2] = scaler.inverse_transform(dummy_pred_t3)[:, target_idx]
                
                # Store for CSV output - T+3
                all_predictions.extend(y_preds_inv[2])
                all_ground_truth.extend(y_trues_inv[2])
                all_horizons.extend(['T+3'] * len(y_preds_inv[2]))
                dates_t3 = test_df['date'].iloc[lookback+2:lookback+2+len(t3_test_preds_raw)].values
                all_dates.extend(dates_t3)
                
                horizon_metrics['T+3'] = evaluate(y_trues_inv[2], y_preds_inv[2])
                horizon_nse[2] = horizon_metrics['T+3']['NSE']
                print(f"NSE for T+3 ({dataset_name}): {horizon_nse[2]:.4f}")
                if len(y_trues_inv[2]) > 0:
                    threshold_t3 = np.percentile(y_trues_inv[2], 90)
                    high_idx_t3 = y_trues_inv[2] >= threshold_t3
                    if np.sum(high_idx_t3) > 0:
                        top10_nse[2] = nse(y_trues_inv[2][high_idx_t3], y_preds_inv[2][high_idx_t3])
                        print(f"Top 10% NSE for T+3 ({dataset_name}): {top10_nse[2]:.4f}")
                    else:
                        print(f"No values found in top 10% for T+3 ({dataset_name}).")
            else:
                print(f"Not enough data to evaluate T+3 for {dataset_name}.")
        else:
            print(f"T+3 test dataset is empty for {dataset_name}.")
    else:
        print(f"T+2 predictions were not available for T+3 evaluation for {dataset_name}.")
    
    # Save predictions and ground truth to CSV
    if len(all_predictions) > 0:
        predictions_df = pd.DataFrame({
            'date': all_dates,
            'horizon': all_horizons,
            'predicted': all_predictions,
            'ground_truth': all_ground_truth,
            'dataset': [dataset_name] * len(all_predictions)
        })
        
        csv_path = os.path.join(output_csv_dir, csv_filename)
        predictions_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")
        
        # Update or create the mapping file
        mapping_file = os.path.join(output_csv_dir, "csv_mapping.txt")
        with open(mapping_file, 'a') as f:
            f.write(f"{csv_filename}, {run_dir if run_dir else 'None'}\n")
        print(f"Mapping updated in: {mapping_file}")
    else:
        print(f"No predictions to save for {dataset_name}")
    
    print(f"\nDetailed Metrics for {dataset_name}:")
    for horizon, metrics in horizon_metrics.items():
        print(f"{horizon}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")
    if run_dir and len(dates) > 0 and len(y_trues_inv[0]) > 0:
        plt.figure(figsize=(18, 12))
        for i in range(3):
            if len(y_trues_inv[i]) > 0:
                plt.subplot(3, 1, i + 1)
                plot_dates_current_horizon = dates[:len(y_trues_inv[i])]
                plt.plot(plot_dates_current_horizon, y_trues_inv[i], label='Observed', color='red', linewidth=1.0)
                plt.plot(plot_dates_current_horizon, y_preds_inv[i], label='Predicted', color='blue', linewidth=1.0)
                if type == "streamflow":
                    plt.title(f'Sequential Streamflow Prediction at T+{i+1} - {dataset_name}')
                    plt.ylabel('Streamflow')
                else:
                    plt.title(f'Sequential Waterlevel Prediction at T+{i+1} - {dataset_name}')
                    plt.ylabel('Waterlevel')
                plt.xlabel('Date')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                print(f"Skipping plot for T+{i+1} as data is empty for {dataset_name}.")
        if plt.get_fignums():
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f'sequential_predictions_{dataset_name.lower().replace(" ", "_")}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print(f"No plots generated for {dataset_name} due to lack of valid data.")
    elif run_dir:
        print(f"Skipping plot generation for {dataset_name} as there's no sufficient data or valid dates.")
    return horizon_metrics, top10_nse, horizon_nse

def main():
    import numpy as np
    import pandas as pd
    import torch
    import argparse
    import os
    from sklearn.preprocessing import MinMaxScaler
    from misc import create_run_directory, save_run_info
    from fitlstm import train_sequential_models, evaluate_sequential_models
    from misc import RatingCurveFitter
    from misc import create_run_directory, save_run_info
    np.random.seed(74)
    torch.manual_seed(74)


    if torch.cuda.is_available():
        torch.cuda.manual_seed(74)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser(description='Sequential Multi-LSTM Streamflow Prediction')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--desc', type=str, required=True, help='Description of the run')
    parser.add_argument('--station_file', type=str, required=True, help='Path to the station CSV file')
    parser.add_argument('--target_variable', type=str, required=True, help='Target variable for prediction')
    parser.add_argument('--features', type=str, required=True, help='Comma-separated list of features')
    parser.add_argument('--output_base_dir', type=str, required=True, help='Base directory for outputs')
    parser.add_argument('--station_name', type=str, default='unknown', help='Station name for output file naming')
    args = parser.parse_args()
    features = [f.strip() for f in args.features.split(',')]
    target = args.target_variable
    run_dir = create_run_directory(args.desc, args.output_base_dir)
    print(f"Created run directory: {run_dir}")
    try:
        df = pd.read_csv(args.station_file, parse_dates=['date'], dayfirst=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['year'] = df['date'].dt.year
    except FileNotFoundError:
        print(f"Error: Station file not found at {args.station_file}. Exiting.")
        return
    except KeyError as e:
        print(f"Error: Required column missing in {args.station_file}: {e}. Ensure 'date', '{target}' and all specified features are present.")
        return
    except Exception as e:
        print(f"An error occurred while loading or processing {args.station_file}: {e}")
        return
    required_cols = features + [target]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in {args.station_file}: {', '.join(missing_cols)}. Exiting.")
        return
    type = target.split("_")[0]
    lookback = 7
    first_10_mask = (df['year'] >= 1961) & (df['year'] <= 1970)
    last_10_mask = (df['year'] >= 2011) & (df['year'] <= 2020)
    middle_mask = (df['year'] > 1970) & (df['year'] < 2011)
    train_df = df[middle_mask].reset_index(drop=True)
    first_10_df = df[first_10_mask].reset_index(drop=True)
    last_10_df = df[last_10_mask].reset_index(drop=True)
    if train_df.empty:
        print(f"Warning: Training dataframe is empty for {args.station_file}. Cannot train model. Exiting.")
        save_run_info(run_dir, args.desc, args,
                      {'T+1': {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan},
                       'T+2': {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan},
                       'T+3': {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan}},
                      {'first_10': {}, 'last_10': {}, 'first_10_top10_nse': [np.nan, np.nan, np.nan], 'last_10_top10_nse': [np.nan, np.nan, np.nan]},
                      type)
        print(f"\nPlaceholder run information saved to: {os.path.join(run_dir, 'run_info.txt')}")
        return
    available_features = [f for f in features if f in train_df.columns]
    if len(available_features) != len(features):
        print(f"Warning: Some requested features ({set(features) - set(available_features)}) are not in the training data. Using available features.")
        features = available_features
        if not features:
            print("Error: No valid features found in the dataframe. Exiting.")
            return
    scaler = MinMaxScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    if not first_10_df.empty:
        missing_test_cols = [col for col in features if col not in first_10_df.columns]
        if missing_test_cols:
            print(f"Warning: Missing features in first_10_df: {', '.join(missing_test_cols)}. Skipping scaling for these.")
            first_10_df = first_10_df.drop(columns=missing_test_cols, errors='ignore')
        if not first_10_df.empty and all(f in first_10_df.columns for f in features):
            first_10_df[features] = scaler.transform(first_10_df[features])
        else:
            print(f"Warning: First 10 years test dataframe for {args.station_file} is empty or missing features after preprocessing. Metrics will be skipped.")
            first_10_df = pd.DataFrame()
    if not last_10_df.empty:
        missing_test_cols = [col for col in features if col not in last_10_df.columns]
        if missing_test_cols:
            print(f"Warning: Missing features in last_10_df: {', '.join(missing_test_cols)}. Skipping scaling for these.")
            last_10_df = last_10_df.drop(columns=missing_test_cols, errors='ignore')
        if not last_10_df.empty and all(f in last_10_df.columns for f in features):
            last_10_df[features] = scaler.transform(last_10_df[features])
        else:
            print(f"Warning: Last 10 years test dataframe for {args.station_file} is empty or missing features after preprocessing. Metrics will be skipped.")
            last_10_df = pd.DataFrame()
    train_dfs = {'train': train_df}
    rating_curve_fitter = None
    waterlevel_col = None
    rating_curve_info = None
    if type == "streamflow":
        if 'waterlevel_final' not in train_df.columns or 'streamflow_final' not in train_df.columns:
            print("Error: 'waterlevel_final' and 'streamflow_final' must be present for streamflow prediction. Exiting.")
            return
        print("Fitting rating curve...")
        rating_curve_fitter = RatingCurveFitter()
        waterlevel_data = train_df['waterlevel_final'].values
        streamflow_data = train_df['streamflow_final'].values
        if not rating_curve_fitter.fit(waterlevel_data, streamflow_data):
            print("Error: Rating curve fitting failed. Exiting.")
            return
        rating_curve_fitter.save(os.path.join(run_dir, 'rating_curve.pkl'))
        rating_curve_info = {
            'name': rating_curve_fitter.best_model['name'],
            'params': rating_curve_fitter.best_model['params'].tolist(),
            'RMSE': rating_curve_fitter.best_model['RMSE'],
            'R2': rating_curve_fitter.best_model['R2']
        }
        waterlevel_col = 'waterlevel_final'
    else:
        rating_curve_fitter = None
        waterlevel_col = None
    print("=== Starting Sequential Model Training ===")
    station_name = args.station_name
    models, train_metrics = train_sequential_models(
        train_dfs, target, features,
        num_epochs=args.epochs, run_dir=run_dir, lookback=lookback,
        rating_curve_fitter=rating_curve_fitter, waterlevel_col=waterlevel_col, scaler=scaler,
        station_name=station_name
    )
    for horizon, model in models.items():
        model_path = os.path.join(run_dir, f"best_sequential_model_{horizon.lower().replace('+', '')}.pt")
        try:
            torch.save(model.state_dict(), model_path)
            print(f"Sequential Model {horizon} saved to: {model_path}")
        except Exception as e:
            print(f"Warning: Could not save model {horizon} to {model_path}: {e}")
    test_metrics = {
        'first_10': {'T+1': {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan},
                     'T+2': {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan},
                     'T+3': {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan}},
        'last_10': {'T+1': {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan},
                    'T+2': {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan},
                    'T+3': {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan}},
        'first_10_top10_nse': [np.nan, np.nan, np.nan],
        'last_10_top10_nse': [np.nan, np.nan, np.nan],
        'first_10_horizon_nse': [np.nan, np.nan, np.nan],
        'last_10_horizon_nse': [np.nan, np.nan, np.nan]
    }
    if 'T+1' in models and models['T+1'] is not None:
        if not first_10_df.empty:
            print("\n=== Evaluating Sequential Models on First 10 Years ===")
            first_10_metrics, first_10_top10_nse, first_10_horizon_nse = evaluate_sequential_models(
                models, first_10_df, lookback, scaler, target, features,
                run_dir, "First 10 Years", type, rating_curve_fitter, waterlevel_col, station_name
            )
            test_metrics['first_10'] = first_10_metrics
            test_metrics['first_10_top10_nse'] = first_10_top10_nse
            test_metrics['first_10_horizon_nse'] = first_10_horizon_nse
        else:
            print("Skipping evaluation for First 10 Years due to empty or invalid data.")
        if not last_10_df.empty:
            print("\n=== Evaluating Sequential Models on Last 10 Years ===")
            last_10_metrics, last_10_top10_nse, last_10_horizon_nse = evaluate_sequential_models(
                models, last_10_df, lookback, scaler, target, features,
                run_dir, "Last 10 Years", type, rating_curve_fitter, waterlevel_col, station_name
            )
            test_metrics['last_10'] = last_10_metrics
            test_metrics['last_10_top10_nse'] = last_10_top10_nse
            test_metrics['last_10_horizon_nse'] = last_10_horizon_nse
        else:
            print("Skipping evaluation for Last 10 Years due to empty or invalid data.")
    else:
        print("Skipping all evaluation steps as T+1 model was not trained successfully.")
    save_run_info(run_dir, args.desc, args, train_metrics, test_metrics, type, rating_curve_info)
    # Combine all prediction CSVs
    combine_train_test_csvs(target, run_dir, station_name)
    print(f"\nSequential run information saved to: {os.path.join(run_dir, 'run_info.txt')}")
    print("\n" + "="*60)
    print("SEQUENTIAL MULTI-LSTM TRAINING COMPLETE")
    print("="*60)
    print(f"Run Directory: {run_dir}")
    print(f"Target Variable: {target}")
    print(f"Lookback Window: {lookback}")
    print(f"Training Epochs: {args.epochs}")
    print(f"Features Used: {len(features)}")
    print("\nFinal Training Metrics:")
    for horizon, metrics in train_metrics.items():
        print(f"    {horizon}: NSE = {metrics['NSE']:.4f}, R2 = {metrics['R2']:.4f}")
    print("\nTest Performance Summary:")
    print("First 10 Years NSE:")
    for i, nse_val in enumerate(test_metrics['first_10_horizon_nse']):
        print(f"    T+{i+1}: {nse_val:.4f}")
    print("Last 10 Years NSE:")
    for i, nse_val in enumerate(test_metrics['last_10_horizon_nse']):
        print(f"    T+{i+1}: {nse_val:.4f}")

def combine_train_test_csvs(target, run_dir, station_name='unknown'):
    import glob
    import pandas as pd
    output_csv_dir = "output_csv"
    pattern = os.path.join(output_csv_dir, f"predictions_*_{station_name}_{target}.csv")
    csv_files = glob.glob(pattern)
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    if dfs:
        combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        combined_csv_path = os.path.join(output_csv_dir, f"predictions_combined_{station_name}_{target}.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined predictions saved to: {combined_csv_path}")
    else:
        print("No prediction CSVs found to combine.")

if __name__ == "__main__":
    main()