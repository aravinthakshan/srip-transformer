import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import date, datetime
import argparse
import os
from tqdm import tqdm
import json
import math

# Metrics Functions
def nse(y_true, y_pred):
    if np.all(y_true == np.mean(y_true)):
        return -np.inf
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def pbias(y_true, y_pred):
    if np.sum(y_true) == 0:
        return np.nan
    return 100 * (np.sum(y_true - y_pred) / np.sum(y_true))

def kge(y_true, y_pred):
    if len(y_true) < 2 or len(y_pred) < 2:
        return np.nan

    r_val = np.corrcoef(y_true, y_pred)
    if r_val.shape != (2, 2):
        r = np.nan
    else:
        r = r_val[0, 1]

    std_y_true = np.std(y_true)
    std_y_pred = np.std(y_pred)

    if std_y_true == 0:
        alpha = np.nan
    else:
        alpha = std_y_pred / std_y_true

    mean_y_true = np.mean(y_true)
    mean_y_pred = np.mean(y_pred)

    if mean_y_true == 0:
        beta = np.nan
    else:
        beta = mean_y_pred / mean_y_true

    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return np.nan

    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def evaluate(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0 or np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return {
            'NSE': np.nan,
            'R2': np.nan,
            'PBIAS': np.nan,
            'KGE': np.nan
        }
    return {
        'NSE': nse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'PBIAS': pbias(y_true, y_pred),
        'KGE': kge(y_true, y_pred)
    }

# # Dataset Classes
# class T1Dataset(Dataset):
#     def __init__(self, data, input_cols, target_col, lookback=30):
#         self.X = data[input_cols].values
#         self.y = data[target_col].values
#         self.lookback = lookback

#     def __len__(self):
#         if len(self.X) <= self.lookback + 1:
#              return 0
#         return len(self.X) - self.lookback - 1

#     def __getitem__(self, idx):
#         x = self.X[idx : idx + self.lookback]
#         y = self.y[idx + self.lookback]
#         return (torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

# class T2Dataset(Dataset):
#     def __init__(self, data, input_cols, target_col, t1_predictions, lookback=30):
#         self.X = data[input_cols].values
#         self.y = data[target_col].values
#         self.t1_preds = t1_predictions
#         self.lookback = lookback

#     def __len__(self):
#         min_len = min(len(self.X) - self.lookback - 1, len(self.t1_preds) - 1)
#         if min_len <= 0:
#             return 0
#         return min_len

#     def __getitem__(self, idx):
#         x = self.X[idx : idx + self.lookback]
#         t1_pred = self.t1_preds[idx]
#         y = self.y[idx + self.lookback + 1]
#         return (torch.tensor(x, dtype=torch.float32), torch.tensor(t1_pred, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

# class T3Dataset(Dataset):
#     def __init__(self, data, input_cols, target_col, t1_predictions, t2_predictions, lookback=30):
#         self.X = data[input_cols].values
#         self.y = data[target_col].values
#         self.t1_preds = t1_predictions
#         self.t2_preds = t2_predictions
#         self.lookback = lookback

#     def __len__(self):
#         min_len = min(len(self.X) - self.lookback - 2, len(self.t2_preds) - 1)
#         if min_len <= 0:
#             return 0
#         return min_len

#     def __getitem__(self, idx):
#         x = self.X[idx : idx + self.lookback]
#         t1_pred = self.t1_preds[idx]
#         t2_pred = self.t2_preds[idx]
#         y = self.y[idx + self.lookback + 2]
#         return (torch.tensor(x, dtype=torch.float32),
#                 torch.tensor(t1_pred, dtype=torch.float32),
#                 torch.tensor(t2_pred, dtype=torch.float32),
#                 torch.tensor(y, dtype=torch.float32))

# Dataset Classes - FIXED VERSION
class T1Dataset(Dataset):
    def __init__(self, data, input_cols, target_col, lookback=30):
        self.X = data[input_cols].values
        self.y = data[target_col].values
        self.lookback = lookback

    def __len__(self):
        # Need at least lookback+1 points to create one sample
        # (lookback points for input + 1 point for T+1 target)
        if len(self.X) <= self.lookback:
             return 0
        return len(self.X) - self.lookback

    def __getitem__(self, idx):
        # Input: from T-lookback to T-1 (excluding T)
        # Target: T+1
        x = self.X[idx : idx + self.lookback]  # T-lookback to T-1
        y = self.y[idx + self.lookback]        # T+1
        return (torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

class T2Dataset(Dataset):
    def __init__(self, data, input_cols, target_col, t1_predictions, lookback=30):
        self.X = data[input_cols].values
        self.y = data[target_col].values
        self.t1_preds = t1_predictions
        self.lookback = lookback

    def __len__(self):
        # Need at least lookback+2 points to create one sample
        # (lookback points for input + 1 for T+1 pred + 1 for T+2 target)
        min_len = min(len(self.X) - self.lookback - 1, len(self.t1_preds))
        if min_len <= 0:
            return 0
        return min_len

    def __getitem__(self, idx):
        # Input: from T-lookback to T-1 (excluding T)
        # T+1 prediction: corresponds to time T+1
        # Target: T+2
        x = self.X[idx : idx + self.lookback]     # T-lookback to T-1
        t1_pred = self.t1_preds[idx]              # T+1 prediction
        y = self.y[idx + self.lookback + 1]       # T+2
        return (torch.tensor(x, dtype=torch.float32), 
                torch.tensor(t1_pred, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.float32))

class T3Dataset(Dataset):
    def __init__(self, data, input_cols, target_col, t1_predictions, t2_predictions, lookback=30):
        self.X = data[input_cols].values
        self.y = data[target_col].values
        self.t1_preds = t1_predictions
        self.t2_preds = t2_predictions
        self.lookback = lookback

    def __len__(self):
        # Need at least lookback+3 points to create one sample
        # (lookback points for input + 1 for T+1 pred + 1 for T+2 pred + 1 for T+3 target)
        min_len = min(len(self.X) - self.lookback - 2, len(self.t2_preds))
        if min_len <= 0:
            return 0
        return min_len

    def __getitem__(self, idx):
        # Input: from T-lookback to T-1 (excluding T)
        # T+1 prediction: corresponds to time T+1
        # T+2 prediction: corresponds to time T+2
        # Target: T+3
        x = self.X[idx : idx + self.lookback]      # T-lookback to T-1
        t1_pred = self.t1_preds[idx]               # T+1 prediction
        t2_pred = self.t2_preds[idx]               # T+2 prediction
        y = self.y[idx + self.lookback + 2]        # T+3
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(t1_pred, dtype=torch.float32),
                torch.tensor(t2_pred, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))
    
# Transformer-LSTM Model Classes
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class T1TransformerLSTMModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, transformer_layers=2, lstm_hidden=32, num_layers=1):
        super().__init__()
        
        # Input projection to transformer dimension
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder for contextual understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
        # LSTM for sequential processing of enriched features
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        
        # Final prediction layer
        self.fc = nn.Linear(lstm_hidden, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Project input to transformer dimension
        x_proj = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x_proj = self.pos_encoder(x_proj.transpose(0, 1)).transpose(0, 1)
        
        # Create contextual encodings with transformer
        contextual_encoding = self.transformer_encoder(x_proj)  # [batch, seq_len, d_model]
        
        # Process contextual encodings with LSTM
        lstm_out, _ = self.lstm(contextual_encoding)
        lstm_out = self.dropout(lstm_out)
        
        # Final prediction using last timestep
        return self.fc(lstm_out[:, -1, :]).squeeze(-1)

class T2TransformerLSTMModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, transformer_layers=2, lstm_hidden=32, num_layers=1):
        super().__init__()
        
        # Input projection to transformer dimension
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder for contextual understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        
        # Combine LSTM output with T+1 prediction
        self.fc1 = nn.Linear(lstm_hidden + 1, lstm_hidden // 2)
        self.fc2 = nn.Linear(lstm_hidden // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, t1_pred):
        # Project input to transformer dimension
        x_proj = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x_proj = self.pos_encoder(x_proj.transpose(0, 1)).transpose(0, 1)
        
        # Create contextual encodings with transformer
        contextual_encoding = self.transformer_encoder(x_proj)  # [batch, seq_len, d_model]
        
        # Process contextual encodings with LSTM
        lstm_out, _ = self.lstm(contextual_encoding)
        lstm_out = self.dropout(lstm_out)
        
        # Combine with T+1 prediction
        combined = torch.cat([lstm_out[:, -1, :], t1_pred.unsqueeze(-1)], dim=-1)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        return self.fc2(out).squeeze(-1)

class T3TransformerLSTMModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, transformer_layers=2, lstm_hidden=32, num_layers=1):
        super().__init__()
        
        # Input projection to transformer dimension
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder for contextual understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        
        # Combine LSTM output with T+1 and T+2 predictions
        self.fc1 = nn.Linear(lstm_hidden + 2, lstm_hidden // 2)
        self.fc2 = nn.Linear(lstm_hidden // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, t1_pred, t2_pred):
        # Project input to transformer dimension
        x_proj = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x_proj = self.pos_encoder(x_proj.transpose(0, 1)).transpose(0, 1)
        
        # Create contextual encodings with transformer
        contextual_encoding = self.transformer_encoder(x_proj)  # [batch, seq_len, d_model]
        
        # Process contextual encodings with LSTM
        lstm_out, _ = self.lstm(contextual_encoding)
        lstm_out = self.dropout(lstm_out)
        
        # Combine with T+1 and T+2 predictions
        combined = torch.cat([lstm_out[:, -1, :], t1_pred.unsqueeze(-1), t2_pred.unsqueeze(-1)], dim=-1)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        return self.fc2(out).squeeze(-1)

# Utility Functions
def create_run_directory(description, output_base_dir):
    base_dir = output_base_dir
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    plot_nums = []
    for d in existing_dirs:
        if d.startswith("plot"):
            try:
                num_str = d.split('_')[0].replace("plot", "")
                num = int(num_str)
                plot_nums.append(num)
            except ValueError:
                continue

    next_num = max(plot_nums) + 1 if plot_nums else 1
    run_dir = os.path.join(base_dir, f"plot{next_num}_{description}")
    os.makedirs(run_dir)

    return run_dir

def save_run_info(run_dir, description, args, train_metrics, test_metrics, type=None):
    info_file = os.path.join(run_dir, "run_info.txt")
    with open(info_file, 'w') as f:
        f.write("=" * 50 + "\n")
        if type == "streamflow":
            f.write("TRANSFORMER-LSTM STREAMFLOW PREDICTION RUN SUMMARY\n")
        else:
            f.write("TRANSFORMER-LSTM WATERLEVEL PREDICTION RUN SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Run Description: {description}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Epochs: {args.epochs}\n")
        f.write("Architecture: Transformer-LSTM Hybrid models\n")
        f.write("    - T+1: Transformer Encoder + LSTM\n")
        f.write("    - T+2: Transformer Encoder + LSTM + T+1 prediction\n")
        f.write("    - T+3: Transformer Encoder + LSTM + T+1 + T+2 predictions\n\n")
        f.write(f"Station File: {args.station_file}\n")
        f.write(f"Target Variable: {args.target_variable}\n")
        f.write(f"Features: {args.features}\n\n")

        f.write("TRAINING METRICS (Final Epoch):\n")
        f.write("-" * 30 + "\n")
        for horizon, metrics in train_metrics.items():
            f.write(f"Transformer-LSTM {horizon}:\n")
            for metric, value in metrics.items():
                f.write(f"    {metric}: {value:.4f}\n")
            f.write("\n")

        f.write("TEST METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write("First 10 Years (1961-1970):\n")
        if test_metrics['first_10']:
            for horizon in ['T+1', 'T+2', 'T+3']:
                f.write(f"{horizon} Metrics:\n")
                for metric, value in test_metrics['first_10'].get(horizon, {}).items():
                    f.write(f"    {metric}: {value:.4f}\n")
        else:
            f.write("    No data for First 10 Years.\n")

        f.write("\nLast 10 Years (2011-2020):\n")
        if test_metrics['last_10']:
            for horizon in ['T+1', 'T+2', 'T+3']:
                f.write(f"{horizon} Metrics:\n")
                for metric, value in test_metrics['last_10'].get(horizon, {}).items():
                    f.write(f"    {metric}: {value:.4f}\n")
        else:
            f.write("    No data for Last 10 Years.\n")

        f.write("\nTOP 10% NSE SCORES:\n")
        f.write("-" * 30 + "\n")
        f.write("First 10 Years:\n")
        for i, nse_val in enumerate(test_metrics['first_10_top10_nse']):
            f.write(f"    T+{i+1}: {nse_val:.4f}\n")

        f.write("\nLast 10 Years:\n")
        for i, nse_val in enumerate(test_metrics['last_10_top10_nse']):
            f.write(f"    T+{i+1}: {nse_val:.4f}\n")

def get_predictions(model, data_loader, device='cpu'):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            try:
                if len(batch) == 2:  # T1Dataset (x, y)
                    x, _ = batch
                    pred = model(x.to(device))
                elif len(batch) == 3:  # T2Dataset (x, t1_pred, y)
                    x, t1_pred, _ = batch
                    pred = model(x.to(device), t1_pred.to(device))
                elif len(batch) == 4:  # T3Dataset (x, t1_pred, t2_pred, y)
                    x, t1_pred, t2_pred, _ = batch
                    pred = model(x.to(device), t1_pred.to(device), t2_pred.to(device))
                else:
                    raise ValueError(f"Unexpected batch format. Expected tuple of length 2, 3, or 4, but got length {len(batch)}")
                
                predictions.append(pred.cpu().numpy())
            except Exception as e:
                print(f"Error in get_predictions: {e}")
                print(f"Batch type: {type(batch)}, Batch length: {len(batch) if hasattr(batch, '__len__') else 'No length'}")
                raise
    
    return np.concatenate(predictions)

# Training Function
def train_sequential_transformer_models(train_dfs, target, features, num_epochs=20, lr=1e-5, run_dir=None, lookback=7): # changed lr note
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = {}
    train_metrics = {}
    
    # Calculate appropriate transformer dimensions based on sequence length and features
    input_size = len(features)
    
    # Adaptive sizing based on sequence length and input features
    if lookback <= 7:
        d_model = 64
        nhead = 4
        transformer_layers = 2
        lstm_hidden = 32
    elif lookback <= 30:
        d_model = 128
        nhead = 8
        transformer_layers = 3
        lstm_hidden = 64
    else:
        d_model = 256
        nhead = 8
        transformer_layers = 4
        lstm_hidden = 128
    
    print(f"Using Transformer config: d_model={d_model}, nhead={nhead}, layers={transformer_layers}, lstm_hidden={lstm_hidden}")
    print(f"Sequence length: {lookback}, Input features: {input_size}")

    print("=== Training T+1 Transformer-LSTM Model ===")
    t1_model = T1TransformerLSTMModel(
        input_size=input_size, 
        d_model=d_model, 
        nhead=nhead, 
        transformer_layers=transformer_layers, 
        lstm_hidden=lstm_hidden
    ).to(device)
    
    t1_criterion = nn.MSELoss()
    t1_optimizer = torch.optim.Adam(t1_model.parameters(), lr=lr, weight_decay=1e-5)
    t1_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(t1_optimizer, patience=5, factor=0.5)

    t1_train_ds = T1Dataset(train_dfs['train'], features, target, lookback)
    if len(t1_train_ds) == 0:
        print("T+1 training dataset is empty. Skipping training for T+1 and subsequent models.")
        return {}, {}
    
    t1_train_loader = DataLoader(t1_train_ds, batch_size=128, shuffle=True)
    t1_train_losses = []
    best_t1_nse = -np.inf
    best_t1_state = None

    for epoch in tqdm(range(num_epochs), desc="Training T+1 Transformer-LSTM"):
        t1_model.train()
        epoch_loss = 0
        epoch_y_true, epoch_y_pred = [], []

        for xb, yb in t1_train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = t1_model(xb)
            loss = t1_criterion(pred, yb)
            t1_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(t1_model.parameters(), max_norm=1.0)
            t1_optimizer.step()

            epoch_loss += loss.item()
            epoch_y_true.append(yb.detach().cpu().numpy())
            epoch_y_pred.append(pred.detach().cpu().numpy())

        avg_loss = epoch_loss / len(t1_train_loader)
        t1_train_losses.append(avg_loss)
        t1_scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            if epoch_y_true and epoch_y_pred:
                y_true = np.concatenate(epoch_y_true)
                y_pred = np.concatenate(epoch_y_pred)
                train_nse = nse(y_true, y_pred)

                if train_nse > best_t1_nse:
                    best_t1_nse = train_nse
                    best_t1_state = t1_model.state_dict().copy()

                print(f"T+1 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, NSE: {train_nse:.4f}, LR: {t1_optimizer.param_groups[0]['lr']:.6f}")
            else:
                print(f"T+1 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, No valid data for NSE calculation.")

    if best_t1_state is not None:
        t1_model.load_state_dict(best_t1_state)
    else:
        print("Warning: No best T+1 model state saved. Using last epoch's model.")
    
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

    print("\n=== Training T+2 Transformer-LSTM Model ===")
    t2_model = T2TransformerLSTMModel(
        input_size=input_size, 
        d_model=d_model, 
        nhead=nhead, 
        transformer_layers=transformer_layers, 
        lstm_hidden=lstm_hidden
    ).to(device)
    
    t2_criterion = nn.MSELoss()
    t2_optimizer = torch.optim.Adam(t2_model.parameters(), lr=lr, weight_decay=1e-5)
    t2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(t2_optimizer, patience=5, factor=0.5)
    
    t2_train_ds = T2Dataset(train_dfs['train'], features, target, t1_train_preds, lookback)
    if len(t2_train_ds) == 0:
        print("T+2 training dataset is empty. Skipping training for T+2 and T+3 models.")
        return models, train_metrics
    
    t2_train_loader = DataLoader(t2_train_ds, batch_size=128, shuffle=True)
    t2_train_losses = []
    best_t2_nse = -np.inf
    best_t2_state = None

    for epoch in tqdm(range(num_epochs), desc="Training T+2 Transformer-LSTM"):
        t2_model.train()
        epoch_loss = 0
        epoch_y_true, epoch_y_pred = [], []

        for xb, t1_pred, yb in t2_train_loader:
            xb, t1_pred, yb = xb.to(device), t1_pred.to(device), yb.to(device)
            pred = t2_model(xb, t1_pred)
            loss = t2_criterion(pred, yb)
            t2_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(t2_model.parameters(), max_norm=1.0)
            t2_optimizer.step()

            epoch_loss += loss.item()
            epoch_y_true.append(yb.detach().cpu().numpy())
            epoch_y_pred.append(pred.detach().cpu().numpy())

        avg_loss = epoch_loss / len(t2_train_loader)
        t2_train_losses.append(avg_loss)
        t2_scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            if epoch_y_true and epoch_y_pred:
                y_true = np.concatenate(epoch_y_true)
                y_pred = np.concatenate(epoch_y_pred)
                train_nse = nse(y_true, y_pred)

                if train_nse > best_t2_nse:
                    best_t2_nse = train_nse
                    best_t2_state = t2_model.state_dict().copy()

                print(f"T+2 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, NSE: {train_nse:.4f}, LR: {t2_optimizer.param_groups[0]['lr']:.6f}")
            else:
                print(f"T+2 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, No valid data for NSE calculation.")

    if best_t2_state is not None:
        t2_model.load_state_dict(best_t2_state)
    else:
        print("Warning: No best T+2 model state saved. Using last epoch's model.")
    
    if epoch_y_true and epoch_y_pred:
        y_true = np.concatenate(epoch_y_true)
        y_pred = np.concatenate(epoch_y_pred)
        train_metrics['T+2'] = evaluate(y_true, y_pred)
    else:
        train_metrics['T+2'] = {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan}
    models['T+2'] = t2_model

    t2_train_preds = get_predictions(t2_model, t2_train_loader, device)
    if len(t2_train_preds) == 0:
        print("T+2 predictions for training T+3 are empty. Skipping T+3 training.")
        return models, train_metrics

    print("\n=== Training T+3 Transformer-LSTM Model ===")
    t3_model = T3TransformerLSTMModel(
        input_size=input_size, 
        d_model=d_model, 
        nhead=nhead, 
        transformer_layers=transformer_layers, 
        lstm_hidden=lstm_hidden
    ).to(device)
    
    t3_criterion = nn.MSELoss()
    t3_optimizer = torch.optim.Adam(t3_model.parameters(), lr=lr, weight_decay=1e-5)
    t3_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(t3_optimizer, patience=5, factor=0.5)
    
    t3_train_ds = T3Dataset(train_dfs['train'], features, target, t1_train_preds, t2_train_preds, lookback)
    if len(t3_train_ds) == 0:
        print("T+3 training dataset is empty. Skipping training for T+3 model.")
        return models, train_metrics
    
    t3_train_loader = DataLoader(t3_train_ds, batch_size=128, shuffle=True)
    t3_train_losses = []
    best_t3_nse = -np.inf
    best_t3_state = None

    for epoch in tqdm(range(num_epochs), desc="Training T+3 Transformer-LSTM"):
        t3_model.train()
        epoch_loss = 0
        epoch_y_true, epoch_y_pred = [], []

        for xb, t1_pred, t2_pred, yb in t3_train_loader:
            xb, t1_pred, t2_pred, yb = xb.to(device), t1_pred.to(device), t2_pred.to(device), yb.to(device)
            pred = t3_model(xb, t1_pred, t2_pred)
            loss = t3_criterion(pred, yb)
            t3_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(t3_model.parameters(), max_norm=1.0)
            t3_optimizer.step()

            epoch_loss += loss.item()
            epoch_y_true.append(yb.detach().cpu().numpy())
            epoch_y_pred.append(pred.detach().cpu().numpy())

        avg_loss = epoch_loss / len(t3_train_loader)
        t3_train_losses.append(avg_loss)
        t3_scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            if epoch_y_true and epoch_y_pred:
                y_true = np.concatenate(epoch_y_true)
                y_pred = np.concatenate(epoch_y_pred)
                train_nse = nse(y_true, y_pred)

                if train_nse > best_t3_nse:
                    best_t3_nse = train_nse
                    best_t3_state = t3_model.state_dict().copy()

                print(f"T+3 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, NSE: {train_nse:.4f}, LR: {t3_optimizer.param_groups[0]['lr']:.6f}")
            else:
                print(f"T+3 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, No valid data for NSE calculation.")

    if best_t3_state is not None:
        t3_model.load_state_dict(best_t3_state)
    else:
        print("Warning: No best T+3 model state saved. Using last epoch's model.")
    
    if epoch_y_true and epoch_y_pred:
        y_true = np.concatenate(epoch_y_true)
        y_pred = np.concatenate(epoch_y_pred)
        train_metrics['T+3'] = evaluate(y_true, y_pred)
    else:
        train_metrics['T+3'] = {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan}
    models['T+3'] = t3_model

    # Save training loss plots
    if run_dir:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(t1_train_losses)
        plt.title('Training Loss - T+1 Transformer-LSTM')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(t2_train_losses)
        plt.title('Training Loss - T+2 Transformer-LSTM')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(t3_train_losses)
        plt.title('Training Loss - T+3 Transformer-LSTM')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'training_losses_transformer_lstm.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return models, train_metrics

# Evaluation Function
def evaluate_sequential_models(models, test_df, lookback, scaler, target, features, run_dir=None, dataset_name="", type=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model in models.values():
        model.to(device)
        model.eval()

    horizon_metrics = {f'T+{i+1}': {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan} for i in range(3)}
    horizon_nse = [np.nan, np.nan, np.nan]
    top10_nse = [np.nan, np.nan, np.nan]
    y_trues_inv = [np.array([]), np.array([]), np.array([])]
    y_preds_inv = [np.array([]), np.array([]), np.array([])]
    dates = np.array([])

    try:
        target_idx = features.index(target)
    except ValueError:
        print(f"Error: Target variable '{target}' not found in the features list used for scaling. Cannot evaluate.")
        return horizon_metrics, top10_nse, horizon_nse

    t1_test_ds = T1Dataset(test_df, features, target, lookback)
    if len(t1_test_ds) > 0:
        t1_test_loader = DataLoader(t1_test_ds, batch_size=256, shuffle=False)
        t1_test_preds_raw = get_predictions(models['T+1'], t1_test_loader, device)

        y_trues_raw_t1 = test_df[target].iloc[lookback : lookback + len(t1_test_preds_raw)].values

        if len(y_trues_raw_t1) > 0:
            dummy_true_t1 = np.zeros((len(y_trues_raw_t1), len(features)))
            dummy_pred_t1 = np.zeros((len(t1_test_preds_raw), len(features)))
            dummy_true_t1[:, target_idx] = y_trues_raw_t1
            dummy_pred_t1[:, target_idx] = t1_test_preds_raw

            y_trues_inv[0] = scaler.inverse_transform(dummy_true_t1)[:, target_idx]
            y_preds_inv[0] = scaler.inverse_transform(dummy_pred_t1)[:, target_idx]

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
            dates = test_df['date'].iloc[lookback : lookback + len(t1_test_preds_raw)].values
        else:
            print(f"Not enough data to evaluate T+1 for {dataset_name}.")
    else:
        print(f"T+1 test dataset is empty for {dataset_name}.")

    if len(t1_test_preds_raw) > 0:
        t2_test_ds = T2Dataset(test_df, features, target, t1_test_preds_raw, lookback)
        if len(t2_test_ds) > 0:
            t2_test_loader = DataLoader(t2_test_ds, batch_size=256, shuffle=False)
            t2_test_preds_raw = get_predictions(models['T+2'], t2_test_loader, device)

            y_trues_raw_t2 = test_df[target].iloc[lookback + 1 : lookback + 1 + len(t2_test_preds_raw)].values

            if len(y_trues_raw_t2) > 0:
                dummy_true_t2 = np.zeros((len(y_trues_raw_t2), len(features)))
                dummy_pred_t2 = np.zeros((len(t2_test_preds_raw), len(features)))
                dummy_true_t2[:, target_idx] = y_trues_raw_t2
                dummy_pred_t2[:, target_idx] = t2_test_preds_raw

                y_trues_inv[1] = scaler.inverse_transform(dummy_true_t2)[:, target_idx]
                y_preds_inv[1] = scaler.inverse_transform(dummy_pred_t2)[:, target_idx]

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

    if 't2_test_preds_raw' in locals() and len(t2_test_preds_raw) > 0:
        t3_test_ds = T3Dataset(test_df, features, target, t1_test_preds_raw, t2_test_preds_raw, lookback)
        if len(t3_test_ds) > 0:
            t3_test_loader = DataLoader(t3_test_ds, batch_size=256, shuffle=False)
            t3_test_preds_raw = get_predictions(models['T+3'], t3_test_loader, device)

            y_trues_raw_t3 = test_df[target].iloc[lookback + 2 : lookback + 2 + len(t3_test_preds_raw)].values

            if len(y_trues_raw_t3) > 0:
                dummy_true_t3 = np.zeros((len(y_trues_raw_t3), len(features)))
                dummy_pred_t3 = np.zeros((len(t3_test_preds_raw), len(features)))
                dummy_true_t3[:, target_idx] = y_trues_raw_t3
                dummy_pred_t3[:, target_idx] = t3_test_preds_raw

                y_trues_inv[2] = scaler.inverse_transform(dummy_true_t3)[:, target_idx]
                y_preds_inv[2] = scaler.inverse_transform(dummy_pred_t3)[:, target_idx]

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
                    plt.title(f'Transformer-LSTM Streamflow Prediction at T+{i+1} - {dataset_name}')
                    plt.ylabel('Streamflow')
                else:
                    plt.title(f'Transformer-LSTM Waterlevel Prediction at T+{i+1} - {dataset_name}')
                    plt.ylabel('Waterlevel')
                plt.xlabel('Date')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                print(f"Skipping plot for T+{i+1} as data is empty for {dataset_name}.")
        if plt.get_fignums():
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f'transformer_lstm_predictions_{dataset_name.lower().replace(" ", "_")}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print(f"No plots generated for {dataset_name} due to lack of valid data.")
    elif run_dir:
        print(f"Skipping plot generation for {dataset_name} as there's no sufficient data or valid dates.")

    return horizon_metrics, top10_nse, horizon_nse

# Main Function
def main():
    import numpy as np
    np.random.seed(74)
    torch.manual_seed(74)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(74)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='Transformer-LSTM Hybrid Streamflow Prediction')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--desc', type=str, required=True, help='Description of the run (e.g., station_target_features)')
    parser.add_argument('--station_file', type=str, default="/home/aravinthakshan/Projects/SRIP-Weather-Forecasting/dataset/cleaned/hierarchial/new_Handia_with_hierarchical_quantiles.csv", help='Path to the station CSV file')
    parser.add_argument('--target_variable', type=str, default="streamflow_final", help='Target variable for prediction')
    parser.add_argument('--features', type=str, default="rainfall, tmin, tmax, waterlevel_final, streamflow_final, Rain_cumulative_7d, streamflow_upstream,waterlevel_upstream,hierarchical_feature_quantile_0.5", help='Comma-separated list of features to use')
    parser.add_argument('--output_base_dir', type=str, default="/home/aravinthakshan/Projects/SRIP-Weather-Forecasting/plots", help='Base directory where all run outputs will be saved')
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

    lookback = 15
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

    print("=== Starting Transformer-LSTM Hybrid Model Training ===")
    models, train_metrics = train_sequential_transformer_models(
        train_dfs, target, features,
        num_epochs=args.epochs, run_dir=run_dir, lookback=lookback
    )

    for horizon, model in models.items():
        model_path = os.path.join(run_dir, f"best_transformer_lstm_model_{horizon.lower().replace('+', '')}.pt")
        try:
            torch.save(model.state_dict(), model_path)
            print(f"Transformer-LSTM Model {horizon} saved to: {model_path}")
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
            print("\n=== Evaluating Transformer-LSTM Models on First 10 Years ===")
            first_10_metrics, first_10_top10_nse, first_10_horizon_nse = evaluate_sequential_models(
                models, first_10_df, lookback, scaler, target, features,
                run_dir, "First 10 Years", type
            )
            test_metrics['first_10'] = first_10_metrics
            test_metrics['first_10_top10_nse'] = first_10_top10_nse
            test_metrics['first_10_horizon_nse'] = first_10_horizon_nse
        else:
            print("Skipping evaluation for First 10 Years due to empty or invalid data.")

        if not last_10_df.empty:
            print("\n=== Evaluating Transformer-LSTM Models on Last 10 Years ===")
            last_10_metrics, last_10_top10_nse, last_10_horizon_nse = evaluate_sequential_models(
                models, last_10_df, lookback, scaler, target, features,
                run_dir, "Last 10 Years", type
            )
            test_metrics['last_10'] = last_10_metrics
            test_metrics['last_10_top10_nse'] = last_10_top10_nse
            test_metrics['last_10_horizon_nse'] = last_10_horizon_nse
        else:
            print("Skipping evaluation for Last 10 Years due to empty or invalid data.")
    else:
        print("Skipping all evaluation steps as T+1 model was not trained successfully.")

    save_run_info(run_dir, args.desc, args, train_metrics, test_metrics, type)
    print(f"\nTransformer-LSTM run information saved to: {os.path.join(run_dir, 'run_info.txt')}")
    print("\n" + "="*60)
    print("TRANSFORMER-LSTM HYBRID TRAINING COMPLETE")
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

if __name__ == "__main__":
    main()

# """
# # Improved Transformer-LSTM Model Classes

# import torch
# import torch.nn as nn
# import math

# class TimeSeriesPositionalEncoding(nn.Module):
#     """Time-aware positional encoding for time series data"""
#     def __init__(self, d_model, max_len=5000, learnable=True):
#         super().__init__()
#         self.d_model = d_model
#         self.learnable = learnable
        
#         if learnable:
#             # Learnable positional embeddings
#             self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.1)
#         else:
#             # Traditional sinusoidal with time-series adaptation
#             pe = torch.zeros(max_len, d_model)
#             position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
#             # Use different frequencies for time series
#             div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
#                                (-math.log(1000.0) / d_model))  # Reduced from 10000
            
#             pe[:, 0::2] = torch.sin(position * div_term)
#             pe[:, 1::2] = torch.cos(position * div_term)
            
#             self.register_buffer('pe', pe)

#     def forward(self, x):
#         seq_len = x.size(1)
#         if self.learnable:
#             return x + self.pos_embedding[:seq_len, :].unsqueeze(0)
#         else:
#             return x + self.pe[:seq_len, :].unsqueeze(0)

# class AttentionPooling(nn.Module):
#     """Attention-based pooling instead of just using last timestep"""
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.Tanh(),
#             nn.Linear(hidden_size // 2, 1),
#             nn.Softmax(dim=1)
#         )
    
#     def forward(self, lstm_output):
#         # lstm_output: [batch, seq_len, hidden_size]
#         attention_weights = self.attention(lstm_output)  # [batch, seq_len, 1]
#         weighted_output = torch.sum(lstm_output * attention_weights, dim=1)  # [batch, hidden_size]
#         return weighted_output

# class ImprovedT1TransformerLSTMModel(nn.Module):
#     def __init__(self, input_size, d_model=64, nhead=4, transformer_layers=2, 
#                  lstm_hidden=32, num_layers=1, use_bidirectional=True, 
#                  use_attention_pooling=True, positional_encoding_type='learnable'):
#         super().__init__()
        
#         self.use_attention_pooling = use_attention_pooling
#         self.use_bidirectional = use_bidirectional
        
#         # Input projection with residual connection capability
#         self.input_projection = nn.Linear(input_size, d_model)
#         self.input_norm = nn.LayerNorm(d_model)
        
#         # Improved positional encoding
#         if positional_encoding_type == 'learnable':
#             self.pos_encoder = TimeSeriesPositionalEncoding(d_model, learnable=True)
#         elif positional_encoding_type == 'time_series':
#             self.pos_encoder = TimeSeriesPositionalEncoding(d_model, learnable=False)
#         else:
#             self.pos_encoder = None
        
#         # Transformer encoder with residual connections
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, 
#             nhead=nhead, 
#             dim_feedforward=d_model * 4,  # Increased feedforward dim
#             dropout=0.1,
#             batch_first=True,
#             activation='gelu'  # GELU often works better than ReLU
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
#         # Bidirectional LSTM
#         lstm_input_size = d_model
#         self.lstm = nn.LSTM(
#             lstm_input_size, lstm_hidden, num_layers, 
#             batch_first=True, 
#             dropout=0.1 if num_layers > 1 else 0,
#             bidirectional=use_bidirectional
#         )
        
#         # Adjust hidden size for bidirectional
#         final_lstm_hidden = lstm_hidden * (2 if use_bidirectional else 1)
        
#         # Attention pooling or simple last timestep
#         if use_attention_pooling:
#             self.attention_pool = AttentionPooling(final_lstm_hidden)
#             fc_input_size = final_lstm_hidden
#         else:
#             fc_input_size = final_lstm_hidden
        
#         # Multi-layer prediction head instead of single linear layer
#         self.prediction_head = nn.Sequential(
#             nn.Linear(fc_input_size, fc_input_size // 2),
#             nn.LayerNorm(fc_input_size // 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(fc_input_size // 2, fc_input_size // 4),
#             nn.LayerNorm(fc_input_size // 4),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(fc_input_size // 4, 1)
#         )
        
#     def forward(self, x):
#         # Project and normalize input
#         x_proj = self.input_norm(self.input_projection(x))
        
#         # Add positional encoding if available
#         if self.pos_encoder is not None:
#             x_proj = self.pos_encoder(x_proj)
        
#         # Transformer encoding
#         contextual_encoding = self.transformer_encoder(x_proj)
        
#         # LSTM processing
#         lstm_out, _ = self.lstm(contextual_encoding)
        
#         # Pooling strategy
#         if self.use_attention_pooling:
#             pooled_output = self.attention_pool(lstm_out)
#         else:
#             if self.use_bidirectional:
#                 # For bidirectional, take the last forward and first backward
#                 pooled_output = lstm_out[:, -1, :]
#             else:
#                 pooled_output = lstm_out[:, -1, :]
        
#         # Final prediction
#         return self.prediction_head(pooled_output).squeeze(-1)

# class ImprovedT2TransformerLSTMModel(nn.Module):
#     def __init__(self, input_size, d_model=64, nhead=4, transformer_layers=2, 
#                  lstm_hidden=32, num_layers=1, use_bidirectional=True, 
#                  use_attention_pooling=True, positional_encoding_type='learnable'):
#         super().__init__()
        
#         self.use_attention_pooling = use_attention_pooling
#         self.use_bidirectional = use_bidirectional
        
#         # Input projection
#         self.input_projection = nn.Linear(input_size, d_model)
#         self.input_norm = nn.LayerNorm(d_model)
        
#         # Positional encoding
#         if positional_encoding_type == 'learnable':
#             self.pos_encoder = TimeSeriesPositionalEncoding(d_model, learnable=True)
#         elif positional_encoding_type == 'time_series':
#             self.pos_encoder = TimeSeriesPositionalEncoding(d_model, learnable=False)
#         else:
#             self.pos_encoder = None
        
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, 
#             nhead=nhead, 
#             dim_feedforward=d_model * 4,
#             dropout=0.1,
#             batch_first=True,
#             activation='gelu'
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
#         # LSTM
#         self.lstm = nn.LSTM(
#             d_model, lstm_hidden, num_layers,
#             batch_first=True, 
#             dropout=0.1 if num_layers > 1 else 0,
#             bidirectional=use_bidirectional
#         )
        
#         final_lstm_hidden = lstm_hidden * (2 if use_bidirectional else 1)
        
#         # Attention pooling
#         if use_attention_pooling:
#             self.attention_pool = AttentionPooling(final_lstm_hidden)
        
#         # Cross-attention to incorporate T+1 prediction
#         self.t1_projection = nn.Linear(1, d_model // 4)
#         self.cross_attention = nn.MultiheadAttention(
#             embed_dim=final_lstm_hidden, 
#             num_heads=max(1, final_lstm_hidden // 32),
#             batch_first=True
#         )
        
#         # Prediction head
#         self.prediction_head = nn.Sequential(
#             nn.Linear(final_lstm_hidden, final_lstm_hidden // 2),
#             nn.LayerNorm(final_lstm_hidden // 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(final_lstm_hidden // 2, final_lstm_hidden // 4),
#             nn.LayerNorm(final_lstm_hidden // 4),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(final_lstm_hidden // 4, 1)
#         )
        
#     def forward(self, x, t1_pred):
#         # Process main input
#         x_proj = self.input_norm(self.input_projection(x))
        
#         if self.pos_encoder is not None:
#             x_proj = self.pos_encoder(x_proj)
        
#         contextual_encoding = self.transformer_encoder(x_proj)
#         lstm_out, _ = self.lstm(contextual_encoding)
        
#         # Pool LSTM output
#         if self.use_attention_pooling:
#             pooled_output = self.attention_pool(lstm_out)
#         else:
#             pooled_output = lstm_out[:, -1, :]
        
#         # Process T+1 prediction as a query for cross-attention
#         t1_expanded = self.t1_projection(t1_pred.unsqueeze(-1)).unsqueeze(1)  # [batch, 1, d_model//4]
        
#         # Use cross-attention to incorporate T+1 information
#         # Pad t1_expanded to match lstm dimensions for attention
#         t1_padded = torch.cat([
#             t1_expanded, 
#             torch.zeros(t1_expanded.size(0), 1, pooled_output.size(-1) - t1_expanded.size(-1), 
#                        device=t1_expanded.device)
#         ], dim=-1)
        
#         attended_output, _ = self.cross_attention(
#             t1_padded, 
#             pooled_output.unsqueeze(1), 
#             pooled_output.unsqueeze(1)
#         )
        
#         # Combine with original output
#         combined_output = pooled_output + attended_output.squeeze(1)
        
#         return self.prediction_head(combined_output).squeeze(-1)

# # Alternative: Skip connections between transformer and LSTM
# class SkipConnectionTransformerLSTM(nn.Module):
#     def __init__(self, input_size, d_model=64, nhead=4, transformer_layers=2, lstm_hidden=32):
#         super().__init__()
        
#         self.input_projection = nn.Linear(input_size, d_model)
#         self.pos_encoder = TimeSeriesPositionalEncoding(d_model, learnable=True)
        
#         # Transformer
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
#             dropout=0.1, batch_first=True, activation='gelu'
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
#         # LSTM
#         self.lstm = nn.LSTM(d_model, lstm_hidden, batch_first=True, bidirectional=True)
        
#         # Skip connection projection
#         self.skip_projection = nn.Linear(d_model, lstm_hidden * 2)
        
#         # Final layers
#         self.attention_pool = AttentionPooling(lstm_hidden * 2)
#         self.prediction_head = nn.Sequential(
#             nn.Linear(lstm_hidden * 2, lstm_hidden),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(lstm_hidden, 1)
#         )
    
#     def forward(self, x):
#         # Input processing
#         x_proj = self.input_projection(x)
#         x_proj = self.pos_encoder(x_proj)
        
#         # Transformer encoding
#         transformer_out = self.transformer(x_proj)
        
#         # LSTM processing
#         lstm_out, _ = self.lstm(transformer_out)
        
#         # Skip connection from transformer output
#         skip_connection = self.skip_projection(transformer_out)
        
#         # Combine LSTM output with skip connection
#         combined = lstm_out + skip_connection
        
#         # Pool and predict
#         pooled = self.attention_pool(combined)
#         return self.prediction_head(pooled).squeeze(-1)
# Major Issues with Current Approach:

# Positional Encoding Mismatch: Standard sinusoidal PE is designed for NLP, not time series. For time series, you want temporal relationships, not just position.
# Linear Bottleneck: Single linear layer loses information. Multi-layer prediction heads work better.
# Information Loss: Using only the last LSTM timestep discards valuable sequence information.
# Transformer-LSTM Integration: They're operating somewhat independently rather than complementing each other.

# Key Improvements:

# Better Positional Encoding:

# Learnable embeddings that adapt to your specific time series patterns
# Lower frequencies (1000 vs 10000) for time series temporal relationships


# Bidirectional LSTM: Captures both forward and backward temporal dependencies
# Attention Pooling: Instead of last timestep, use attention to weight all timesteps
# Multi-layer Prediction Head: Replaces the bottleneck linear layer with a proper neural network
# Skip Connections: Direct paths between transformer and final output to prevent gradient vanishing
# Cross-attention Integration: For T2/T3 models, use attention mechanisms to better incorporate previous predictions

# Additional Suggestions:

# Layer Normalization: Stabilizes training
# GELU Activation: Often works better than ReLU for transformers
# Larger Feedforward Dimension: Standard practice is 4x the model dimension

# The transformer might actually be hurting because:

# Time series have strong sequential dependencies that LSTM handles well
# Transformer's attention might be creating noise without enough data
# The combination isn't leveraging the strengths of both architectures
# lets try to figure out better methods  sequnetial lstm is doing well wihtout trasnformer so I though adding it would make it better  now I think  1) positonal encoding some alternatvie migt help for time sries  2) bi drectinoal lstm  3) that fc linear layer might be a bottle neck alternatvei to that maybe   do you think anything else might help or is wrogn with my current apprajc
# """

# bi directional 
# with trasnfomrer
# with differnt encoding other thna positional
# replacement for fc/linear
