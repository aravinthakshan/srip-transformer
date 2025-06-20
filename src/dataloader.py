import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

### ------------------------ Dataloaders ------------------------ ###
class T1Dataset(Dataset):
    """
    For T+1 forecasting:
    - Uses data from t-n to t-1 (lookback period)
    - Predicts t+1
    """
    def __init__(self, data, input_cols, target_col, lookback=30):  # Fixed: was missing __
        self.X = data[input_cols].values
        self.y = data[target_col].values
        self.lookback = lookback
    
    def __len__(self):  # Fixed: was missing __
        # Need at least lookback+1 points to create one sample
        if len(self.X) <= self.lookback:
            return 0
        return len(self.X) - self.lookback
    
    def __getitem__(self, idx):  # Fixed: was missing __
        # Input: t-n to t-1 (lookback points ending at idx+lookback-1)
        # Target: t+1 (idx+lookback)
        x = self.X[idx : idx + self.lookback]
        y = self.y[idx + self.lookback]
        return (torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

class T2Dataset(Dataset):
    """
    For T+2 forecasting:
    - Uses data from t-n to t-1 (lookback period)
    - Uses T+1 prediction
    - Predicts t+2
    """
    def __init__(self, data, input_cols, target_col, t1_predictions, rating_curve_preds=None, lookback=30):  # Fixed: was missing __
        self.X = data[input_cols].values
        self.y = data[target_col].values
        self.t1_preds = t1_predictions
        self.rating_curve_preds = rating_curve_preds
        self.lookback = lookback
        self.use_rating_curve = rating_curve_preds is not None
        
        # Ensure we don't go beyond available predictions
        max_samples = min(len(self.X) - self.lookback - 1, len(self.t1_preds))
        if self.use_rating_curve:
            max_samples = min(max_samples, len(self.rating_curve_preds) - self.lookback - 1)
        self.max_samples = max(0, max_samples)
    
    def __len__(self):  # Fixed: was missing __
        return self.max_samples
    
    def __getitem__(self, idx):  # Fixed: was missing __
        # Input: t-n to t-1 (lookback points ending at idx+lookback-1)
        # T+1 prediction corresponds to index idx (T1 predictions start from index 0)
        # Target: t+2 (idx+lookback+1)
        x = self.X[idx : idx + self.lookback]
        t1_pred = self.t1_preds[idx]  # T+1 predictions are aligned with dataset indices
        y = self.y[idx + self.lookback + 1]
        
        if self.use_rating_curve:
            rating_pred = self.rating_curve_preds[idx + self.lookback + 1]  # Rating curve for t+2
            return (torch.tensor(x, dtype=torch.float32), 
                   torch.tensor(t1_pred, dtype=torch.float32),
                   torch.tensor(rating_pred, dtype=torch.float32),
                   torch.tensor(y, dtype=torch.float32))
        else:
            return (torch.tensor(x, dtype=torch.float32), 
                   torch.tensor(t1_pred, dtype=torch.float32),
                   torch.tensor(0.0, dtype=torch.float32),  # Dummy rating curve value
                   torch.tensor(y, dtype=torch.float32))

class T3Dataset(Dataset):
    """
    For T+3 forecasting:
    - Uses data from t-n to t-1 (lookback period)
    - Uses T+1 and T+2 predictions
    - Predicts t+3
    """
    def __init__(self, data, input_cols, target_col, t1_predictions, t2_predictions,   # Fixed: was missing __
                 rating_curve_preds_t1=None, rating_curve_preds_t2=None, lookback=30):
        self.X = data[input_cols].values
        self.y = data[target_col].values
        self.t1_preds = t1_predictions
        self.t2_preds = t2_predictions
        self.rating_curve_preds_t1 = rating_curve_preds_t1
        self.rating_curve_preds_t2 = rating_curve_preds_t2
        self.lookback = lookback
        self.use_rating_curve = (rating_curve_preds_t1 is not None and 
                                rating_curve_preds_t2 is not None)
        
        # Ensure we don't go beyond available predictions
        max_samples = min(len(self.X) - self.lookback - 2, len(self.t1_preds), len(self.t2_preds))
        if self.use_rating_curve:
            max_samples = min(max_samples, len(self.rating_curve_preds_t1), 
                             len(self.rating_curve_preds_t2) - self.lookback - 2)
        self.max_samples = max(0, max_samples)
    
    def __len__(self):  # Fixed: was missing __
        return self.max_samples
    
    def __getitem__(self, idx):  # Fixed: was missing __
        # Input: t-n to t-1 (lookback points ending at idx+lookback-1)
        # T+1 prediction corresponds to index idx
        # T+2 prediction corresponds to index idx
        # Target: t+3 (idx+lookback+2)
        x = self.X[idx : idx + self.lookback]
        t1_pred = self.t1_preds[idx]  # T+1 predictions are aligned with dataset indices
        t2_pred = self.t2_preds[idx]  # T+2 predictions are aligned with dataset indices
        y = self.y[idx + self.lookback + 2]
        
        if self.use_rating_curve:
            rating_pred_t1 = self.rating_curve_preds_t1[idx]                    # Rating curve for t+1
            rating_pred_t2 = self.rating_curve_preds_t2[idx + self.lookback + 1]  # Rating curve for t+2
            return (torch.tensor(x, dtype=torch.float32),
                   torch.tensor(t1_pred, dtype=torch.float32),
                   torch.tensor(t2_pred, dtype=torch.float32),
                   torch.tensor(rating_pred_t1, dtype=torch.float32),
                   torch.tensor(rating_pred_t2, dtype=torch.float32),
                   torch.tensor(y, dtype=torch.float32))
        else:
            return (torch.tensor(x, dtype=torch.float32),
                   torch.tensor(t1_pred, dtype=torch.float32),
                   torch.tensor(t2_pred, dtype=torch.float32),
                   torch.tensor(0.0, dtype=torch.float32),  # Dummy rating curve values
                   torch.tensor(0.0, dtype=torch.float32),
                   torch.tensor(y, dtype=torch.float32))