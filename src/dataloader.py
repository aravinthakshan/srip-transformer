import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

### ------------------------ Dataloaders ------------------------ ###
class T1Dataset(Dataset):
    def __init__(self, data, input_cols, target_col, lookback=30):
        self.X = data[input_cols].values
        self.y = data[target_col].values
        self.lookback = lookback

    def __len__(self):
        # Need lookback (t-n to t-1) + 1 skipped (t) + 1 for y (t+1)
        return max(0, len(self.X) - self.lookback - 1)

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.lookback]              # t-n to t-1
        y = self.y[idx + self.lookback + 1]                # t+1
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))

class T2Dataset(Dataset):
    def __init__(self, data, input_cols, target_col, t1_predictions, rating_curve_preds=None, lookback=30):
        self.X = data[input_cols].values
        self.y = data[target_col].values
        self.t1_preds = t1_predictions
        self.rating_curve_preds = rating_curve_preds
        self.lookback = lookback
        self.use_rating_curve = rating_curve_preds is not None

        max_samples = min(len(self.X) - lookback - 2, len(self.t1_preds))  # Skip t, predict t+2
        if self.use_rating_curve:
            max_samples = min(max_samples, len(rating_curve_preds) - lookback - 2)
        self.max_samples = max(0, max_samples)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.lookback]                # t-n to t-1
        t1_pred = self.t1_preds[idx]                         # t+1
        y = self.y[idx + self.lookback + 2]                  # t+2
        if self.use_rating_curve:
            rating_pred = self.rating_curve_preds[idx + self.lookback + 2]  # t+2
            return (torch.tensor(x, dtype=torch.float32),
                    torch.tensor(t1_pred, dtype=torch.float32),
                    torch.tensor(rating_pred, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32))
        else:
            return (torch.tensor(x, dtype=torch.float32),
                    torch.tensor(t1_pred, dtype=torch.float32),
                    torch.tensor(0.0, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32))

class T3Dataset(Dataset):
    def __init__(self, data, input_cols, target_col, t1_predictions, t2_predictions,
                 rating_curve_preds_t1=None, rating_curve_preds_t2=None, lookback=30):
        self.X = data[input_cols].values
        self.y = data[target_col].values
        self.t1_preds = t1_predictions
        self.t2_preds = t2_predictions
        self.rating_curve_preds_t1 = rating_curve_preds_t1
        self.rating_curve_preds_t2 = rating_curve_preds_t2
        self.lookback = lookback
        self.use_rating_curve = (rating_curve_preds_t1 is not None and rating_curve_preds_t2 is not None)

        max_samples = min(len(self.X) - lookback - 3, len(t1_predictions), len(t2_predictions))
        if self.use_rating_curve:
            max_samples = min(max_samples,
                              len(rating_curve_preds_t1),
                              len(rating_curve_preds_t2) - lookback - 3)
        self.max_samples = max(0, max_samples)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.lookback]                       # t-n to t-1
        t1_pred = self.t1_preds[idx]                                # t+1
        t2_pred = self.t2_preds[idx]                                # t+2
        y = self.y[idx + self.lookback + 3]                         # t+3

        if self.use_rating_curve:
            rating_pred_t1 = self.rating_curve_preds_t1[idx]        # t+1
            rating_pred_t2 = self.rating_curve_preds_t2[idx + self.lookback + 2]  # t+2
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
                    torch.tensor(0.0, dtype=torch.float32),
                    torch.tensor(0.0, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32))
