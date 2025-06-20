import numpy as np 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
        return {'NSE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'KGE': np.nan}
    return {
        'NSE': nse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'PBIAS': pbias(y_true, y_pred),
        'KGE': kge(y_true, y_pred)
    }