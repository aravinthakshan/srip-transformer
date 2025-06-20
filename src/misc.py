import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import pickle
import os 
from datetime import datetime

def quadratic(wse, a, b, c):
    return a*(wse**2) + b * wse + c

def exponential_offset(wse, a, b):
    return a * np.exp(b * wse)

def shifted_quadratic(wse, a, b):
    return a * (wse - b)**2

def shifted_cubic(wse, a, b):
    return a * (wse - b)**3

### ------------------------ Curve Fitting for Rating Waterlevel to Streamflow Curves ------------------------ ###
class RatingCurveFitter:
    def __init__(self):  # Fixed: was missing __
        self.functions = {
            "Quadratic": (quadratic, [1, 1, 1]),
            "Exponential Offset": (exponential_offset, [1e-5, 0.01]),
            "Shifted Quadratic": (shifted_quadratic, [1, 260]),
            "Shifted Cubic": (shifted_cubic, [1, 260])
        }
        self.best_model = None
        
    def fit(self, waterlevel, streamflow):
        results = []
        for name, (func, init_params) in self.functions.items():
            try:
                params, _ = curve_fit(func, waterlevel, streamflow, p0=init_params, maxfev=10000)  # Fixed: was missing _
                q_pred = func(waterlevel, *params)
                rmse = np.sqrt(mean_squared_error(streamflow, q_pred))
                mae = mean_absolute_error(streamflow, q_pred)
                r2 = r2_score(streamflow, q_pred)
                results.append({
                    "name": name,
                    "function": func,
                    "params": params,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2
                })
            except Exception as e:
                print(f"Rating curve {name} failed to fit: {e}")
        if results:
            results.sort(key=lambda x: x["RMSE"])
            self.best_model = results[0]
            print(f"Best Rating Curve: {self.best_model['name']}")
            print(f"RMSE: {self.best_model['RMSE']:.3f}, R²: {self.best_model['R2']:.3f}")
            return True
        else:
            print("Warning: No rating curve could be fitted")
            return False
    
    def predict(self, waterlevel):
        if self.best_model is None:
            raise ValueError("Rating curve not fitted yet")
        try:
            return self.best_model["function"](waterlevel, *self.best_model["params"])
        except:
            return np.full_like(waterlevel, np.nan)
    
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.best_model, f)
    
    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.best_model = pickle.load(f)


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

def save_run_info(run_dir, description, args, train_metrics, test_metrics, type=None, rating_curve_info=None):
    info_file = os.path.join(run_dir, "run_info.txt")
    with open(info_file, 'w') as f:
        f.write("=" * 50 + "\n")
        if type == "streamflow":
            f.write("ENHANCED SEQUENTIAL MULTI-LSTM STREAMFLOW PREDICTION WITH RATING CURVE\n")
        else:
            f.write("SEQUENTIAL MULTI-LSTM WATERLEVEL PREDICTION RUN SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Run Description: {description}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Epochs: {args.epochs}\n")
        f.write("Architecture: Sequential LSTM models with Rating Curve Integration\n")
        f.write("    - T+1: Base LSTM (Waterlevel)\n")
        f.write("    - T+2: LSTM + T+1 prediction + Rating Curve (if streamflow)\n")
        f.write("    - T+3: LSTM + T+1 + T+2 predictions + Rating Curves (if streamflow)\n\n")
        if type == "streamflow":
            if not rating_curve_info:
                print("Error: Rating curve information required for streamflow runs.")
                return
            f.write("RATING CURVE INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best Model: {rating_curve_info['name']}\n")
            f.write(f"Parameters: {rating_curve_info['params']}\n")
            f.write(f"RMSE: {rating_curve_info['RMSE']:.4f}\n")
            f.write(f"R²: {rating_curve_info['R2']:.4f}\n\n")
        f.write(f"Station File: {args.station_file}\n")
        f.write(f"Target Variable: {args.target_variable}\n")
        f.write(f"Features: {args.features}\n\n")
        f.write("TRAINING METRICS (Final Epoch):\n")
        f.write("-" * 30 + "\n")
        for horizon, metrics in train_metrics.items():
            f.write(f"LSTM {horizon}:\n")
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
