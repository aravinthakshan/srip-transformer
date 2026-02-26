"""
Utilities Module
================
Shared utilities for config loading, data preparation, and metrics.
Enforces hard constraints for ablation experiments.
"""

import os
import yaml
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================================================
# HARD CONSTRAINTS - DO NOT MODIFY BETWEEN EXPERIMENTS
# ============================================================================
HARD_CONSTRAINTS = {
    "lookback": 7,  # Days of history (t-7 to t-1, NOT t)
    "batch_size": 256,
    "learning_rate": 1e-3,
    "epochs": 30,
    "random_seed": 74,
    "optimizer": "Adam",
    "loss": "MSE",
    "train_years": (1971, 2010),
    "test_first_years": (1961, 1970),
    "test_last_years": (2011, 2020),
}

# ============================================================================
# Configuration Loading
# ============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Merges with hard constraints and validates.

    Args:
        config_path: Path to YAML config file

    Returns:
        Complete configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Flatten nested config
    flat_config = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat_config[k] = v
        else:
            flat_config[section] = values

    # Enforce hard constraints (override any user settings)
    flat_config["lookback"] = HARD_CONSTRAINTS["lookback"]
    flat_config["batch_size"] = HARD_CONSTRAINTS["batch_size"]
    flat_config["learning_rate"] = HARD_CONSTRAINTS["learning_rate"]
    flat_config["epochs"] = HARD_CONSTRAINTS["epochs"]
    flat_config["random_seed"] = HARD_CONSTRAINTS["random_seed"]

    return flat_config


def get_config_name(config_path: str) -> str:
    """Extract config name from path."""
    return Path(config_path).stem


# ============================================================================
# Random Seed Setting
# ============================================================================


def set_seed(seed: int = HARD_CONSTRAINTS["random_seed"]):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Data Loading and Preparation
# ============================================================================


def load_station_data(
    csv_path: str, features: List[str], target: str, date_col: str = "date"
) -> pd.DataFrame:
    """
    Load and validate station data.

    Args:
        csv_path: Path to station CSV
        features: List of feature column names
        target: Target column name
        date_col: Date column name

    Returns:
        Cleaned DataFrame sorted by date
    """
    df = pd.read_csv(csv_path, parse_dates=[date_col], dayfirst=True)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df["year"] = df[date_col].dt.year

    # Validate columns
    required_cols = features + [target, date_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def split_data(
    df: pd.DataFrame,
    train_years: Tuple[int, int] = HARD_CONSTRAINTS["train_years"],
    test_first_years: Tuple[int, int] = HARD_CONSTRAINTS["test_first_years"],
    test_last_years: Tuple[int, int] = HARD_CONSTRAINTS["test_last_years"],
) -> Dict[str, pd.DataFrame]:
    """
    Split data into train/test sets based on years.

    Enforces consistent splits for all experiments.

    Args:
        df: Input DataFrame with 'year' column
        train_years: (start_year, end_year) for training
        test_first_years: (start_year, end_year) for first test period
        test_last_years: (start_year, end_year) for last test period

    Returns:
        Dict with 'train', 'test_first', 'test_last' DataFrames
    """
    train_mask = (df["year"] >= train_years[0]) & (df["year"] <= train_years[1])
    test_first_mask = (df["year"] >= test_first_years[0]) & (
        df["year"] <= test_first_years[1]
    )
    test_last_mask = (df["year"] >= test_last_years[0]) & (
        df["year"] <= test_last_years[1]
    )

    return {
        "train": df[train_mask].reset_index(drop=True),
        "test_first": df[test_first_mask].reset_index(drop=True),
        "test_last": df[test_last_mask].reset_index(drop=True),
    }


def scale_features(
    train_df: pd.DataFrame, test_dfs: List[pd.DataFrame], features: List[str]
) -> Tuple[pd.DataFrame, List[pd.DataFrame], MinMaxScaler]:
    """
    Fit scaler on training data and transform all datasets.

    Args:
        train_df: Training DataFrame
        test_dfs: List of test DataFrames
        features: Feature columns to scale

    Returns:
        Scaled train_df, list of scaled test_dfs, fitted scaler
    """
    scaler = MinMaxScaler()

    train_df = train_df.copy()
    train_df[features] = scaler.fit_transform(train_df[features])

    scaled_test_dfs = []
    for test_df in test_dfs:
        test_df = test_df.copy()
        test_df[features] = scaler.transform(test_df[features])
        scaled_test_dfs.append(test_df)

    return train_df, scaled_test_dfs, scaler


# ============================================================================
# Metrics
# ============================================================================


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    if np.all(y_true == np.mean(y_true)):
        return -np.inf
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def pbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percent Bias."""
    if np.sum(y_true) == 0:
        return np.nan
    return 100 * (np.sum(y_true - y_pred) / np.sum(y_true))


def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kling-Gupta Efficiency."""
    if len(y_true) < 2:
        return np.nan

    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else np.nan
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) > 0 else np.nan

    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return np.nan

    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all metrics."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[valid], y_pred[valid]

    if len(y_true) == 0:
        return {"NSE": np.nan, "R2": np.nan, "PBIAS": np.nan, "KGE": np.nan}

    return {
        "NSE": nse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "PBIAS": pbias(y_true, y_pred),
        "KGE": kge(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def compute_extreme_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, percentiles: List[float] = [90, 95]
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for extreme (high) flow events.

    Args:
        y_true: Ground truth
        y_pred: Predictions
        percentiles: Percentile thresholds for "extreme" definition

    Returns:
        Dict mapping percentile -> metrics dict
    """
    results = {}

    for pct in percentiles:
        threshold = np.percentile(y_true, pct)
        mask = y_true >= threshold

        if np.sum(mask) < 2:
            results[f"p{pct}"] = {"NSE": np.nan, "count": 0}
        else:
            results[f"p{pct}"] = {
                "NSE": nse(y_true[mask], y_pred[mask]),
                "count": int(np.sum(mask)),
                "threshold": float(threshold),
            }

    return results


def compute_peak_capture(
    y_true: np.ndarray, y_pred: np.ndarray, window_size: int = 7
) -> Dict[str, float]:
    """
    Compute peak capture percentage using rolling windows.

    For each window, find the peak in ground truth and measure
    what percentage of it was captured by the prediction.

    Args:
        y_true: Ground truth values
        y_pred: Predictions
        window_size: Size of rolling window (days)

    Returns:
        Statistics about peak capture
    """
    peak_captures = []

    for i in range(len(y_true) - window_size + 1):
        window_true = y_true[i : i + window_size]
        window_pred = y_pred[i : i + window_size]

        peak_idx = np.argmax(window_true)
        true_peak = window_true[peak_idx]
        pred_at_peak = window_pred[peak_idx]

        if true_peak > 0:
            capture = min(100, max(0, (pred_at_peak / true_peak) * 100))
            peak_captures.append(capture)

    if not peak_captures:
        return {"mean": np.nan, "median": np.nan, "std": np.nan}

    return {
        "mean": np.mean(peak_captures),
        "median": np.median(peak_captures),
        "std": np.std(peak_captures),
        "min": np.min(peak_captures),
        "max": np.max(peak_captures),
    }


# ============================================================================
# Output Directory Management
# ============================================================================


def create_run_directory(config_name: str, output_base: str = "runs") -> str:
    """
    Create directory for experiment outputs.

    Args:
        config_name: Name of the config (used in directory name)
        output_base: Base directory for all runs

    Returns:
        Path to created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_base, f"{config_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_results(
    run_dir: str,
    config: Dict[str, Any],
    train_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    model_info: Dict[str, Any] = None,
):
    """
    Save experiment results to JSON.

    Args:
        run_dir: Output directory
        config: Configuration used
        train_metrics: Training metrics
        test_metrics: Test metrics
        model_info: Additional model information (params, etc.)
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "hard_constraints": HARD_CONSTRAINTS,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_info": model_info or {},
    }

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)


# ============================================================================
# Dataset Classes (PyTorch)
# ============================================================================


class SequenceDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for sequence-to-prediction tasks.

    Implements the lookback constraint: uses data from t-n to t-1
    to predict t+1, t+2, t+3 (NOT including day t).

    Args:
        data: DataFrame with features and target
        features: List of feature column names
        target: Target column name
        lookback: Number of historical days to use
        horizon: Which horizon to return target for (1, 2, or 3)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        target: str,
        lookback: int = HARD_CONSTRAINTS["lookback"],
        horizon: int = 1,
    ):
        self.X = data[features].values
        self.y = data[target].values
        self.lookback = lookback
        self.horizon = horizon

        # Valid range: need lookback days + horizon offset
        self.max_idx = len(self.X) - lookback - horizon

    def __len__(self):
        return max(0, self.max_idx)

    def __getitem__(self, idx):
        # Input: t-lookback to t-1 (NOT including t)
        x = self.X[idx : idx + self.lookback]

        # Target: t+horizon (offset from end of lookback)
        y = self.y[idx + self.lookback + self.horizon - 1]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class MultiHorizonDataset(torch.utils.data.Dataset):
    """
    Dataset returning targets for all three horizons.

    Args:
        data: DataFrame with features and target
        features: Feature column names
        target: Target column name
        lookback: Historical window size
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        target: str,
        lookback: int = HARD_CONSTRAINTS["lookback"],
    ):
        self.X = data[features].values
        self.y = data[target].values
        self.lookback = lookback

        # Need lookback + 3 (for T+3 target)
        self.max_idx = len(self.X) - lookback - 3

    def __len__(self):
        return max(0, self.max_idx)

    def __getitem__(self, idx):
        # Input: t-lookback to t-1
        x = self.X[idx : idx + self.lookback]

        # Targets: T+1, T+2, T+3
        y1 = self.y[idx + self.lookback]
        y2 = self.y[idx + self.lookback + 1]
        y3 = self.y[idx + self.lookback + 2]

        return (
            torch.tensor(x, dtype=torch.float32),
            {
                "t1": torch.tensor(y1, dtype=torch.float32),
                "t2": torch.tensor(y2, dtype=torch.float32),
                "t3": torch.tensor(y3, dtype=torch.float32),
            },
        )


class MultiHorizonDualTargetDataset(torch.utils.data.Dataset):
    """
    Dataset returning targets for both primary variable (streamflow) and
    water level for all three horizons.

    Used when the fitter module is active: the model predicts water level
    at each horizon, converts via rating curve to expected streamflow,
    and uses that to predict final streamflow. We need WL ground truth
    for the auxiliary water level loss.

    Args:
        data: DataFrame with features and targets
        features: Feature column names
        primary_target: Primary target column (e.g. 'streamflow_final')
        wl_target: Water level target column (e.g. 'waterlevel_final')
        lookback: Historical window size
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        primary_target: str,
        wl_target: str,
        lookback: int = HARD_CONSTRAINTS["lookback"],
    ):
        self.X = data[features].values
        self.y_primary = data[primary_target].values
        self.y_wl = data[wl_target].values
        self.lookback = lookback

        # Need lookback + 3 (for T+3 target)
        self.max_idx = len(self.X) - lookback - 3

    def __len__(self):
        return max(0, self.max_idx)

    def __getitem__(self, idx):
        # Input: t-lookback to t-1
        x = self.X[idx : idx + self.lookback]

        # Primary targets (streamflow): T+1, T+2, T+3
        y1 = self.y_primary[idx + self.lookback]
        y2 = self.y_primary[idx + self.lookback + 1]
        y3 = self.y_primary[idx + self.lookback + 2]

        # Water level targets: T+1, T+2, T+3
        wl1 = self.y_wl[idx + self.lookback]
        wl2 = self.y_wl[idx + self.lookback + 1]
        wl3 = self.y_wl[idx + self.lookback + 2]

        return (
            torch.tensor(x, dtype=torch.float32),
            {
                "t1": torch.tensor(y1, dtype=torch.float32),
                "t2": torch.tensor(y2, dtype=torch.float32),
                "t3": torch.tensor(y3, dtype=torch.float32),
                "wl_t1": torch.tensor(wl1, dtype=torch.float32),
                "wl_t2": torch.tensor(wl2, dtype=torch.float32),
                "wl_t3": torch.tensor(wl3, dtype=torch.float32),
            },
        )
