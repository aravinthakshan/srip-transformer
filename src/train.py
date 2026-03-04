#!/usr/bin/env python
"""
Training Script
===============
Unified config-driven training loop for ablation experiments.

Usage:
    python train.py --config configs/baseline.yaml --station path/to/station.csv --target streamflow_final
    python train.py --config configs/seq_fitter.yaml --station path/to/station.csv --target streamflow_final
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_config,
    set_seed,
    load_station_data,
    split_data,
    scale_features,
    create_run_directory,
    save_results,
    compute_metrics,
    compute_extreme_metrics,
    compute_peak_capture,
    MultiHorizonDataset,
    HARD_CONSTRAINTS,
)
from models import SeqLSTMModel
from models.full_model import ModelConfig
from models.fitter import RatingCurveFitter


def print_step(step_num: int, title: str, char: str = "="):
    """Print a formatted step header."""
    print(f"\n{char * 60}")
    print(f"  STEP {step_num}: {title}")
    print(f"{char * 60}")


def print_substep(title: str):
    """Print a formatted substep."""
    print(f"\n  → {title}")


def print_config_summary(config_dict: dict):
    """Print configuration summary."""
    print("\n  Configuration:")
    print(
        f"    • Sequential:    {'✓ ON  (3 LSTM blocks)' if config_dict.get('sequential', False) else '✗ OFF (single LSTM)'}"
    )
    print(
        f"    • Bidirectional: {'✓ ON' if config_dict.get('bidirectional', False) else '✗ OFF'}"
    )
    print(
        f"    • Attention:     {'✓ ON  (MHA)' if config_dict.get('use_attention', False) else '✗ OFF'}"
    )
    print(
        f"    • Rating Curve:  {'✓ ON' if config_dict.get('use_fitter', False) else '✗ OFF'}"
    )
    print(
        f"    • Hierarchical:  {'✓ ON' if config_dict.get('use_hierarchical', False) else '✗ OFF'}"
    )

    print(f"\n  LSTM Settings:")
    print(f"    • Hidden dim:    {config_dict.get('hidden_dim', 64)}")
    print(f"    • Num layers:    {config_dict.get('num_layers', 1)}")
    print(f"    • Dropout:       {config_dict.get('dropout', 0.1)}")

    if config_dict.get("use_attention", False):
        print(f"\n  Attention Settings:")
        print(f"    • Num heads:     {config_dict.get('num_heads', 8)}")


def print_hard_constraints():
    """Print hard constraints being enforced."""
    print("\n  Hard Constraints (enforced globally):")
    print(f"    • Lookback:      {HARD_CONSTRAINTS['lookback']} days")
    print(f"    • Batch size:    {HARD_CONSTRAINTS['batch_size']}")
    print(f"    • Learning rate: {HARD_CONSTRAINTS['learning_rate']}")
    print(f"    • Epochs:        {HARD_CONSTRAINTS['epochs']}")
    print(f"    • Optimizer:     {HARD_CONSTRAINTS['optimizer']}")
    print(f"    • Loss:          {HARD_CONSTRAINTS['loss']}")
    print(f"    • Random seed:   {HARD_CONSTRAINTS['random_seed']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SeqLSTM model with config")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--station", type=str, required=True, help="Path to station CSV file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs", help="Base directory for outputs"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="streamflow_final",
        help="Target variable column name",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature columns (auto-detected if not provided)",
    )
    parser.add_argument(
        "--station_name", type=str, default=None, help="Station name for logging"
    )
    return parser.parse_args()


# Pre-computed hierarchical feature column names
HIERARCHICAL_FEATURE_COLS = [
    "hierarchical_feature_quantile_0.1",
    "hierarchical_feature_quantile_0.5",
    "hierarchical_feature_quantile_0.9",
    "hierarchical_feature_quantile_0.95",
]


def detect_hierarchical_features(df) -> list:
    """Detect if pre-computed hierarchical features exist in dataframe."""
    available = [col for col in HIERARCHICAL_FEATURE_COLS if col in df.columns]
    return available


def get_default_features(target: str, df=None, use_hierarchical: bool = False) -> list:
    """
    Get default feature list based on target type and config.

    Args:
        target: Target variable name
        df: DataFrame to check for available columns
        use_hierarchical: Whether to include hierarchical features (from config)

    Returns:
        List of feature column names
    """
    base_features = [
        "rainfall",
        "tmax",
        "tmin",
        "waterlevel_upstream",
        "streamflow_upstream",
    ]

    if "streamflow" in target.lower():
        features = base_features + ["waterlevel_final", "streamflow_final"]
    else:
        features = base_features + ["waterlevel_final"]

    # Add pre-computed hierarchical features ONLY if enabled in config
    if use_hierarchical and df is not None:
        hier_features = detect_hierarchical_features(df)
        if hier_features:
            features = features + hier_features
            print(f"    ✓ Including {len(hier_features)} hierarchical features")
        else:
            print(f"    ⚠ use_hierarchical=True but no pre-computed features found")

    return features


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    has_wl_targets: bool = False,
    wl_target_loader=None,
) -> dict:
    """
    Train for one epoch with progress bar.

    Args:
        has_wl_targets: If True, model outputs include 'wl_t1', 'wl_t2', 'wl_t3'
                       and we add an auxiliary water level loss.
        wl_target_loader: Not used (WL targets come via the dataset when needed).

    Returns:
        Dict with loss and predictions for metrics
    """
    model.train()
    total_loss = 0
    all_preds = {"t1": [], "t2": [], "t3": []}
    all_targets = {"t1": [], "t2": [], "t3": []}

    pbar = tqdm(
        dataloader,
        desc=f"  Epoch {epoch + 1:2d}/{total_epochs}",
        leave=False,
        ncols=100,
    )

    for batch_idx, (batch_x, batch_y) in enumerate(pbar):
        batch_x = batch_x.to(device)

        # Move targets to device
        targets = {k: v.to(device) for k, v in batch_y.items()}

        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch_x)

        # Compute loss for all horizons
        loss = 0
        for horizon in ["t1", "t2", "t3"]:
            horizon_loss = criterion(predictions[horizon], targets[horizon])
            loss += horizon_loss

            all_preds[horizon].extend(predictions[horizon].detach().cpu().numpy())
            all_targets[horizon].extend(targets[horizon].cpu().numpy())

        loss = loss / 3  # Average across horizons

        # Add auxiliary water level loss if fitter mode is active
        # and WL targets are available in the batch
        if has_wl_targets:
            wl_loss = 0
            wl_count = 0
            for horizon in ["t1", "t2", "t3"]:
                wl_key = f"wl_{horizon}"
                wl_target_key = f"wl_{horizon}"
                if wl_key in predictions and wl_target_key in targets:
                    wl_loss += criterion(predictions[wl_key], targets[wl_target_key])
                    wl_count += 1
            if wl_count > 0:
                # Weight the WL auxiliary loss (0.3x the main loss)
                loss += 0.3 * (wl_loss / wl_count)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)

    # Compute metrics per horizon
    metrics = {}
    for horizon in ["t1", "t2", "t3"]:
        preds = np.array(all_preds[horizon])
        trues = np.array(all_targets[horizon])
        metrics[horizon] = compute_metrics(trues, preds)

    return {"loss": avg_loss, "metrics": metrics}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    scaler=None,
    target_idx: int = None,
    desc: str = "Evaluating",
) -> dict:
    """
    Evaluate model on dataset.

    Returns comprehensive metrics including extreme flow analysis.
    """
    model.eval()

    all_preds = {"t1": [], "t2": [], "t3": []}
    all_targets = {"t1": [], "t2": [], "t3": []}

    pbar = tqdm(dataloader, desc=f"  {desc}", leave=False, ncols=100)

    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        predictions = model(batch_x)

        for horizon in ["t1", "t2", "t3"]:
            all_preds[horizon].extend(predictions[horizon].cpu().numpy())
            all_targets[horizon].extend(batch_y[horizon].numpy())

    results = {}

    for horizon in ["t1", "t2", "t3"]:
        preds = np.array(all_preds[horizon])
        trues = np.array(all_targets[horizon])

        # Inverse transform if scaler provided
        if scaler is not None and target_idx is not None:
            n = len(preds)
            # Create dummy arrays for inverse transform
            dummy_pred = np.zeros((n, scaler.n_features_in_))
            dummy_true = np.zeros((n, scaler.n_features_in_))
            dummy_pred[:, target_idx] = preds
            dummy_true[:, target_idx] = trues

            preds = scaler.inverse_transform(dummy_pred)[:, target_idx]
            trues = scaler.inverse_transform(dummy_true)[:, target_idx]

        results[horizon] = {
            "metrics": compute_metrics(trues, preds),
            "extreme": compute_extreme_metrics(trues, preds, [90, 95]),
            "peak_capture": compute_peak_capture(trues, preds),
            "predictions": preds.tolist(),
            "ground_truth": trues.tolist(),
        }

    return results


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  SeqLSTM TRAINING PIPELINE")
    print("  Modular Ablation Framework")
    print("=" * 60)

    # ==========================================
    # STEP 1: Load Configuration
    # ==========================================
    print_step(1, "LOADING CONFIGURATION")

    print(f"  Config file: {args.config}")
    config_dict = load_config(args.config)
    config_name = Path(args.config).stem
    print(f"  Config name: {config_name}")

    print_config_summary(config_dict)
    print_hard_constraints()

    # Set seed
    print_substep(f"Setting random seed: {HARD_CONSTRAINTS['random_seed']}")
    set_seed(HARD_CONSTRAINTS["random_seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_substep(f"Using device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    # ==========================================
    # STEP 2: Load and Prepare Data
    # ==========================================
    print_step(2, "LOADING AND PREPARING DATA")

    print_substep(f"Loading station data: {args.station}")
    raw_df = pd.read_csv(args.station, parse_dates=["date"], dayfirst=True)
    print(f"    Raw data shape: {raw_df.shape}")
    print(f"    Date range: {raw_df['date'].min()} to {raw_df['date'].max()}")

    # Detect pre-computed hierarchical features
    hier_features = detect_hierarchical_features(raw_df)
    has_precomputed_hier = len(hier_features) > 0
    use_hierarchical = config_dict.get("use_hierarchical", False)

    if has_precomputed_hier:
        print_substep(f"Detected pre-computed hierarchical features in CSV")
        for f in hier_features:
            print(f"    • {f}")
        if use_hierarchical:
            print(f"    ✓ Will be INCLUDED (use_hierarchical=True)")
        else:
            print(f"    ✗ Will be EXCLUDED (use_hierarchical=False)")

    # Determine features
    print_substep("Determining feature set")
    if args.features:
        features = [f.strip() for f in args.features.split(",")]
        print(f"    Using user-specified features")
    else:
        features = get_default_features(
            args.target,
            df=raw_df if has_precomputed_hier else None,
            use_hierarchical=use_hierarchical,
        )
        print(f"    Using auto-detected features")

    print(f"    Total features: {len(features)}")
    for i, f in enumerate(features):
        print(f"      {i + 1:2d}. {f}")
    print(f"    Target: {args.target}")

    # Determine if we need water level targets for fitter training
    is_streamflow = "streamflow" in args.target.lower()
    use_fitter = config_dict.get("use_fitter", False)
    need_fitter = is_streamflow and use_fitter

    # Load and prepare data with determined features
    print_substep("Loading processed data")
    df = load_station_data(args.station, features, args.target)

    station_name = args.station_name or Path(args.station).stem
    print(f"    Station: {station_name}")
    print(f"    Processed shape: {df.shape}")

    # Split data
    print_substep("Splitting data by year")
    splits = split_data(df)
    print(f"    Train (1971-2010):        {len(splits['train']):,} samples")
    print(f"    Test (1961-1970):         {len(splits['test_first']):,} samples")
    print(f"    Test (2011-2020):         {len(splits['test_last']):,} samples")

    # Scale features
    print_substep("Scaling features (MinMaxScaler)")
    train_df, [test_first_df, test_last_df], scaler = scale_features(
        splits["train"], [splits["test_first"], splits["test_last"]], features
    )
    print(f"    Scaler fitted on {len(features)} features")

    # Get target index for inverse scaling
    target_idx = features.index(args.target) if args.target in features else None

    # Create datasets
    print_substep("Creating PyTorch datasets")
    lookback = HARD_CONSTRAINTS["lookback"]

    # If fitter mode, we need a dataset that also provides water level targets
    if need_fitter:
        from utils import MultiHorizonDualTargetDataset

        wl_target_col = "waterlevel_final"

        train_dataset = MultiHorizonDualTargetDataset(
            train_df, features, args.target, wl_target_col, lookback
        )
        test_first_dataset = MultiHorizonDualTargetDataset(
            test_first_df, features, args.target, wl_target_col, lookback
        )
        test_last_dataset = MultiHorizonDualTargetDataset(
            test_last_df, features, args.target, wl_target_col, lookback
        )
        has_wl_targets = True
    else:
        train_dataset = MultiHorizonDataset(train_df, features, args.target, lookback)
        test_first_dataset = MultiHorizonDataset(
            test_first_df, features, args.target, lookback
        )
        test_last_dataset = MultiHorizonDataset(
            test_last_df, features, args.target, lookback
        )
        has_wl_targets = False

    print(f"    Lookback window: {lookback} days (t-{lookback} to t-1)")
    print(f"    Train dataset:   {len(train_dataset):,} sequences")
    print(f"    Test 61-70:      {len(test_first_dataset):,} sequences")
    print(f"    Test 11-20:      {len(test_last_dataset):,} sequences")
    if has_wl_targets:
        print(f"    ✓ Dual-target mode: streamflow + water level targets")

    # Create dataloaders
    print_substep("Creating DataLoaders")
    batch_size = HARD_CONSTRAINTS["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_first_loader = DataLoader(
        test_first_dataset, batch_size=batch_size, shuffle=False
    )
    test_last_loader = DataLoader(
        test_last_dataset, batch_size=batch_size, shuffle=False
    )

    print(f"    Batch size: {batch_size}")
    print(f"    Train batches: {len(train_loader)}")

    # ==========================================
    # STEP 3: Setup Output Directory
    # ==========================================
    print_step(3, "SETTING UP OUTPUT DIRECTORY")

    run_dir = create_run_directory(config_name, args.output_dir)
    print(f"  Run directory: {run_dir}")

    # ==========================================
    # STEP 4: Fit Rating Curve (if applicable)
    # ==========================================
    rating_curve = None

    if need_fitter:
        print_step(4, "FITTING RATING CURVE")

        rating_curve = RatingCurveFitter()

        print_substep("Extracting water level and streamflow data")
        wl = splits["train"]["waterlevel_final"].values
        sf = splits["train"]["streamflow_final"].values
        print(f"    Data points: {len(wl):,}")

        print_substep("Fitting rating curve models")
        if rating_curve.fit(wl, sf):
            rating_curve_path = os.path.join(run_dir, "rating_curve.pkl")
            rating_curve.save(rating_curve_path)
            config_dict["rating_curve_path"] = rating_curve_path
            print(f"    ✓ Best model: {rating_curve.best_model['name']}")
            print(f"    ✓ Rating curve saved: {rating_curve_path}")
        else:
            print(f"    ✗ Rating curve fitting failed, proceeding without fitter")
            need_fitter = False
            has_wl_targets = False
            config_dict["use_fitter"] = False
    else:
        print_step(4, "RATING CURVE (SKIPPED)")
        if not is_streamflow:
            print(f"  Target is water level — fitter not applicable")
        elif not use_fitter:
            print(f"  Fitter disabled in config")

    # ==========================================
    # STEP 5: Create Model
    # ==========================================
    print_step(5, "CREATING MODEL")

    config_dict["input_dim"] = len(features)
    config_dict["is_streamflow"] = is_streamflow

    print_substep("Initializing SeqLSTMModel")
    model_config = ModelConfig.from_dict(config_dict)
    model = SeqLSTMModel(model_config).to(device)

    # If fitter was fitted, inject it into the model
    if need_fitter and rating_curve is not None and rating_curve.fitted:
        model.set_fitter(rating_curve)
        print(f"    ✓ Rating curve injected into model")

    print(f"    Input dimension: {len(features)}")
    print(f"    Model variant:   {model.get_config_summary()}")

    # Count parameters
    print_substep("Counting parameters & FLOPs")
    param_counts = model.count_parameters()
    print(f"    Total parameters: {param_counts['total']:,}")
    for component, count in param_counts.items():
        if count > 0 and component != "total":
            print(f"      • {component}: {count:,}")

    # Compute FLOPs
    lookback = HARD_CONSTRAINTS["lookback"]
    dummy_input = torch.randn(1, lookback, len(features)).to(device)
    try:
        from thop import profile, clever_format
        model.eval()
        with torch.no_grad():
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        model.train()
        flops_str, _ = clever_format([flops, 0], "%.3f")
        print(f"    FLOPs (per forward pass): {flops_str}")
    except Exception as e:
        print(f"    FLOPs: could not compute via thop ({e})")

    # ==========================================
    # STEP 6: Training
    # ==========================================
    print_step(6, "TRAINING MODEL")

    print_substep("Setting up optimizer and loss function")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=HARD_CONSTRAINTS["learning_rate"]
    )
    print(f"    Optimizer: Adam (lr={HARD_CONSTRAINTS['learning_rate']})")
    print(f"    Loss: MSELoss")

    epochs = HARD_CONSTRAINTS["epochs"]
    best_nse = -np.inf
    best_state = None
    train_history = []

    print_substep(f"Starting training loop ({epochs} epochs)")
    print(f"    Progress will be shown every 5 epochs\n")

    for epoch in range(epochs):
        result = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            epochs,
            has_wl_targets=has_wl_targets,
        )
        train_history.append(result)

        # Track best model by T+1 NSE
        t1_nse = result["metrics"]["t1"]["NSE"]
        improved = ""
        if t1_nse > best_nse:
            best_nse = t1_nse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = " ★ NEW BEST"

        # Print every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            t2_nse = result["metrics"]["t2"]["NSE"]
            t3_nse = result["metrics"]["t3"]["NSE"]
            print(
                f"  Epoch {epoch + 1:2d}/{epochs} │ Loss: {result['loss']:.4f} │ "
                f"T+1: {t1_nse:.4f} │ T+2: {t2_nse:.4f} │ T+3: {t3_nse:.4f}{improved}"
            )

    # Load best model
    print_substep("Loading best model checkpoint")
    if best_state:
        model.load_state_dict(best_state)
        print(f"    Best T+1 NSE: {best_nse:.4f}")

    # Re-inject fitter after loading state dict (it's not saved in state_dict)
    if need_fitter and rating_curve is not None and rating_curve.fitted:
        model.set_fitter(rating_curve)

    # Save model
    print_substep("Saving model")
    model_path = os.path.join(run_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"    Model saved: {model_path}")

    # ==========================================
    # STEP 7: Evaluation
    # ==========================================
    print_step(7, "EVALUATING MODEL")

    print_substep("Evaluating on Test Set 1961-1970")
    test_first_results = evaluate(
        model, test_first_loader, device, scaler, target_idx, desc="Testing 1961-1970"
    )

    print("\n    Results (1961-1970):")
    print(
        f"    {'Horizon':<8} {'NSE':>8} {'R²':>8} {'PBIAS':>10} │ {'P90':>8} {'P95':>8} │ {'Peak%':>8}"
    )
    print(f"    {'-' * 70}")
    for h in ["t1", "t2", "t3"]:
        m = test_first_results[h]["metrics"]
        e = test_first_results[h]["extreme"]
        p = test_first_results[h]["peak_capture"]
        print(
            f"    {h.upper():<8} {m['NSE']:>8.4f} {m['R2']:>8.4f} {m['PBIAS']:>9.2f}% │ "
            f"{e['p90']['NSE']:>8.4f} {e['p95']['NSE']:>8.4f} │ {p['mean']:>7.1f}%"
        )

    print_substep("Evaluating on Test Set 2011-2020")
    test_last_results = evaluate(
        model, test_last_loader, device, scaler, target_idx, desc="Testing 2011-2020"
    )

    print("\n    Results (2011-2020):")
    print(
        f"    {'Horizon':<8} {'NSE':>8} {'R²':>8} {'PBIAS':>10} │ {'P90':>8} {'P95':>8} │ {'Peak%':>8}"
    )
    print(f"    {'-' * 70}")
    for h in ["t1", "t2", "t3"]:
        m = test_last_results[h]["metrics"]
        e = test_last_results[h]["extreme"]
        p = test_last_results[h]["peak_capture"]
        print(
            f"    {h.upper():<8} {m['NSE']:>8.4f} {m['R2']:>8.4f} {m['PBIAS']:>9.2f}% │ "
            f"{e['p90']['NSE']:>8.4f} {e['p95']['NSE']:>8.4f} │ {p['mean']:>7.1f}%"
        )

    # ==========================================
    # STEP 8: Save Results
    # ==========================================
    print_step(8, "SAVING RESULTS")

    train_metrics = {
        "final_loss": train_history[-1]["loss"],
        "best_nse": best_nse,
        "metrics_per_horizon": train_history[-1]["metrics"],
    }

    test_metrics = {"first_10": test_first_results, "last_10": test_last_results}

    print_substep("Saving results.json")
    save_results(
        run_dir=run_dir,
        config=config_dict,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        model_info={
            "param_counts": param_counts,
            "config_name": config_name,
            "station": station_name,
            "target": args.target,
            "model_variant": model.get_config_summary(),
        },
    )

    # Save training history
    print_substep("Saving train_history.json")
    with open(os.path.join(run_dir, "train_history.json"), "w") as f:
        json.dump(train_history, f, indent=2, default=str)

    print(f"\n  All results saved to: {run_dir}")

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "=" * 60)
    print("  ✓ TRAINING COMPLETE")
    print("=" * 60)

    print(f"\n  Configuration:  {config_name}")
    print(f"  Station:        {station_name}")
    print(f"  Target:         {args.target}")
    print(f"  Parameters:     {param_counts['total']:,}")
    print(f"  Architecture:   {model.get_config_summary()}")

    print(f"\n  Best Results (2011-2020):")
    for h in ["t1", "t2", "t3"]:
        nse = test_last_results[h]["metrics"]["NSE"]
        print(f"    {h.upper()} NSE: {nse:.4f}")

    print(f"\n  Output: {run_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
