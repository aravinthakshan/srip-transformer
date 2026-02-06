#!/usr/bin/env python
"""
Training Script
===============
Unified config-driven training loop for ablation experiments.

Usage:
    python train.py --config configs/full.yaml --station path/to/station.csv
    python train.py --config configs/baseline.yaml --station path/to/station.csv --output_dir runs
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
    load_config, set_seed, load_station_data, split_data, scale_features,
    create_run_directory, save_results, compute_metrics, compute_extreme_metrics,
    compute_peak_capture, MultiHorizonDataset, HARD_CONSTRAINTS
)
from models import SeqLSTMModel
from models.full_model import ModelConfig
from models.fitter import RatingCurveFitter
from models.hierarchical_features import HierarchicalFeatures


def parse_args():
    parser = argparse.ArgumentParser(description='Train SeqLSTM model with config')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--station', type=str, required=True,
                        help='Path to station CSV file')
    parser.add_argument('--output_dir', type=str, default='runs',
                        help='Base directory for outputs')
    parser.add_argument('--target', type=str, default='streamflow_final',
                        help='Target variable column name')
    parser.add_argument('--features', type=str, default=None,
                        help='Comma-separated feature columns (auto-detected if not provided)')
    parser.add_argument('--station_name', type=str, default=None,
                        help='Station name for logging')
    return parser.parse_args()


def get_default_features(target: str) -> list:
    """Get default feature list based on target type."""
    base_features = [
        'rainfall', 'tmax', 'tmin',
        'waterlevel_upstream', 'streamflow_upstream'
    ]
    
    if 'streamflow' in target.lower():
        return base_features + ['waterlevel_final', 'streamflow_final']
    else:
        return base_features + ['waterlevel_final']


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """
    Train for one epoch.
    
    Returns:
        Dict with loss and predictions for metrics
    """
    model.train()
    total_loss = 0
    all_preds = {'t1': [], 't2': [], 't3': []}
    all_targets = {'t1': [], 't2': [], 't3': []}
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        
        # Move targets to device
        targets = {k: v.to(device) for k, v in batch_y.items()}
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch_x)
        
        # Compute loss for all horizons
        loss = 0
        for horizon in ['t1', 't2', 't3']:
            horizon_loss = criterion(predictions[horizon], targets[horizon])
            loss += horizon_loss
            
            all_preds[horizon].extend(predictions[horizon].detach().cpu().numpy())
            all_targets[horizon].extend(targets[horizon].cpu().numpy())
            
        loss = loss / 3  # Average across horizons
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    
    # Compute metrics per horizon
    metrics = {}
    for horizon in ['t1', 't2', 't3']:
        preds = np.array(all_preds[horizon])
        trues = np.array(all_targets[horizon])
        metrics[horizon] = compute_metrics(trues, preds)
        
    return {'loss': avg_loss, 'metrics': metrics}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    scaler=None,
    target_idx: int = None
) -> dict:
    """
    Evaluate model on dataset.
    
    Returns comprehensive metrics including extreme flow analysis.
    """
    model.eval()
    
    all_preds = {'t1': [], 't2': [], 't3': []}
    all_targets = {'t1': [], 't2': [], 't3': []}
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        predictions = model(batch_x)
        
        for horizon in ['t1', 't2', 't3']:
            all_preds[horizon].extend(predictions[horizon].cpu().numpy())
            all_targets[horizon].extend(batch_y[horizon].numpy())
            
    results = {}
    
    for horizon in ['t1', 't2', 't3']:
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
            'metrics': compute_metrics(trues, preds),
            'extreme': compute_extreme_metrics(trues, preds, [90, 95]),
            'peak_capture': compute_peak_capture(trues, preds),
            'predictions': preds.tolist(),
            'ground_truth': trues.tolist()
        }
        
    return results


def main():
    args = parse_args()
    
    # Load config
    print(f"Loading config: {args.config}")
    config_dict = load_config(args.config)
    config_name = Path(args.config).stem
    
    # Set seed
    set_seed(HARD_CONSTRAINTS['random_seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine features
    if args.features:
        features = [f.strip() for f in args.features.split(',')]
    else:
        features = get_default_features(args.target)
        
    print(f"Features: {features}")
    print(f"Target: {args.target}")
    
    # Load and prepare data
    print(f"Loading station data: {args.station}")
    df = load_station_data(args.station, features, args.target)
    
    station_name = args.station_name or Path(args.station).stem
    print(f"Station: {station_name}")
    
    # Split data
    splits = split_data(df)
    print(f"Train samples: {len(splits['train'])}")
    print(f"Test (1961-1970): {len(splits['test_first'])}")
    print(f"Test (2011-2020): {len(splits['test_last'])}")
    
    # Scale features
    train_df, [test_first_df, test_last_df], scaler = scale_features(
        splits['train'], [splits['test_first'], splits['test_last']], features
    )
    
    # Get target index for inverse scaling
    target_idx = features.index(args.target) if args.target in features else None
    
    # Create datasets
    lookback = HARD_CONSTRAINTS['lookback']
    
    train_dataset = MultiHorizonDataset(train_df, features, args.target, lookback)
    test_first_dataset = MultiHorizonDataset(test_first_df, features, args.target, lookback)
    test_last_dataset = MultiHorizonDataset(test_last_df, features, args.target, lookback)
    
    # Create dataloaders
    batch_size = HARD_CONSTRAINTS['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_first_loader = DataLoader(test_first_dataset, batch_size=batch_size, shuffle=False)
    test_last_loader = DataLoader(test_last_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    
    # Create run directory
    run_dir = create_run_directory(config_name, args.output_dir)
    print(f"Run directory: {run_dir}")
    
    # ==========================
    # Fit Rating Curve (if streamflow)
    # ==========================
    rating_curve = None
    rating_curve_path = None
    
    if 'streamflow' in args.target.lower() and config_dict.get('use_fitter', True):
        print("\nFitting rating curve...")
        rating_curve = RatingCurveFitter()
        
        # Use unscaled data for rating curve
        wl = splits['train']['waterlevel_final'].values
        sf = splits['train']['streamflow_final'].values
        
        if rating_curve.fit(wl, sf):
            rating_curve_path = os.path.join(run_dir, 'rating_curve.pkl')
            rating_curve.save(rating_curve_path)
            config_dict['rating_curve_path'] = rating_curve_path
            print(f"Rating curve saved: {rating_curve_path}")
            
    # ==========================
    # Fit Hierarchical Model (if enabled)
    # ==========================
    hier_model_path = None
    
    if config_dict.get('use_hierarchical', False):
        print("\nFitting hierarchical model...")
        hier = HierarchicalFeatures()
        
        # Define lagged columns
        lag = 1  # 1-day lag for upstream
        
        # Check if required columns exist
        required_cols = ['streamflow_upstream', 'rainfall', 'tmax', 'tmin', 'waterlevel_upstream']
        if all(c in splits['train'].columns for c in required_cols):
            try:
                hier.fit(
                    df=splits['train'],
                    target_col=args.target,
                    feeder_col='streamflow_upstream',
                    rainfall_col='rainfall',
                    covariate_cols=['tmax', 'tmin', 'waterlevel_upstream']
                )
                hier_model_path = os.path.join(run_dir, 'hierarchical_model.pkl')
                hier.save(hier_model_path)
                config_dict['hierarchical_model_path'] = hier_model_path
                print(f"Hierarchical model saved: {hier_model_path}")
            except Exception as e:
                print(f"Warning: Hierarchical model fitting failed: {e}")
                config_dict['use_hierarchical'] = False
        else:
            print("Warning: Missing columns for hierarchical model. Disabling.")
            config_dict['use_hierarchical'] = False
            
    # ==========================
    # Create Model
    # ==========================
    config_dict['input_dim'] = len(features)
    config_dict['is_streamflow'] = 'streamflow' in args.target.lower()
    
    model_config = ModelConfig.from_dict(config_dict)
    model = SeqLSTMModel(model_config).to(device)
    
    # Count parameters
    param_counts = model.count_parameters()
    print(f"\nModel parameters: {param_counts['total']:,}")
    for component, count in param_counts.items():
        if count > 0 and component != 'total':
            print(f"  {component}: {count:,}")
            
    # ==========================
    # Training
    # ==========================
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=HARD_CONSTRAINTS['learning_rate'])
    
    epochs = HARD_CONSTRAINTS['epochs']
    best_nse = -np.inf
    best_state = None
    train_history = []
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in tqdm(range(epochs), desc="Training"):
        result = train_epoch(model, train_loader, optimizer, criterion, device)
        train_history.append(result)
        
        # Track best model by T+1 NSE
        t1_nse = result['metrics']['t1']['NSE']
        if t1_nse > best_nse:
            best_nse = t1_nse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Loss: {result['loss']:.4f}")
            for h in ['t1', 't2', 't3']:
                print(f"  {h.upper()} NSE: {result['metrics'][h]['NSE']:.4f}")
                
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model (NSE: {best_nse:.4f})")
        
    # Save model
    model_path = os.path.join(run_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    
    # ==========================
    # Evaluation
    # ==========================
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    print("\nEvaluating on Test Set (1961-1970)...")
    test_first_results = evaluate(model, test_first_loader, device, scaler, target_idx)
    
    print("\nEvaluating on Test Set (2011-2020)...")
    test_last_results = evaluate(model, test_last_loader, device, scaler, target_idx)
    
    # Print results
    for name, results in [("1961-1970", test_first_results), ("2011-2020", test_last_results)]:
        print(f"\n--- {name} ---")
        for horizon in ['t1', 't2', 't3']:
            m = results[horizon]['metrics']
            e90 = results[horizon]['extreme']['p90']['NSE']
            e95 = results[horizon]['extreme']['p95']['NSE']
            peak = results[horizon]['peak_capture']['mean']
            print(f"  {horizon.upper()}: NSE={m['NSE']:.4f}, P90-NSE={e90:.4f}, P95-NSE={e95:.4f}, Peak={peak:.1f}%")
            
    # ==========================
    # Save Results
    # ==========================
    train_metrics = {
        'final_loss': train_history[-1]['loss'],
        'best_nse': best_nse,
        'metrics_per_horizon': train_history[-1]['metrics']
    }
    
    test_metrics = {
        'first_10': test_first_results,
        'last_10': test_last_results
    }
    
    save_results(
        run_dir=run_dir,
        config=config_dict,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        model_info={
            'param_counts': param_counts,
            'config_name': config_name,
            'station': station_name,
            'target': args.target
        }
    )
    
    # Save training history
    with open(os.path.join(run_dir, 'train_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2, default=str)
        
    print(f"\nResults saved to: {run_dir}")
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Final summary
    print(f"\nConfig: {config_name}")
    print(f"Model: {model.get_config_summary()}")
    print(f"Parameters: {param_counts['total']:,}")
    
    for name, results in [("1961-1970", test_first_results), ("2011-2020", test_last_results)]:
        print(f"\n{name} Results:")
        for h in ['t1', 't2', 't3']:
            nse = results[h]['metrics']['NSE']
            print(f"  {h.upper()} NSE: {nse:.4f}")


if __name__ == '__main__':
    main()
