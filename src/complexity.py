#!/usr/bin/env python
"""
Complexity Analysis Script
==========================
Compute FLOPs and parameter counts for all model configurations.

Only counts LSTM and Attention components (excludes CatBoost and rating curve).

Usage:
    python complexity.py --all-configs
    python complexity.py --config configs/full.yaml
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config, HARD_CONSTRAINTS
from models import SeqLSTMModel
from models.full_model import ModelConfig, get_capacity_matched_hidden_dim

# Try to import profiling tools
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. FLOPs will be estimated manually.")
    print("Install with: pip install thop")


def parse_args():
    parser = argparse.ArgumentParser(description='Model complexity analysis')
    parser.add_argument('--config', type=str, default=None,
                        help='Single config to analyze')
    parser.add_argument('--all-configs', action='store_true',
                        help='Analyze all configs in configs/')
    parser.add_argument('--input_dim', type=int, default=8,
                        help='Number of input features')
    parser.add_argument('--output', type=str, default='complexity_analysis.csv',
                        help='Output CSV file')
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable parameters by component."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    counts = {'total': total}
    
    # Count by named modules
    for name, module in model.named_modules():
        if hasattr(module, 'parameters'):
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if n_params > 0 and '.' not in name:  # Top-level modules only
                counts[name] = n_params
                
    return counts


def estimate_lstm_flops(input_dim: int, hidden_dim: int, seq_len: int, num_layers: int = 1, bidirectional: bool = False) -> int:
    """
    Estimate FLOPs for LSTM forward pass.
    
    LSTM cell FLOPs per timestep ≈ 8 * hidden_dim * (input_dim + hidden_dim)
    Total = seq_len * num_layers * (8 * H * (I + H))
    """
    directions = 2 if bidirectional else 1
    
    # First layer
    flops_first = seq_len * 8 * hidden_dim * (input_dim + hidden_dim)
    
    # Subsequent layers (input is hidden_dim * directions)
    flops_rest = (num_layers - 1) * seq_len * 8 * hidden_dim * (hidden_dim * directions + hidden_dim)
    
    total_flops = directions * (flops_first + flops_rest)
    
    return int(total_flops)


def estimate_attention_flops(hidden_dim: int, seq_len: int, num_heads: int = 8) -> int:
    """
    Estimate FLOPs for Multi-Head Attention.
    
    Attention FLOPs ≈ 4 * seq_len * hidden_dim^2 + 2 * seq_len^2 * hidden_dim
    """
    # Q, K, V projections
    qkv_flops = 3 * seq_len * hidden_dim * hidden_dim
    
    # Attention scores: QK^T
    attn_scores = seq_len * seq_len * hidden_dim
    
    # Attention @ V
    attn_output = seq_len * seq_len * hidden_dim
    
    # Output projection
    out_proj = seq_len * hidden_dim * hidden_dim
    
    total_flops = qkv_flops + attn_scores + attn_output + out_proj
    
    return int(total_flops)


def analyze_model(
    config_dict: Dict[str, Any],
    input_dim: int,
    config_name: str = 'unknown'
) -> Dict[str, Any]:
    """Analyze a single model configuration."""
    
    config_dict['input_dim'] = input_dim
    
    model_config = ModelConfig.from_dict(config_dict)
    model = SeqLSTMModel(model_config)
    
    # Count parameters
    param_counts = count_parameters(model)
    
    # Estimate FLOPs
    lookback = HARD_CONSTRAINTS['lookback']
    hidden_dim = config_dict.get('hidden_dim', 64)
    num_layers = config_dict.get('num_layers', 1)
    bidirectional = config_dict.get('bidirectional', False)
    num_heads = config_dict.get('num_heads', 8)
    
    lstm_flops = estimate_lstm_flops(
        input_dim, hidden_dim, lookback, num_layers, bidirectional
    )
    
    attention_flops = 0
    if config_dict.get('use_attention', False):
        effective_hidden = hidden_dim * (2 if bidirectional else 1)
        attention_flops = estimate_attention_flops(effective_hidden, lookback, num_heads)
        
    # Use thop if available for more accurate count
    total_flops = lstm_flops + attention_flops
    
    if THOP_AVAILABLE:
        try:
            dummy_input = torch.randn(1, lookback, input_dim)
            model.eval()
            with torch.no_grad():
                flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            total_flops = int(flops)
        except Exception as e:
            print(f"Warning: thop profiling failed for {config_name}: {e}")
            
    return {
        'config': config_name,
        'parameters': param_counts['total'],
        'lstm_flops': lstm_flops,
        'attention_flops': attention_flops,
        'total_flops': total_flops,
        'sequential': config_dict.get('sequential', False),
        'attention': config_dict.get('use_attention', False),
        'hierarchical': config_dict.get('use_hierarchical', False),
        'fitter': config_dict.get('use_fitter', False),
        'hidden_dim': hidden_dim,
        'num_layers': num_layers
    }


def compute_capacity_matched_baseline(
    target_config: Dict[str, Any],
    input_dim: int
) -> Dict[str, Any]:
    """
    Compute hidden_dim for capacity-matched baseline.
    
    Creates a baseline LSTM (no attention) with approximately
    the same parameter count as the target configuration.
    """
    # First, get target parameter count
    target_config['input_dim'] = input_dim
    target_model_config = ModelConfig.from_dict(target_config)
    target_model = SeqLSTMModel(target_model_config)
    target_params = sum(p.numel() for p in target_model.parameters() if p.requires_grad)
    
    # Create baseline config (no attention, no hier, no fitter)
    baseline_config = {
        'use_lstm': True,
        'use_attention': False,
        'use_hierarchical': False,
        'use_fitter': False,
        'sequential': False,
        'input_dim': input_dim,
        'num_layers': 1,
        'dropout': 0.1
    }
    
    # Binary search for hidden_dim
    low, high = 32, 512
    best_hidden_dim = 64
    best_diff = float('inf')
    
    while low <= high:
        mid = (low + high) // 2
        baseline_config['hidden_dim'] = mid
        
        test_config = ModelConfig.from_dict(baseline_config)
        test_model = SeqLSTMModel(test_config)
        test_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
        
        diff = abs(test_params - target_params)
        
        if diff < best_diff:
            best_diff = diff
            best_hidden_dim = mid
            
        if test_params < target_params:
            low = mid + 1
        else:
            high = mid - 1
            
    baseline_config['hidden_dim'] = best_hidden_dim
    
    return baseline_config


def main():
    args = parse_args()
    
    results = []
    
    if args.all_configs:
        config_dir = Path(__file__).parent / 'configs'
        config_files = sorted(config_dir.glob('*.yaml'))
        
        if not config_files:
            print(f"No config files found in {config_dir}")
            return
            
        print(f"Analyzing {len(config_files)} configurations...\n")
        
        for config_path in config_files:
            config_name = config_path.stem
            print(f"Analyzing: {config_name}")
            
            config_dict = load_config(str(config_path))
            analysis = analyze_model(config_dict, args.input_dim, config_name)
            results.append(analysis)
            
    elif args.config:
        config_name = Path(args.config).stem
        print(f"Analyzing: {config_name}")
        
        config_dict = load_config(args.config)
        analysis = analyze_model(config_dict, args.input_dim, config_name)
        results.append(analysis)
        
        # Also compute capacity-matched baseline
        print("\nComputing capacity-matched baseline...")
        baseline_config = compute_capacity_matched_baseline(config_dict, args.input_dim)
        baseline_analysis = analyze_model(baseline_config, args.input_dim, 'capacity_matched_baseline')
        results.append(baseline_analysis)
        
    else:
        print("Specify --config or --all-configs")
        return
        
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPLEXITY ANALYSIS SUMMARY")
    print("="*80)
    
    # Format FLOPs nicely
    df_display = df.copy()
    df_display['parameters'] = df_display['parameters'].apply(lambda x: f"{x:,}")
    df_display['total_flops'] = df_display['total_flops'].apply(lambda x: f"{x/1e6:.2f}M")
    
    print(df_display[['config', 'parameters', 'total_flops', 'hidden_dim', 
                      'sequential', 'attention', 'hierarchical', 'fitter']].to_string(index=False))
    
    # Print capacity matching info
    print("\n" + "="*80)
    print("CAPACITY MATCHING")
    print("="*80)
    
    if 'full' in df['config'].values:
        full_params = df[df['config'] == 'full']['parameters'].values[0]
        print(f"Target (full model): {full_params:,} parameters")
        
        if 'wide_lstm' in df['config'].values:
            wide_params = df[df['config'] == 'wide_lstm']['parameters'].values[0]
            wide_hidden = df[df['config'] == 'wide_lstm']['hidden_dim'].values[0]
            print(f"Wide LSTM (capacity-matched): {wide_params:,} parameters")
            print(f"  hidden_dim = {wide_hidden}")
            print(f"  Difference: {abs(full_params - wide_params):,} ({abs(full_params - wide_params)/full_params*100:.1f}%)")


if __name__ == '__main__':
    main()
