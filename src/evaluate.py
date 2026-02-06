#!/usr/bin/env python
"""
Evaluation Script
=================
Evaluate trained models and generate paper-ready outputs.
Uses matplotlib/seaborn for plotting (similar to plots.py style).

Usage:
    python evaluate.py --run_dir runs/full_20260206_183000
    python evaluate.py --compare runs/baseline* runs/full* --output comparison.csv
"""

import os
import sys
import argparse
import json
import glob
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--run_dir', type=str, default=None,
                        help='Single run directory to evaluate')
    parser.add_argument('--compare', nargs='+', default=None,
                        help='Glob patterns for runs to compare')
    parser.add_argument('--output', type=str, default='comparison.csv',
                        help='Output file for comparison')
    parser.add_argument('--plots', action='store_true',
                        help='Generate paper-ready plots')
    parser.add_argument('--output_dir', type=str, default='ablation_plots',
                        help='Directory for plot outputs')
    return parser.parse_args()


def load_run_results(run_dir: str) -> Dict[str, Any]:
    """Load results from a run directory."""
    results_path = os.path.join(run_dir, 'results.json')
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results.json found in {run_dir}")
        
    with open(results_path, 'r') as f:
        return json.load(f)


def extract_metrics_for_comparison(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics for ablation table."""
    metrics = {
        'config': results.get('model_info', {}).get('config_name', 'unknown')
    }
    
    # Extract test metrics for both periods
    for period, period_name in [('first_10', '61-70'), ('last_10', '11-20')]:
        period_results = results.get('test_metrics', {}).get(period, {})
        
        for horizon in ['t1', 't2', 't3']:
            h_results = period_results.get(horizon, {})
            h_metrics = h_results.get('metrics', {})
            h_extreme = h_results.get('extreme', {})
            h_peak = h_results.get('peak_capture', {})
            
            prefix = f"{horizon}_{period_name}"
            metrics[f'{prefix}_NSE'] = h_metrics.get('NSE', np.nan)
            metrics[f'{prefix}_P90'] = h_extreme.get('p90', {}).get('NSE', np.nan)
            metrics[f'{prefix}_P95'] = h_extreme.get('p95', {}).get('NSE', np.nan)
            metrics[f'{prefix}_Peak'] = h_peak.get('mean', np.nan)
            
    # Model info
    model_info = results.get('model_info', {})
    metrics['params'] = model_info.get('param_counts', {}).get('total', np.nan)
    
    # Config flags
    config = results.get('config', {})
    metrics['sequential'] = config.get('sequential', False)
    metrics['attention'] = config.get('use_attention', False)
    metrics['hierarchical'] = config.get('use_hierarchical', False)
    metrics['fitter'] = config.get('use_fitter', False)
    
    return metrics


def compare_runs(run_dirs: List[str]) -> pd.DataFrame:
    """Compare multiple runs and create ablation table."""
    all_metrics = []
    
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            # Handle glob patterns
            matching_dirs = glob.glob(run_dir)
            for d in matching_dirs:
                if os.path.isdir(d):
                    try:
                        results = load_run_results(d)
                        metrics = extract_metrics_for_comparison(results)
                        metrics['run_dir'] = d
                        all_metrics.append(metrics)
                    except Exception as e:
                        print(f"Warning: Failed to load {d}: {e}")
        else:
            try:
                results = load_run_results(run_dir)
                metrics = extract_metrics_for_comparison(results)
                metrics['run_dir'] = run_dir
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Warning: Failed to load {run_dir}: {e}")
                
    return pd.DataFrame(all_metrics)


def print_ablation_table(df: pd.DataFrame):
    """Print ablation table to terminal in a nice format."""
    
    # Sort by config name to match ablation matrix order
    config_order = [
        'baseline', 'seq', 'seq_mha', 'seq_mha_no_hier',
        'seq_mha_no_fitter', 'full', 'wide_lstm'
    ]
    
    df = df.copy()
    df['sort_order'] = df['config'].map(
        {c: i for i, c in enumerate(config_order)}
    ).fillna(100)
    df = df.sort_values('sort_order')
    
    print("\n" + "="*100)
    print(" " * 30 + "ABLATION STUDY RESULTS")
    print("="*100)
    
    # Header
    print(f"\n{'Config':<20} {'Seq':^5} {'MHA':^5} {'Hier':^5} {'Fit':^5} │ "
          f"{'T+1':^7} {'T+2':^7} {'T+3':^7} │ "
          f"{'P90-1':^7} {'P90-2':^7} {'P90-3':^7} │ {'Params':>10}")
    print("-"*100)
    
    for _, row in df.iterrows():
        seq = "✓" if row['sequential'] else "✗"
        mha = "✓" if row['attention'] else "✗"
        hier = "✓" if row['hierarchical'] else "✗"
        fit = "✓" if row['fitter'] else "✗"
        
        t1 = f"{row['t1_11-20_NSE']:.3f}" if not np.isnan(row.get('t1_11-20_NSE', np.nan)) else "  -  "
        t2 = f"{row['t2_11-20_NSE']:.3f}" if not np.isnan(row.get('t2_11-20_NSE', np.nan)) else "  -  "
        t3 = f"{row['t3_11-20_NSE']:.3f}" if not np.isnan(row.get('t3_11-20_NSE', np.nan)) else "  -  "
        
        p1 = f"{row['t1_11-20_P90']:.3f}" if not np.isnan(row.get('t1_11-20_P90', np.nan)) else "  -  "
        p2 = f"{row['t2_11-20_P90']:.3f}" if not np.isnan(row.get('t2_11-20_P90', np.nan)) else "  -  "
        p3 = f"{row['t3_11-20_P90']:.3f}" if not np.isnan(row.get('t3_11-20_P90', np.nan)) else "  -  "
        
        params = f"{int(row['params']):,}" if not np.isnan(row.get('params', np.nan)) else "  -  "
        
        print(f"{row['config']:<20} {seq:^5} {mha:^5} {hier:^5} {fit:^5} │ "
              f"{t1:^7} {t2:^7} {t3:^7} │ "
              f"{p1:^7} {p2:^7} {p3:^7} │ {params:>10}")
    
    print("-"*100)
    
    # Print statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (2011-2020 Test Period)")
    print("="*60)
    
    for horizon in ['t1', 't2', 't3']:
        col = f'{horizon}_11-20_NSE'
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"\n{horizon.upper()} NSE:")
                print(f"  Best:  {values.max():.4f} ({df.loc[values.idxmax(), 'config']})")
                print(f"  Worst: {values.min():.4f} ({df.loc[values.idxmin(), 'config']})")
                print(f"  Mean:  {values.mean():.4f}")


def plot_nse_comparison(df: pd.DataFrame, output_dir: str):
    """Generate grouped bar plot comparing NSE across models."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style similar to plots.py
    plt.style.use('default')
    sns.set_palette("viridis")
    
    configs = df['config'].tolist()
    
    # Prepare data for plotting
    plot_data = []
    for _, row in df.iterrows():
        for horizon in ['t1', 't2', 't3']:
            for period, period_name in [('11-20', '2011-2020'), ('61-70', '1961-1970')]:
                col = f'{horizon}_{period}_NSE'
                if col in df.columns:
                    plot_data.append({
                        'Config': row['config'],
                        'Horizon': f'T+{horizon[1]}',
                        'Period': period_name,
                        'NSE': row[col]
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create faceted plot by horizon
    g = sns.catplot(
        data=plot_df,
        x='Config',
        y='NSE',
        hue='Period',
        col='Horizon',
        kind='bar',
        height=6, aspect=1.2,
        palette=['#2196F3', '#4CAF50'],
        edgecolor='black',
        linewidth=0.7
    )
    
    g.set_axis_labels("Configuration", "NSE Score", fontsize=14)
    g.set_titles("Forecast Horizon: {col_name}", fontsize=16, fontweight='bold')
    g.set_xticklabels(rotation=45, ha='right', fontsize=11)
    g.set(ylim=(-0.1, 1.05))
    
    g.fig.suptitle('NSE Performance Across Model Configurations', 
                   fontsize=18, fontweight='bold', y=1.02)
    
    # Add reference lines and value labels
    for ax in g.axes.flat:
        ax.axhline(y=0.75, color='forestgreen', linestyle='--', linewidth=1.5, alpha=0.7, label='Very Good')
        ax.axhline(y=0.5, color='darkorange', linestyle=':', linewidth=1.5, alpha=0.7, label='Good')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        # Add value labels on bars
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if not np.isnan(height) and height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'nse_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_extreme_comparison(df: pd.DataFrame, output_dir: str):
    """Generate bar plot for extreme flow NSE (P90 and P95)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    sns.set_palette("plasma")
    
    # Prepare data
    plot_data = []
    for _, row in df.iterrows():
        for horizon in ['t1', 't2', 't3']:
            for pct, pct_name in [('P90', '90th Percentile'), ('P95', '95th Percentile')]:
                col = f'{horizon}_11-20_{pct}'
                if col in df.columns:
                    plot_data.append({
                        'Config': row['config'],
                        'Horizon': f'T+{horizon[1]}',
                        'Percentile': pct_name,
                        'NSE': row[col]
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    g = sns.catplot(
        data=plot_df,
        x='Config',
        y='NSE',
        hue='Percentile',
        col='Horizon',
        kind='bar',
        height=6, aspect=1.2,
        palette=['#FF9800', '#F44336'],
        edgecolor='black',
        linewidth=0.7
    )
    
    g.set_axis_labels("Configuration", "NSE Score", fontsize=14)
    g.set_titles("Extreme Flow NSE - {col_name}", fontsize=16, fontweight='bold')
    g.set_xticklabels(rotation=45, ha='right', fontsize=11)
    
    g.fig.suptitle('Extreme Flow NSE (2011-2020)', fontsize=18, fontweight='bold', y=1.02)
    
    for ax in g.axes.flat:
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if not np.isnan(height) and height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'extreme_nse_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_params_comparison(df: pd.DataFrame, output_dir: str):
    """Generate parameter count comparison bar plot."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    
    # Sort configs
    config_order = ['baseline', 'seq', 'seq_mha', 'seq_mha_no_hier',
                    'seq_mha_no_fitter', 'full', 'wide_lstm']
    df = df.copy()
    df['sort_order'] = df['config'].map({c: i for i, c in enumerate(config_order)}).fillna(100)
    df = df.sort_values('sort_order')
    
    configs = df['config'].tolist()
    params = df['params'].values / 1000  # In thousands
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(configs)))
    bars = ax.bar(configs, params, color=colors, edgecolor='black', linewidth=0.7)
    
    ax.set_ylabel('Parameters (K)', fontsize=14)
    ax.set_xlabel('Configuration', fontsize=14)
    ax.set_title('Model Complexity Comparison', fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add value labels
    for bar, val in zip(bars, params):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}K', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'params_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_ablation_heatmap(df: pd.DataFrame, output_dir: str):
    """Generate heatmap of NSE values across configs and horizons."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for heatmap
    config_order = ['baseline', 'seq', 'seq_mha', 'seq_mha_no_hier',
                    'seq_mha_no_fitter', 'full', 'wide_lstm']
    
    heatmap_data = []
    for config in config_order:
        row_data = df[df['config'] == config]
        if len(row_data) > 0:
            row = row_data.iloc[0]
            heatmap_data.append({
                'Config': config,
                'T+1 (2011-20)': row.get('t1_11-20_NSE', np.nan),
                'T+2 (2011-20)': row.get('t2_11-20_NSE', np.nan),
                'T+3 (2011-20)': row.get('t3_11-20_NSE', np.nan),
                'T+1 P90': row.get('t1_11-20_P90', np.nan),
                'T+2 P90': row.get('t2_11-20_P90', np.nan),
                'T+3 P90': row.get('t3_11-20_P90', np.nan),
            })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index('Config')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0.5, vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'NSE Score'})
    
    ax.set_title('Ablation Study: NSE Heatmap', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Metric', fontsize=14)
    ax.set_ylabel('Configuration', fontsize=14)
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'ablation_heatmap.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_peak_capture(df: pd.DataFrame, output_dir: str):
    """Generate peak capture percentage comparison."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    
    # Prepare data
    plot_data = []
    for _, row in df.iterrows():
        for horizon in ['t1', 't2', 't3']:
            col = f'{horizon}_11-20_Peak'
            if col in df.columns:
                plot_data.append({
                    'Config': row['config'],
                    'Horizon': f'T+{horizon[1]}',
                    'Peak Capture (%)': row[col]
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    g = sns.catplot(
        data=plot_df,
        x='Config',
        y='Peak Capture (%)',
        hue='Horizon',
        kind='bar',
        height=6, aspect=1.8,
        palette='Set2',
        edgecolor='black',
        linewidth=0.7
    )
    
    g.set_axis_labels("Configuration", "Peak Capture (%)", fontsize=14)
    g.fig.suptitle('Peak Capture Performance (2011-2020)', fontsize=18, fontweight='bold', y=1.02)
    
    plt.xticks(rotation=45, ha='right', fontsize=11)
    
    for ax in g.axes.flat:
        ax.axhline(y=80, color='forestgreen', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=60, color='darkorange', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if not np.isnan(height) and height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'peak_capture_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def print_single_run_results(results: Dict[str, Any]):
    """Print results for a single run."""
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    config = results.get('config', {})
    model_info = results.get('model_info', {})
    
    print(f"\nConfiguration: {model_info.get('config_name', 'unknown')}")
    print(f"Station: {model_info.get('station', 'unknown')}")
    print(f"Target: {model_info.get('target', 'unknown')}")
    
    print(f"\nArchitecture:")
    print(f"  Sequential:    {'✓' if config.get('sequential', False) else '✗'}")
    print(f"  Attention:     {'✓' if config.get('use_attention', False) else '✗'}")
    print(f"  Hierarchical:  {'✓' if config.get('use_hierarchical', False) else '✗'}")
    print(f"  Fitter:        {'✓' if config.get('use_fitter', False) else '✗'}")
    
    param_counts = model_info.get('param_counts', {})
    print(f"\nParameters: {param_counts.get('total', 'N/A'):,}")
    
    for period, period_name in [('first_10', '1961-1970'), ('last_10', '2011-2020')]:
        print(f"\n{'='*50}")
        print(f"Test Period: {period_name}")
        print(f"{'='*50}")
        
        period_results = results.get('test_metrics', {}).get(period, {})
        
        print(f"\n{'Horizon':<10} {'NSE':>8} {'R²':>8} {'PBIAS':>8} {'KGE':>8} │ {'P90':>8} {'P95':>8} │ {'Peak%':>8}")
        print("-"*80)
        
        for horizon in ['t1', 't2', 't3']:
            h_results = period_results.get(horizon, {})
            m = h_results.get('metrics', {})
            e = h_results.get('extreme', {})
            p = h_results.get('peak_capture', {})
            
            nse = f"{m.get('NSE', np.nan):.4f}" if not np.isnan(m.get('NSE', np.nan)) else "   -   "
            r2 = f"{m.get('R2', np.nan):.4f}" if not np.isnan(m.get('R2', np.nan)) else "   -   "
            pbias = f"{m.get('PBIAS', np.nan):.2f}" if not np.isnan(m.get('PBIAS', np.nan)) else "   -   "
            kge = f"{m.get('KGE', np.nan):.4f}" if not np.isnan(m.get('KGE', np.nan)) else "   -   "
            p90 = f"{e.get('p90', {}).get('NSE', np.nan):.4f}" if not np.isnan(e.get('p90', {}).get('NSE', np.nan)) else "   -   "
            p95 = f"{e.get('p95', {}).get('NSE', np.nan):.4f}" if not np.isnan(e.get('p95', {}).get('NSE', np.nan)) else "   -   "
            peak = f"{p.get('mean', np.nan):.1f}%" if not np.isnan(p.get('mean', np.nan)) else "   -   "
            
            print(f"{horizon.upper():<10} {nse:>8} {r2:>8} {pbias:>8} {kge:>8} │ {p90:>8} {p95:>8} │ {peak:>8}")


def main():
    args = parse_args()
    
    if args.compare:
        # Compare multiple runs
        print("Comparing runs...")
        all_dirs = []
        for pattern in args.compare:
            matching = glob.glob(pattern)
            all_dirs.extend([d for d in matching if os.path.isdir(d)])
            
        if not all_dirs:
            print("No matching run directories found.")
            return
            
        print(f"Found {len(all_dirs)} runs to compare")
        
        df = compare_runs(all_dirs)
        
        # Save comparison CSV
        df.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")
        
        # Print ablation table to terminal
        print_ablation_table(df)
        
        # Generate plots if requested
        if args.plots:
            print(f"\nGenerating plots in {args.output_dir}/...")
            plot_nse_comparison(df, args.output_dir)
            plot_extreme_comparison(df, args.output_dir)
            plot_params_comparison(df, args.output_dir)
            plot_ablation_heatmap(df, args.output_dir)
            plot_peak_capture(df, args.output_dir)
            print(f"\nAll plots saved to: {args.output_dir}/")
            
    elif args.run_dir:
        # Evaluate single run
        print(f"Evaluating: {args.run_dir}")
        results = load_run_results(args.run_dir)
        print_single_run_results(results)
        
    else:
        print("Specify either --run_dir or --compare")


if __name__ == '__main__':
    main()
