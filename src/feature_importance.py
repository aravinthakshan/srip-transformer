#!/usr/bin/env python
"""
Feature Importance Script
=========================
Computes permutation-based feature importance for a trained SeqLSTM model.

Method: For each feature, shuffle its values across all test samples and
measure the drop in T+1 NSE compared to the baseline. A larger drop means
the feature is more important.

No extra libraries needed (no SHAP, no captum) -- pure PyTorch + numpy.

Usage:
    # Point at a run directory (must contain model.pt + results.json)
    python feature_importance.py \\
        --run_dir runs/baseline_20260304_123456 \\
        --station hierarchial/new_Barmanghat_with_hierarchical_quantiles.csv

    # Repeat shuffle N times and average (more stable)
    python feature_importance.py \\
        --run_dir runs/full_20260304_123456 \\
        --station hierarchial/new_Barmanghat_with_hierarchical_quantiles.csv \\
        --n_repeats 10 \\
        --split test_last          # test_first | test_last | both (default: test_last)

    # Save bar chart
    python feature_importance.py --run_dir runs/... --station ... --plot
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_config,
    load_station_data,
    split_data,
    scale_features,
    MultiHorizonDataset,
    HARD_CONSTRAINTS,
    nse,
)
from models import SeqLSTMModel
from models.full_model import ModelConfig


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Permutation feature importance for SeqLSTM"
    )
    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to a completed run directory (must contain model.pt and results.json)",
    )
    p.add_argument(
        "--station",
        type=str,
        required=True,
        help="Path to the station CSV used for this run",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test_last",
        choices=["test_first", "test_last", "both"],
        help="Which test split to evaluate on (default: test_last)",
    )
    p.add_argument(
        "--n_repeats",
        type=int,
        default=5,
        help="Number of permutation repeats per feature (default: 5)",
    )
    p.add_argument(
        "--horizon",
        type=str,
        default="t1",
        choices=["t1", "t2", "t3"],
        help="Prediction horizon to measure importance on (default: t1)",
    )
    p.add_argument(
        "--plot", action="store_true", help="Save a bar chart of feature importances"
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results CSV (default: <run_dir>/feature_importance.csv)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core permutation logic
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_all(model, loader, device, horizon="t1"):
    """Collect all predictions and ground-truth for the given horizon."""
    model.eval()
    all_preds, all_true = [], []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        out = model(batch_x)
        all_preds.extend(out[horizon].cpu().numpy())
        all_true.extend(batch_y[horizon].numpy())
    return np.array(all_preds), np.array(all_true)


@torch.no_grad()
def permutation_importance(
    model,
    X_tensor,
    y_true,
    feature_names,
    horizon="t1",
    n_repeats=5,
    device=None,
    batch_size=256,
):
    """
    Compute permutation importance.

    Parameters
    ----------
    model       : trained SeqLSTMModel
    X_tensor    : FloatTensor [N, T, F]  (full scaled test set)
    y_true      : np.ndarray [N]         (inverse-scaled ground truth)
    feature_names : list of str          (length F)
    horizon     : 't1' | 't2' | 't3'
    n_repeats   : how many shuffles per feature
    device      : torch.device

    Returns
    -------
    importance  : dict  feature_name -> mean NSE drop across repeats
    std_dev     : dict  feature_name -> std of NSE drop across repeats
    baseline_nse: float
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    def _forward_batch(X):
        """Run model on a float32 numpy array [N, T, F], return predictions."""
        preds = []
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            out = model(batch)
            preds.extend(out[horizon].cpu().numpy())
        return np.array(preds)

    X_np = X_tensor.numpy()  # [N, T, F]

    # Baseline NSE
    baseline_preds = _forward_batch(X_np)
    baseline_nse_val = nse(y_true, baseline_preds)

    n_features = X_np.shape[2]
    importance = {}
    std_dev = {}

    for feat_idx, feat_name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_np.copy()
            # Shuffle this feature across the sample dimension
            perm = np.random.permutation(len(X_perm))
            X_perm[:, :, feat_idx] = X_perm[perm, :, feat_idx]

            perm_preds = _forward_batch(X_perm)
            perm_nse = nse(y_true, perm_preds)
            drops.append(baseline_nse_val - perm_nse)

        importance[feat_name] = float(np.mean(drops))
        std_dev[feat_name] = float(np.std(drops))

    return importance, std_dev, baseline_nse_val


# ---------------------------------------------------------------------------
# Feature detection (same logic as train.py)
# ---------------------------------------------------------------------------

HIERARCHICAL_FEATURE_COLS = [
    "hierarchical_feature_quantile_0.1",
    "hierarchical_feature_quantile_0.5",
    "hierarchical_feature_quantile_0.9",
    "hierarchical_feature_quantile_0.95",
]


def get_features_from_results(results_json: dict, df) -> list:
    """
    Reconstruct the feature list that was used during training,
    reading it from results.json if saved, or falling back to auto-detection.
    """
    # results.json may have a 'features' key saved by train.py
    cfg = results_json.get("config", {})
    use_hierarchical = cfg.get("use_hierarchical", False)
    target = results_json.get("model_info", {}).get("target", "streamflow_final")

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

    if use_hierarchical:
        hier = [c for c in HIERARCHICAL_FEATURE_COLS if c in df.columns]
        features = features + hier

    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    run_dir = Path(args.run_dir)
    model_pt = run_dir / "model.pt"
    results_f = run_dir / "results.json"

    # --- Validate run dir ---
    if not model_pt.exists():
        print(f"[ERROR] model.pt not found in {run_dir}")
        sys.exit(1)
    if not results_f.exists():
        print(f"[ERROR] results.json not found in {run_dir}")
        sys.exit(1)

    with open(results_f) as f:
        results_json = json.load(f)

    config_dict = results_json.get("config", {})
    print(f"\nRun dir  : {run_dir}")
    print(f"Config   : {results_json.get('model_info', {}).get('config_name', '?')}")
    print(f"Station  : {results_json.get('model_info', {}).get('station', '?')}")
    print(
        f"Target   : {results_json.get('model_info', {}).get('target', 'streamflow_final')}"
    )

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")

    # --- Load data ---
    target = results_json.get("model_info", {}).get("target", "streamflow_final")
    raw_df = __import__("pandas").read_csv(
        args.station, parse_dates=["date"], dayfirst=True
    )
    features = get_features_from_results(results_json, raw_df)
    print(f"\nFeatures ({len(features)}):")
    for i, f in enumerate(features):
        print(f"  {i + 1:2d}. {f}")

    df = load_station_data(args.station, features, target)
    splits = split_data(df)
    _, [test_first_df, test_last_df], scaler = scale_features(
        splits["train"], [splits["test_first"], splits["test_last"]], features
    )
    target_idx = features.index(target) if target in features else None

    # Choose split(s)
    lookback = HARD_CONSTRAINTS["lookback"]

    def make_Xy(split_df):
        ds = MultiHorizonDataset(split_df, features, target, lookback)
        X = torch.stack([ds[i][0] for i in range(len(ds))])  # [N, T, F]
        # Inverse-scale the target for interpretable NSE
        raw_preds_placeholder = np.zeros((len(ds), scaler.n_features_in_))
        raw_preds_placeholder[:, target_idx] = np.array(
            [ds[i][1][args.horizon].item() for i in range(len(ds))]
        )
        y = scaler.inverse_transform(raw_preds_placeholder)[:, target_idx]
        return X, y

    splits_to_run = []
    if args.split in ("test_last", "both"):
        splits_to_run.append(("test_last (2011-2020)", test_last_df))
    if args.split in ("test_first", "both"):
        splits_to_run.append(("test_first (1961-1970)", test_first_df))

    # --- Load model ---
    config_dict["input_dim"] = len(features)
    config_dict.setdefault("is_streamflow", "streamflow" in target.lower())
    model_config = ModelConfig.from_dict(config_dict)
    model = SeqLSTMModel(model_config).to(device)
    model.load_state_dict(torch.load(model_pt, map_location=device))
    model.eval()
    print(f"\nModel    : {model.get_config_summary()}")
    print(f"Params   : {sum(p.numel() for p in model.parameters()):,}")

    # --- Run permutation importance for each split ---
    all_results = []

    for split_name, split_df in splits_to_run:
        print(f"\n{'=' * 60}")
        print(f"  Permutation Importance — {split_name}")
        print(f"  Horizon: {args.horizon.upper()}  |  Repeats: {args.n_repeats}")
        print(f"{'=' * 60}")

        X, y_true = make_Xy(split_df)

        importance, std_dev, baseline_nse_val = permutation_importance(
            model,
            X,
            y_true,
            features,
            horizon=args.horizon,
            n_repeats=args.n_repeats,
            device=device,
        )

        print(f"\n  Baseline {args.horizon.upper()} NSE : {baseline_nse_val:.4f}")
        print(f"\n  {'Feature':<45}  {'NSE Drop':>10}  {'Std':>8}  {'Rank':>5}")
        print(f"  {'-' * 75}")

        ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for rank, (feat, drop) in enumerate(ranked, 1):
            bar = (
                "#"
                * max(
                    0,
                    int(drop * 40 / max(v for v in importance.values() if v > 0) + 0.5),
                )
                if any(v > 0 for v in importance.values())
                else ""
            )
            print(
                f"  {feat:<45}  {drop:>+10.4f}  {std_dev[feat]:>8.4f}  {rank:>5}   {bar}"
            )
            all_results.append(
                {
                    "split": split_name,
                    "feature": feat,
                    "nse_drop": drop,
                    "nse_drop_std": std_dev[feat],
                    "rank": rank,
                    "baseline_nse": baseline_nse_val,
                    "horizon": args.horizon,
                }
            )

    # --- Save CSV ---
    import pandas as pd

    out_path = args.output or str(run_dir / "feature_importance.csv")
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(out_path, index=False)
    print(f"\n  Results saved: {out_path}")

    # --- Optional plot ---
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.rcParams["font.family"] = "DejaVu Sans"

            for split_name, _ in splits_to_run:
                sub = df_out[df_out["split"] == split_name].sort_values(
                    "nse_drop", ascending=True
                )
                fig, ax = plt.subplots(figsize=(8, max(4, len(sub) * 0.4)))

                colors = ["#e74c3c" if v >= 0 else "#3498db" for v in sub["nse_drop"]]
                bars = ax.barh(
                    sub["feature"],
                    sub["nse_drop"],
                    xerr=sub["nse_drop_std"],
                    color=colors,
                    edgecolor="white",
                    linewidth=0.5,
                    error_kw=dict(ecolor="gray", capsize=3),
                )

                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel(
                    f"NSE Drop when feature shuffled ({args.horizon.upper()})"
                )
                ax.set_title(
                    f"Permutation Feature Importance\n{run_dir.name} | {split_name}"
                )
                ax.tick_params(axis="y", labelsize=8)
                plt.tight_layout()

                plot_path = str(
                    run_dir
                    / f"feature_importance_{split_name.split()[0]}_{args.horizon}.png"
                )
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  Plot saved : {plot_path}")
        except ImportError:
            print("  [WARN] matplotlib not available, skipping plot")

    print()


if __name__ == "__main__":
    main()
