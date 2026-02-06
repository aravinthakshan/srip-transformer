# SeqLSTM-MHA: Modular Ablation Framework for Streamflow and Water Level Prediction

A config-driven deep learning framework for multi-horizon hydrological forecasting with ablation study capabilities.

## 🎯 Overview

This framework implements a modular architecture for predicting:
- **Streamflow** (T+1, T+2, T+3 days ahead)
- **Water Level** (T+1, T+2, T+3 days ahead)

The key innovation is a **config-driven ablation system** that allows systematic comparison of architectural components:

| Component | Description |
|-----------|-------------|
| **Sequential Prediction** | Feed T+1 prediction into T+2, T+2 into T+3 |
| **Multi-Head Attention** | Attend over LSTM hidden states |
| **Hierarchical Features** | Pre-computed CatBoost quantile features |
| **Rating Curve Fitter** | Physics-based streamflow estimation |

## 📁 Project Structure

```
srip-transformer/
├── src/
│   ├── models/                 # Core model components
│   │   ├── lstm_backbone.py    # LSTM encoder
│   │   ├── attention.py        # Multi-Head Attention
│   │   ├── sequential_wrapper.py # Sequential vs Direct prediction
│   │   ├── hierarchical_features.py # CatBoost feature extraction
│   │   ├── fitter.py           # Rating curve module
│   │   └── full_model.py       # Model composer
│   ├── configs/                # Ablation configurations
│   │   ├── baseline.yaml       # Direct LSTM (no components)
│   │   ├── seq.yaml            # Sequential only
│   │   ├── seq_mha.yaml        # Sequential + Attention
│   │   ├── seq_mha_no_hier.yaml # Seq + MHA + Fitter
│   │   ├── seq_mha_no_fitter.yaml # Seq + MHA + Hierarchical
│   │   ├── full.yaml           # All components
│   │   └── wide_lstm.yaml      # Capacity-matched baseline
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation & comparison
│   ├── complexity.py           # FLOPs & parameter analysis
│   ├── utils.py                # Shared utilities
│   ├── plots.py                # Visualization
│   └── run_all.py              # Batch experiment runner
├── hierarchial/                # Pre-computed hierarchical features
├── runs/                       # Experiment outputs (gitignored)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Single Experiment

```bash
# Streamflow prediction with full model
python src/train.py --config src/configs/full.yaml \
    --station hierarchial/new_Handia_with_hierarchical_quantiles.csv \
    --target streamflow_final

# Water level prediction with baseline
python src/train.py --config src/configs/baseline.yaml \
    --station hierarchial/new_Handia_with_hierarchical_quantiles.csv \
    --target waterlevel_final
```

### Run All Ablation Experiments

```bash
python src/run_all.py
```

This runs 14 experiments (7 configs × 2 targets).

### Compare Results

```bash
python src/evaluate.py --compare "runs/*_streamflow_*" --output comparison.csv --plots
```

## ⚙️ Configuration System

### Hard Constraints (Fixed Across All Experiments)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Lookback | 7 days | Input window (t-7 to t-1) |
| Batch Size | 256 | |
| Learning Rate | 1e-3 | |
| Epochs | 30 | |
| Optimizer | Adam | |
| Loss | MSE | |
| Train Years | 1971-2010 | |
| Test Years | 1961-1970, 2011-2020 | Before & after training |

### Ablation Matrix

| Config | Sequential | MHA | Hierarchical | Fitter |
|--------|:----------:|:---:|:------------:|:------:|
| baseline | ✗ | ✗ | ✗ | ✗ |
| seq | ✓ | ✗ | ✗ | ✗ |
| seq_mha | ✓ | ✓ | ✗ | ✗ |
| seq_mha_no_hier | ✓ | ✓ | ✗ | ✓ |
| seq_mha_no_fitter | ✓ | ✓ | ✓ | ✗ |
| full | ✓ | ✓ | ✓ | ✓ |
| wide_lstm | ✗ | ✗ | ✗ | ✗ |

*`wide_lstm` has increased hidden_dim for capacity matching*

## 📊 Metrics

The framework computes comprehensive metrics:

- **NSE** (Nash-Sutcliffe Efficiency)
- **KGE** (Kling-Gupta Efficiency)
- **R²** (Coefficient of Determination)
- **PBIAS** (Percent Bias)
- **RMSE** / **MAE**
- **Extreme Flow NSE** (P90, P95 percentiles)
- **Peak Capture %**

## 📈 Output Structure

Each run creates a directory in `runs/`:

```
runs/full_20260206_210000/
├── model.pt              # Trained model weights
├── results.json          # All metrics and predictions
├── train_history.json    # Per-epoch training metrics
├── rating_curve.pkl      # Fitted rating curve (if applicable)
└── hierarchical_model.pkl # CatBoost model (if trained)
```

## 🔬 Model Architecture

```
Input [B, 7, D] → LSTM Backbone [B, 7, H]
                       ↓
              (Optional) MHA [B, H]
                       ↓
              (Optional) + Hierarchical Features
                       ↓
         Sequential/Direct Predictor
                       ↓
              T+1, T+2, T+3 Predictions
                       ↓
              (Optional) Rating Curve
```

## 📝 Citation

If you use this framework, please cite:

```bibtex
@software{seqlstm_mha_2026,
  title={SeqLSTM-MHA: Modular Ablation Framework for Hydrological Forecasting},
  year={2026},
  author={SRIP Team}
}
```

## 📄 License

MIT License
