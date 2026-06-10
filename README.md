# SeqLSTM-MHA: Modular Ablation Framework for Streamflow and Water Level Prediction

A config-driven deep learning framework for multi-horizon hydrological forecasting with
ablation study capabilities.

## Overview

This framework implements a modular LSTM-based architecture for predicting:

- **Streamflow** (T+1, T+2, T+3 days ahead)
- **Water Level** (T+1, T+2, T+3 days ahead)

The key innovation is a **config-driven ablation system** that allows systematic comparison
of architectural components. Every model variant is defined by a small YAML file that toggles
components on/off, while a single training script and a shared set of hard constraints keep
every experiment directly comparable:

| Component | Description |
|-----------|-------------|
| **Sequential Prediction** | Feed the T+1 prediction into T+2, and T+2 into T+3 (3 stacked LSTM blocks). |
| **Multi-Head Attention** | Attend over the LSTM hidden states. |
| **Hierarchical Features** | Pre-computed CatBoost quantile features (q0.1, q0.5, q0.9, q0.95). |
| **Rating Curve Fitter** | Physics-based water-level → streamflow estimation. |
| **Bidirectional LSTM** | Encode the lookback window in both directions. |

### Model architecture

```
Input [B, 7, D] → LSTM Backbone [B, 7, H]
                       ↓
              (Optional) MHA [B, H]
                       ↓
              (Optional) + Hierarchical Features
                       ↓
         Sequential / Direct Predictor
                       ↓
              T+1, T+2, T+3 Predictions
                       ↓
              (Optional) Rating Curve
```

---

# How to Run

This guide is a complete, reproducible, step-by-step walkthrough for installing the
environment, training the models, running the full ablation suite, and generating
all evaluation outputs (metrics tables, plots, feature importance, and complexity
analysis).

It is written so that someone who has **never seen this code before** can reproduce
every result from a clean checkout.

---

## 1. What this code does

The framework trains LSTM-based deep learning models to forecast, for a given river
gauging station, **3 days ahead** (T+1, T+2, T+3) of either:

- **Streamflow** (`streamflow_final`) — the default target, or
- **Water level** (`waterlevel_final`).

It is built as a **config-driven ablation framework**: each model variant is defined
by a small YAML file that toggles architectural components on/off. The same training
script (`src/train.py`) is reused for every variant, so all results are produced under
identical, hard-coded training constraints (see [Section 6](#6-hard-constraints-fixed-for-every-experiment)).

### Model components that can be toggled

| Component | Config flag | Description |
|-----------|-------------|-------------|
| Sequential prediction | `sequential` | 3 stacked LSTM blocks; block *k* sees the lookback window **plus** predictions 1…*k*−1. If off, a single LSTM with 3 independent heads is used. |
| Bidirectional LSTM | `bidirectional` | Encodes the lookback window in both directions. |
| Multi-Head Attention | `use_attention` | MHA attends over the LSTM hidden states. |
| Hierarchical features | `use_hierarchical` | Adds 4 pre-computed CatBoost quantile features (q0.1, q0.5, q0.9, q0.95) already stored in the station CSVs. |
| Rating-curve fitter | `use_fitter` | Physics-based water-level → streamflow conversion; only active for streamflow targets. |

---

## 2. Repository layout

```
srip-transformer/
├── src/
│   ├── train.py                 # Train ONE model (one config × one station × one target)
│   ├── evaluate.py              # Compare runs, build ablation tables + paper plots
│   ├── complexity.py            # FLOPs & parameter counts for each config
│   ├── feature_importance.py    # Permutation feature importance for a trained run
│   ├── plots.py                 # Multi-station comparison plots from a summary CSV
│   ├── utils.py                 # Data loading, scaling, metrics, HARD_CONSTRAINTS
│   ├── configs/                 # 8 ablation YAML files (see Section 5)
│   └── models/                  # Model components
│       ├── lstm_backbone.py     #   LSTM encoder
│       ├── attention.py         #   Multi-Head Attention block
│       ├── sequential_wrapper.py#   Sequential vs Direct predictor
│       ├── hierarchical_features.py # CatBoost hierarchical feature handling
│       ├── fitter.py            #   Rating-curve fitter
│       └── full_model.py        #   Model composer (SeqLSTMModel + ModelConfig)
├── run_all.py                   # Batch runner: ALL configs × ALL stations (the "final" entry point)
├── hierarchial/                 # 7 main-station CSVs WITH hierarchical quantile features
├── stations_csvs/               # 9 additional/upstream station CSVs (raw, no hierarchical features)
├── requirements.txt             # Python dependencies
├── setup_venv.sh                # Linux helper to create a venv and install deps
├── run_all.ps1                  # Windows PowerShell equivalent of run_all.py
├── runs/                        # Output of every training run (created automatically, git-ignored)
└── run_logs/                    # Per-experiment text logs from run_all.py
```

> **Note:** The "final" reproducible entry point is `run_all.py` at the **repository root**
> (not inside `src/`). It calls `src/train.py` once per (config, station) pair.

---

## 3. Requirements

- **Python** 3.10+ (developed/tested on 3.12)
- The packages in `requirements.txt`:
  - `torch>=2.0.0`, `numpy>=1.24.0`, `pandas>=2.0.0`, `scikit-learn>=1.3.0`
  - `pyyaml>=6.0`, `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `tqdm>=4.65.0`, `thop>=0.1.1`
- A GPU is **optional**. The code auto-detects CUDA and falls back to CPU. Each model is
  small (~tens of thousands of parameters, 30 epochs), so CPU training is feasible.
- `catboost` is **not required** to reproduce results — the hierarchical quantile features
  are already pre-computed and stored as columns inside the CSVs in `hierarchial/`.

---

## 4. Installation

### Option A — Linux helper script

```bash
cd srip-transformer
bash setup_venv.sh          # creates ./venv and installs requirements.txt
source venv/bin/activate
```

### Option B — Manual (Linux / macOS)

```bash
cd srip-transformer
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option C — Manual (Windows PowerShell)

```powershell
cd srip-transformer
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import torch, pandas, sklearn, yaml, seaborn, thop; print('OK', torch.__version__)"
```

---

## 5. The ablation configurations

The configs live in `src/configs/`. Eight YAML files are provided:

| Config file | Sequential | Bidirectional | Attention (MHA) | Hierarchical | Fitter |
|-------------|:----------:|:-------------:|:---------------:|:------------:|:------:|
| `baseline.yaml`      | ✗ | ✗ | ✗ | ✗ | ✗ |
| `bidirectional.yaml` | ✗ | ✓ | ✗ | ✗ | ✗ |
| `baseline_mha.yaml`  | ✗ | ✗ | ✓ | ✗ | ✗ |
| `seq.yaml`           | ✓ | ✗ | ✗ | ✗ | ✗ |
| `seq_hier.yaml`      | ✓ | ✗ | ✗ | ✓ | ✗ |
| `seq_mha.yaml`       | ✓ | ✗ | ✓ | ✗ | ✗ |
| `seq_fitter.yaml`    | ✓ | ✗ | ✗ | ✓ | ✓ |
| `full.yaml`          | ✓ | ✗ | ✓ | ✓ | ✓ |

The batch runner `run_all.py` uses these **7** configs in ablation order:
`baseline → seq → seq_hier → seq_fitter → full → baseline_mha → bidirectional`.
(`seq_mha.yaml` is available for manual runs but is not part of the default batch list.)

All LSTM settings are identical across configs (`hidden_dim: 64`, `num_layers: 1`,
`dropout: 0.1`, `num_heads: 8` where attention is used). Training hyper-parameters in the
YAML are **overridden** by the global hard constraints (Section 6), so they cannot drift
between experiments.

---

## 6. Hard constraints (fixed for every experiment)

Defined in `src/utils.py` as `HARD_CONSTRAINTS` and enforced for **every** run regardless
of the YAML:

| Parameter | Value |
|-----------|-------|
| Lookback window | 7 days (t−7 … t−1) |
| Batch size | 256 |
| Learning rate | 1e-3 |
| Epochs | 30 |
| Optimizer | Adam |
| Loss | MSE (mean over T+1, T+2, T+3) |
| Random seed | 74 |
| Train years | 1971–2010 |
| Test set #1 | 1961–1970 (before training period) |
| Test set #2 | 2011–2020 (after training period) |

Features are scaled with `MinMaxScaler` fit on the training split only. The best checkpoint
is selected by the **highest T+1 NSE on the training set** during the 30 epochs.

---

## 7. Input data

Two data folders are included:

- **`hierarchial/`** — 7 main gauging stations, each as
  `new_<Station>_with_hierarchical_quantiles.csv`. These are the CSVs used for training,
  because they already contain the pre-computed hierarchical quantile feature columns:
  - `Barmanghat, Garudeshwar, Handia, Hoshangabad, Mandleshwar, Manot, Sandia`
- **`stations_csvs/`** — 9 additional/upstream stations (raw, no hierarchical columns):
  `BamniBanjar, Belkhedi, Chhidgaon, Dindori, Gadarwara, Kogaon, Mohgaon, Patan, Pati`.

Each training CSV contains a `date` column (day-first format) plus the model features:
`rainfall, tmax, tmin, waterlevel_upstream, streamflow_upstream`, the targets
`waterlevel_final, streamflow_final`, and the four hierarchical columns
`hierarchical_feature_quantile_{0.1,0.5,0.9,0.95}`.

> Feature selection is automatic: for streamflow targets the model uses the base features
> plus `waterlevel_final` and `streamflow_final`; for water level it drops `streamflow_final`.
> Hierarchical columns are appended **only** when the config sets `use_hierarchical: true`.

---

## 8. Running the code

> Run all commands from the **repository root** (`srip-transformer/`) with the virtual
> environment activated.

### 8.1 Quick smoke test (recommended first step)

Print every command that the full suite would run, without executing anything:

```bash
python run_all.py --dry-run
```

Then train a single fast model to confirm the environment works end-to-end:

```bash
python src/train.py \
    --config src/configs/baseline.yaml \
    --station hierarchial/new_Handia_with_hierarchical_quantiles.csv \
    --target streamflow_final
```

This prints an 8-step pipeline (config → data → output dir → fitter → model → training →
evaluation → save) and writes a folder under `runs/`.

### 8.2 Train a single model (general form)

```bash
python src/train.py \
    --config   src/configs/<CONFIG>.yaml \
    --station  <PATH_TO_STATION_CSV> \
    --target   <streamflow_final | waterlevel_final> \
    --output_dir runs \
    --station_name <optional friendly name>
```

`src/train.py` arguments:

| Argument | Required | Default | Meaning |
|----------|:--------:|---------|---------|
| `--config` | ✓ | — | Path to a YAML config in `src/configs/`. |
| `--station` | ✓ | — | Path to a station CSV. |
| `--target` | | `streamflow_final` | Target column to predict. |
| `--output_dir` | | `runs` | Base directory for run outputs. |
| `--features` | | auto | Comma-separated override of feature columns. |
| `--station_name` | | CSV stem | Label used in logs / `results.json`. |

Examples:

```bash
# Full model, streamflow
python src/train.py --config src/configs/full.yaml \
    --station hierarchial/new_Handia_with_hierarchical_quantiles.csv \
    --target streamflow_final

# Baseline, water level
python src/train.py --config src/configs/baseline.yaml \
    --station hierarchial/new_Handia_with_hierarchical_quantiles.csv \
    --target waterlevel_final
```

### 8.3 Run the FULL ablation suite (the "final" run)

`run_all.py` trains **7 configs × 7 stations = 49 experiments** sequentially, writing one
log per experiment to `run_logs/` and all model outputs to `runs/`:

```bash
python run_all.py
```

Useful flags:

```bash
python run_all.py --dry-run                 # print the 49 commands, run nothing
python run_all.py --config full             # only the 'full' config, across all 7 stations
python run_all.py --start-from 15           # resume the suite from experiment #15
```

> By default `run_all.py` trains the **streamflow** target (the `train.py` default). To produce
> water-level results as well, run the suite/targets explicitly via `src/train.py`, or duplicate
> the loop with `--target waterlevel_final`.

On Windows you can instead use the PowerShell version:

```powershell
.\run_all.ps1
```

---

## 9. Outputs of a run

Every training run creates a timestamped directory `runs/<config>_<YYYYMMDD_HHMMSS>/`
containing:

| File | Description |
|------|-------------|
| `model.pt` | Best model weights (selected by training T+1 NSE). |
| `results.json` | Config, parameter counts, and full test metrics for both test periods. |
| `train_history.json` | Per-epoch loss and per-horizon metrics. |
| `feature_saliency.json` | Gradient × input saliency: `[N samples × 7 timesteps × n_features]` per horizon. |
| `attention_weights.json` | Per-day attention weights `[N × 7 timesteps × n_heads]` (only for attention configs; a stub otherwise). |
| `rating_curve.pkl` | Fitted rating curve (only when `use_fitter: true` and target is streamflow). |

### Metrics computed

For each horizon (T+1, T+2, T+3) on each test period (1961–1970 and 2011–2020):

- **NSE** (Nash–Sutcliffe Efficiency), **KGE** (Kling–Gupta Efficiency), **R²**
- **PBIAS** (percent bias), **RMSE**, **MAE**
- **Extreme-flow NSE** at the P90 and P95 percentiles
- **Peak capture %**

---

## 10. Evaluating and comparing results

### 10.1 Inspect a single run

```bash
python src/evaluate.py --run_dir runs/full_20260206_183000
```

Prints a formatted metrics table for both test periods.

### 10.2 Build the ablation comparison table + paper plots

```bash
python src/evaluate.py \
    --compare "runs/baseline_*" "runs/seq_*" "runs/full_*" \
    --output comparison.csv \
    --plots \
    --output_dir ablation_plots
```

- `--compare` accepts one or more glob patterns over run directories.
- `--output` writes the combined metrics table as CSV.
- `--plots` saves, into `--output_dir` (default `ablation_plots/`):
  - `nse_comparison.png`, `extreme_nse_comparison.png`, `params_comparison.png`,
    `ablation_heatmap.png`, `peak_capture_comparison.png`.

### 10.3 Permutation feature importance for a trained run

```bash
python src/feature_importance.py \
    --run_dir runs/full_20260206_183000 \
    --station hierarchial/new_Handia_with_hierarchical_quantiles.csv \
    --split test_last \
    --n_repeats 10 \
    --horizon t1 \
    --plot
```

Shuffles each feature across the test set and measures the resulting NSE drop. Results are
saved to `<run_dir>/feature_importance.csv` (override with `--output`).

### 10.4 Model complexity (FLOPs & parameters)

```bash
# All configs at once → complexity_analysis.csv
python src/complexity.py --all-configs

# A single config
python src/complexity.py --config src/configs/full.yaml --input_dim 8
```

### 10.5 Multi-station summary plots (optional)

`src/plots.py` builds grouped comparison plots from a pre-built summary CSV:

```bash
python src/plots.py <summary_csv_path> <output_base_dir> "<feature set description>"
```

---

## 11. End-to-end reproduction recipe

```bash
# 1. Environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Sanity check
python run_all.py --dry-run

# 3. Train the full ablation suite (49 runs → runs/, logs → run_logs/)
python run_all.py

# 4. Build ablation table + figures across all runs
python src/evaluate.py --compare "runs/*" --output comparison.csv --plots

# 5. Complexity analysis for every config
python src/complexity.py --all-configs

# 6. (Optional) feature importance on the best 'full' run
python src/feature_importance.py \
    --run_dir runs/full_<timestamp> \
    --station hierarchial/new_Handia_with_hierarchical_quantiles.csv --plot
```

---

## 12. Reproducibility notes & troubleshooting

- **Determinism:** the seed is fixed at 74 and cuDNN is set to deterministic mode in
  `set_seed()`. Results are reproducible on the same hardware/library versions; minor
  numerical differences can occur across different GPUs or PyTorch builds.
- **CPU vs GPU:** no flags needed — CUDA is auto-detected, otherwise CPU is used.
- **`runs/` is git-ignored** (see `.gitignore`), so a fresh checkout starts with no outputs;
  everything is regenerated by the steps above.
- **Encoding on Windows:** the scripts force UTF-8 output, so the ✓/✗/→ symbols and progress
  bars render correctly in `cmd`/PowerShell.
- **Hierarchical features missing:** if you point `--station` at a CSV from `stations_csvs/`
  (which has no hierarchical columns) while using a `use_hierarchical: true` config, the model
  prints a warning and proceeds without them. Use the `hierarchial/` CSVs to include them.
- **`thop` FLOPs show 0:** `thop` cannot trace `nn.LSTM` kernels; the scripts fall back to a
  manual FLOPs estimate, which is the value to report.
