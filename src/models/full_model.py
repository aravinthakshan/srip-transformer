"""
Full Model Composer
===================
The ONLY place where all modules are combined into a complete model.
Configuration-driven composition enables ablation experiments.

Model Variants:
    1. Vanilla LSTM        - DirectPredictor with bidirectional=False
    2. Bidirectional LSTM  - DirectPredictor with bidirectional=True
    3. Sequential LSTM     - SequentialLSTM (3 LSTM blocks, auto-regressive)
    4. Seq LSTM + Fitter   - SequentialLSTM with rating curve for streamflow

    Any of the above can have MHA (Multi-Head Attention) layered on top.
    MHA attends over LSTM hidden states instead of using only the last one.
"""

import torch
import torch.nn as nn
from typing import Dict
from dataclasses import dataclass

from .sequential_wrapper import SequentialLSTM, DirectPredictor
from .fitter import RatingCurveFitter


@dataclass
class ModelConfig:
    """
    Configuration for SeqLSTMModel.

    All ablation experiments are defined by this configuration.
    No code edits should be needed per experiment.

    Hard constraints (these should NOT vary between experiments):
        - lookback: 7
        - epochs: 30
        - batch_size: 256
        - lr: 1e-3
        - optimizer: Adam
        - loss: MSE
    """

    # Core architecture switches (ABLATION AXES)
    sequential: bool = True  # Sequential (3 LSTM blocks) vs Direct
    use_attention: bool = False  # MHA on top of LSTM hidden states
    use_fitter: bool = True  # Rating curve for streamflow (only used with sequential)
    use_hierarchical: bool = True  # Hierarchical features on/off
    bidirectional: bool = False  # For Vanilla/Bidirectional LSTM (non-sequential mode)

    # LSTM parameters
    input_dim: int = 8  # Number of input features
    hidden_dim: int = 64  # LSTM hidden dimension
    num_layers: int = 1  # Number of LSTM layers per block
    dropout: float = 0.1  # Dropout probability

    # Attention parameters
    num_heads: int = 8  # Number of attention heads (when use_attention=True)

    # Training constraints (FIXED)
    lookback: int = 7  # Days of history (t-7 to t-1)
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 30
    random_seed: int = 74

    # Target variable type
    is_streamflow: bool = True  # True for streamflow, False for water level

    # External model paths
    rating_curve_path: str = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create config from dictionary (e.g., from YAML)."""
        return cls(
            **{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class SeqLSTMModel(nn.Module):
    """
    Modular Sequence-to-Prediction Model for Streamflow/Water Level Forecasting.

    This is the UNIFIED model composer. All architectural variations are
    controlled via the config, enabling clean ablation experiments.

    Architecture Variants:
        1. Vanilla LSTM (sequential=False, bidirectional=False):
           Single LSTM → 3 independent linear heads

        2. Bidirectional LSTM (sequential=False, bidirectional=True):
           Bidirectional LSTM → 3 independent linear heads

        3. Sequential LSTM (sequential=True, use_fitter=False):
           3 separate LSTM blocks. Block k processes the original
           sequence + predictions from blocks 1..k-1 appended as
           new timesteps.

        4. Sequential LSTM + Fitter (sequential=True, use_fitter=True, is_streamflow=True):
           Same as (3), but each block first predicts water level,
           converts via rating curve to expected streamflow, and
           uses that signal alongside LSTM features to predict streamflow.

    Args:
        config: ModelConfig object defining all architectural choices
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        if config.sequential:
            # --- Sequential LSTM (3 blocks) ---
            # Determine if fitter should actually be used:
            # only for streamflow targets with fitter enabled
            effective_use_fitter = config.use_fitter and config.is_streamflow

            # Load rating curve if needed
            fitter = None
            if effective_use_fitter and config.rating_curve_path:
                fitter = RatingCurveFitter()
                fitter.load(config.rating_curve_path)

            self.predictor = SequentialLSTM(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                use_attention=config.use_attention,
                num_heads=config.num_heads,
                use_fitter=effective_use_fitter,
                fitter=fitter,
            )
        else:
            # --- Vanilla / Bidirectional LSTM ---
            self.predictor = DirectPredictor(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                bidirectional=config.bidirectional,
                use_attention=config.use_attention,
                num_heads=config.num_heads,
            )

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape [B, T, input_dim]
               T = lookback (t-7 to t-1), NOT including day t
            **kwargs: Passed through to the predictor (e.g. return_attn_weights=True)

        Returns:
            predictions: Dict with keys 't1', 't2', 't3' containing predictions [B]
                        (may also contain 'wl_t1', 'wl_t2', 'wl_t3' if fitter is used)
        """
        return self.predictor(x, **kwargs)

    def set_fitter(self, fitter: RatingCurveFitter):
        """
        Set or update the rating curve fitter on the sequential predictor.

        This is called during training after the fitter has been fitted
        on the training data.

        Args:
            fitter: Fitted RatingCurveFitter instance
        """
        if isinstance(self.predictor, SequentialLSTM):
            self.predictor.fitter = fitter
        else:
            print("Warning: set_fitter called on non-sequential model (no-op)")

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters by component.

        Returns dictionary with per-component and total counts.
        """
        counts = {}
        counts["total"] = self.predictor.count_parameters()

        if isinstance(self.predictor, SequentialLSTM):
            # Break down by LSTM block
            counts["lstm_block1"] = sum(
                p.numel()
                for p in self.predictor.lstm_block1.parameters()
                if p.requires_grad
            )
            counts["lstm_block2"] = sum(
                p.numel()
                for p in self.predictor.lstm_block2.parameters()
                if p.requires_grad
            )
            counts["lstm_block3"] = sum(
                p.numel()
                for p in self.predictor.lstm_block3.parameters()
                if p.requires_grad
            )
            counts["projection"] = sum(
                p.numel()
                for p in self.predictor.pred_to_timestep.parameters()
                if p.requires_grad
            )
            if self.config.use_attention:
                counts["attention"] = sum(
                    p.numel()
                    for name, p in self.predictor.named_parameters()
                    if p.requires_grad and "mha_block" in name
                )
            else:
                counts["attention"] = 0
            counts["heads"] = counts["total"] - (
                counts["lstm_block1"]
                + counts["lstm_block2"]
                + counts["lstm_block3"]
                + counts["projection"]
                + counts["attention"]
            )
        elif isinstance(self.predictor, DirectPredictor):
            counts["lstm"] = sum(
                p.numel() for p in self.predictor.lstm.parameters() if p.requires_grad
            )
            if self.config.use_attention:
                counts["attention"] = sum(
                    p.numel()
                    for p in self.predictor.mha.parameters()
                    if p.requires_grad
                )
            else:
                counts["attention"] = 0
            counts["heads"] = sum(
                p.numel() for p in self.predictor.heads.parameters() if p.requires_grad
            )

        return counts

    def get_config_summary(self) -> str:
        """Get human-readable config summary."""
        if self.config.sequential:
            parts = ["Sequential"]
            if self.config.use_fitter and self.config.is_streamflow:
                parts.append("Fitter")
        else:
            if self.config.bidirectional:
                parts = ["Bidirectional"]
            else:
                parts = ["Vanilla"]

        if self.config.use_attention:
            parts.append("MHA")

        if self.config.use_hierarchical:
            parts.append("Hier")

        return "_".join(parts)


def create_model_from_config(config_dict: dict) -> SeqLSTMModel:
    """
    Factory function to create model from config dictionary.

    Args:
        config_dict: Configuration dictionary (typically from YAML)

    Returns:
        Instantiated SeqLSTMModel
    """
    config = ModelConfig.from_dict(config_dict)
    return SeqLSTMModel(config)
