"""
Full Model Composer
===================
The ONLY place where all modules are combined into a complete model.
Configuration-driven composition enables ablation experiments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from .lstm_backbone import LSTMBackbone
from .attention import MHABlock, PooledMHABlock
from .sequential_wrapper import SequentialPredictor, DirectPredictor
from .hierarchical_features import HierarchicalFeatures
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
    use_lstm: bool = True          # Always True for our experiments
    use_attention: bool = True      # MHA on/off
    use_hierarchical: bool = True   # Hierarchical features on/off
    use_fitter: bool = True         # Rating curve on/off
    sequential: bool = True         # Sequential vs direct prediction
    
    # LSTM backbone parameters
    input_dim: int = 8              # Number of input features
    hidden_dim: int = 64            # LSTM hidden dimension
    num_layers: int = 1             # Number of LSTM layers
    bidirectional: bool = False     # Bidirectional LSTM
    dropout: float = 0.1            # Dropout probability
    
    # Attention parameters
    num_heads: int = 8              # Number of attention heads
    attention_pooling: str = 'last' # Pooling strategy
    
    # Training constraints (FIXED)
    lookback: int = 7               # Days of history (t-7 to t-1)
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 30
    random_seed: int = 74
    
    # Target variable type
    is_streamflow: bool = True      # True for streamflow, False for water level
    
    # External model paths
    hierarchical_model_path: str = None
    rating_curve_path: str = None
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create config from dictionary (e.g., from YAML)."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class SeqLSTMModel(nn.Module):
    """
    Modular Sequence-to-Prediction Model for Streamflow/Water Level Forecasting.
    
    This is the UNIFIED model composer. All architectural variations are
    controlled via the config, enabling clean ablation experiments.
    
    Architecture:
        1. LSTM Backbone: Encodes input sequence [B, T, D] -> [B, T, H]
        2. Attention (optional): Attends over sequence [B, T, H] -> [B, H]
        3. Hierarchical Features (optional): Injects CatBoost μ/σ features
        4. Sequential/Direct Predictor: Produces T+1, T+2, T+3 predictions
        5. Rating Curve Fitter (optional): Converts water level to streamflow
    
    Args:
        config: ModelConfig object defining all architectural choices
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Compute effective hidden dimension (accounts for bidirectional)
        self.effective_hidden_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # ===== 1. LSTM Backbone =====
        if config.use_lstm:
            self.backbone = LSTMBackbone(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                bidirectional=config.bidirectional
            )
        else:
            # Fallback: simple linear encoding (for testing)
            self.backbone = nn.Linear(config.input_dim, self.effective_hidden_dim)
            
        # ===== 2. Multi-Head Attention (Optional) =====
        if config.use_attention:
            self.mha = MHABlock(
                embed_dim=self.effective_hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
        else:
            self.mha = None
            
        # ===== 3. Hierarchical Features (Optional) =====
        if config.use_hierarchical:
            self.hier = HierarchicalFeatures(
                model_path=config.hierarchical_model_path
            )
            # Hierarchical features add: μ, σ (2 features per horizon)
            self.hier_feature_dim = 2
        else:
            self.hier = None
            self.hier_feature_dim = 0
            
        # ===== 4. Rating Curve Fitter (Optional) =====
        if config.use_fitter and config.is_streamflow:
            self.fitter = RatingCurveFitter()
            if config.rating_curve_path:
                self.fitter.load(config.rating_curve_path)
        else:
            self.fitter = None
            
        # ===== 5. Sequential/Direct Predictor =====
        predictor_input_dim = self.effective_hidden_dim
        
        if config.sequential:
            self.predictor = SequentialPredictor(
                hidden_dim=predictor_input_dim,
                num_horizons=3,
                sequential=True,
                dropout=config.dropout
            )
        else:
            self.predictor = DirectPredictor(
                hidden_dim=predictor_input_dim,
                num_horizons=3,
                dropout=config.dropout
            )
            
        # Projection for when not using attention (need to reduce sequence to single vector)
        if not config.use_attention:
            self.sequence_pooler = nn.Sequential(
                nn.Linear(self.effective_hidden_dim, self.effective_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
            
    def forward(
        self,
        x: torch.Tensor,
        hierarchical_inputs: Dict[str, torch.Tensor] = None,
        water_level_targets: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [B, T, input_dim]
               T = lookback (t-n to t-1), NOT including day t
            hierarchical_inputs: Optional dict with keys 'feeder', 'rainfall', 'covariates'
                                for hierarchical feature extraction
            water_level_targets: Optional dict with 't1', 't2', 't3' water level predictions
                                for rating curve application
                                
        Returns:
            predictions: Dict with keys 't1', 't2', 't3' containing predictions [B]
        """
        batch_size = x.size(0)
        
        # ===== Step 1: Encode sequence =====
        if isinstance(self.backbone, LSTMBackbone):
            hidden_states = self.backbone(x)  # [B, T, H]
        else:
            # Linear fallback
            hidden_states = self.backbone(x)  # [B, T, H]
            
        # ===== Step 2: Apply attention or pool =====
        if self.mha is not None:
            features = self.mha(hidden_states)  # [B, H]
        else:
            # Use last hidden state and pool
            last_hidden = hidden_states[:, -1, :]  # [B, H]
            features = self.sequence_pooler(last_hidden)  # [B, H]
            
        # ===== Step 3: Prepare external features (hierarchical + fitter) =====
        external_features = {}
        
        # Hierarchical features (if enabled and inputs provided)
        if self.hier is not None and hierarchical_inputs is not None:
            try:
                with torch.no_grad():
                    # Extract μ, σ from hierarchical model (CPU operation)
                    feeder = hierarchical_inputs['feeder'].cpu().numpy()
                    rainfall = hierarchical_inputs['rainfall'].cpu().numpy()
                    covariates = hierarchical_inputs.get('covariates')
                    if covariates is not None:
                        covariates = covariates.cpu().numpy()
                        
                    mu, sigma = self.hier.transform(feeder, rainfall, covariates)
                    
                    # Convert back to tensor and add to external features
                    hier_feats = torch.tensor(
                        np.column_stack([mu, sigma]),
                        dtype=x.dtype, device=x.device
                    )
                    
                    # Apply to all horizons
                    for horizon in ['t1', 't2', 't3']:
                        external_features[horizon] = hier_feats
            except Exception as e:
                print(f"Warning: Hierarchical feature extraction failed: {e}")
                
        # Rating curve predictions (if enabled)
        if self.fitter is not None and self.fitter.fitted and water_level_targets is not None:
            for horizon in ['t1', 't2', 't3']:
                if horizon in water_level_targets:
                    try:
                        wl = water_level_targets[horizon].cpu().numpy()
                        q_fit = self.fitter.predict(wl)
                        q_fit_tensor = torch.tensor(
                            q_fit, dtype=x.dtype, device=x.device
                        ).unsqueeze(-1)  # [B, 1]
                        
                        if horizon in external_features:
                            external_features[horizon] = torch.cat(
                                [external_features[horizon], q_fit_tensor], dim=-1
                            )
                        else:
                            external_features[horizon] = q_fit_tensor
                    except Exception as e:
                        print(f"Warning: Rating curve prediction failed for {horizon}: {e}")
                        
        # ===== Step 4: Make predictions =====
        if external_features:
            predictions = self.predictor(features, external_features)
        else:
            predictions = self.predictor(features)
            
        return predictions
    
    def forward_t1_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for T+1 prediction only.
        
        Args:
            x: Input [B, T, D]
            
        Returns:
            t1_pred: [B]
        """
        preds = self.forward(x)
        return preds['t1']
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters by component.
        
        Returns dictionary with per-component and total counts.
        Excludes non-trainable components (hierarchical, fitter).
        """
        counts = {}
        
        if isinstance(self.backbone, LSTMBackbone):
            counts['backbone'] = self.backbone.count_parameters()
        else:
            counts['backbone'] = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            
        if self.mha is not None:
            counts['attention'] = self.mha.count_parameters()
        else:
            counts['attention'] = 0
            
        counts['predictor'] = self.predictor.count_parameters()
        
        if hasattr(self, 'sequence_pooler'):
            counts['pooler'] = sum(p.numel() for p in self.sequence_pooler.parameters() if p.requires_grad)
        else:
            counts['pooler'] = 0
            
        counts['total'] = sum(counts.values())
        
        # Note: hierarchical and fitter have 0 trainable params
        counts['hierarchical'] = 0
        counts['fitter'] = 0
        
        return counts
    
    def get_config_summary(self) -> str:
        """Get human-readable config summary."""
        parts = []
        
        if self.config.sequential:
            parts.append("Seq")
        else:
            parts.append("Direct")
            
        if self.config.use_attention:
            parts.append("MHA")
            
        if self.config.use_hierarchical:
            parts.append("Hier")
            
        if self.config.use_fitter:
            parts.append("Fitter")
            
        return "_".join(parts) if parts else "Baseline"


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


def get_capacity_matched_hidden_dim(target_params: int, config: ModelConfig) -> int:
    """
    Calculate hidden_dim for capacity-matched baseline.
    
    Given a target parameter count, finds the hidden_dim that
    produces approximately that many parameters for a baseline
    (no attention) LSTM model.
    
    Args:
        target_params: Target parameter count
        config: Base configuration
        
    Returns:
        hidden_dim: Adjusted hidden dimension
    """
    # LSTM params ≈ 4 * hidden_dim * (input_dim + hidden_dim + 1)
    # Simplified quadratic: 4h^2 + 4h*input_dim - target = 0
    
    input_dim = config.input_dim
    
    # Solve quadratic: 4h^2 + 4*input_dim*h - target = 0
    a = 4
    b = 4 * input_dim
    c = -target_params
    
    discriminant = b**2 - 4*a*c
    hidden_dim = int((-b + np.sqrt(discriminant)) / (2*a))
    
    return max(32, hidden_dim)  # Minimum 32
