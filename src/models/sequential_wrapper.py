"""
Sequential Prediction Wrapper
=============================
Controls direct vs sequential multi-horizon forecasting logic.
Core ablation axis for comparing sequential vs parallel prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable


class HorizonHead(nn.Module):
    """
    Single prediction head for one forecast horizon.
    
    Maps from feature representation to scalar prediction.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Internal hidden dimension (default: input_dim // 2)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or input_dim // 2
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict scalar output from features.
        
        Args:
            x: Features of shape [B, input_dim]
            
        Returns:
            prediction: Shape [B] (squeezed scalar predictions)
        """
        return torch.abs(self.head(x).squeeze(-1))  # Ensure non-negative


class SequentialPredictor(nn.Module):
    """
    Sequential prediction wrapper for multi-horizon forecasting.
    
    Controls whether predictions are made sequentially (T+1 feeds into T+2)
    or directly (all horizons predicted independently).
    
    Args:
        hidden_dim: Feature dimension from backbone/attention
        num_horizons: Number of forecast horizons (default: 3 for T+1, T+2, T+3)
        sequential: If True, feed predictions forward; if False, predict all at once
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_horizons: int = 3,
        sequential: bool = True,
        dropout: float = 0.1,
        use_previous_pred_as_feature: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons
        self.sequential = sequential
        self.use_previous_pred_as_feature = use_previous_pred_as_feature
        
        if sequential and use_previous_pred_as_feature:
            # Each subsequent horizon gets previous predictions as features
            # T+1: hidden_dim features
            # T+2: hidden_dim + 1 (T+1 pred) features
            # T+3: hidden_dim + 2 (T+1, T+2 preds) features
            self.heads = nn.ModuleDict({
                't1': HorizonHead(hidden_dim, dropout=dropout),
                't2': HorizonHead(hidden_dim + 1, dropout=dropout),
                't3': HorizonHead(hidden_dim + 2, dropout=dropout)
            })
            
            # Projection layers to incorporate previous predictions
            self.t2_projection = nn.Linear(hidden_dim + 1, hidden_dim + 1)
            self.t3_projection = nn.Linear(hidden_dim + 2, hidden_dim + 2)
        else:
            # All heads have same input dimension
            self.heads = nn.ModuleDict({
                f't{i+1}': HorizonHead(hidden_dim, dropout=dropout)
                for i in range(num_horizons)
            })
            
    def forward(
        self,
        features: torch.Tensor,
        external_features: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions for all horizons.
        
        Args:
            features: Base features from backbone/attention of shape [B, hidden_dim]
            external_features: Optional dict with additional features per horizon
                              (e.g., rating curve predictions, hierarchical features)
                              
        Returns:
            predictions: Dict with keys 't1', 't2', 't3' and tensor values of shape [B]
        """
        predictions = {}
        external_features = external_features or {}
        
        if self.sequential and self.use_previous_pred_as_feature:
            # Sequential: feed forward predictions
            
            # T+1 prediction
            t1_features = features
            if 't1' in external_features:
                t1_features = torch.cat([t1_features, external_features['t1']], dim=-1)
            pred_t1 = self.heads['t1'](t1_features)
            predictions['t1'] = pred_t1
            
            # T+2 prediction: uses T+1 prediction
            t2_features = torch.cat([features, pred_t1.unsqueeze(-1)], dim=-1)
            if 't2' in external_features:
                t2_features = torch.cat([t2_features, external_features['t2']], dim=-1)
            t2_features_proj = self.t2_projection(t2_features[:, :self.hidden_dim + 1])
            if 't2' in external_features:
                t2_features_proj = torch.cat([t2_features_proj, external_features['t2']], dim=-1)
            pred_t2 = self.heads['t2'](t2_features[:, :self.hidden_dim + 1])
            predictions['t2'] = pred_t2
            
            # T+3 prediction: uses T+1 and T+2 predictions
            t3_features = torch.cat([
                features, 
                pred_t1.unsqueeze(-1), 
                pred_t2.unsqueeze(-1)
            ], dim=-1)
            if 't3' in external_features:
                t3_features = torch.cat([t3_features, external_features['t3']], dim=-1)
            pred_t3 = self.heads['t3'](t3_features[:, :self.hidden_dim + 2])
            predictions['t3'] = pred_t3
            
        else:
            # Direct: all horizons predicted independently
            for horizon in ['t1', 't2', 't3']:
                h_features = features
                if horizon in external_features:
                    h_features = torch.cat([h_features, external_features[horizon]], dim=-1)
                predictions[horizon] = self.heads[horizon](h_features)
                
        return predictions
    
    def forward_single_horizon(
        self,
        features: torch.Tensor,
        horizon: str,
        previous_preds: Dict[str, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Make prediction for a single horizon (useful for inference/evaluation).
        
        Args:
            features: Base features of shape [B, hidden_dim]
            horizon: Which horizon ('t1', 't2', or 't3')
            previous_preds: Dict of previous predictions (needed for sequential mode)
            
        Returns:
            prediction: Shape [B]
        """
        previous_preds = previous_preds or {}
        
        if self.sequential and self.use_previous_pred_as_feature:
            if horizon == 't1':
                return self.heads['t1'](features)
            elif horizon == 't2':
                if 't1' not in previous_preds:
                    raise ValueError("T+1 prediction required for T+2 in sequential mode")
                t2_features = torch.cat([features, previous_preds['t1'].unsqueeze(-1)], dim=-1)
                return self.heads['t2'](t2_features)
            elif horizon == 't3':
                if 't1' not in previous_preds or 't2' not in previous_preds:
                    raise ValueError("T+1 and T+2 predictions required for T+3 in sequential mode")
                t3_features = torch.cat([
                    features,
                    previous_preds['t1'].unsqueeze(-1),
                    previous_preds['t2'].unsqueeze(-1)
                ], dim=-1)
                return self.heads['t3'](t3_features)
        else:
            return self.heads[horizon](features)
            
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DirectPredictor(nn.Module):
    """
    Simplified direct (non-sequential) predictor.
    
    All horizons are predicted from the same features independently.
    Useful as baseline for ablation studies.
    
    Args:
        hidden_dim: Input feature dimension
        num_horizons: Number of horizons to predict
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_horizons: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Single shared backbone with multiple output heads
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_horizons)
        ])
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict all horizons directly.
        
        Args:
            features: Shape [B, hidden_dim]
            
        Returns:
            predictions: Dict with 't1', 't2', 't3' keys
        """
        shared = self.shared_layer(features)
        
        predictions = {}
        for i, head in enumerate(self.heads):
            predictions[f't{i+1}'] = torch.abs(head(shared).squeeze(-1))
            
        return predictions
