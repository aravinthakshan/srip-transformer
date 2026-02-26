"""
Sequential Prediction Wrapper
=============================
3 separate LSTM blocks for sequential multi-horizon forecasting.

Block 1: [B, 7, D] → LSTM (→ optional MHA) → predict T+1
Block 2: [B, 8, D] → LSTM (→ optional MHA) → predict T+2  (appends T+1 forecast)
Block 3: [B, 9, D] → LSTM (→ optional MHA) → predict T+3  (appends T+1, T+2 forecasts)

For streamflow with fitter: each block also predicts water level, converts
via rating curve to expected streamflow, and uses that to improve the
streamflow prediction.

MHA can be optionally layered on top of any variant. Instead of using
the last hidden state, MHA attends over all LSTM hidden states to
produce a richer representation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

from .attention import MHABlock


class SequentialLSTM(nn.Module):
    """
    Sequential multi-horizon LSTM predictor with 3 separate LSTM blocks.

    Each block processes an increasingly longer sequence as previous
    predictions are appended as new timesteps.

    Optional MHA attends over each block's hidden states instead of
    just using the last hidden state.

    For streamflow prediction with fitter enabled:
        - Each block first predicts water level
        - Rating curve converts water level → expected streamflow
        - Expected streamflow is concatenated with LSTM features
          to produce the final streamflow prediction

    Args:
        input_dim: Number of input features per timestep
        hidden_dim: LSTM hidden dimension (default: 64)
        num_layers: Number of LSTM layers per block (default: 1)
        dropout: Dropout probability (default: 0.1)
        use_attention: Whether to use MHA on top of LSTM (default: False)
        num_heads: Number of attention heads (default: 8)
        use_fitter: Whether to use rating curve for streamflow (default: False)
        fitter: Pre-fitted RatingCurveFitter instance (default: None)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_attention: bool = False,
        num_heads: int = 8,
        use_fitter: bool = False,
        fitter=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_fitter = use_fitter
        self.fitter = fitter  # RatingCurveFitter instance (not a nn.Module)

        # ----- 3 separate LSTM blocks -----
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm_block1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.lstm_block2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.lstm_block3 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # ----- Optional MHA blocks (one per LSTM block) -----
        if use_attention:
            self.mha_block1 = MHABlock(
                embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout
            )
            self.mha_block2 = MHABlock(
                embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout
            )
            self.mha_block3 = MHABlock(
                embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout
            )

        # ----- Projection layers -----
        # Map a scalar prediction (1-dim) → input_dim so it can be
        # appended as a new timestep to the input sequence.
        self.pred_to_timestep = nn.Linear(1, input_dim)

        # ----- Prediction heads -----
        self.dropout_layer = nn.Dropout(dropout)

        if use_fitter:
            # Water level prediction heads (one per horizon)
            self.wl_head1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.wl_head2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.wl_head3 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

            # Streamflow prediction heads
            # Input: hidden_dim (LSTM/MHA features) + 1 (expected streamflow from fitter)
            sf_input_dim = hidden_dim + 1
            self.sf_head1 = nn.Sequential(
                nn.Linear(sf_input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.sf_head2 = nn.Sequential(
                nn.Linear(sf_input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.sf_head3 = nn.Sequential(
                nn.Linear(sf_input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            # Direct target prediction heads (water level OR streamflow without fitter)
            self.head1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.head2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.head3 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

    def _extract_features(
        self, hidden_states: torch.Tensor, mha_block=None
    ) -> torch.Tensor:
        """
        Extract a single feature vector from LSTM hidden states.

        If MHA is enabled, attends over all timesteps.
        Otherwise, uses the last hidden state.

        Args:
            hidden_states: [B, T, H] from LSTM
            mha_block: Optional MHABlock to use

        Returns:
            features: [B, H]
        """
        if self.use_attention and mha_block is not None:
            # MHA attends over all timesteps using last as query → [B, H]
            features = mha_block(hidden_states)
        else:
            # Just use the last hidden state
            features = hidden_states[:, -1, :]

        return self.dropout_layer(features)

    def _apply_fitter(self, water_level_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert predicted water level to expected streamflow via rating curve.

        Args:
            water_level_pred: Water level predictions [B, 1]

        Returns:
            expected_streamflow: [B, 1] tensor on same device
        """
        if self.fitter is None or not self.fitter.fitted:
            return torch.zeros_like(water_level_pred)

        with torch.no_grad():
            wl_np = water_level_pred.detach().cpu().numpy().flatten()
            try:
                sf_np = self.fitter.predict(wl_np)
            except Exception:
                sf_np = np.zeros_like(wl_np)

            expected_sf = torch.tensor(
                sf_np, dtype=water_level_pred.dtype, device=water_level_pred.device
            ).unsqueeze(-1)  # [B, 1]

        return expected_sf

    def _project_pred_to_timestep(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Project a scalar prediction to input_dim-dimensional vector
        so it can be appended as a new timestep.

        Args:
            pred: Prediction tensor [B] or [B, 1]

        Returns:
            timestep: [B, 1, input_dim]
        """
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)  # [B, 1]
        projected = self.pred_to_timestep(pred)  # [B, input_dim]
        return projected.unsqueeze(1)  # [B, 1, input_dim]

    def forward(
        self, x: torch.Tensor, external_features: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the 3 sequential LSTM blocks.

        Args:
            x: Input tensor [B, T, input_dim] where T = lookback (7)
            external_features: Unused, kept for interface compatibility

        Returns:
            predictions: Dict with keys 't1', 't2', 't3' and values [B]

            If use_fitter is True, also includes 'wl_t1', 'wl_t2', 'wl_t3'
            for the intermediate water level predictions (useful for loss).
        """
        predictions = {}

        # Get optional MHA blocks
        mha1 = self.mha_block1 if self.use_attention else None
        mha2 = self.mha_block2 if self.use_attention else None
        mha3 = self.mha_block3 if self.use_attention else None

        if self.use_fitter:
            # ===== STREAMFLOW MODE WITH FITTER =====

            # --- Block 1: predict WL T+1, then SF T+1 ---
            h1, _ = self.lstm_block1(x)  # [B, T, H]
            h1_feat = self._extract_features(h1, mha1)  # [B, H]

            wl_t1 = self.wl_head1(h1_feat)  # [B, 1]
            expected_sf_t1 = self._apply_fitter(wl_t1)  # [B, 1]
            sf_features_t1 = torch.cat([h1_feat, expected_sf_t1], dim=-1)  # [B, H+1]
            sf_t1 = torch.abs(self.sf_head1(sf_features_t1).squeeze(-1))  # [B]
            predictions["t1"] = sf_t1
            predictions["wl_t1"] = torch.abs(wl_t1.squeeze(-1))  # [B]

            # --- Block 2: append T+1 prediction, predict T+2 ---
            t1_timestep = self._project_pred_to_timestep(sf_t1.detach())  # [B, 1, D]
            x2 = torch.cat([x, t1_timestep], dim=1)  # [B, T+1, D]

            h2, _ = self.lstm_block2(x2)  # [B, T+1, H]
            h2_feat = self._extract_features(h2, mha2)  # [B, H]

            wl_t2 = self.wl_head2(h2_feat)
            expected_sf_t2 = self._apply_fitter(wl_t2)
            sf_features_t2 = torch.cat([h2_feat, expected_sf_t2], dim=-1)
            sf_t2 = torch.abs(self.sf_head2(sf_features_t2).squeeze(-1))
            predictions["t2"] = sf_t2
            predictions["wl_t2"] = torch.abs(wl_t2.squeeze(-1))

            # --- Block 3: append T+1 and T+2 predictions, predict T+3 ---
            t2_timestep = self._project_pred_to_timestep(sf_t2.detach())
            x3 = torch.cat([x, t1_timestep, t2_timestep], dim=1)  # [B, T+2, D]

            h3, _ = self.lstm_block3(x3)  # [B, T+2, H]
            h3_feat = self._extract_features(h3, mha3)  # [B, H]

            wl_t3 = self.wl_head3(h3_feat)
            expected_sf_t3 = self._apply_fitter(wl_t3)
            sf_features_t3 = torch.cat([h3_feat, expected_sf_t3], dim=-1)
            sf_t3 = torch.abs(self.sf_head3(sf_features_t3).squeeze(-1))
            predictions["t3"] = sf_t3
            predictions["wl_t3"] = torch.abs(wl_t3.squeeze(-1))

        else:
            # ===== WATER LEVEL MODE (or streamflow without fitter) =====

            # --- Block 1 ---
            h1, _ = self.lstm_block1(x)  # [B, T, H]
            h1_feat = self._extract_features(h1, mha1)  # [B, H]
            pred_t1 = torch.abs(self.head1(h1_feat).squeeze(-1))  # [B]
            predictions["t1"] = pred_t1

            # --- Block 2 ---
            t1_timestep = self._project_pred_to_timestep(pred_t1.detach())
            x2 = torch.cat([x, t1_timestep], dim=1)

            h2, _ = self.lstm_block2(x2)
            h2_feat = self._extract_features(h2, mha2)
            pred_t2 = torch.abs(self.head2(h2_feat).squeeze(-1))
            predictions["t2"] = pred_t2

            # --- Block 3 ---
            t2_timestep = self._project_pred_to_timestep(pred_t2.detach())
            x3 = torch.cat([x, t1_timestep, t2_timestep], dim=1)

            h3, _ = self.lstm_block3(x3)
            h3_feat = self._extract_features(h3, mha3)
            pred_t3 = torch.abs(self.head3(h3_feat).squeeze(-1))
            predictions["t3"] = pred_t3

        return predictions

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DirectPredictor(nn.Module):
    """
    Direct (non-sequential) predictor.

    Uses a single LSTM backbone and predicts all horizons independently
    from the same feature vector. Used for Vanilla LSTM and
    Bidirectional LSTM baselines.

    Optional MHA attends over LSTM hidden states instead of just
    using the last hidden state.

    Args:
        input_dim: Number of input features per timestep
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional LSTM
        use_attention: Whether to use MHA on top of LSTM
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        use_attention: bool = False,
        num_heads: int = 8,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        self.effective_hidden = hidden_dim * self.num_directions

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Optional MHA block
        if use_attention:
            self.mha = MHABlock(
                embed_dim=self.effective_hidden,
                num_heads=num_heads,
                dropout=dropout,
            )

        self.dropout = nn.Dropout(dropout)

        # 3 independent prediction heads (one per horizon)
        self.heads = nn.ModuleDict(
            {
                "t1": nn.Sequential(
                    nn.Linear(self.effective_hidden, self.effective_hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.effective_hidden // 2, 1),
                ),
                "t2": nn.Sequential(
                    nn.Linear(self.effective_hidden, self.effective_hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.effective_hidden // 2, 1),
                ),
                "t3": nn.Sequential(
                    nn.Linear(self.effective_hidden, self.effective_hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.effective_hidden // 2, 1),
                ),
            }
        )

    def forward(
        self, x: torch.Tensor, external_features: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: single LSTM (+ optional MHA) + independent heads.

        Args:
            x: Input [B, T, input_dim]
            external_features: Unused, kept for interface compatibility

        Returns:
            predictions: Dict with 't1', 't2', 't3' keys
        """
        hidden_states, (h_n, _) = self.lstm(x)  # [B, T, H*dirs]

        if self.use_attention:
            # MHA attends over all hidden states → [B, effective_hidden]
            features = self.mha(hidden_states)
        else:
            # Use last hidden state
            if self.bidirectional:
                forward_final = h_n[-2, :, :]  # [B, H]
                backward_final = h_n[-1, :, :]  # [B, H]
                features = torch.cat([forward_final, backward_final], dim=-1)
            else:
                features = h_n[-1, :, :]  # [B, H]

            features = self.dropout(features)

        predictions = {}
        for horizon, head in self.heads.items():
            predictions[horizon] = torch.abs(head(features).squeeze(-1))  # [B]

        return predictions

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
