"""
LSTM Backbone Module
====================
Pure LSTM encoder with single responsibility: encode input sequence to hidden states.
No forecasting logic - just sequence encoding.
"""

import torch
import torch.nn as nn


class LSTMBackbone(nn.Module):
    """
    Pure LSTM encoder backbone.
    
    Encodes input sequences to hidden representations without any task-specific logic.
    This is the foundational encoding layer used by all model variants.
    
    Args:
        input_dim: Number of input features per timestep
        hidden_dim: LSTM hidden dimension (default: 64)
        num_layers: Number of stacked LSTM layers (default: 1)
        dropout: Dropout probability between LSTM layers (default: 0.1)
        bidirectional: Use bidirectional LSTM (default: False)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Output dimension accounts for bidirectional concatenation
        self.output_dim = hidden_dim * self.num_directions
        
        # LSTM layer - dropout only applied if num_layers > 1
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to hidden states.
        
        Args:
            x: Input tensor of shape [B, T, input_dim]
               B = batch size, T = sequence length (lookback)
               
        Returns:
            hidden_states: Tensor of shape [B, T, output_dim]
                          where output_dim = hidden_dim * num_directions
        """
        # x: [B, T, input_dim] -> hidden_states: [B, T, hidden_dim * num_directions]
        hidden_states, (h_n, c_n) = self.lstm(x)
        
        return hidden_states
    
    def get_final_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the final hidden state (useful for non-attention models).
        
        Args:
            x: Input tensor of shape [B, T, input_dim]
            
        Returns:
            final_hidden: Tensor of shape [B, output_dim]
        """
        hidden_states, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            # Concatenate final states from both directions
            # h_n shape: [num_layers * 2, B, hidden_dim]
            forward_final = h_n[-2, :, :]  # Last layer, forward
            backward_final = h_n[-1, :, :]  # Last layer, backward
            final_hidden = torch.cat([forward_final, backward_final], dim=-1)
        else:
            # h_n shape: [num_layers, B, hidden_dim]
            final_hidden = h_n[-1, :, :]  # Last layer
            
        return final_hidden

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
