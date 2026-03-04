"""
Multi-Head Attention Module
===========================
Standalone MHA block for attending over LSTM hidden states.
No task-specific logic - pure attention mechanism.
"""

import torch
import torch.nn as nn
import math


class MHABlock(nn.Module):
    """
    Multi-Head Attention block.
    
    Applies scaled dot-product attention over sequence of hidden states.
    Can be used for self-attention or cross-attention with optional query.
    
    Args:
        embed_dim: Embedding/hidden dimension
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        use_layer_norm: Apply layer normalization (default: True)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Optional layer normalization (residual connection style)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(embed_dim)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        h: torch.Tensor,
        query: torch.Tensor = None,
        return_weights: bool = False
    ) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            h: Hidden states tensor of shape [B, T, H]
               B = batch size, T = sequence length, H = embed_dim
            query: Optional query tensor of shape [B, 1, H] for cross-attention.
                   If None, uses last timestep of h as query (standard approach).
            return_weights: If True, also return attention weights
            
        Returns:
            attended: Attended representation of shape [B, H]
            attn_weights: (optional) Attention weights of shape [B, num_heads, 1, T]
        """
        batch_size, seq_len, hidden_dim = h.shape
        
        # Default query: use last timestep
        if query is None:
            query = h[:, -1:, :]  # [B, 1, H]
        
        # Apply multi-head attention
        # query: [B, 1, H], key/value: [B, T, H]
        attn_output, attn_weights = self.attention(
            query=query,
            key=h,
            value=h,
            need_weights=return_weights,
            average_attn_weights=False,
        )
        
        # attn_output: [B, 1, H] -> [B, H]
        attended = attn_output.squeeze(1)
        
        # Apply dropout
        attended = self.dropout(attended)
        
        # Optional: layer norm with residual (add query)
        if self.use_layer_norm:
            attended = self.layer_norm(attended + query.squeeze(1))
        
        if return_weights:
            return attended, attn_weights
        return attended
    
    def forward_full_sequence(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention over the full sequence.
        
        Useful for when you want attention-weighted representations
        for every timestep, not just a single query.
        
        Args:
            h: Hidden states tensor of shape [B, T, H]
            
        Returns:
            attended_seq: Attention-enhanced sequence of shape [B, T, H]
        """
        # Self-attention: query = key = value = h
        attn_output, _ = self.attention(
            query=h,
            key=h,
            value=h,
            need_weights=False
        )
        
        # Apply dropout and layer norm with residual
        attn_output = self.dropout(attn_output)
        
        if self.use_layer_norm:
            # Apply layer norm per-timestep
            attn_output = self.layer_norm(attn_output + h)
            
        return attn_output

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PooledMHABlock(MHABlock):
    """
    MHA block with configurable pooling strategy.
    
    Extends MHABlock with various pooling options for creating
    a single vector representation from attended sequence.
    
    Args:
        embed_dim: Embedding/hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        pooling: Pooling strategy ('last', 'mean', 'max', 'attention_weighted')
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        pooling: str = 'last'
    ):
        super().__init__(embed_dim, num_heads, dropout)
        
        self.pooling = pooling
        
        # For attention-weighted pooling
        if pooling == 'attention_weighted':
            self.pooling_weights = nn.Linear(embed_dim, 1)
            
    def forward_pooled(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention and pool to single vector.
        
        Args:
            h: Hidden states tensor of shape [B, T, H]
            
        Returns:
            pooled: Pooled representation of shape [B, H]
        """
        # First apply self-attention over sequence
        attended_seq = self.forward_full_sequence(h)  # [B, T, H]
        
        # Then pool based on strategy
        if self.pooling == 'last':
            pooled = attended_seq[:, -1, :]
        elif self.pooling == 'mean':
            pooled = torch.mean(attended_seq, dim=1)
        elif self.pooling == 'max':
            pooled, _ = torch.max(attended_seq, dim=1)
        elif self.pooling == 'attention_weighted':
            weights = torch.softmax(self.pooling_weights(attended_seq), dim=1)  # [B, T, 1]
            pooled = torch.sum(attended_seq * weights, dim=1)  # [B, H]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
            
        return pooled
