import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class T1LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Final output layer
        self.fc = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply multi-head attention (self-attention on LSTM outputs)
        attn_out, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        
        # Layer normalization and residual connection
        attn_out = self.layer_norm(attn_out + lstm_out)
        
        # Use the last timestep output
        final_out = attn_out[:, -1, :]
        
        return self.fc(final_out).squeeze(-1)


class T2LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_heads=8, use_rating_curve=True):
        super().__init__()
        self.use_rating_curve = use_rating_curve
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Multi-head attention for LSTM outputs
        self.lstm_multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Determine feature dimension based on use_rating_curve
        if self.use_rating_curve:
            feature_dim = hidden_size + 2  # LSTM + T+1 pred + rating curve
        else:
            feature_dim = hidden_size + 1  # LSTM + T+1 pred only
        
        # Multi-head attention for combined features
        self.feature_multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=min(num_heads, feature_dim),  # Ensure num_heads <= feature_dim
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        self.fc = nn.Linear(feature_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, t1_pred, rating_pred=None):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply multi-head attention to LSTM outputs
        attn_lstm_out, _ = self.lstm_multihead_attn(lstm_out, lstm_out, lstm_out)
        attn_lstm_out = self.layer_norm1(attn_lstm_out + lstm_out)
        
        # Use the last timestep output from attention
        lstm_final = attn_lstm_out[:, -1, :]
        
        # Combine with predictions
        if self.use_rating_curve and rating_pred is not None:
            combined = torch.cat([lstm_final, t1_pred.unsqueeze(-1), rating_pred.unsqueeze(-1)], dim=-1)
        else:
            combined = torch.cat([lstm_final, t1_pred.unsqueeze(-1)], dim=-1)
        
        # Add sequence dimension for attention (treating combined features as a sequence of length 1)
        combined_seq = combined.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # Apply multi-head attention to combined features
        attn_combined, _ = self.feature_multihead_attn(combined_seq, combined_seq, combined_seq)
        attn_combined = self.layer_norm2(attn_combined + combined_seq)
        
        # Remove sequence dimension and apply final layer
        final_features = attn_combined.squeeze(1)
        out = self.relu(final_features)
        
        return self.fc(out).squeeze(-1)


class T3LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_heads=8, use_rating_curve=True):
        super().__init__()
        self.use_rating_curve = use_rating_curve
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Multi-head attention for LSTM outputs
        self.lstm_multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Determine feature dimension based on use_rating_curve
        if self.use_rating_curve:
            feature_dim = hidden_size + 4  # LSTM + T+1 + T+2 + 2 rating curves
        else:
            feature_dim = hidden_size + 2  # LSTM + T+1 + T+2 only
        
        # Multi-head attention for combined features
        self.feature_multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=min(num_heads, feature_dim),  # Ensure num_heads <= feature_dim
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        self.fc = nn.Linear(feature_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, t1_pred, t2_pred, rating_pred_t1=None, rating_pred_t2=None):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply multi-head attention to LSTM outputs
        attn_lstm_out, _ = self.lstm_multihead_attn(lstm_out, lstm_out, lstm_out)
        attn_lstm_out = self.layer_norm1(attn_lstm_out + lstm_out)
        
        # Use the last timestep output from attention
        lstm_final = attn_lstm_out[:, -1, :]
        
        # Combine with predictions
        if self.use_rating_curve and rating_pred_t1 is not None and rating_pred_t2 is not None:
            combined = torch.cat([lstm_final, t1_pred.unsqueeze(-1), t2_pred.unsqueeze(-1),
                                rating_pred_t1.unsqueeze(-1), rating_pred_t2.unsqueeze(-1)], dim=-1)
        else:
            combined = torch.cat([lstm_final, t1_pred.unsqueeze(-1), t2_pred.unsqueeze(-1)], dim=-1)
        
        # Add sequence dimension for attention (treating combined features as a sequence of length 1)
        combined_seq = combined.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # Apply multi-head attention to combined features
        attn_combined, _ = self.feature_multihead_attn(combined_seq, combined_seq, combined_seq)
        attn_combined = self.layer_norm2(attn_combined + combined_seq)
        
        # Remove sequence dimension and apply final layer
        final_features = attn_combined.squeeze(1)
        out = self.relu(final_features)
        
        return self.fc(out).squeeze(-1)
    
# ### ------------------------ Models ------------------------ ###
# class T1LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1):  # Fixed: was missing __
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
    
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :]).squeeze(-1)

# class T2LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True):  # Fixed: was missing __
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
#         # Adjust the input size of the first fully connected layer based on use_rating_curve
#         if self.use_rating_curve:
#             self.fc1 = nn.Linear(hidden_size + 2, hidden_size // 2)  # LSTM + T+1 pred + rating curve
#         else:
#             self.fc1 = nn.Linear(hidden_size + 1, hidden_size // 2)  # LSTM + T+1 pred only
            
#         self.fc2 = nn.Linear(hidden_size // 2, 1)
#         self.relu = nn.ReLU()
    
#     def forward(self, x, t1_pred, rating_pred=None):
#         out, _ = self.lstm(x)
#         lstm_out = out[:, -1, :]
        
#         if self.use_rating_curve and rating_pred is not None:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), rating_pred.unsqueeze(-1)], dim=-1)
#         else:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1)], dim=-1)
            
#         out = self.relu(self.fc1(combined))
#         return self.fc2(out).squeeze(-1)

# class T3LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True):  # Fixed: was missing __
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
#         # Adjust the input size based on use_rating_curve
#         if self.use_rating_curve:
#             self.fc1 = nn.Linear(hidden_size + 4, hidden_size // 2)  # LSTM + T+1 + T+2 + 2 rating curves
#         else:
#             self.fc1 = nn.Linear(hidden_size + 2, hidden_size // 2)  # LSTM + T+1 + T+2 only
            
#         self.fc2 = nn.Linear(hidden_size // 2, 1)
#         self.relu = nn.ReLU()
    
#     def forward(self, x, t1_pred, t2_pred, rating_pred_t1=None, rating_pred_t2=None):
#         out, _ = self.lstm(x)
#         lstm_out = out[:, -1, :]
        
#         if self.use_rating_curve and rating_pred_t1 is not None and rating_pred_t2 is not None:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), t2_pred.unsqueeze(-1),
#                                  rating_pred_t1.unsqueeze(-1), rating_pred_t2.unsqueeze(-1)], dim=-1)
#         else:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), t2_pred.unsqueeze(-1)], dim=-1)
            
#         out = self.relu(self.fc1(combined))
#         return self.fc2(out).squeeze(-1)


# import torch
# import torch.nn as nn

# ### ------------------------ Models ------------------------ ###

# class T1LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional
        
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :]).squeeze(-1)

# class T2LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True):
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
#         # Adjust input size for bidirectional output
#         if self.use_rating_curve:
#             self.fc1 = nn.Linear(hidden_size * 2 + 2, hidden_size // 2)  # Bidirectional LSTM + T+1 pred + rating curve
#         else:
#             self.fc1 = nn.Linear(hidden_size * 2 + 1, hidden_size // 2)  # Bidirectional LSTM + T+1 pred only
            
#         self.fc2 = nn.Linear(hidden_size // 2, 1)
#         self.relu = nn.ReLU()
        
#     def forward(self, x, t1_pred, rating_pred=None):
#         out, _ = self.lstm(x)
#         lstm_out = out[:, -1, :]
        
#         if self.use_rating_curve and rating_pred is not None:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), rating_pred.unsqueeze(-1)], dim=-1)
#         else:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1)], dim=-1)
            
#         out = self.relu(self.fc1(combined))
#         return self.fc2(out).squeeze(-1)

# class T3LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True):
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
#         # Adjust input size for bidirectional output
#         if self.use_rating_curve:
#             self.fc1 = nn.Linear(hidden_size * 2 + 4, hidden_size // 2)  # Bidirectional LSTM + T+1 + T+2 + 2 rating curves
#         else:
#             self.fc1 = nn.Linear(hidden_size * 2 + 2, hidden_size // 2)  # Bidirectional LSTM + T+1 + T+2 only
            
#         self.fc2 = nn.Linear(hidden_size // 2, 1)
#         self.relu = nn.ReLU()
        
#     def forward(self, x, t1_pred, t2_pred, rating_pred_t1=None, rating_pred_t2=None):
#         out, _ = self.lstm(x)
#         lstm_out = out[:, -1, :]
        
#         if self.use_rating_curve and rating_pred_t1 is not None and rating_pred_t2 is not None:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), t2_pred.unsqueeze(-1),
#                                 rating_pred_t1.unsqueeze(-1), rating_pred_t2.unsqueeze(-1)], dim=-1)
#         else:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), t2_pred.unsqueeze(-1)], dim=-1)
            
#         out = self.relu(self.fc1(combined))
#         return self.fc2(out).squeeze(-1)


# import torch
# import torch.nn as nn
# import math

# ### ------------------------ Models ------------------------ ###

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]

# class T1LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=32, num_layers=1, d_model=64, nhead=4, transformer_layers=2):
#         super().__init__()
        
#         # Input projection to transformer dimension
#         self.input_projection = nn.Linear(input_size, d_model)
        
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(d_model)
        
#         # Transformer encoder for contextual understanding
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, 
#             nhead=nhead, 
#             dim_feedforward=d_model * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
#         # Bidirectional LSTM
#         self.lstm = nn.LSTM(d_model, hidden_size, num_layers, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, x):
#         # Project input to transformer dimension
#         x_proj = self.input_projection(x)
        
#         # Add positional encoding
#         x_proj = self.pos_encoder(x_proj.transpose(0, 1)).transpose(0, 1)
        
#         # Transformer encoding
#         contextual_encoding = self.transformer_encoder(x_proj)
        
#         # LSTM processing
#         out, _ = self.lstm(contextual_encoding)
#         out = self.dropout(out)
        
#         return self.fc(out[:, -1, :]).squeeze(-1)

# class T2LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=32, num_layers=1, use_rating_curve=True, d_model=64, nhead=4, transformer_layers=2):
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
        
#         # Input projection to transformer dimension
#         self.input_projection = nn.Linear(input_size, d_model)
        
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(d_model)
        
#         # Transformer encoder for contextual understanding
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, 
#             nhead=nhead, 
#             dim_feedforward=d_model * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
#         # Bidirectional LSTM
#         self.lstm = nn.LSTM(d_model, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
#         # Adjust input size for bidirectional output
#         if self.use_rating_curve:
#             self.fc1 = nn.Linear(hidden_size * 2 + 2, hidden_size // 2)  # Bidirectional LSTM + T+1 pred + rating curve
#         else:
#             self.fc1 = nn.Linear(hidden_size * 2 + 1, hidden_size // 2)  # Bidirectional LSTM + T+1 pred only
            
#         self.fc2 = nn.Linear(hidden_size // 2, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, x, t1_pred, rating_pred=None):
#         # Project input to transformer dimension
#         x_proj = self.input_projection(x)
        
#         # Add positional encoding
#         x_proj = self.pos_encoder(x_proj.transpose(0, 1)).transpose(0, 1)
        
#         # Transformer encoding
#         contextual_encoding = self.transformer_encoder(x_proj)
        
#         # LSTM processing
#         out, _ = self.lstm(contextual_encoding)
#         lstm_out = self.dropout(out)
#         lstm_out = lstm_out[:, -1, :]
        
#         if self.use_rating_curve and rating_pred is not None:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), rating_pred.unsqueeze(-1)], dim=-1)
#         else:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1)], dim=-1)
            
#         out = self.relu(self.fc1(combined))
#         out = self.dropout(out)
#         return self.fc2(out).squeeze(-1)

# class T3LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=32, num_layers=1, use_rating_curve=True, d_model=64, nhead=4, transformer_layers=2):
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
        
#         # Input projection to transformer dimension
#         self.input_projection = nn.Linear(input_size, d_model)
        
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(d_model)
        
#         # Transformer encoder for contextual understanding
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, 
#             nhead=nhead, 
#             dim_feedforward=d_model * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
#         # Bidirectional LSTM
#         self.lstm = nn.LSTM(d_model, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
#         # Adjust input size for bidirectional output
#         if self.use_rating_curve:
#             self.fc1 = nn.Linear(hidden_size * 2 + 4, hidden_size // 2)  # Bidirectional LSTM + T+1 + T+2 + 2 rating curves
#         else:
#             self.fc1 = nn.Linear(hidden_size * 2 + 2, hidden_size // 2)  # Bidirectional LSTM + T+1 + T+2 only
            
#         self.fc2 = nn.Linear(hidden_size // 2, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, x, t1_pred, t2_pred, rating_pred_t1=None, rating_pred_t2=None):
#         # Project input to transformer dimension
#         x_proj = self.input_projection(x)
        
#         # Add positional encoding
#         x_proj = self.pos_encoder(x_proj.transpose(0, 1)).transpose(0, 1)
        
#         # Transformer encoding
#         contextual_encoding = self.transformer_encoder(x_proj)
        
#         # LSTM processing
#         out, _ = self.lstm(contextual_encoding)
#         lstm_out = self.dropout(out)
#         lstm_out = lstm_out[:, -1, :]
        
#         if self.use_rating_curve and rating_pred_t1 is not None and rating_pred_t2 is not None:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), t2_pred.unsqueeze(-1),
#                                 rating_pred_t1.unsqueeze(-1), rating_pred_t2.unsqueeze(-1)], dim=-1)
#         else:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), t2_pred.unsqueeze(-1)], dim=-1)
            
#         out = self.relu(self.fc1(combined))
#         out = self.dropout(out)
#         return self.fc2(out).squeeze(-1)

# import torch
# import torch.nn as nn
# import math

# ### ------------------------ Models ------------------------ ###

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]

# class T1LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, d_model=64, nhead=4, transformer_layers=2):
#         super().__init__()
        
#         # Input projection to transformer dimension
#         self.input_projection = nn.Linear(input_size, d_model)
        
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(d_model)
        
#         # Transformer encoder for contextual understanding
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, 
#             nhead=nhead, 
#             dim_feedforward=d_model * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
#         # LSTM
#         self.lstm = nn.LSTM(d_model, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, x):
#         # Project input to transformer dimension
#         x_proj = self.input_projection(x)
        
#         # Add positional encoding
#         x_proj = self.pos_encoder(x_proj.transpose(0, 1)).transpose(0, 1)
        
#         # Transformer encoding
#         contextual_encoding = self.transformer_encoder(x_proj)
        
#         # LSTM processing
#         out, _ = self.lstm(contextual_encoding)
#         out = self.dropout(out)
        
#         return self.fc(out[:, -1, :]).squeeze(-1)

# class T2LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True, d_model=64, nhead=4, transformer_layers=2):
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
        
#         # Input projection to transformer dimension
#         self.input_projection = nn.Linear(input_size, d_model)
        
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(d_model)
        
#         # Transformer encoder for contextual understanding
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, 
#             nhead=nhead, 
#             dim_feedforward=d_model * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
#         # LSTM
#         self.lstm = nn.LSTM(d_model, hidden_size, num_layers, batch_first=True)
        
#         # Adjust input size for unidirectional LSTM
#         if self.use_rating_curve:
#             self.fc1 = nn.Linear(hidden_size + 2, hidden_size // 2)  # LSTM + T+1 pred + rating curve
#         else:
#             self.fc1 = nn.Linear(hidden_size + 1, hidden_size // 2)  # LSTM + T+1 pred only
            
#         self.fc2 = nn.Linear(hidden_size // 2, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.0)
        
#     def forward(self, x, t1_pred, rating_pred=None):
#         # Project input to transformer dimension
#         x_proj = self.input_projection(x)
        
#         # Add positional encoding
#         x_proj = self.pos_encoder(x_proj.transpose(0, 1)).transpose(0, 1)
        
#         # Transformer encoding
#         contextual_encoding = self.transformer_encoder(x_proj)
        
#         # LSTM processing
#         out, _ = self.lstm(contextual_encoding)
#         lstm_out = self.dropout(out)
#         lstm_out = lstm_out[:, -1, :]
        
#         if self.use_rating_curve and rating_pred is not None:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), rating_pred.unsqueeze(-1)], dim=-1)
#         else:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1)], dim=-1)
            
#         out = self.relu(self.fc1(combined))
#         out = self.dropout(out)
#         return self.fc2(out).squeeze(-1)

# class T3LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True, d_model=64, nhead=4, transformer_layers=2):
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
        
#         # Input projection to transformer dimension
#         self.input_projection = nn.Linear(input_size, d_model)
        
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(d_model)
        
#         # Transformer encoder for contextual understanding
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, 
#             nhead=nhead, 
#             dim_feedforward=d_model * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
#         # LSTM
#         self.lstm = nn.LSTM(d_model, hidden_size, num_layers, batch_first=True)
        
#         # Adjust input size for unidirectional LSTM
#         if self.use_rating_curve:
#             self.fc1 = nn.Linear(hidden_size + 4, hidden_size // 2)  # LSTM + T+1 + T+2 + 2 rating curves
#         else:
#             self.fc1 = nn.Linear(hidden_size + 2, hidden_size // 2)  # LSTM + T+1 + T+2 only
            
#         self.fc2 = nn.Linear(hidden_size // 2, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, x, t1_pred, t2_pred, rating_pred_t1=None, rating_pred_t2=None):
#         # Project input to transformer dimension
#         x_proj = self.input_projection(x)
        
#         # Add positional encoding
#         x_proj = self.pos_encoder(x_proj.transpose(0, 1)).transpose(0, 1)
        
#         # Transformer encoding
#         contextual_encoding = self.transformer_encoder(x_proj)
        
#         # LSTM processing
#         out, _ = self.lstm(contextual_encoding)
#         lstm_out = self.dropout(out)
#         lstm_out = lstm_out[:, -1, :]
        
#         if self.use_rating_curve and rating_pred_t1 is not None and rating_pred_t2 is not None:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), t2_pred.unsqueeze(-1),
#                                 rating_pred_t1.unsqueeze(-1), rating_pred_t2.unsqueeze(-1)], dim=-1)
#         else:
#             combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), t2_pred.unsqueeze(-1)], dim=-1)
            
#         out = self.relu(self.fc1(combined))
#         out = self.dropout(out)
#         return self.fc2(out).squeeze(-1)