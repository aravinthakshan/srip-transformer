
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class T1LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=False):
        super().__init__()
        self.use_rating_curve = use_rating_curve
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        
        # Adjust final layer input size if using rating curve
        final_input_size = hidden_size
        if use_rating_curve:
            final_input_size += 1  # Add 1 for rating curve input
            
        self.fc = nn.Linear(final_input_size, 1)
        
    def forward(self, x, rating_pred=None):
        lstm_out, _ = self.lstm(x)
        # Use the last output as query, all outputs as key and value
        query = lstm_out[:, -1:, :]  # Shape: (batch, 1, hidden_size)
        attn_out, _ = self.attention(query, lstm_out, lstm_out)
        attn_features = attn_out.squeeze(1)  # Shape: (batch, hidden_size)
        
        if self.use_rating_curve and rating_pred is not None:
            # Concatenate attention output with rating curve prediction
            combined = torch.cat([attn_features, rating_pred.unsqueeze(-1)], dim=-1)
            return self.fc(combined).squeeze(-1)
        else:
            return self.fc(attn_features).squeeze(-1)

class T2LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True):
        super().__init__()
        self.use_rating_curve = use_rating_curve
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Create embeddings for additional inputs
        if self.use_rating_curve:
            self.additional_features = 2  # T+1 pred + rating curve
        else:
            self.additional_features = 1  # T+1 pred only
            
        self.feature_projection = nn.Linear(self.additional_features, hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, t1_pred, rating_pred=None):
        lstm_out, _ = self.lstm(x)
        
        # Prepare additional features
        if self.use_rating_curve and rating_pred is not None:
            additional_features = torch.stack([t1_pred, rating_pred], dim=-1)
        else:
            additional_features = t1_pred.unsqueeze(-1)
            
        # Project additional features to hidden_size
        additional_projected = self.feature_projection(additional_features).unsqueeze(1)  # (batch, 1, hidden_size)
        
        # Combine LSTM output with additional features
        combined_features = torch.cat([lstm_out, additional_projected], dim=1)  # (batch, seq_len+1, hidden_size)
        
        # Use the additional features as query
        attn_out, _ = self.attention(additional_projected, combined_features, combined_features)
        return self.fc(attn_out.squeeze(1)).squeeze(-1)

class T3LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True):
        super().__init__()
        self.use_rating_curve = use_rating_curve
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Create embeddings for additional inputs
        if self.use_rating_curve:
            self.additional_features = 4  # T+1 + T+2 + 2 rating curves
        else:
            self.additional_features = 2  # T+1 + T+2 only
            
        self.feature_projection = nn.Linear(self.additional_features, hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, t1_pred, t2_pred, rating_pred_t1=None, rating_pred_t2=None):
        lstm_out, _ = self.lstm(x)
        
        # Prepare additional features
        if self.use_rating_curve and rating_pred_t1 is not None and rating_pred_t2 is not None:
            additional_features = torch.stack([t1_pred, t2_pred, rating_pred_t1, rating_pred_t2], dim=-1)
        else:
            additional_features = torch.stack([t1_pred, t2_pred], dim=-1)
            
        # Project additional features to hidden_size
        additional_projected = self.feature_projection(additional_features).unsqueeze(1)  # (batch, 1, hidden_size)
        
        # Combine LSTM output with additional features
        combined_features = torch.cat([lstm_out, additional_projected], dim=1)  # (batch, seq_len+1, hidden_size)
        
        # Use the additional features as query
        attn_out, _ = self.attention(additional_projected, combined_features, combined_features)
        return self.fc(attn_out.squeeze(1)).squeeze(-1)
