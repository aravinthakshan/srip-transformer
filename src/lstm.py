import torch
import torch.nn as nn
import torch
import torch.nn as nn

class T1LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, activation='softplus'):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, 1)
        
        # Choose activation function for intermediate layer
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = None
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        query = lstm_out[:, -1:, :]
        attn_out, _ = self.attention(query, lstm_out, lstm_out)
        
        # Use intermediate activation but not at the final output
        hidden = self.fc1(attn_out.squeeze(1))
        if self.activation is not None:
            hidden = self.activation(hidden)
        output = self.fc2(hidden).squeeze(-1)
        
        return abs(output)


class T2LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True, activation='softplus'):
        super().__init__()
        self.use_rating_curve = use_rating_curve
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        if self.use_rating_curve:
            self.additional_features = 2
        else:
            self.additional_features = 1
            
        self.feature_projection = nn.Linear(self.additional_features, hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, 1)
        
        # Choose activation function for intermediate layer
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = None
        
    def forward(self, x, t1_pred, rating_pred=None):
        lstm_out, _ = self.lstm(x)
        
        if self.use_rating_curve and rating_pred is not None:
            additional_features = torch.stack([t1_pred, rating_pred], dim=-1)
        else:
            additional_features = t1_pred.unsqueeze(-1)
            
        additional_projected = self.feature_projection(additional_features).unsqueeze(1)
        combined_features = torch.cat([lstm_out, additional_projected], dim=1)
        
        attn_out, _ = self.attention(additional_projected, combined_features, combined_features)
        
        # Use intermediate activation but not at the final output
        hidden = self.fc1(attn_out.squeeze(1))
        if self.activation is not None:
            hidden = self.activation(hidden)
        output = self.fc2(hidden).squeeze(-1)
        
        return abs(output)


class T3LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True, activation='softplus'):
        super().__init__()
        self.use_rating_curve = use_rating_curve
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        if self.use_rating_curve:
            self.additional_features = 4
        else:
            self.additional_features = 2
            
        self.feature_projection = nn.Linear(self.additional_features, hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, 1)
        
        # Choose activation function for intermediate layer
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = None
        
    def forward(self, x, t1_pred, t2_pred, rating_pred_t1=None, rating_pred_t2=None):
        lstm_out, _ = self.lstm(x)
        
        if self.use_rating_curve and rating_pred_t1 is not None and rating_pred_t2 is not None:
            additional_features = torch.stack([t1_pred, t2_pred, rating_pred_t1, rating_pred_t2], dim=-1)
        else:
            additional_features = torch.stack([t1_pred, t2_pred], dim=-1)
            
        additional_projected = self.feature_projection(additional_features).unsqueeze(1)
        combined_features = torch.cat([lstm_out, additional_projected], dim=1)
        
        attn_out, _ = self.attention(additional_projected, combined_features, combined_features)
        
        # Use intermediate activation but not at the final output
        hidden = self.fc1(attn_out.squeeze(1))
        if self.activation is not None:
            hidden = self.activation(hidden)
        output = self.fc2(hidden).squeeze(-1)
        
        return abs(output)