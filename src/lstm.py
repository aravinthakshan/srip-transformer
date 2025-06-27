# import torch
# import torch.nn as nn
# import torch
# import torch.nn as nn

# class T1LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, activation='softplus'):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
#         self.fc1 = nn.Linear(hidden_size, hidden_size//2)
#         self.fc2 = nn.Linear(hidden_size//2, 1)
        
#         # Choose activation function for intermediate layer
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'softplus':
#             self.activation = nn.Softplus()
#         elif activation == 'elu':
#             self.activation = nn.ELU()
#         else:
#             self.activation = None
        
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         query = lstm_out[:, -1:, :]
#         attn_out, _ = self.attention(query, lstm_out, lstm_out)
        
#         # Use intermediate activation but not at the final output
#         hidden = self.fc1(attn_out.squeeze(1))
#         if self.activation is not None:
#             hidden = self.activation(hidden)
#         output = self.fc2(hidden).squeeze(-1)
        
#         return abs(output)


# class T2LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True, activation='softplus'):
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
#         if self.use_rating_curve:
#             self.additional_features = 2
#         else:
#             self.additional_features = 1
            
#         self.feature_projection = nn.Linear(self.additional_features, hidden_size)
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
#         self.fc1 = nn.Linear(hidden_size, hidden_size//2)
#         self.fc2 = nn.Linear(hidden_size//2, 1)
        
#         # Choose activation function for intermediate layer
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'softplus':
#             self.activation = nn.Softplus()
#         elif activation == 'elu':
#             self.activation = nn.ELU()
#         else:
#             self.activation = None
        
#     def forward(self, x, t1_pred, rating_pred=None):
#         lstm_out, _ = self.lstm(x)
        
#         if self.use_rating_curve and rating_pred is not None:
#             additional_features = torch.stack([t1_pred, rating_pred], dim=-1)
#         else:
#             additional_features = t1_pred.unsqueeze(-1)
            
#         additional_projected = self.feature_projection(additional_features).unsqueeze(1)
#         combined_features = torch.cat([lstm_out, additional_projected], dim=1)
        
#         attn_out, _ = self.attention(additional_projected, combined_features, combined_features)
        
#         # Use intermediate activation but not at the final output
#         hidden = self.fc1(attn_out.squeeze(1))
#         if self.activation is not None:
#             hidden = self.activation(hidden)
#         output = self.fc2(hidden).squeeze(-1)
        
#         return abs(output)


# class T3LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True, activation='softplus'):
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
#         if self.use_rating_curve:
#             self.additional_features = 4
#         else:
#             self.additional_features = 2
            
#         self.feature_projection = nn.Linear(self.additional_features, hidden_size)
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
#         self.fc1 = nn.Linear(hidden_size, hidden_size//2)
#         self.fc2 = nn.Linear(hidden_size//2, 1)
        
#         # Choose activation function for intermediate layer
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'softplus':
#             self.activation = nn.Softplus()
#         elif activation == 'elu':
#             self.activation = nn.ELU()
#         else:
#             self.activation = None
        
#     def forward(self, x, t1_pred, t2_pred, rating_pred_t1=None, rating_pred_t2=None):
#         lstm_out, _ = self.lstm(x)
        
#         if self.use_rating_curve and rating_pred_t1 is not None and rating_pred_t2 is not None:
#             additional_features = torch.stack([t1_pred, t2_pred, rating_pred_t1, rating_pred_t2], dim=-1)
#         else:
#             additional_features = torch.stack([t1_pred, t2_pred], dim=-1)
            
#         additional_projected = self.feature_projection(additional_features).unsqueeze(1)
#         combined_features = torch.cat([lstm_out, additional_projected], dim=1)
        
#         attn_out, _ = self.attention(additional_projected, combined_features, combined_features)
        
#         # Use intermediate activation but not at the final output
#         hidden = self.fc1(attn_out.squeeze(1))
#         if self.activation is not None:
#             hidden = self.activation(hidden)
#         output = self.fc2(hidden).squeeze(-1)
        
#         return abs(output)

# ## Option 1
# import torch
# import torch.nn as nn

# class T1LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, activation='softplus'):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        
#         # For T1, we don't have external predictions, so we use the final LSTM state as guidance
#         self.guidance_projection = nn.Linear(hidden_size, hidden_size)
#         self.final_projection = nn.Linear(hidden_size * 2, hidden_size)  # temporal + guidance
        
#         self.fc1 = nn.Linear(hidden_size, hidden_size//2)
#         self.fc2 = nn.Linear(hidden_size//2, 1)
        
#         # Choose activation function for intermediate layer
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'softplus':
#             self.activation = nn.Softplus()
#         elif activation == 'elu':
#             self.activation = nn.ELU()
#         else:
#             self.activation = None
        
#     def forward(self, x):
#         lstm_out, (hidden, _) = self.lstm(x)  # Get final hidden state
        
#         # Use final hidden state as guidance to attend over the sequence
#         guidance = self.guidance_projection(hidden.squeeze(0)).unsqueeze(1)  # (batch, 1, hidden)
        
#         # Use guidance to attend over the temporal sequence
#         attn_out, _ = self.attention(guidance, lstm_out, lstm_out)
        
#         # Combine attended temporal info with original guidance
#         temporal_summary = attn_out.squeeze(1)
#         combined = torch.cat([temporal_summary, guidance.squeeze(1)], dim=-1)
#         combined = self.final_projection(combined)
        
#         # Apply final layers
#         hidden = self.fc1(combined)
#         if self.activation is not None:
#             hidden = self.activation(hidden)
#         output = self.fc2(hidden).squeeze(-1)
        
#         return abs(output)


# class T2LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True, activation='softplus'):
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
#         # Determine number of guidance features
#         if self.use_rating_curve:
#             self.guidance_features = 2  # t1_pred + rating_pred
#         else:
#             self.guidance_features = 1  # t1_pred only
            
#         self.guidance_projection = nn.Linear(self.guidance_features, hidden_size)
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
#         self.final_projection = nn.Linear(hidden_size + self.guidance_features, hidden_size)
        
#         self.fc1 = nn.Linear(hidden_size, hidden_size//2)
#         self.fc2 = nn.Linear(hidden_size//2, 1)
        
#         # Choose activation function for intermediate layer
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'softplus':
#             self.activation = nn.Softplus()
#         elif activation == 'elu':
#             self.activation = nn.ELU()
#         else:
#             self.activation = None
        
#     def forward(self, x, t1_pred, rating_pred=None):
#         lstm_out, _ = self.lstm(x)
        
#         # Create guidance from predictions
#         if self.use_rating_curve and rating_pred is not None:
#             guidance_features = torch.stack([t1_pred, rating_pred], dim=-1)
#         else:
#             guidance_features = t1_pred.unsqueeze(-1)
            
#         # Project guidance to hidden dimension
#         guidance_projected = self.guidance_projection(guidance_features).unsqueeze(1)  # (batch, 1, hidden)
        
#         # Use guidance to attend over the temporal sequence
#         attn_out, _ = self.attention(guidance_projected, lstm_out, lstm_out)
        
#         # Combine attended temporal info with original guidance features
#         temporal_summary = attn_out.squeeze(1)
#         combined = torch.cat([temporal_summary, guidance_features], dim=-1)
#         combined = self.final_projection(combined)
        
#         # Apply final layers
#         hidden = self.fc1(combined)
#         if self.activation is not None:
#             hidden = self.activation(hidden)
#         output = self.fc2(hidden).squeeze(-1)
        
#         return abs(output)


# class T3LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True, activation='softplus'):
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
#         # Determine number of guidance features
#         if self.use_rating_curve:
#             self.guidance_features = 4  # t1_pred + t2_pred + rating_pred_t1 + rating_pred_t2
#         else:
#             self.guidance_features = 2  # t1_pred + t2_pred only
            
#         self.guidance_projection = nn.Linear(self.guidance_features, hidden_size)
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
#         self.final_projection = nn.Linear(hidden_size + self.guidance_features, hidden_size)
        
#         self.fc1 = nn.Linear(hidden_size, hidden_size//2)
#         self.fc2 = nn.Linear(hidden_size//2, 1)
        
#         # Choose activation function for intermediate layer
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'softplus':
#             self.activation = nn.Softplus()
#         elif activation == 'elu':
#             self.activation = nn.ELU()
#         else:
#             self.activation = None
        
#     def forward(self, x, t1_pred, t2_pred, rating_pred_t1=None, rating_pred_t2=None):
#         lstm_out, _ = self.lstm(x)
        
#         # Create guidance from all predictions
#         if self.use_rating_curve and rating_pred_t1 is not None and rating_pred_t2 is not None:
#             guidance_features = torch.stack([t1_pred, t2_pred, rating_pred_t1, rating_pred_t2], dim=-1)
#         else:
#             guidance_features = torch.stack([t1_pred, t2_pred], dim=-1)
            
#         # Project guidance to hidden dimension
#         guidance_projected = self.guidance_projection(guidance_features).unsqueeze(1)  # (batch, 1, hidden)
        
#         # Use guidance to attend over the temporal sequence
#         attn_out, _ = self.attention(guidance_projected, lstm_out, lstm_out)
        
#         # Combine attended temporal info with original guidance features
#         temporal_summary = attn_out.squeeze(1)
#         combined = torch.cat([temporal_summary, guidance_features], dim=-1)
#         combined = self.final_projection(combined)
        
#         # Apply final layers
#         hidden = self.fc1(combined)
#         if self.activation is not None:
#             hidden = self.activation(hidden)
#         output = self.fc2(hidden).squeeze(-1)
        
#         return abs(output)
    


# ## Tempor
# import torch
# import torch.nn as nn

# class T1LSTMModel_Option2(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, activation='softplus'):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        
#         # Pooling options for temporal summary
#         self.pooling_type = 'mean'  # 'mean', 'max', 'last', or 'attention_weighted'
#         if self.pooling_type == 'attention_weighted':
#             self.pooling_weights = nn.Linear(hidden_size, 1)
        
#         self.fc1 = nn.Linear(hidden_size, hidden_size//2)
#         self.fc2 = nn.Linear(hidden_size//2, 1)
        
#         # Choose activation function for intermediate layer
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'softplus':
#             self.activation = nn.Softplus()
#         elif activation == 'elu':
#             self.activation = nn.ELU()
#         else:
#             self.activation = None
        
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
        
#         # Self-attention on temporal sequence
#         temporal_attn, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
#         # Create temporal summary
#         if self.pooling_type == 'mean':
#             temporal_summary = torch.mean(temporal_attn, dim=1)
#         elif self.pooling_type == 'max':
#             temporal_summary = torch.max(temporal_attn, dim=1)[0]
#         elif self.pooling_type == 'last':
#             temporal_summary = temporal_attn[:, -1, :]
#         elif self.pooling_type == 'attention_weighted':
#             weights = torch.softmax(self.pooling_weights(temporal_attn), dim=1)
#             temporal_summary = torch.sum(temporal_attn * weights, dim=1)
        
#         # For T1, we only have temporal info
#         combined = temporal_summary
        
#         # Apply final layers
#         hidden = self.fc1(combined)
#         if self.activation is not None:
#             hidden = self.activation(hidden)
#         output = self.fc2(hidden).squeeze(-1)
        
#         return abs(output)


# class T2LSTMModel_Option2(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True, activation='softplus'):
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        
#         # Determine context features
#         if self.use_rating_curve:
#             self.context_features = 2  # t1_pred + rating_pred
#         else:
#             self.context_features = 1  # t1_pred only
        
#         # Pooling for temporal summary
#         self.pooling_type = 'attention_weighted'
#         self.pooling_weights = nn.Linear(hidden_size, 1)
        
#         # Combine temporal understanding with predictive context
#         self.final_projection = nn.Linear(hidden_size + self.context_features, hidden_size)
        
#         self.fc1 = nn.Linear(hidden_size, hidden_size//2)
#         self.fc2 = nn.Linear(hidden_size//2, 1)
        
#         # Choose activation function for intermediate layer
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'softplus':
#             self.activation = nn.Softplus()
#         elif activation == 'elu':
#             self.activation = nn.ELU()
#         else:
#             self.activation = None
        
#     def forward(self, x, t1_pred, rating_pred=None):
#         lstm_out, _ = self.lstm(x)
        
#         # Self-attention on temporal sequence
#         temporal_attn, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
#         # Create weighted temporal summary
#         weights = torch.softmax(self.pooling_weights(temporal_attn), dim=1)
#         temporal_summary = torch.sum(temporal_attn * weights, dim=1)
        
#         # Use predictions as additional context
#         if self.use_rating_curve and rating_pred is not None:
#             context = torch.stack([t1_pred, rating_pred], dim=-1)
#         else:
#             context = t1_pred.unsqueeze(-1)
        
#         # Combine temporal understanding with predictive context
#         combined = torch.cat([temporal_summary, context], dim=-1)
#         combined = self.final_projection(combined)
        
#         # Apply final layers
#         hidden = self.fc1(combined)
#         if self.activation is not None:
#             hidden = self.activation(hidden)
#         output = self.fc2(hidden).squeeze(-1)
        
#         return abs(output)


# class T3LSTMModel_Option2(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True, activation='softplus'):
#         super().__init__()
#         self.use_rating_curve = use_rating_curve
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        
#         # Determine context features
#         if self.use_rating_curve:
#             self.context_features = 4  # t1_pred + t2_pred + rating_pred_t1 + rating_pred_t2
#         else:
#             self.context_features = 2  # t1_pred + t2_pred only
        
#         # Pooling for temporal summary
#         self.pooling_type = 'attention_weighted'
#         self.pooling_weights = nn.Linear(hidden_size, 1)
        
#         # Combine temporal understanding with predictive context
#         self.final_projection = nn.Linear(hidden_size + self.context_features, hidden_size)
        
#         self.fc1 = nn.Linear(hidden_size, hidden_size//2)
#         self.fc2 = nn.Linear(hidden_size//2, 1)
        
#         # Choose activation function for intermediate layer
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'softplus':
#             self.activation = nn.Softplus()
#         elif activation == 'elu':
#             self.activation = nn.ELU()
#         else:
#             self.activation = None
        
#     def forward(self, x, t1_pred, t2_pred, rating_pred_t1=None, rating_pred_t2=None):
#         lstm_out, _ = self.lstm(x)
        
#         # Self-attention on temporal sequence
#         temporal_attn, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
#         # Create weighted temporal summary
#         weights = torch.softmax(self.pooling_weights(temporal_attn), dim=1)
#         temporal_summary = torch.sum(temporal_attn * weights, dim=1)
        
#         # Use all predictions as context
#         if self.use_rating_curve and rating_pred_t1 is not None and rating_pred_t2 is not None:
#             context = torch.stack([t1_pred, t2_pred, rating_pred_t1, rating_pred_t2], dim=-1)
#         else:
#             context = torch.stack([t1_pred, t2_pred], dim=-1)
        
#         # Combine temporal understanding with predictive context
#         combined = torch.cat([temporal_summary, context], dim=-1)
#         combined = self.final_projection(combined)
        
#         # Apply final layers
#         hidden = self.fc1(combined)
#         if self.activation is not None:
#             hidden = self.activation(hidden)
#         output = self.fc2(hidden).squeeze(-1)
        
#         return abs(output)

import torch
import torch.nn as nn

class T1LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, activation='softplus'):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        
        # Pooling options for temporal summary
        self.pooling_type = 'mean'  # 'mean', 'max', 'last', or 'attention_weighted'
        if self.pooling_type == 'attention_weighted':
            self.pooling_weights = nn.Linear(hidden_size, 1)
        
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
        
        # Self-attention on temporal sequence
        temporal_attn, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Create temporal summary
        if self.pooling_type == 'mean':
            temporal_summary = torch.mean(temporal_attn, dim=1)
        elif self.pooling_type == 'max':
            temporal_summary = torch.max(temporal_attn, dim=1)[0]
        elif self.pooling_type == 'last':
            temporal_summary = temporal_attn[:, -1, :]
        elif self.pooling_type == 'attention_weighted':
            weights = torch.softmax(self.pooling_weights(temporal_attn), dim=1)
            temporal_summary = torch.sum(temporal_attn * weights, dim=1)
        
        # For T1, we only have temporal info
        combined = temporal_summary
        
        # Apply final layers
        hidden = self.fc1(combined)
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
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        
        # Determine context features
        if self.use_rating_curve:
            self.context_features = 2  # t1_pred + rating_pred
        else:
            self.context_features = 1  # t1_pred only
        
        # Pooling for temporal summary
        self.pooling_type = 'attention_weighted'
        self.pooling_weights = nn.Linear(hidden_size, 1)
        
        # Combine temporal understanding with predictive context
        self.final_projection = nn.Linear(hidden_size + self.context_features, hidden_size)
        
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
        
        # Self-attention on temporal sequence
        temporal_attn, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Create weighted temporal summary
        weights = torch.softmax(self.pooling_weights(temporal_attn), dim=1)
        temporal_summary = torch.sum(temporal_attn * weights, dim=1)
        
        # Use predictions as additional context
        if self.use_rating_curve and rating_pred is not None:
            context = torch.stack([t1_pred, rating_pred], dim=-1)
        else:
            context = t1_pred.unsqueeze(-1)
        
        # Combine temporal understanding with predictive context
        combined = torch.cat([temporal_summary, context], dim=-1)
        combined = self.final_projection(combined)
        
        # Apply final layers
        hidden = self.fc1(combined)
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
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        
        # Determine context features
        if self.use_rating_curve:
            self.context_features = 4  # t1_pred + t2_pred + rating_pred_t1 + rating_pred_t2
        else:
            self.context_features = 2  # t1_pred + t2_pred only
        
        # Pooling for temporal summary
        self.pooling_type = 'attention_weighted'
        self.pooling_weights = nn.Linear(hidden_size, 1)
        
        # Combine temporal understanding with predictive context
        self.final_projection = nn.Linear(hidden_size + self.context_features, hidden_size)
        
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
        
        # Self-attention on temporal sequence
        temporal_attn, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Create weighted temporal summary
        weights = torch.softmax(self.pooling_weights(temporal_attn), dim=1)
        temporal_summary = torch.sum(temporal_attn * weights, dim=1)
        
        # Use all predictions as context
        if self.use_rating_curve and rating_pred_t1 is not None and rating_pred_t2 is not None:
            context = torch.stack([t1_pred, t2_pred, rating_pred_t1, rating_pred_t2], dim=-1)
        else:
            context = torch.stack([t1_pred, t2_pred], dim=-1)
        
        # Combine temporal understanding with predictive context
        combined = torch.cat([temporal_summary, context], dim=-1)
        combined = self.final_projection(combined)
        
        # Apply final layers
        hidden = self.fc1(combined)
        if self.activation is not None:
            hidden = self.activation(hidden)
        output = self.fc2(hidden).squeeze(-1)
        
        return abs(output)