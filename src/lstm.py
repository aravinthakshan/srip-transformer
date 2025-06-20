import torch
import torch.nn as nn

### ------------------------ Models ------------------------ ###
class T1LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):  # Fixed: was missing __
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class T2LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True):  # Fixed: was missing __
        super().__init__()
        self.use_rating_curve = use_rating_curve
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Adjust the input size of the first fully connected layer based on use_rating_curve
        if self.use_rating_curve:
            self.fc1 = nn.Linear(hidden_size + 2, hidden_size // 2)  # LSTM + T+1 pred + rating curve
        else:
            self.fc1 = nn.Linear(hidden_size + 1, hidden_size // 2)  # LSTM + T+1 pred only
            
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, t1_pred, rating_pred=None):
        out, _ = self.lstm(x)
        lstm_out = out[:, -1, :]
        
        if self.use_rating_curve and rating_pred is not None:
            combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), rating_pred.unsqueeze(-1)], dim=-1)
        else:
            combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1)], dim=-1)
            
        out = self.relu(self.fc1(combined))
        return self.fc2(out).squeeze(-1)

class T3LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, use_rating_curve=True):  # Fixed: was missing __
        super().__init__()
        self.use_rating_curve = use_rating_curve
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Adjust the input size based on use_rating_curve
        if self.use_rating_curve:
            self.fc1 = nn.Linear(hidden_size + 4, hidden_size // 2)  # LSTM + T+1 + T+2 + 2 rating curves
        else:
            self.fc1 = nn.Linear(hidden_size + 2, hidden_size // 2)  # LSTM + T+1 + T+2 only
            
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, t1_pred, t2_pred, rating_pred_t1=None, rating_pred_t2=None):
        out, _ = self.lstm(x)
        lstm_out = out[:, -1, :]
        
        if self.use_rating_curve and rating_pred_t1 is not None and rating_pred_t2 is not None:
            combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), t2_pred.unsqueeze(-1),
                                 rating_pred_t1.unsqueeze(-1), rating_pred_t2.unsqueeze(-1)], dim=-1)
        else:
            combined = torch.cat([lstm_out, t1_pred.unsqueeze(-1), t2_pred.unsqueeze(-1)], dim=-1)
            
        out = self.relu(self.fc1(combined))
        return self.fc2(out).squeeze(-1)