"""
LSTM model for trajectory prediction
Predicts X,Y coordinates from PL and RMS features
"""

import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=3, output_size=2, dropout=0.3):
        """
        Improved LSTM model for trajectory prediction
        
        Parameters:
        -----------
        input_size : int
            Number of input features (default: 7)
        hidden_size : int
            Hidden layer size
        num_layers : int
            Number of LSTM layers
        output_size : int
            Output size (X, Y coordinates = 2)
        dropout : float
            Dropout rate
        """
        super(TrajectoryLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Output layers with residual connections
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights properly"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
        --------
        torch.Tensor : Output tensor of shape (batch_size, seq_len, output_size)
        """
        # Normalize input
        x = self.input_norm(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply output layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        output = self.fc2(out)
        
        return output