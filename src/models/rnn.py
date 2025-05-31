import torch
import torch.nn as nn


class TrajectoryRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        """
        Initialize the RNN model for trajectory prediction
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of features in the hidden state
        num_layers : int
            Number of recurrent layers
        dropout : float
            Dropout rate between RNN layers (except last layer)
        """
        super(TrajectoryRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Main RNN layer (using LSTM cells)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True  # Input shape: (batch, seq_len, features)
        )
        
        # Output layer to predict X,Y coordinates
        self.output_layer = nn.Linear(hidden_size, 2)  # 2 for X,Y coordinates
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights for better training"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, x, hidden=None):
        """
        Forward pass of the model
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size)
        hidden : tuple of torch.Tensor, optional
            Initial hidden state and cell state for LSTM
            
        Returns:
        --------
        torch.Tensor
            Predicted coordinates of shape (batch_size, sequence_length, 2)
        tuple of torch.Tensor
            Final hidden state and cell state
        """
        # Run RNN/LSTM
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Predict coordinates for each timestep
        predictions = self.output_layer(rnn_out)
        
        return predictions, hidden
    
    def init_hidden(self, batch_size, device='cpu'):
        """
        Initialize hidden state
        
        Parameters:
        -----------
        batch_size : int
            Size of the batch
        device : str
            Device to create the hidden state on ('cpu' or 'cuda')
            
        Returns:
        --------
        tuple of torch.Tensor
            Initial hidden state and cell state
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        return (h0, c0)
