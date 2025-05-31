import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.data.loader import load_cir_data
from src.data.preprocessing import scale_and_sequence
from src.config import DATA_CONFIG
import numpy as np
import pandas as pd
import random
import time
import math

class MixtureGRUModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, num_layers=3, num_mixtures=10, dropout=0.4):
        super(MixtureGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        
        # Increase capacity for better range handling
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Range predictor with wider range
        self.range_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Predict min and max
        )
        
        # Attention with range awareness
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),  # +2 for range info
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Mixture parameters with range awareness
        self.mixture_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim * 2),  # +2 for range info
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_mixtures * 3)
        )
        
        # Value range embeddings
        self.range_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Predict value range with extended range
        range_context = torch.mean(gru_out, dim=1)
        predicted_range = self.range_predictor(range_context)
        range_min, range_max = predicted_range[:, 0], predicted_range[:, 1]
        
        # Add padding to range to allow for higher values
        range_span = range_max - range_min
        range_min = range_min - range_span * 0.2  # Increase lower padding
        range_max = range_max + range_span * 0.3  # Increase upper padding more
        range_span = range_max - range_min  # Recalculate span
        
        # Create range embeddings
        range_info = torch.stack([range_min, range_max], dim=1)
        range_embed = self.range_embedding(range_info)
        
        # Attention with range awareness
        range_expanded = range_info.unsqueeze(1).expand(-1, gru_out.size(1), -1)
        attention_input = torch.cat([gru_out, range_expanded], dim=2)
        attention_weights = F.softmax(self.attention(attention_input), dim=1)
        context = torch.sum(attention_weights * gru_out, dim=1)
        
        # Add range information to context
        context_with_range = torch.cat([context, range_info], dim=1)
        
        # Generate mixture parameters with range awareness
        mixture_params = self.mixture_layer(context_with_range)
        
        # Split and process mixture parameters
        means = mixture_params[:, :self.num_mixtures]
        stds = F.softplus(mixture_params[:, self.num_mixtures:2*self.num_mixtures]) + 1e-3
        weights = F.softmax(mixture_params[:, 2*self.num_mixtures:], dim=1)
        
        # Scale means with more aggressive range
        means = range_min.unsqueeze(1) + F.sigmoid(means) * range_span.unsqueeze(1) * 1.2
        # Scale stds based on range span with increased variation
        stds = stds * range_span.unsqueeze(1) * 0.15
        
        if self.training:
            # Sample from mixture during training
            component_idx = torch.multinomial(weights, 1).squeeze()
            selected_means = means[torch.arange(batch_size), component_idx]
            selected_stds = stds[torch.arange(batch_size), component_idx]
            
            # Add noise scaled by the range with increased variation
            noise = torch.randn_like(selected_means) * selected_stds * 1.2
            predictions = selected_means + noise
            
            # Allow predictions to go beyond the range more during training
            predictions = torch.clamp(predictions, 
                                   min=range_min - range_span * 0.2,
                                   max=range_max + range_span * 0.3)
        else:
            # During inference, use weighted average with controlled randomness
            predictions = torch.sum(weights.unsqueeze(-1) * means.unsqueeze(-1), dim=1).squeeze()
            avg_std = torch.sum(weights * stds, dim=1)
            
            # Add scaled noise during inference with more variation
            noise = torch.randn_like(predictions) * avg_std * 1.2
            predictions = predictions + noise
            
            # Allow predictions to go beyond the range more during inference
            predictions = torch.clamp(predictions, 
                                   min=range_min - range_span * 0.2,
                                   max=range_max + range_span * 0.3)
        
        return predictions, means, stds, weights, range_min, range_max

class MixtureLoss(nn.Module):
    def __init__(self, range_weight=0.15):  # Reduced range weight to allow more flexibility
        super(MixtureLoss, self).__init__()
        self.range_weight = range_weight
    
    def forward(self, predictions, targets, means, stds, weights, range_min, range_max):
        # Basic mixture loss
        exponents = -0.5 * ((targets.unsqueeze(1) - means) / stds) ** 2
        component_probs = (1.0 / (stds * math.sqrt(2 * math.pi))) * torch.exp(exponents)
        weighted_probs = weights * component_probs
        nll = -torch.log(torch.sum(weighted_probs, dim=1) + 1e-10)
        
        # Range prediction loss with more flexibility
        batch_min = targets.min()
        batch_max = targets.max()
        range_loss = F.smooth_l1_loss(range_min, batch_min) + F.smooth_l1_loss(range_max, batch_max)
        
        # Add MSE loss for stability
        mse = F.smooth_l1_loss(predictions, targets)  # Changed to smooth L1 loss
        
        # Enhanced diversity loss
        diversity_loss = -torch.mean(torch.std(means, dim=1)) - 0.1 * torch.mean(torch.abs(means[:, 1:] - means[:, :-1]))
        
        # Combine losses with adjusted weights
        total_loss = nll.mean() + 0.15 * mse + self.range_weight * range_loss + 0.15 * diversity_loss
        return total_loss

def train_gru_on_all(processed_dir: str, model_variant: str = "gru", 
                     batch_size: int = 32, epochs: int = 300, 
                     lr: float = 0.001, seq_len: int = 15):
    """
    Train GRU model with specific optimizations for position estimation
    """
    # Generate random seed based on current time
    random_seed = int(time.time() * 1000) % 100000
    print(f"Using random seed: {random_seed}")
    
    # Set random seeds
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Load data using dataset from config
    df = load_cir_data(processed_dir, filter_keyword=DATA_CONFIG['datasets'][0])
    print(f"Loaded {len(df)} data points from {DATA_CONFIG['datasets'][0]}")
    
    # Scale and create sequences
    X_seq, y_seq, x_scaler, y_scaler = scale_and_sequence(df, seq_len=seq_len)
    
    # Split data with random seed
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=random_seed, shuffle=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(random_seed + 1)  # Different seed for loader
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=batch_size,
        drop_last=False
    )
    
    # Create model with mixture outputs
    model = MixtureGRUModel(
        input_dim=2,
        hidden_dim=256,
        num_layers=3,
        num_mixtures=10,
        dropout=0.4
    )
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)  # Use orthogonal initialization
                elif 'bias' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -0.1, 0.1)
    
    model.apply(init_weights)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Use mixture loss
    criterion = MixtureLoss(range_weight=0.15).to(device)
    
    # Use Lion optimizer for better convergence
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.99)
    )
    
    # Use OneCycle learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    train_loss_hist = []
    val_loss_hist = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds, means, stds, weights, range_min, range_max = model(X_batch)
            loss = criterion(preds, y_batch, means, stds, weights, range_min, range_max)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        y_val_actual, y_val_pred = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds, means, stds, weights, range_min, range_max = model(X_batch)
                loss = criterion(preds, y_batch, means, stds, weights, range_min, range_max)
                val_loss += loss.item()
                val_batches += 1
                
                y_val_actual.extend(y_batch.cpu().numpy())
                y_val_pred.extend(preds.cpu().numpy())
        
        val_loss /= val_batches
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            # Check prediction diversity
            pred_std = np.std(y_val_pred)
            print(f"  Prediction std: {pred_std:.6f}")
    
    # Generate predictions on full dataset
    model.eval()
    with torch.no_grad():
        full_preds_scaled = model(X_seq.to(device))[0].cpu().numpy()
        full_targets_scaled = y_seq.numpy()
    
    # Inverse transform
    full_preds = y_scaler.inverse_transform(full_preds_scaled.reshape(-1, 1)).flatten()
    full_targets = y_scaler.inverse_transform(full_targets_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(np.mean((full_targets - full_preds) ** 2))
    
    print(f"\nFinal Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"Prediction range: [{full_preds.min():.2f}, {full_preds.max():.2f}]")
    print(f"Target range: [{full_targets.min():.2f}, {full_targets.max():.2f}]")
    print(f"Prediction std: {np.std(full_preds):.4f}")
    print(f"Target std: {np.std(full_targets):.4f}")
    
    return {
        'r_actual': full_targets.tolist(),
        'r_pred': full_preds.tolist(),
        'train_loss': train_loss_hist,
        'val_loss': val_loss_hist,
        'rmse': rmse,
        'original_df_size': len(df),
        'sequence_size': len(full_targets),
        'seq_len': seq_len
    }

# Usage example
if __name__ == "__main__":
    # Example usage with your data
    results = train_gru_on_all(
        processed_dir="your_data_directory",
        batch_size=32,
        epochs=300,
        lr=0.001,
        seq_len=15
    )