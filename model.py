import torch
import torch.nn as nn

class DisasterLSTM(nn.Module):
    def __init__(self, num_regions, num_features, embed_dim, hidden_size, num_layers, pred_len):
        super().__init__()
        
        # Region Embedding
        self.region_embed = nn.Embedding(num_regions, embed_dim)
        
        # LSTM for the weather sequence
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully connected head
        # Input size: LSTM hidden_size + Embedding dim + 1 (for the region baseline scalar)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + embed_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len)
        )

    def forward(self, x_seq, region_idx, baseline):
        # Get region embedding
        embed = self.region_embed(region_idx) 
        
        # Pass sequence through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_seq)
        final_h = h_n[-1] 
        
        # Concatenate temporal context, region identity, and the historical baseline anchor
        combined = torch.cat([final_h, embed, baseline.unsqueeze(1)], dim=1)
        
        # Predict the next 5 weeks
        predictions = self.fc(combined)
        return predictions