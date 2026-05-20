import torch
import torch.nn as nn

class DisasterLSTM(nn.Module):
    def __init__(self, num_regions, num_features, embed_dim, hidden_size, num_layers, pred_len):
        super().__init__()
        
        # Region Embedding: Let the network learn similarities between regions natively
        self.region_embed = nn.Embedding(num_regions, embed_dim)
        
        # LSTM for the weather sequence
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully connected head to output the 5 weeks
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len)
        )

    def forward(self, x_seq, region_idx):
        # x_seq: (batch, seq_len, features)
        # region_idx: (batch)
        
        # Get region embedding
        embed = self.region_embed(region_idx)  # (batch, embed_dim)
        
        # Pass sequence through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_seq)
        
        # Take the final hidden state from the top layer of the LSTM
        final_h = h_n[-1]  # (batch, hidden_size)
        
        # Concatenate temporal context with region identity
        combined = torch.cat([final_h, embed], dim=1)
        
        # Predict the next 5 weeks
        predictions = self.fc(combined)
        return predictions