import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: (B, Seq, 1) -> (B, Seq, d_model)
        x = self.input_proj(x)
        if x.size(1) > self.pos_encoder.size(1):
             # basic safety
             x = x[:, :self.pos_encoder.size(1), :]
             
        # Add simpler positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]
        out = self.transformer_encoder(x)
        # Take last time step
        return self.fc(out[:, -1, :]).squeeze(1)
