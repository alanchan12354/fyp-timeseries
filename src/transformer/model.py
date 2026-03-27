import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1, input_size=8):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: (B, Seq, input_size) -> (B, Seq, d_model)
        x = self.input_proj(x)
        seq_len = x.size(1)
        base_pos_len = self.pos_encoder.size(1)

        # Keep full sequence context even when seq_len exceeds the learned
        # positional table by tiling the table instead of truncating inputs.
        if seq_len <= base_pos_len:
            pos = self.pos_encoder[:, :seq_len, :]
        else:
            repeats = math.ceil(seq_len / base_pos_len)
            pos = self.pos_encoder.repeat(1, repeats, 1)[:, :seq_len, :]

        x = x + pos
        out = self.transformer_encoder(x)
        # Take last time step
        return self.fc(out[:, -1, :]).squeeze(1)
