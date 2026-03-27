import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, hidden=64, layers=2, input_size=8):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(1)
