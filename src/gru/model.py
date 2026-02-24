import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        self.gru = nn.GRU(1, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(1)
