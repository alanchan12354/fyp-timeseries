import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, hidden=64, layers=2, input_size=8):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(1)
