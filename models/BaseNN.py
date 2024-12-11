import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class BaselineNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaselineNN, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.activation(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
