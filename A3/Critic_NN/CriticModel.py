import torch.nn as nn
import torch.nn.functional as F

# Define Neural Network model
class ValueFunctionModel(nn.Module):
    def __init__(self):
        super(ValueFunctionModel, self).__init__()
        self.input = nn.Linear(1, 16)
        self.h1 = nn.Linear(16, 32)
        self.h2 = nn.Linear(32, 64)
        self.h3 = nn.Linear(64, 64)
        self.h4 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        x = F.relu(self.h4(x))
        x = self.output(x)
        return x
    