import torch
from torch.utils.data import Dataset

# Create a PyTorch Dataset
class OCPDataset(Dataset):
    def __init__(self, buffer):
        self.x = torch.tensor([item[0] for item in buffer], dtype=torch.float32).unsqueeze(1)
        self.V = torch.tensor([item[1] for item in buffer], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.V[idx]