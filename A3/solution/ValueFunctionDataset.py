import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class OcpDataset(Dataset):
    def __init__(self, data_file):
        """
        Args:
            csv_file (str): Path to the CSV file with 'x_init' and 'cost' columns.
        """
        # Read the CSV file using pandas
        data = pd.read_csv(data_file)
        
        # Ensure the required columns are present
        if not {'x_init', 'cost'}.issubset(data.columns):
            raise ValueError("CSV file must contain 'x_init' and 'cost' columns.")
        
        # Extract 'x_init' and 'cost' columns and convert to tensors
        self.x = torch.tensor(data['x_init'].values, dtype=torch.float32).unsqueeze(1)
        self.V = torch.tensor(data['cost'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.V[idx]


def get_data_loader(data_dir, batch_size, num_workers=0):
    train_data = OcpDataset(os.path.join(data_dir, "train.csv"))
    val_data = OcpDataset(os.path.join(data_dir, "validate.csv"))
    test_data = OcpDataset(os.path.join(data_dir, "test.csv"))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loader("dataset", 64, 0)
    dataset = OcpDataset("dataset/train.csv")
    item = dataset.__getitem__(0)
    print(item)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        print(inputs.view(-1, 1).shape)
        break