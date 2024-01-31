import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class CustomImageDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, header=None)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # return [torch.from_numpy(numpy.array(self.data.iloc[idx, 0])).to(dtype=torch.float64)], [torch.from_numpy(numpy.array(self.data.iloc[idx, 1])).to(dtype=torch.float64)]
        return np.array(self.data.iloc[idx, 0]), self.data.iloc[idx, 1]


def get_data_loader(data_dir, batch_size, num_workers=0):
    train_data = CustomImageDataset(os.path.join(data_dir, "train.csv"))
    val_data = CustomImageDataset(os.path.join(data_dir, "validate.csv"))
    test_data = CustomImageDataset(os.path.join(data_dir, "test.csv"))

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
    dataset = CustomImageDataset("dataset/train.csv")
    item = dataset.__getitem__(0)
    print(item)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        print(inputs.view(-1, 1).shape)
        break