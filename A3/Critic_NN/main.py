import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


from trainFunctions import train
from solution.Critic.ValueFunctionDataset import get_data_loader
from CriticModel import ValueFunctionModel

DATA_DIR="dataset/2000000Samples"
BATCH_SIZE=16
EPOCHS=100
DEVICE="cuda"

# Get Dataloaders
train_loader, val_loader, test_loader = get_data_loader(DATA_DIR, BATCH_SIZE, 0)

# Define Loos Function
loss_fn = torch.nn.MSELoss()

model = ValueFunctionModel().to(DEVICE, dtype=torch.float32)

# print(model.input.weight.dytpe)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

train(train_loader, val_loader, model, loss_fn, optimizer, EPOCHS, DEVICE)
