import argparse
import datetime
import os
import json

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from ValueFunctionDataset import OcpDataset
from Critic.Model import Critic
from Actor.Model import Actor

actor = Actor().to('cuda')
critic = Critic().to('cuda')

delta_t = 0.1

def save_training_params(params, filename='training_params.json'):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)

# Define the loss function for Actor
def actor_loss(x):
    u = actor(x)
    # Compute l(x, u)
    l_val = 0.5 * u**2 + (x - 1.9)*(x - 1.0)*(x - 0.6)*(x + 0.5)*(x + 1.2)*(x + 2.1)
    # Compute f(x, u)
    f_val = x + delta_t * u
    # Predict V(f(x, u)) using Critic
    V_val = critic(f_val)
    # Action-value
    Q = l_val + V_val
    return Q.mean()

def train(num_epochs, lr, dataset_filename):
    dataset = OcpDataset(dataset_filename)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(actor.parameters(), lr=lr)

    # Training loop for Actor
    for epoch in range(num_epochs):
        for batch_x, _ in dataloader:
            optimizer.zero_grad()
            loss = actor_loss(batch_x.to('cuda'))
            loss.backward()
            optimizer.step()
        print(f'Actor Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

def save(save_name):
    print("Saving Critic Model")
    torch.save(actor.state_dict(), save_name + '/model.pt')

if __name__ == "__main__":
    # Read from shell parameters the number of epochs, the learning rate and the filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    parser = argparse.ArgumentParser(description='Train the Critic model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for the optimizer')
    parser.add_argument('--dataset', type=str, default='dataset_v2.csv', help='Filename of the dataset')
    parser.add_argument('--project', type=str, default='actor_training',help='Filename to save the trained model')
    parser.add_argument('--dt', type=float, default='0.1',help='Time step')
    args = parser.parse_args()

    # Create a new project folder
    project = args.project
    project = f"actor_runs/{args.project}_{timestamp}"
    os.makedirs(project, exist_ok=True)

    delta_t = args.dt

    # Save training parameters to a file
    training_params = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'dataset': args.dataset,
        'project': args.project
    }
    save_training_params(training_params, project + '/training_params.json')

    train(args.epochs, args.lr, args.dataset)
    save(args.project)