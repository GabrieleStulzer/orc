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

critic = Critic(3).to('cuda')

def save_training_params(params, filename='training_params.json'):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)

def train(num_epochs, lr, dataset_filename):
    dataset = OcpDataset(dataset_filename)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(critic.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, batch_V in dataloader:
            optimizer.zero_grad()
            V_pred = critic(batch_x.to('cuda'))
            loss = criterion(V_pred, batch_V.to('cuda'))
            loss.backward()
            optimizer.step()
        # if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

def save(save_name):
    print("Saving Critic Model")
    torch.save(critic.state_dict(), save_name + '/model.pt')

if __name__ == "__main__":
    # Read from shell parameters the number of epochs, the learning rate and the filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    parser = argparse.ArgumentParser(description='Train the Critic model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for the optimizer')
    parser.add_argument('--dataset', type=str, default='dataset_v2.csv', help='Filename of the dataset')
    parser.add_argument('--project', type=str, default='critic_training',help='Filename to save the trained model')
    parser.add_argument('--version', type=int, default='1',help='Version of the Critic model')
    args = parser.parse_args()

    critic = Critic(args.version).to('cuda')

    # Create a new project folder
    project = args.project
    project = f"critic_runs/{args.project}_{timestamp}"
    os.makedirs(project, exist_ok=True)

    # Save training parameters to a file
    training_params = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'dataset': args.dataset,
        'project': args.project,
        'version': args.version
    }

    # Get time counter
    start_time = datetime.datetime.now()
    train(args.epochs, args.lr, args.dataset)
    end_time = datetime.datetime.now()

    training_params['start_time'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
    training_params['end_time'] = end_time.strftime('%Y-%m-%d %H:%M:%S')
    training_params['duration'] = str(end_time - start_time)

    save_training_params(training_params, project + '/training_params.json')
    save(project)