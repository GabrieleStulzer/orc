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

critic = Critic(4).to('cuda')

def save_training_params(params, filename='training_params.json'):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)

def train(num_epochs, lr, project, loss_fn, lr_adjust, dataset_filename):
    dataset = OcpDataset(dataset_filename)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    ## TODO: Change the loss function to Huber Loss
    criterion = loss_fn

    optimizer = optim.Adam(critic.parameters(), lr=lr)
    ## TODO: Implement rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.001, patience=10, verbose=True)

    losses = []
    patience = 10


    last_loss = 999999999

    ## TODO: Implement Gradient clipping
    ## IDEA: Consider weight decay

    for epoch in range(num_epochs):
        for batch_x, batch_V in dataloader:
            optimizer.zero_grad()
            V_pred = critic(batch_x.to('cuda'))
            loss = criterion(V_pred, batch_V.to('cuda'))
            loss.backward()
            optimizer.step()
        if lr_adjust:
            scheduler.step(loss)

        # Save the model with the best loss
        if loss.item() < last_loss:
            last_loss = loss.item()
            torch.save(critic.state_dict(), project + '/best_model.pt')
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        losses.append(loss.item())
    
    return losses

def save(save_name):
    print("Saving Critic Model")
    torch.save(critic.state_dict(), save_name + '/model.pt')

if __name__ == "__main__":
    # Read from shell parameters the number of epochs, the learning rate and the filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    parser = argparse.ArgumentParser(description='Train the Critic model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for the optimizer')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function for the optimizer')
    parser.add_argument('--lr_adjust', type=bool, default=False, help='Learning rate for the optimizer')
    parser.add_argument('--dataset', type=str, default='dataset_v2.csv', help='Filename of the dataset')
    parser.add_argument('--project', type=str, default='critic_training',help='Filename to save the trained model')
    parser.add_argument('--version', type=int, default='1',help='Version of the Critic model')
    args = parser.parse_args()

    critic = Critic(args.version).to('cuda')

    if args.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif args.loss == 'huber':
        loss_fn = nn.SmoothL1Loss()
    else:
        raise ValueError('Invalid loss function')

    # Create a new project folder
    project = args.project
    project = f"critic_runs/{args.project}_{timestamp}"
    os.makedirs(project, exist_ok=True)

    # Save training parameters to a file
    training_params = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'lr_adjust': args.lr_adjust,
        'loss_fn': args.loss,
        'dataset': args.dataset,
        'project': args.project,
        'version': args.version
    }

    # Get time counter
    start_time = datetime.datetime.now()
    losses = train(args.epochs, args.lr, project=project, loss_fn=loss_fn, lr_adjust=args.lr_adjust, dataset_filename=args.dataset)
    end_time = datetime.datetime.now()

    training_params['start_time'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
    training_params['end_time'] = end_time.strftime('%Y-%m-%d %H:%M:%S')
    training_params['duration'] = str(end_time - start_time)
    training_params['losses'] = losses

    save_training_params(training_params, project + '/training_params.json')
    save(project)