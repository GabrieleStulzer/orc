import matplotlib.pyplot as plt
import torch
import argparse
import json

import numpy as np
from multiprocessing import Pool

from OcpDefinition import OcpDefinition
from Parameters import Parameters
from Critic.Model import Critic

parser = argparse.ArgumentParser(description='Evalute Crtic Performance')
parser.add_argument('--model', type=str, default="critic_runs/default/model.pt", help='Critic model file')
parser.add_argument('--folder', type=str, default="critic_runs/default/", help='Critic model data folder')

args = parser.parse_args()

N = 50          # horizon size
dt = 0.1        # time step
w_u = 1e-2
u_min = -1      # min control input
u_max = 1       # max control input
PLOT = False
SAMPLES = 1000
x_range = [-3, 3]

def solveOneProblem(params: Parameters):
    ocp = OcpDefinition(params.dt, params.w_u, params.u_min, params.u_max)
    sol = ocp.solve(params.x_init, params.N)
    return params.x_init, sol[0]

def generate_initial_states(num_samples, x_range):
    return np.random.uniform(x_range[0], x_range[1], num_samples)


# Read training_params.json from folder
with open(args.folder + 'training_params.json', 'r') as f:
    training_params = json.load(f)
    version = training_params['version']


test_initial_states = generate_initial_states(SAMPLES, x_range)
params = [Parameters(i, N, dt, i, w_u, u_min, u_max) for i in test_initial_states]

with Pool(10) as p:
    sols = p.map(solveOneProblem, params)

critic = Critic(version).to('cuda')
critic.load_state_dict(torch.load(args.folder + 'model.pt', weights_only=True))
critic.eval()

test_buffer = []

for x0 in test_initial_states:
    ocp = OcpDefinition(dt, w_u, u_min, u_max)
    V_opt, _ = ocp.solve(x0, N)

    if V_opt is not None:
        test_buffer.append((x0, V_opt))

# Predict V using Critic
with torch.no_grad():
    x_test = torch.tensor([item[0] for item in test_buffer], dtype=torch.float32).to('cuda').unsqueeze(1)
    V_pred = critic(x_test).cpu().squeeze().numpy()

V_true = np.array([item[1] for item in test_buffer])

# Plot Critic performance
plt.scatter(V_true, V_pred)
plt.xlabel('True V(x0)')
plt.ylabel('Predicted V(x0)')
plt.title('Critic Performance')
plt.plot([V_true.min(), V_true.max()], [V_true.min(), V_true.max()], 'r--')
plt.show()

# If training_params.json contains the key losses, plot the losses
if 'losses' in training_params:
    plt.plot(training_params['losses'])
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Critic Training Loss')
    plt.show()