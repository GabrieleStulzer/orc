import matplotlib.pyplot as plt
import torch
import argparse

import numpy as np
from multiprocessing import Pool

from OcpDefinition import OcpDefinition
from Parameters import Parameters
from Actor.Model import Actor

parser = argparse.ArgumentParser(description='Evalute Actor Performance')
parser.add_argument('--model', type=str, default="actor_runs/test/model.pt", help='Actor model file')

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

test_initial_states = generate_initial_states(SAMPLES, x_range)
params = [Parameters(i, N, dt, i, w_u, u_min, u_max) for i in test_initial_states]

with Pool(10) as p:
    sols = p.map(solveOneProblem, params)

actor = Actor().to('cuda')
actor.load_state_dict(torch.load(args.model, weights_only=True))
actor.eval()

# Use Actor to control the system and compute the resulting cost
def simulate_actor(x0):
    x = x0
    total_cost = 0
    for _ in range(N):
        u = actor(torch.tensor([x], dtype=torch.float32).to('cuda')).cpu().item()
        cost = 0.5 * u**2 + (x - 1.9)*(x - 1.0)*(x - 0.6)*(x + 0.5)*(x + 1.2)*(x + 2.1)
        total_cost += cost
        x = x + dt * u
    return total_cost

actor_costs = [simulate_actor(x0) for x0 in test_initial_states]
ocp_costs = [item[1] for item in sols]

# Plot comparison
plt.scatter(ocp_costs, actor_costs)
plt.xlabel('OCP Cost')
plt.ylabel('Actor Cost')
plt.title('Actor Performance')
plt.plot([min(ocp_costs), max(ocp_costs)], [min(ocp_costs), max(ocp_costs)], 'r--')
plt.show()