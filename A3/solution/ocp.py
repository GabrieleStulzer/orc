import casadi as ca
import numpy as np

print("Defining OCP Problem")

# Time step
delta_t = 0.1

# Define symbolic variables
x = ca.SX.sym('x')
u = ca.SX.sym('u')

# System dynamics for single integrator
f = x + delta_t * u

# Running cost
l = 0.5 * u**2 + (x - 1.9)*(x - 1.0)*(x - 0.6)*(x + 0.5)*(x + 1.2)*(x + 2.1)

# Define CasADi functions
f_dyn = ca.Function('f_dyn', [x, u], [f])
running_cost = ca.Function('running_cost', [x, u], [l])

def solve_ocp(x0, N=50):
    # N: Horizon length
    opti = ca.Opti()

    # Decision variables
    X = opti.variable(N+1)
    U = opti.variable(N)

    # Initial condition
    opti.set_initial(X[0], x0)
    opti.subject_to(X[0] == x0)

    # Dynamics constraints
    for k in range(N):
        opti.subject_to(X[k+1] == f_dyn(X[k], U[k]))

    # Define the cost
    cost = 0
    for k in range(N):
        cost += running_cost(X[k], U[k])

    opti.minimize(cost)

    # Set solver options
    opts = {"print_time": False, "ipopt": {"print_level": 0}}
    opti.solver('ipopt', opts)

    # Solve the OCP
    try:
        sol = opti.solve()
        total_cost = sol.value(cost)
        optimal_U = sol.value(U)
        return total_cost, optimal_U
    except:
        # Handle infeasibility or solver failure
        return None, None

print("Solving multpile problems")

import random

def generate_initial_states(num_samples, x_range):
    return np.random.uniform(x_range[0], x_range[1], num_samples)

# Example parameters
num_samples = 100000
x_range = [-3, 3]

buffer = []

initial_states = generate_initial_states(num_samples, x_range)

for x0 in initial_states:
    V, _ = solve_ocp(x0)
    if V is not None:
        buffer.append((x0, V))

print("Training Critic")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Create a PyTorch Dataset
class OCPDataset(Dataset):
    def __init__(self, buffer):
        self.x = torch.tensor([item[0] for item in buffer], dtype=torch.float32).unsqueeze(1)
        self.V = torch.tensor([item[1] for item in buffer], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.V[idx]

dataset = OCPDataset(buffer)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the Critic network
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

critic = Critic()
criterion = nn.MSELoss()
optimizer = optim.Adam(critic.parameters(), lr=1e-3)

# Training loop for Critic
num_epochs = 1000

for epoch in range(num_epochs):
    for batch_x, batch_V in dataloader:
        optimizer.zero_grad()
        V_pred = critic(batch_x)
        loss = criterion(V_pred, batch_V)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

print("Saving critic")
torch.save(critic.state_dict(), "critic_gpt_1000")
    
print("Training Actor")

# Define the Actor network
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

actor = Actor()
optimizer_actor = optim.Adam(actor.parameters(), lr=1e-3)

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

# Training loop for Actor
for epoch in range(num_epochs):
    for batch_x, _ in dataloader:
        optimizer_actor.zero_grad()
        loss = actor_loss(batch_x)
        loss.backward()
        optimizer_actor.step()
    if (epoch+1) % 10 == 0:
        print(f'Actor Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

print("Saving actor")
torch.save(actor.state_dict(), "actor_gpt_1000")

import matplotlib.pyplot as plt

print("Evaluation")

# Evaluate on a new set of initial states
test_initial_states = generate_initial_states(100, x_range)
test_buffer = []

for x0 in test_initial_states:
    V_opt, _ = solve_ocp(x0)
    if V_opt is not None:
        test_buffer.append((x0, V_opt))

# Predict V using Critic
critic.eval()
with torch.no_grad():
    x_test = torch.tensor([item[0] for item in test_buffer], dtype=torch.float32).unsqueeze(1)
    V_pred = critic(x_test).squeeze().numpy()

V_true = np.array([item[1] for item in test_buffer])

# Plot Critic performance
plt.scatter(V_true, V_pred)
plt.xlabel('True V(x0)')
plt.ylabel('Predicted V(x0)')
plt.title('Critic Performance')
plt.plot([V_true.min(), V_true.max()], [V_true.min(), V_true.max()], 'r--')
plt.show()

# Use Actor to control the system and compute the resulting cost
def simulate_actor(x0, N=50):
    x = x0
    total_cost = 0
    for _ in range(N):
        u = actor(torch.tensor([x], dtype=torch.float32)).item()
        cost = 0.5 * u**2 + (x - 1.9)*(x - 1.0)*(x - 0.6)*(x + 0.5)*(x + 1.2)*(x + 2.1)
        total_cost += cost
        x = x + delta_t * u
    return total_cost

actor_costs = [simulate_actor(x0) for x0 in test_initial_states]
ocp_costs = [item[1] for item in test_buffer]

# Plot comparison
plt.scatter(ocp_costs, actor_costs)
plt.xlabel('OCP Cost')
plt.ylabel('Actor Cost')
plt.title('Actor Performance')
plt.plot([ocp_costs.min(), ocp_costs.max()], [ocp_costs.min(), ocp_costs.max()], 'r--')
plt.show()

