import numpy as np
import matplotlib.pyplot as plt
from casadi import SX, vertcat, Function, Opti
import casadi as ca

T = 5.0
# w_u = w_u
# u_min = u_min
# u_max = u_max
x = ca.SX.sym('x')
u = ca.SX.sym('u')
    
def dynamics(x, u):
    f = x + dt*u
    return ca.Function('f_dyn', [x, u], [f])
    
def running_cost(x, u):
    l = 0.5*u**2 + (x - 1.9)*(x - 1.0)*(x - 0.6)*(x + 0.5)*(x + 1.2)*(x + 2.1)
    return ca.Function('running_cost', [x, u], [l])
    
x0 = 0.0
N = 50
dt = T/N
opti = ca.Opti()

time = np.linspace(0, T, N + 1)

X = opti.variable(N+1)
U = opti.variable(N)

opti.set_initial(X[0], x0)
opti.subject_to(X[0] == x0)

f_dyn = dynamics(x, u)
running_cost = running_cost(x, u)

for k in range(N):
    opti.subject_to(X[k+1] == f_dyn(X[k], U[k]))

cost = 0
for k in range(N):
    cost += running_cost(X[k], U[k])
    #TODO: Add cost on controls

opti.minimize(cost)

#TODO: Add constraints on control bounds


# s_opts = {"max_iter": 100}
opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
opti.solver("ipopt", opts) #, s_opts)

# Solve the OCP
try:
    sol = opti.solve()
    total_cost = sol.value(cost)
    optimal_U = sol.value(U)
except:
    # Handle infeasibility or solver failure
    print("Infeasible")

# Extract solution
x_sol = sol.value(X).flatten()
u_sol = sol.value(U).flatten()

# Generate cost values for the solution
cost_values = np.zeros(len(x_sol) - 1)
for i in range(len(cost_values)):
    cost_values[i] = running_cost(x_sol[i], u_sol[i])

# Create a combined plot
plt.figure(figsize=(12, 6))

# Plot state trajectory
plt.plot(time, x_sol, label='State $x(t)$', color='blue', linewidth=2)

# Plot cost values normalized for visualization
normalized_cost = (cost_values - np.min(cost_values)) / (np.max(cost_values) - np.min(cost_values))
plt.plot(time[:-1], normalized_cost, label='Normalized Cost', linestyle='--', color='orange', linewidth=2)

plt.title('State Trajectory and Normalized Cost Function', fontsize=14)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('State / Cost', fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Plot the results
plt.figure(figsize=(10, 6))

# State plot
plt.subplot(2, 1, 1)
plt.plot(time, x_sol, label='State $x(t)$')
plt.title('Optimal State and Control Trajectories')
plt.xlabel('Time [s]')
plt.ylabel('State')
plt.grid(True)
plt.legend()

# Control plot
plt.subplot(2, 1, 2)
plt.step(time[:-1], u_sol, where='post', label='Control $u(t)$')
plt.xlabel('Time [s]')
plt.ylabel('Control Input')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()