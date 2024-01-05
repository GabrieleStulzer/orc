import numpy as np

def single_pendulum(x, u):
    dt = 0.01  # time step
    l = 1  # length of the pendulum
    m = 1  # mass of the pendulum
    g = 9.81  # gravity

    q, dq = x[0], x[1]
    u1 = u[0]

    ddq = (-g / l * np.sin(q) + u1 / (m * l ** 2))  # Equation of motion for a single pendulum

    x_next = x + dt * np.array([dq, ddq])
    return x_next
