import os
import csv
import random
import argparse
import casadi as ca
import numpy as np
from OcpDefinition import OcpDefinition
from Parameters import Parameters
from multiprocessing import Pool


def solveOneProblem(params: Parameters):
    ocp = OcpDefinition(params.dt, params.w_u, params.u_min, params.u_max)
    sol = ocp.solve(params.x_init, params.N)
    return params.x_init, sol[0]

def generate_initial_states(num_samples, x_range):
    return np.random.uniform(x_range[0], x_range[1], num_samples)

def solveMultipleProblems(N, dt, w_u, u_min, u_max, x_range, samples):
    test_initial_states = generate_initial_states(samples, x_range)
    params = [Parameters(i, N, dt, i, w_u, u_min, u_max) for i in test_initial_states]

    with Pool(10) as p:
        sols = p.map(solveOneProblem, params)

    return sols

def saveDataset(sols, filename):
    with open (filename, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a header row
        writer.writerow(['x_init', 'cost'])

        for sol in sols:
            row = [sol[0], sol[1]]
            writer.writerow(row)

def saveParams(N, dt, w_u, u_min, u_max, x_range, samples, filename):
    with open (filename, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a header row
        writer.writerow(['N', 'dt', 'x_interval', 'w_u', 'u_min', 'u_max'])

        row = [N, dt, x_range, w_u, u_min, u_max]
        writer.writerow(row)

if __name__=="__main__":
    import matplotlib.pyplot as plt
    N = 50          # horizon size
    dt = 0.1        # time step
    w_u = 1e-2
    u_min = -1      # min control input
    u_max = 1       # max control input
    x_interval = [-3, 3]

    parser = argparse.ArgumentParser(description='Evalute Actor Performance')
    parser.add_argument('--dataset', type=str, default="default", help='Name of the dataset')

    args = parser.parse_args()

    folder = f"datasets/{args.dataset}"
    os.makedirs(folder, exist_ok=True)

    PLOT = False
    SAMPLES = 100

    sols = solveMultipleProblems(N, dt, w_u, u_min, u_max, x_interval, SAMPLES)
    saveDataset(sols, f"{folder}/dataset.csv")
    saveParams( N, dt, w_u, u_min, u_max, x_interval, SAMPLES, f"{folder}/params.csv")