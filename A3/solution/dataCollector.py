import csv
import random
import casadi as ca
import numpy as np
from OcpDefinition import OcpDefinition
from Parameters import Parameters
from multiprocessing import Pool


def solveOneProblem(params: Parameters):
    ocp = OcpDefinition(params.dt, params.w_u, params.u_min, params.u_max)
    sol = ocp.solve(params.x_init, params.N)
    return params.x_init, sol[0]

if __name__=="__main__":
    import matplotlib.pyplot as plt
    N = 50          # horizon size
    dt = 0.1        # time step
    w_u = 1e-2
    u_min = -1      # min control input
    u_max = 1       # max control input
    PLOT = False
    SAMPLES = 1000000

    params = [Parameters(i, N, dt, np.random.uniform(-3.0, 3.0), w_u, u_min, u_max) for i in range(SAMPLES)]

    with Pool(15) as p:
        sols = p.map(solveOneProblem, params)
    
    with open('dataset_v2.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a header row
        writer.writerow(['x_init', 'cost'])

        for sol in sols:
            row = [sol[0], sol[1]]
            writer.writerow(row)