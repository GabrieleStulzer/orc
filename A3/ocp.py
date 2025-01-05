import numpy as np
import csv
import casadi
from multiprocessing import Pool

class OcpSingleIntegrator:

    def __init__(self, dt, w_u, u_min=None, u_max=None):
        self.dt = dt
        self.w_u = w_u
        self.u_min = u_min
        self.u_max = u_max

    def solve(self, x_init, N, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()
        self.x = self.opti.variable(N+1)
        self.u = self.opti.variable(N)
        x = self.x
        u = self.u

        if(X_guess is not None):
            for i in range(N+1):
                self.opti.set_initial(x[i], X_guess[i,:])
        else:
            for i in range(N+1):
                self.opti.set_initial(x[i], x_init)
        if(U_guess is not None):
            for i in range(N):
                self.opti.set_initial(u[i], U_guess[i,:])

        self.cost = 0
        self.running_costs = [None,]*(N+1)
        for i in range(N+1):
            self.running_costs[i] = (x[i]-1.9)*(x[i]-1.0)*(x[i]-0.6)*(x[i]+0.5)*(x[i]+1.2)*(x[i]+2.1)
            if(i<N):
                self.running_costs[i] += self.w_u * u[i]*u[i]
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        for i in range(N):
            self.opti.subject_to( x[i+1]==x[i] + self.dt*u[i] )
        if(self.u_min is not None and self.u_max is not None):
            for i in range(N):
                self.opti.subject_to( self.opti.bounded(self.u_min, u[i], self.u_max) )
        self.opti.subject_to(x[0]==x_init)

        # s_opts = {"max_iter": 100}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts) #, s_opts)

        return self.opti.solve()

class Parameters:
    def __init__(self, index, N, dt, x_init, w_u, u_min, u_max):
        self.index = index
        self.N = N
        self.dt = dt
        self.w_u = w_u
        self.u_min = u_min
        self.u_max = u_max
        self.x_init = x_init
    
    def setXInit(self, x_init):
        self.x_init = x_init

    def __str__(self) -> str:
        return f"Index = {self.index} - Parameters(N={self.N}, dt={self.dt}, x_init={self.x_init}, w_u={self.w_u}, u_min={self.u_min}, u_max={self.u_max})"

    
def solveOneProblem(parameters: Parameters):
    # print(f"Solving problem with parameters: {parameters}")
    ocp = OcpSingleIntegrator(parameters.dt, parameters.w_u, parameters.u_min, parameters.u_max)
    sol = ocp.solve(parameters.x_init, parameters.N)
    print(f"==== {parameters.index} solved\n")
    return (parameters.x_init, sol.value(ocp.cost))

if __name__=="__main__":
    import matplotlib.pyplot as plt
    N = 20          # horizon size
    dt = 0.01        # time step
    x_init = -1.0   # initial state
    w_u = 1e-2
    u_min = -1      # min control input
    u_max = 1       # max control input
    PLOT = False
    SAMPLES = 2

    params = [Parameters(i, N, dt, np.random.uniform(-2.2, 2.0), w_u, u_min, u_max) for i in range(SAMPLES)]

    with Pool(10) as p:
        sols = p.map(solveOneProblem, params)
    
    with open('data_2.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a header row
        writer.writerow(['x_init', 'cost'])

        for sol in sols:
            row = [sol[0], sol[1]]
            writer.writerow(row)
    
    

    # for i in range(10):
    #     x_init = np.random.uniform(-2.2, 2.0)
    #     print(f"Random x_init: {x_init}")


    #     ocp = OcpSingleIntegrator(dt, w_u, u_min, u_max)
    #     sol = ocp.solve(x_init, N)
    #     print("Optimal value of x:\n", sol.value(ocp.x))
    #     costs = [(sol.value(ocp.x[i]), sol.value(ocp.running_costs[i])) for i in range(N+1)]
    #     print(costs)

    #     x1, y1 = costs[0]
    #     x2, y2 = costs[-1]
    #     distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    #     print(f"Value function (Euclidean distance 2D): {distance}")

    #      # Get the final optimal value
    #     final_optimal_value = sol.value(ocp.x[N])

    #     # Calculate the Euclidean distance between initial guess and final optimal value
    #     value_function = np.linalg.norm(final_optimal_value - x_init)

    #     print(f"Value function (Euclidean distance): {value_function}")

    #     print(f"Ocp Cost: {sol.value(ocp.cost)}")


    #     if PLOT:
    #         X = np.linspace(-2.2, 2.0, 100)
    #         costs = [sol.value(ocp.running_costs[0], [ocp.x==x_val]) for x_val in X]
    #         plt.plot(X, costs)
    #         for i in range(N+1):
    #             plt.plot(sol.value(ocp.x[i]), sol.value(ocp.running_costs[i]), 
    #                     'xr', label='x_'+str(i))
    #         plt.legend()
    #         plt.show()
