import casadi as ca
import numpy as np
import Parameters

class OcpDefinition:
    def __init__(self, dt, w_u, u_min=None, u_max=None):
        self.dt = dt
        self.w_u = w_u
        self.u_min = u_min
        self.u_max = u_max
        self.x = ca.SX.sym('x')
        self.u = ca.SX.sym('u')
    
    def dynamics(self, x, u):
        f = x + self.dt*u
        return ca.Function('f_dyn', [x, u], [f])
    
    def running_cost(self, x, u):
        l = 0.5*u**2 + (x - 1.9)*(x - 1.0)*(x - 0.6)*(x + 0.5)*(x + 1.2)*(x + 2.1)
        return ca.Function('running_cost', [x, u], [l])
    
    def solve(self, x0, N=50):
        opti = ca.Opti()

        X = opti.variable(N+1)
        U = opti.variable(N)

        opti.set_initial(X[0], x0)
        opti.subject_to(X[0] == x0)

        f_dyn = self.dynamics(self.x, self.u)
        running_cost = self.running_cost(self.x, self.u)

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
            return total_cost, optimal_U
        except:
            # Handle infeasibility or solver failure
            return None, None
