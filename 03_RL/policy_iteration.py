'''
Example of policy iteration with a simple discretized 1-DoF pendulum.
'''

import numpy as np
from dpendulum import DPendulum, plot_policy, plot_V_table
from policy_evaluation import policy_eval, render_policy
import signal
import time

### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" % RANDOM_SEED)
np.random.seed(RANDOM_SEED)

### --- Hyper paramaters
MAX_EVAL_ITERS    = 200     # Max number of iterations for policy evaluation
MAX_IMPR_ITERS    = 20      # Max number of iterations for policy improvement
VALUE_THR         = 1e-3    # convergence threshold for policy evaluation
POLICY_THR        = 1e-4    # convergence threshold for policy improvement
NPRINT            = 1       # print some info every NPRINT iterations
DISCOUNT          = 0.9     # Discount factor 

nq=51   # number of discretization steps for the joint angle q
nv=21   # number of discretization steps for the joint velocity v
nu=11   # number of discretization steps for the joint torque u

### --- Environment
env = DPendulum(nq, nv, nu)
V  = np.zeros(env.nx)                       # Value table initialized to 0
pi = env.c2du(0.0)*np.ones(env.nx, np.int)  # policy table initialized to zero torque

def policy(env, x):
    return pi[x]
    
Q  = np.zeros(env.nu)           # temporary array to store value of different controls
for k in range(MAX_IMPR_ITERS):
    # evaluate current policy using policy evaluation
    V = policy_eval(env, DISCOUNT, policy, V, MAX_EVAL_ITERS, VALUE_THR, False)
    if not k%NPRINT: 
        print('PI - Iter #%d done' % (k))
        plot_policy(env, pi)
        plot_V_table(env, V)
    
    pi_old = np.copy(pi) # make a copy of current policy table
    for x in range(env.nx):     # for every state
        for u in range(env.nu): # for every control
            env.reset(x)                        # set the environment state to x
            x_next,cost = env.step(u)           # apply the control u
            Q[u] = cost + DISCOUNT * V[x_next]  # store value associated to u
            
        # Rather than simply using argmin we do something slightly more complex
        # to ensure simmetry of the policy when multiply control inputs
        # result in the same value. In these cases we prefer the more extreme
        # actions
#        pi[x] = np.argmin(Q)
        u_best = np.where(Q==np.min(Q))[0]
        if(u_best[0]>env.c2du(0.0)):    # if all the best action corresponds to a positive torque
            pi[x] = u_best[-1]          # take the largest torque
        elif(u_best[-1]<env.c2du(0.0)): # if all the best action corresponds to a negative torque
            pi[x] = u_best[0]           # take the smallest torque (largest in abs value)
        else:                           # otherwise take the average value among the best actions
            pi[x] = u_best[int(u_best.shape[0]/2)]
            
    # check for convergence
    pi_err = np.max(np.abs(pi-pi_old))
    if(pi_err<POLICY_THR):
        print("PI converged after %d iters with error"%k, pi_err)
        print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V)) 
        # 4.699 -9.99999 -3.13810
        plot_policy(env, pi)
        plot_V_table(env, V)
        break
        
    if not k%NPRINT: 
        print('PI - Iter #%d done' % (k))
        print("|pi - pi_old|=%.5f"%(pi_err))
        
render_policy(env, policy, env.x2i(env.c2d([np.pi,0.])))