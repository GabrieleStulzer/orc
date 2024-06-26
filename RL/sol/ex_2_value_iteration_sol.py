#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 05:30:56 2021

@author: adelprete
"""
import numpy as np

def value_iteration(env, gamma, V, maxIters, value_thr, plot=False, nprint=1000):
    ''' Policy iteration algorithm 
        env: environment used for evaluating the policy
        gamma: discount factor
        V: initial guess of the Value table
        maxIters: max number of iterations of the algorithm
        value_thr: convergence threshold
        plot: if True it plots the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    # IMPLEMENT VALUE ITERATION HERE
    
    # Create an array to store the Q value of different controls
    # Iterate for at most maxIters loops
    # You can make a copy of current Value table using np.copy()
    # The number of states is env.nx
    # The number of controls is env.nu
    # You can set the state of the robot using env.reset(x)
    # You can simulate the robot using: x_next,cost = env.step(u)
    # You can get the minimum/maximum of an array using np.min() / np.max()
    # You can get the absolute values of the element of an array using np.abs()
    # Check convergence based on how much the Value has changed since the previous loop
    # At the end return the Value table

    Q = np.zeros(env.nu)

    for i in range(maxIters):
        V_old = np.copy(V)

        for x in range(env.nx):
            for u in range(env.nu):
                env.reset(x)
                n, c = env.step(u)
                Q[u] = c + gamma*V[n]

            V[x] = np.min(Q)
        
        err = np.max(np.abs(V - V_old))
        if(err < value_thr):
            print("Policy Iteration has converged in", i, "iters with err", err)
            break;

        if i % nprint == 0:
            print("Policy evaluation - Iter", i, "error", err)
            if plot:
                env.plot_V_table(V)
    return V