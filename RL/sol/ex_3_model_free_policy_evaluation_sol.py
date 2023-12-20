#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:56:07 2021

@author: adelprete
"""
from random import randrange
import numpy as np

def mc_policy_eval(env, gamma, pi, nEpisodes, maxEpisodeLength, 
                   V_real, plot=False, nprint=1000):
    ''' Monte-Carlo Policy Evaluation:
        env: environment 
        gamma: discount factor
        pi: policy to evaluate
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        V_real: real Value table
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    # create a vector N to store the number of times each state has been visited
    N = np.zeros(env.nx)
    # create a vector C to store the cumulative cost associated to each state
    C = np.zeros(env.nx)
    # create a vector V to store the Value
    # create a list V_err to store history of the error between real and estimated V table
    
    # for each episode
    # reset the environment to a random state
    # keep track of the states visited in this episode
    # keep track of the costs received at each state in this episode
    # simulate the system using the policy pi   
    # Update the V-Table by computing the cost-to-go J backward in time        
    # compute V_err as: mean(abs(V-V_real))
    V = np.zeros(env.nx);
    V_err = []


    for e in range(nEpisodes):
        env.reset() # Selects random state

        X = []
        costs = []

        for i in range(maxEpisodeLength):
            X.append(env.x)
            u = pi(env, X[i])
            x_next, c = env.step(u)
            costs.append(c)
        
        J = 0
        for i in range(len(X)-1, -1, -1):
            J = costs[i] + gamma*J
            x = X[i]
            N[x] += 1
            C[x] += J
            V[x] = C[x] / N[x]
        
        V_err.append(np.mean(np.abs(V - V_real)))

        if e%nprint == 0:
            print("Iter", e, "V_err", V_err[e])
            if plot:
                env.plot_V_table(V)

    
    return V, V_err


def td0_policy_eval(env, gamma, pi, V0, nEpisodes, maxEpisodeLength, 
                    V_real, learningRate, plot=False, nprint=1000):
    ''' TD(0) Policy Evaluation:
        env: environment 
        gamma: discount factor
        pi: policy to evaluate
        V0: initial guess for V table
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        V_real: real Value table
        learningRate: learning rate of the algorithm
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    
    # make a copy of V0 using np.copy(V0)
    # create a list V__err to store the history of the error between real and estimated V table
    # for each episode
    # reset environment to random initial state
    # simulate the system using the policy pi
    # at each simulation step update the Value of the current state         
    # compute V_err as: mean(abs(V-V_real))
    V = np.copy(V0)
    V_err = []

    for e in range(nEpisodes):
        env.reset()

        for i in range(maxEpisodeLength):
            x = env.x
            u = pi(env, x)
            x_next, c = env.step(u)

            V[x] += learningRate*(c + gamma*V[x_next] - V[x])

        V_err.append(np.mean(np.abs(V - V_real)))

        if e%nprint == 0:
            print("Iter", e, "V_err", V_err[e])
            if plot:
                env.plot_V_table(V)
    
    return V, V_err