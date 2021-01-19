#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 00:11:38 2021

@author: erwanrahis
"""
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import animation
from matplotlib.colors import PowerNorm  
import numpy as np
from numpy.random import choice, normal, dirichlet, beta, gamma, multinomial, exponential, binomial

from functions import stick_breaking, _logphi

#ADD THE DATA
data = np.loadtxt("simulated_data.txt")

#%% Plot of the data
plt.figure(figsize=(15,6))
plt.plot(np.ravel(data))
plt.title('Simulated data')
#%% PARAMETERS

#Parameters prior
L = 10  #nb of states at the beginning
alpha = 1 #parameter of the DP
gma = 1 #parameter of the DP

#Sticky parameter
kappa = 2 * data.size

# Hyperparameters
nu = 2
a = 2
b = 2


#%% Initialize
#Shape of the data
T, n = data.shape

#Random assignment of states
state = choice(L, data.shape)
#Distribution parameters of states assignment
std = np.std(data)
mu = normal(0, std, L)
sigma = np.ones(L) * std

for i in range(L):
    idx = np.where(state == i)
    if idx[0].size:
        cluster = data[idx]
        mu[i] = np.mean(cluster)
        sigma[i] = np.std(cluster)
        
#Stick breaking prior on beta
stickbreaking = stick_breaking(gma)
betas = np.array([next(stickbreaking) for i in range(L)])

#Matrix N with Njk number of transition from state j to k
N = np.zeros((L, L))
for t in range(1, T):
    for i in range(n):
        N[state[t-1, i], state[t, i]] += 1

#Matrix with the mjk number of observation in cluster considering state k
M = np.zeros(N.shape)
#Initialization of the probabilities of transition
PI = (N.T / (np.sum(N, axis=1) + 1e-7)).T

plt.matshow(PI)
    
#%% Sampling function
       
def sampler(PI, state, betas, N):
    for obs in range(n):
        # Step 1: messages
        #Init messages to 1
        messages = np.zeros((T, L))
        messages[-1, :] = 1
        #With a backward loop, compute the messages for each k
        for t in range(T - 1, 0, -1):
            messages[t-1, :] = PI.dot(messages[t, :] * np.exp(_logphi(data[t, obs], mu, sigma)))
            messages[t-1, :] /= np.max(messages[t-1, :])
        # Step 2: states by MH algorithm
        for t in range(1, T):
            j = choice(L) # proposal
            k = state[t, obs] 

            logprob_accept = (np.log(messages[t, k]) -
                              np.log(messages[t, j]) +
                              np.log(PI[state[t-1, obs], k]) -
                              np.log(PI[state[t-1, obs], j]) +
                              _logphi(data[t-1, obs], 
                                           mu[k], 
                                           sigma[k]) -
                              _logphi(data[t-1, obs], 
                                           mu[j], 
                                           sigma[j]))
            if exponential(1) > logprob_accept:
                #print('accept')
                state[t, obs] = j
                
                #N[state[t-1, obs], j] += 1
                #N[state[t-1, obs], k] -= 1  
            #Re compute the transition matrix once the
                N = np.zeros((L, L))
                for j in range(1, T):
                    for i in range(n):
                        N[state[j-1, i], state[j, i]] += 1
            
    # Step 3: auxiliary variables
    P = np.tile(betas, (L, 1)) + n
    np.fill_diagonal(P, np.diag(P) + kappa)
    P = 1 - n / P
    for i in range(L):
        for j in range(L):
            M[i, j] = binomial(M[i, j], P[i, j])

    w = np.array([binomial(M[i, i], 1 / (1 + betas[i])) for i in range(L)])
    m_bar = np.sum(M, axis=0) - w
    
    # Step 4: beta and parameters of clusters
    betas = dirichlet(np.ones(L) * (gma / L) + m_bar)
    
    #Change here
    #N[N<0]=0
    #print(N)
    # Step 5: transition matrix
    PI =  np.tile(alpha * betas, (L, 1)) + N


    for i in range(L):
        PI[i, :] = dirichlet(PI[i, :])
        idx = np.where(state == i)
        cluster = data[idx]
        nc = cluster.size
        if nc:
            xmean = np.mean(cluster)
            mu[i] = xmean / (nu/ nc + 1)
            sigma[i] = (2 * b + (nc - 1) * np.var(cluster) + 
                             nc * xmean ** 2 / (nu + nc)) / (2 * a + nc - 1)
        else:
            mu[i] = normal(0, np.sqrt(nu))
            sigma[i] = 1 / gamma(a, b)
    return PI, state, betas, N
            


#%% Run Sampling

for z in range(100):
    print(z)
    PI, state, betas, N = sampler(PI, state, betas, N)
    

plt.matshow(PI)

# %% PLOT THE FINAL STATES
plt.figure(figsize=(20,6))
plt.plot(np.ravel(state))

# %% ESTIMATED SAMPLE PATH
pathss = []
for h in range(n):
    paths = np.zeros(data.shape[0])
    for i, mu_ in enumerate(mu):
        paths[np.where(state[:, h] == i)] = mu_
    pathss.append(paths)
    
pathsplot = np.concatenate(pathss)
#%% PLOT
plt.figure(figsize=(20,6))
plt.plot(pahs)
