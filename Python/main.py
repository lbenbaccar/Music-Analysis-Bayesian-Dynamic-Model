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
L = 5  #nb of states at the beginning
alpha = 1 #parameter of the DP
gamma_ = 1 #parameter of the DP

#Sticky parameter
kappa = 1 #* data.size

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
stickbreaking = stick_breaking(gamma_)
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

#plt.matshow(PI, norm=PowerNorm(0.2, 0, 1), vmin=0, vmax=0.1, aspect='auto')
    
#%% Sampling function
       
def sampler(PI, state, betas, N, mu, sigma, M):
    for subsequence in range(n):
        # Step 1: messages
        #Init messages to 1
        messages = np.zeros((T, L))
        messages[-1, :] = 1
        #With a backward loop, compute the messages for each k
        for t in range(T - 1, 0, -1):
            messages[t-1, :] = PI.dot(messages[t, :] * np.exp(_logphi(data[t, subsequence], mu, sigma)))
            messages[t-1, :] /= np.max(messages[t-1, :])
        # Step 2: states by MH algorithm
        for t in range(1, T):
            j = choice(L) # proposal
            k = state[t, subsequence] 

            logprob_accept = (np.log(messages[t, k]) -
                              np.log(messages[t, j]) +
                              np.log(PI[state[t-1, subsequence], k]) -
                              np.log(PI[state[t-1, subsequence], j]) +
                              _logphi(data[t-1, subsequence], 
                                           mu[k], 
                                           sigma[k]) -
                              _logphi(data[t-1, subsequence], 
                                           mu[j], 
                                           sigma[j]))
            if exponential(1) > logprob_accept:
                state[t, subsequence] = j
                
                N[state[t-1, subsequence], j] += 1
                N[state[t-1, subsequence], k] -= 1 
                if t != (T-1):
                    N[k, state[t+1, subsequence]] -= 1 
                    N[j, state[t+1, subsequence]] += 1 

            
    # Step 3: auxiliary variables
    #P contains the parameters for the Bernoulli distribution 
    P = np.tile(betas, (L, 1))*alpha + n
    #Adding the Kappa in diagonal
    np.fill_diagonal(P, np.diag(P) + kappa)
    P = 1 - n / P
    
    #Computes the M binomial distribution for each state transition
    for i in range(L):
        for j in range(L):
            M[i, j] = binomial(N[i, j], P[i, j])
            
    #Computes the override random variables
    w = np.array([binomial(M[i, i], kappa / (kappa + alpha*betas[i])) for i in range(L)])
    
    #Computes the number of considered states
    m_bar = np.sum(M, axis=0) - w
    
    # Step 4: beta and parameters of clusters
    betas = dirichlet(np.ones(L) * (gamma_ / L) + m_bar)
    
    # Step 5: transition matrix
    PI =  np.tile(alpha * betas, (L, 1)) + N
    np.fill_diagonal(PI, np.diag(PI) + kappa)


    for i in range(L):
       #Derive PI from a Dirichlet 
        PI[i, :] = dirichlet(PI[i, :])
        
        #Find the clusters and update the parameters
        state_index = np.where(state == i)
        cluster = data[state_index]
        cardinal = cluster.size
        if cardinal:
            meancluster= np.mean(cluster)
            mu[i] = meancluster / (nu/ cardinal + 1)
            sigma[i] = (2 * b + (cardinal - 1) * np.var(cluster) + 
                             cardinal * meancluster ** 2 / (nu + cardinal)) / (2 * a + cardinal - 1)
        else:
            mu[i] = normal(0, np.sqrt(nu))
            sigma[i] = 1 / gamma(a, b)
            
    return PI, state, betas, N, mu, sigma, M
            


# %% Run Sampling
max_iter = 300
for z in range(max_iter):
    print(str(z)+'/'+str(max_iter))
    PI, state, betas, N, mu, sigma, M = sampler(PI, state, betas, N, mu, sigma, M)
    
plt.matshow(PI,norm=PowerNorm(0.2, 0, 1), vmin=0, vmax=0.1, aspect='auto')

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
plt.plot(np.ravel(pathss))
#plt.yscale('log')