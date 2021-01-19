#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 00:19:50 2021

@author: erwanrahis
"""
import numpy as np
from numpy.random import beta


def _logphi(x, mu, sigma):
    """
    Compute log-likelihood of the base measure
    """
    return -np.power((x - mu) / sigma,2) / 2 - np.log(sigma)
    
def stick_breaking(param_gamma):
    """
    Generate the stick-breaking process with parameter param_gamma.
    """
    beta_save = 1
    while True:
        beta_k = beta(1, param_gamma) * beta_save
        beta_save -= beta_k
        yield beta_k
 