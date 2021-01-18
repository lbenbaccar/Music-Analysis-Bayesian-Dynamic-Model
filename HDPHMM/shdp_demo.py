import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import animation
from matplotlib.colors import PowerNorm  

from shdp import StickyHDPHMM

 
np.random.seed(100)
H = 3
L = 30

data = np.loadtxt("simulated_data.txt")
T = data.shape[0]


logdata = np.log10(data)
hdp = StickyHDPHMM(logdata, L=L, kappa=0)
#plot simulated date
#%%
plt.figure(figsize=(20,4))
plt.plot(np.ravel(data))
#%%
#plot sticky hmdmdmdm
estimates_shdp = np.array([10 ** hdp.getPath(h) for h in range(H)]).T
plt.plot(np.arange(T*3), np.ravel(estimates_shdp))


#plot transition matrix
matrix_transition= hdp.PI
plt.matshow(matrix_transition, norm=PowerNorm(0.2, 0, 1))#, vmin=0, vmax=0.1, aspect='auto')



#%%    
def update(t):
    hdp.sampler()     
    return hdp.PI.copy()

for t in range(1000): 
    print(t)
    matrix_transition = update(t)


#%%
update(t)
estimates_shdp = np.array([10**hdp.getPath(h) for h in range(H)]).T
plt.figure(figsize=(20,4))
plt.plot(np.ravel(estimates_shdp))

