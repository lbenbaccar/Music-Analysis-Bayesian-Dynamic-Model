#!\usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 21:58:14 2021

@author: erwanrahis
"""

# %% Imports
import sys, os
import librosa
import librosa.display
from librosa.feature import mfcc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy

# %% Visualization

path_temp =  'Documents/Cours/MSDS/S1/Statistique Bayesienne/MusicAnalysis-Bayesian_DynamicModel.nosync/Data/beatles.mp3' #Mettre le mP3 ici

amplitude, sr = librosa.load(path_temp)
print(amplitude.shape)

# %% Waveplots
librosa.display.waveplot(amplitude, sr, alpha=0.8)
song_melspecto = librosa.feature.melspectrogram(amplitude, sr)
sbd = librosa.power_to_db(song_melspectro)

# %% Decibel graphs
librosa.display.specshow(sbd, sr=sr, hop_length=500, x_axis='time', y_axis='log')
# %%  Show melspectogram
librosa.display.specshow(song_melspecto, sr=sr, hop_length=500, x_axis='time', y_axis='log')

# %%
#mfccs = mfcc(sr=sr, y=amplitude, n_mfcc=40)
librosa.display.specshow(sbd, x_axis='time', y_axis='log')

#%% Test de la fonction mfcc
mfccs = librosa.feature.mfcc(y=amplitude, sr=sr, n_mfcc=40)

# %% Graph des mfcc
librosa.display.specshow(mfccs, x_axis='time', y_axis='log')


# %% Vector Quantization
whiten  = scipy.cluster.vq.whiten(mfccs)
codebook,_ = scipy.cluster.vq.kmeans(whiten, 16 )
# %% Plot des kmeans
plt.scatter(whiten[:,20], whiten[:,21])
plt.scatter(codebook[:,20], codebook[:,21])

#%% VQ
code, _ = scipy.cluster.vq.vq(whiten, codebook)

#%% VQ Plot
plt.plot(code)