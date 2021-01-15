#!\usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 21:58:14 2021

@author: erwanrahis
"""

# %% Libraries
import sys, os
import librosa
import librosa.display
from librosa.feature import mfcc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy

# %% Loading the song
#Path to the song
path_temp =  'Documents/Cours/MSDS/S1/Statistique Bayesienne/MusicAnalysis-Bayesian_DynamicModel.nosync/Data/beatles.mp3' #Mettre le mP3 ici
#Loading using librosa
amplitude, sr = librosa.load(path_temp)
#%%Plot of the amplitude
plt.figure(figsize=(10, 4))
librosa.display.waveplot(amplitude, sr, alpha=0.8)
plt.title('Waveform : amplitude ')


# %% Melspectrograme
song_melspecto = librosa.feature.melspectrogram(y=amplitude, sr=sr)
#Convert from power to decibels
sbd = librosa.power_to_db(song_melspecto, ref=np.max)
#Plot mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(sbd, sr=sr, hop_length=500, x_axis='time', y_axis='mel')
plt.title('Mel spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()


#%% Compute MFCCs
mfccs = librosa.feature.mfcc(y=amplitude, sr=sr, n_mfcc=6000)
#Plot MFCCs
plt.figure(figsize=(10,4))
librosa.display.specshow(mfccs, hop_length=500, x_axis='time')
plt.title('MFCCs')
plt.colorbar()

# %% Clustering (kmeans)
#Standardize the columns
whiten  = scipy.cluster.vq.whiten(mfccs)
#Creates 16 code books
codebook,_ = scipy.cluster.vq.kmeans(whiten, 2400)

# %% Vector Quantization (VQ)
code, _ = scipy.cluster.vq.vq(whiten, codebook)
#Plot
plt.plot(code)

#%% Time split
time = np.arange(0,len(amplitude))/(60*sr) #Convert in minutes

plt.plot(time, amplitude, alpha=0.8)
