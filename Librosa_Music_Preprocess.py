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
codebook,_ = scipy.cluster.vq.kmeans(whiten, 16)

# %% Vector Quantization (VQ)
code, _ = scipy.cluster.vq.vq(whiten, codebook)
#Plot
plt.plot(code)

#%% Time conversion
time = np.arange(0,len(amplitude))/(60*sr) #Convert in minutes
plt.plot(time, amplitude, alpha=0.8)

#%% Split time
frame_100ms = int(sr/10)*20
code_final=[]

for i in range(0, len(amplitude)-frame_100ms, frame_100ms):
    print(str(i+frame_100ms)+'/'+str(len(amplitude)))
    i_ = i+frame_100ms
    frame_amplitude = amplitude[i:i_]
    frame_mfccs = librosa.feature.mfcc(y=frame_amplitude, sr=sr, n_mfcc=40)
    frame_whiten  = scipy.cluster.vq.whiten(frame_mfccs)
    frame_codebook,_ = scipy.cluster.vq.kmeans(frame_whiten, 16)
    frame_code, _ = scipy.cluster.vq.vq(frame_whiten, frame_codebook)
    code_final.append(frame_code)
    
test = np.concatenate(code_final)

#%% Plot
plt.figure(figsize=(10,4))
plt.plot(test)



