#!/usr/bin/env python3
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
import matplotlib as plt


# %% Visualization

path_temp =  '' #Mettre le mP3 ici

amplitude, sr = librosa.load(path_temp)
print(amplitude.shape)

librosa.display.waveplot(amplitude, song, alpha=0.8)
song_melspecto = librosa.feature.melspectrogram(amplitude, s)
sbd = librosa.power_to_db(song_melspecto)

# %% Decibel graphs
librosa.display.specshow(sbd, sr=song, hop_length=500, x_axis='time', y_axis='log')


# Utilsier la fonction mfcc pour extraire les paramètres now 
mfcc(sr=22050, S=sbd, n_mfcc=40)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
