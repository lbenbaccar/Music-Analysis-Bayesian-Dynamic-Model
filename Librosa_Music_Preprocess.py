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

amplitude, song = librosa.load(path_temp)  

librosa.display.waveplot(amplitude, song, alpha=0.8)
song_melspecto = librosa.feature.melspectrogram(amplitude, sr=song, power=2.0)
sbd = librosa.power_to_db(song_melspecto)

# %% Decibel graphs
librosa.display.specshow(sbd, sr=song, hop_length=500, x_axis='time', y_axis='log')


# Utilsier la fonction mfcc pour extraire les paramètres now 
