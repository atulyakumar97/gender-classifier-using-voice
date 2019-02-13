#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:50:28 2019

@author: bhargavdesai
"""
import numpy as np 
import librosa
from keras.models import load_model
import os
import soundfile as sf


test_path="C:\\Users\\Atulya\\OneDrive\\Desktop\\Test\\"
Model = load_model('gc2_(50-800)_trained-on-mac.h5')
Model.summary()
g=[]
for audiofile in os.listdir(test_path):
            try:
                y_in, sr = sf.read(os.path.join(test_path,audiofile))
                y_in = librosa.resample(y_in, sr, 8000)
                y_in = y_in[0:40000]
                y_in = np.concatenate((y_in, [0]* (40000 - y_in.shape[0])), axis=0)
                g.append(y_in)
            except RuntimeError:
                print(".DS_Store file detected and dismissed")
                pass
x_in = np.array(g)
x_in.shape
x_in = x_in.reshape(x_in.shape[0],50,800)
result = Model.predict(x_in)
m=0
for r in result:
    if r>0.2:
        m=m+1
print("Accuracy is")
print(m/len(result))