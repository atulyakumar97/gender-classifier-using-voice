#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import re
import scipy.stats as stats
from scipy.io import wavfile
import numpy as np
import os

raw_folder= "C:\\Users\\Atulya\\Documents\\GitHub\\gender-classifier-using-voice\\Female\\"

def get_frequencies(rate,data):
  
  #extract list of dominant frequencies in sliding windows of duration defined by 'step' for each of the 10 wav files and return an array
  #get dominating frequencies in sliding windows of 200ms
  step = int(rate/5) #3200 sampling points every 1/5 sec 
  window_frequencies = []
  frequencies_lol=[]

  for i in list(range(0,len(data),step)):
      ft = np.fft.fft(data[i:i+step]) #fft returns the list N complex numbers
      freqs = np.fft.fftfreq(len(ft)) #fftq tells you the frequencies associated with the coefficients
      imax = np.argmax(np.abs(ft))
      freq = freqs[imax]
      freq_in_hz = abs(freq *rate)
      window_frequencies.append(freq_in_hz)
      #filtered_frequencies = [f for f in window_frequencies if 20<f<280 and not 46<f<66] # I see noise at 50Hz and 60Hz
      filtered_frequencies = [f for f in window_frequencies if not 46<f<66] # I see noise at 50Hz and 60Hz
      frequencies_lol.append(filtered_frequencies)
      #frequencies_lol.append(window_frequencies)
      #frequencies_lol.append(freq_in_hz)

  frequencies = [item for sublist in frequencies_lol for item in sublist]
  return(frequencies)
  #return(frequencies_lol)

def get_features(frequencies):

  #print("\nExtracting features ")
  nobs, minmax, mean, variance, skew, kurtosis =  stats.describe(frequencies)
  median   = np.median(frequencies)
  mode     = stats.mode(frequencies).mode[0]
  std      = np.std(frequencies)
  low,peak = minmax
  q75,q25  = np.percentile(frequencies, [75 ,25])
  iqr      = q75 - q25
  return(nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr)

def main():
  directory=os.listdir(raw_folder)
  for wav_file in directory:
    rate, data = wavfile.read(raw_folder+wav_file)
    print(wav_file)
    frequencies=get_frequencies(rate,data)
    features=get_features(frequencies)
    print(features)
    


rate, data = wavfile.read(raw_folder+'00258.wav')
frequencies=get_frequencies(rate,data)
features=get_features(frequencies)
print(features)
