import pandas as pd
import re
import scipy.stats as stats
from scipy.io import wavfile
import numpy as np
import os
import librosa

raw_folder1= "C:\\Users\\Atulya\\Documents\\GitHub\\gender-classifier-using-voice\\Female\\"
raw_folder2= "C:\\Users\\Atulya\\Documents\\GitHub\\gender-classifier-using-voice\\Male\\"

def get_frequencies(y,sr):
    '''Spectral centroid
    Each frame of a magnitude spectrogram is normalized and treated as a distribution over frequency bins,
    from which the mean (centroid) is extracted per frame.'''
    frequencies=librosa.feature.spectral_centroid(y=y, sr=sr)
    
    return frequencies
    
def get_features(frequencies):

    #print("\nExtracting features ")
    nobs, minmax, mean, variance, skew, kurtosis =  stats.describe(frequencies)
    mean     =  np.mean(frequencies)
    median   = np.median(frequencies)
    mode     = stats.mode(frequencies).mode[0][0]
    std      = np.std(frequencies)
    sortedfreq = np.sort(frequencies)
    minfreq = sortedfreq[0][0]
    peak = sortedfreq[0][-1]
    q75,q25  = np.percentile(frequencies, [75 ,25])
    iqr      = q75 - q25
    print('\nmean = ', mean,'\nmedian = ', median,'\nmode = ', mode,'\nstd = ', std,'\nminfreq = ', minfreq,'\npeak = ', peak,'\nq25 = ', q25,'\nq75 = ', q75,'\niqr = ', iqr)
    return(mean, median, mode, std, minfreq, peak, q25, q75, iqr)

def main():
    directory=os.listdir(raw_folder1)
    for wav_file in directory:
        y, sr = librosa.load(raw_folder1+wav_file)
        print(wav_file)
        frequencies=get_frequencies(y,sr)
        features=get_features(frequencies)
    
##print('Female sample')
##y, sr = librosa.load(raw_folder1+'00259.wav')
##frequencies=get_frequencies(y,sr)
##features=get_features(frequencies)
##
##print('\nMale sample')
##y, sr = librosa.load(raw_folder2+'00259.wav')
##frequencies=get_frequencies(y,sr)
##features=get_features(frequencies)

main()

