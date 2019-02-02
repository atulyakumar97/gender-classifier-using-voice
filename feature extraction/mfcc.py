import os
import librosa
import numpy as np

path_male = "C:\\Users\\Atulya\\Documents\\GitHub\\gender-classifier-using-voice\\Male";
path_female = "C:\\Users\\Atulya\\Documents\\GitHub\\gender-classifier-using-voice\\Female";

for audiofile in os.listdir(path_male):
    y, sr = librosa.load(os.path.join(path_male,audiofile));
    y = librosa.resample(y, sr, 8000);
    y = y[0:40000];
    y = np.concatenate((y, [0]* (40000 - y.shape[0])), axis=0);
      #Mel-frequency cepstral coefficients 
    mfcc=librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10,hop_length=4000);
    mfcc_feature=np.reshape(mfcc, (1,110))
