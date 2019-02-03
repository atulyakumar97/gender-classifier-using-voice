import pandas as pd
import os
from mfcc import get_mfcc
from frequencies import get_frequencies,get_features
import librosa

path_male = "C:\\Users\\Atulya\\Documents\\GitHub\\gender-classifier-using-voice\\Male\\";
path_female = "C:\\Users\\Atulya\\Documents\\GitHub\\gender-classifier-using-voice\\Female\\";

def main():
    df = pd.DataFrame()
    directory=os.listdir(path_male)
    for wav_file in directory:
        write_features=[]
        y, sr = librosa.load(path_male+wav_file)
        print(wav_file)
        
        frequencies=get_frequencies(y,sr)
        freq_features=get_features(frequencies)
        mfcc_features=get_mfcc(y,sr)
        
        print(freq_features)
        print(mfcc_features)
        
        write_features=freq_features+mfcc_features.tolist()[0]
        df = df.append([write_features], ignore_index=True)
        break #remove break to execute for all files
    
    df.to_csv('features.csv')
    return(df)

main()