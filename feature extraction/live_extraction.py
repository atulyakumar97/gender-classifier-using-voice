import pandas as pd
import os
from pitch import get_pitch
from mfcc import get_mfcc
import librosa
import  scipy.io.wavfile as wav

path = "C:\\Users\\Atulya\\Documents\\GitHub\\gender-classifier-using-voice\\feature extraction\\";

freq_col=['pitch']
mfcc_col=['mfcc'+str(i+1) for i in list(range(110))]
col = freq_col+mfcc_col+['label','filename']

def main(path):
    df = pd.DataFrame()
    print('Extracting features')
   
    directory=os.listdir(path)
    for wav_file in directory:
        print(wav_file,end='')
        label=int(input(' Enter gender (1 for male, 0 for female): ')) #1 = male, 0 = female
        write_features=[]
        y, sr = librosa.load(path+wav_file)
        fs,x = wav.read(path+wav_file)
        
        pitch=get_pitch(fs,x)
        mfcc_features=get_mfcc(y,sr)
        
        write_features=[pitch]+mfcc_features.tolist()[0]+[label, wav_file]
        df = df.append([write_features])
    df.columns = col
    df.to_csv('recorded_audio_features.csv')

main(path+"recorded_audio\\")
