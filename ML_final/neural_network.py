#recording folder must exist at ML_final\recorded_audio

import numpy as np 
import librosa
from keras.models import load_model
import os
import soundfile as sf
from record_audio import record

path="C:\\Users\\Atulya\Documents\\GitHub\\gender-classifier-using-voice\\ML_final\\"
record(path)

Model = load_model('neural_network.h5')
g=[]
for audiofile in os.listdir(path+'recorded_audio\\'):
            try:
                y_in, sr = sf.read(os.path.join(path+'recorded_audio\\',audiofile))
                y_in = librosa.resample(y_in, sr, 8000)
                y_in = y_in[0:40000]
                y_in = np.concatenate((y_in, [0]* (40000 - y_in.shape[0])), axis=0)
                g.append(y_in)
            except RuntimeError:
                print(".DS_Store file detected and dismissed")
                pass
x_in = np.array(g)
x_in.shape
x_in = x_in.reshape(x_in.shape[0],200,200)
#result = Model.predict(x_in)
m=0
for r in result:
    if r>0.5:
        print("Male Voice Predicted")
    else:
        print("Female Voice Predicted")