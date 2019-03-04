import numpy as np 
import pandas as pd
import tensorflow as tf
from keras.backend import tensorflow_backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

x_train = pd.read_csv('/Users/bhargavdesai/Desktop/Data/x_train.csv')
x_test = pd.read_csv('/Users/bhargavdesai/Desktop/Data/x_test.csv')
y_train = pd.read_csv('/Users/bhargavdesai/Desktop/Data/y_train.csv')
y_test = pd.read_csv('/Users/bhargavdesai/Desktop/Data/y_test.csv')
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values
x_train = x_train.values
x_train = x_train.reshape(x_train.shape[0],200,200)
x_test = x_test.reshape(x_test.shape[0],200,200)
print("Loading done")

def model(x_train,x_test,y_train,y_test):
    model = Sequential() 
    model.add(LSTM(80, input_shape=(x_train.shape[1:]), kernel_initializer= 'glorot_uniform', dropout=0.2))
    model.add(Dense(40, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(20, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, kernel_initializer='glorot_uniform', activation = 'sigmoid'))
    return model


with tf.Session(config=tf.ConfigProto(
                    intra_op_parallelism_threads=4)) as sess:
    K.set_session(sess) 
    Model = model(x_train,x_test,y_train,y_test)
    Model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    Model.summary()
    Model.fit(x=x_train,y=y_train, epochs=180, batch_size=32)
    Model.save('gc6_(200-200)_trained-on-mac.h5')
    del Model
    print("-------End of Line-------------")

K.clear_session()    
    

    




 