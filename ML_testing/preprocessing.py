# -*- coding: utf-8 -*-
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot

#Importing the dataset
dataset=pd.read_csv('voice.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1:].values

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=0, strategy='mean')
imputer=imputer.fit(X[:,:])
X[:,:]= imputer.transform(X[:,:])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0315, random_state=None, shuffle=True)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

