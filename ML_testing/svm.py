# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('C:\\Users\\Atulya\\Documents\\GitHub\\gender-classifier-using-voice\\feature extraction\\features.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1:].values

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy='mean')
imputer = imputer.fit(X[:,:])
X[:,:] = imputer.transform(X[:,:])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y.ravel())

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.02, random_state = 0,shuffle=True)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = None,max_iter=10000,class_weight='balanced')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# And find the final test error
correct_pred=sum(y_pred == y_test)
print(correct_pred)
print('accuracy = ', correct_pred*100/(y_pred.shape[0]))
