# Kernel SVM

# Importing the libraries
import numpy as np
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('C:\\Users\\Atulya\\Documents\\GitHub\\gender-classifier-using-voice\\feature extraction\\features.csv')
dataset_transpose=dataset.T
X = dataset_transpose.iloc[1:-1,:].values
y = dataset_transpose.iloc[-1:,:].values

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
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size = 0.0125, random_state = None,shuffle=True)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(C=3,gamma=0.02,kernel = 'rbf', random_state =None,max_iter=100000)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# And find the final test error
correct_pred=sum(y_pred == y_test)
print(correct_pred, ' classified correctly out of ',np.shape(y_test)[0])
#print('accuracy = ', correct_pred*100/(y_pred.shape[0]))

print('Train set accuracy = ',classifier.score(X_train,y_train)*100)
print('Test set accuracy = ',classifier.score(X_test,y_test)*100)
#
##from sklearn.model_selection import cross_val_score
##accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
##print(accuracies.mean())
##print(accuracies.std())
#
#
##---------------------------------Testing on recorded audio ---------------------------#

recorded_dataset = pd.read_csv('C:\\Users\\Atulya\\Documents\\GitHub\\gender-classifier-using-voice\\feature extraction\\recorded_audio_features.csv')
X_recorded = recorded_dataset.iloc[:,1:-2].values           #-2
y_recorded = recorded_dataset.iloc[:,-2:-1].values

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy='most_frequent')
imputer = imputer.fit(X_recorded[:,:])
X_recorded[:,:] = imputer.transform(X_recorded[:,:])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y_recorded_test = labelencoder_y.fit_transform(y_recorded.ravel())

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_recorded_test = sc.fit_transform(X_recorded)

# Predicting the Test set results
y_recorded_pred = classifier.predict(X_recorded_test)

# And find the final test error
correct_pred=sum(y_recorded_pred == y_recorded_test)
print(correct_pred, ' classified correctly out of ',np.shape(y_recorded_test)[0],' recorded files')