
# coding: utf-8

# In[2]:


# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('voice.csv')
X = dataset.iloc[:, 0:19].values
y = dataset.iloc[:, 20].values

print(X)
print(y)


# In[3]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


# In[4]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[5]:


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = None)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# And find the final test error
correct_pred=sum(y_pred == y_test)
print(correct_pred)
print('accuracy = ', correct_pred*100/(y_pred.shape[0]))


# In[6]:


from sklearn import svm

for c in range(1, 11):
    model = svm.LinearSVC(C=c*1.)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    correct_pred=sum(y_pred == y_test)
    print (c, correct_pred, 'accuracy = ', correct_pred*100/(y_pred.shape[0]) )

