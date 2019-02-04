import pandas as pd
from sklearn import svm
import random
import decimal

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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=0, shuffle=True)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


best_score = 0  
best_params = {'C': None, 'gamma': None}

#for a preset number of iterations
for i in range(100):
    #try random values for each hyperparameter
    C=float(decimal.Decimal(random.randrange(0, 1000))/100)
    gamma=float(decimal.Decimal(random.randrange(10, 300))/100)
    svc = svm.SVC(C=C,gamma=gamma,kernel = 'rbf', random_state = 0,max_iter=1000000,class_weight='balanced')
    print(C,gamma)
    svc.fit(X_train, y_train)
    score = svc.score(X_test, y_test)
    print(score*100)

    if score > best_score:
        best_score = score
        best_params['C'] = C
        best_params['gamma'] = gamma

print(best_score, best_params)

