# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import theano
import tensorflow
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import mode
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
import math
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



# Importing the dataset
dataset = pd.read_csv('train.csv')
#print(dataset)


#dataset['GOV_SPEND_EDU'] = dataset.groupby(['LOCATION'])['GOV_SPEND_EDU'].transform(lambda x: x.fillna(x.mean()))



dataset = dataset.dropna(axis=0,how='any')

X = dataset.iloc[:,1:40].values
Y = dataset.iloc[:,-1].values

labelencoder_X_1 = LabelEncoder()
X[:,0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_3 = LabelEncoder()
X[:,2] = labelencoder_X_3.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features = [0, 2])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 0:]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=67, units=25))
classifier.add(Dropout(0.3, noise_shape=None, seed=None))
# Adding the second hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=25))
classifier.add(Dropout(0.3, noise_shape=None, seed=None))
# Adding the output layer
classifier.add(Dense(kernel_initializer="uniform", activation="sigmoid", units=1))

# Compiling Neural Network
classifier.compile(optimizer = 'adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, Y_train, batch_size = 10, epochs = 15)

print(classifier.summary())
print(classifier.get_weights())

Y_pred = classifier.predict(X_test)

count = 0;
summean = 0
for entry in Y_test:
    #print(entry)
    #print(Y_pred[count])
    num = Y_pred[count]
    #print(abs((entry - num)**2))
    summean += abs((entry - num)**2)
    count+=1

print("sum mean " + str(summean))
rooterror = math.sqrt((1/134)*(summean))
print("error " + str(rooterror))

#print(X[0:20])

# -*- coding: utf-8 -*-

