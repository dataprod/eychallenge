#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 19:22:27 2018

@author: stals
"""

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
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
import csv


# Importing the dataset
dataset = pd.read_csv('train.csv')
#print(dataset)

final_set = pd.read_csv('test.csv')

columns_test = list(final_set.columns.values)

for column in columns_test[4:]:
    final_set[column] = final_set.groupby(['LOCATION'])[column].transform(lambda x: x.fillna(x.mean()))

#columns = list(dataset.columns.values)
#for column in columns[4:]:
 #   dataset[column] = dataset.groupby(['LOCATION'])[column].transform(lambda x: x.fillna(x.mean()))

dataset = dataset.dropna(axis=0,how='any')


'''
for column in columns[10:]:
    
    mean = dataset.groupby(['LOCATION'])[column].mean()  
    mean_overall = mean.mean()
    counted = 0
    for num in mean:
        if (math.isnan(num)):
            mean[counted] = mean_overall
            counted+=1
    
    number = 0
    for num in mean:
        dataset[column][number] = mean[number]
'''

#dataset = dataset.dropna(axis=0,how='any')
#final_set = final_set.dropna(axis=0, how='any')


indexes = final_set.iloc[:, 0]

X = dataset.iloc[:,2:40].values
Y_train = dataset.iloc[:,-1].values
X_test = final_set.iloc[:,2:40].values



#abelencoder_X_1 = LabelEncoder()
#X[:,0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_3 = LabelEncoder()
X[:,1] = labelencoder_X_3.fit_transform(X[:, 1])


#labelencoder_X_test = LabelEncoder()
#X_test[:,0] = labelencoder_X_test.fit_transform(X_test[:, 0])
labelencoder_X_test_2 = LabelEncoder()
X_test[:,1] = labelencoder_X_test_2.fit_transform(X_test[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

onehotencoder2 = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder2.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]


sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)


classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=38, units=25))
# Adding the second hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=25))
# Adding the second hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=15))
# Adding the output layer
classifier.add(Dense(kernel_initializer="uniform", activation="sigmoid", units=1))

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X, Y_train, batch_size = 10, epochs = 15)

Y_pred = classifier.predict(X_test)

counter = 0
dic = {}
        
for each in Y_pred:
    dic[indexes[counter]] = each[0]
    print(each)
    print(counter)
    counter+=1

with open("csvfile.csv", "w") as csvfile:
    fieldnames = ['id', 'target']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for key in dic.keys():
        writer.writerow({'id': key, 'target': dic[key]})
