#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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



# Importing the dataset
dataset = pd.read_csv('train.csv')
#print(dataset)

final_set = pd.read_csv('test.csv')

columns_test = list(final_set.columns.values)

#for column in columns_test[4:]:
 #   final_set[column].fillna((final_set[column].mean()), inplace=True)

columns = list(dataset.columns.values)

for column in columns[4:]:
        dataset[column].fillna((dataset[column].mean()), inplace=True)
#dataset = dataset.dropna(axis=0,how='any')


values = dataset.iloc[:,1:40].values

labelencoder_X_3 = LabelEncoder()
values[:,0] = labelencoder_X_3.fit_transform(values[:, 0])
labelencoder_X_3 = LabelEncoder()
values[:,2] = labelencoder_X_3.fit_transform(values[:, 2])

print(values[:,0])

onehotencoder = OneHotEncoder(categorical_features = [2])
values = onehotencoder.fit_transform(values).toarray()
values = values[:,0:]


unique_countries = np.unique(values[:,2])
countries_index = {}

number = 0
for country in unique_countries:
    countries_index[country] = number
    number+=1

unique_countries = len(unique_countries)

rows = len(dataset)

first_value = countries_index[values[0][2]]
sequences = np.zeros((unique_countries,rows,38))
sequence_counter = 0
timestep_counter = 0
position_counter = 0
for value in values[0:]:
    if countries_index[value[2]] == first_value:
        sequences[sequence_counter] = first_value
        sequences[sequence_counter][timestep_counter] = value[3]
        sequences[sequence_counter][timestep_counter][0] = value[0]
        sequences[sequence_counter][timestep_counter][1] = value[1]
        for i in range(4,40):
            sequences[sequence_counter][timestep_counter][i-2] = value[i]
        timestep_counter+=1
    if position_counter == len(values) - 1:
        continue
    if countries_index[values[position_counter + 1][2]] != first_value:
        first_value = countries_index[values[position_counter + 1][2]]
        sequence_counter+=1
    position_counter+=1

'''
groups = [4,40]
#for j in range(4, 10):
 #   groups.append(j)
    
i = 1
#for i in range(4, 41):
    
x = values[:,2]
y = values[:,40]
plt.scatter(y, x)
plt.plot(figsize=(100,100))
plt.title(dataset.columns[2], loc='right')
plt.show()
'''