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

for column in columns_test[4:]:
    final_set[column].fillna((final_set[column].mean()), inplace=True)

columns = list(dataset.columns.values)

for column in columns[4:]:
        dataset[column].fillna((dataset[column].mean()), inplace=True)





X = dataset.iloc[:,1:40].values
Y_train = dataset.iloc[:,-1].values
X_test = final_set.iloc[:,1:40].values



labelencoder_X_1 = LabelEncoder()
X[:,0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_3 = LabelEncoder()
X[:,2] = labelencoder_X_3.fit_transform(X[:, 2])


labelencoder_X_test = LabelEncoder()
X_test[:,0] = labelencoder_X_test.fit_transform(X_test[:, 0])
labelencoder_X_test_2 = LabelEncoder()
X_test[:,2] = labelencoder_X_test_2.fit_transform(X_test[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [0, 2])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 0:]

onehotencoder2 = OneHotEncoder(categorical_features = [0, 2])
X_test = onehotencoder2.fit_transform(X_test).toarray()
X_test = X_test[:, 0:]


sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)


classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=67, units=33))
# Adding the second hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=33))
# Adding the output layer
classifier.add(Dense(kernel_initializer="uniform", activation="sigmoid", units=1))

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, Y_train, batch_size = 10, epochs = 15)

Y_pred = classifier.predict(X_test)


print(Y_pred)
