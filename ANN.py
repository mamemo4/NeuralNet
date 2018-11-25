# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime 


# Importing Foundation data
FNDdata = pd.read_csv('FoundationData.csv', parse_dates = True)

# Convert Date to datetime
FNDdata['GiftDate2'] = pd.to_datetime(FNDdata['GiftDate'])

# Extract only the year from datetime
FNDdata['GiftYear'] = FNDdata['GiftDate2'].dt.year

# Set index as GiftYear
FNDdata = FNDdata.set_index('GiftYear')


# Importing the Annual Unemployment Data
Unemploy = pd.read_csv('UnemploymentData.csv', parse_dates = True)

# Set the index of Unmeploy to Year
Unemploy = Unemploy.set_index('Year')

# Joining the FND data with Unemploy on the year
Join1 = FNDdata.join(Unemploy, rsuffix = '_unemp')

# Import S&P 500 Data
SP = pd.read_csv('SP500Annual.csv')

# Set index
SP = SP.set_index('Year')

# Join the three tables
Join2 = Join1.join(SP, rsuffix = '_sp')

# Import GDP Annual Growth % data
GDP = pd.read_csv('GDP.csv')

# Set the index
GDP = GDP.set_index('Year')

# Join all 4 tables
Join3 = Join2.join(GDP, rsuffix = 'gdp')

# PRE - Neural Network

NN_DataPreProcess = Join3.reset_index()

NN_Data = NN_DataPreProcess[['Average_Open','Annual', 'United_States']]

# Filters out those in 2018
NN_Data = NN_Data.dropna(subset=['Average_Open'])

# Filter out those in 2018
Test = NN_DataPreProcess[NN_DataPreProcess.index < 26298]

Output = Test.iloc[:, 2].values

# Setting up dataframes for 10 - cross validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(NN_Data, Output, test_size = 0.1,random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# NEURAL NETWORK TIME!

import keras
from keras.model import Sequential
from keras.layers import Dense


NN = Sequential()
# Input and 1st hidden layer

NN.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu', input_dim = 3))

# Second hidden layer
NN.add(Dense(output_dim = 1), init = 'uniform', activation = 'relu')

# training the NN
NN.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
NN.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
