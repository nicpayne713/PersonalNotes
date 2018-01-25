# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:26:33 2017

@author: paynen3
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# label encoder encodes variables
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
# OneHotEncoder takes encoded categories and creates Dummy Variables
onehotendoer = OneHotEncoder(categorical_features = [3])
X = onehotendoer.fit_transform(X).toarray()

# Avoid the Dummy Variable Trap
X = X[:,1:] # Remove one Dummy Variable for analysis

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Leaner Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test,)

error = y_pred-y_test

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,3]] # remove x2 after 1 iteration, and so on and so forth
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
