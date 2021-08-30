# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:34:01 2021

@author: mich

Data camp Data ananlysis- Module 10 Chapter 2

Regression target variable is continious
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


"""
Import data
"""
# Read the CSV file into a DataFrame: boston
boston = pd.read_csv("boston.csv")
print("Head() of data: ",boston.head())
print("Describe() of data: ", boston.describe())
print("Info() of data: ",boston.info())

"""   
Creating feature and target arrays
"""
# Create arrays for features and target variable
# y = df['target'].values
# X = df['feature'].values
X = boston.drop('MEDV', axis=1).values
y = boston['MEDV'].values
# Print the dimensions of y and X before reshaping
print("Dimensions of y before reshaping: ", y.shape)
print("Dimensions of X before reshaping: ", X.shape)


"""
Predicting house value from a single feature
"""
#from a single feature
X_rooms = X[:,5]
print(type(X_rooms))
print(type(y))

# Reshape X and y
# y_reshaped = y.reshape(-1,1)
# X_reshaped = X.reshape(-1,1)
y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1)
# Print the dimensions of y_reshaped and X_reshaped
print("Dimensions of y after reshaping: ", y.shape)
print("Dimensions of X after reshaping: ", X_rooms.shape)


"""
Plotting house valuevs number of rooms
"""
plt.scatter(X_rooms, y)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show();
plt.scatter(X_rooms, y)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show();


"""
Fitting regression model
"""
# instantiate model
reg = LinearRegression()

#Fit model to data - performing OLS
reg.fit(X_rooms, y)

#Make a prediction on target 
prediction_space = np.linspace(min(X_rooms),max(X_rooms)).reshape(-1, 1)


plt.scatter(X_rooms, y, color='blue')
y_pred = reg.predict(prediction_space)
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()

# # Compute and print R^2 and RMSE
# print("R^2: {}".format(reg.score(X_rooms, y)))
# rmse = np.sqrt(mean_squared_error(y, y_pred))
# print("Root Mean Squared Error: {}".format(rmse))

"""Hold-out set in practice II: Regression"""
#Hold-out set in practice II: Regression
#Remember lasso and ridge regression from the 
#previous chapter? Lasso used the  penalty to regularize, 
#while ridge used the  penalty. There is another type of 
#regularized regression known as the elastic net. 
#In elastic net regularization, the penalty term 
#is a linear combination of the l1 and l2  penalties
# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))

#Ridge - example 
# Import necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)
# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)
# Print the cross-validated scores
print("cross-validated scores: ",ridge_cv)



















