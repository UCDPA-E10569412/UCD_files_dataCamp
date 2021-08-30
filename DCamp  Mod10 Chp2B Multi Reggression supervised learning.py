# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:34:01 2021

@author: mich

Data camp Data ananlysis- Module 10 Chapter 2
Regression models are models which predict a continuous outcome. target variable
Linear regression on all features - Predicting house value from ALL features


Problem Statement:
What you think Data will shouw?:

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



"""Import data"""
# Read the CSV file into a DataFrame: boston
boston = pd.read_csv("boston.csv")
print("Head() of data: ",boston.head())
print("Describe() of data: ", boston.describe())
print("Info() of data: ",boston.info())
print("Shape of dataframe: ", boston.shape)



"""Data preprocessing"""


"""Scale data"""
#The third line normalizes the predictors. This is done because the units of the 
#variables differ significantly and may influence the modeling process. 
#To prevent this, we will do normalization via scaling of the predictors between 0 and 1.

# target_column = ['unemploy'] 
# predictors = list(set(list(df.columns))-set(target_column))
# df[predictors] = df[predictors]/df[predictors].max()
# df.describe()


"""Create Feature and Target arrays"""
# target_column = ['unemploy'] 
# predictors = list(set(list(df.columns))-set(target_column))
# df[predictors] = df[predictors]/df[predictors].max()
# df.describe()



# Create arrays for features and target variable
# y = df['target'].values
# X = df['feature'].values
X = boston.drop('MEDV', axis=1).values
y = boston['MEDV'].values
# Print the dimensions of y and X before reshaping
print("Contents of y : ", y)
print("Contents of X : ", X)
print("Dimensions of y : ", y.shape)
print("Dimensions of X : ", X.shape)


"""Split data into Train and Test"""
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)





# Set up linear reggression models

""" Set up Linear Reggeion model Reg"""
#The simplest form of regression is the linear regression, which assumes that the predictors 
# have a linear relationship with the target variable. The input variables are assumed to 
# have a Gaussian distribution. Another assumption is that the predictors are 
# not highly correlated with each other (a problem called multi-collinearity).
# The parameters a and b of the model are selected through the Ordinary least 
# squares (OLS) method. It works by minimizing the sum of squares of residuals (actual value - predicted value).
"""================================="""
"""Linear Reggeion - example 1 """
from sklearn.linear_model import LinearRegression

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

pred_test = y_pred                         # output of prediction(X_test)
pred_train = reg_all.predict(X_train)      # output of prediction(X_train)
print("mean() of results in y_pred : ", y_pred.mean())

from sklearn.metrics import mean_squared_error

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


""" Compute 5-fold cross-validation scores: cv_scores """
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(reg_all, X,y,cv=5)
# Print the 5-fold cross-validation scores
print("5-Fold CV Scores: ", cv_scores)
print("Average 5-Fold CV Score: {}".format(cv_scores.mean()))











"""Ridge regression model """
#Ridge regression is an extension of linear regression where the loss function is modified to minimize the complexity of the model. This modification is done by adding a penalty parameter that is equivalent to the square of the magnitude of the coefficients.
#A low alpha value can lead to over-fitting, whereas a high alpha value can lead to under-fitting# from sklearn.linear_model import Ridge
"""================================="""

"""Ridge - example 1 """
# from sklearn.linear_model import Ridge

# # Instantiate a ridge regressor: ridge
# ridge = Ridge(alpha=0.1, normalize=True)

# # Fit the regressor to the training data 
# ridge.fit(X_train, y_train)

# # Predict on the test data:
# ridge_pred = ridge.predict(X_test)

# pred_test  = ridge_pred                  # output of prediction(X_test)
# pred_train = ridge.predict(X_train)      # output of prediction(X_train)

# print("Ridge score for model: ", ridge.score(X_test, y_test))



""" Ridge - example 2 """
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import Ridge

# # Instantiate a ridge regressor: ridge
# ridge = Ridge(alpha=0.5, normalize=True)

# # Perform 5-fold cross-validation: ridge_cv
# ridge_cv = cross_val_score(ridge, X, y, cv=5)

# # Print the cross-validated scores
# print("cross-validated scores: ",ridge_cv)


# # Fit the regressor to complete Evaluation below 
# ridge.fit(X_train, y_train)
# # Predict on the test data:
# ridge_pred = ridge.predict(X_test)

# pred_test  = ridge_pred                  # output of prediction(X_test)
# pred_train = ridge.predict(X_train)      # output of prediction(X_train)



""" Ridge - example 3 """
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import cross_val_score

# # Setup the array of alphas and lists to store scores
# alpha_space = np.logspace(-4, 0, 50)
# ridge_scores = []
# ridge_scores_std = []

# # Create a ridge regressor: ridge
# ridge = Ridge(normalize=True)

# # Compute scores over range of alphas
# for alpha in alpha_space:
#     # Specify the alpha value to use: ridge.alpha
#     ridge.alpha = alpha
#     # Perform 10-fold CV: ridge_cv_scores
#     ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
#     # Append the mean of ridge_cv_scores to ridge_scores
#     ridge_scores.append(np.mean(ridge_cv_scores))
#     # Append the std of ridge_cv_scores to ridge_scores_std
#     ridge_scores_std.append(np.std(ridge_cv_scores))


# # Display the plot
# cv_scores  = ridge_scores
# cv_scores_std = ridge_scores_std
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(alpha_space, cv_scores)
# std_error = cv_scores_std / np.sqrt(10)
# ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
# ax.set_ylabel('CV Score +/- Std Error')
# ax.set_xlabel('Alpha')
# ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
# ax.set_xlim([alpha_space[0], alpha_space[-1]])
# ax.set_xscale('log')
# plt.show()


# # Fit the regressor to complete Evaluation below 
# ridge.fit(X_train, y_train)
# # Predict on the test data:
# ridge_pred = ridge.predict(X_test)

# pred_test  = ridge_pred                  # output of prediction(X_test)
# pred_train = ridge.predict(X_train)      # output of prediction(X_train)










"""Lasso regression model  """
#Lasso regression, or the Least Absolute Shrinkage and Selection Operator, 
#is also a modification of linear regression. In Lasso, the loss function 
#is modified to minimize the complexity of the model by limiting the 
#sum of the absolute values of the model coefficients 
#(also called the l1-norm).
#In the above loss function, alpha is the penalty parameter we need 
#to select. Using an l1 norm constraint forces some weight values 
#to zero to allow other coefficients to take non-zero values.
#Can be used to select important features of a dataset, 
#Shrinks the coecients of less important features to exactly 0
"""================================="""

"""Linear Reggeion- example 1"""
# from sklearn.linear_model import Lasso

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# lasso = Lasso(alpha=0.1, normalize=True)

# lasso.fit(X_train, y_train)

# lasso_pred = lasso.predict(X_test)

# pred_test  = lasso_pred                  # output of prediction(X_test)
# pred_train = lasso.predict(X_train)      # output of prediction(X_train)

# print("Lasso score for model: ", lasso.score(X_test, y_test))
# names = boston.drop('MEDV', axis=1).columns
# lasso = Lasso(alpha=0.1)
# lasso_coef = lasso.fit(X, y).coef_
# # _ = plt.plot(range(len(names)), lasso_coef)
# # _ = plt.xticks(range(len(names)), names, rotation=60)
# # _ = plt.ylabel('Coefficients')
# plt.plot(range(len(names)), lasso_coef)
# plt.xticks(range(len(names)), names, rotation=60)
# plt.ylabel('Coefficients')
# plt.show()


"""#Lasso regression model - example 2"""
# from sklearn.linear_model import Lasso

# # Instantiate a lasso regressor: lasso
# lasso = Lasso(alpha=0.4, normalize=True)

# # Fit the regressor to the data
# lasso.fit(X, y)

# # Compute and print the coefficients
# lasso_coef = lasso.coef_
# print(lasso_coef)

# """
# # Plot the coefficients
# plt.plot(range(len(boston)), lasso_coef)
# plt.xticks(range(len(boston)), boston.values, rotation=60)
# plt.margins(0.02)
# plt.show()
# """

# # Fit the regressor to complete Evaluation below 
# lasso.fit(X_train, y_train)
# # Predict on the test data:
# lasso_pred = lasso.predict(X_test)

# pred_test  = lasso_pred                  # output of prediction(X_test)
# pred_train = lasso.predict(X_train)      # output of prediction(X_train)









"""ElasticNet Regression model  """
#ElasticNet combines the properties of both Ridge and Lasso regression. 
#It works by penalizing the model using both the l2-norm and the l1-norm.
"""================================="""

"""ElasticNet Regression- example 1 Hold-out set in practice II: Regression"""

# #Hold-out set in practice II: Regression
# #Remember lasso and ridge regression from the 
# #previous chapter? Lasso used the  penalty to regularize, 
# #while ridge used the  penalty. There is another type of 
# #regularized regression known as the elastic net. 
# #In elastic net regularization, the penalty term 
# #is a linear combination of the l1 and l2  penalties
# # Import necessary modules
# from sklearn.linear_model import ElasticNet
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split


# # Create the hyperparameter grid
# l1_space = np.linspace(0, 1, 30)
# param_grid = {'l1_ratio': l1_space}

# # Instantiate the ElasticNet regressor: elastic_net
# elastic_net = ElasticNet()

# # Setup the GridSearchCV object: gm_cv
# gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# # Fit it to the training data
# gm_cv.fit(X_train, y_train)

# # Predict on the test set and compute metrics
# y_pred = gm_cv.predict(X_test)
# r2 = gm_cv.score(X_test, y_test)
# mse = mean_squared_error(y_test, y_pred)
# print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
# print("Tuned ElasticNet R squared: {}".format(r2))
# print("Tuned ElasticNet MSE: {}".format(mse))














""" Evaluate the Model """
# Evaluation Metrics - We will evaluate the performance of the model using two metrics - R-squared value and Root Mean Squared Error (RMSE).
# R-squared values range from 0 to 1 and are commonly stated as percentages. 
# It is a statistical measure that represents the proportion of the variance for a target variable that is explained by the independent variables. 
# The other commonly used metric for regression problems is; 
# RMSE, that measures the average magnitude of the residuals or error. 
# Ideally, lower RMSE and higher R-squared values are indicative of a good model.
# The most ideal result would be an RMSE value of zero and R-squared value of 1
"""==================================="""
from sklearn.metrics import mean_squared_error

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# #Evaluation Option 2:
from sklearn.metrics import r2_score
#code below predicts on the training set. The second and third lines of code prints the evaluation metrics - RMSE and R-squared - on the training set
print("The RMSE score for the training set: ",np.sqrt(mean_squared_error(y_train,pred_train)))
print("The R2 score for the training set: ",r2_score(y_train, pred_train))
#code below predicts on the training set. The second and third lines of code prints the evaluation metrics - RMSE and R-squared - on the training set
print("The RMSE score for the test set: ",np.sqrt(mean_squared_error(y_test,pred_test))) 
print("The R2 score for the test set: ",r2_score(y_test, pred_test))


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Printout relevant metrics
model = reg_all
print("Model Coefficients:", model.coef_)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Coefficient of Determination:", r2_score(y_test, y_pred))
# Results



"""There are other iterations that can be done to improve 
model performance. We have assigned the value of alpha to be 0.01, 
but this can be altered by hyper parameter tuning to arrive at the 
optimal alpha value. Cross-validation can also be tried along with 
feature selection technique"""








































