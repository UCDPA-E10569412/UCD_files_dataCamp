# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:53:37 2021

@author: micha

Data camp Data ananlysis- Module 10 Chapter 1

Supervised learning

Classifiers: target variable consistes of catagories.
Classification models are models which predict a categorical label

"""

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Import necessary modules
from sklearn.model_selection import train_test_split



""" Import data and visulaise"""
iris = datasets.load_iris()



"""Examine the data downloaded"""
print(type(iris))
print(iris.keys())
print(type(iris.data))
print(type(iris.target))
# print(iris.shape)
# print(iris.describe)

""" exporatory data analysis, """
#Table EDA, 

#Visual EDA
#_ = pd.plotting.scatter_matrix(df, c = y, figsize = [8, 8], s=150, marker = 'D')


""" Create Features (X) and Target (y) data sets"""
# Create feature and target arrays
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
# Create arrays for the features and the response variable
# y = df['diabetes'].values
# X = df.drop('diabetes', axis=1).values 
print(df.head())


""" set up Training and Test data sets """
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)


#Start of classifiers
""" Set up classifier - KNN  """
""" ======================== """
from sklearn.neighbors import KNeighborsClassifier 
# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

""" Predict on Data set """
# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

""" Accuracy """
# Print the accuracy
print(knn.score(X_test, y_test))
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


"""Hyperparameter tuning with RandomizedSearchCV""" 
#GridSearchCVinscikit-learn
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)
print("knn_cv.best_params: ",knn_cv.best_params_)
print("knn_cv.best_score: ",knn_cv.best_score_)





""" Set up classifier - LogReg  """
""" =========================== """
""" Logistic regression and The ROC curve"""
# #Logisticregressioninscikit-learn
# #Logistic regression for binary classification Logistic regression outputs probabilities 
# #If theprobability ‘p’ is greater than 0.5: The data is labeled ‘1’ 
# #If the probability ‘p’ is lessthan 0.5:The data is labeled ‘0’
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# # Create the classifier: logreg
# logreg = LogisticRegression()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# # Fit the classifier to the training data
# logreg.fit(X_train, y_train)
# # Predict the labels of the test set: y_pred
# y_pred = logreg.predict(X_test)
# print("Print out y_pred for logreg: ", y_pred)
# #Plotting the ROC curve
# from sklearn.metrics import roc_curve
# y_pred_prob = logreg.predict_proba(X_test)[:,1]
# # Generate ROC curve values: fpr, tpr, thresholds
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# # Plot ROC curve
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='Logistic Regression')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Logistic Regression ROC Curve')
# plt.show();

# logreg.predict_proba(X_test)[:,1]
# #Larger area under the ROC curve=better model


""" Setr up Classifier Decision tree """
""" ================================ """
# # Import necessary modules
# from scipy.stats import randint
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import RandomizedSearchCV
# Setup the parameters and distributions to sample from: param_dist
# param_dist = {"max_depth": [3, None],
#               "max_features": randint(1, 9),
#               "min_samples_leaf": randint(1, 9),
#               "criterion": ["gini", "entropy"]}
# # Instantiate a Decision Tree classifier: tree
# tree = DecisionTreeClassifier()
# # Instantiate the RandomizedSearchCV object: tree_cv
# tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
# # Fit it to the data
# tree_cv.fit(X, y)
# # Print the tuned parameters and score
# print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
# print("Best score is {}".format(tree_cv.best_score_))



""" Evaluate model"""
""" confusion matrix and generating a classification report."""
# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Evaluation Metrics -  evaluate the performance of the model using four metrics Accuracy 
# Accuracy is the fraction of cases correctly classified. 
# For a binary classifier, it is represented as accuracy = (TP+TN)/(TP+TN+FP+FN), where 
# 
# True Positive or TP are cases with positive labels which have been correctly classified as positive. 
# True Negative or TN are cases with negative labels which have been correctly classified as negative. 
# False Positive or FP are cases with negative labels which have been incorrectly classified as positive. 
# False Negative or FN are cases with positive labels which have been incorrectly classified as negative.
# 
# Precision Precision is the fraction of correctly classified label cases out of all cases classified with that label value. It is represented as Precision = P = TP / (TP+ FP)
# 
# Recall Recall is the fraction of cases of a label value correctly classified out of all cases that actually have that label value. It is represented as Recall = R = TP / (TP+FN)
# 
# F1-score The F1 statistic is a weighted average of precision and recall. It is represented as F1 = =2(PR) / (P+R)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))












"""
Hold-out set in practice I: Classification
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
# Fit it to the training data
logreg_cv.fit(X_train, y_train)
# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))
"""








"""
# Hyperparameter tuning with GridSearchCV
# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}
# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()
#Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
# Fit it to the data
logreg_cv.fit(X, y)
# Print the tuned parameter and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))
"""









"""
#AUCinscikit-learn
from sklearn.metrics import roc_auc_score
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)

#AUC using cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
print(cv_scores)
# Import necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

"""

