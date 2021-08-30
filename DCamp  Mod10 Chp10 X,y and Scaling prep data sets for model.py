# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 18:13:13 2021

@author: micha
"""




"""Create in X and y data sets"""
# # Reindex data using a DatetimeIndex
# df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)

# target_column = ['unemploy'] 
# predictors = list(set(list(df.columns))-set(target_column))
# df[predictors] = df[predictors]/df[predictors].max()
# df.describe()

# y=df["loan"]
# X=df[["age","children","saving"]]

# # Create arrays for the features and the response variable
# y = df['diabetes'].values
# X = df.drop('diabetes', axis=1).values 

# # Create arrays for features and target variable
# # y = df['target'].values
# # X = df['feature'].values
# X = boston.drop('MEDV', axis=1).values
# y = boston['MEDV'].values


# #Predicting house value from a single feature
# #from a single feature
# X_rooms = X[:,5]
# print(type(X_rooms))
# print(type(y))

# # Reshape X and y
# # y_reshaped = y.reshape(-1,1)
# # X_reshaped = X.reshape(-1,1)
# y = y.reshape(-1, 1)
# X_rooms = X_rooms.reshape(-1, 1)
# # Print the dimensions of y_reshaped and X_reshaped
# print("Dimensions of y after reshaping: ", y.shape)







"""Scaling in scikit-learn """
# #option 1 scale

# boston = pd.read_csv("boston.csv")
# from sklearn.preprocessing import scale
# X = boston
# print("mean not scaled: ", np.mean(X))
# print("STD not scaled: ", np.std(X))
# X_scaled = scale(X)

# print("mean X scaled ", np.mean(X_scaled))
# print("STD x scaled: ", np.std(X_scaled))

# #option 2 scale

# print("option 1 scale")
# boston = pd.read_csv("boston.csv")
# X=boston
# import sklearn.preprocessing as preprocessing
# minmax = preprocessing.MinMaxScaler()
# # X is a matrix with float type
# minmax.fit(X)
# X_minmax = minmax.transform(X)
# print("option 2 scale")
# print("mean not scaled: ", np.mean(X))
# print("STD not scaled: ", np.std(X))
# X_scaled = scale(X)
# print("mean X scaled ", np.mean(X_scaled))
# print("STD x scaled: ", np.std(X_scaled))

# #Option 3

# # The third line normalizes the predictors. This is done because the units of the 
# # variables differ significantly and may influence the modeling process. 
# # To prevent this, we will do normalization via scaling of the predictors between 0 and 1.
# target_column = ['unemploy'] 
# predictors = list(set(list(df.columns))-set(target_column))
# df[predictors] = df[predictors]/df[predictors].max()
# df.describe()



# #Scaling in scikit-learn 
# boston = pd.read_csv("boston.csv")
# from sklearn.preprocessing import scale
# X = boston
# print("mean not scaled: ", np.mean(X))
# print("STD not scaled: ", np.std(X))
# X_scaled = scale(X)

# print("mean X scaled ", np.mean(X_scaled))
# print("STD x scaled: ", np.std(X_scaled))























