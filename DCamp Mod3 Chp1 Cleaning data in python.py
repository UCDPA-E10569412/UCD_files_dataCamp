# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 12:40:03 2021

@author: micha

1. Import data
2. create data frame
3. review data frame
Comun contents confirm same type
4. Correct index
5. Correct date format
6. Review missing values
7. Review duplicate values
8. 
8. Cross field validation




"""
import pandas as pd
import matplotlib.pyplot as plt

print("\n1. Import Data\n")
print("===================================")
# Import CSV file and output header
df = pd.read_csv('boston.csv')


print("\n2. Create dataFrame\n")
print("===================================")
df = pd.DataFrame()


print("\n3. data review on imported Data\n")
print("===================================")
# EDA data check
print("\nHead() of data: \n",df.head())
print("\nTail() of data: \n",df.tail())
print("\nColumn types of data: \n",df.dtypes)
print("\nDescribe() of data: \n", df.describe())
print("\nInfo() of data: \n",df.info())
print("\nshape of data: \n",df.shape)
# Print shape of original DataFrame
print("\nShape of Original DataFrame: {}".format(df.shape))
print("\ntype() of data: \n",type(df))




print("\n4. Initial tidy of Column contents\n")
print("===================================")

# Replace "+" with "00"
phones["Phone number"] = phones["Phone number"].str.replace("+", "00")
print(phones)

# Replace phone numbers with lower than 10 digits to NaN
digits = phones['Phone number'].str.len()
phones.loc[digits < 10, "Phone number"] = np.nanphones
print(phones)

# Find length of each row in Phone number column 
sanity_check = phone['Phone number'].str.len()

# Assert minmum phone number length is 10
assert sanity_check.min() >= 10

# Assert all numbers do not have "+" or "-"
assert phone['Phone number'].str.contains("+|-").any() == False

# Replace letters with nothing 
phones['Phone number'] = phones['Phone number'].str.replace(r'\D+', '')
phones.head()

# Print sum of all Revenue column
df['x'].sum()
# Remove $ from Revenue column
df['x'] = df['x'].str.strip('$')
#Strip duration of minutes
df['x'] = df['x'].str.strip('minutes') 
#Convert duration to integer
df['x'] = df['x'].astype('int')

# This will pass
assert 1+1 == 2
# This will not pass 
assert 1+1 == 3
#Verify that Revenue is now an integer
assert df['x'].dtype == 'int'

# Catagorial descriptions# - eg marraige: single, married, divorced
df = pd.DataFrame()
df['x'] = (0,1,3,2,1,1,1,2,3,2,3,2,1,2,3,2,1)
df['x'].describe()
# Convert to categorical 
df["x"] = df["x"].astype('category')
df['x'].describe()

# Get marriage status column (series)
marriage_status = df['marriage_status']
marriage_status.value_counts()
# Get value counts on DataFrame
marriage_status.groupby('marriage_status').count()
#Strip all spaces - lead and trainng whirte spaces
demographics = demographics['marriage_status'].str.strip()
demographics['marriage_status'].value_counts()

# Capitalize
marriage_status['marriage_status'] = marriage_status['marriage_status'].str.upper()
marriage_status['marriage_status'].value_counts()

# Lower case 
marriage_status['marriage_status'] = marriage_status['marriage_status'].str.lower()
marriage_status['marriage_status'].value_counts()


# Collapsing data into categories
# Using qcut()
import pandas as pd
group_names = ['0-200K', '200K-500K', '500K+']
demographics['income_group'] = pd.qcut(demographics['household_income'], q = 3,  labels = group_names)
# Print income_group column
demographics[['income_group', 'household_income']]


# data with in range
# Drop values using filtering
movies = movies[movies['avg_rating'] <= 5]
# Drop values using .drop()
movies.drop(movies[movies['avg_rating'] > 5].index, inplace = True)
# Assert results
assert movies['avg_rating'].max() <= 5
# Convert avg_rating > 5 to 5
movies.loc[movies['avg_rating'] > 5, 'avg_rating'] = 5
# Assert statement
assert movies['avg_rating'].max() <= 5


print("\n5. correct index / date \n")
print("===================================")
Timeuser_signups['subscription_date'] = pd.to_datetime(user_signups['subscription_date'])
# Assert that conversion happened
assert user_signups['subscription_date'].dtype == 'datetime64[ns]'
today_date = dt.date.today()
# Drop values using filtering
user_signups = user_signups[user_signups['subscription_date'] < today_date]
# Drop values using .drop()
user_signups.drop(user_signups[user_signups['subscription_date'] > today_date].index, inplace = True)
#Hard code dates with upper limit
# Drop values using filtering
user_signups.loc[user_signups['subscription_date'] > today_date, 'subscription_date'] = today_date
# Assert is true
assert user_signups.subscription_date.max().date() <= today_date

#Treating date data - Attempt to infer format of each date - Return NA for rows where conversion failed
birthdays['Birthday'] = pd.to_datetime(birthdays['Birthday'], infer_datetime_format=True, errors = 'coerce')
birthdays['Birthday'] = birthdays['Birthday'].dt.strftime("%d-%m-%Y")


print("\n6. Duplicate values\n")
print("====================")
#How to find duplicate rows? The.duplicated() method 
#subset: List of column names t ocheck fo rduplication.
#keep:Whether to keep first('first'),last('last') or all (False) duplicatevalues.
# Column names to check for duplication
##>column_names = ['first_name','last_name','address']
##>duplicates = height_weight.duplicated(subset = column_names, keep = False)


#Identify Duplicates\n")
# Get duplicates across all columns show true or false for each line duplicated
print(df.info())
duplicates = df.duplicated()
print(duplicates)


# Get duplicate rows
duplicates = df.duplicated()
print(df[duplicates])

#Column names to check for duplication
column_names = ['AGE']
duplicates = df.duplicated(subset = column_names, keep = False)
# Output duplicate values
print("\ndf[duplicates]: \n",df[duplicates])

# sort Output duplicate values by header
print("\nOutput duplicate values: \n",df[duplicates].sort_values(by = 'INDUS'))

# Drop Duplicates\n"
# How to treat duplicate values? The.drop_duplicates() method 
# subset: List of column names to check for duplication. 
# keep:Whethertokeeprst('first'),last('last')orall(False) duplicate values.
# inplace:Drop duplicated rows directly inside Data Frame without creating new object(True)

#Drop duplicates
df.drop_duplicates(inplace = True)

#treat duplicate values - The .groupby() and .agg() methods
# Group method
# Group by column names and produce statistical summaries
column_names = ['first_name','last_name','address']
summaries = {'height': 'max', 'weight': 'mean'}
df = df.groupby(by = column_names).agg(summaries).reset_index()


# Aggregation method
# Make sure aggregation is done
duplicates = df.duplicated(subset = column_names, keep = False)
df[duplicates].sort_values(by = 'first_name')


print("\n7. Missing values")
print("===================================")
# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Convert '?' to NaN
df[df == '?'] = np.nan
# Convert '' to NaN
df[df == ''] = np.nan
#Drop missing data per column
df.ZN.replace(0, np.nan, inplace=True)#ZN is a column
# Print the number of NaNs
print(df.isnull().sum())

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()
# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))



#Imputing missing data - us mean on missing data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imp.fit(X)
# X = imp.transform(X)
imp.fit(df)
X = imp.transform(df)

# Return missing values
airquality = pd.read_csv('airquality.csv')
airquality.isna()
Get summary of missingness 
airquality.isna().sum()

# Isolate missing and complete values aside
missing = airquality[airquality['CO2'].isna()]
complete = airquality[~airquality['CO2'].isna()]
#Describe complete DataFrame
complete.describe()
#Describe missing DataFramee
missing.describe()

#Drop missing data 
boston = pd.read_csv("boston.csv")
boston = boston.dropna()
# Drop missing values
airquality_dropped = airquality.dropna(subset = ['CO2'])
airquality_dropped.head()

# Replacing with statistical measures
co2_mean = airquality['CO2'].mean()
airquality_imputed = airquality.fillna({'CO2': co2_mean})
airquality_imputed.head()







print("\n7. Cross field validation")
print("===================================")

#Cross field validation: The use of multiple fields in a data set to sanity check data integrity = adding columns - across observation - features are columns
sum_classes = flights[['economy_class', 'business_class', 'first_class']].sum(axis = 1)
passenger_equ = sum_classes == flights['total_passengers']
# Find and filter out rows with inconsistent passenger totals
inconsistent_pass = flights[~passenger_equ]
consistent_pass = flights[passenger_equ]




print("\n3. EDA graphical review and EDA Quantative on  Data\n")
print("===================================")
# Data - Data uniformity - EDA visual check
#EX.1 - Create scatter plot
plt.scatter(x = 'Date', y = 'Temperature', data = temperatures)
# Create title, xlabel and ylabel
plt.title('Temperature in Celsius March 2019 - NYC')
plt.xlabel('Dates')
plt.ylabel('Temperature in Celsius')
# Show plot
plt.show()

#EX.2 - missing data
import missingno as msno
import matplotlib.pyplot as plt
# Visualize missingness
msno.matrix(df)
plt.show()





