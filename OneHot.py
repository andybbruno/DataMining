import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# READ AND DROP COLUMNS
df = pd.read_csv('training_cleaned.csv')
df.drop(df.columns[0], axis=1, inplace=True)
df.drop(columns=['RefId', 'PurchDate', 'Make', 'IsOnlineSale', 'Transmission',
                 'Nationality', 'IsBase', 'WheelTypeID', 'Model'], inplace=True)


y = df['IsBadBuy']
X = df.drop(columns=['IsBadBuy'])

# ONE-HOT TRAINING
colNames = {}
for col in list(X):
    coltype = str(X[col].dtype)
    if (coltype == "object"):
        one_hot = pd.get_dummies(X[col])
        columns_ = pd.get_dummies(X[col]).columns
        new_columns_names = [str(col + x) for x in columns_]
        renamed = dict(zip(columns_, new_columns_names))
        one_hot.rename(columns=renamed, inplace=True)
        X = X.drop(col, axis=1)
        X = X.join(one_hot)
        colNames.update({col: new_columns_names})

print(X.shape)

# READ AND DROP COLUMNS
df = pd.read_csv('test_cleaned.csv')
df.drop(df.columns[0], axis=1, inplace=True)
df.drop(columns=['RefId', 'PurchDate', 'Make', 'IsOnlineSale', 'Transmission',
                 'Nationality', 'IsBase', 'WheelTypeID', 'Model'], inplace=True)


y = df['IsBadBuy']
X = df.drop(columns=['IsBadBuy'])

# ONE-HOT TEST
for col in list(X):
    coltype = str(X[col].dtype)
    if (coltype == "object"):
        columns_ = pd.get_dummies(X[col]).columns
        new_columns_names = [str(col + x) for x in columns_]  
        known_categories = colNames[col]
        car_type = pd.Series(new_columns_names)
        car_type = pd.Categorical(car_type, categories=known_categories)
        one_hot = pd.get_dummies(car_type)
        X = X.drop(col, axis=1)
        X = X.join(one_hot)

print(X.shape)