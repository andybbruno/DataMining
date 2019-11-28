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
training = pd.read_csv('training_cleaned.csv')
training.drop(training.columns[0], axis=1, inplace=True)
training.drop(columns=['RefId', 'PurchDate', 'Make', 'IsOnlineSale', 'Transmission',
                       'Nationality', 'IsBase', 'WheelTypeID'], inplace=True)


# READ AND DROP COLUMNS
test = pd.read_csv('test_cleaned.csv')
test.drop(test.columns[0], axis=1, inplace=True)
test.drop(columns=['RefId', 'PurchDate', 'Make', 'IsOnlineSale', 'Transmission',
                   'Nationality', 'IsBase', 'WheelTypeID'], inplace=True)

training['isTraining'] = 1
test['isTraining'] = 0
        
combined = pd.concat([training, test])
for col in list(training):
    coltype = str(training[col].dtype)
    if (coltype == "object"):
        df = pd.get_dummies(combined[col])
        combined = combined.drop(col, axis=1)
        tmp = pd.concat([combined, df], axis=1)

train_df = tmp[tmp['isTraining'] == 1]
test_df = tmp[tmp['isTraining'] == 0]
train = train_df.drop(['isTraining'], axis=1)
test = test_df.drop(['isTraining'], axis=1)

train.to_csv('training_cleaned_ONEHOT.csv')
test.to_csv('test_cleaned_ONEHOT.csv')


# y = df['IsBadBuy']
# X = df.drop(columns=['IsBadBuy'])

# #HASHING
# for col in list(X):
#     coltype = str(X[col].dtype)
#     if (coltype == "object"):
#         lst = [hash(col + " " + x) for x in X[col]]
#         dframe = pd.DataFrame(lst)
#         X = X.drop(col, axis=1)
#         X[col] = lst


# # ONE-HOT TRAINING
# colNames={}
# for col in list(X):
#     coltype=str(X[col].dtype)
#     if (coltype == "object"):
#         one_hot=pd.get_dummies(X[col])
#         columns_=pd.get_dummies(X[col]).columns
#         new_columns_names=[str(col + x) for x in columns_]
#         renamed=dict(zip(columns_, new_columns_names))
#         one_hot.rename(columns = renamed, inplace = True)
#         X=X.drop(col, axis = 1)
#         X=X.join(one_hot)
#         colNames.update({col: new_columns_names})

# print(X.shape)

# # READ AND DROP COLUMNS
# df=pd.read_csv('test_cleaned.csv')
# df.drop(df.columns[0], axis = 1, inplace = True)
# df.drop(columns = ['RefId', 'PurchDate', 'Make', 'IsOnlineSale', 'Transmission',
#                  'Nationality', 'IsBase', 'WheelTypeID'], inplace = True)


# y=df['IsBadBuy']
# X=df.drop(columns = ['IsBadBuy'])

# # ONE-HOT TEST
# for col in list(X):
#     coltype=str(X[col].dtype)
#     if (coltype == "object"):
#         columns_=pd.get_dummies(X[col]).columns
#         new_columns_names=[str(col + x) for x in columns_]
#         known_categories=colNames[col]
#         car_type=pd.Series(new_columns_names)
#         car_type=pd.Categorical(car_type, categories = known_categories)
#         one_hot=pd.get_dummies(car_type)
#         X=X.drop(col, axis = 1)
#         X=X.join(one_hot)

# print(X.shape)
