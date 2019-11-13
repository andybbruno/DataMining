"""# Decision Tree + OverSampling"""

# Commented out IPython magic to ensure Python compatibility.
# %reset -f
# %matplotlib inline
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#####################################################################
#                       DATA LOADING                                #
#####################################################################
df = pd.read_csv('training_cleaned.csv')
df.drop(df.columns[0], axis=1, inplace=True)

df.drop(columns=['RefId',
                 'PurchDate',
                 'Make',
                 'Model'], inplace=True)

uniqueVals = {}
for col in list(df):
    coltype = str(df[col].dtype)
    if (coltype == "object"):
        lst = list(df[col].unique())
        uniqueVals[col] = lst
        le = preprocessing.LabelEncoder()
        le.fit(lst)
        tmp = le.transform(df[col]).tolist()
        df.drop(columns=[col], inplace=True)
        df[col] = tmp

#####################################################################

df_NEW = df
y = df_NEW['IsBadBuy']
X = df_NEW.drop(columns=['IsBadBuy', 'IsOnlineSale', 'Transmission',
                         'Nationality', 'IsBase', 'WheelTypeID', 'SubModel'])
attributes = [col for col in X.columns if col != 'class']


rus = RandomOverSampler()
X, y = rus.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=100,
                                                    stratify=y)

clf = tree.DecisionTreeClassifier(criterion='gini',
                                  max_depth=None,
                                  min_samples_split=2,
                                  min_samples_leaf=1)

clf.fit(X_train, y_train)

res = []
for col, imp in zip(attributes, clf.feature_importances_):
    res.append([col, imp])
tm = pd.DataFrame(res, columns=['name', 'importance'])
tm.sort_values('importance', ascending=False, inplace=True)
print(tm)

y_pred = clf.predict(X_test)
print("############################## training.csv ##############################")
print(classification_report(y_test, y_pred))

# TESTING ON TEST SET

df_test = pd.read_csv('test_cleaned.csv')
df_test.drop(df_test.columns[0], axis=1, inplace=True)
ids = pd.DataFrame(df_test['RefId'])
df_test.drop(columns=['RefId',
                      'PurchDate',
                      'Make',
                      'Model'], inplace=True)

for col in list(df_test):
    coltype = str(df_test[col].dtype)
    if (coltype == "object"):
        lst = uniqueVals[col]
        le = preprocessing.LabelEncoder()
        le.fit(lst)
        tmp = le.transform(df_test[col]).tolist()
        # print(df_test[col].head())
        df_test.drop(columns=[col], inplace=True)
        df_test[col] = tmp
        #print("@@@" , df_test[col].head())


y = df_test['IsBadBuy']
X = df_test.drop(columns=['IsBadBuy', 'IsOnlineSale', 'Transmission',
                          'Nationality', 'IsBase', 'WheelTypeID', 'SubModel'])


y_pred = clf.predict(X)
print("############################## test.csv ##############################")
print(classification_report(y, y_pred))

# ids.insert(column="IsBadBuy", value=y_pred, loc=1)

#ids.to_csv("RESULTS.csv", index=False)
#graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())
