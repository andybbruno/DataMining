import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


df = pd.read_csv('training_cleaned.csv')
df.drop(df.columns[0], axis=1, inplace=True)

df.drop(columns=['RefId', 'PurchDate', 'Make',
                 'Model', 'IsOnlineSale', 'Transmission',
                 'Nationality', 'IsBase', 'WheelTypeID', 'SubModel'], inplace=True)

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


attributes = [col for col in df.columns if col != 'IsBadBuy']
X = df[attributes].values
y = df['IsBadBuy']


# AFTER 100 CV

# Model with rank: 1
# Mean validation score: 0.877 (std: 0.000)
# Parameters: {'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 3}

# Model with rank: 1
# Mean validation score: 0.877 (std: 0.000)
# Parameters: {'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 3}

# Model with rank: 3
# Mean validation score: 0.877 (std: 0.000)
# Parameters: {'min_samples_split': 10, 'min_samples_leaf': 50, 'max_depth': 2}

# Model with rank: 3
# Mean validation score: 0.877 (std: 0.000)
# Parameters: {'min_samples_split': 30, 'min_samples_leaf': 50, 'max_depth': 2}



X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=100,
                                                    stratify=y)
                                                    
clf = DecisionTreeClassifier(
    criterion='gini',
    min_samples_split=5,
    min_samples_leaf=1,
    max_depth=3
)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_tr = clf.predict(X_train)

print(classification_report(y_test, y_pred))


res = []
for col, imp in zip(attributes, clf.feature_importances_):
    res.append([col, imp])
tm = pd.DataFrame(res, columns=['name', 'importance'])
tm.sort_values('importance', ascending=False, inplace=True)
print(tm)
