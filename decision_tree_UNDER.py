"""# Decision Tree + UnderSampling"""

# Commented out IPython magic to ensure Python compatibility.
# %reset -f
# %matplotlib inline
import sys
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from sklearn import tree
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import  RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import numbers
import warnings
from collections import defaultdict
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from IPython.display import Image
warnings.filterwarnings('ignore')

#####################################################################
#                       DATA LOADING                                #
#####################################################################
df = pd.read_csv('training_cleaned.csv')
df.drop(df.columns[0], axis=1, inplace=True)

df.drop(columns=['RefId',
                 'PurchDate',
                 'Make',
                 'Model'], inplace=True)

for col in list(df):
  coltype = str(df[col].dtype)
  if (coltype == "object"):
    lst = list(df[col].unique())
    le = preprocessing.LabelEncoder()
    le.fit(lst)
    tmp = le.transform(df[col]).tolist()
    df.drop(columns=[col], inplace = True)
    df[col] = tmp

#####################################################################

df_NEW = df
y = df_NEW['IsBadBuy']
X = df_NEW.drop(columns=['IsBadBuy','IsOnlineSale','Transmission','Nationality','IsBase','WheelTypeID','SubModel'])
attributes = [col for col in X.columns if col != 'class']


rus = RandomUnderSampler(random_state=0)
X, y = rus.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
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

tm = pd.DataFrame(res, columns=['name','importance'])
tm.sort_values('importance', ascending=False, inplace=True)
#tm.drop(tm.columns[0], axis=1, inplace=True)
#print(tm)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

#TESTING ON TEST SET
 
df_test = pd.read_csv('test_cleaned.csv')
df_test.drop(df_test.columns[0], axis=1, inplace=True)
df_test.drop(columns=['RefId',
                 'PurchDate',
                 'Make',
                 'Model'], inplace=True)

for col in list(df_test):
  coltype = str(df_test[col].dtype)
  if (coltype == "object"):
    lst = list(df_test[col].unique())
    le = preprocessing.LabelEncoder()
    le.fit(lst)
    tmp = le.transform(df_test[col]).tolist()
    df_test.drop(columns=[col], inplace = True)
    df_test[col] = tmp

y_ = df_test['IsBadBuy']
X_ = df_test.drop(columns=['IsBadBuy','IsOnlineSale','Transmission','Nationality','IsBase','WheelTypeID','SubModel'])

y_pred = clf.predict(X_)
print(classification_report(y_, y_pred))

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=attributes,
                                class_names=['Bad Buy' if x == 1 else 'Good Buy' for x in clf.classes_],  
                                filled=True, rounded=True,  
                                special_characters=True,
                                max_depth=4
                                )  
#graph = pydotplus.graph_from_dot_data(dot_data)  
#Image(graph.create_png())