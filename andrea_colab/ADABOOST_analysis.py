import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

fig = plt.figure()
fig.suptitle('ADABOOST', fontsize=14, fontweight='bold')

ax = plt.subplot(121)
file = '/AdaBoost/CV_AdaBoost_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df[[metric,'learning_rate','n_estimators']]
pivot = df.pivot('learning_rate','n_estimators',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('F1')


ax = plt.subplot(122)
file = '/AdaBoost/CV_AdaBoost_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df[[metric,'learning_rate','n_estimators']]
pivot = df.pivot('learning_rate','n_estimators',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('RECALL')

plt.show()

##################################

fig = plt.figure()
fig.suptitle('ADABOOST + OVERSAMPLING', fontsize=14, fontweight='bold')

ax = plt.subplot(121)
file = '/AdaBoost/CV_OverSampling_AdaBoost_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df[[metric,'learning_rate','n_estimators']]
pivot = df.pivot('learning_rate','n_estimators',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('F1')


ax = plt.subplot(122)
file = '/AdaBoost/CV_OverSampling_AdaBoost_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df[[metric,'learning_rate','n_estimators']]
pivot = df.pivot('learning_rate','n_estimators',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('RECALL')

plt.show()

##################################

fig = plt.figure()
fig.suptitle('ADABOOST + UNDERSAMPLING', fontsize=14, fontweight='bold')

ax = plt.subplot(121)
file = '/AdaBoost/CV_UnderSampling_AdaBoost_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df[[metric,'learning_rate','n_estimators']]
pivot = df.pivot('learning_rate','n_estimators',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('F1')


ax = plt.subplot(122)
file = '/AdaBoost/CV_UnderSampling_AdaBoost_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df[[metric,'learning_rate','n_estimators']]
pivot = df.pivot('learning_rate','n_estimators',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('RECALL')


plt.show()