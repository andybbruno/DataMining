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
df = df[[metric,'learning_rate','n_estimators', 'std_dev']]
pivot = df.pivot('learning_rate','n_estimators',metric)
labels = pivot.as_matrix().astype(str)
x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1
sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "F1\quad" + "}$")



ax = plt.subplot(122)
file = '/AdaBoost/CV_AdaBoost_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df[[metric,'learning_rate','n_estimators', 'std_dev']]
pivot = df.pivot('learning_rate','n_estimators',metric)
labels = pivot.as_matrix().astype(str)
x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1
sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "RECALL\quad" + "}$")

plt.show()

##################################

fig = plt.figure()
fig.suptitle('ADABOOST + OVERSAMPLING', fontsize=14, fontweight='bold')

ax = plt.subplot(121)
file = '/AdaBoost/CV_OverSampling_AdaBoost_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df[[metric,'learning_rate','n_estimators', 'std_dev']]
pivot = df.pivot('learning_rate','n_estimators',metric)
labels = pivot.as_matrix().astype(str)
x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1
sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "F1\quad" + "}$")


ax = plt.subplot(122)
file = '/AdaBoost/CV_OverSampling_AdaBoost_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df[[metric,'learning_rate','n_estimators', 'std_dev']]
pivot = df.pivot('learning_rate','n_estimators',metric)
labels = pivot.as_matrix().astype(str)
x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1
sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "RECALL\quad" + "}$")

plt.show()

##################################

fig = plt.figure()
fig.suptitle('ADABOOST + UNDERSAMPLING', fontsize=14, fontweight='bold')

ax = plt.subplot(121)
file = '/AdaBoost/CV_UnderSampling_AdaBoost_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df[[metric,'learning_rate','n_estimators', 'std_dev']]
pivot = df.pivot('learning_rate','n_estimators',metric)
labels = pivot.as_matrix().astype(str)
x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1
sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "F1\quad" + "}$")

ax = plt.subplot(122)
file = '/AdaBoost/CV_UnderSampling_AdaBoost_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df[[metric,'learning_rate','n_estimators', 'std_dev']]
pivot = df.pivot('learning_rate','n_estimators',metric)
labels = pivot.as_matrix().astype(str)
x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1
sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "RECALL\quad" + "}$")


plt.show()