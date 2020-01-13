import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

fig.suptitle('DECISION TREE', fontsize=14, fontweight='bold')

criterion = 'gini'

ax = plt.subplot(141)
file = '/DecisionTree/CV_Decision_Tree_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.min_samples_leaf == 1)][[metric,'min_samples_split','max_depth', 'std_dev']]
pivot = df.pivot('max_depth','min_samples_split',metric)
labels = pivot.as_matrix().astype(str)

x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1

sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "F1\quad" + criterion + "}$")

ax = plt.subplot(142)
file = '/DecisionTree/CV_Decision_Tree_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.min_samples_leaf == 1)][[metric,'min_samples_split','max_depth', 'std_dev']]
pivot = df.pivot('max_depth','min_samples_split',metric)
labels = pivot.as_matrix().astype(str)

x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1

sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "RECALL\quad" + criterion + "}$")


criterion = 'entropy'
ax = plt.subplot(143)
file = '/DecisionTree/CV_Decision_Tree_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.min_samples_leaf == 1)][[metric,'min_samples_split','max_depth', 'std_dev']]
pivot = df.pivot('max_depth','min_samples_split',metric)
labels = pivot.as_matrix().astype(str)

x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1

sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "F1\quad" + criterion + "}$")

ax = plt.subplot(144)
file = '/DecisionTree/CV_Decision_Tree_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.min_samples_leaf == 1)][[metric,'min_samples_split','max_depth', 'std_dev']]
pivot = df.pivot('max_depth','min_samples_split',metric)
labels = pivot.as_matrix().astype(str)

x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1

sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "RECALL\quad" + criterion + "}$")

# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

##################################

fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

fig.suptitle('DECISION TREE OVERSAMPLING', fontsize=14, fontweight='bold')

criterion = 'gini'

ax = plt.subplot(141)
file = '/DecisionTree/CV_OverSampling_Decision_Tree_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.min_samples_leaf == 1)][[metric,'min_samples_split','max_depth', 'std_dev']]
pivot = df.pivot('max_depth','min_samples_split',metric)
labels = pivot.as_matrix().astype(str)

x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1

sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "F1\quad" + criterion + "}$")

ax = plt.subplot(142)
file = '/DecisionTree/CV_OverSampling_Decision_Tree_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.min_samples_leaf == 1)][[metric,'min_samples_split','max_depth', 'std_dev']]
pivot = df.pivot('max_depth','min_samples_split',metric)
labels = pivot.as_matrix().astype(str)

x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1

sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "RECALL\quad" + criterion + "}$")


criterion = 'entropy'
ax = plt.subplot(143)
file = '/DecisionTree/CV_OverSampling_Decision_Tree_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.min_samples_leaf == 1)][[metric,'min_samples_split','max_depth', 'std_dev']]
pivot = df.pivot('max_depth','min_samples_split',metric)
labels = pivot.as_matrix().astype(str)

x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1

sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "F1\quad" + criterion + "}$")

ax = plt.subplot(144)
file = '/DecisionTree/CV_OverSampling_Decision_Tree_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.min_samples_leaf == 1)][[metric,'min_samples_split','max_depth', 'std_dev']]
pivot = df.pivot('max_depth','min_samples_split',metric)
labels = pivot.as_matrix().astype(str)

x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1

sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "RECALL\quad" + criterion + "}$")

# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

##################################

fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

fig.suptitle('DECISION TREE UNDERSAMPLING', fontsize=14, fontweight='bold')

criterion = 'gini'

ax = plt.subplot(141)
file = '/DecisionTree/CV_UnderSampling_Decision_Tree_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.min_samples_leaf == 1)][[metric,'min_samples_split','max_depth', 'std_dev']]
pivot = df.pivot('max_depth','min_samples_split',metric)
labels = pivot.as_matrix().astype(str)

x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1

sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "F1\quad" + criterion + "}$")

ax = plt.subplot(142)
file = '/DecisionTree/CV_UnderSampling_Decision_Tree_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.min_samples_leaf == 1)][[metric,'min_samples_split','max_depth', 'std_dev']]
pivot = df.pivot('max_depth','min_samples_split',metric)
labels = pivot.as_matrix().astype(str)

x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1

sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "RECALL\quad" + criterion + "}$")


criterion = 'entropy'
ax = plt.subplot(143)
file = '/DecisionTree/CV_UnderSampling_Decision_Tree_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.min_samples_leaf == 1)][[metric,'min_samples_split','max_depth', 'std_dev']]
pivot = df.pivot('max_depth','min_samples_split',metric)
labels = pivot.as_matrix().astype(str)

x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1

sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "F1\quad" + criterion + "}$")

ax = plt.subplot(144)
file = '/DecisionTree/CV_UnderSampling_Decision_Tree_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.min_samples_leaf == 1)][[metric,'min_samples_split','max_depth', 'std_dev']]
pivot = df.pivot('max_depth','min_samples_split',metric)
labels = pivot.as_matrix().astype(str)

x = 0
for i in range(0, pivot.shape[0]):
    for j in range(0, pivot.shape[1]):
        labels[i,j] = str(labels[i,j])[:5] + "\n+/- " + "{0:.3f}".format(df['std_dev'].iloc[x])
        x += 1

sns.heatmap(pivot, cmap="YlGnBu", annot=labels, fmt="", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text(r"$\bf{" + "RECALL\quad" + criterion + "}$")

# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

##################################