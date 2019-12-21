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
df = df.loc[(df.criterion == criterion) & (df.max_depth == 'None')][[metric,'min_samples_split','min_samples_leaf']]
pivot = df.pivot('min_samples_split','min_samples_leaf',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('F1' + " " + criterion)

ax = plt.subplot(142)
file = '/DecisionTree/CV_Decision_Tree_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.max_depth == 'None')][[metric,'min_samples_split','min_samples_leaf']]
pivot = df.pivot('min_samples_split','min_samples_leaf',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('RECALL' + " " + criterion)


criterion = 'entropy'
ax = plt.subplot(143)
file = '/DecisionTree/CV_Decision_Tree_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.max_depth == 'None')][[metric,'min_samples_split','min_samples_leaf']]
pivot = df.pivot('min_samples_split','min_samples_leaf',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('F1'  + " " + criterion)

ax = plt.subplot(144)
file = '/DecisionTree/CV_Decision_Tree_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.max_depth == 'None')][[metric,'min_samples_split','min_samples_leaf']]
pivot = df.pivot('min_samples_split','min_samples_leaf',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('RECALL' + " " + criterion)

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
df = df.loc[(df.criterion == criterion) & (df.max_depth == 'None')][[metric,'min_samples_split','min_samples_leaf']]
pivot = df.pivot('min_samples_split','min_samples_leaf',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('F1' + " " + criterion)

ax = plt.subplot(142)
file = '/DecisionTree/CV_OverSampling_Decision_Tree_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.max_depth == 'None')][[metric,'min_samples_split','min_samples_leaf']]
pivot = df.pivot('min_samples_split','min_samples_leaf',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('RECALL' + " " + criterion)


criterion = 'entropy'
ax = plt.subplot(143)
file = '/DecisionTree/CV_OverSampling_Decision_Tree_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.max_depth == 'None')][[metric,'min_samples_split','min_samples_leaf']]
pivot = df.pivot('min_samples_split','min_samples_leaf',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('F1'  + " " + criterion)

ax = plt.subplot(144)
file = '/DecisionTree/CV_OverSampling_Decision_Tree_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.max_depth == 'None')][[metric,'min_samples_split','min_samples_leaf']]
pivot = df.pivot('min_samples_split','min_samples_leaf',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('RECALL' + " " + criterion)

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
df = df.loc[(df.criterion == criterion) & (df.max_depth == 'None')][[metric,'min_samples_split','min_samples_leaf']]
pivot = df.pivot('min_samples_split','min_samples_leaf',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('F1' + " " + criterion)

ax = plt.subplot(142)
file = '/DecisionTree/CV_UnderSampling_Decision_Tree_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.max_depth == 'None')][[metric,'min_samples_split','min_samples_leaf']]
pivot = df.pivot('min_samples_split','min_samples_leaf',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('RECALL' + " " + criterion)


criterion = 'entropy'
ax = plt.subplot(143)
file = '/DecisionTree/CV_UnderSampling_Decision_Tree_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.max_depth == 'None')][[metric,'min_samples_split','min_samples_leaf']]
pivot = df.pivot('min_samples_split','min_samples_leaf',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('F1'  + " " + criterion)

ax = plt.subplot(144)
file = '/DecisionTree/CV_UnderSampling_Decision_Tree_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.criterion == criterion) & (df.max_depth == 'None')][[metric,'min_samples_split','min_samples_leaf']]
pivot = df.pivot('min_samples_split','min_samples_leaf',metric)
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".4f", cbar=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('RECALL' + " " + criterion)

plt.show()

##################################