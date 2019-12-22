import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

fig.suptitle('KNN', fontsize=14, fontweight='bold')

weights = 'uniform'

ax = plt.subplot(141)
file = '/KNN/CV_KNN_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df.loc[(df.weights == weights)][[metric,'n_neighbors','std_dev']]
x = df['n_neighbors']
y = df[metric]
dy = df['std_dev']
ax.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0, markersize=8)
ax.plot(x,y,'m-',linewidth=4)
ax.title.set_text('F1 ' + "<" + weights + ">")

ax = plt.subplot(142)
file = '/KNN/CV_KNN_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.weights == weights)][[metric,'n_neighbors','std_dev']]
x = df['n_neighbors']
y = df[metric]
dy = df['std_dev']
ax.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0, markersize=8)
ax.plot(x,y,'c-',linewidth=4)
ax.title.set_text('RECALL ' + "<" + weights + ">")


#####################################

weights = 'distance'

ax = plt.subplot(143)
file = '/KNN/CV_KNN_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df.loc[(df.weights == weights)][[metric,'n_neighbors','std_dev']]
x = df['n_neighbors']
y = df[metric]
dy = df['std_dev']
ax.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0, markersize=8)
ax.plot(x,y,'m-',linewidth=4)
ax.title.set_text('F1 ' + "<" + weights + ">")

ax = plt.subplot(144)
file = '/KNN/CV_KNN_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.weights == weights)][[metric,'n_neighbors','std_dev']]
x = df['n_neighbors']
y = df[metric]
dy = df['std_dev']
ax.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0, markersize=8)
ax.plot(x,y,'c-',linewidth=4)
ax.title.set_text('RECALL ' + "<" + weights + ">")

plt.show()