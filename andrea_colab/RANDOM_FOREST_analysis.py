import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


kwargs = dict(ecolor='k', color='k', capsize=2,
              elinewidth=1.1, linewidth=0.6, ms=7)

fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

fig.suptitle('RANDOM FOREST', fontsize=14, fontweight='bold')

ax = plt.subplot(121)
file = '/RandomForest/CV_RandomForest_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
x = df['n_estimators']
y = df[metric]
dy = df['std_dev']
ax.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0, markersize=8)
ax.plot(x,y,'m-',linewidth=4)
ax.title.set_text('F1')

ax = plt.subplot(122)
file = '/RandomForest/CV_RandomForest_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
x = df['n_estimators']
y = df[metric]
dy = df['std_dev']
ax.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0,markersize=8)
ax.plot(x,y,'c-',linewidth=4)
ax.title.set_text('RECALL')

plt.show()


################################

fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

fig.suptitle('RANDOM FOREST OVERSAMPLING', fontsize=14, fontweight='bold')

ax = plt.subplot(121)
file = '/RandomForest/CV_OverSampling_Random_Forest_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
x = df['n_estimators']
y = df[metric]
dy = df['std_dev']
ax.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0,markersize=8)
ax.plot(x,y,'m-',linewidth=4)
ax.title.set_text('F1')

ax = plt.subplot(122)
file = '/RandomForest/CV_OverSampling_Random_Forest_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
x = df['n_estimators']
y = df[metric]
dy = df['std_dev']
ax.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0,markersize=8)
ax.plot(x,y,'c-',linewidth=4)
ax.title.set_text('RECALL')

plt.show()

################################

fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

fig.suptitle('RANDOM FOREST OVERSAMPLING', fontsize=14, fontweight='bold')

ax = plt.subplot(121)
file = '/RandomForest/CV_UnderSampling_Random_Forest_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
x = df['n_estimators']
y = df[metric]
dy = df['std_dev']
ax.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0,markersize=8)
ax.plot(x,y,'m-',linewidth=4)
ax.title.set_text('F1')

ax = plt.subplot(122)
file = '/RandomForest/CV_UnderSampling_Random_Forest_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
x = df['n_estimators']
y = df[metric]
dy = df['std_dev']
ax.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0,markersize=8)
ax.plot(x,y,'c-',linewidth=4)
ax.title.set_text('RECALL')

plt.show()

################################

