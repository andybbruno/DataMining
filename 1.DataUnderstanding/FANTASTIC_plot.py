import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.express as px
from plotly import graph_objects as go
import seaborn as sns
#import missingno as msno

from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from scipy.stats.stats import pearsonr

df = pd.read_csv('kids_train_cleaned.csv')

import random

x = df[['Make', 'WarrantyCost']].groupby('Make').mean()
x = [item for sublist in x.values.tolist() for item in sublist]

y = df[['Make', 'VehBCost']].groupby('Make').mean()
y = [item for sublist in y.values.tolist() for item in sublist]

z = list(dict(df.groupby(['Make']).size()).values())
z = [float(i)/100 for i in z]
z = [3 if i < 3 else i for i in z]
names = list(dict(df.groupby(['Make']).size()).keys())

# Create figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers+text',
    text=names,
    textposition="middle center",
    marker=dict(
        size=z,
        color=[random.randint(0, 100) for i in range(len(z))],
        colorscale="tealrose"
    )
))

fig.update_layout(
    title="FANTASTIC PLOT",
    xaxis=dict(
        title="WarrantyCost",
        type='log'
    ),

    yaxis=dict(
        title="VehBCost",
    )
)
fig.show()