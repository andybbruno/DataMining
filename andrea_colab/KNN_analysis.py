import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots



weights = 'uniform'

fig = go.Figure()
file = '/KNN/CV_KNN_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df.loc[(df.weights == weights)][[metric, 'n_neighbors', 'std_dev']]
x = df['n_neighbors']
y = df[metric]
dy = df['std_dev']

n = [str(round(y.iloc[j],3))[:5] + "\n+/- " + str(dy.iloc[j])[:5] for j in range(len(dy))]

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    line = dict(width=8)
))

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    error_y=dict(
        type='data',
        array=dy,
        color='lightgray',
        thickness=3,
        width=3),
    marker=dict(color='black', size=12)
))

for i, txt in enumerate(n):
    if i == 3:
        fig.add_annotation(
            go.layout.Annotation(
                xref='x', yref='y',
                ax=40,
                ay= -100,
                x=x.iloc[i],
                y=y.iloc[i],
                text=txt,
                arrowsize=4)
        )
    else:
        fig.add_annotation(
            go.layout.Annotation(
                xref='x', yref='y',
                ax=75,
                ay= ((i%2)+1) * -75,
                x=x.iloc[i],
                y=y.iloc[i],
                text=txt,
                arrowsize=4)
        )
fig.update_layout(
    font=dict(size=20),
    template='plotly_white',
    showlegend=False, 
    title={
        'text': "KNN <uniform> F1",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
)

fig.show()


fig = go.Figure()
file = '/KNN/CV_KNN_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.weights == weights)][[metric, 'n_neighbors', 'std_dev']]
x = df['n_neighbors']
y = df[metric]
dy = df['std_dev']

n = [str(round(y.iloc[j],3))[:5] + "\n+/- " + str(dy.iloc[j])[:5] for j in range(len(dy))]

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    line = dict(color='orangered', width=8)
))

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    error_y=dict(
        type='data',
        array=dy,
        color='lightgray',
        thickness=3,
        width=3),
    marker=dict(color='black', size=12)
))

for i, txt in enumerate(n):
    fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=75,
            ay= ((i%2)+1) * -75,
            x=x.iloc[i],
            y=y.iloc[i],
            text=txt,
            arrowsize=4)
    )
fig.update_layout(
    font=dict(size=20),
    template='plotly_white',
    showlegend=False, 
    title={
        'text': "KNN <uniform> RECALL",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
)

fig.show()

#####################################

weights = 'distance'

fig = go.Figure()
file = '/KNN/CV_KNN_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
df = df.loc[(df.weights == weights)][[metric, 'n_neighbors', 'std_dev']]
x = df['n_neighbors']
y = df[metric]
dy = df['std_dev']

n = [str(round(y.iloc[j],3))[:5] + "\n+/- " + str(dy.iloc[j])[:5] for j in range(len(dy))]

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    line = dict(width=8)
))

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    error_y=dict(
        type='data',
        array=dy,
        color='lightgray',
        thickness=3,
        width=3),
    marker=dict(color='black', size=12)
))

for i, txt in enumerate(n):
    fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=75,
            ay= ((i%2)+1) * -75,
            x=x.iloc[i],
            y=y.iloc[i],
            text=txt,
            arrowsize=4)
    )
fig.update_layout(
    font=dict(size=20),
    template='plotly_white',
    showlegend=False, 
    title={
        'text': "KNN <distance> F1",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
)

fig.show()


fig = go.Figure()
file = '/KNN/CV_KNN_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
df = df.loc[(df.weights == weights)][[metric, 'n_neighbors', 'std_dev']]
x = df['n_neighbors']
y = df[metric]
dy = df['std_dev']

n = [str(round(y.iloc[j],3))[:5] + "\n+/- " + str(dy.iloc[j])[:5] for j in range(len(dy))]

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    line = dict(color='orangered', width=8)
))

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    error_y=dict(
        type='data',
        array=dy,
        color='lightgray',
        thickness=3,
        width=3),
    marker=dict(color='black', size=12)
))

for i, txt in enumerate(n):
    fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=75,
            ay= ((i%2)+1) * -75,
            x=x.iloc[i],
            y=y.iloc[i],
            text=txt,
            arrowsize=4)
    )
fig.update_layout(
    font=dict(size=20),
    template='plotly_white',
    showlegend=False, 
    title={
        'text': "KNN <distance> RECALL",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
)

fig.show()