import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go



kwargs = dict(ecolor='k', color='k', capsize=2,
              elinewidth=1.1, linewidth=0.6, ms=7)

fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

fig.suptitle('RANDOM FOREST', fontsize=14, fontweight='bold')

fig = go.Figure()
file = '/RandomForest/CV_RandomForest_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
x = df['n_estimators']
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
    marker=dict(color='black', size=15)
))


for i, txt in enumerate(n):
    if i == 0:
        fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=-75,
            ay=-75,
            x=x.iloc[0],
            y=y.iloc[0],
            text=n[0],
            arrowsize=4)
        )
    elif i == 1:
        fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=75,
            ay=75,
            x=x.iloc[1],
            y=y.iloc[1],
            text=n[1],
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
        'text': "RANDOM FOREST F1",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
)

fig.show()


fig = go.Figure()
file = '/RandomForest/CV_RandomForest_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
x = df['n_estimators']
y = df[metric]
dy = df['std_dev']
n = [str(round(y.iloc[j],3))[:5] + "\n+/- " + str(dy.iloc[j])[:5] for j in range(len(dy))]

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    line = dict(color='orangered',  width=8)
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
    marker=dict(color='black', size=15)
))


for i, txt in enumerate(n):
    if i == 0:
        fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=-75,
            ay=-75,
            x=x.iloc[0],
            y=y.iloc[0],
            text=n[0],
            arrowsize=4)
        )
    elif i == 1:
        fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=75,
            ay=75,
            x=x.iloc[1],
            y=y.iloc[1],
            text=n[1],
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
        'text': "RANDOM FOREST RECALL",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
)

fig.show()



################################

fig = plt.figure()
fig.canvas.manager.full_screen_toggle()


fig = go.Figure()
file = '/RandomForest/CV_OverSampling_Random_Forest_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
x = df['n_estimators']
y = df[metric]
dy = df['std_dev']
n = [str(round(y.iloc[j],3))[:5] + "\n+/- " + str(dy.iloc[j])[:5] for j in range(len(dy))]

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    line = dict( width=8)
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
    marker=dict(color='black', size=15)
))


for i, txt in enumerate(n):
    if i == 0:
        fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=-75,
            ay=-75,
            x=x.iloc[0],
            y=y.iloc[0],
            text=n[0],
            arrowsize=4)
        )
    elif i == 1:
        fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=75,
            ay=75,
            x=x.iloc[1],
            y=y.iloc[1],
            text=n[1],
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
        'text': "RANDOM FOREST OVERSAMPLING F1",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
)

fig.show()



fig = go.Figure()
file = '/RandomForest/CV_OverSampling_Random_Forest_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
x = df['n_estimators']
y = df[metric]
dy = df['std_dev']
n = [str(round(y.iloc[j],3))[:5] + "\n+/- " + str(dy.iloc[j])[:5] for j in range(len(dy))]

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    line = dict(color='orangered',  width=8)
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
    marker=dict(color='black', size=15)
))


for i, txt in enumerate(n):
    if i == 0:
        fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=-75,
            ay=-75,
            x=x.iloc[0],
            y=y.iloc[0],
            text=n[0],
            arrowsize=4)
        )
    elif i == 1:
        fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=75,
            ay=75,
            x=x.iloc[1],
            y=y.iloc[1],
            text=n[1],
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
        'text': "RANDOM FOREST OVERSAMPLING RECALL",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
)

fig.show()

################################

fig = go.Figure()
file = '/RandomForest/CV_UnderSampling_Random_Forest_f1.csv'
file = os.getcwd() + file
metric = 'f1'
df = pd.read_csv(file)
x = df['n_estimators']
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
    marker=dict(color='black', size=15)
))


for i, txt in enumerate(n):
    if i == 0:
        fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=-75,
            ay=-75,
            x=x.iloc[0],
            y=y.iloc[0],
            text=n[0],
            arrowsize=4)
        )
    elif i == 1:
        fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=75,
            ay=75,
            x=x.iloc[1],
            y=y.iloc[1],
            text=n[1],
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
        'text': "RANDOM FOREST UNDERSAMPLING F1",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
)

fig.show()



fig = go.Figure()
file = '/RandomForest/CV_UnderSampling_Random_Forest_recall.csv'
file = os.getcwd() + file
metric = 'recall'
df = pd.read_csv(file)
x = df['n_estimators']
y = df[metric]
dy = df['std_dev']
n = [str(round(y.iloc[j],3))[:5] + "\n+/- " + str(dy.iloc[j])[:5] for j in range(len(dy))]

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    line = dict(color='orangered',  width=8)
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
    marker=dict(color='black', size=15)
))


for i, txt in enumerate(n):
    if i == 0:
        fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=-75,
            ay=-75,
            x=x.iloc[0],
            y=y.iloc[0],
            text=n[0],
            arrowsize=4)
        )
    elif i == 1:
        fig.add_annotation(
        go.layout.Annotation(
            xref='x', yref='y',
            ax=75,
            ay=75,
            x=x.iloc[1],
            y=y.iloc[1],
            text=n[1],
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
        'text': "RANDOM FOREST UNDERSAMPLING RECALL",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
)

fig.show()

################################

