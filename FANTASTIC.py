##############

    x = df[['Make', 'WarrantyCost']].groupby('Make').mean()
    x = [item for sublist in x.values.tolist() for item in sublist]

    y = df[['Make', 'VehBCost']].groupby('Make').mean()
    y = [item for sublist in y.values.tolist() for item in sublist]

    z = list(dict(df.groupby(['Make']).size()).values())
    z = [float(i)/100 for i in z]
    # z = [50 for i in z]
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
            color=[random.randint(0, 2000) for i in range(len(z))],
            colorscale="Rainbow"
        )
    ))

    fig.update_layout(
        title="FANTASTIC PLOT",
        xaxis=dict(
            title="VehBCost",
            type='log'
        ),

        yaxis=dict(
            title="WarrantyCost",
        )
    )
    fig.show()

    ##############