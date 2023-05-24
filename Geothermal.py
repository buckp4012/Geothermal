import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import dash

from dash import dcc
from dash import html
import math
from dash import dash_table



df = pd.read_excel('New Geo Tracker.xlsx')
customers = df['Customer'].unique()
df['INSTALL DATE'] = pd.to_datetime(df['INSTALL DATE']) # Convert Install Date column to datetime format
df['INSTALL DATE'] = df['INSTALL DATE'].dt.strftime('%m-%d-%Y') # Format date as mm-dd-yyyy

max_runtimes = df.groupby('Customer')['Runtime'].max()

# Create a pivot table from the DataFrame
pivot_table = pd.pivot_table(df, index=['Customer'], values=['Installs', 'Runtime','R/NR/Waiting', 'INSTALL DATE'], aggfunc={'Runtime': 'mean','R/NR/Waiting': lambda x: sum(x == 'NR'),'Installs': 'nunique', 'INSTALL DATE': 'max'}).reset_index()
  
# Add the max_runtimes variable to the pivot table
pivot_table['Runtime'] = pivot_table['Runtime'].round()
pivot_table = pivot_table.merge(max_runtimes, on='Customer', how='left')
pivot_table = pivot_table.rename(columns={'Runtime_x': 'Average Runtime'})
pivot_table = pivot_table.rename(columns={'Runtime_y': 'Max Runtime'})
# Rename columns
pivot_table = pivot_table.rename(columns={'R/NR/Waiting': 'Failures'})
pivot_table = pivot_table.rename(columns={'INSTALL DATE': 'First Install'})



# Reorder the columns of the pivot table
#pivot_table = pivot_table[['Customer', 'Installs', 'Failures', 'Average Runtime', 'Max Runtime', 'First Install']]
data=df.groupby('Customer').filter(lambda x: (x['R/NR/Waiting'] == 'NR').sum() >= 2)
customers = data['Customer'].unique()
options = [{'label': 'All Customers', 'value': 'all'}] + \
          [{'label': customer, 'value': customer} for customer in customers]
app = dash.Dash(__name__)
server = app.server
dash_table = dash_table.DataTable(
    id='table',
    columns=[{'name': i, 'id': i} for i in pivot_table.columns],
    data=pivot_table.to_dict('records'),
    style_table={ 'width': '50%'},

)

app.layout = html.Div(
    style={'backgroundColor': '#E00000', 'height': '75px'},
    
    children=[

        html.Hr(),
        html.H1('Summit ESP- A Halliburton Service: Geothermal Dashboard',
                style={"text-align": "center", "font-size": "2rem"}),

        dcc.Dropdown(
            id='customer-dropdown',
            options=options,
            value='all',
            placeholder='Select a customer...'
        ),
        html.Center(dash_table),
        html.Br(),
        
        #dcc.Graph(id='data-table'),
        dcc.Graph(id='failure-points-pie-chart'),
        dcc.Graph(id='reason-for-pull-pie-chart'),
        dcc.Graph(id='normal-dist-plot'),
        dcc.Graph(id='survive-plot'),
        dcc.Graph(id='reliability-plot'),
        html.Footer(children=[html.P("Summit ESP - Global Technical Service; Created by Buck Pettit - 2023",
                                     style={"font-size": "small"})])
        ])

@app.callback(
    [#dash.dependencies.Output('data-table', 'figure'),
        dash.dependencies.Output('failure-points-pie-chart', 'figure'),
     dash.dependencies.Output('reason-for-pull-pie-chart', 'figure'),
    dash.dependencies.Output('normal-dist-plot', 'figure'),
    dash.dependencies.Output('survive-plot', 'figure'),
    dash.dependencies.Output('reliability-plot', 'figure')],
    [
     dash.dependencies.Input('customer-dropdown', 'value')
     ])
def update_pie_charts(selected_customer):
    
    if selected_customer != 'all':
        filtered_data = data[data['Customer'] == selected_customer]
    else:
        filtered_data=data
    failure_points = filtered_data['Failure Points'].value_counts()
    reason_for_pull = filtered_data['Reason for Pull'].value_counts()

    Runtime_rows = filtered_data[filtered_data['R/NR/Waiting'] == 'NR']
    drop_blanks = filtered_data['Runtime'].dropna()
    Sum_run = sum(drop_blanks)
    Runtime = Runtime_rows['Runtime'].tolist()
    sorted_runtime = sorted(Runtime)
    mean = np.mean(sorted_runtime)
    std_dev = np.std(sorted_runtime)
    dist = norm(loc=mean, scale=std_dev)
    pdf_values = []
    for value in sorted_runtime:
        pdf_values.append(dist.pdf(value))
    max_1 = np.max(pdf_values)
    failure_points_fig = px.pie(
        names=failure_points.index,
        values=failure_points.values,
        title=f'Failure Points (Total: {sum(failure_points)})'
    )
    reason_for_pull_fig = px.pie(
        names=reason_for_pull.index,
        values=reason_for_pull.values,
        title=f'Reason for Pull (Total: {sum(failure_points)})'
    )
    normal_dist_fig = px.scatter(
        x=sorted_runtime,
        y=pdf_values,
        title=f'Normal Distribution of Runtime (Total: {sum(failure_points)})'
    )
    normal_dist_fig.update_layout(
        xaxis_title="Days",
        yaxis_title="Normal Distribution",
        shapes=[
            dict(
                type='line',
                x0=mean,
                y0=0,
                x1=mean,
                y1=max_1,
                line=dict(color='red', width=2),
            )
        ],
        annotations=[
            dict(
                x=mean,
                y=max_1,
                xref="x",
                yref="y",
                text="mean =" + str(round(mean)) + " days",

            )
        ], )
    failure_count = []
    prob_array = []

    for i in range(len(sorted_runtime)):
        failure_count.append(i + 1)
    for i in range(len(sorted_runtime)):
        results = 1 - (failure_count[i] / len(failure_count))
        prob_array.append(results)
    survivability_fig = px.line(x=sorted_runtime, y=prob_array,
                                title=f'Survivability Curve for (Total: {sum(failure_points)})')
    survivability_fig.update_layout(
        xaxis_title="Days",
        yaxis_title="Survival Probability",
    )
    Total_failures = np.max(failure_count)
    mttf = (Sum_run / Total_failures)

    reliability = []
    for i in range(len(sorted_runtime)):
        reliability.append(math.exp((-1 / mttf) * sorted_runtime[i]))
    reliability_fig = px.line(x=sorted_runtime, y=reliability,
                              title=f'Reliability Curve for (Total: {sum(failure_points)})')
    reliability_fig.update_layout(
        xaxis_title="Days",
        yaxis_title="Overall Reliability", )
    return  failure_points_fig, reason_for_pull_fig, normal_dist_fig, survivability_fig, reliability_fig
if __name__ == '__main__':
    app.run_server(debug=True)
