#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 20:53:21 2020

@author: asherhensley
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import yulesimon as ys
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import numpy as np
import dash_table

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#000000',
    'text': '#4ae2ed'
}

fig1 = make_subplots()
# fig1.update_layout(
#     autosize=False,
#     height=400,
#     width=600,
#     showlegend=False,
#     #margin=dict(l=0,r=0,b=50,t=50),
#     )

fig2 = make_subplots()
# fig2.update_layout(
#     autosize=False,
#     height=400,
#     width=600,
#     showlegend=False,
#     #margin=dict(l=0,r=0,b=50,t=50),
#     )

fig3 = make_subplots()

colors = {
    'background': '#000000',
    'text': '#7FDBFF'
}

df = pd.DataFrame(data={
        "Key Statistics":[6],
        "Values":[4]})

app.layout = html.Div(children=[
    
    html.H1(children='CIRCLON-8', style={'textAlign':'left'}),
    
    html.Div(children=[
        'Ticker: ',
        dcc.Input(id='Ticker',value='MSFT',type='text', size='50'),
        html.Button('Search',id='Search',n_clicks=0)]
        ),
    
    html.Br(),
    
    html.H6(id='Status',children='Ready', style={'textAlign':'left'}),
    
    # dash_table.DataTable(
    #     id='table',
    #     columns=[{"name": "Key Statistics", "id": "Key Statistics"},
    #              {"name": "Values", "id": "Values"}],
    #     data=df.to_dict('records')
    #     ),
    
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Prices/Returns',
                children=[dcc.Graph(id='Figure1',figure=fig1)]),
        dcc.Tab(label='Volatility Profile', 
                children=[dcc.Graph(id='Figure2',figure=fig2)]),
        dcc.Tab(label='Modeling Analysis', 
                children=[dcc.Graph(id='Figure3',figure=fig2)]),
    ]),
    html.Div(id='tabs-content')
    
])

@app.callback(
    Output(component_id='Status', component_property='children'),
    Input(component_id='Search', component_property='n_clicks')
    )
def set_status(n_clicks):
    status = 'Searching...'
    if n_clicks==0:
        status = 'Initializing...'
    return status

@app.callback(
    Output(component_id='Figure1', component_property='figure'),
    Output(component_id='Figure2', component_property='figure'),
    Output(component_id='Figure3', component_property='figure'),
    Output(component_id='Status', component_property='children'),
    Input(component_id='Ticker', component_property='value'),
    Input(component_id='Search', component_property='n_clicks')
    )
def update_figure(ticker_in, n_clicks):
    
    ctx = dash.callback_context
    
    if not ctx.triggered:
        ticker = 'MSFT'
    else:
        callback_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if callback_id=='Search':
            ticker = ticker_in
        else:
            ticker = None
            
    if ticker==None:
        raise PreventUpdate
    else:
        
        # Run Model
        closing_prices, log_returns, dates = ys.GetYahooFeed(ticker,5)
        Chain = ys.TimeSeries(log_returns)
        nsteps = 200
        burnin = nsteps/2.0
        downsample = 2
        history = Chain.step(nsteps)
        sigma, sample_size = ys.ExpectedValue(history.std_deviation, burnin, downsample)
        mu, sample_size = ys.ExpectedValue(history.mean, burnin, downsample)
        
        
        z = np.arange(-0.2,0.2,0.001)
        yulesimon_PDF = ys.MixtureModel(z,mu/100,sigma/100)
        H,b = np.histogram(log_returns,200)
        delta = b[1]-b[0]
        bctr = b[1:]-delta/2.0
        empirical_PDF = H/(sum(H)*delta)
        gaussian_PDF = ys.Gaussian(z,np.mean(log_returns),1/np.var(log_returns))
        
        # Update Prices/Returns
        fig1 = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.05)
        fig1.add_trace(go.Scatter(x=dates[1:],y=closing_prices[1:],
                                  fill='tozeroy',
                                  line_color='#0000ff',
                                  fillcolor='#7474f7'), row=1,col=1)
        fig1.add_trace(go.Scatter(x=dates[1:],y=mu/100+2*sigma/100,
                                  fill='tozeroy',
                                  fillcolor='#ffb0b0',
                                  mode='none'), row=2,col=1)
        fig1.add_trace(go.Scatter(x=dates[1:],y=mu/100-2*sigma/100,
                                  fill='tozeroy',
                                  fillcolor='#ffb0b0',
                                  mode='none'), row=2,col=1)
        fig1.add_trace(go.Scatter(x=dates[1:],y=log_returns,
                                  line_color='#ff0000'), row=2,col=1)
        fig1.add_trace(go.Scatter(x=dates[1:],y=mu,
                                  line_color='#000000'), row=2,col=1)
        #fig1.add_trace(go.Scatter(x=dates[1:],y=mu*0,line=dict(dash='dash'),
        #                          line_color='#000000'), row=2,col=1)
        
        fig1.update_layout(
            showlegend=False,
            height=700
            )
        
        fig1.update_yaxes(title_text='Daily Close',row=1,col=1)
        fig1.update_yaxes(title_text='Daily Log-Return',row=2,col=1)
        
        # Update Volatility Profile
        fig2 = make_subplots(rows=1,cols=2,
                             shared_xaxes=True,
                             subplot_titles=("Linear Scale","Log Scale"))
        
        fig2.add_trace(go.Scatter(x=bctr,y=empirical_PDF,mode='markers',marker_color='#ff0000'),row=1,col=1)
        #fig2.add_trace(go.Scatter(x=z,y=gaussian_PDF,line_color='#edc24a',),row=1,col=1)
        fig2.add_trace(go.Scatter(x=z,y=yulesimon_PDF,line_color='#0000ff',),row=1,col=1)
        fig2.add_trace(go.Scatter(x=bctr,y=empirical_PDF,mode='markers',marker_color='#ff0000'),row=1,col=2)
        #fig2.add_trace(go.Scatter(x=z,y=gaussian_PDF,line_color='#edc24a',),row=1,col=2)
        fig2.add_trace(go.Scatter(x=z,y=yulesimon_PDF,line_color='#0000ff',),row=1,col=2)

        fig2.update_xaxes(title_text='Log Returns',row=1,col=1)
        fig2.update_yaxes(title_text='Probability Density',row=1,col=1)
        fig2.update_xaxes(title_text='Log Returns',row=1,col=2)
        fig2.update_yaxes(title_text='Probability Density',type="log",row=1,col=2)
        
        fig2.update_layout(showlegend=False)
        
        # Update Modeling Analysis Tab
        fig3 = make_subplots(rows=1,cols=2)
        fig3.add_trace(go.Scatter(y=history.log_likelihood,line_color='#0000ff',),row=1,col=1)
        fig3.add_trace(go.Scatter(y=history.pvalue,line_color='#ff0000',),row=1,col=2)
        fig3.update_xaxes(title_text='Iteration',row=1,col=1)
        fig3.update_yaxes(title_text='Log-Likelihood',row=1,col=1)
        fig3.update_xaxes(title_text='Iteration',row=1,col=2)
        fig3.update_yaxes(title_text='p-Value',type="log",row=1,col=2)
        fig3.update_layout(showlegend=False)
        
  
        return fig1, fig2, fig3, 'Ready'


if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
    
    
    
    
    
    
    