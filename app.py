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


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#000000'
}

fig = make_subplots()
fig.update_layout(
    autosize=False,
    height=10,
    width=10,
    )

app.layout = html.Div(children=[
    
    html.H1(children='Simon Analytics', style={'textAlign':'left'}),
    
    html.Div(children=
        ['Ticker: ',
         dcc.Input(id='Ticker',value='MSFT',type='text', size='50'),
         html.Button('Search',id='Search',n_clicks=0)]
        ),
    
    html.Br(),
    
    html.H6(id='Status',children='Status: Ready', style={'textAlign':'left'}),
    
    dcc.Graph(
        id='Figure',
        figure=fig
    ),    
    
])

@app.callback(
    Output(component_id='Status', component_property='children'),
    Input(component_id='Search', component_property='n_clicks')
    )
def set_status(n_clicks):
    status = 'Status: Processing...'
    if n_clicks==0:
        status = 'Status: Initializing...'
    return status

@app.callback(
    Output(component_id='Figure', component_property='figure'),
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
        closing_prices, log_returns, dates = ys.GetYahooFeed(ticker,5)
        Chain = ys.TimeSeries(log_returns)
        nsteps = 200
        burnin = nsteps/2.0
        downsample = 2
        history = Chain.step(nsteps)
        sigma, sample_size = ys.ExpectedValue(history.std_deviation, burnin, downsample)
        mu, sample_size = ys.ExpectedValue(history.mean, burnin, downsample)
        
        sigma_t = sigma/100
        mu_t = mu/100
        
        log_returns_std = mu_t + np.std(log_returns)*(log_returns-mu_t) / sigma_t
        
        detrended = closing_prices[0] * np.cumprod(np.exp(log_returns_std))
        
        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=dates[1:],y=closing_prices[1:],
                                  fill='tozeroy',
                                  line_color='#0000ff',
                                  fillcolor='#7474f7'), row=1,col=1)
        
        fig.add_trace(go.Scatter(x=dates[1:],y=detrended,
                                  line_color='#ff0000'), row=1,col=1)
        
        fig.add_trace(go.Scatter(x=dates[1:],y=mu/100+2*sigma/100,
                                  fill='tozeroy',
                                  fillcolor='#ffb0b0',
                                  mode='none'), row=2,col=1)
        fig.add_trace(go.Scatter(x=dates[1:],y=mu/100-2*sigma/100,
                                  fill='tozeroy',
                                  fillcolor='#ffb0b0',
                                  mode='none'), row=2,col=1)
        fig.add_trace(go.Scatter(x=dates[1:],y=log_returns,
                                  line_color='#ff0000'), row=2,col=1)
        fig.add_trace(go.Scatter(x=dates[1:],y=mu,
                                  line_color='#000000'), row=2,col=1)
        fig.add_trace(go.Scatter(x=dates[1:],y=mu*0,line=dict(dash='dash'),
                                  line_color='#000000'), row=2,col=1)
        
        fig.update_layout(
            autosize=False,
            height=500,
            width=600,
            showlegend=False,
            margin=dict(l=0,r=0,b=0,t=0),
            )
    
        return fig, 'Status: Ready'


if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
    
    
    
    
    
    
    