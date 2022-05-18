import os
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import plotly.express as px
import dash
from dash import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc

from dash import html

from dash.dependencies import Input, Output, State


def encode(name):
  char_indices={'a': 1,
 'b': 12,
 'c': 11,
 'd': 13,
 'e': 9,
 'f': 20,
 'g': 24,
 'h': 4,
 'i': 5,
 'j': 21,
 'k': 22,
 'l': 0,
 'm': 23,
 'n': 25,
 'o': 16,
 'p': 14,
 'q': 18,
 'r': 15,
 's': 3,
 't': 10,
 'u': 7,
 'v': 17,
 'w': 2,
 'x': 19,
 'y': 8,
 'z': 6}
  name=name.lower()
  X = np.zeros((1, 15, 26 ), dtype=bool)
  for t, char in enumerate(name):
      X[0, t, char_indices[char]] = 1   
  return(X)


#from utils import preprocess

# Load the model
model_path = 'gender.h5'
pred_model = load_model(model_path)

# Setup the Dash App
external_stylesheets = [dbc.themes.LITERA]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Server
server = app.server

# FAQ section
with open('faq.md', 'r') as file:
    faq = file.read()

# App Layout
app.layout = html.Table([
    html.Tr([
        html.H1(html.Center(html.B('Predict Gender from A Name'))),
        html.Div(
            html.Center("Check if a name is a Boy's name or a Girl's name"),
            style={'fontSize': 20}),
        html.Br(),
        html.Div(
            dbc.Input(id='names',
                      value='Yunfan',
                      placeholder='Enter an English Name',
                      style={'width': '700px'})),
        html.Br(),
        html.Center(children=[
            dbc.Button('Submit',
                       id='submit-button',
                       n_clicks=0,
                       color='primary',
                       type='submit'),
            dbc.Button('Reset',
                       id='reset-button',
                       color='secondary',
                       type='submit',
                       style={"margin-left": "50px"})
        ]),
        html.Br(),
        dcc.Loading(id='table-loading',
                    type='default',
                    children=html.Div(id='predictions',
                                      children=[],
                                      style={'width': '700px'})),
        dcc.Store(id='selected-names'),
        html.Br(),
        dcc.Markdown(faq, style={'width': '700px'})
    ])
],
                        style={
                            'marginLeft': 'auto',
                            'marginRight': 'auto'
                        })


# Callbacks
@app.callback([Output('submit-button', 'n_clicks'),
               Output('names', 'value')], Input('reset-button', 'n_clicks'),
              State('names', 'value'))
def update(n_clicks, value):
    if n_clicks is not None and n_clicks > 0:
        return -1, ''
    else:
        return 0, value


@app.callback(
    [Output('predictions', 'children'),
     Output('selected-names', 'data')], Input('submit-button', 'n_clicks'),
    State('names', 'value'))
def predict(n_clicks, value):
    if n_clicks >= 0:
        # Split on all non-alphabet characters
        name = re.sub("[^a-zA-Z]+", "", value)
        name=np.array([name])
        pred_df = pd.DataFrame({'name': name})

        # Predictions
        result = pred_model.predict(encode(name[0]))[0,0]
        pred_df['Gender'] = [
            'Male' if result > 0.5 else 'Female' 
        ]
        pred_df['Probability'] = [
            result if result > 0.5 else 1.0 - result
        ]

        # Format the output
        pred_df['name'] = name
        pred_df.rename(columns={'name': 'Name'}, inplace=True)
        pred_df['Probability'] = pred_df['Probability'].round(2)
        pred_df.drop_duplicates(inplace=True)

        return [
            dash_table.DataTable(
                id='pred-table',
                columns=[{
                    'name': col,
                    'id': col,
                } for col in pred_df.columns],
                data=pred_df.to_dict('records'),
                sort_action="native",  # give user capability to sort columns
                sort_mode="single",  # sort across 'multi' or 'single' columns
                page_current=0,  # page number that user is on
                page_size=10,  # number of rows visible per page
                style_cell={
                    'fontFamily': 'Arial',
                    'textAlign': 'center',
                    'padding': '10px',
                    'backgroundColor': 'rgb(255, 255, 254)',
                    'height': 'auto',
                    'font-size': '18px',
                    'font-weight':"bold"
                },
                style_header={
                    'backgroundColor': '#5185e5',
                    'font-weight':"normal",
                    'color': 'white',
                    'textAlign': 'center',
                    'font-size':'14px'
                },
                export_format='csv')
        ], name
    else:
        return [], ''




if __name__ == '__main__':
    app.run_server(port=8020)