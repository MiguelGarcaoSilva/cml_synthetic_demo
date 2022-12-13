import dash
from dash import html, dcc, callback, Input, Output, State, exceptions
import dash_bootstrap_components as dbc
import logging

dash.register_page(__name__)

layout = dbc.Container([

    dcc.Link('Go back', href='/home'),
    dcc.Location(id='url-viz', refresh=False),
    dbc.Row(dbc.Col(html.H4("Season-Trend decomposition with LOESS (STL)"), width={'size': 12, 'offset': 0, 'order': 0}), style={
            'textAlign': 'center', 'paddingBottom': '1%'}),
    dbc.Row(dbc.Col(html.Iframe(id="ts-decomp-stl-plots-iframe", srcDoc=None,
            style={"border-width": "5", "width": "100%", "height": "1000px"}), id="ts-decomp-stl-plots")),

    dbc.Row(dbc.Col(html.H4("Trend Regression (OLS)"), width={'size': 12, 'offset': 0, 'order': 0}), style={
        'textAlign': 'center', 'paddingBottom': '1%'}),
    dbc.Row(dbc.Col(html.Iframe(id="ts-decomp-trendregression-plot-iframe", srcDoc=None,
            style={"border-width": "5", "width": "100%", "height": "300px"}), id="ts-decomp-trendregression--plots"))

])

@callback(
        Output("ts-decomp-stl-plots-iframe", "srcDoc"),
        Output("ts-decomp-trendregression-plot-iframe", "srcDoc"),
        Input('url-viz','pathname'),
        Input('ts-decomp-stl-plots-iframe-memory', "data"),
        Input('ts-decomp-trendregression-plots-iframe-memory', "data")
)
def render_plots(url, html_matplotlib_stl, html_matplotlib_trend_regression):
        if url != '/decomp-viz': raise exceptions.PreventUpdate
        return html_matplotlib_stl, html_matplotlib_trend_regression



