import logging
from dash import Dash, dcc, html
import dash
import dash_bootstrap_components as dbc
from dash.long_callback import DiskcacheLongCallbackManager
from flask import Flask
import diskcache

# Create app.
server = Flask(__name__)


cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)
app = Dash(server=server, external_stylesheets=[
           dbc.themes.FLATLY], long_callback_manager=long_callback_manager, prevent_initial_callbacks=True, use_pages=True)


app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Mobility Data in Lisbon"), width={'size': 12, 'offset': 0, 'order': 0}), style={
            'textAlign': 'center', 'paddingBottom': '1%'}),
    dcc.Store(id='checklist-store-memory', data={},storage_type="session"),
    dcc.Store(id="ts-decomp-table-store-memory", data={}, storage_type="session"),
    dcc.Store(id='ts-decomp-stl-plots-iframe-memory', data={}, storage_type='session'),
    dcc.Store(id='ts-decomp-trendregression-plots-iframe-memory', data={}, storage_type='session'),

	dash.page_container
])
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting WebApp")
    app.run_server(debug=True)