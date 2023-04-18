import dash
from dash import html, dcc, callback, Input, Output, State, exceptions
import dash_bootstrap_components as dbc

dash.register_page(__name__)

layout = dbc.Container([
    dcc.Link('Go back', href='/home'),
    dcc.Location(id='url-viz', refresh=False),
    dbc.Row(dbc.Col(html.H4("Seasonal Decomposition"), width={'size': 12, 'offset': 0, 'order': 0}), style={
        'textAlign': 'center', 'paddingBottom': '1%'}),
    dbc.Row(dbc.Col(html.Iframe(id="ts-decomp-stl-plots-iframe", srcDoc="<h3>Loading...</h3>",
            style={"border-width": "5", "width": "100%", "height": "1000px",'paddingBottom': '2%'}), id="ts-decomp-stl-plots")),

    # needs to be here, to solve bug of layout
    dbc.Row(dbc.Col(html.H4(""), width={'size': 12, 'offset': 0, 'order': 0}), style={
        'textAlign': 'center', 'paddingBottom': '10%'}),
    # solves race condition of callback running faster then the rendering of the page layout
    dcc.Interval(id='interval-component', interval=1*1000, max_intervals=1)
])

@callback(
    Output("ts-decomp-stl-plots-iframe", "srcDoc"),
    Input("url-viz", "pathname"),
    Input("ts-decomp-stl-plots-iframe-memory", "data"),
    Input("interval-component", "n_intervals")
)
def render_plots(url, html_matplotlib_stl, interval):
    if url != "/decomp-viz":
        raise exceptions.PreventUpdate
    if html_matplotlib_stl is None:
        return "<h3>Loading...</h3>"
    return html_matplotlib_stl
