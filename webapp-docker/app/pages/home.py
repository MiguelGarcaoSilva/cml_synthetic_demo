import logging
from dash import Dash, callback, register_page, dcc, html, Input, State, Output, exceptions, callback_context, no_update, dash_table
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import arrow_function, assign
import plotly.express as px
from dotenv import load_dotenv
from sqlalchemy import create_engine
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import diskcache as dc
import os
import re
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import psycopg2.extras
import time
import math
import mpld3
import concurrent.futures


params = {'axes.labelsize': 28}
pylab.rcParams.update(params)

register_page(__name__, path='/home')

# SGBD configs
DB_HOST = os.getenv('PG_HOST')
DB_PORT= os.getenv('PG_PORT')
DB_USER = os.getenv('PG_USER')
DB_DATABASE = os.getenv('PG_DBNAME')
DB_PASSWORD = os.getenv('PG_PASSWORD')


engine_string = "postgresql+psycopg2://%s:%s@%s:%s/%s" % (
    DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_DATABASE)
engine = create_engine(engine_string)


cache = dc.Cache("/path/to/cache_directory")


def get_current_dataset(time_agg, space_agg):
    # Check if data exists in cache
    cache_key = f"{time_agg}_{space_agg}"
    cached_data = cache.get(cache_key)

    if cached_data is not None:
        # Data found in cache, use the cached data
        data = cached_data
        logging.warning("Got data from cache.")
    else:
        # Data not found in cache, retrieve it from source and cache it
        data = get_dataset(time_agg, space_agg)
        cache.set(cache_key, data)
        logging.warning("Data retrieved from source and cached.")
    return data




def get_dataset(time_agg, space_agg):
    start = time.process_time()
    view_name = f"mob_data_aggregated_{time_agg.lower()}_{space_agg.lower()}_withgeom_view"
    logging.warning(view_name)
    query = f"SELECT * FROM {view_name}"
    geom_col = f"wkt_{space_agg.lower()}"
    gdf = gpd.read_postgis(query, engine, geom_col= geom_col, crs="EPSG:4326")
    gdf["datetime"] = pd.to_datetime(
                gdf["one_time"])
    gdf = gdf.drop("one_time", axis=1)
    logging.warning(time.process_time() - start)
    return gdf


def get_info(feature=None, space_agg=None):
    header = [html.H4("Lisbon - Number of Terminals")]
    if not feature:
        return header
    if space_agg == "Cell":
        return header + [html.B(feature["properties"]["township_name"] + " - " + str(feature["properties"]["location_id"])), html.Br(),
                         "Aprox: {:0.0f} terminals".format(
                             feature["properties"]["sum_terminals"]), html.Br(),
                         "Approx In: {:0.0f}, Out:{:0.0f}, Staying:{:0.0f}  ".format(feature["properties"]["sum_terminals_in"], feature["properties"]["sum_terminals_out"], feature["properties"]["sum_terminals_stayed"]),  html.Br()]
    if space_agg == "TAZ":
        return header + [html.B(feature["properties"]["taz_name"] + " - " + str(feature["properties"]["taz_id"])), html.Br(),
                         "Aprox: {:0.0f} terminals".format(feature["properties"]["sum_terminals"]), html.Br()]
    if space_agg == "Township":
        return header + [html.B(feature["properties"]["township_name"] + " - " + str(feature["properties"]["dicofre_code"])), html.Br(),
                         "Aprox: {:0.0f} terminals".format(feature["properties"]["sum_terminals"]), html.Br()]

    return header

def get_info_statistics(feature=None, space_feature_name=None):
    header = [html.H4("Lisbon - Statistics of Interest")]
    if not feature:
        return header
    return header + [html.B(feature["properties"][space_feature_name]), html.Br(),
                        "F_T: {:.2f}, F_S:{:.2f}, F_R:{:.2f}  ".format(feature["properties"]["F_T"], feature["properties"]["F_S"], feature["properties"]["F_R"]),  html.Br()]


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}+'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


colorscale = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']
style = dict(weight=2, opacity=1, color='blue',
             dashArray='3', fillOpacity=0.7)
style_handle = assign("""function(feature, context){
    const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value > classes[i]) {
            style.fillColor = colorscale[i];  // set the fill color according to the class
        }
    }
    return style;
}""")


info = html.Div(children=get_info(), id="info", className="info",
                style={"position": "absolute", "top": "10px", "right": "10px", "z-index": "1000"})
info_statistics = html.Div(children=get_info_statistics(), id="info_statistics", className="info",
                style={"position": "absolute", "top": "10px", "right": "10px", "z-index": "1000"})



layout = dbc.Container([

    dcc.Location(id='url-home', refresh=False),

    html.Div([
        html.Div([
            dcc.Dropdown(
                ["Hourly", "Daily", "Weekly", "Monthly"],
                'Daily',
                id='timeagg-dropdown', persistence=True, persistence_type="session"
            )
        ], style={'width': '30%', "float": "left", "text-align": "left"}),

        html.Div([
            dcc.Dropdown(
                ["Cell", "TAZ", "Township"],
                'TAZ',
                id='spaceagg-dropdown', persistence=True, persistence_type="session"
            )
        ], style={'width': '30%', 'display': 'inline-block', "text-align": "left"}),
        html.Div([
            dcc.Dropdown(
                options={
                    'sum_terminals': 'Terminals',
                    'sum_roaming_terminals': 'Roamings'
                },
                value='sum_terminals',
                id='datafeature-dropdown', persistence=True, persistence_type="session"
            )
        ], style={'width': '30%', "float": "right", "text-align": "left"})
    ], style={"text-align": "center"}),


    dbc.Row(dbc.Col(children=[html.Div(children=[dl.Map(id="my-map-mobility", children=[dl.TileLayer(), dl.GeoJSON(id="geojson_layer"), None, info], center=(38.74, -9.14), zoom=12, style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"})
                                            ], id="map-container"),
                                dcc.Loading(
                                    id="loading",
                                    type="default", fullscreen = True,
                                    children=[
                                        html.Div(
                                            dcc.RangeSlider(
                                                id="range-slider-mobility",
                                                min=0,
                                                max=0,
                                                value=[0, 0],
                                                marks={},
                                                step=None,
                                                persistence=True,
                                                persistence_type="session"
                                            ),
                                            id="slider_div"
                                        )
                                    ]
                                )
                        ]), style={"margin-bottom": "50px"}),
            

    dbc.Row(children=[
            dbc.Col(html.Div(dcc.Graph(id="ts-plots-chart"),
                    id="multivariate-time-series_div"), width=6),
            dbc.Col(children=[
                    html.Div(dcc.Checklist(
                        ["All"], [], id="all-checklist", inline=True, style={"margin-top": "50px"}, persistence=True, persistence_type="session")),
                    html.Div(dcc.Checklist(
                        id="space-checklist",
                        options=[],
                        value=[],
                        inline=True,
                        style={"height": "200px", "overflow": "scroll", "margin-top": "10px"}, persistence=True, persistence_type="session"), id="checklist_div"),

                    html.Div("Time range to perform TS Decomposition:"),

                    html.Div([
                        dcc.DatePickerRange(
                            id="date-range",
                            persistence=True, 
                            persistence_type="session"
                        ), 
                       dcc.Input(id="start-time-range", type="datetime-local", style={"display": "none"}, persistence=True, persistence_type="session"),
                       dcc.Input(id="end-time-range", type="datetime-local", style={"display": "none"}, persistence=True, persistence_type="session")],
                        id="date-range-div", style={'textAlign': 'center', "margin": "auto", "margin-top": "10px"}),

                    html.Div([
                            html.Label('Data Features:'),
                            dcc.Dropdown(
                                id='datafeature-dropdown-decomp',
                                options=[],
                                multi=True,
                                value=[],  
                                style={'width': '100%'}, persistence=True, persistence_type="session"
                            )
                        ], style={'margin-bottom': '10px'}),

                    html.Div([

                            dcc.Dropdown(
                                options=[
                                    {"label": "Classic Additive", "value": "seasonal_decompose"},
                                    {"label": "STL", "value": "stl"},
                                    {"label": "MSTL", "value": "mstl"}
                                ],
                                value="seasonal_decompose",
                                id="decompose_algo-dropdown",
                                persistence=True,
                                persistence_type="session",
                                style={'width': '60%'}
                            ),
                        html.Label("Seasonal Period(s):", style={"width": "25%"}),
                        dcc.Input(
                            id='period-input',
                            type='number',
                            value=[],
                            placeholder="e.g., '7'",
                            persistence=True,
                            persistence_type="session",
                            style={"width": "20%"}        
                        ), 
                    ], style={'display': 'flex', 'textAlign': 'center', "margin": "auto", "margin-top": "10px"}),


                    html.Div([
                        dbc.Button("Run TS Decomp!",id="tsdecomp_button", className="mr-2", color="primary"),
                        dbc.Button( "Cancel Running Job!", id="cancel_tsdecomp_button", className="mr-2", color="danger", style={
                            "margin-left": "15px"})], style={'textAlign': 'center', "margin": "auto", "margin-top": "20px"}),

                    html.Div([dbc.Progress(id="tsdecomp_progress_bar", value="0")],  style={
                             'textAlign': 'center', "margin": "auto", "margin-top": "30px"})
                    ], width=6)]),



                    
    dbc.Row(children=[dbc.Col(html.Div(dash_table.DataTable(id="ts-decomp-table", editable=True,
                                                            filter_action="native",
                                                            sort_action="native",
                                                            sort_mode="multi",
                                                            column_selectable="single",
                                                            row_selectable="multi",
                                                            row_deletable=False,
                                                            selected_columns=[],
                                                            selected_rows=[],
                                                            page_action="native",
                                                            page_current=0,
                                                            page_size=10, persistence=True, persistence_type="session", persisted_props=["columns.name", "data"]),
                                       id="ts-decomp-table_div", style={'textAlign': 'center', "margin": "auto", "margin-top": "30px"}))]),

    dbc.Row(children=[dbc.Col(html.Div(dcc.Link(dbc.Button("Check Decomposition Visualizations!", id="checkfull_viz_button", className="mr-2", color="primary"),
            href="/decomp-viz"), style={ "margin": "auto", "margin-top": "30px", "text-align": "center"})), dbc.Col(dbc.Button("Download Decomposition Results!", id="download_decomp_button", className="mr-2", color="primary"), style={ "margin": "auto", "margin-top": "30px", "text-align": "center"}), dcc.Download(id="download-results")], justify="center"),

   html.Div([
        dcc.Dropdown(
            options={
                'F_T': 'F_T',
                'ROC':'ROC',
                'Unified Score':'Unified Score',
                'F_S': 'F_S',
                'F_R': 'F_R'
            },
            value='F_R',
            id='statistics-dropdown', persistence=True, persistence_type="session"
        )
    ], style={'width': '30%', "float": "right", "text-align": "left", "margin-top": "30px"}),
    dbc.Row(dbc.Col(children=[html.Div(children=[dl.Map(id="my-map-statistics", children=[dl.TileLayer(), dl.GeoJSON(id="geojson_layer_statistics"), None, info_statistics], center=(38.74, -9.14), zoom=12, style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"}),
                                                 ], id="map-statistics-container"),
                              ]), style={"margin-top": "30px"})

])


    

@callback(
    Output("info", "children"),
    Input("geojson_layer", "hover_feature"),
    State("spaceagg-dropdown", "value"))
def info_hover(feature, space_agg):
    return get_info(feature, space_agg)

@callback(
    Output("info_statistics", "children"),
    Input("geojson_layer_statistics", "hover_feature"),
    State("spaceagg-dropdown", "value"))
def info_hover_statistics(feature, space_agg):
    if space_agg == "Cell":
        space_feature_name = "location_id"
    elif space_agg == "TAZ":
        space_feature_name = "taz_name"
    else:
        space_feature_name = "township_name"
    return get_info_statistics(feature, space_feature_name)


@callback(
    Output("space-checklist", "value"),
    Output("all-checklist", "value"),
    Output("timeagg-store-memory", "data"),
    Output("spaceagg-store-memory", "data"),
    Output("checklist-store-memory", "data"),
    Input("space-checklist", "value"),
    Input("all-checklist", "value"),
    Input("timeagg-dropdown", "value"),
    Input('spaceagg-dropdown', 'value'),
    State("checklist-store-memory", "data"),
    prevent_initial_call=True,
    background=True
)
def sync_checklists(space_selected, all_selected, time_agg, space_agg, stored_checklists):


    ctx = callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if space_agg == "Cell":
        space_feature_name = "location_id"
    elif space_agg == "TAZ":
        space_feature_name = "taz_name"
    else:
        space_feature_name = "township_name"

    gdf_data = get_current_dataset(time_agg,space_agg)
    space_names = sorted(gdf_data[space_feature_name].unique())
    if input_id == "space-checklist":
        all_selected = ["All"] if set(
            space_selected) == set(space_names) else []
    elif input_id == "all-checklist":
        space_selected = space_names if all_selected else []
    else:
        if stored_checklists:
            space_selected, all_selected = stored_checklists
        else:
            space_selected = []
            all_selected = []

    #we update here the timeagg and spaceagg memory, because the function gets input from that
    return space_selected, all_selected, time_agg, space_agg, [space_selected, all_selected]

@callback(
    Output("date-range-div", "children"),
    Input('timeagg-dropdown', 'value'),
    Input('spaceagg-dropdown', 'value'),
    background=True
)
def change_between_time_data_range(time_agg,space_agg):

    gdf_data = get_current_dataset(time_agg, space_agg)
    sorted_dates_datetime = sorted(gdf_data['datetime'].unique())

    min_date_string = np.datetime_as_string(sorted_dates_datetime[0], unit='m')
    max_date_string = np.datetime_as_string(sorted_dates_datetime[-1], unit='m')

    if time_agg == "Hourly":
        time_inputs = [html.Label("Start Time:"),
                       dcc.Input(id="start-time-range", type="datetime-local", value=min_date_string,
                                 min=min_date_string, max=max_date_string, step="3600", style={"margin-left": "5px"}, persistence=True, persistence_type="session"),
                       html.Label("End Time:", style={
                           "margin-left": "20px"}),
                       dcc.Input(id="end-time-range", type="datetime-local", value=max_date_string, min=min_date_string, max=max_date_string, step="3600", style={"margin-left": "5px"}, persistence=True, persistence_type="session")]
        date_range = dcc.DatePickerRange(
            id="date-range",
            style={"display": "none"}, persistence=True, persistence_type="session"  # hide the datepicker by default
        )
    else:
        time_inputs = [
                       dcc.Input(id="start-time-range", type="datetime-local", style={"display": "none"}, persistence=True, persistence_type="session"),
                       dcc.Input(id="end-time-range", type="datetime-local", style={"display": "none"}, persistence=True, persistence_type="session")]
        date_range = dcc.DatePickerRange(
            id="date-range",
            min_date_allowed=min_date_string,
            max_date_allowed=max_date_string,
            display_format="DD-MM-YYYY",
            month_format='MMMM Y',
            end_date_placeholder_text='MMMM Y',
            start_date=min_date_string,
            end_date=max_date_string,
            persistence=True,
            persistence_type="session"
        )

    return [date_range] + time_inputs


@callback(
    Output("datafeature-dropdown", "options"),
    Output("datafeature-dropdown-decomp", "options"),
    Input("spaceagg-dropdown","value"),
    State('timeagg-dropdown', 'value'),
    background=True
)
def update_datafeature_dropdown(space_agg, time_agg):

    gdf_data = get_current_dataset(time_agg, space_agg)
    patterns = [r".*_id", r".*code", r".*_name", r"datetime", r"wkt.*"]
    data_columns =  [x for x in gdf_data.columns.tolist() if not any(re.match(pattern, x) for pattern in patterns)]

    return data_columns, data_columns




@callback(
    Output('date-range-store-memory', 'data'),
    Input("date-range-div", "children"),
    Input('timeagg-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('start-time-range', 'value'),
    Input('end-time-range', 'value'))
def update_date_range_store(div, time_agg, start_date, end_date, start_hour, end_hour):
    
    if time_agg == "Hourly":
        return {'start_date': start_hour, 'end_date': end_hour}
    return {'start_date': start_date, 'end_date': end_date}

@callback(
    Output("period-input", "type"),
    Input("decompose_algo-dropdown", "value"),
    prevent_initial_call=True
)
def update_period_input(decomp_algo):
    if decomp_algo == "mstl":
        return "text"
    return "number"


@callback(
    Output("checklist_div", "children"),
    Input("geojson_layer", "click_feature"),
    Input('spaceagg-dropdown', 'value'),
    State('timeagg-dropdown', 'value'),
    State("space-checklist", "value"),
    prevent_initial_call=True
)
def update_checklist_from_map_and_spacedropdown(selected_space, space_agg, time_agg, checklist_values):
    # new checklist from space aggregation
    if callback_context.triggered_id == "spaceagg-dropdown":
        checklist = html.Div(dcc.Checklist(
            id="space-checklist",
            options=space_names,
            value=[],
            style={"height": "200px", "overflow": "scroll", "margin-top": "10px"},
            inline=False, persistence=True, persistence_type="session"))
        return checklist

    # Sync map click with checklist
    if callback_context.triggered_id == "geojson_layer":
        if space_agg == "Cell":
            space_feature_name = "location_id"
        elif space_agg == "TAZ":
            space_feature_name = "taz_name"
        else:
            space_feature_name = "township_name"

        if selected_space:  # prevents from triggering when map is updated but not clicked
            checklist_values = checklist_values + \
                [selected_space["properties"][space_feature_name]]

        gdf_data = get_current_dataset(time_agg, space_agg)
        space_names = sorted(gdf_data[space_feature_name].unique())
        checklist = dcc.Checklist(
            id="space-checklist",
            options=space_names,
            value=checklist_values,
            style={"height": "200px", "overflow": "scroll", "margin-top": "10px"},
            inline=True, persistence=True, persistence_type="session")
        return checklist


@callback(
    Output("ts-plots-chart", "figure"),
    Input("space-checklist", "value"),
    State('timeagg-dropdown', 'value'),
    State('spaceagg-dropdown', 'value'),
    State("datafeature-dropdown", "value"),
    prevent_initial_callback=True,
    background=True
)
def update_plot(checklist_values, time_agg, space_agg , data_feature):
    if space_agg == "Cell":
        space_feature_name = "location_id"
    elif space_agg == "TAZ":
        space_feature_name = "taz_name"
    else:
        space_feature_name = "township_name"

    gdf_data = get_current_dataset(time_agg, space_agg)

    filtered_gdf_data = gdf_data[gdf_data[space_feature_name].isin(
        checklist_values)].copy()
    
    # needs to be sorted by datetime to plot with px.line
    filtered_gdf_data.sort_values(by="datetime", inplace=True)
    fig = px.line(filtered_gdf_data, x="datetime", y=data_feature, color=space_feature_name,
                  line_group=space_feature_name, labels={"datetime": "Time"}, hover_name=space_feature_name)
    
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(showlegend=False)

    return fig



def fix_wkt_taz(row, wkt_feature_name):
    # Convert the wkt_taz string to a Shapely geometry
    geometry = row[wkt_feature_name]
    # Check if the geometry is valid
    if not geometry.is_valid:
        # If the geometry is not valid, apply the buffer(0) method to repair it
        geometry = geometry.buffer(0)
    # Return the fixed geometry
    return geometry



@callback(
    Output('map-container', 'children'),
    Output('slider_div', 'children'),
    Input('timeagg-dropdown', 'value'),
    Input('spaceagg-dropdown', 'value'),
    Input("datafeature-dropdown", "value"),
    Input('range-slider-mobility', 'value'),
    prevent_initial_callback=True,
    background=True
)
def update_figure_and_slider(time_agg, space_agg, data_feature, selected_time_range_agg):

    if space_agg == "Cell":
        space_feature_name = "location_id"
        wkt_feature_name = "wkt_cell"
    elif space_agg == "TAZ":
        space_feature_name = "taz_name"
        wkt_feature_name = "wkt_taz"
    else:
        space_feature_name = "township_name"
        wkt_feature_name = "wkt_township"

    # Apply the fix_wkt_taz function to the wkt_taz column
    gdf_data = get_current_dataset(time_agg, space_agg)
    gdf_data[wkt_feature_name] = gdf_data.apply(lambda row: fix_wkt_taz(row, wkt_feature_name), axis=1)

    numdate = [x for x in range(len(gdf_data['datetime'].unique()))]

    if callback_context.triggered_id == "range-slider-mobility":
        selected_range = [sorted(gdf_data['datetime'].unique())[selected_time_range_agg[0]], sorted(gdf_data['datetime'].unique())[
            selected_time_range_agg[1]]]
    else:
        selected_range = [sorted(gdf_data['datetime'].unique())[
            0], sorted(gdf_data['datetime'].unique())[0]]

    if selected_range[0] == selected_range[1]:
        filtered_gdf = gdf_data[(gdf_data['datetime'] == selected_range[0])].drop(
            "datetime", axis=1)
    else:
        filtered_gdf = gdf_data[(gdf_data['datetime'] >= selected_range[0]) & (gdf_data['datetime'] <= selected_range[1])].drop(
            "datetime", axis=1)
        agg_dict = {col: 'sum' if col in [data_feature] else 'first'
                    for col in filtered_gdf.columns if col != wkt_feature_name}
        filtered_gdf = filtered_gdf.dissolve(
            by=space_feature_name, aggfunc=agg_dict)

    max_value = filtered_gdf[data_feature].max()
    classes_scale = math.floor(
        max_value/np.power(10, int(math.log10(max_value))))*np.power(10, int(math.log10(max_value)))
    classes = [i * (classes_scale // 5) for i in range(5)]
    ctg = [human_format(cls) for cls in classes]
    colorbar = dlx.categorical_colorbar(categories=ctg,
                                        colorscale=colorscale,
                                        width=300,
                                        height=30,
                                        position="bottomright")

    geojson = dl.GeoJSON(data=json.loads(filtered_gdf.to_json()),
                         options=dict(style=style_handle),
                         hoverStyle=arrow_function(
                             dict(weight=5, color='#666', dashArray='')),
                         hideout=dict(colorscale=colorscale, classes=classes,
                                      style=style, colorProp=data_feature),
                         id="geojson_layer")

    map = dl.Map(id="my-map-mobility", children=[dl.TileLayer(), geojson, colorbar, info], center=(
        38.74, -9.14), zoom=12, style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"})

    if callback_context.triggered_id == "range-slider-mobility":
        return map, no_update

    if time_agg == "Hourly":
        range_slider = dcc.RangeSlider(
            id="range-slider-mobility",
            min=numdate[0],
            max=numdate[-1],
            value=[numdate[0], numdate[0]],
            marks={numd: {"label": None, "style": {"writing-mode": "vertical-rl"}}
                   for numd, date in zip(numdate, sorted(gdf_data['datetime'].unique()))},
            tooltip={"placement": "bottom", "always_visible": True},
            step=None)

    else:
        range_slider = dcc.RangeSlider(
            id="range-slider-mobility",
            min=numdate[0],
            max=numdate[-1],
            value=[numdate[0], numdate[0]],
            marks={numd: {"label": pd.to_datetime(str(date)).strftime('%d/%m/%Y'), "style": {"writing-mode": "vertical-rl"}}
                   for numd, date in zip(numdate, sorted(gdf_data['datetime'].unique()))},
            step=None)

    return map, range_slider

def process_batch(batch_idx, batch_size, group_names, table_df, space_feature_name, data_features, time_agg, decomp_algo, decomp_period, axes_stl):
    
    logging.warning("processing batch: "+str(batch_idx))
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(group_names))
    batch_group_names = group_names[start_idx:end_idx]
    
    decomp_resids = {}
    decomp_stats_table = pd.DataFrame()

    for i, group_name in enumerate(batch_group_names):

        df_group = table_df[table_df[space_feature_name] == group_name]

        for data_feature in data_features:
            time_serie = df_group[["datetime", data_feature]]
            time_serie = time_serie.set_index("datetime")
            if time_agg == "Hourly":
                time_serie = time_serie.asfreq('H')
            elif time_agg == "Daily":
                time_serie = time_serie.asfreq('D')
            elif time_agg == "Weekly":
                time_serie = time_serie.asfreq('W-MON')
            else:
                time_serie = time_serie.asfreq('MS')

            if decomp_algo == "seasonal_decompose":
                res = seasonal_decompose(np.squeeze(time_serie), model="additive", period=decomp_period)
            elif decomp_algo == "stl": 
                res = STL(np.squeeze(time_serie), period=decomp_period).fit()
            else: #mstl
                res = MSTL(np.squeeze(time_serie), periods=decomp_period).fit()

            #as a lit because dash only stores json serializable data
            decomp_resids[str(group_name)+"_"+data_feature] = [res.resid.index, res.resid.values]

            time_serie_trend = res.trend.dropna(axis=0).to_frame().reset_index()
            time_serie_trend["Time"] = time_serie_trend.index.values
            model_ols = LinearRegression()
            X = time_serie_trend.loc[:, ['Time']]
            y = time_serie_trend.loc[:, "trend"]
            res_ols = model_ols.fit(X, y)
            y_pred = pd.Series(model_ols.predict(X), index=time_serie_trend.datetime)
            formula = f'y = {res_ols.coef_[0]:.2f}x + {res_ols.intercept_:.2f}'

            # Modify batch_axes_stl accordingly
            plotseasonal(axes_stl[:, start_idx+i], res, data_feature, y_pred, formula, group_name)


            # variances explained
            var_resid = np.var(res.resid)
            var_observed = np.var(res.observed)
            trend_strength = max(0, 1 - (var_resid/np.var(res.trend+res.resid)))
            noise_strength = var_resid/var_observed

            if decomp_algo == "mstl":
                seasonal_individial_strengths = {}
                #get strength of individual seasonality components
                for period in res.seasonal:
                    seasonal_individial_strengths["F_"+str(period)] = max(0, 1 - (var_resid/np.var(res.seasonal[period] + res.resid)))
                seasonal_strength = max(0, 1 - (var_resid/np.var(res.seasonal.sum(axis=1) + res.resid)))
            else:
                seasonal_strength = max(0, 1 - (var_resid/np.var(res.seasonal + res.resid)))

            rate_change = (y_pred[-1] - y_pred[0]) / y_pred[0]
            r2 = res_ols.score(X, y)
            a1, a2, a3 = 0, 0, 0
            unified = ((a1 + trend_strength)**1) * ((a2 + r2)**1) * ((a3 + abs(rate_change))**1) * math.copysign(1,rate_change)
            unified = 0 if (unified <= 0 and unified >= -0.001) else unified
            #unified_trendimpact =  ((a1 + trend_strength)**2) * ((a2 + r2)**1) * ((a3 + abs(rate_change))**(0.5)) * math.copysign(1,rate_change)
            #unified_trendimpact = 0 if (unified_trendimpact <= 0 and unified_trendimpact >= -0.001) else unified_trendimpact
            #unified_ratechangeimpact =  ((a1 + trend_strength)**(0.5)) * ((a2 + r2)**1) * ((a3 + abs(rate_change))**2) * math.copysign(1,rate_change)
            #unified_ratechangeimpact = 0 if (unified_ratechangeimpact <= 0 and unified_ratechangeimpact >= -0.001) else unified_ratechangeimpact

            stats_df = {"Name": group_name, "Feature": data_feature,
                        "Slope": round(res_ols.coef_[0], 3), "ROC": round(rate_change, 3), "R2": round(r2, 3),
                        "F_T": round(trend_strength, 3),"Unified Score": round(unified, 3), 
                        "F_S": round(seasonal_strength, 3), "F_R": round(noise_strength, 3)}
            
            #add individual seasonal strengths to stats_df, rounded with 3 decimals
            if decomp_algo == "mstl":
                for period in seasonal_individial_strengths:
                    stats_df[period] = round(seasonal_individial_strengths[period], 3)
                
            
            decomp_stats_table = pd.concat(
                [decomp_stats_table, pd.DataFrame(stats_df, index=[0])], ignore_index=True)
        
    return decomp_resids, decomp_stats_table


@callback(
    [
        Output("ts-decomp-stl-plots-iframe-memory", "data"),
        Output("ts-decomp-resids-store-memory", "data"),
        Output('ts-decomp-table', 'data'),
        Output('ts-decomp-table', 'columns'),
        Output("ts-decomp-table-store-memory", "data"),
        Output("datafeature-store-memory", "data")
    ],
    [Input("url-home", "pathname"),
     Input("tsdecomp_button", "n_clicks"),
     State("decompose_algo-dropdown", "value"),
     State("period-input", "value"),
     State("ts-decomp-resids-store-memory", "data"),
     State("ts-decomp-table-store-memory", "data"),
     State('ts-decomp-stl-plots-iframe-memory', "data"),
     State('timeagg-dropdown', 'value'),
     State('spaceagg-dropdown', 'value'),
     State("datafeature-dropdown-decomp", "value"),
     State("date-range-store-memory", "data"),
     State("checklist-store-memory", "data")],
    running=[
        (Output("timeagg-dropdown", "disabled"), True, False),
        (Output("spaceagg-dropdown", "disabled"), True, False),
        (Output("tsdecomp_button", "disabled"), True, False),
        (Output("datafeature-dropdown", "disabled"), True, False),
        (Output("cancel_tsdecomp_button", "disabled"), False, True),
        (Output("checkfull_viz_button", "disabled"), True, False),
        (
            Output("tsdecomp_progress_bar", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        ),
    ],
    cancel=[Input("cancel_tsdecomp_button", "n_clicks")],
    progress=[Output("tsdecomp_progress_bar", "value"), Output("tsdecomp_progress_bar", "max")],
    prevent_initial_call=True,
    background=True)
def perform_tscomp_ifnotin_cache(set_progress, url, n_clicks, decomp_algo, decomp_periods, stored_decomp_resid,
                                 stored_table, stored_html_matplotlib_stl, time_agg, space_agg,
                                 data_features, stored_daterange, checklist_values_memory):
    if url != '/home' or checklist_values_memory == {}:
        raise exceptions.PreventUpdate

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    checklist_values = checklist_values_memory[0]

    if trigger_id == "url-home" and stored_table:
        return stored_html_matplotlib_stl, stored_decomp_resid, stored_table[0], stored_table[1], stored_table, data_features

    if space_agg == "Cell":
        space_feature_name = "location_id"
    elif space_agg == "TAZ":
        space_feature_name = "taz_name"
    else:
        space_feature_name = "township_name"

    gdf_data = get_current_dataset(time_agg, space_agg)
    filtered_gdf_data = gdf_data[gdf_data[space_feature_name].isin(
        checklist_values)]

    datetime_mask = (filtered_gdf_data['datetime'] >= stored_daterange["start_date"]) & (filtered_gdf_data['datetime'] <= stored_daterange["end_date"])

    filtered_gdf_data = filtered_gdf_data.loc[datetime_mask]

    batch_size = 10

    table_df = filtered_gdf_data[[space_feature_name, "datetime"] + data_features]

    if decomp_algo == "mstl":
        if isinstance(decomp_periods, str):
            decomp_period = [int(x) for x in decomp_periods.split(",")]
        else:
            decomp_period = [decomp_periods]

        if len(decomp_period) <= 1:
            raise exceptions.PreventUpdate
        nrows = 3 + len(decomp_period)
    else:
        decomp_period = int(decomp_periods)
        nrows = 4
    if checklist_values:
        fig_stl, axes_stl = plt.subplots(ncols=len(checklist_values), nrows=nrows, figsize=(
            len(checklist_values) * 4, 8), squeeze=False)

        decomp_stats_table = pd.DataFrame(
            columns=["Name", "Feature", "Slope", "R2", "ROC", "F_T", "Unified Score", "F_S", "F_R"])
        decomp_resids = {}

        group_names = table_df[space_feature_name].unique()  # Get unique group names
        num_groups = len(group_names)
        num_batches = math.ceil(num_groups / batch_size)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, batch_idx, batch_size, group_names, table_df, space_feature_name, data_features, time_agg, decomp_algo, decomp_period, axes_stl) for batch_idx in range(num_batches)]
            
            # Initialize lists to store results
            all_decomp_resids = []
            all_decomp_stats_tables = []
            # Continuously update progress as futures complete
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                decomp_resids, decomp_stats_table = future.result()
                
                # Append results to the lists
                all_decomp_resids.append(decomp_resids)
                all_decomp_stats_tables.append(decomp_stats_table)

                # Logging progress
                completed += 1
                set_progress((completed, str(num_batches)))
        # Combine the results from all batches
        combined_decomp_resids = {}
        combined_decomp_stats_table = pd.DataFrame()

        for decomp_resids in all_decomp_resids:
            combined_decomp_resids.update(decomp_resids)

        for decomp_stats_table in all_decomp_stats_tables:
            combined_decomp_stats_table = pd.concat([combined_decomp_stats_table, decomp_stats_table], ignore_index=True)
        

        axes_stl[0, 0].set_ylabel("Observed")
        axes_stl[1, 0].set_ylabel("Trend")
        ax_idx = 2
        if decomp_algo == "mstl":
            for period in sorted(decomp_period):
                axes_stl[ax_idx, 0].set_ylabel("Seas_"+str(period))
                ax_idx += 1
        else:
            axes_stl[ax_idx, 0].set_ylabel("Seasonal")
            ax_idx += 1
        axes_stl[ax_idx, 0].set_ylabel("Residual")

        fig_stl.tight_layout()
        html_matplotlib_stl = mpld3.fig_to_html(fig_stl)

        table = combined_decomp_stats_table.to_dict(orient='records')
        table_columns = [{'name': col, 'id': col}
                         for col in decomp_stats_table.columns]
        stored_table = [table, table_columns]
        set_progress((0, str(len(checklist_values))))
        return html_matplotlib_stl, decomp_resids, table, table_columns, stored_table, data_features

    # empty list
    set_progress((0, str(len(checklist_values))))
    return None, None, None, None, [None, None], None


# @callback(
#     [
#         Output("ts-decomp-stl-plots-iframe-memory", "data"),
#         Output("ts-decomp-resids-store-memory", "data"),
#         Output('ts-decomp-table', 'data'),
#         Output('ts-decomp-table', 'columns'),
#         Output("ts-decomp-table-store-memory", "data"),
#         Output("datafeature-store-memory", "data")
#     ],
#     [Input("url-home", "pathname"),
#      Input("tsdecomp_button", "n_clicks"),
#      State("decompose_algo-dropdown", "value"),
#      State("ts-decomp-resids-store-memory", "data"),
#      State("ts-decomp-table-store-memory", "data"),
#      State('ts-decomp-stl-plots-iframe-memory', "data"),
#      State('timeagg-dropdown', 'value'),
#      State('spaceagg-dropdown', 'value'),
#      State("datafeature-dropdown-decomp", "value"),
#      State("date-range-store-memory", "data"),
#      State("checklist-store-memory", "data")],
#     running=[
#         (Output("timeagg-dropdown", "disabled"), True, False),
#         (Output("spaceagg-dropdown", "disabled"), True, False),
#         (Output("tsdecomp_button", "disabled"), True, False),
#         (Output("datafeature-dropdown", "disabled"), True, False),
#         (Output("cancel_tsdecomp_button", "disabled"), False, True),
#         (Output("checkfull_viz_button", "disabled"), True, False),
#         (
#             Output("tsdecomp_progress_bar", "style"),
#             {"visibility": "visible"},
#             {"visibility": "hidden"},
#         ),
#     ],
#     cancel=[Input("cancel_tsdecomp_button", "n_clicks")],
#     progress=[Output("tsdecomp_progress_bar", "value"), Output("tsdecomp_progress_bar", "max")],
#     prevent_initial_call=True,
#     background=True)
# def perform_tscomp_ifnotin_cache(set_progress, url, n_clicks, decomp_algo, stored_decomp_resid, stored_table, stored_html_matplotlib_stl, time_agg, space_agg, data_features, stored_daterange, checklist_values_memory):
#     if url != '/home' or checklist_values_memory == {}:
#         raise exceptions.PreventUpdate

#     trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]

#     checklist_values = checklist_values_memory[0]

#     if trigger_id == "url-home" and stored_table:
#         return stored_html_matplotlib_stl, stored_decomp_resid, stored_table[0], stored_table[1], stored_table, data_features

#     if space_agg == "Cell":
#         space_feature_name = "location_id"
#     elif space_agg == "TAZ":
#         space_feature_name = "taz_name"
#     else:
#         space_feature_name = "township_name"

#     gdf_data = get_current_dataset(time_agg, space_agg)
#     filtered_gdf_data = gdf_data[gdf_data[space_feature_name].isin(
#         checklist_values)]

#     datetime_mask = (filtered_gdf_data['datetime'] >= stored_daterange["start_date"]) & (filtered_gdf_data['datetime'] <= stored_daterange["end_date"])

#     filtered_gdf_data = filtered_gdf_data.loc[datetime_mask]

#     batch_size = 10

#     table_df = filtered_gdf_data[[space_feature_name, "datetime"] + data_features]

#     if checklist_values:
#         fig_stl, axes_stl = plt.subplots(ncols=len(checklist_values), nrows=4, figsize=(
#             len(checklist_values) * 5, 10), squeeze=False)

#         decomp_stats_table = pd.DataFrame(
#             columns=["Name", "Feature", "Slope", "R2", "ROC", "F_T", "Unified Score", "F_S", "F_R"])
#         decomp_resids = {}

#         group_names = table_df[space_feature_name].unique()  # Get unique group names
#         num_groups = len(group_names)
#         num_batches = math.ceil(num_groups / batch_size)

#         for batch_idx in range(num_batches):
#             start_idx = batch_idx * batch_size
#             end_idx = min((batch_idx + 1) * batch_size, num_groups)
#             batch_group_names = group_names[start_idx:end_idx]

#             for i, group_name in enumerate(batch_group_names):
#                 set_progress((str(batch_idx * batch_size + i + 1), str(len(checklist_values))))

#                 df_group = table_df[table_df[space_feature_name] == group_name]

#                 for data_feature in data_features:
#                     time_serie = df_group[["datetime", data_feature]]
#                     time_serie = time_serie.set_index("datetime")
#                     if time_agg == "Hourly":
#                         time_serie = time_serie.asfreq('H')
#                     elif time_agg == "Daily":
#                         time_serie = time_serie.asfreq('D')
#                     elif time_agg == "Weekly":
#                         time_serie = time_serie.asfreq('W-MON')
#                     else:
#                         time_serie = time_serie.asfreq('MS')

#                     if decomp_algo == "seasonal_decompose":
#                         res = seasonal_decompose(time_serie, model="additive")
#                     else:  # stl
#                         res = STL(time_serie).fit()

#                     decomp_resids[str(group_name)+"_"+data_feature] = [time_serie.index, res.resid]

#                     time_serie_trend = res.trend.dropna(axis=0).to_frame().reset_index()
#                     time_serie_trend["Time"] = time_serie_trend.index.values
#                     model_ols = LinearRegression()
#                     X = time_serie_trend.loc[:, ['Time']]
#                     y = time_serie_trend.loc[:, "trend"]
#                     res_ols = model_ols.fit(X, y)
#                     y_pred = pd.Series(model_ols.predict(X), index=time_serie_trend.datetime)
#                     formula = f'y = {res_ols.coef_[0]:.2f}x + {res_ols.intercept_:.2f}'

#                     plotseasonal(axes_stl[:, i], res, data_feature, y_pred, formula, group_name)

#                     # variances explained
#                     var_trend = np.var(res.trend)
#                     var_seasonal = np.var(res.seasonal)
#                     var_resid = np.var(res.resid)
#                     var_observed = var_trend+var_seasonal+var_resid

#                     rate_change = (y_pred[-1] - y_pred[0]) / y_pred[0]
#                     r2 = res_ols.score(X, y)
#                     trend_strength = max(0, 1 - (var_resid/np.var(res.trend+res.resid)))
#                     a1, a2, a3 = 0, 0, 0
#                     unified = ((a1 + trend_strength)**1) * ((a2 + r2)**1) * ((a3 + abs(rate_change))**1) * math.copysign(1,rate_change)
#                     unified = 0 if (unified <= 0 and unified >= -0.001) else unified
#                     #unified_trendimpact =  ((a1 + trend_strength)**2) * ((a2 + r2)**1) * ((a3 + abs(rate_change))**(0.5)) * math.copysign(1,rate_change)
#                     #unified_trendimpact = 0 if (unified_trendimpact <= 0 and unified_trendimpact >= -0.001) else unified_trendimpact
#                     #unified_ratechangeimpact =  ((a1 + trend_strength)**(0.5)) * ((a2 + r2)**1) * ((a3 + abs(rate_change))**2) * math.copysign(1,rate_change)
#                     #unified_ratechangeimpact = 0 if (unified_ratechangeimpact <= 0 and unified_ratechangeimpact >= -0.001) else unified_ratechangeimpact

#                     seasonal_strength = max(0, 1 - (var_resid/np.var(res.seasonal+res.resid)))
#                     noise_strength = var_resid/var_observed

#                     stats_df = {"Name": group_name, "Feature": data_feature,
#                                 "Slope": round(res_ols.coef_[0], 3), "ROC": round(rate_change, 3), "R2": round(r2, 3),
#                                 "F_T": round(trend_strength, 3),"Unified Score": round(unified, 3), 
#                                 "F_S": round(seasonal_strength, 3), "F_R": round(noise_strength, 3)}

#                     decomp_stats_table = pd.concat(
#                         [decomp_stats_table, pd.DataFrame(stats_df, index=[0])], ignore_index=True)

#         axes_stl[0, 0].set_ylabel("Observed")
#         axes_stl[1, 0].set_ylabel("Trend")
#         axes_stl[2, 0].set_ylabel("Seasonal")
#         axes_stl[3, 0].set_ylabel("Residual")

#         fig_stl.tight_layout()
#         html_matplotlib_stl = mpld3.fig_to_html(fig_stl)


#         table = decomp_stats_table.to_dict(orient='records')
#         table_columns = [{'name': col, 'id': col}
#                          for col in decomp_stats_table.columns]
#         stored_table = [table, table_columns]
#         set_progress((0, str(len(checklist_values))))
#         return html_matplotlib_stl, decomp_resids, table, table_columns, stored_table, data_features

#     # empty list
#     set_progress((0, str(len(checklist_values))))
#     return None, None, None, None, [None, None], None

@callback(
    Output("download-results", "data"),
    Input("download_decomp_button", "n_clicks"),
    State("ts-decomp-table", "data"),
    State("ts-decomp-table", "columns"),
    prevent_initial_call=True
)
def download_decomp_results(n_clicks, table_data, table_columns):
    df = pd.DataFrame(table_data, columns=[c['name'] for c in table_columns])
    csv_string = df.to_csv(index=False, encoding='utf-8')
    if len(df["Name"].unique()) > 5:
        filename = "all_decomp_results.csv"
    else:
        filename = "_".join(df["Name"].unique()) + "_decomp_results.csv" 

    return dict(content=csv_string, filename=filename)
    
def plotseasonal(axes, res, data_feature, predicted, formula, ts_name):    
    axes[0].plot(res.observed.index, res.observed.values, label=None)
    axes[0].set_xlabel("")
    axes[0].set_title(ts_name)
    #axes[0].legend(loc='upper right')
    
    axes[1].plot(res.trend.index, res.trend.values, label="")
    axes[1].plot(predicted.index, predicted.values, label=formula, color="r")
    axes[1].set_xlabel("") 
    axes[1].legend(loc='upper right')

    ax_idx = 2
    if isinstance(res.seasonal, pd.DataFrame):
        ax_idx = 2
        for i, period in enumerate(res.seasonal):
            axes[ax_idx].plot(res.seasonal.index, res.seasonal[period], label=period)
            axes[2+i].set_xlabel("")
            ax_idx += 1
    else:
        axes[ax_idx].plot(res.seasonal.index, res.seasonal.values)
        axes[ax_idx].set_xlabel("")

    axes[-1].plot(res.resid.index, res.resid.values, marker='.', label=None)
    axes[-1].set_xlabel("")
    return None


@callback(
    Output('map-statistics-container', 'children'),
    [Input('ts-decomp-table', 'selected_rows'),
     Input('statistics-dropdown', 'value'),
     State('ts-decomp-table', 'data'),
     State('timeagg-dropdown', 'value'),
     State('spaceagg-dropdown', 'value')],
    prevent_initial_call=True,
    background=True)
def draw_map_from_selectedrows(selected_rows, data_feature, table_data, time_agg, space_agg):

    if space_agg == "Cell":
        space_feature_name = "location_id"
        wkt_feature_name = "wkt_cell"
    elif space_agg == "TAZ":
        space_feature_name = "taz_name"
        wkt_feature_name = "wkt_taz"
    else:
        space_feature_name = "township_name"
        wkt_feature_name = "wkt_township"

    gdf_data = get_current_dataset(time_agg, space_agg)
    gdf_data = gdf_data[[space_feature_name,wkt_feature_name]].drop_duplicates()

    filtered_table_df = pd.DataFrame([table_data[i] for i in selected_rows])

    filtered_table_df = pd.merge(filtered_table_df, gdf_data, left_on="Name", right_on=space_feature_name)
    filtered_table_df = filtered_table_df.drop(columns=["Name"])
   
    filtered_gdf = gpd.GeoDataFrame(filtered_table_df, geometry=wkt_feature_name)


    if data_feature == "Unified Score" or data_feature == "ROC": 
        # if positive
        if float(filtered_gdf[data_feature].min()) > 0  : 
            classes = [0,0.2,0.4,0.6,0.8, 1]
            colorscale = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026', '#3d010d']
        # if negative 
        elif float(filtered_gdf[data_feature].min()) <= 0 and float(filtered_gdf[data_feature].max()) <=0:
            classes = [-1, -0.8, -0.6, -0.4, -0.2, 0]
            colorscale = ['#10097d','#4239d6','#39aad6','#a4e6a0','#ecffd1', '#ffffb2']
        # if negative and positive
        else: 
            classes = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
            colorscale = ['#10097d','#4239d6','#39aad6','#a4e6a0','#ecffd1', '#ffffb2','#fecc5c', '#fd8d3c', '#f03b20', '#bd0026', '#3d010d']

    else:
        classes = [0,0.2,0.4,0.6,0.8]
        colorscale = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']
    ctg = [str(cls)+"+" for cls in classes]
    colorbar = dlx.categorical_colorbar(categories=ctg,
                                        colorscale=colorscale,
                                        width=300,
                                        height=30,
                                        position="bottomright")

    geojson_layer_statistics = dl.GeoJSON(data=json.loads(filtered_gdf.to_json()),
                         options=dict(style=style_handle),
                         hoverStyle=arrow_function(
                             dict(weight=5, color='#666', dashArray='')),
                         hideout=dict(colorscale=colorscale, classes=classes,
                                      style=style, colorProp=data_feature),
                         id="geojson_layer_statistics")


    map = dl.Map(id="my-map-statistics", children=[dl.TileLayer(), geojson_layer_statistics, colorbar, info_statistics], center=(
        38.74, -9.14), zoom=12, style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"})


    return map
