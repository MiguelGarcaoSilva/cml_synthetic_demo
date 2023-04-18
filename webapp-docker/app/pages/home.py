import logging
from dash import Dash, callback, register_page, dcc, html, Input, State, Output, exceptions, callback_context, no_update, dash_table
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import arrow_function, assign
import plotly.express as px
from dotenv import load_dotenv
from sqlalchemy import create_engine
from statsmodels.tsa.seasonal import STL, seasonal_decompose
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from datetime import date
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import psycopg2
import psycopg2.extras
import time
import math
import mpld3


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


# aggregated by cell
# start = time.process_time()
# query = "SELECT * FROM mob_data_aggregated_hourly_cell_withgeom_view;"
# gdf_mobdata_hourly_cell = gpd.read_postgis(query, engine, geom_col="wkt_cell", crs="EPSG:4326")
# gdf_mobdata_hourly_cell["datetime"] = pd.to_datetime(gdf_mobdata_hourly_cell["one_time"])
# gdf_mobdata_hourly_cell = gdf_mobdata_hourly_cell.drop("one_time", axis=1)
# logging.warning("mob_data_aggregated_hourly_cell_withgeom_view:")
# logging.warning(time.process_time() - start)

# start = time.process_time()
# query = "SELECT * FROM mob_data_aggregated_daily_cell_withgeom_view;"
# gdf_mobdata_daily_cell = gpd.read_postgis(query, engine, geom_col="wkt_cell", crs="EPSG:4326")
# gdf_mobdata_daily_cell["datetime"] = pd.to_datetime(gdf_mobdata_daily_cell["one_time"])
# gdf_mobdata_daily_cell = gdf_mobdata_daily_cell.drop("one_time", axis=1)
# logging.warning("mob_data_aggregated_daily_cell_withgeom_view:")
# logging.warning(time.process_time() - start)

start = time.process_time()
query = "SELECT * FROM mob_data_aggregated_weekly_cell_withgeom_view;"
gdf_mobdata_weekly_cell = gpd.read_postgis(
    query, engine, geom_col="wkt_cell", crs="EPSG:4326")
gdf_mobdata_weekly_cell["datetime"] = pd.to_datetime(
    gdf_mobdata_weekly_cell["one_time"])
gdf_mobdata_weekly_cell = gdf_mobdata_weekly_cell.drop("one_time", axis=1)
logging.warning("mob_data_aggregated_weekly_cell_withgeom_view:")
logging.warning(time.process_time() - start)


start = time.process_time()
query = "SELECT * FROM mob_data_aggregated_monthly_cell_withgeom_view;"
gdf_mobdata_monthly_cell = gpd.read_postgis(
    query, engine, geom_col="wkt_cell", crs="EPSG:4326")
gdf_mobdata_monthly_cell["datetime"] = pd.to_datetime(
    gdf_mobdata_monthly_cell["one_time"])
gdf_mobdata_monthly_cell = gdf_mobdata_monthly_cell.drop("one_time", axis=1)
logging.warning("mob_data_aggregated_monthly_cell_withgeom_view:")
logging.warning(time.process_time() - start)


# aggregated by TAZ
# start = time.process_time()
# query = "SELECT * FROM mob_data_aggregated_hourly_taz_withgeom_view;"
# gdf_mobdata_hourly_taz = gpd.read_postgis(
#     query, engine, geom_col="wkt_taz", crs="EPSG:4326")
# gdf_mobdata_hourly_taz["datetime"] = pd.to_datetime(gdf_mobdata_hourly_taz["one_time"])
# gdf_mobdata_hourly_taz = gdf_mobdata_hourly_taz.drop("one_time", axis=1)
# logging.warning("mob_data_aggregated_hourly_taz_withgeom_view:")
# logging.warning(time.process_time() - start)

start = time.process_time()
query = "SELECT * FROM mob_data_aggregated_daily_taz_withgeom_view;"
gdf_mobdata_daily_taz = gpd.read_postgis(
    query, engine, geom_col="wkt_taz", crs="EPSG:4326")
gdf_mobdata_daily_taz["datetime"] = pd.to_datetime(
    gdf_mobdata_daily_taz["one_time"])
gdf_mobdata_daily_taz = gdf_mobdata_daily_taz.drop("one_time", axis=1)
logging.warning("mob_data_aggregated_daily_taz_withgeom_view:")
logging.warning(time.process_time() - start)

start = time.process_time()
query = "SELECT * FROM mob_data_aggregated_weekly_taz_withgeom_view;"
gdf_mobdata_weekly_taz = gpd.read_postgis(
    query, engine, geom_col="wkt_taz", crs="EPSG:4326")
gdf_mobdata_weekly_taz["datetime"] = pd.to_datetime(
    gdf_mobdata_weekly_taz["one_time"])
gdf_mobdata_weekly_taz = gdf_mobdata_weekly_taz.drop("one_time", axis=1)
logging.warning("mob_data_aggregated_weekly_taz_withgeom_view:")
logging.warning(time.process_time() - start)

start = time.process_time()
query = "SELECT * FROM mob_data_aggregated_monthly_taz_withgeom_view;"
gdf_mobdata_monthly_taz = gpd.read_postgis(
    query, engine, geom_col="wkt_taz", crs="EPSG:4326")
gdf_mobdata_monthly_taz["datetime"] = pd.to_datetime(
    gdf_mobdata_monthly_taz["one_time"])
gdf_mobdata_monthly_taz = gdf_mobdata_monthly_taz.drop("one_time", axis=1)
logging.warning("mob_data_aggregated_monthly_taz_withgeom_view:")
logging.warning(time.process_time() - start)


# aggregated by township
# start = time.process_time()
# query = "SELECT * FROM mob_data_aggregated_hourly_township_withgeom_view;"
# gdf_mobdata_hourly_township = gpd.read_postgis(
#     query, engine, geom_col="wkt_township", crs="EPSG:4326")
# gdf_mobdata_hourly_township["datetime"] = pd.to_datetime(gdf_mobdata_hourly_township["one_time"])
# gdf_mobdata_hourly_township = gdf_mobdata_hourly_township.drop("one_time", axis=1)
# logging.warning("mob_data_aggregated_hourly_township_withgeom_view:")
# logging.warning(time.process_time() - start)

start = time.process_time()
query = "SELECT * FROM mob_data_aggregated_daily_township_withgeom_view;"
gdf_mobdata_daily_township = gpd.read_postgis(
    query, engine, geom_col="wkt_township", crs="EPSG:4326")
gdf_mobdata_daily_township["datetime"] = pd.to_datetime(
    gdf_mobdata_daily_township["one_time"])
gdf_mobdata_daily_township = gdf_mobdata_daily_township.drop(
    "one_time", axis=1)
logging.warning("mob_data_aggregated_daily_township_withgeom_view:")
logging.warning(time.process_time() - start)

start = time.process_time()
query = "SELECT * FROM mob_data_aggregated_weekly_township_withgeom_view;"
gdf_mobdata_weekly_township = gpd.read_postgis(
    query, engine, geom_col="wkt_township", crs="EPSG:4326")
gdf_mobdata_weekly_township["datetime"] = pd.to_datetime(
    gdf_mobdata_weekly_township["one_time"])
gdf_mobdata_weekly_township = gdf_mobdata_weekly_township.drop(
    "one_time", axis=1)
logging.warning("mob_data_aggregated_weekly_township_withgeom_view:")
logging.warning(time.process_time() - start)

start = time.process_time()
query = "SELECT * FROM mob_data_aggregated_monthly_township_withgeom_view;"
gdf_mobdata_monthly_township = gpd.read_postgis(
    query, engine, geom_col="wkt_township", crs="EPSG:4326")
gdf_mobdata_monthly_township["datetime"] = pd.to_datetime(
    gdf_mobdata_monthly_township["one_time"])
gdf_mobdata_monthly_township = gdf_mobdata_monthly_township.drop(
    "one_time", axis=1)
logging.warning("mob_data_aggregated_monthly_township_withgeom_view:")
logging.warning(time.process_time() - start)


numdate = [x for x in range(
    len(gdf_mobdata_monthly_township['datetime'].unique()))]
filtered_gdf_civiltownship = gdf_mobdata_monthly_township[(
    gdf_mobdata_monthly_township['datetime'] == gdf_mobdata_monthly_township['datetime'].unique()[0])].drop("datetime", axis=1)
space_names = sorted(filtered_gdf_civiltownship["township_name"].unique())
colorscale = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']
max_value = filtered_gdf_civiltownship.sum_terminals.max()
classes_scale = math.floor(
    max_value/np.power(10, int(math.log10(max_value))))*np.power(10, int(math.log10(max_value)))
classes = [i * (classes_scale // 5) for i in range(5)]
ctg = [human_format(cls) for i, cls in enumerate(classes)]
colorbar = dlx.categorical_colorbar(categories=ctg,
                                    colorscale=colorscale,
                                    width=300,
                                    height=30,
                                    position="bottomright")

sorted_dates = sorted(gdf_mobdata_monthly_township['datetime'].unique())
min_date_string = np.datetime_as_string(sorted_dates[0], unit='D')
max_date_string = np.datetime_as_string(sorted_dates[-1], unit='D')
marks = {numd: {"label": pd.to_datetime(str(date)).strftime('%d/%m/%Y'), "style": {"writing-mode": "vertical-rl"}}
         for numd, date in zip(numdate, sorted_dates)}
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
geojson_layer = dl.GeoJSON(data=json.loads(filtered_gdf_civiltownship.to_json()),
                           options=dict(style=style_handle),
                           hoverStyle=arrow_function(
                               dict(weight=5, color='#666', dashArray='')),
                           hideout=dict(colorscale=colorscale, classes=classes,
                                        style=style, colorProp="sum_terminals"),
                           id="geojson_layer")

geojson_layer_statistics = dl.GeoJSON(data=json.loads(pd.DataFrame().to_json()),
                        options=dict(style=style_handle),
                        hoverStyle=arrow_function(
                            dict(weight=5, color='#666', dashArray='')),
                        hideout=dict(colorscale=colorscale, classes=classes,
                                    style=style, colorProp="F_S"),
                        id="geojson_layer_statistics")


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
                'Monthly',
                id='timeagg-dropdown', persistence=True, persistence_type="session"
            )
        ], style={'width': '30%', "float": "left", "text-align": "left"}),

        html.Div([
            dcc.Dropdown(
                ["Cell", "TAZ", "Township"],
                'Township',
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

    dbc.Row(dbc.Col(children=[html.Div(children=[dl.Map(id="my-map-mobility", children=[dl.TileLayer(), geojson_layer, colorbar, info], center=(38.74, -9.14), zoom=12, style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"}),
                                                 ], id="map-container"),
                              html.Div(dcc.RangeSlider(
                                  id="range-slider-mobility",
                                  min=numdate[0],
                                  max=numdate[-1],
                                  value=[numdate[0], numdate[0]],
                                  marks=marks,
                                  step=None,
                              ), id="slider_div")
                              ]), style={"margin-bottom": "50px"}),




    dbc.Row(children=[
            dbc.Col(html.Div(dcc.Graph(id="ts-plots-chart"),
                    id="multivariate-time-series_div"), width=6),
            dbc.Col(children=[
                    html.Div(dcc.Checklist(
                        ["All"], [], id="all-checklist", inline=True, style={"margin-top": "50px"}, persistence=True, persistence_type="session")),
                    html.Div(dcc.Checklist(
                        id="space-checklist",
                        options=space_names,
                        value=[],
                        inline=True,
                        style={"height": "200px", "overflow": "scroll", "margin-top": "10px"}, persistence=True, persistence_type="session"), id="checklist_div"),

                    html.Div("Time range to perform TS Decomposition:"),

                    html.Div([
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=min_date_string,
                            max_date_allowed=max_date_string,
                            display_format="DD-MM-YYYY",
                            month_format='MMMM Y',
                            end_date_placeholder_text='MMMM Y',
                            start_date=min_date_string,
                            end_date=max_date_string
                        )],
                        id="date-range-div", style={'textAlign': 'center', "margin": "auto", "margin-top": "10px"}),

                    html.Div([dcc.Dropdown(
                        options=[{"label": "Classic Additive", "value": "seasonal_decompose"}, {
                            "label": "STL", "value": "stl"}],
                        value="seasonal_decompose", id="decompose_algo-dropdown",
                        persistence=True, persistence_type="session")]),

                    html.Div([
                        html.Button(id="button_run",
                                    children="Run TS Decomp!"),
                        html.Button(id="cancel_button_run", children="Cancel Running Job!", style={
                            "margin-left": "15px"})], style={'textAlign': 'center', "margin": "auto", "margin-top": "20px"}),

                    html.Div([html.Progress(id="progress_bar", value="0")],  style={
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

    dbc.Row(children=[dbc.Col(html.Div(dcc.Link(html.Button(id="button_checkfull_viz", children="Check Decomposition Visualizations!"),
            href="/decomp-viz"), style={'textAlign': 'center', "margin": "auto", "margin-top": "30px"}))]),

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
    dbc.Row(dbc.Col(children=[html.Div(children=[dl.Map(id="my-map-statistics", children=[dl.TileLayer(), geojson_layer_statistics, None, info_statistics], center=(38.74, -9.14), zoom=12, style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"}),
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
    Output("checklist-store-memory", "data"),
    Input("space-checklist", "value"),
    Input("all-checklist", "value"),
    State('timeagg-dropdown', 'value'),
    Input('spaceagg-dropdown', 'value'),
    State("checklist-store-memory", "data")
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

    gdf_data = get_dataset(time_agg, space_agg)
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

    return space_selected, all_selected, [space_selected, all_selected]


@callback(
    Output("date-range-div", "children"),
    Input('timeagg-dropdown', 'value'),
    State("spaceagg-dropdown", "value"),
    prevent_initial_call=True
)
def change_between_time_data_range(time_agg, space_agg):
    gdf_data = get_dataset(time_agg, space_agg)

    sorted_dates_datetime = sorted(gdf_data['datetime'].unique())
    if time_agg == "Hourly":
        min_date_string = np.datetime_as_string(
            sorted_dates_datetime[0], unit='m')
        max_date_string = np.datetime_as_string(
            sorted_dates_datetime[-1], unit='m')
        range_selector = [html.Label("Start Time:"),
                          dcc.Input(id="start-time-range", type="datetime-local", value=min_date_string,
                                    min=min_date_string, max=max_date_string, step="3600", style={"margin-left": "5px"}),
                          html.Label("End Time:", style={
                                     "margin-left": "20px"}),
                          dcc.Input(id="end-time-range", type="datetime-local", value=max_date_string, min=min_date_string, max=max_date_string, step="3600", style={"margin-left": "5px"})]
    else:
        min_date_string = np.datetime_as_string(
            sorted_dates_datetime[0], unit='D')
        max_date_string = np.datetime_as_string(
            sorted_dates_datetime[-1], unit='D')
        range_selector = [dcc.DatePickerRange(
            id="date-range",
            min_date_allowed=min_date_string,
            max_date_allowed=max_date_string,
            display_format="DD-MM-YYYY",
            month_format='MMMM Y',
            end_date_placeholder_text='MMMM Y',
            start_date=min_date_string,
            end_date=max_date_string,
        )]
    return range_selector


@callback(
    Output("checklist_div", "children"),
    Input("geojson_layer", "click_feature"),
    State("space-checklist", "value"),
    State('timeagg-dropdown', 'value'),
    Input('spaceagg-dropdown', 'value'),
    prevent_initial_call=True
)
def update_checklist_from_map_and_spacedropdown(selected_space, checklist_values, time_agg, space_agg):
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

        gdf_data = get_dataset(time_agg, space_agg)

        if selected_space:  # prevents from triggering when map is updated but not clicked
            checklist_values = checklist_values + \
                [selected_space["properties"][space_feature_name]]

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
    State("datafeature-dropdown", "value")
)
def update_plot(checklist_values, time_agg, space_agg, data_feature):
    if space_agg == "Cell":
        space_feature_name = "location_id"
    elif space_agg == "TAZ":
        space_feature_name = "taz_name"
    else:
        space_feature_name = "township_name"

    data_feature_dict = {
        'sum_terminals': 'Terminals',
        'sum_roaming_terminals': 'Roamings'
    }
    gdf_data = get_dataset(time_agg, space_agg)

    filtered_gdf_data = gdf_data[gdf_data[space_feature_name].isin(
        checklist_values)]

    fig = px.line(filtered_gdf_data, x="datetime", y=data_feature, color=space_feature_name,
                  line_group=space_feature_name, labels={data_feature: data_feature_dict[data_feature], "datetime": "Time"}, hover_name=space_feature_name)
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

    gdf_data = get_dataset(time_agg, space_agg)
    # Apply the fix_wkt_taz function to the wkt_taz column
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


# TODO: ONLY working for non hourly data
@callback(
    [
        Output("ts-decomp-stl-plots-iframe-memory", "data"),
        Output('ts-decomp-table', 'data'),
        Output('ts-decomp-table', 'columns'),
        Output("ts-decomp-table-store-memory", "data")
    ],
    [Input("url-home", "pathname"),
     Input("button_run", "n_clicks"),
     State("decompose_algo-dropdown", "value"),
     State("ts-decomp-table-store-memory", "data"),
     State('ts-decomp-stl-plots-iframe-memory', "data"),
     State('timeagg-dropdown', 'value'),
     State('spaceagg-dropdown', 'value'),
     State("datafeature-dropdown", "value"),
     State("date-range", "start_date"),
     State("date-range", "end_date"),
     State("checklist-store-memory", "data")],
    running=[
        (Output("timeagg-dropdown", "disabled"), True, False),
        (Output("spaceagg-dropdown", "disabled"), True, False),
        (Output("button_run", "disabled"), True, False),
        (Output("datafeature-dropdown", "disabled"), True, False),
        (Output("cancel_button_run", "disabled"), False, True),
        (Output("button_checkfull_viz", "disabled"), True, False),
        (
            Output("progress_bar", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        ),
    ],
    cancel=[Input("cancel_button_run", "n_clicks")],
    progress=[Output("progress_bar", "value"), Output("progress_bar", "max")],
    prevent_initial_call=True,
    background=True)
def perform_tscomp_ifnotin_cache(set_progress, url, n_clicks, decomp_algo, stored_table, stored_html_matplotlib_stl, time_agg, space_agg, data_feature, start_date, end_date, checklist_values_memory):
    if url != '/home':
        raise exceptions.PreventUpdate

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    if checklist_values_memory == {}:
        raise exceptions.PreventUpdate

    checklist_values = checklist_values_memory[0]

    if trigger_id == "url-home" and stored_table:
        return stored_html_matplotlib_stl, stored_table[0], stored_table[1], stored_table

    if space_agg == "Cell":
        space_feature_name = "location_id"
    elif space_agg == "TAZ":
        space_feature_name = "taz_name"
    else:
        space_feature_name = "township_name"

    gdf_data = get_dataset(time_agg, space_agg)

    filtered_gdf_data = gdf_data[gdf_data[space_feature_name].isin(
        checklist_values)]

    if time_agg == "Hourly":
        datetime_mask = (filtered_gdf_data['datetime'] >= start_hour) & (filtered_gdf_data['datetime'] <= end_hour)
    else:
        datetime_mask = (filtered_gdf_data['datetime'] >= start_date) & (filtered_gdf_data['datetime'] <= end_date)

    filtered_gdf_data = filtered_gdf_data.loc[datetime_mask]

    table_df = filtered_gdf_data[[
        space_feature_name, "datetime", data_feature]]

    if checklist_values:
        fig_stl, axes_stl = plt.subplots(ncols=len(checklist_values), nrows=4, figsize=(
            len(checklist_values)*5, 10), squeeze=False)

        decomp_stats_table = pd.DataFrame(
            columns=["Name", "Slope", "R2", "ROC", "F_T", "Unified Score",  "F_S", "F_R"])
        for i, (group_name, df_group) in enumerate(table_df.groupby(space_feature_name)):
            set_progress((str(i + 1), str(len(checklist_values))))

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
                res = seasonal_decompose(time_serie, model="additive")
            else:  # stl
                res = STL(time_serie).fit()


            time_serie_trend = res.trend.dropna(axis=0).to_frame().reset_index()
            time_serie_trend["Time"] = time_serie_trend.index.values
            model_ols = LinearRegression()
            X = time_serie_trend.loc[:, ['Time']]
            y = time_serie_trend.loc[:, "trend"]
            res_ols = model_ols.fit(X, y)
            y_pred = pd.Series(model_ols.predict(X), index=time_serie_trend.datetime)
            formula = f'y = {res_ols.coef_[0]:.2f}x + {res_ols.intercept_:.2f}'

            plotseasonal(axes_stl[:, i], res,  y_pred, formula, group_name)

            # variances explained
            var_trend = np.var(res.trend)
            var_seasonal = np.var(res.seasonal)
            var_resid = np.var(res.resid)
            var_observed = var_trend+var_seasonal+var_resid

            rete_change = (y_pred[-1] - y_pred[0]) / y_pred[0]
            r2 = res_ols.score(X, y)      
            trend_strength = max(0, 1 - (var_resid/np.var(res.trend+res.resid)))
            comb_trend = rete_change * trend_strength * r2
            seasonal_strength = max(0, 1 - (var_resid/np.var(res.seasonal+res.resid)))
            noise_strength = var_resid/var_observed


            stats_df = {"Name": group_name,
                        "Slope": round(res_ols.coef_[0], 3), "ROC": round(rete_change, 3), "R2": round(r2, 3),
                        "F_T": round(trend_strength, 3),"Unified Score": round(comb_trend, 3), "F_S": round(seasonal_strength, 3), "F_R": round(noise_strength, 3)}

            decomp_stats_table = pd.concat(
                [decomp_stats_table, pd.DataFrame(stats_df, index=[0])], ignore_index=True)

        axes_stl[0, 0].set_ylabel("Observed")
        axes_stl[1, 0].set_ylabel("Trend")
        axes_stl[2, 0].set_ylabel("Seasonal")
        axes_stl[3, 0].set_ylabel("Residual")

        fig_stl.tight_layout()
        html_matplotlib_stl = mpld3.fig_to_html(fig_stl)

        table = decomp_stats_table.to_dict(orient='records')
        table_columns = [{'name': col, 'id': col}
                         for col in decomp_stats_table.columns]
        set_progress((0, str(len(checklist_values))))

        stored_table = [table, table_columns]
        return html_matplotlib_stl, table, table_columns, stored_table

    # empty list
    set_progress((0, str(len(checklist_values))))
    return None, None, None, [None, None]


def plotseasonal(axes, res, predicted, formula, ts_name):
    res.observed.plot(ax=axes[0], legend=False, xlabel="")
    axes[0].set_title(ts_name)
    res.trend.plot(ax=axes[1], label="", xlabel="")
    predicted.plot(ax=axes[1], color="r", label=formula, xlabel="")
    axes[1].legend(loc='upper right')
    res.seasonal.plot(ax=axes[2], legend=False, xlabel="")
    res.resid.plot(ax=axes[3], style=".", legend=False, xlabel="")
    return None



@callback(
    Output('map-statistics-container', 'children'),
    [Input('ts-decomp-table', 'selected_rows'),
     Input('statistics-dropdown', 'value'),
     State('ts-decomp-table', 'data'),
     State('timeagg-dropdown', 'value'),
     State('spaceagg-dropdown', 'value')],
    prevent_initial_call=True)
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

    gdf_data = get_dataset(time_agg, space_agg)[[space_feature_name,wkt_feature_name]].drop_duplicates()

    filtered_table_df = pd.DataFrame([table_data[i] for i in selected_rows])

    filtered_table_df = pd.merge(filtered_table_df, gdf_data, left_on="Name", right_on=space_feature_name)
    filtered_table_df = filtered_table_df.drop(columns=["Name"])
   
    filtered_gdf = gpd.GeoDataFrame(filtered_table_df, geometry=wkt_feature_name)


    if data_feature == "Unified Score" or data_feature == "ROC": 
        # if positive
        if filtered_gdf[data_feature].min() > 0  : 
            classes = [0,0.2,0.4,0.6,0.8, 1]
            colorscale = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026', '#3d010d']
        # if negative 
        elif filtered_gdf[data_feature].min() <= 0 and filtered_gdf[data_feature].max() <=0:
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





def get_dataset(time_agg, space_agg):
    if space_agg == "Cell":
        if time_agg == "Hourly":
            gdf_data = gdf_mobdata_hourly_cell
        elif time_agg == "Daily":
            gdf_data = gdf_mobdata_daily_cell
        elif time_agg == "Weekly":
            gdf_data = gdf_mobdata_weekly_cell
        else:
            gdf_data = gdf_mobdata_monthly_cell
    elif space_agg == "TAZ":
        if time_agg == "Hourly":
            gdf_data = gdf_mobdata_hourly_taz
        elif time_agg == "Daily":
            gdf_data = gdf_mobdata_daily_taz
        elif time_agg == "Weekly":
            gdf_data = gdf_mobdata_weekly_taz
        else:
            gdf_data = gdf_mobdata_monthly_taz
    else:
        if time_agg == "Hourly":
            gdf_data = gdf_mobdata_hourly_township
        elif time_agg == "Daily":
            gdf_data = gdf_mobdata_daily_township
        elif time_agg == "Weekly":
            gdf_data = gdf_mobdata_weekly_township
        else:
            gdf_data = gdf_mobdata_monthly_township

    return gdf_data
