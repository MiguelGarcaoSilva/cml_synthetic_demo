import logging
import dash
from dash import Dash, callback, register_page, dcc, html, Input, State, Output, exceptions, callback_context, dash_table
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine

import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import diskcache as dc

import numpy as np
import pandas as pd
import geopandas as gpd
import os
import time
import re

import stumpy
import mpld3


dash.register_page(__name__)


# SGBD configs
DB_HOST = os.getenv('PG_HOST')
DB_PORT= os.getenv('PG_PORT')
DB_USER = os.getenv('PG_USER')
DB_DATABASE = os.getenv('PG_DBNAME')
DB_PASSWORD = os.getenv('PG_PASSWORD')


engine_string = "postgresql+psycopg2://%s:%s@%s:%s/%s" % (
    DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_DATABASE)
engine = create_engine(engine_string)


layout = dbc.Container([
    dcc.Link('Go back', href='/home'),
    dcc.Location(id='url-decompviz', refresh=False),
    dbc.Row(dbc.Col(html.H4("Seasonal Decomposition"), width={'size': 12, 'offset': 0, 'order': 0}), style={
        'textAlign': 'center', 'paddingBottom': '1%'}),
    dbc.Row(dbc.Col(html.Iframe(id="ts-decomp-stl-plots-iframe", srcDoc="<h3>Loading...</h3>",
            style={"border-width": "5", "width": "100%", "height": "1000px",'paddingBottom': '2%'}), id="ts-decomp-stl-plots")),

    dbc.Row(
        dbc.Col(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label('Max #Motifs per Location:'),
                                dcc.Input(
                                    id='max-motifs-input',
                                    type='number',
                                    value=[],
                                    placeholder = "e.g., '5'",
                                    style={'width': '100%'},
                                    persistence=True, persistence_type="session"
                                )
                            ],
                            width=3,
                            style={'margin-bottom': '10px'}
                        ),

                        dbc.Col(
                            [
                                html.Label('Subsequence Lengths:'),
                                dcc.Input(
                                    id='motifsizes-input',
                                    type='text',
                                    value=[],
                                    placeholder = "e.g., '3,5'",
                                    style={'width': '100%'},
                                    persistence=True, persistence_type="session"
                                )
                            ],
                            width=2,
                            style={'margin-bottom': '10px'}
                        ),

                        dbc.Col(
                            [
                                html.Label('Minimum #Matches:'),
                                dcc.Input(
                                    id='min-matches-input',
                                    type='number',
                                    value=[],
                                    placeholder = "e.g., '3'",
                                    style={'width': '100%'},
                                    persistence=True, persistence_type="session"
                                )
                            ],
                            width=2,
                            style={'margin-bottom': '10px'}
                        ),
                        dbc.Col(
                            [
                                html.Label('Maximum #Matches:'),
                                dcc.Input(
                                    id='max-matches-input',
                                    type='number',
                                    value=[],
                                    placeholder = "e.g., '10'",
                                    style={'width': '100%'},
                                    persistence=True, persistence_type="session"
                                )
                            ],
                            width=2,
                            style={'margin-bottom': '10px'}
                        ),

                        dbc.Col(
                            [
                                html.Label('Data Features:'),
                                dcc.Dropdown(
                                    id='datafeature-dropdown-mp',
                                    options=[],
                                    multi=True,
                                    value=[],
                                    style={'width': '100%'},
                                    persistence=True, persistence_type="session"
                                )
                            ],
                            width=3,
                            style={'margin-bottom': '10px'}
                        )

                    ],
                    style={'flex-wrap': 'wrap'}
                ),

                dbc.Row(
                    [
                    dbc.Col(
                        [
                            dcc.Checklist(
                                id='normalization-checklist',
                                options=[
                                    {'label': 'Normalize', 'value': True},
                                ],
                                value=[True],
                                style={'width': '100%'},
                                persistence=True, persistence_type="session"
                            ),
                        ],
                        width=2,
                        style={'margin-bottom': '10px'}
                    ),
                    dbc.Col(
                        [
                            dcc.Checklist(
                                id='guided-search-checklist',
                                options=[
                                    {'label': 'Complexity Correction', 'value': True},
                                ],
                                value=[False],
                                style={'width': '100%'},
                                persistence=True, persistence_type="session"
                            ),
                        ],
                        width=2,
                        style={'margin-bottom': '10px'}
                    ),
                    dbc.Col(
                        [
                            html.Label('Actionability Bias:'),
                            dcc.Dropdown(
                                id='av-options-dropdown',
                                options=[
                                    {'label': day, 'value': day} for day in [
                                        'Weekends', 'Weekdays', 'Mondays', 'Tuesdays',
                                        'Wednesdays', 'Thursdays', 'Fridays', 'Saturdays',
                                        'Sundays', 'Mornings', 'Afternoons', 'Nights'
                                    ]
                                ],
                                value=[],
                                multi=False,
                                style={'width': '100%'}
                            ),
                        ],
                        width=4,
                        style={'margin-bottom': '10px'}
                    ),
                    dbc.Col(
                            [
                                html.Label('Top-K MP:'),
                                dcc.Input(
                                    id='top-k-mp-input',
                                    type='number',
                                    value=[],
                                    placeholder = "e.g., '1'",
                                    style={'width': '100%'},
                                    persistence=True, persistence_type="session"
                                )
                            ],
                            width=2,
                            style={'margin-bottom': '10px'}
                    ),
                    dbc.Col(
                        [
                            html.Label("Unified Weights:"),
                            dcc.Input(
                                id='weights-input',
                                type='text',
                                value=[],
                                placeholder = "e.g., '0.33,0.33,0.33'",
                                style={'width': '100%'},
                                persistence=True, persistence_type="session"
                            )
                        ],
                        width=2,
                        style={'margin-bottom': '10px'}
                    ),
                    dbc.Col(
                            [
                                html.Label('Top-K Unified Score:'),
                                dcc.Input(
                                    id='top-k-unifiedscore-input',
                                    type='number',
                                    value=[],
                                    placeholder = "e.g., '5'",
                                    style={'width': '100%'},
                                    persistence=True, persistence_type="session"
                                )
                            ],
                            width=2,
                            style={'margin-bottom': '10px', 'display': 'none'}
                    ),
                    ]),

                dbc.Button(
                    "Find Motifs in Residuals", 
                    id="matrixprofile_button", 
                    className="mr-2",
                    color="primary"
                ),
                dbc.Button(
                    "Cancel", 
                    id="cancel_matrixprofile_button", 
                    className="mr-2",
                    color="danger"
                ),
                dbc.Progress(
                    id="matrixprofile-progress_bar",
                    value=0
                ),
            ], 
            width={'size': 12, 'offset': 0, 'order': 0}
        ),
        style={'textAlign': 'center', 'paddingBottom': '1%'}
    ),

    dbc.Row(children=[dbc.Col(html.Div(dash_table.DataTable(id="resid-mp-table", editable=True,
                                                            filter_action="native",
                                                            sort_action="native",
                                                            sort_mode="multi",
                                                            column_selectable="single",
                                                            row_selectable="single",
                                                            row_deletable=False,
                                                            selected_columns=[],
                                                            selected_rows=[],
                                                            page_action="native",
                                                            page_current=0,
                                                            page_size=10, persistence=True, persistence_type="session", persisted_props=["columns.name", "data"]),
                                       id="resid-mp-table_div", style={'textAlign': 'center', "margin": "auto", "margin-top": "30px"}))]),

    dbc.Row(children=[
        dbc.Col(dbc.Button("Download MP Results!", id="button_download_mp", className="mr-2", color="primary", disabled=True), style={ "margin": "auto", "margin-top": "30px", "text-align": "center"}),
        dcc.Download(id="download-mp-results")
        ]),

    dbc.Row(dbc.Col(html.Iframe(id="ts-decomp-resid_matrixprofile-motif-plot-iframe", srcDoc="<h3>Loading...</h3>",
           style={"border-width": "5", "width": "100%", "height": "400px",'paddingBottom': '2%'}), id="ts-decomp-resid_matrixprofile-plots")),
    
    # solves race condition of callback running faster then the rendering of the page layout
    dcc.Interval(id='interval-component', interval=1*1000, max_intervals=1)
])


cache = dc.Cache("/path/to/cache_directory")

@callback(
    Output('av-options-dropdown', 'options'),
    Output('av-options-dropdown', 'multi'),
    Input('av-options-dropdown', 'value'),
    State('av-options-dropdown', 'options'),
    State('timeagg-store-memory', 'data')
)
def update_dropdown_avoptions_dropdown(values, options, time_agg):
    if time_agg == "Hourly":
        if 'Weekdays' in values or 'Weekends' in values:
            return options, False
        else:
            return options, True
    else:
        return [option for option in options if option not in ["Morning","Afternoon","Night"]], True

@callback(
    Output("download-mp-results", "data"),
    Input("button_download_mp", "n_clicks"),
    State("resid-mp-table", "data"),
    State("resid-mp-table", "columns"),
    prevent_initial_call=True
)
def download_mp_results(n_clicks, table_data, table_columns):
    df = pd.DataFrame(table_data, columns=[c['name'] for c in table_columns])
    csv_string = df.to_csv(index=False, encoding='utf-8')
   
    df_unique_names = df['ID'].str.split("_", n = 1, expand = True)[0].unique()
    df_unique_features = df['Features'].unique()
    df_unique_m = df['m'].unique().astype(str)
    filename = "mp_results_{}_{}_{}.csv".format("_".join(df_unique_names),
                                                 "_".join(df_unique_features), 
                                                 "_".join(df_unique_m))

    return dict(content=csv_string, filename=filename)




@callback(
    Output("ts-decomp-stl-plots-iframe", "srcDoc"),
    Input("url-decompviz", "pathname"),
    Input("ts-decomp-stl-plots-iframe-memory", "data"),
    Input("interval-component", "n_intervals"),
    prevent_initial_call=True
)
def render_plots(url, html_matplotlib_stl, interval):
    if url != "/decomp-viz":
        raise exceptions.PreventUpdate
    if html_matplotlib_stl is None:
        return "<h3>Loading...</h3>", "<h3>Loading...</h3>"
    return html_matplotlib_stl

@callback(
    Output("datafeature-dropdown-mp", "options"),
    Input("datafeature-store-memory", "data")
)
def update_datafeature_dropdown(datafeatures_decomp):
    return datafeatures_decomp

@callback(
    [ 
        Output('resid-mp-table', 'data'),
        Output('resid-mp-table', 'columns'),
        Output("resid-mp-table-store-memory", "data"),
        Output("motifsizes-store-memory", "data")
    ],
    [   
        Input("matrixprofile_button", "n_clicks"),
        State("max-motifs-input", "value"),
        State('motifsizes-input', 'value'),
        State('min-matches-input', 'value'),
        State('max-matches-input', 'value'),
        State("datafeature-dropdown-mp", "value"),
        State("normalization-checklist", "value"),
        State("guided-search-checklist", "value"),
        State("av-options-dropdown", "value"),
        State("top-k-mp-input", "value"),
        State("weights-input", "value"),
        State("top-k-unifiedscore-input", "value"),
        State("resid-mp-table-store-memory", "data"),
        State("ts-decomp-resids-store-memory", "data")
    ],
    running=[
        (Output("matrixprofile_button", "disabled"), True, False),
        (Output("cancel_matrixprofile_button", "disabled"), False, True),
        (Output("button_download_mp", "disabled"), True, False),
        (
            Output("matrixprofile-progress_bar", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        ),
    ],
    cancel=[Input("cancel_matrixprofile_button", "n_clicks")],
    progress=[Output("matrixprofile-progress_bar", "value"), Output("matrixprofile-progress_bar", "max")],
    prevent_initial_call=True,
    background=True)
def perform_matrixprofile_ifnotin_cache(set_progress, n_clicks, max_motifs, subsequence_lengths, motif_min_matches, motif_max_matches, data_features, normalize, simplicity_bias, actionability_options, topk_mp, unified_weights, k_unified, stored_table_mp, stored_decomp_resid):

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]


    if trigger_id == "matrixprofile_button" and (n_clicks is None) and not stored_table_mp:
        raise exceptions.PreventUpdate

    subsequence_lengths = [int(s.strip()) for s in subsequence_lengths.split(',')]
    
    if stored_table_mp and (n_clicks is None):
        return stored_table_mp[0], stored_table_mp[1], stored_table_mp, subsequence_lengths    

    if stored_decomp_resid:

        # Create an empty dataframe
        df = pd.DataFrame()

        # Iterate over the decomp_resids dictionary
        for key, (resid_index, resid_values) in stored_decomp_resid.items():
            location_name, data_feature = key.split("_",1)

            # Create a temporary dataframe for the current location_name and data_feature
            temp_df = pd.DataFrame({data_feature: resid_values}, index=pd.to_datetime(resid_index))

            # Assign the location_name as a column level
            temp_df.columns = pd.MultiIndex.from_tuples([(location_name, data_feature)])

            # Concatenate the temporary dataframe with the main dataframe
            df = pd.concat([df, temp_df], axis=1)

        
        mp_stats_table = pd.DataFrame()

        for i, (location_name, location_time_series) in enumerate(df.groupby(level=0, axis=1)):
            set_progress((str(i + 1), str(len(stored_decomp_resid))))
            location_time_series = location_time_series.droplevel(level=0, axis=1).dropna()

            if len(data_features) == 1:
                mp, mp_stats_table_loc = perform_matrixprofile(location_time_series, location_name, data_features[0],
                                                                normalize, simplicity_bias, actionability_options, subsequence_lengths,k=topk_mp, unified_weights=unified_weights,
                                                                min_neighbors=motif_min_matches-1, max_matches=motif_max_matches, max_motifs=max_motifs, k_unified=k_unified)
                mp_stats_table = pd.concat([mp_stats_table, mp_stats_table_loc], ignore_index=True)   
            else:

                mp, mp_stats_table_loc = perform_multidimmatrixprofile(location_time_series, location_name, data_features,
                                                                normalize, simplicity_bias, actionability_options, subsequence_lengths, unified_weights=unified_weights,
                                                                min_neighbors=motif_min_matches-1, max_matches=motif_max_matches, max_motifs=max_motifs, num_dims=None, include=None, k_unified=k_unified)
                mp_stats_table = pd.concat([mp_stats_table, mp_stats_table_loc], ignore_index=True)   


        table = mp_stats_table.to_dict(orient='records')
        table_columns = [{'name': col, 'id': col}
                         for col in mp_stats_table.columns]
        set_progress((0, str(len(stored_decomp_resid))))

        stored_table_mp = [table, table_columns]
        return table, table_columns, stored_table_mp, subsequence_lengths


    # empty list
    set_progress((0, str(len(stored_decomp_resid))))
    return  None, None, [None, None], None


#function triggers with user selecting rows in the table and update the plot with the latest
@callback(
    Output("ts-decomp-resid_matrixprofile-motif-plot-iframe","srcDoc"),
    Input("resid-mp-table", "selected_rows"),
    State("resid-mp-table", "data"),
    State("resid-mp-table", "columns"),
    State("ts-decomp-resids-store-memory", "data"),
    prevent_initial_call=True
)
def render_plots_motifs(selected_rows, table_data, table_columns, stored_decomp_resid):

    if selected_rows is None:
        raise exceptions.PreventUpdate
    
    if stored_decomp_resid is None:
        raise exceptions.PreventUpdate
    motif_row = table_data[selected_rows[0]]


    location_name = motif_row["ID"].split("_",1)[0]
    data_features = motif_row["Features"].split(",")
    m = motif_row["m"]
    motif_indexes = motif_row["Indices"]
    motif_indexes = re.findall(r'\d+', motif_indexes)
    motif_indexes = np.array([int(i) for i in motif_indexes])

    if len(data_features) <= 1:
        data_feature = data_features[0]
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(10, 3), squeeze=False)
        #create time serie as dataframe from the stored_decomp_resid with index as datetime
        time_series = pd.DataFrame({data_feature: stored_decomp_resid[location_name+"_"+data_feature][1]},
                                index=pd.to_datetime(stored_decomp_resid[location_name+"_"+data_feature][0]))
        
        #removed nones because of the nan values in the resid
        time_series = time_series.dropna()

        #plot as a line and original datetime index
        axes[0,2].plot(time_series, color='black', linewidth=0.5, alpha=0.5)

        colors = plt.cm.tab20(np.linspace(0, 1, len(motif_indexes)))
        axes[0,0].set_prop_cycle('color', colors)
        axes[0,1].set_prop_cycle('color', colors)
        axes[0,2].set_prop_cycle('color', colors)
        #get the time serie motif
        for index in motif_indexes:
            subsequence_match = time_series[index : index + m]
            normalized_subsequence_match = (subsequence_match - np.mean(subsequence_match))/np.std(subsequence_match)
            axes[0,0].plot(normalized_subsequence_match.values)
            axes[0,1].plot(subsequence_match.values)
            axes[0,2].plot(subsequence_match, linewidth=2) 
    
    else:
        fig, axes = plt.subplots(ncols=3, nrows=len(data_features), figsize=(10, 3), squeeze=False)

        #create multidim time serie as dataframe from the stored_decomp_resid with index as datetime
        time_series = {}
        for data_feature in data_features:
            #removed nones because of the nan values in the resid
            time_series[data_feature] = pd.DataFrame({data_feature: stored_decomp_resid[location_name+"_"+data_feature][1]},
                                index=pd.to_datetime(stored_decomp_resid[location_name+"_"+data_feature][0])).dropna()
    

        for k, data_feature in enumerate(data_features):
            axes[k,2].plot(time_series[data_feature], color='black', linewidth=0.5, alpha=0.5)
            colors = plt.cm.tab20(np.linspace(0, 1, len(motif_indexes)))
            axes[k,0].set_prop_cycle('color', colors)
            axes[k,1].set_prop_cycle('color', colors)
            axes[k,2].set_prop_cycle('color', colors)
            #get the time serie motif
            for index in motif_indexes:
                subsequence_match = time_series[data_feature][index : index + m]
                normalized_subsequence_match = (subsequence_match - np.mean(subsequence_match))/np.std(subsequence_match)
                axes[k,0].plot(normalized_subsequence_match.values)
                axes[k,1].plot(subsequence_match.values)
                axes[k,2].plot(subsequence_match, linewidth=2)
            
    axes[0,0].set_title("Z-Normalized Subsequences")
    axes[0,1].set_title("Raw Subsequences")
    axes[0,2].set_title("Motif in Residual TS")

    fig.tight_layout()
    return mpld3.fig_to_html(fig)


# k - number of top k smallest distances used to construct the matrix profile.
# min_neighbors - min of subsequences in motif to be considered.
# max_matches - max of subsequences in a motif.
# max_motifs - max number of motifs to be returned
def perform_matrixprofile(time_series, location_name, data_feature, normalize ,simplicity_bias, actionability_options, m_dict, k = 1, unified_weights="0.33,0.33,0.33", min_neighbors=2, max_matches=10, max_motifs=10, k_unified=None):

    mp_stats_table = pd.DataFrame()

    time_series  = time_series[data_feature]

    motif_index = 0

    for i, m in enumerate(m_dict):

        # k = 1 (default), the first column is the matrix profile
        # the second column consists of the matrix profile indices
        # the third and fourth column consists of the left and right matrix profile indices, respectively
        # left matrix profile indices are the indices of the subsequence that is the closest neighbors to the left of the  current subsequence 
        # right matrix profile indices """ to the right of the current subsequence
        # Example:
        #[ [0.0 4 -1 4]
        #  [0.91 5 -1 5]
        #  [1.39 8 0 8]
        #  [2.66 7 0 7]
        #  [0.0 0 0 8]
        #  [0.91 1 1 7]
        #  [2.03 1 1 8]
        #  [1.21 1 1 -1]
        #  [1.39 2 2 -1]]
        # k > 1, the output array will contain 2 * k + 2 columns. 
        # The first k columns (i.e., out[:, :k]) consists of the top-k matrix profile
        # the next set of k columns (i.e., out[:, k:2k]) are the corresponding top-k matrix profile indices, 
        # and the last two columns  are top-1 left and right matrix profile indices
        # Example k = 3:
        # [[0.0 1.57 2.74 4 8 2 -1 4]
        #  [0.91 1.21 2.03 5 7 6 -1 5]
        #  [1.39 2.74 2.74 8 0 4 0 8]
        #  [2.66 2.74 2.90 7 8 0 0 7]
        #  [0.0 1.57 2.74 0 8 2 0 8]
        #  [0.91 1.82 3.26 1 7 3 1 7]
        #  [2.03 2.77 3.00 1 2 4 1 8]
        #  [1.21 1.82 2.66 1 5 3 1 -1]
        #  [1.39 1.57 1.57 2 4 0 2 -1]]
        
        out = stumpy.stump(time_series.to_numpy(), m=m, k=k, normalize=normalize)

        if k == 1:
            mp = out[:, 0]
        else:
            #TODO: the top k can be trivial between them, so can be biased. we can increase k and remove the trivial ones
            mp = out[:, :k].mean(axis=1)

        av = np.ones(len(time_series) - m + 1)
        if simplicity_bias:
            av = make_av_simplicitybias(time_series, m)

        if actionability_options:
            av *= make_av_actionability(time_series, m, actionability_options)

        cmp = mp + ((1-av) * np.max(mp))
        mp = cmp

        motif_distances, motif_indices = stumpy.motifs(time_series.to_numpy(), mp, min_neighbors=min_neighbors+1, max_distance=None, cutoff=np.inf, max_matches=max_matches, max_motifs=max_motifs, normalize = normalize)
        
        
        for motif_indice, match_indices in enumerate(motif_indices):

            #remove filling values of -1 and Nansfrom motif_indices and match_distances
            match_indices = match_indices[match_indices != -1]
            match_distances = motif_distances[motif_indice]
            match_distances = match_distances[~np.isnan(match_distances)]

            #if is empty, skip
            if len(match_indices) == 0:
                continue

            #get the time serie motif
            subsequence = time_series[match_indices[0]:match_indices[0]+m]

            #minmax normalize subsequence
            norm_subsequence = (subsequence - np.min(subsequence)) / (np.max(subsequence) - np.min(subsequence))
            ce_norm_subsequence = subsequence_complexity(norm_subsequence)
            norm_ce_norm_subsequence = ce_norm_subsequence/np.sqrt(len(subsequence)-1)

            #complexity_zerocrossings = subsequence_complexity_zerocrossings(subsequence)
            max_dist = np.max(match_distances)
            min_dist = np.min(match_distances[1:])

            if not k_unified:
                med_dist = np.median(match_distances[1:])
            else:
                med_dist = np.median(match_distances[1:int(k_unified)+1])
            
            #D is distance profile between the motif and Time series
            D = stumpy.mass(subsequence.to_numpy(), time_series.to_numpy(), normalize=normalize)
            max_allowed_dist = np.nanmax([np.nanmean(D) - 2.0 * np.nanstd(D), np.nanmin(D)])
            excl_zone = np.ceil(m/4)

            weights = list(map(float, unified_weights.split(",")))
            w1, w2, w3 = weights[0], weights[1], weights[2]
            unified = w1 * (1-(med_dist/max_allowed_dist)) + w2 * (len(match_indices)/(len(time_series)-excl_zone)) + w3 * norm_ce_norm_subsequence

            #remove timepoints from time series in match all indices + m
            time_series_nomatches = time_series.copy()
            #list of indexes to remove
            indexes_to_remove = [i for index in match_indices for i in range(index, index + m)]
            #put zero in the indexes to remove
            time_series_nomatches[indexes_to_remove] = 0

            #calculate variance explained by the motif
            variance_explained = 100 - (np.var(time_series_nomatches)*100)/np.var(time_series)

            stats_df = {"ID": location_name+"_"+str(motif_index), "Features":data_feature, "m":m, 
                        "#Matches": len(match_indices)-1,
                         "Indices":str(match_indices),
                         "CE": np.around(norm_ce_norm_subsequence,3), "Score Unified": np.around(unified,3),
                         "max(dists)": np.around(max_dist,3), "min(dists)": np.around(min_dist,3),
                         "med(dists)": np.around(med_dist,3),
                         "Explained Var(%)": np.around(variance_explained,2)
                         }
            
            mp_stats_table = pd.concat(
                [mp_stats_table, pd.DataFrame.from_records([stats_df])], ignore_index=True)
            
            motif_index += 1


    return mp, mp_stats_table


# k - number of top k smallest distances used to construct the matrix profile.
# min_neighbors - min of subsequences in motif to be considered.
# max_matches - max of subsequences in a motif.
# max_motifs - max number of motifs to be returned
# num_dims - number of dimensions (k + 1) required for discovering all motifs
# include - A list of (zero-based) indices corresponding to the dimensions in T that must be included in the constrained multidimensional motif search.
def perform_multidimmatrixprofile(time_series, location_name, data_features, normalize, simplicity_bias, actionability_options, m_dict, unified_weights="0.33,0.33,0.33",
                                   min_neighbors=2, max_matches=10, max_motifs=10, num_dims=None, include = None, k_unified=None):

    mp_stats_table = pd.DataFrame()

    multivar_time_serie  = time_series[data_features]

    motif_index = 0

    for i, m in enumerate(m_dict):

        logging.warning(multivar_time_serie)

        multivar_time_serie_array = time_series.to_numpy().T
        
        #mp - (numpy.ndarray) – The multi-dimensional matrix profile. 
        #Each row of the array corresponds to each matrix profile for a given dimension 
        # (i.e., the first row is the 1-D matrix profile and the second row is the 2-D matrix profile).

        #mp_indices - The multi-dimensional matrix profile index where each row of 
        # the array corresponds to each matrix profile index for a given dimension.
        mp, mp_indices = stumpy.mstump(multivar_time_serie_array, m=m, include=include, normalize=normalize)


        #av = np.ones(len(multivar_time_serie_array[0]) - m + 1)

        #TODO: bias basta fazer av multidim?
        #if simplicity_bias:
        #    av = make_av_simplicitybias(time_series, m)

        #if actionability_options:
        #    av *= make_av_actionability(time_series, m, actionability_options)

        #TODO: works para multivariate?
        #cmp = mp + ((1-av) * np.max(mp))
        #mp = cmp

        motif_distances, motif_indices, motif_subspaces, motif_mdls = stumpy.mmotifs(multivar_time_serie_array, mp, mp_indices, min_neighbors=min_neighbors+1, max_distance=None,
                                                                     cutoffs=np.inf, max_matches=max_matches, max_motifs=max_motifs, k=num_dims, include=include, normalize=normalize)       
        for motif_indice, match_indices in enumerate(motif_indices):

            dimensions = motif_subspaces[motif_indice]
            
            logging.warning(dimensions)

            #remove filling values of -1 and Nansfrom motif_indices and match_distances
            match_indices = match_indices[match_indices != -1]
            match_distances = motif_distances[motif_indice]
            match_distances = match_distances[~np.isnan(match_distances)]

            #if is empty, skip
            if len(match_indices) == 0:
                continue

            #get the multidim time serie motif in the dimensions
            multivar_subsequence = multivar_time_serie_array[dimensions][:,match_indices[0]:match_indices[0]+m]


            #minmax normalize subsequence
            #norm_subsequence = (subsequence - np.min(subsequence)) / (np.max(subsequence) - np.min(subsequence))
            #ce_norm_subsequence = subsequence_complexity(norm_subsequence)
            #norm_ce_norm_subsequence = ce_norm_subsequence/np.sqrt(len(subsequence)-1)
            norm_ce_norm_subsequence = 0

            #complexity_zerocrossings = subsequence_complexity_zerocrossings(subsequence)
            max_dist = np.max(match_distances)
            min_dist = np.min(match_distances[1:])
            avg_dist = np.mean(match_distances[1:])
            std_dist = np.std(match_distances[1:])
            med_dist = np.median(match_distances[1:])
            
            #D is distance profile between the motif and Time serie
            #D = stumpy.mass(subsequence.to_numpy(), time_series.to_numpy(), normalize = normalize)
            #max_allowed_dist = np.nanmax([np.nanmean(D) - 2.0 * np.nanstd(D), np.nanmin(D)])
            #excl_zone = np.ceil(m/4)

            weights = list(map(float, unified_weights.split(",")))
            w1, w2, w3 = weights[0], weights[1], weights[2]

            #TODO: unified score
            #unified = w1 * (1-(med_dist/max_allowed_dist)) + w2 * (len(match_indices)/(len(time_serie)-excl_zone)) + w3 * norm_ce_norm_subsequence
            unified = 0

            stats_df = {"ID": location_name+"_"+str(motif_index), "Features":",".join([data_features[i] for i in dimensions]), "m":m, 
                        "#Matches": len(match_indices)-1,
                         "Indices":str(match_indices),
                         "CE": np.around(norm_ce_norm_subsequence,3), "Score Unified": np.around(unified,3),
                         "max(dists)": np.around(max_dist,3), "min(dists)": np.around(min_dist,3),
                         "med(dists)": np.around(med_dist,3)
                         }
            
            mp_stats_table = pd.concat(
                [mp_stats_table, pd.DataFrame.from_records([stats_df])], ignore_index=True)
            
            motif_index += 1

    return mp, mp_stats_table



def subsequence_complexity(x):
    return np.sqrt(np.sum(np.square(np.diff(x))))


def subsequence_complexity_zerocrossings(x):
    x = (x - np.mean(x)) / np.std(x)
    return np.sum(np.diff(np.sign(x)) != 0)


def make_av_simplicitybias(data, m):
    av = np.zeros(len(data) - m + 1)
    for i in range(len(data) - m + 1):
        x = data[i:i+m]
        x = (x - np.mean(x)) / np.std(x)
        av[i] = subsequence_complexity(x)

    av = (av - np.min(av)) / (np.max(av) - np.min(av))
    return av

def make_av_actionability(data, m, options):
    days = [day for day in options if day in ["Mondays","Tuesdays","Wednesdays","Thursdays","Fridays","Saturdays","Sundays"]]
    dayparts = [daypart for daypart in options if daypart in ["Mornings","Afternoons","Nights"]]
    if options == "Weekends":
        return make_av_weekend(data, m, True)
    elif options == "Weekdays":
        return make_av_weekend(data, m, False)
    elif days: # can contain dayparts
        return make_av_days(data, m, days, dayparts) 
    elif dayparts:
        return make_av_dayparts(data, m, dayparts)
    else:
        return np.ones(len(data) - m + 1)

def make_av_weekend(data, m, weekends=True):
    av = np.zeros(len(data) - m + 1)
    for i in range(len(data) - m + 1):
        if weekends:
            # 0 if weekday, 1 if weekend
            if data.index[i].weekday() >= 5:
                av[i] = 1
        else:
            # 0 if weekend, 1 if weekday
            if data.index[i].weekday() < 5:
                av[i] = 1
    return av

def make_av_days(data, m, days, dayparts):
    #days to index
    days_to_value = dict({"Mondays": 0, "Tuesdays": 1, "Wednesdays": 2, 
                          "Thursdays": 3, "Fridays": 4, "Saturdays": 5, "Sundays": 6})
    days = [days_to_value[day] for day in days]
    av = np.zeros(len(data) - m + 1)
    for i in range(len(data) - m + 1):
        if data.index[i].weekday() in days:
            if dayparts:
                for part in dayparts:
                    if part == "Mornings":
                        if data.index[i].hour >= 6 and data.index[i].hour < 12:
                            av[i] = 1
                    elif part == "Afternoons":
                        if data.index[i].hour >= 12 and data.index[i].hour < 18:
                            av[i] = 1
                    else: # Night
                        av[i] = 1
            else:
                av[i] = 1
  
                
    return av

def make_av_dayparts(data, m, dayparts):
    av = np.zeros(len(data) - m + 1)
    for i in range(len(data) - m + 1):
        for part in dayparts:
            if part == "Mornings":
                if data.index[i].hour >= 6 and data.index[i].hour < 12:
                    av[i] = 1
            elif part == "Afternoons":
                if data.index[i].hour >= 12 and data.index[i].hour < 18:
                    av[i] = 1
            else: # Night
                av[i] = 1
    return av
