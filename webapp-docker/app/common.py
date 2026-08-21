"""Shared database access, dataset caching and space-aggregation config."""
import logging
import os
import time

import diskcache as dc
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine

# SGBD configs
DB_HOST = os.getenv('PG_HOST')
DB_PORT = os.getenv('PG_PORT')
DB_USER = os.getenv('PG_USER')
DB_DATABASE = os.getenv('PG_DBNAME')
DB_PASSWORD = os.getenv('PG_PASSWORD')

engine_string = "postgresql+psycopg2://%s:%s@%s:%s/%s" % (
    DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_DATABASE)
engine = create_engine(engine_string)

cache = dc.Cache("./dataset_cache")

# feature column and geometry column for each spatial aggregation level
SPACE_CONFIG = {
    "Cell": ("location_id", "wkt_cell"),
    "TAZ": ("taz_name", "wkt_taz"),
    "Township": ("township_name", "wkt_township"),
}


def space_feature(space_agg):
    return SPACE_CONFIG[space_agg][0]


def space_wkt(space_agg):
    return SPACE_CONFIG[space_agg][1]


def get_current_dataset(time_agg, space_agg):
    # Check if data exists in cache
    cache_key = f"{time_agg}_{space_agg}"
    cached_data = cache.get(cache_key)

    if cached_data is not None:
        # Data found in cache, use the cached data
        data = cached_data
        logging.info("Got data from cache.")
    else:
        # Data not found in cache, retrieve it from source and cache it
        data = get_dataset(time_agg, space_agg)
        cache.set(cache_key, data)
        logging.info("Data retrieved from source and cached.")
    return data


def get_dataset(time_agg, space_agg):
    start = time.process_time()
    view_name = f"mob_data_aggregated_{time_agg.lower()}_{space_agg.lower()}_withgeom_view"
    logging.info(view_name)
    query = f"SELECT * FROM {view_name}"
    geom_col = f"wkt_{space_agg.lower()}"
    gdf = gpd.read_postgis(query, engine, geom_col=geom_col, crs="EPSG:4326")
    gdf["datetime"] = pd.to_datetime(gdf["one_time"])
    gdf = gdf.drop("one_time", axis=1)
    # repair invalid geometries once at load time, so callbacks don't have to
    invalid = ~gdf[geom_col].is_valid
    if invalid.any():
        gdf.loc[invalid, geom_col] = gdf.loc[invalid, geom_col].buffer(0)
    logging.info(time.process_time() - start)
    return gdf
