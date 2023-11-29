#!/usr/bin/python3
from dotenv import load_dotenv, main
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from sqlalchemy import create_engine, Integer
from geoalchemy2 import Geometry

import psycopg2
import psycopg2.extras
import logging
import json
import os


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Script")
    # SGBD configs
    DB_HOST = os.getenv('PG_HOST')
    DB_USER = os.getenv('PG_USER')
    DB_DATABASE = os.getenv('PG_DBNAME')
    DB_PORT = os.getenv('PG_PORT')
    DB_PASSWORD = os.getenv('PG_PASSWORD')


    engine_string_synthetic = "postgresql+psycopg2://%s:%s@%s:%s/%s" % (
        DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_DATABASE)
    engine_synthetic= create_engine(engine_string_synthetic)


    DB_CONNECTION_STRING = "host=%s port=%s dbname=%s user=%s password=%s" % (
        DB_HOST, DB_PORT, DB_DATABASE, DB_USER, DB_PASSWORD)

          
    dbConn = None
    cursor = None
    try:
        dbConn = psycopg2.connect(DB_CONNECTION_STRING)
        cursor = dbConn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # Print PostgreSQL details
        print("PostgreSQL server information")
        print(dbConn.get_dsn_parameters(), "\n")
        # Executing a SQL query
        cursor.execute("SELECT version();")
        # Fetch result
        record = cursor.fetchone()
        print("You are connected to - ", record, "\n")
    except Exception as e:
        print("Error while connecting to PostgreSQL", e)
    finally:
        if dbConn:

            #path_eixos_folder = "../../Data/CML Data/VODAFONE_EIXOS_2021/"
            path_grelha_folder = "/wd/VODAFONE_GRELHA_2021/"
            path_taz_folder = "/wd/TAZ/"
            path_data_synthetic = "/wd/CML Mobility/"

            # Populating Country Table
            logging.info("Populating Database with %s into %s..." %
                         ("country_list.csv", "Country"))
            df_countries = pd.read_csv(os.path.join(path_grelha_folder, "country_list.csv"), usecols=[
                                       "name", "alpha-3", "country-code"])


            df_countries_mapping = pd.read_csv(os.path.join(
                path_grelha_folder, "df_country_mapping.csv"), usecols=["country_string", "closest_match"])
            country_mapping_dict = dict(zip(
                df_countries_mapping["closest_match"], df_countries_mapping["country_string"]))

            df_countries["name"] = df_countries["name"].replace(
                country_mapping_dict)

            country_to_id_dict = dict(
                zip(df_countries["name"], df_countries["country-code"]))
            df_countries.columns = ["country_name",
                                    "alpha3_iso_code", "numeric_iso_code"]
            df_countries.to_sql("country", engine_synthetic,
                                index=False, if_exists="append")

            # Populating CivilTownship Table
            logging.info("Populating Database with %s into %s..." %
                         ("freguesias-metadata.json", "CivilTownship"))
            with open(os.path.join(path_grelha_folder, "freguesias-metadata.json"), 'r') as f:
                data = json.loads(f.read())
            df_townships = pd.json_normalize(data, record_path=["d"])
            df_townships = df_townships[["dicofre", "freguesia", "concelho", "distrito"]].rename(
                columns={"freguesia": "township_name"})
            df_townships = df_townships.loc[df_townships['distrito'] == "Lisboa"]
            df_townships = df_townships.loc[df_townships['concelho'] == "Lisboa"]

            df_townships_boundaries = gpd.read_file(os.path.join(path_grelha_folder, "freguesias.geojson")).rename(
                columns={"NOME": "township_name", "FREGUESIAS53": "old_township_name"})
            gdf_townships_boundaries = gpd.GeoDataFrame(
                df_townships_boundaries, crs="EPSG:4326")
            gdf_townships_boundaries = gdf_townships_boundaries[[
                "township_name", "old_township_name", "geometry"]]
            gdf_townships = gdf_townships_boundaries.merge(
                df_townships, on='township_name')
            gdf_townships = gdf_townships[[
                "dicofre", "township_name", "old_township_name", "concelho", "distrito", "geometry"]]

            gdf_townships.columns = ["dicofre_code", "township_name",
                                     "old_township_name", "municipality", "district", "wkt"]
            gdf_townships = gdf_townships.set_geometry("wkt")
            gdf_townships.to_postgis("civiltownship", engine_synthetic, if_exists="append", dtype={
                                     'dicofre_code': Integer, "wkt": Geometry("POLYGON", 4326)})


            # Populating TrafficAnalysisZone Table
            logging.info("Populating Database with %s into %s..." %
                         ("Zonamento_zone_Project.shp", "TrafficAnalysisZone"))
            gpd_taz = gpd.read_file(os.path.join(
                path_taz_folder+'Zonamento_zone_Project.shp'))
            gpd_taz.crs = 'EPSG:3763'
            gpd_taz = gpd_taz.to_crs('EPSG:4326')
            gpd_taz = gpd_taz.loc[gpd_taz['TYPENO'] == '1']
            gpd_taz = gpd_taz[["NO", "NAME", "AREA_ZONA", "TTR", "geometry"]]
            gpd_taz.columns = ["taz_id", "taz_name", "area", "ttr", "wkt"]
            gpd_taz = gpd_taz.set_geometry("wkt")
            gpd_taz.to_postgis("trafficanalysiszone", engine_synthetic, if_exists="append", dtype={
                               'taz_id': Integer, "geometry": Geometry("POLYGON", 4326)})
            
            # Populating SpatialLocation Table
            logging.info("Populating Database with %s into %s..." %
                         ("vodafone_grelha.csv", "SpatialLocation"))
            #not using the dicofre code from the file as it seems to be wrong.
            df_spatiallocation = pd.read_csv(os.path.join(path_grelha_folder, "vodafone_grelha.csv"), usecols=[
                                             "grelha_id", "grelha_x", "grelha_y", "latitude", "longitude", "nome", "wkt"], sep=";")
            df_spatiallocation["wkt"] = df_spatiallocation["wkt"].apply(
                wkt.loads)
            df_spatiallocation.columns = [
                "location_id", "x_square", "y_square", "latitude", "longitude", "loc_name", "wkt"]
            gdf_spatiallocation = gpd.GeoDataFrame(
                df_spatiallocation, crs="EPSG:4326", geometry="wkt")
            # Iterate through each row in gdf_spatiallocation
            for index, row in gdf_spatiallocation.iterrows():
                # Get the cell geometry
                cell = row['wkt']
                
                # Find all the townships that intersect the cell
                intersecting_townships = gdf_townships[gdf_townships.intersects(cell)]
                
                if not intersecting_townships.empty:
                    # Find the township with the highest intersection area
                    intersecting_townships['area'] = intersecting_townships.apply(lambda x: x['wkt'].buffer(0).intersection(cell).area, axis=1)
                    township = intersecting_townships.sort_values(by='area', ascending=False).iloc[0]
                    # Set the value of the "dicofre_code" column for the current row to the selected township
                    gdf_spatiallocation.loc[index, 'dicofre_code'] = int(township['dicofre_code'])
                else:
                    # Find the closest township
                    gdf_townships['distance'] = gdf_townships.apply(lambda x: x['wkt'].buffer(0).distance(cell), axis=1)
                    township = gdf_townships.sort_values(by='distance').iloc[0]
                    # Set the value of the "dicofre_code" column for the current row to the selected township
                    gdf_spatiallocation.loc[index, 'dicofre_code'] = int(township['dicofre_code'])
            gdf_spatiallocation['dicofre_code'] = gdf_spatiallocation['dicofre_code'].astype(int)
            # Iterate through each row in gdf_spatiallocation
            for index, row in gdf_spatiallocation.iterrows():
                # Get the cell geometry
                cell = row['wkt']
                
                # Find all the TAZs that intersect the cell
                intersecting_tazs = gpd_taz[gpd_taz.intersects(cell)]
                
                if not intersecting_tazs.empty:
                    # Find the TAZ with the highest intersection area
                    intersecting_tazs['area'] = intersecting_tazs.apply(lambda x: x['wkt'].buffer(0).intersection(cell).area, axis=1)
                    taz = intersecting_tazs.sort_values(by='area', ascending=False).iloc[0]
                    # Set the value of the "taz_id" column for the current row to the selected TAZ
                    gdf_spatiallocation.loc[index, 'taz_id'] = int(taz['taz_id'])
                else:
                    # Find the closest TAZ
                    gpd_taz['distance'] = gpd_taz.apply(lambda x: x['wkt'].buffer(0).distance(cell), axis=1)
                    taz = gpd_taz.sort_values(by='distance').iloc[0]
                    # Set the value of the "taz_id" column for the current row to the selected TAZ
                    gdf_spatiallocation.loc[index, 'taz_id'] = int(taz['taz_id'])
            gdf_spatiallocation['taz_id'] = gdf_spatiallocation['taz_id'].astype(int)
            gdf_spatiallocation.to_postgis("spatiallocation", engine_synthetic, if_exists="append", dtype={
                                           'location_id': Integer,'taz_id':Integer, 'dicofre_code': Integer, "wkt": Geometry("POLYGON", 4326)})   
            

            # Populating MobilityData Table
            logging.info("Populating Database with %s into %s..." % (
                path_data_synthetic+'synthetic_data.csv', "MobilityData"))

            #only columns with header n_terminals, roaming_terminals, n_phonecalls
            synthetic_df = pd.read_csv(path_data_synthetic+'synthetic_data.csv', header=0, usecols=["time", "location_id", "n_terminals", "n_roaming_terminals", "n_phonecalls"])
            logging.info(synthetic_df.head())
            #send to database in batches without index
            synthetic_df.to_sql("mobilitydata", engine_synthetic, if_exists="append", index=False, chunksize=1000)

            cursor.close()
            dbConn.close()
            print("PostgreSQL connection is closed")


if __name__ == "__main__":
    main()
