CREATE EXTENSION IF NOT EXISTS timescaledb VERSION '2.5.1' CASCADE;
CREATE EXTENSION IF NOT EXISTS postgis VERSION '3.2.1' CASCADE;


DROP TABLE IF EXISTS MobilityData CASCADE;
DROP TABLE IF EXISTS SpatialLocation CASCADE;
DROP TABLE IF EXISTS CivilTownShip CASCADE;
DROP TABLE IF EXISTS Country CASCADE;
DROP TABLE IF EXISTS TrafficAnalysisZone CASCADE;
DROP TABLE IF EXISTS IntersectsTaz CASCADE;


CREATE TABLE Country (
   numeric_iso_code INTEGER NOT NULL,
	alpha3_iso_code CHAR(3) NOT NULL UNIQUE,
	country_name VARCHAR NOT NULL UNIQUE,
	PRIMARY KEY(numeric_iso_code)
);


CREATE TABLE TrafficAnalysisZone (
   taz_id INTEGER NOT NULL,
   taz_name VARCHAR NOT NULL UNIQUE,
   area DECIMAL NOT NULL,
   ttr DECIMAL NOT NULL,
   wkt GEOMETRY(POLYGON, 4326),
   PRIMARY KEY(taz_id)
);



CREATE TABLE CivilTownShip (
   dicofre_code INTEGER NOT NULL,
   township_name VARCHAR NOT NULL,
   old_township_name VARCHAR,
   municipality VARCHAR NOT NULL,
   district VARCHAR NOT NULL,
   wkt GEOMETRY(POLYGON, 4326),
   PRIMARY KEY(dicofre_code)
);
-- Add a spatial index
CREATE INDEX CivilTownShip_wkt_idx ON CivilTownShip USING GIST (wkt);

CREATE TABLE SpatialLocation (
   location_id  SMALLINT NOT NULL UNIQUE,
   dicofre_code INTEGER NOT NULL,
   loc_name VARCHAR NOT NULL,
   x_square SMALLINT NOT NULL,
   y_square SMALLINT NOT NULL,
   latitude DECIMAL(8,6) NOT NULL,
   longitude DECIMAL(9,6) NOT NULL,
   wkt GEOMETRY(MULTIPOLYGON, 4326),
   PRIMARY KEY(location_id),
   CONSTRAINT fk_civiltownship FOREIGN KEY (dicofre_code) REFERENCES CivilTownShip(dicofre_code) ON DELETE CASCADE
);

CREATE INDEX SpatialLocation_latlong_idx ON SpatialLocation USING BTREE(latitude,longitude);
-- Add a spatial index
CREATE INDEX SpatialLocation_wtk_idx ON SpatialLocation USING GIST (wkt);


CREATE TABLE MobilityData (
   time TIMESTAMP NOT NULL,
   location_id INTEGER NOT NULL,
   n_terminals INTEGER NOT NULL,
   n_roaming_terminals INTEGER NOT NULL,
   n_terminals_stayed INTEGER NOT NULL,
   n_roaming_terminals_stayed INTEGER NOT NULL,
   n_terminals_in INTEGER NOT NULL,
   n_terminals_out INTEGER NOT NULL,
   n_roaming_terminals_in INTEGER NOT NULL,
   n_roaming_terminals_out INTEGER NOT NULL,
   PRIMARY KEY (time, location_id),
   CONSTRAINT fk_location FOREIGN KEY (location_id) REFERENCES SpatialLocation(location_id) ON DELETE CASCADE
);

SELECT create_hypertable ('MobilityData', 'time', chunk_time_interval => INTERVAL '1 month');



CREATE TABLE IntersectsTaz (
   location_id INTEGER NOT NULL,
   taz_id INTEGER NOT NULL,
   PRIMARY KEY(location_id, taz_id),
   CONSTRAINT fk_spatiallocation FOREIGN KEY (location_id) REFERENCES SpatialLocation(location_id) ON DELETE CASCADE
);