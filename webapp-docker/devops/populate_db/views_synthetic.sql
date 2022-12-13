DROP MATERIALIZED VIEW mob_data_aggregated_hourly_cell_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_hourly_cell_withgeom_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_hourly_taz_withgeom_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_hourly_township_withgeom_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_daily_cell_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_daily_cell_withgeom_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_daily_taz_withgeom_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_daily_township_withgeom_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_weekly_cell_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_weekly_cell_withgeom_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_weekly_taz_withgeom_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_weekly_township_withgeom_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_monthly_cell_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_monthly_cell_withgeom_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_monthly_taz_withgeom_view CASCADE;
DROP MATERIALIZED VIEW mob_data_aggregated_monthly_township_withgeom_view CASCADE;



----HOURLY
CREATE MATERIALIZED VIEW mob_data_aggregated_hourly_cell_view (location_id , one_time, avg_terminals, avg_roaming_terminals,
avg_terminals_stayed, avg_roaming_terminals_stayed, avg_terminals_in, avg_terminals_out, avg_roaming_terminals_in, avg_roaming_terminals_out)
WITH (timescaledb.continuous) AS 
    SELECT location_id, time_bucket('1 hour', time) as one_time, AVG(n_terminals) as avg_terminals,
    AVG(n_roaming_terminals) as avg_roaming_terminals, AVG(n_terminals_stayed) as avg_terminals_stayed,
    AVG(n_roaming_terminals_stayed) as avg_roaming_terminals_stayed, AVG(n_terminals_in) as avg_terminals_in,
    AVG(n_terminals_out) as avg_terminals_out, AVG(n_roaming_terminals_in) as avg_roaming_terminals_in,
    AVG(n_roaming_terminals_out) as avg_roaming_terminals_out
    FROM MobilityData
    GROUP BY (location_id, one_time);


CREATE MATERIALIZED VIEW mob_data_aggregated_hourly_cell_withgeom_view (location_id, township_name, one_time, avg_terminals, avg_roaming_terminals,
avg_terminals_stayed, avg_roaming_terminals_stayed, avg_terminals_in, avg_terminals_out, avg_roaming_terminals_in, avg_roaming_terminals_out, wkt_cell) AS
    SELECT location_id, CivilTownShip.township_name, one_time, avg_terminals, avg_roaming_terminals, avg_terminals_stayed, 
    avg_roaming_terminals_stayed, avg_terminals_in, avg_terminals_out, avg_roaming_terminals_in, avg_roaming_terminals_out, SpatialLocation.wkt
    FROM mob_data_aggregated_hourly_cell_view NATURAL JOIN SpatialLocation JOIN CivilTownShip ON (SpatialLocation.dicofre_code=CivilTownShip.dicofre_code)
    ORDER BY one_time DESC;



CREATE MATERIALIZED VIEW mob_data_aggregated_hourly_taz_withgeom_view (taz_id, taz_name, one_time, avg_terminals, avg_roaming_terminals, wkt_taz) AS
SELECT taz_id, taz_name, one_time, avg_sum_terminals_per_taz_hour, avg_sum_roaming_terminals_per_taz_hour, wkt
FROM (SELECT one_time, taz_id, AVG(sum_terminals_per_cell_hour) as avg_sum_terminals_per_taz_hour,  AVG(sum_roaming_terminals_per_cell_hour) as avg_sum_roaming_terminals_per_taz_hour
      FROM ( SELECT location_id, taz_id, date_trunc('hour', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_hour, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_hour 
             FROM MobilityData NATURAL JOIN SpatialLocation NATURAL JOIN IntersectsTaz 
             GROUP BY (location_id, taz_id, one_time)) as total_per_cell_hour 
      GROUP BY (taz_id,one_time)) as TotalPertaz_andtime NATURAL JOIN TrafficAnalysisZone
ORDER BY one_time DESC;


CREATE MATERIALIZED VIEW mob_data_aggregated_hourly_township_withgeom_view (dicofre_code, township_name, one_time, avg_terminals, avg_roaming_terminals, wkt_township) AS
SELECT dicofre_code, township_name, one_time, avg_sum_terminals_per_township_hour, avg_sum_roaming_terminals_per_township_hour, wkt
FROM (SELECT one_time, dicofre_code, AVG(sum_terminals_per_cell_hour) as avg_sum_terminals_per_township_hour, AVG(sum_roaming_terminals_per_cell_hour) as avg_sum_roaming_terminals_per_township_hour
      FROM ( SELECT location_id, dicofre_code, date_trunc('hour', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_hour, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_hour 
             FROM MobilityData NATURAL JOIN SpatialLocation
             GROUP BY (location_id, dicofre_code, one_time)) as total_per_cell_hour
      GROUP BY (dicofre_code,one_time)) as TotalPerTownship_andtime NATURAL JOIN CivilTownShip
ORDER BY one_time DESC;

----DAILY

CREATE MATERIALIZED VIEW mob_data_aggregated_daily_cell_view (location_id , one_time, avg_terminals, avg_roaming_terminals,
avg_terminals_stayed, avg_roaming_terminals_stayed, avg_terminals_in, avg_terminals_out, avg_roaming_terminals_in, avg_roaming_terminals_out)
WITH (timescaledb.continuous) AS 
    SELECT location_id, time_bucket('1 day', time) as one_time, AVG(n_terminals) as avg_terminals,
    AVG(n_roaming_terminals) as avg_roaming_terminals, AVG(n_terminals_stayed) as avg_terminals_stayed,
    AVG(n_roaming_terminals_stayed) as avg_roaming_terminals_stayed, AVG(n_terminals_in) as avg_terminals_in,
    AVG(n_terminals_out) as avg_terminals_out, AVG(n_roaming_terminals_in) as avg_roaming_terminals_in,
    AVG(n_roaming_terminals_out) as avg_roaming_terminals_out
    FROM MobilityData
    GROUP BY (location_id, one_time);


CREATE MATERIALIZED VIEW mob_data_aggregated_daily_cell_withgeom_view (location_id, township_name, one_time, avg_terminals, avg_roaming_terminals,
avg_terminals_stayed, avg_roaming_terminals_stayed, avg_terminals_in, avg_terminals_out, avg_roaming_terminals_in, avg_roaming_terminals_out, wkt_cell) AS
    SELECT location_id, CivilTownShip.township_name, one_time, avg_terminals, avg_roaming_terminals, avg_terminals_stayed, 
    avg_roaming_terminals_stayed, avg_terminals_in, avg_terminals_out, avg_roaming_terminals_in, avg_roaming_terminals_out, SpatialLocation.wkt
    FROM mob_data_aggregated_daily_cell_view NATURAL JOIN SpatialLocation JOIN CivilTownShip ON (SpatialLocation.dicofre_code=CivilTownShip.dicofre_code)
    ORDER BY one_time DESC;


CREATE MATERIALIZED VIEW mob_data_aggregated_daily_taz_withgeom_view (taz_id, taz_name, one_time, avg_terminals, avg_roaming_terminals, wkt_taz) AS
SELECT taz_id, taz_name, one_time, avg_sum_terminals_per_taz_day, avg_sum_roaming_terminals_per_taz_day, wkt
FROM (SELECT one_time, taz_id, AVG(sum_terminals_per_cell_day) as avg_sum_terminals_per_taz_day,  AVG(sum_roaming_terminals_per_cell_day) as avg_sum_roaming_terminals_per_taz_day
      FROM ( SELECT location_id, taz_id, date_trunc('day', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_day, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_day 
             FROM MobilityData NATURAL JOIN SpatialLocation NATURAL JOIN IntersectsTaz 
             GROUP BY (location_id, taz_id, one_time)) as total_per_cell_day 
      GROUP BY (taz_id,one_time)) as TotalPertaz_andtime NATURAL JOIN TrafficAnalysisZone
ORDER BY one_time DESC;

CREATE MATERIALIZED VIEW mob_data_aggregated_daily_township_withgeom_view (dicofre_code, township_name, one_time, avg_terminals, avg_roaming_terminals, wkt_township) AS
SELECT dicofre_code, township_name, one_time, avg_sum_terminals_per_township_day, avg_sum_roaming_terminals_per_township_day, wkt
FROM (SELECT one_time, dicofre_code, AVG(sum_terminals_per_cell_day) as avg_sum_terminals_per_township_day, AVG(sum_roaming_terminals_per_cell_day) as avg_sum_roaming_terminals_per_township_day
      FROM ( SELECT location_id, dicofre_code, date_trunc('day', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_day, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_day 
             FROM MobilityData NATURAL JOIN SpatialLocation
             GROUP BY (location_id, dicofre_code, one_time)) as total_per_cell_day
      GROUP BY (dicofre_code,one_time)) as TotalPerTownship_andtime NATURAL JOIN CivilTownShip
ORDER BY one_time DESC;


----WEEKLY


CREATE MATERIALIZED VIEW mob_data_aggregated_weekly_cell_view (location_id , one_time, avg_terminals, avg_roaming_terminals,
avg_terminals_stayed, avg_roaming_terminals_stayed, avg_terminals_in, avg_terminals_out, avg_roaming_terminals_in, avg_roaming_terminals_out)
WITH (timescaledb.continuous) AS 
    SELECT location_id, time_bucket('1 week', time) as one_time, AVG(n_terminals) as avg_terminals,
    AVG(n_roaming_terminals) as avg_roaming_terminals, AVG(n_terminals_stayed) as avg_terminals_stayed,
    AVG(n_roaming_terminals_stayed) as avg_roaming_terminals_stayed, AVG(n_terminals_in) as avg_terminals_in,
    AVG(n_terminals_out) as avg_terminals_out, AVG(n_roaming_terminals_in) as avg_roaming_terminals_in,
    AVG(n_roaming_terminals_out) as avg_roaming_terminals_out
    FROM MobilityData
    GROUP BY (location_id, one_time);


CREATE MATERIALIZED VIEW mob_data_aggregated_weekly_cell_withgeom_view (location_id, township_name, one_time, avg_terminals, avg_roaming_terminals,
avg_terminals_stayed, avg_roaming_terminals_stayed, avg_terminals_in, avg_terminals_out, avg_roaming_terminals_in, avg_roaming_terminals_out, wkt_cell) AS
    SELECT location_id, CivilTownShip.township_name, one_time, avg_terminals, avg_roaming_terminals, avg_terminals_stayed, 
    avg_roaming_terminals_stayed, avg_terminals_in, avg_terminals_out, avg_roaming_terminals_in, avg_roaming_terminals_out, SpatialLocation.wkt
    FROM mob_data_aggregated_weekly_cell_view NATURAL JOIN SpatialLocation JOIN CivilTownShip ON (SpatialLocation.dicofre_code=CivilTownShip.dicofre_code)
    ORDER BY one_time DESC;


CREATE MATERIALIZED VIEW mob_data_aggregated_weekly_taz_withgeom_view (taz_id, taz_name, one_time, avg_terminals, avg_roaming_terminals, wkt_taz) AS
SELECT taz_id, taz_name, one_time, avg_sum_terminals_per_taz_week, avg_sum_roaming_terminals_per_taz_week, wkt
FROM (SELECT one_time, taz_id, AVG(sum_terminals_per_cell_week) as avg_sum_terminals_per_taz_week,  AVG(sum_roaming_terminals_per_cell_week) as avg_sum_roaming_terminals_per_taz_week
      FROM ( SELECT location_id, taz_id, date_trunc('week', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_week, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_week 
             FROM MobilityData NATURAL JOIN SpatialLocation NATURAL JOIN IntersectsTaz 
             GROUP BY (location_id, taz_id, one_time)) as total_per_cell_week 
      GROUP BY (taz_id,one_time)) as TotalPertaz_andtime NATURAL JOIN TrafficAnalysisZone
ORDER BY one_time DESC;

CREATE MATERIALIZED VIEW mob_data_aggregated_weekly_township_withgeom_view (dicofre_code, township_name, one_time, avg_terminals, avg_roaming_terminals, wkt_township) AS
SELECT dicofre_code, township_name, one_time, avg_sum_terminals_per_township_week, avg_sum_roaming_terminals_per_township_week, wkt
FROM (SELECT one_time, dicofre_code, AVG(sum_terminals_per_cell_week) as avg_sum_terminals_per_township_week, AVG(sum_roaming_terminals_per_cell_week) as avg_sum_roaming_terminals_per_township_week
      FROM ( SELECT location_id, dicofre_code, date_trunc('week', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_week, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_week 
             FROM MobilityData NATURAL JOIN SpatialLocation
             GROUP BY (location_id, dicofre_code, one_time)) as total_per_cell_week
      GROUP BY (dicofre_code,one_time)) as TotalPerTownship_andtime NATURAL JOIN CivilTownShip
ORDER BY one_time DESC;



----MONTHLY

CREATE MATERIALIZED VIEW mob_data_aggregated_monthly_cell_view (location_id , one_time, avg_terminals, avg_roaming_terminals,
avg_terminals_stayed, avg_roaming_terminals_stayed, avg_terminals_in, avg_terminals_out, avg_roaming_terminals_in, avg_roaming_terminals_out) AS 
    SELECT location_id, date_trunc('month', time) as one_time, AVG(n_terminals) as avg_terminals,
    AVG(n_roaming_terminals) as avg_roaming_terminals, AVG(n_terminals_stayed) as avg_terminals_stayed,
    AVG(n_roaming_terminals_stayed) as avg_roaming_terminals_stayed, AVG(n_terminals_in) as avg_terminals_in,
    AVG(n_terminals_out) as avg_terminals_out, AVG(n_roaming_terminals_in) as avg_roaming_terminals_in,
    AVG(n_roaming_terminals_out) as avg_roaming_terminals_out
    FROM MobilityData
    GROUP BY (location_id, one_time);


CREATE MATERIALIZED VIEW mob_data_aggregated_monthly_cell_withgeom_view (location_id, township_name, one_time, avg_terminals, avg_roaming_terminals,
avg_terminals_stayed, avg_roaming_terminals_stayed, avg_terminals_in, avg_terminals_out, avg_roaming_terminals_in, avg_roaming_terminals_out, wkt_cell) AS
    SELECT location_id, CivilTownShip.township_name, one_time, avg_terminals, avg_roaming_terminals, avg_terminals_stayed, 
    avg_roaming_terminals_stayed, avg_terminals_in, avg_terminals_out, avg_roaming_terminals_in, avg_roaming_terminals_out, SpatialLocation.wkt
    FROM mob_data_aggregated_monthly_cell_view NATURAL JOIN SpatialLocation JOIN CivilTownShip ON (SpatialLocation.dicofre_code=CivilTownShip.dicofre_code)
    ORDER BY one_time DESC;


CREATE MATERIALIZED VIEW mob_data_aggregated_monthly_taz_withgeom_view (taz_id, taz_name, one_time, avg_terminals, avg_roaming_terminals, wkt_taz) AS
SELECT taz_id, taz_name, one_time, avg_sum_terminals_per_taz_month, avg_sum_roaming_terminals_per_taz_month, wkt
FROM (SELECT one_time, taz_id, AVG(sum_terminals_per_cell_month) as avg_sum_terminals_per_taz_month,  AVG(sum_roaming_terminals_per_cell_month) as avg_sum_roaming_terminals_per_taz_month
      FROM ( SELECT location_id, taz_id, date_trunc('month', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_month, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_month 
             FROM MobilityData NATURAL JOIN SpatialLocation NATURAL JOIN IntersectsTaz 
             GROUP BY (location_id, taz_id, one_time)) as total_per_cell_month 
      GROUP BY (taz_id,one_time)) as TotalPertaz_andtime NATURAL JOIN TrafficAnalysisZone
ORDER BY one_time DESC;

CREATE MATERIALIZED VIEW mob_data_aggregated_monthly_township_withgeom_view (dicofre_code, township_name, one_time, avg_terminals, avg_roaming_terminals, wkt_township) AS
SELECT dicofre_code, township_name, one_time, avg_sum_terminals_per_township_month, avg_sum_roaming_terminals_per_township_month, wkt
FROM (SELECT one_time, dicofre_code, AVG(sum_terminals_per_cell_month) as avg_sum_terminals_per_township_month, AVG(sum_roaming_terminals_per_cell_month) as avg_sum_roaming_terminals_per_township_month
      FROM ( SELECT location_id, dicofre_code, date_trunc('month', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_month, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_month 
             FROM MobilityData NATURAL JOIN SpatialLocation
             GROUP BY (location_id, dicofre_code, one_time)) as total_per_cell_month
      GROUP BY (dicofre_code,one_time)) as TotalPerTownship_andtime NATURAL JOIN CivilTownShip
ORDER BY one_time DESC;

