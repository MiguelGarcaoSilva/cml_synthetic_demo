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
CREATE MATERIALIZED VIEW mob_data_aggregated_hourly_cell_view (location_id , one_time, sum_terminals, sum_roaming_terminals,
sum_terminals_stayed, sum_roaming_terminals_stayed, sum_terminals_in, sum_terminals_out, sum_roaming_terminals_in, sum_roaming_terminals_out)
WITH (timescaledb.continuous) AS 
    SELECT location_id, time_bucket('1 hour', time) as one_time, SUM(n_terminals) as sum_terminals,
    SUM(n_roaming_terminals) as sum_roaming_terminals, SUM(n_terminals_stayed) as sum_terminals_stayed,
    SUM(n_roaming_terminals_stayed) as sum_roaming_terminals_stayed, SUM(n_terminals_in) as sum_terminals_in,
    SUM(n_terminals_out) as sum_terminals_out, SUM(n_roaming_terminals_in) as sum_roaming_terminals_in,
    SUM(n_roaming_terminals_out) as sum_roaming_terminals_out
    FROM MobilityData
    GROUP BY (location_id, one_time);


CREATE MATERIALIZED VIEW mob_data_aggregated_hourly_cell_withgeom_view (location_id, township_name, one_time, sum_terminals, sum_roaming_terminals,
sum_terminals_stayed, sum_roaming_terminals_stayed, sum_terminals_in, sum_terminals_out, sum_roaming_terminals_in, sum_roaming_terminals_out, wkt_cell) AS
    SELECT location_id, CivilTownShip.township_name, one_time, sum_terminals, sum_roaming_terminals, sum_terminals_stayed, 
    sum_roaming_terminals_stayed, sum_terminals_in, sum_terminals_out, sum_roaming_terminals_in, sum_roaming_terminals_out, SpatialLocation.wkt
    FROM mob_data_aggregated_hourly_cell_view NATURAL JOIN SpatialLocation JOIN CivilTownShip ON (SpatialLocation.dicofre_code=CivilTownShip.dicofre_code)
    ORDER BY one_time DESC;



CREATE MATERIALIZED VIEW mob_data_aggregated_hourly_taz_withgeom_view (taz_id, taz_name, one_time, sum_terminals, sum_roaming_terminals, wkt_taz) AS
SELECT taz_id, taz_name, one_time, sum_sum_terminals_per_taz_hour, sum_sum_roaming_terminals_per_taz_hour, wkt
FROM (SELECT one_time, taz_id, SUM(sum_terminals_per_cell_hour) as sum_sum_terminals_per_taz_hour,  SUM(sum_roaming_terminals_per_cell_hour) as sum_sum_roaming_terminals_per_taz_hour
      FROM ( SELECT location_id, taz_id, date_trunc('hour', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_hour, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_hour 
             FROM MobilityData NATURAL JOIN SpatialLocation
             GROUP BY (location_id, taz_id, one_time)) as total_per_cell_hour 
      GROUP BY (taz_id,one_time)) as TotalPertaz_andtime NATURAL JOIN TrafficAnalysisZone
ORDER BY one_time DESC;


CREATE MATERIALIZED VIEW mob_data_aggregated_hourly_township_withgeom_view (dicofre_code, township_name, one_time, sum_terminals, sum_roaming_terminals, wkt_township) AS
SELECT dicofre_code, township_name, one_time, sum_sum_terminals_per_township_hour, sum_sum_roaming_terminals_per_township_hour, wkt
FROM (SELECT one_time, dicofre_code, SUM(sum_terminals_per_cell_hour) as sum_sum_terminals_per_township_hour, SUM(sum_roaming_terminals_per_cell_hour) as sum_sum_roaming_terminals_per_township_hour
      FROM ( SELECT location_id, dicofre_code, date_trunc('hour', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_hour, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_hour 
             FROM MobilityData NATURAL JOIN SpatialLocation
             GROUP BY (location_id, dicofre_code, one_time)) as total_per_cell_hour
      GROUP BY (dicofre_code,one_time)) as TotalPerTownship_andtime NATURAL JOIN CivilTownShip
ORDER BY one_time DESC;

----DAILY

CREATE MATERIALIZED VIEW mob_data_aggregated_daily_cell_view (location_id , one_time, sum_terminals, sum_roaming_terminals,
sum_terminals_stayed, sum_roaming_terminals_stayed, sum_terminals_in, sum_terminals_out, sum_roaming_terminals_in, sum_roaming_terminals_out)
WITH (timescaledb.continuous) AS 
    SELECT location_id, time_bucket('1 day', time) as one_time, SUM(n_terminals) as sum_terminals,
    SUM(n_roaming_terminals) as sum_roaming_terminals, SUM(n_terminals_stayed) as sum_terminals_stayed,
    SUM(n_roaming_terminals_stayed) as sum_roaming_terminals_stayed, SUM(n_terminals_in) as sum_terminals_in,
    SUM(n_terminals_out) as sum_terminals_out, SUM(n_roaming_terminals_in) as sum_roaming_terminals_in,
    SUM(n_roaming_terminals_out) as sum_roaming_terminals_out
    FROM MobilityData
    GROUP BY (location_id, one_time);


CREATE MATERIALIZED VIEW mob_data_aggregated_daily_cell_withgeom_view (location_id, township_name, one_time, sum_terminals, sum_roaming_terminals,
sum_terminals_stayed, sum_roaming_terminals_stayed, sum_terminals_in, sum_terminals_out, sum_roaming_terminals_in, sum_roaming_terminals_out, wkt_cell) AS
    SELECT location_id, CivilTownShip.township_name, one_time, sum_terminals, sum_roaming_terminals, sum_terminals_stayed, 
    sum_roaming_terminals_stayed, sum_terminals_in, sum_terminals_out, sum_roaming_terminals_in, sum_roaming_terminals_out, SpatialLocation.wkt
    FROM mob_data_aggregated_daily_cell_view NATURAL JOIN SpatialLocation JOIN CivilTownShip ON (SpatialLocation.dicofre_code=CivilTownShip.dicofre_code)
    ORDER BY one_time DESC;


CREATE MATERIALIZED VIEW mob_data_aggregated_daily_taz_withgeom_view (taz_id, taz_name, one_time, sum_terminals, sum_roaming_terminals, wkt_taz) AS
SELECT taz_id, taz_name, one_time, sum_sum_terminals_per_taz_day, sum_sum_roaming_terminals_per_taz_day, wkt
FROM (SELECT one_time, taz_id, SUM(sum_terminals_per_cell_day) as sum_sum_terminals_per_taz_day,  SUM(sum_roaming_terminals_per_cell_day) as sum_sum_roaming_terminals_per_taz_day
      FROM ( SELECT location_id, taz_id, date_trunc('day', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_day, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_day 
             FROM MobilityData NATURAL JOIN SpatialLocation
             GROUP BY (location_id, taz_id, one_time)) as total_per_cell_day 
      GROUP BY (taz_id,one_time)) as TotalPertaz_andtime NATURAL JOIN TrafficAnalysisZone
ORDER BY one_time DESC;

CREATE MATERIALIZED VIEW mob_data_aggregated_daily_township_withgeom_view (dicofre_code, township_name, one_time, sum_terminals, sum_roaming_terminals, wkt_township) AS
SELECT dicofre_code, township_name, one_time, sum_sum_terminals_per_township_day, sum_sum_roaming_terminals_per_township_day, wkt
FROM (SELECT one_time, dicofre_code, SUM(sum_terminals_per_cell_day) as sum_sum_terminals_per_township_day, SUM(sum_roaming_terminals_per_cell_day) as sum_sum_roaming_terminals_per_township_day
      FROM ( SELECT location_id, dicofre_code, date_trunc('day', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_day, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_day 
             FROM MobilityData NATURAL JOIN SpatialLocation
             GROUP BY (location_id, dicofre_code, one_time)) as total_per_cell_day
      GROUP BY (dicofre_code,one_time)) as TotalPerTownship_andtime NATURAL JOIN CivilTownShip
ORDER BY one_time DESC;


----WEEKLY


CREATE MATERIALIZED VIEW mob_data_aggregated_weekly_cell_view (location_id , one_time, sum_terminals, sum_roaming_terminals,
sum_terminals_stayed, sum_roaming_terminals_stayed, sum_terminals_in, sum_terminals_out, sum_roaming_terminals_in, sum_roaming_terminals_out)
WITH (timescaledb.continuous) AS 
    SELECT location_id, time_bucket('1 week', time) as one_time, SUM(n_terminals) as sum_terminals,
    SUM(n_roaming_terminals) as sum_roaming_terminals, SUM(n_terminals_stayed) as sum_terminals_stayed,
    SUM(n_roaming_terminals_stayed) as sum_roaming_terminals_stayed, SUM(n_terminals_in) as sum_terminals_in,
    SUM(n_terminals_out) as sum_terminals_out, SUM(n_roaming_terminals_in) as sum_roaming_terminals_in,
    SUM(n_roaming_terminals_out) as sum_roaming_terminals_out
    FROM MobilityData
    GROUP BY (location_id, one_time);


CREATE MATERIALIZED VIEW mob_data_aggregated_weekly_cell_withgeom_view (location_id, township_name, one_time, sum_terminals, sum_roaming_terminals,
sum_terminals_stayed, sum_roaming_terminals_stayed, sum_terminals_in, sum_terminals_out, sum_roaming_terminals_in, sum_roaming_terminals_out, wkt_cell) AS
    SELECT location_id, CivilTownShip.township_name, one_time, sum_terminals, sum_roaming_terminals, sum_terminals_stayed, 
    sum_roaming_terminals_stayed, sum_terminals_in, sum_terminals_out, sum_roaming_terminals_in, sum_roaming_terminals_out, SpatialLocation.wkt
    FROM mob_data_aggregated_weekly_cell_view NATURAL JOIN SpatialLocation JOIN CivilTownShip ON (SpatialLocation.dicofre_code=CivilTownShip.dicofre_code)
    ORDER BY one_time DESC;


CREATE MATERIALIZED VIEW mob_data_aggregated_weekly_taz_withgeom_view (taz_id, taz_name, one_time, sum_terminals, sum_roaming_terminals, wkt_taz) AS
SELECT taz_id, taz_name, one_time, sum_sum_terminals_per_taz_week, sum_sum_roaming_terminals_per_taz_week, wkt
FROM (SELECT one_time, taz_id, SUM(sum_terminals_per_cell_week) as sum_sum_terminals_per_taz_week,  SUM(sum_roaming_terminals_per_cell_week) as sum_sum_roaming_terminals_per_taz_week
      FROM ( SELECT location_id, taz_id, date_trunc('week', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_week, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_week 
             FROM MobilityData NATURAL JOIN SpatialLocation
             GROUP BY (location_id, taz_id, one_time)) as total_per_cell_week 
      GROUP BY (taz_id,one_time)) as TotalPertaz_andtime NATURAL JOIN TrafficAnalysisZone
ORDER BY one_time DESC;

CREATE MATERIALIZED VIEW mob_data_aggregated_weekly_township_withgeom_view (dicofre_code, township_name, one_time, sum_terminals, sum_roaming_terminals, wkt_township) AS
SELECT dicofre_code, township_name, one_time, sum_sum_terminals_per_township_week, sum_sum_roaming_terminals_per_township_week, wkt
FROM (SELECT one_time, dicofre_code, SUM(sum_terminals_per_cell_week) as sum_sum_terminals_per_township_week, SUM(sum_roaming_terminals_per_cell_week) as sum_sum_roaming_terminals_per_township_week
      FROM ( SELECT location_id, dicofre_code, date_trunc('week', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_week, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_week 
             FROM MobilityData NATURAL JOIN SpatialLocation
             GROUP BY (location_id, dicofre_code, one_time)) as total_per_cell_week
      GROUP BY (dicofre_code,one_time)) as TotalPerTownship_andtime NATURAL JOIN CivilTownShip
ORDER BY one_time DESC;



----MONTHLY

CREATE MATERIALIZED VIEW mob_data_aggregated_monthly_cell_view (location_id , one_time, sum_terminals, sum_roaming_terminals,
sum_terminals_stayed, sum_roaming_terminals_stayed, sum_terminals_in, sum_terminals_out, sum_roaming_terminals_in, sum_roaming_terminals_out) AS 
    SELECT location_id, date_trunc('month', time) as one_time, SUM(n_terminals) as sum_terminals,
    SUM(n_roaming_terminals) as sum_roaming_terminals, SUM(n_terminals_stayed) as sum_terminals_stayed,
    SUM(n_roaming_terminals_stayed) as sum_roaming_terminals_stayed, SUM(n_terminals_in) as sum_terminals_in,
    SUM(n_terminals_out) as sum_terminals_out, SUM(n_roaming_terminals_in) as sum_roaming_terminals_in,
    SUM(n_roaming_terminals_out) as sum_roaming_terminals_out
    FROM MobilityData
    GROUP BY (location_id, one_time);


CREATE MATERIALIZED VIEW mob_data_aggregated_monthly_cell_withgeom_view (location_id, township_name, one_time, sum_terminals, sum_roaming_terminals,
sum_terminals_stayed, sum_roaming_terminals_stayed, sum_terminals_in, sum_terminals_out, sum_roaming_terminals_in, sum_roaming_terminals_out, wkt_cell) AS
    SELECT location_id, CivilTownShip.township_name, one_time, sum_terminals, sum_roaming_terminals, sum_terminals_stayed, 
    sum_roaming_terminals_stayed, sum_terminals_in, sum_terminals_out, sum_roaming_terminals_in, sum_roaming_terminals_out, SpatialLocation.wkt
    FROM mob_data_aggregated_monthly_cell_view NATURAL JOIN SpatialLocation JOIN CivilTownShip ON (SpatialLocation.dicofre_code=CivilTownShip.dicofre_code)
    ORDER BY one_time DESC;


CREATE MATERIALIZED VIEW mob_data_aggregated_monthly_taz_withgeom_view (taz_id, taz_name, one_time, sum_terminals, sum_roaming_terminals, wkt_taz) AS
SELECT taz_id, taz_name, one_time, sum_sum_terminals_per_taz_month, sum_sum_roaming_terminals_per_taz_month, wkt
FROM (SELECT one_time, taz_id, SUM(sum_terminals_per_cell_month) as sum_sum_terminals_per_taz_month,  SUM(sum_roaming_terminals_per_cell_month) as sum_sum_roaming_terminals_per_taz_month
      FROM ( SELECT location_id, taz_id, date_trunc('month', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_month, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_month 
             FROM MobilityData NATURAL JOIN SpatialLocation
             GROUP BY (location_id, taz_id, one_time)) as total_per_cell_month 
      GROUP BY (taz_id,one_time)) as TotalPertaz_andtime NATURAL JOIN TrafficAnalysisZone
ORDER BY one_time DESC;

CREATE MATERIALIZED VIEW mob_data_aggregated_monthly_township_withgeom_view (dicofre_code, township_name, one_time, sum_terminals, sum_roaming_terminals, wkt_township) AS
SELECT dicofre_code, township_name, one_time, sum_sum_terminals_per_township_month, sum_sum_roaming_terminals_per_township_month, wkt
FROM (SELECT one_time, dicofre_code, SUM(sum_terminals_per_cell_month) as sum_sum_terminals_per_township_month, SUM(sum_roaming_terminals_per_cell_month) as sum_sum_roaming_terminals_per_township_month
      FROM ( SELECT location_id, dicofre_code, date_trunc('month', time) as one_time, SUM(n_terminals) as sum_terminals_per_cell_month, SUM(n_roaming_terminals) as sum_roaming_terminals_per_cell_month 
             FROM MobilityData NATURAL JOIN SpatialLocation
             GROUP BY (location_id, dicofre_code, one_time)) as total_per_cell_month
      GROUP BY (dicofre_code,one_time)) as TotalPerTownship_andtime NATURAL JOIN CivilTownShip
ORDER BY one_time DESC;

