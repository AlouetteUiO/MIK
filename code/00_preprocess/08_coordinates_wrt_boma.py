import os
import pandas as pd
import geopandas as gpd 
from datetime import date, time, datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime, timedelta, timezone
from shapely.geometry import Point
from scipy.interpolate import griddata

"""
Compute coordinates of the drone with respect to the center of the boma.
"""

# open dataframe
dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
df = pd.read_csv("07_df.csv", index_col='datetime', dtype=dtype_options)
df.index = pd.to_datetime(df.index, format='ISO8601')

# make geopandas dataframe
gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df['RTK.aircraftLongitude_x'], df['RTK.aircraftLatitude_x'], crs=4326))
gdf = gdf.to_crs(32737)

# coordinates of the corners of the mass balance flight: S, E, N, W, the center of the boma C, 
# and the base from which we started the drone. 
# note that this is the new home point (denoted as such in the flight_notes file).
# we do not use the experiment with the old home point in our study.
d = {   'name' : ['S', 'E', 'N', 'W', 'C', 'BASE'],
        'geometry': [   Point(37.132793, -1.613556),
                        Point(37.132965, -1.613383),
                        Point(37.132793, -1.613210),
                        Point(37.132622, -1.613383),
                        Point(37.132793, -1.613383),
                        Point(37.132341, -1.613654)],
    }
gdf_boma = gpd.GeoDataFrame(d, crs=4326)
gdf_boma.set_index('name', inplace=True)
gdf_boma = gdf_boma.to_crs(32737)

# interpolate to get surface altitude at boma coordinates
x = gdf.geometry.x
y = gdf.geometry.y
z = gdf['PROC.LiDAR.surfaceAltitude']
xi = gdf_boma.geometry.x
yi = gdf_boma.geometry.y
zi = griddata((x, y), z, (xi, yi), method='linear')
gdf_boma['PROC.LiDAR.surfaceAltitude'] = zi
gdf_boma['PROC.z_surface'] = gdf_boma['PROC.LiDAR.surfaceAltitude'] - gdf_boma.loc['C','PROC.LiDAR.surfaceAltitude']

# save coordinates in local coordinate system
df['geometry'] = gdf['geometry']

# use center of boma as reference point and compute x and y
x_boma, y_boma = gdf_boma.loc['C'].geometry.x, gdf_boma.loc['C'].geometry.y
df['PROC.x'] = gdf.geometry.x - x_boma
df['PROC.y'] = gdf.geometry.y - y_boma

# use center of boma as reference point for z_surface and z_drone
df['PROC.z_surface'] = df['PROC.LiDAR.surfaceAltitude'] - gdf_boma.loc['C','PROC.LiDAR.surfaceAltitude']
df['PROC.aircraftAltitude'] = df['RTK.aircraftHeight [m]'] + df['RTK.baseStationAltitude [m]'] 
df['PROC.h_drone'] = df['PROC.aircraftAltitude'] - df['PROC.LiDAR.surfaceAltitude']
df['PROC.z_drone'] = df['PROC.aircraftAltitude'] - gdf_boma.loc['C','PROC.LiDAR.surfaceAltitude']

# we also compute h_drone and z_drone based on vpsHeight
df['PROC.h_drone_vps'] = df['OSD.vpsHeight [m]'] 
df['PROC.z_drone_vps'] = df['OSD.vpsHeight [m]'] + df['PROC.z_surface']

# during data processing, we observe that h_drone can deviate up to 2 meters from vpsHeight for the different mass balance flights. 

# we compute h_drone and z_drone without using RTK.baseStationAltitude
# but using RTK aircraftHeight and LiDAR.surfaceAltitude at the approximate BASE location
# we find that this is the most reliable option and we use z_drone_base in further analysis
df['PROC.h_drone_base'] = df['RTK.aircraftHeight [m]'] + gdf_boma.loc['BASE','PROC.LiDAR.surfaceAltitude'] - df['PROC.LiDAR.surfaceAltitude']
df['PROC.z_drone_base'] = df['RTK.aircraftHeight [m]'] + gdf_boma.loc['BASE','PROC.LiDAR.surfaceAltitude'] - gdf_boma.loc['C','PROC.LiDAR.surfaceAltitude']

# save df and gdf_boma
df.to_csv("08_df.csv", index=True, decimal='.', sep=',')
gdf_boma.to_csv("gdf_boma.csv", index=True, decimal='.', sep=',')