import os
import pandas as pd
from datetime import date, time, datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime, timedelta, timezone

"""
Add surface altitude from DEM.
"""

# open dataframe
dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
df = pd.read_csv("06_df.csv", index_col='datetime', dtype=dtype_options)
df.index = pd.to_datetime(df.index, format='ISO8601')

# open dataframe, this dataframe includes SurfaceLayerAltitude information from the DEM at the drone locations, available from zenodo
df_DEM = pd.read_csv("./06_df_withZ_v2.csv", usecols=['datetime', 'SurfaceLayerAltitude', 'RTK.aircraftLongitude', 'RTK.aircraftLatitude'], sep=',')
df_DEM['datetime'] = pd.to_datetime(df_DEM['datetime'], format='ISO8601')
df_DEM.set_index('datetime', inplace=True)

# the df includes some duplicated items, keep the first one
# print(df.index.duplicated().sum())
# print(df_DEM.index.duplicated().sum())
df = df[~df.index.duplicated(keep='first')]
df_DEM = df_DEM[~df_DEM.index.duplicated(keep='first')]

df_DEM.rename(columns={'SurfaceLayerAltitude': 'PROC.LiDAR.surfaceAltitude'}, inplace=True)

# merge and save the dataframes
merged_df = pd.merge(df, df_DEM['PROC.LiDAR.surfaceAltitude'], left_index=True, right_index=True, how='inner')
merged_df = merged_df[~merged_df.index.duplicated()]
merged_df.to_csv("07_df.csv", index=True, decimal='.', sep=',')