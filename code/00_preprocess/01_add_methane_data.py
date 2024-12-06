import os
import pandas as pd
from datetime import date, time, datetime

"""
Add CH4 concentration data from AERIS sensor
"""

# empty dataframe
df = pd.DataFrame()

columns_to_read = ['Time Stamp',
                   'GPS Time',
                   'Latitude',
                   'Longitude',
                   'Alt. (m)',
                   'CH4 (ppm)', 
                   'H2O (ppm)', 
                   'C2H6 (ppb)']

directory = "./Strato"  # change path to storage location of methane sensor data, available from zenodo
suffix = "Eng.txt"

for filename in os.listdir(directory):
    if filename.endswith(suffix):
        fileloc = f"{directory}/{filename}"

        # read data and create datetime index
        df_temp = pd.read_csv(fileloc, usecols=columns_to_read)
        df_temp['datetime'] = pd.to_datetime(df_temp['GPS Time'], unit="s").dt.tz_localize('UTC')
        df_temp = df_temp.set_index('datetime')
        df_temp = df_temp.drop(columns = ['Time Stamp', 'GPS Time'])

        # append to complete dataframe
        df = pd.concat([df, df_temp], ignore_index=False)

# rename columns and sort index
df.rename(columns={ 'CH4 (ppm)': 'AERIS.CH4 [ppm]',
                    'H2O (ppm)': 'AERIS.H2O [ppm]',
                    'C2H6 (ppb)': 'AERIS.C2H6 [ppb]',
                    'Latitude': 'AERIS.latitude',
                    'Longitude': 'AERIS.longitude',
                    'Alt. (m)': 'AERIS.Alt. [m]'}, inplace=True)
df = df.sort_index()

# add to leading dataframe
dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str'}
df_drone = pd.read_csv("00_df.csv", index_col='datetime', dtype=dtype_options)
df_drone.index = pd.to_datetime(df_drone.index, format='ISO8601')

df.index = pd.to_datetime(df.index, format='ISO8601')
df = pd.merge(df, df_drone, on='datetime', how='right') # keep the index of the dji data

# save dataframe
df.to_csv("01_df.csv", index=True, decimal='.', sep=',')