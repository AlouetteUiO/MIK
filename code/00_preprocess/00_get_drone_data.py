import os
import pandas as pd
from datetime import date, time, datetime
import numpy as np
import matplotlib.pyplot as plt

"""
Get drone data processed by FlightReader: flightreader.com
"""

# create empty dataframe
df = pd.DataFrame()

# collect drone data
directory = "./FlighReader" # replace with path to FlightReader folder, data available from zenodo
suffix = "minimal.csv"

for folder in os.listdir(directory):
    for filename in os.listdir(f"{directory}/{folder}/"):
        if filename.endswith(suffix):
            fileloc = f"{directory}/{folder}/{filename}"

            print(f"processing {filename}")

            # read data and create datetime index
            df_temp = pd.read_csv(fileloc, sep=';')
            df_temp['datetime'] = (pd.to_datetime(df_temp['CUSTOM.updateTime [epoch]'], unit="ms")).dt.tz_localize('UTC')
            df_temp = df_temp.set_index('datetime')
            df_temp = df_temp.drop(columns = ['CUSTOM.updateTime [epoch]', 'CUSTOM.updateDateTime24 [UTC]', 'CUSTOM.updateDateTime24', 
                'OSD.directionOfTravel', 'OSD.gpsNum', 'OSD.gpsLevel', 'OSD.isGPSUsed'])

            # filter out the rows with flight data
            df_temp = df_temp[df_temp['OSD.flycState'].isin(['Tripod', 'P-GPS (Brake)', 'P-GPS', 'Waypoints'])]

            # append to complete dataframe
            df = pd.concat([df, df_temp], ignore_index=False)

# change columns to numeric data
numeric_columns = ['OSD.flyTime [s]', 'OSD.latitude', 'OSD.longitude', 'OSD.height [m]', 
            'OSD.vpsHeight [m]', 'OSD.altitude [m]', 'OSD.pitch', 'OSD.roll', 
            'OSD.yaw', 'OSD.yaw [360]', 'RTK.aircraftLatitude', 'RTK.aircraftLongitude',	
            'RTK.aircraftHeight [m]', 'RTK.baseStationAltitude [m]', 'HOME.height [m]', 'WEATHER.windSpeed [m/s]']
#for col in numeric_columns[:-1]: 
for col in numeric_columns: 
    df[col] = pd.to_numeric(df[col].str.replace(',', '.'))

# sort index of df
df = df.sort_index()

# check for duplicated timestamps and remove these
duplicate_timestamps = df.index.duplicated()
if duplicate_timestamps.any():
    print(f"Duplicate timestamps detected: {df.index[duplicate_timestamps]}. Removing duplicates...")
    df = df[~duplicate_timestamps]  

# resample numeric data by taking the mean of the bin
df_resample_numeric = df.loc[:, numeric_columns].resample('S').mean()

# resample the string data by taking the most often mentioned string in the bin
def mode_or_default(x):
    mode_df = x.mode()
    if mode_df.empty:
        return None 
    else:
        return mode_df.iloc[0]

df_resample_string = pd.DataFrame()
df_resample_string['OSD.flycState'] = df['OSD.flycState'].resample('1S').agg(mode_or_default)
df_resample_string['OSD.flightAction'] = df['OSD.flightAction'].resample('1S').agg(mode_or_default)
df_resample_string['WEATHER.windDirection'] = df['WEATHER.windDirection'].resample('1S').agg(mode_or_default)

# merge the numeric df and string df
df_resampled = pd.merge(df_resample_numeric, df_resample_string, on='datetime', how='outer')

# remove all empty rows
df_resampled_cleaned = df_resampled.dropna(how='all')

# save dataframe
df_resampled_cleaned.to_csv("data/00_df.csv", index=True, decimal='.', sep=',')  # change path location for storing leading dataframe