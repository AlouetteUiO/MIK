import os
import pandas as pd
from datetime import date, time, datetime
import numpy as np
import matplotlib.pyplot as plt

"""
Select the mass balance flights, and save each flight in a separate file.

Mass balance method workflow:
1) Filter one box flight (script 00_select_flights.py)
2) Determine to which side of the box the point belongs, NE, SE, SW, NW, top (script 01_select_wall.py)
3) Determine the perpendicular wind speed (script 02_perpendicular_windspeed.py)
4) Compute all the point fluxes (script 03_interpolate.py)
5) Make grid (script 03_interpolate.py)
6) Integrating across each wall (script 03_interpolate.py)
"""

# open combined dataframe
dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
df = pd.read_csv("data/11_df.csv", index_col='datetime', dtype=dtype_options)
df.index = pd.to_datetime(df.index, format='ISO8601')

# open dataframe with notes
fileloc_notes = "./flight_notes.xlsx"  # change to location of flight notes, available from zenodo
df_notes = pd.read_excel(fileloc_notes, dtype={'date':str, 'starttime':str, 'endtime':str})

df_notes['datetime_start'] = pd.to_datetime( df_notes['date'].str.split().str[0] + ' ' + df_notes['starttime'], format='%Y-%m-%d %H:%M:%S')
df_notes['datetime_end'] = pd.to_datetime( df_notes['date'].str.split().str[0] + ' ' + df_notes['endtime'], format='%Y-%m-%d %H:%M:%S')

df_notes['datetime_start'] = df_notes['datetime_start'].dt.tz_localize('Africa/Nairobi')
df_notes['datetime_end'] = df_notes['datetime_end'].dt.tz_localize('Africa/Nairobi')
df_notes['datetime_start'] = df_notes['datetime_start'].dt.tz_convert('UTC')
df_notes['datetime_end'] = df_notes['datetime_end'].dt.tz_convert('UTC')
df_notes = df_notes.drop(columns = ['date', 'starttime', 'endtime'])

for index, row in df_notes.iterrows():
    if row.typeOfFlight == 'box':
        start_time = row.datetime_start
        end_time = row.datetime_end
        date_string = start_time.strftime("%Y%m%d_%H%M%S")
        animal_string = row.animals.replace(" ", "_")
        df_box = df[ (df.index>start_time) & (df.index<end_time) ]
        df_box.to_csv(f"data/mass_balance/{date_string}_{animal_string}_00.csv", index=True, decimal='.', sep=',')