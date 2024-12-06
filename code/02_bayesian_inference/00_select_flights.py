import os
import pandas as pd
from datetime import date, time, datetime
import numpy as np
import matplotlib.pyplot as plt

"""
Select the bayesian inference flights, and save each flight in a separate file.
"""

directory = "data/bayesian/inference"

# open combined dataframe
dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
df = pd.read_csv("data/11_df.csv", index_col='datetime', dtype=dtype_options)
df.index = pd.to_datetime(df.index, format='ISO8601')

# open dataframe with notes
fileloc_notes = "./flight_notes.xlsx"  # location of flight notes, available from zenodo
df_notes = pd.read_excel(fileloc_notes, dtype={'date':str, 'starttime':str, 'endtime':str})

df_notes['datetime_start'] = pd.to_datetime( df_notes['date'].str.split().str[0] + ' ' + df_notes['starttime'], format='%Y-%m-%d %H:%M:%S')
df_notes['datetime_end'] = pd.to_datetime( df_notes['date'].str.split().str[0] + ' ' + df_notes['endtime'], format='%Y-%m-%d %H:%M:%S')

df_notes['datetime_start'] = df_notes['datetime_start'].dt.tz_localize('Africa/Nairobi')
df_notes['datetime_end'] = df_notes['datetime_end'].dt.tz_localize('Africa/Nairobi')
df_notes['datetime_start'] = df_notes['datetime_start'].dt.tz_convert('UTC')
df_notes['datetime_end'] = df_notes['datetime_end'].dt.tz_convert('UTC')
df_notes = df_notes.drop(columns = ['date', 'starttime', 'endtime'])

for index, row in df_notes.iterrows():
    # octagon is the half-octagon flight used for most flights, at the start of the field campaign
    # we used a narrower V-shaped (plough) for a few flights.
    if row.typeOfFlight == 'octagon' or row.typeOfFlight == 'plough':
        start_time = row.datetime_start
        end_time = row.datetime_end
        date_string = start_time.strftime("%Y%m%d_%H%M%S")
        animal_string = row.animals.replace(" ", "_")
        df_octagon = df[ (df.index>start_time) & (df.index<end_time) ]

        df_octagon.to_csv(f"{directory}/{date_string}_{animal_string}_00.csv", index=True, decimal='.', sep=',')
