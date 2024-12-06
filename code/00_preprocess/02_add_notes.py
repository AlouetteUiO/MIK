import os
import pandas as pd
from datetime import date, time, datetime

"""
Add field notes.
"""

df = pd.read_csv("01_df.csv", index_col='datetime')
df.index = pd.to_datetime(df.index, format='ISO8601')

# open dataframe with notes
fileloc_notes = "./flight_notes.xlsx"  # change to location of field notes data, available from zenodo
df_notes = pd.read_excel(fileloc_notes, dtype={'date':str, 'starttime':str, 'endtime':str})

df_notes['datetime_start'] = pd.to_datetime( df_notes['date'].str.split().str[0] + ' ' + df_notes['starttime'], format='%Y-%m-%d %H:%M:%S')
df_notes['datetime_end'] = pd.to_datetime( df_notes['date'].str.split().str[0] + ' ' + df_notes['endtime'], format='%Y-%m-%d %H:%M:%S')

df_notes['datetime_start'] = df_notes['datetime_start'].dt.tz_localize('Africa/Nairobi')
df_notes['datetime_end'] = df_notes['datetime_end'].dt.tz_localize('Africa/Nairobi')
df_notes['datetime_start'] = df_notes['datetime_start'].dt.tz_convert('UTC')
df_notes['datetime_end'] = df_notes['datetime_end'].dt.tz_convert('UTC')
df_notes = df_notes.drop(columns = ['date', 'starttime', 'endtime'])

# add notes to leading dataframe
for index, row in df_notes.iterrows():
    print(f"start = {row.datetime_start} and end = {row.datetime_end}")
    df.loc[(df.index>row.datetime_start)&(df.index<row.datetime_end), 'NOTE.animals'] = row.animals
    df.loc[(df.index>row.datetime_start)&(df.index<row.datetime_end), 'NOTE.amount'] = row.amount
    df.loc[(df.index>row.datetime_start)&(df.index<row.datetime_end), 'NOTE.typeOfFlight'] = row.typeOfFlight
    df.loc[(df.index>row.datetime_start)&(df.index<row.datetime_end), 'NOTE.location'] = row.location
    df.loc[(df.index>row.datetime_start)&(df.index<row.datetime_end), 'NOTE.note'] = row.note

    if isinstance(row.amount, int) or isinstance(row.amount, float):
        amount_int = int(row.amount)
    elif "+" in row.amount:
        numbers = row.amount.split("+")
        amount_int = 0
        for number in numbers:
            amount_int += int(number)
    else:
        raise Exception("number of animals not correctly defined")
    df.loc[(df.index>row.datetime_start)&(df.index<row.datetime_end), 'NOTE.amount_int'] = amount_int

# save dataframe
df.to_csv("02_df.csv", index=True, decimal='.', sep=',')