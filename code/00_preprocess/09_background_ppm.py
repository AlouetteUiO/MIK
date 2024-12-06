import os
import pandas as pd
from datetime import date, time, datetime
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

"""
Plot background methane concentration for inspection.
No data is saved in the df.
"""

# open dataframe
dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
df = pd.read_csv("08_df.csv", index_col='datetime', dtype=dtype_options)
df.index = pd.to_datetime(df.index, format='ISO8601')

# make a quick plot to see the variability in the methane concentration
# when there are no animals in the boma
# the figure shows that the background ppm varies between 1.65 to 1.8. 
df_no = df[df['NOTE.animals'] == 'none']
plt.scatter(df_no.index, df_no['AERIS.CH4 [ppm]'], s=1)
plt.ylim([1.65, 1.82])
plt.ylabel('CH4 [ppm]')
plt.title('Methane concentration at EC tower (no animals in boma)')
plt.savefig(f"09_background_ppm.png")
plt.close()

def get_flight(df, start_time, end_time):
    start_time = datetime.strptime(start_time,'%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end_time,'%Y-%m-%d %H:%M:%S') 
    start_time = start_time.replace(tzinfo=timezone.utc)
    end_time = end_time.replace(tzinfo=timezone.utc) 
    df = df[(df.index>start_time)&(df.index<end_time)]
    return df

# zoom in on one flight
start_time = '2024-03-05 00:00:00'
end_time = '2024-03-05 23:00:00'
df_flight = get_flight(df_no, start_time, end_time)
plt.scatter(df_flight.index, df_flight['AERIS.CH4 [ppm]'], s=1)
plt.ylim([1.65, 1.82])
plt.ylabel('CH4 [ppm]')
plt.title('Methane concentration at EC tower (no animals in boma)')
plt.savefig(f"09_background_ppm_0305.png")
plt.close()

# zoom in on another flight
start_time = '2024-03-07 13:30:00'
end_time = '2024-03-07 23:00:00'
df_flight = get_flight(df_no, start_time, end_time)
plt.scatter(df_flight.index, df_flight['AERIS.CH4 [ppm]'], s=1)
plt.ylim([1.65, 1.82])
plt.ylabel('CH4 [ppm]')
plt.title('Methane concentration at EC tower (no animals in boma)')
plt.savefig(f"09_background_ppm_0307.png")
plt.close()