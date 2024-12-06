
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import os

"""
Add estimated z_source data (average mouth height of 10 animals in each herd, based on direct measurements)
"""

# open dataframe
dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
df = pd.read_csv("08_df.csv", index_col='datetime', dtype=dtype_options)
df.index = pd.to_datetime(df.index, format='ISO8601')

# add estimated z_source data
df.loc[df['NOTE.animals'] == "heifers", 'PROC.z_source_mean'] = 0.909 
df.loc[df['NOTE.animals'] == "steers", 'PROC.z_source_mean'] = 0.814
df.loc[df['NOTE.animals'] == "commerical boran cows", 'PROC.z_source_mean'] = 0.871
df.loc[df['NOTE.animals'] == "slick herd", 'PROC.z_source_mean'] = 0.839
df.loc[df['NOTE.animals'] == "camels", 'PROC.z_source_mean'] = 2.487
df.loc[df['NOTE.animals'] == "dry does", 'PROC.z_source_mean'] = 0.748
df.loc[df['NOTE.animals'] == "weaners", 'PROC.z_source_mean'] = 0.738
df.loc[df['NOTE.animals'] == "pregnant does", 'PROC.z_source_mean'] = 0.733
df.loc[df['NOTE.animals'] == "pregnant does + kids", 'PROC.z_source_mean'] = 0.733
df.loc[df['NOTE.animals'] == "lactating ewe + lambs", 'PROC.z_source_mean'] = 0.687

# save dataframe
df.to_csv("10_df.csv", index=True, decimal='.', sep=',')