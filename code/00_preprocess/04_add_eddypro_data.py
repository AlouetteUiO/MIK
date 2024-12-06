import os
import pandas as pd
from datetime import date, time, datetime
import numpy as np
import zipfile

"""
Add processed Eddy-covariance data (from EddyPro)
"""

directory = "./ECtower_Kapiti"  # change to location of eddy-covariance data, available from zenodo
prefix = "eddypro/eddypro_exp_full_output_"

df_total = pd.DataFrame()

columns_to_read = [ 'date',
                    'time',
                    'qc_Tau', 
                    'u_unrot',
                    'v_unrot',
                    'w_unrot', 
                    'wind_speed', 
                    'wind_dir',
                    'u*',
                    'L']

for ghgfile in os.listdir(directory):
    zip_file_path = f"{directory}/{ghgfile}"
    #print(f"processing {ghgfile}")

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()

        for filename in file_list:
            if filename.startswith(prefix):
                target_filename = filename
                zip_ref.extract(target_filename)

    target_file_path = os.path.join(os.getcwd(), target_filename)

    try:
        df = pd.read_csv(target_file_path, skiprows=[0,2], header=0, sep=',', usecols=columns_to_read)
        os.remove(target_file_path)

        new_column_names = ['EDDYPRO.'+name for name in columns_to_read]
        df.rename(columns=dict(zip(columns_to_read, new_column_names)), inplace=True)

        df['datetime'] = pd.to_datetime( df['EDDYPRO.date'].str.split().str[0] + ' ' + df['EDDYPRO.time'], format='%Y-%m-%d %H:%M').dt.tz_localize('Africa/Nairobi')
        df['datetime'] = df['datetime'].dt.tz_convert('UTC')
        df = df.set_index('datetime')
        df = df.drop(columns=['EDDYPRO.date', 'EDDYPRO.time'])

        df_total = pd.concat([df_total, df], ignore_index=False)
    except:
        print(f"something wrong with {target_file_path}")
        pass

# sort index of df_total
df_total = df_total.sort_index()

# add dataframe to leading dataframe
dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
df_old = pd.read_csv("03_df.csv", index_col='datetime', dtype=dtype_options)
df_old.index = pd.to_datetime(df_old.index, format='ISO8601')

# merge keeping the index of both dataframes
df_merged = pd.merge(df_old, df_total, left_index=True, right_index=True, how='outer')

# backwards fill the eddypro columns (eddypro gives half hour values)
df_merged_filled = df_merged.copy()
df_merged_filled[new_column_names[2:]] = df_merged[new_column_names[2:]].bfill()

# save dataframe
df_merged_filled.to_csv("04_df.csv", index=True, decimal='.', sep=',')